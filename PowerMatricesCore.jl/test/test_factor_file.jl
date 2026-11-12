import HDF5
import SparseArrays

function _five_bus_core(
    ref_key::Int = 1;
    arc_live_members = ones(Int, 6),
    b = [10.0, 5.0, 8.0, 4.0, 6.0, 7.0],
    tol = 0.0,
)
    bus_ax = [1, 2, 3, 4, 5]
    arc_ax = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    n_bus = length(bus_ax)
    n_arc = length(arc_ax)
    A = SparseArrays.spzeros(Int8, n_arc, n_bus)
    BA = SparseArrays.spzeros(Float64, n_bus, n_arc)
    for (e, (f, t)) in enumerate(arc_ax)
        A[e, f] = 1
        A[e, t] = -1
        BA[f, e] = b[e]
        BA[t, e] = -b[e]
    end
    return PMC.FactorCore(
        BA,
        A,
        Set([1]),
        (arc_ax, bus_ax),
        Dict(ref_key => (copy(arc_ax), copy(bus_ax))),
        arc_live_members;
        network_data = nothing,
        linear_solver = "KLU",
        tol = tol,
    )
end

function _ptdf_rows(core)
    return [PMC.ptdf_row(core, e) for e in eachindex(PMC.get_arc_axis(core))]
end

function _write_five_bus(
    path;
    ref_key::Int = 1,
    reverse_bus_map = Dict{Int, Int}(),
    arc_live_members = ones(Int, 6),
    tol = 0.0,
)
    core = _five_bus_core(ref_key; arc_live_members = arc_live_members, tol = tol)
    HDF5.h5open(path, "w") do f
        PMC.write_factor_core(
            f,
            core;
            shift_injection = zeros(length(PMC.get_arc_axis(core))),
            dist_slack = [0.0, 1.0, 1.0, 0.0, 2.0],
            reverse_bus_map = reverse_bus_map,
        )
    end
    return core
end

@testset "Factor file: core round trip on a 5-bus case" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        core = _write_five_bus(path)
        restored = PMC.read_factor_core(path; linear_solver = "KLU")
        @test isnothing(PMC.get_network_data(restored))
        @test PMC.get_axes(restored) == PMC.get_axes(core)
        @test PMC.get_ref_bus(restored) == PMC.get_ref_bus(core)
        @test PMC.get_tol_spec(restored) == PMC.get_tol_spec(core)
        @test _ptdf_rows(restored) == _ptdf_rows(core)
        HDF5.h5open(path, "r") do f
            budget = PMC.ReadBudget(path)
            @test PMC.read_dist_slack(f, restored, budget) == [0.0, 1.0, 1.0, 0.0, 2.0]
            @test PMC.read_shift_injection(f, restored, budget) == zeros(6)
        end
    end
end

# Bus 99 is not on the bus axis. A reduction merged it into the reference bus 1.
@testset "Factor file: subnetwork key on a merged bus" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        core = _write_five_bus(path; ref_key = 99, reverse_bus_map = Dict(99 => 1, 98 => 3))
        restored = PMC.read_factor_core(path; linear_solver = "KLU")
        @test PMC.get_ref_bus(restored) == [99]
        @test _ptdf_rows(restored) == _ptdf_rows(core)
        HDF5.h5open(path, "r") do f
            arrays = PMC.read_factor_arrays(f, PMC.ReadBudget(path))
            @test arrays.reverse_bus_map == Dict(99 => 1, 98 => 3)
        end
    end
end

# Each case writes a valid file, changes one entry, and expects an error that names it.
function _hostile(change!, expected, reader = _read_core, writer = _write_five_bus)
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        writer(path)
        HDF5.h5open(change!, path, "r+")
        @test_throws expected reader(path)
    end
    return
end

function _set_column!(g, name::String, row::Int, value)
    column = read(g[name])
    column[row] = value
    _replace_entry!(g, name, column)
    return
end

function _read_core(path)
    return PMC.read_factor_core(path; linear_solver = "KLU")
end

@testset "Factor file: hostile core input" begin
    _hostile(
        Regex("format version $(PMC.FACTOR_FILE_FORMAT_VERSION + 1)"),
    ) do f
        HDF5.delete_attribute(f, "format_version")
        HDF5.attributes(f)["format_version"] = PMC.FACTOR_FILE_FORMAT_VERSION + 1
    end
    _hostile(r"arc_live_members") do f
        HDF5.delete_object(f, "arc_live_members")
        f["arc_live_members"] = ones(Float64, 6)
    end
    _hostile(r"arc_to") do f
        HDF5.delete_object(f, "arc_to")
        f["arc_to"] = [2, 3, 3, 4, 5]
    end
    _hostile(
        r"reduced_bus_from holds bus 4, which is on the bus axis",
    ) do f
        HDF5.delete_object(f, "reduced_bus_from")
        HDF5.delete_object(f, "reduced_bus_to")
        f["reduced_bus_from"] = [4]
        f["reduced_bus_to"] = [1]
    end
end

@testset "Factor file: absolute tol must be finite and not negative" begin
    for bad in (Inf, NaN, -1.0)
        _hostile(
            Regex("tol_value is $bad; expected a finite value >= 0"),
        ) do f
            HDF5.delete_attribute(f, "tol_value")
            HDF5.attributes(f)["tol_value"] = bad
        end
    end
end

function _hostile_slack(weights::Vector{Float64}, expected::Regex)
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        core = PMC.read_factor_core(path; linear_solver = "KLU")
        HDF5.h5open(path, "r+") do f
            HDF5.delete_object(f, "dist_slack")
            f["dist_slack"] = weights
        end
        HDF5.h5open(path, "r") do f
            @test_throws expected PMC.read_dist_slack(f, core)
        end
    end
    return
end

@testset "Factor file: dist_slack weights" begin
    _hostile_slack(zeros(5), r"the sum must be finite and positive")
    _hostile_slack([1.0, -1.0, 1.0, 1.0, 1.0], r"must not be negative")
end

function _replace_entry!(g, name::String, value)
    HDF5.delete_object(g, name)
    g[name] = value
    return
end

function _fixed_type(width::Int)
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, width)
    return dt
end

@testset "Factor file: hostile storage" begin
    _hostile(r"link arc_from is not a hard link") do f
        other = joinpath(mktempdir(), "other.h5")
        HDF5.h5open(g -> g["arc_from"] = [1, 2], other, "w")
        HDF5.delete_object(f, "arc_from")
        HDF5.create_external(f, "arc_from", other, "arc_from")
    end
    _hostile(r"link alias is not a hard link") do f
        HDF5.API.h5l_create_soft(
            "bus_axis",
            f,
            "alias",
            HDF5.API.H5P_DEFAULT,
            HDF5.API.H5P_DEFAULT,
        )
    end
    _hostile(r"arc_susceptances names an external file") do f
        other = joinpath(mktempdir(), "data.bin")
        write(other, zeros(6))
        HDF5.delete_object(f, "arc_susceptances")
        HDF5.create_dataset(f, "arc_susceptances", Float64, (6,); external = (other, 0, 48))
    end
    _hostile(r"arc_susceptances has the layout virtual") do f
        HDF5.delete_object(f, "arc_susceptances")
        space = HDF5.dataspace((6,))
        dcpl = HDF5.create_property(HDF5.API.H5P_DATASET_CREATE)
        HDF5.API.h5p_set_virtual(dcpl, space, ".", "arc_from", space)
        HDF5.API.h5d_create(
            f,
            "arc_susceptances",
            HDF5.datatype(Float64),
            space,
            HDF5.API.H5P_DEFAULT,
            dcpl,
            HDF5.API.H5P_DEFAULT,
        )
    end
    _hostile(r"arc_susceptances has a filter") do f
        HDF5.delete_object(f, "arc_susceptances")
        ds = HDF5.create_dataset(
            f,
            "arc_susceptances",
            Float64,
            (6,);
            chunk = (6,),
            deflate = 3,
        )
        ds[:] = ones(6)
    end
    _hostile(r"entry arc_from is a HDF5.Group") do f
        HDF5.delete_object(f, "arc_from")
        HDF5.create_group(f, "arc_from")
    end
    _hostile(r"entry arc_from has the HDF5 class 6") do f
        HDF5.delete_object(f, "arc_from")
        f["arc_from"] = [(a = 1, b = 2.0)]
    end
    _hostile(r"bus_axis has rank 2") do f
        _replace_entry!(f, "bus_axis", [1 2; 3 4])
    end
end

@testset "Factor file: hostile types and sizes" begin
    _hostile(r"bus_axis has the HDF5 class 0 and 4 bytes") do f
        _replace_entry!(f, "bus_axis", Int32[1, 2, 3])
    end
    _hostile(r"bus_axis has the HDF5 class 3") do f
        HDF5.delete_object(f, "bus_axis")
        HDF5.create_dataset(
            f,
            "bus_axis",
            _fixed_type(2^30),
            HDF5.dataspace((1,));
            chunk = (1,),
        )
    end
    _hostile(r"declares 1099511627776 elements") do f
        HDF5.delete_object(f, "arc_from")
        HDF5.create_dataset(f, "arc_from", Int, (2^40,); chunk = (1024,))
    end
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        HDF5.h5open(path, "r+") do f
            HDF5.create_dataset(
                f,
                "wide",
                _fixed_type(2^26),
                HDF5.dataspace((16,));
                chunk = (1,),
            )
        end
        HDF5.h5open(path, "r") do f
            @test_throws "string of 67108864 bytes per element" PMC._check_before_read(
                f["wide"],
                "wide",
                Vector{String},
                PMC.ReadBudget(filesize(path)),
            )
        end
    end
    @test_throws "string of 1073741824 bytes" PMC._check_stored_type(
        String,
        "system_uuid",
        (Int(HDF5.API.H5T_STRING), 2^30, false),
    )
    @test_throws "declares 16 elements of 67108864 bytes" PMC._charge!(
        PMC.ReadBudget(100),
        "x",
        16,
        2^26,
    )
    @test_throws "declares" PMC._charge!(PMC.ReadBudget(100), "x", typemax(Int) ÷ 2, 4)
end

@testset "Factor file: one budget for all reads" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        n = (filesize(path) ÷ 8) * 6 ÷ 10
        HDF5.h5open(path, "r+") do f
            for i in 1:2
                HDF5.create_dataset(f, "big$i", Int, (n,); chunk = (min(n, 1024),))
            end
        end
        HDF5.h5open(path, "r") do f
            budget = PMC.ReadBudget(filesize(path))
            @test PMC._check_before_read(f["big1"], "big1", Vector{Int}, budget) == 8
            @test_throws "remaining read budget" PMC._check_before_read(
                f["big2"],
                "big2",
                Vector{Int},
                budget,
            )
        end
    end
end

@testset "Factor file: hostile system_uuid" begin
    _hostile(r"system_uuid has 3 values; expected 2") do f
        HDF5.attributes(f)["system_uuid"] = Int64[1, 2, 3]
    end
    _hostile(
        r"system_uuid has the HDF5 class 3 and 10 bytes",
    ) do f
        PMC._write_string_attribute!(HDF5.attributes(f), "system_uuid", "not-a-uuid")
    end
end

@testset "Factor file: hostile network arrays" begin
    _hostile(
        r"entry /subnetworks is a dataset; expected a group",
    ) do f
        _replace_entry!(f, "subnetworks", [1, 2])
    end
    _hostile(
        r"entry /subnetworks/1 is a dataset; expected a group",
    ) do f
        _replace_entry!(f["subnetworks"], "1", [1, 2])
    end
    _hostile(r"subnetworks and only") do f
        for i in 1:6
            HDF5.create_group(f["subnetworks"], "extra$i")
        end
    end
    _hostile(r"reference bus positions for") do f
        _replace_entry!(f, "ref_bus_positions", Int[])
    end
    _hostile(r"more than once") do f
        _replace_entry!(f, "ref_bus_positions", [1, 1])
    end
    _hostile(r"below 1") do f
        _replace_entry!(f, "arc_live_members", fill(-1, 6))
    end
    _hostile(r"at position 1 is 1.0e308") do f
        _replace_entry!(f, "arc_susceptances", [1e308; fill(1.0, 5)])
    end
    _hostile(r"is a self-loop") do f
        _replace_entry!(f, "arc_to", [1, 3, 3, 4, 5, 5])
    end
    _hostile(r"bus axis has the entry") do f
        _replace_entry!(f, "bus_axis", [1, 1, 3, 4, 5])
    end
    _hostile(r"reduced_bus_from has the value") do f
        _replace_entry!(f, "reduced_bus_from", [9, 9])
        _replace_entry!(f, "reduced_bus_to", [1, 2])
    end
    _hostile(r"not the reference bus") do f
        _replace_entry!(f["subnetworks"]["1"], "ref", 99999)
        _replace_entry!(f, "reduced_bus_from", [99999])
        _replace_entry!(f, "reduced_bus_to", [2])
    end
    _hostile(r"no reduction maps it") do f
        _replace_entry!(f["subnetworks"]["1"], "ref", 99999)
    end
end

function _write_names!(
    f;
    types = ["Line", "Line", "Transformer"],
    names = ["a", "b", "a"],
    from = [1, 1, 2],
    to = [2, 2, 4],
)
    g = HDF5.create_group(f, "names")
    PMC._write_strings!(g, "type", types)
    PMC._write_strings!(g, "name", names)
    g["from"] = from
    g["to"] = to
    return
end

@testset "Factor file: branch names round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        core = _write_five_bus(path)
        @test_throws "no entry /names" PMC.read_branch_names(path)
        HDF5.h5open(_write_names!, path, "r+")
        names = PMC.read_branch_names(path)
        @test names == Dict(
            ("Line", "a") => (1, 2),
            ("Line", "b") => (1, 2),
            ("Transformer", "a") => (2, 4),
        )
        @test PMC.get_arc_lookup(core)[names[("Transformer", "a")]] == 4
    end
end

@testset "Factor file: hostile branch names" begin
    _hostile(
        "maps a to (2, 5) at row 3, which is not an arc",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        _write_names!(f; to = [2, 2, 5])
    end
    _hostile(
        "maps b to (2, 1) at row 2, which is not an arc",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        _write_names!(f; from = [1, 2, 2], to = [2, 1, 4])
    end
    _hostile("more than once", PMC.read_branch_names, _write_five_bus) do f
        _write_names!(f; names = ["a", "a", "a"])
    end
    _hostile("columns of unequal length", PMC.read_branch_names, _write_five_bus) do f
        _write_names!(f; from = [1, 1])
    end
    _hostile(
        "entry /names is a dataset; expected a group",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        f["names"] = [1, 2]
    end
    _hostile(
        "entry /names is a committed datatype; expected a group",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        HDF5.commit_datatype(
            f,
            "names",
            HDF5.Datatype(HDF5.API.h5t_copy(HDF5.hdf5_type_id(Int64))),
        )
    end
    _hostile("expected a string", PMC.read_branch_names, _write_five_bus) do f
        _write_names!(f)
        HDF5.delete_object(f["names"], "type")
        f["names"]["type"] = [1, 2, 3]
    end
    _hostile(
        "name is a variable-length string",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        _write_names!(f)
        HDF5.delete_object(f["names"], "name")
        f["names"]["name"] = fill("x", 3)
    end
    _hostile(
        "string of 65537 bytes per element",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        _write_names!(f)
        HDF5.delete_object(f["names"], "name")
        HDF5.create_dataset(
            f["names"],
            "name",
            _fixed_type(PMC.MAX_FIXED_STRING_BYTES + 1),
            HDF5.dataspace((3,));
            chunk = (1,),
        )
    end
    _hostile(
        "Unsupported factor file format version",
        PMC.read_branch_names,
        _write_five_bus,
    ) do f
        _write_names!(f)
        HDF5.delete_attribute(f, "format_version")
        HDF5.attributes(f)["format_version"] = PMC.FACTOR_FILE_FORMAT_VERSION + 1
    end
end
