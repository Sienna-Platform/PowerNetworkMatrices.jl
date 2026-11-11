import HDF5
import SparseArrays

function _five_bus_core(ref_key::Int = 1)
    bus_ax = [1, 2, 3, 4, 5]
    arc_ax = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    b = [10.0, 5.0, 8.0, 4.0, 6.0, 7.0]
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
        ones(Int, n_arc);
        network_data = nothing,
        linear_solver = "KLU",
        tol = 0.0,
    )
end

function _ptdf_rows(core)
    n_arc = length(PMC.get_arc_axis(core))
    return PMC.with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K, work_ba_col, temp_data
        [copy(PMC.ptdf_row!(temp_data, K, work_ba_col, core, e)) for e in 1:n_arc]
    end
end

function _write_five_bus(path; ref_key::Int = 1, reverse_bus_map = Dict{Int, Int}())
    core = _five_bus_core(ref_key)
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
function _hostile(change!, expected::Regex)
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        HDF5.h5open(change!, path, "r+")
        @test_throws expected PMC.read_factor_core(path; linear_solver = "KLU")
    end
    return
end

@testset "Factor file: hostile core input" begin
    _hostile(r"format version 2") do f
        HDF5.delete_attribute(f, "format_version")
        HDF5.attributes(f)["format_version"] = 2
    end
    _hostile(r"arc_live_members") do f
        HDF5.delete_object(f, "arc_live_members")
        f["arc_live_members"] = ones(Float64, 6)
    end
    _hostile(r"arc_to") do f
        HDF5.delete_object(f, "arc_to")
        f["arc_to"] = [2, 3, 3, 4, 5]
    end
    _hostile(r"reduced_bus_from holds bus 4, which is on the bus axis") do f
        HDF5.delete_object(f, "reduced_bus_from")
        HDF5.delete_object(f, "reduced_bus_to")
        f["reduced_bus_from"] = [4]
        f["reduced_bus_to"] = [1]
    end
end
