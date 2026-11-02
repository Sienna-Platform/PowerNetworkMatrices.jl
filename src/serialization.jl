const FACTOR_FILE_FORMAT_VERSION = 1

_cutoff_kind(::AbsoluteCutoff) = "absolute"
_cutoff_kind(::RelativeCutoff) = "relative"

_restore_cutoff(::Val{:absolute}, value::Float64) = AbsoluteCutoff(value)
_restore_cutoff(::Val{:relative}, value::Float64) = RelativeCutoff(value)

function _restore_cutoff(::Val{K}, ::Float64) where {K}
    return error("Unknown cutoff kind \"$K\" in factor file.")
end

function _write_sparse(g, M::SparseArrays.SparseMatrixCSC)
    HDF5.attributes(g)["m"] = M.m
    HDF5.attributes(g)["n"] = M.n
    g["colptr"] = M.colptr
    g["rowval"] = M.rowval
    g["nzval"] = M.nzval
    return
end

function _read_sparse(g, ::Type{T}) where {T}
    a = HDF5.attributes(g)
    return SparseArrays.SparseMatrixCSC{T, Int}(
        read(a["m"]),
        read(a["n"]),
        read(g["colptr"]),
        read(g["rowval"]),
        Vector{T}(read(g["nzval"])),
    )
end

_system_uuid_string(::Nothing) = ""
_system_uuid_string(uuid::Base.UUID) = string(uuid)

_restore_system_uuid(s::AbstractString) = _restore_system_uuid(Val(isempty(s)), s)
_restore_system_uuid(::Val{true}, ::AbstractString) = nothing
_restore_system_uuid(::Val{false}, s::AbstractString) = Base.UUID(s)

function _write_factor_core(f, core::VirtualFactorCore, dist_slack::Vector{Float64})
    nr = get_network_reduction_data(core)
    arc_ax, bus_ax = core.axes
    a = HDF5.attributes(f)
    a["format_version"] = FACTOR_FILE_FORMAT_VERSION
    a["system_uuid"] = _system_uuid_string(core.system_uuid)
    a["cutoff_kind"] = _cutoff_kind(core.tol)
    a["cutoff_value"] = cutoff_value(core.tol)
    a["n_bus"] = length(bus_ax)
    a["n_arc"] = length(arc_ax)
    _write_sparse(HDF5.create_group(f, "BA"), core.BA)
    _write_sparse(HDF5.create_group(f, "A"), core.A)
    f["bus_axis"] = bus_ax
    f["arc_from"] = first.(arc_ax)
    f["arc_to"] = last.(arc_ax)
    f["ref_bus_positions"] = sort(get_ref_bus_position(core))
    f["arc_susceptances"] = core.arc_susceptances
    f["arc_live_members"] = core.arc_live_members
    f["shift_injection"] = Float64[arc_dc_shift_injection(nr, arc) for arc in arc_ax]
    f["dist_slack"] = dist_slack
    f["reduced_bus_from"] = collect(keys(nr.reverse_bus_search_map))
    f["reduced_bus_to"] = collect(values(nr.reverse_bus_search_map))

    types = String[]
    names = String[]
    from = Int[]
    to = Int[]
    for (T, entries) in get_name_to_arc_maps(get_branch_catalog(core))
        for (name, arc) in entries
            push!(types, string(nameof(T)))
            push!(names, name)
            push!(from, arc[1])
            push!(to, arc[2])
        end
    end
    g = HDF5.create_group(f, "names")
    g["type"] = types
    g["name"] = names
    g["from"] = from
    g["to"] = to

    g = HDF5.create_group(f, "subnetworks")
    for (ref, (axis_arcs, axis_buses)) in core.subnetwork_axes
        sg = HDF5.create_group(g, string(ref))
        sg["arc_from"] = first.(axis_arcs)
        sg["arc_to"] = last.(axis_arcs)
        sg["bus"] = axis_buses
    end
    return
end

function _check_can_write(filename::AbstractString, force::Bool)
    if isfile(filename) && !force
        error("File $filename exists. Pass `force = true` to overwrite it.")
    end
    return
end

"""
    to_hdf5(mat::VirtualPTDF, filename; force = false)

Write the factor data of `mat` to the HDF5 file `filename`. The file holds the arrays that
[`from_hdf5`](@ref) needs to rebuild the matrix without a `System`. The function does not
write cached rows. If the file exists and `force` is `false`, the function throws an error.
"""
function to_hdf5(mat::VirtualPTDF, filename::AbstractString; force::Bool = false)
    _check_can_write(filename, force)
    HDF5.h5open(filename, "w") do f
        _write_factor_core(f, get_core(mat), mat.dist_slack)
    end
    return filename
end

"""
    to_hdf5(mat::VirtualMODF, filename; force = false)

Write the factor data of `mat` to the HDF5 file `filename`. The file has the same layout as
the file for a `VirtualPTDF`, with an empty `dist_slack` dataset. If the file exists and
`force` is `false`, the function throws an error.
"""
function to_hdf5(mat::VirtualMODF, filename::AbstractString; force::Bool = false)
    _check_can_write(filename, force)
    HDF5.h5open(filename, "w") do f
        _write_factor_core(f, get_core(mat), Float64[])
    end
    return filename
end

function _read_factor_core(f; linear_solver::String)
    a = HDF5.attributes(f)
    version = read(a["format_version"])
    if version != FACTOR_FILE_FORMAT_VERSION
        error(
            "Unsupported factor file format version $version; " *
            "this version of PowerNetworkMatrices reads version $FACTOR_FILE_FORMAT_VERSION.",
        )
    end
    BA = _read_sparse(f["BA"], Float64)
    A = _read_sparse(f["A"], Int8)
    bus_ax = Vector{Int}(read(f["bus_axis"]))
    arc_ax = Tuple{Int, Int}[zip(read(f["arc_from"]), read(f["arc_to"]))...]
    cutoff = _restore_cutoff(
        Val(Symbol(read(a["cutoff_kind"]))),
        Float64(read(a["cutoff_value"])),
    )

    nrd = NetworkReductionData(;
        reverse_bus_search_map = Dict{Int, Int}(
            zip(read(f["reduced_bus_from"]), read(f["reduced_bus_to"])),
        ),
    )
    name_to_arc = NAME_TO_ARC()
    g = f["names"]
    for (t, n, from, to) in
        zip(read(g["type"]), read(g["name"]), read(g["from"]), read(g["to"]))
        T = getfield(PSY, Symbol(t))
        entries = get!(name_to_arc, T) do
            DataStructures.SortedDict{String, ARC_ENTRY}()
        end
        entries[n] = (from, to)
    end
    catalog = BranchCatalog(
        nrd,
        ARC_TABLE(),
        name_to_arc,
        COMPONENT_TO_ENTRY(),
        COMPONENT_NAME_INDEX(),
    )

    subnetwork_axes = Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}}()
    for k in keys(f["subnetworks"])
        sg = f["subnetworks"][k]
        subnetwork_axes[parse(Int, k)] = (
            Tuple{Int, Int}[zip(read(sg["arc_from"]), read(sg["arc_to"]))...],
            Vector{Int}(read(sg["bus"])),
        )
    end

    return VirtualFactorCore(
        BA,
        A,
        Set{Int}(read(f["ref_bus_positions"])),
        (arc_ax, bus_ax),
        subnetwork_axes,
        Vector{Int}(read(f["arc_live_members"])),
        catalog;
        linear_solver = linear_solver,
        tol = cutoff,
        system_uuid = _restore_system_uuid(read(a["system_uuid"])),
    )
end

"""
    from_hdf5(::Type{VirtualPTDF}, filename; linear_solver, max_cache_size) -> VirtualPTDF

Read a `VirtualPTDF` from a file that [`to_hdf5`](@ref) wrote. The function does not need a
`System`. It factorizes the ABA matrix again with `linear_solver`.

The restored matrix has no components and no reduction maps. `NetworkModification(mat, arc)`,
`NetworkModification(mat, branch)` and `mat[name, bus]` do not work on it. The function
throws an error if the format version of the file is not 1.
"""
function from_hdf5(
    ::Type{VirtualPTDF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
)
    return HDF5.h5open(filename, "r") do f
        core = _read_factor_core(f; linear_solver = linear_solver)
        dist_slack = Vector{Float64}(read(f["dist_slack"]))
        if isempty(dist_slack)
            dist_slack_normalized = Float64[]
        else
            dist_slack_normalized = dist_slack / sum(dist_slack)
        end
        n_bus = length(core.axes[2])
        return VirtualPTDF(
            core,
            dist_slack,
            dist_slack_normalized,
            _persistent_row_cache(
                max_cache_size,
                core.lookup,
                Vector{Tuple{Int, Int}}(),
                n_bus,
            ),
            ReentrantLock(),
        )
    end
end

"""
    from_hdf5(::Type{VirtualMODF}, filename; linear_solver, max_cache_size) -> VirtualMODF

Read a `VirtualMODF` from a file that [`to_hdf5`](@ref) wrote. The restored matrix has no
registered outages. The same limits apply as for the restored `VirtualPTDF`.
"""
function from_hdf5(
    ::Type{VirtualMODF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
)
    core = HDF5.h5open(filename, "r") do f
        return _read_factor_core(f; linear_solver = linear_solver)
    end
    return VirtualMODF(
        core,
        Dict{Int, ContingencySpec}(),
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache{RowCacheValue}}(),
        max_cache_size * MiB,
    )
end
