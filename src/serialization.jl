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

_branch_tag(nr, branch::PSY.ACTransmission) = string(_resolve_branch_arc(nr, branch)[1])
_branch_tag(nr, ::PSY.ThreeWindingTransformer) = "3wt"

function _write_row_arrays(g, mods::Vector{NetworkModification}, starts, lens)
    arc_index = Int[]
    delta_b = Float64[]
    delta_shift = Float64[]
    opened = Int[]
    for mod in mods
        push!(starts, length(arc_index) + 1)
        push!(lens, length(mod.arc_modifications))
        for m in mod.arc_modifications
            push!(arc_index, m.arc_index)
            push!(delta_b, m.delta_b)
            push!(delta_shift, m.delta_shift_injection)
            push!(opened, m.opened)
        end
    end
    g["start"] = starts
    g["len"] = lens
    g["arc_index"] = arc_index
    g["delta_b"] = delta_b
    g["delta_shift"] = delta_shift
    g["opened"] = opened
    return
end

function _collect_branch_outages!(rows, skipped, mat, branches)
    nr = get_network_reduction_data(mat)
    for branch in branches
        try
            mod = NetworkModification(mat, branch)
            row = (
                string(nameof(typeof(branch))),
                PSY.get_name(branch),
                _branch_tag(nr, branch),
                mod,
            )
            push!(rows, row)
        catch e
            push!(skipped, sprint(showerror, e))
        end
    end
    return
end

function _write_branch_outages(f, mat, sys::PSY.System)
    rows = Tuple{String, String, String, NetworkModification}[]
    skipped = String[]
    _collect_branch_outages!(
        rows,
        skipped,
        mat,
        PSY.get_components(PSY.ACTransmission, sys),
    )
    _collect_branch_outages!(
        rows,
        skipped,
        mat,
        PSY.get_components(PSY.ThreeWindingTransformer, sys),
    )
    if !isempty(skipped)
        @warn "$(length(skipped)) branches have no outage row because a network reduction absorbed them. First error: $(first(skipped))"
    end
    g = HDF5.create_group(f, "branch_outages")
    g["type"] = [r[1] for r in rows]
    g["name"] = [r[2] for r in rows]
    g["tag"] = [r[3] for r in rows]
    g["is_islanding"] = [r[4].is_islanding for r in rows]
    _write_row_arrays(g, [r[4] for r in rows], Int[], Int[])
    return
end

_write_registered_outages(f, ::VirtualPTDF) = nothing

function _write_registered_outages(f, mat::VirtualMODF)
    registered = get_registered_contingencies(mat)
    if isempty(registered)
        return
    end
    ids = sort!(collect(keys(registered)))
    mods = NetworkModification[registered[id].modification for id in ids]
    g = HDF5.create_group(f, "outages")
    g["id"] = ids
    g["label"] = [m.label for m in mods]
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods, Int[], Int[])
    return
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename, sys::PSY.System; force = false)

Write the factor data of `mat` as the method without `sys` does. Also write the group
`branch_outages` with one row for each branch of `sys` that `NetworkModification(mat, branch)`
accepts, so `from_hdf5(BranchOutageTable, filename)` can build outages by name. The function
skips the branches that a network reduction absorbed and logs one warning with their count.
For a `VirtualMODF`, the function also writes the group `outages` with the registered
contingencies.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString,
    sys::PSY.System;
    force::Bool = false,
)
    to_hdf5(mat, filename; force = force)
    HDF5.h5open(filename, "r+") do f
        _write_branch_outages(f, mat, sys)
        _write_registered_outages(f, mat)
    end
    return filename
end

"""
    BranchOutageTable

Branch outages that `to_hdf5(mat, filename, sys)` stored. Read it with
`from_hdf5(BranchOutageTable, filename)`. Build a `NetworkModification` from it with
`NetworkModification(table, type, name)`.

# Fields
- `rows`: for the key `(type, name)`, the tag, the islanding flag and the row range.
- `arc_index`, `delta_b`, `delta_shift`, `opened`: the arrays that the row ranges index.
"""
struct BranchOutageTable
    rows::Dict{Tuple{String, String}, Tuple{String, Bool, UnitRange{Int}}}
    arc_index::Vector{Int}
    delta_b::Vector{Float64}
    delta_shift::Vector{Float64}
    opened::Vector{Int}
end

function _row_ranges(g)
    return [s:(s + l - 1) for (s, l) in zip(read(g["start"]), read(g["len"]))]
end

function _arc_modifications(arc_index, delta_b, delta_shift, opened, range)
    z = zero(YBUS_ELTYPE)
    return ArcModification[
        ArcModification(arc_index[i], delta_b[i], delta_shift[i], z, z, z, z, opened[i])
        for
        i in range
    ]
end

"""
    from_hdf5(::Type{BranchOutageTable}, filename) -> BranchOutageTable

Read the branch outage table from a file that `to_hdf5(mat, filename, sys)` wrote. The
function throws an error if the file has no such table.
"""
function from_hdf5(::Type{BranchOutageTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        if !haskey(f, "branch_outages")
            error(
                "File $filename has no branch outage table. " *
                "The file was written without a system.",
            )
        end
        g = f["branch_outages"]
        rows = Dict{Tuple{String, String}, Tuple{String, Bool, UnitRange{Int}}}()
        for (t, n, tag, islanding, range) in zip(
            read(g["type"]),
            read(g["name"]),
            read(g["tag"]),
            read(g["is_islanding"]),
            _row_ranges(g),
        )
            rows[(t, n)] = (tag, islanding, range)
        end
        return BranchOutageTable(
            rows,
            Vector{Int}(read(g["arc_index"])),
            Vector{Float64}(read(g["delta_b"])),
            Vector{Float64}(read(g["delta_shift"])),
            Vector{Int}(read(g["opened"])),
        )
    end
end

_dedupe_series!(::Val, ::Set{Int}, ::Int) = true

function _dedupe_series!(::Val{:series}, seen::Set{Int}, arc_index::Int)
    if arc_index in seen
        return false
    end
    push!(seen, arc_index)
    return true
end

"""
    NetworkModification(table::BranchOutageTable, type, name)
    NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})

Build the outage of one branch, or of several branches at once, from a
[`BranchOutageTable`](@ref). `type` is a `String` or a `Type`, for example
`"Line"` or `PSY.Line`. Two members of one series chain give one row of the whole arc.
The modification has no Ybus deltas and no shunt modifications. It serves the DC row
kernels only.
"""
function NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})
    mods = ArcModification[]
    seen_series = Set{Int}()
    islanding = false
    for key in keys
        tag, is_islanding, range = table.rows[key]
        islanding |= is_islanding
        for m in _arc_modifications(
            table.arc_index,
            table.delta_b,
            table.delta_shift,
            table.opened,
            range,
        )
            if _dedupe_series!(Val(Symbol(tag)), seen_series, m.arc_index)
                push!(mods, m)
            end
        end
    end
    return NetworkModification(
        join((key[2] for key in keys), "+"),
        mods,
        ShuntModification[],
        islanding,
    )
end

function NetworkModification(table::BranchOutageTable, type::String, name::String)
    return NetworkModification(table, [(type, name)])
end

function NetworkModification(table::BranchOutageTable, ::Type{T}, name::String) where {T}
    return NetworkModification(table, string(nameof(T)), name)
end

function _read_registered_outages!(f, cache::Dict{Int, ContingencySpec})
    g = f["outages"]
    arc_index = read(g["arc_index"])
    delta_b = read(g["delta_b"])
    delta_shift = read(g["delta_shift"])
    opened = read(g["opened"])
    for (id, label, islanding, range) in zip(
        read(g["id"]),
        read(g["label"]),
        read(g["is_islanding"]),
        _row_ranges(g),
    )
        mods = _arc_modifications(arc_index, delta_b, delta_shift, opened, range)
        cache[id] = ContingencySpec(
            id,
            NetworkModification(label, mods, ShuntModification[], islanding),
        )
    end
    return
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

Read a `VirtualMODF` from a file that [`to_hdf5`](@ref) wrote. If the file has the group
`outages`, the restored matrix has those registered contingencies. Their modifications serve
the DC row kernels only. The same limits apply as for the restored `VirtualPTDF`.
"""
function from_hdf5(
    ::Type{VirtualMODF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
)
    contingencies = Dict{Int, ContingencySpec}()
    core = HDF5.h5open(filename, "r") do f
        if haskey(f, "outages")
            _read_registered_outages!(f, contingencies)
        end
        return _read_factor_core(f; linear_solver = linear_solver)
    end
    return VirtualMODF(
        core,
        contingencies,
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache{RowCacheValue}}(),
        max_cache_size * MiB,
    )
end
