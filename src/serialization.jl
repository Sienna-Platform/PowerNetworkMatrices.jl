const FACTOR_FILE_FORMAT_VERSION = 1

# The names table maps a type name to a PSY type through this fixed table. The reader never
# looks a file string up in a module, so a file cannot name an arbitrary binding.
const FACTOR_FILE_BRANCH_TYPES = Dict{String, DataType}(
    string(nameof(T)) => T for T in (
        PSY.Line,
        PSY.TwoWindingTransformer,
        PSY.ThreeWindingTransformer,
        PSY.DiscreteControlledACBranch,
        PSY.GenericArcImpedance,
    )
)

function _branch_type(name::String)
    haskey(FACTOR_FILE_BRANCH_TYPES, name) ||
        error("Unknown branch type \"$name\" in factor file.")
    return FACTOR_FILE_BRANCH_TYPES[name]
end

function _branch_type_name(::Type{T}) where {T}
    name = string(nameof(T))
    haskey(FACTOR_FILE_BRANCH_TYPES, name) ||
        error("Branch type $T has no entry in FACTOR_FILE_BRANCH_TYPES.")
    return name
end

# Every value in a factor file is untrusted. `_typed` admits the one type the layout
# prescribes for an entry and errors with the entry name on anything else.
_typed(x::T, ::Type{T}, ::String) where {T} = x
function _typed(x, ::Type{T}, name::String) where {T}
    return error("Factor file entry $name holds a $(typeof(x)); expected $T.")
end

function _read(
    g::Union{HDF5.File, HDF5.Group, HDF5.Attributes},
    name::String,
    ::Type{T},
) where {T}
    haskey(g, name) || error("Factor file has no entry $name.")
    return _typed(read(g[name]), T, name)
end

# Columns of one table must agree in length, or `zip` would drop rows in silence.
function _check_same_length(table::String, columns::Pair{String, <:AbstractVector}...)
    lengths = map(column -> first(column) => length(last(column)), columns)
    allequal(map(last, lengths)) ||
        error("Factor file table $table has columns of unequal length: $lengths.")
    return
end

function _write_tol!(a::HDF5.Attributes, tol::Float64)
    a["tol_kind"] = "absolute"
    a["tol_value"] = tol
    return
end

# A numeric `data_precision` is stored as a Float64. `:auto` leaves the attribute absent.
function _write_data_precision!(a::HDF5.Attributes, dp::Float64)
    a["auto_data_precision"] = dp
    return
end
_write_data_precision!(::HDF5.Attributes, ::Symbol) = nothing

function _write_tol!(a::HDF5.Attributes, tol::AutoTolerance)
    a["tol_kind"] = "auto"
    _write_data_precision!(a, tol.data_precision)
    a["auto_safety"] = tol.safety
    a["auto_quantile"] = tol.quantile
    return
end

function _read_data_precision(a::HDF5.Attributes)
    if haskey(a, "auto_data_precision")
        return _read(a, "auto_data_precision", Float64)
    end
    return :auto
end

function _read_tol(a::HDF5.Attributes)
    kind = _read(a, "tol_kind", String)
    if kind == "absolute"
        return _read(a, "tol_value", Float64)
    elseif kind == "auto"
        return AutoTolerance(;
            data_precision = _read_data_precision(a),
            safety = _read(a, "auto_safety", Float64),
            quantile = _read(a, "auto_quantile", Float64),
        )
    end
    return error("Unknown tolerance kind \"$kind\" in factor file.")
end

_system_uuid_string(::Nothing) = ""
_system_uuid_string(uuid::Base.UUID) = string(uuid)

function _restore_system_uuid(s::AbstractString)
    if isempty(s)
        return nothing
    end
    return Base.UUID(s)
end

function _write_factor_core(
    f::HDF5.File,
    core::VirtualFactorCore,
    dist_slack::Vector{Float64},
)
    nr = get_network_reduction_data(core)
    arc_ax, bus_ax = core.axes
    reduced = get_reverse_bus_search_map(nr)
    a = HDF5.attributes(f)
    a["format_version"] = FACTOR_FILE_FORMAT_VERSION
    a["system_uuid"] = _system_uuid_string(get_system_uuid(core))
    _write_tol!(a, get_tol_spec(core))
    f["bus_axis"] = bus_ax
    f["arc_from"] = first.(arc_ax)
    f["arc_to"] = last.(arc_ax)
    f["ref_bus_positions"] = sort(get_ref_bus_position(core))
    f["arc_susceptances"] = core.arc_susceptances .* core.arc_susceptance_signs
    f["arc_live_members"] = core.arc_live_members
    f["shift_injection"] = Float64[arc_dc_shift_injection(nr, arc) for arc in arc_ax]
    f["dist_slack"] = dist_slack
    f["reduced_bus_from"] = collect(keys(reduced))
    f["reduced_bus_to"] = collect(values(reduced))

    types = String[]
    names = String[]
    from = Int[]
    to = Int[]
    for (T, entries) in get_name_to_arc_maps(get_branch_catalog(core))
        for (name, arc) in entries
            push!(types, _branch_type_name(T))
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
        sg["ref"] = ref
        sg["arc_from"] = first.(axis_arcs)
        sg["arc_to"] = last.(axis_arcs)
        sg["bus"] = axis_buses
    end
    return
end

_dist_slack(mat::VirtualPTDF) = get_dist_slack(mat)
_dist_slack(::VirtualMODF) = Float64[]

function _write_factor_file(
    write_outages,
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString,
    force::Bool,
)
    if isfile(filename) && !force
        error("File $filename exists. Pass `force = true` to overwrite it.")
    end
    HDF5.h5open(filename, "w") do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
        write_outages(f)
    end
    return filename
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename; force = false)

Write the factor data of `mat` to the HDF5 file `filename`. The file holds the arrays that
[`from_hdf5`](@ref) needs to rebuild the matrix without a `System`. The function does not
write cached rows. A `VirtualMODF` writes an empty `dist_slack` dataset. If the file exists
and `force` is `false`, the function throws an error.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString;
    force::Bool = false,
)
    return _write_factor_file(f -> nothing, mat, filename, force)
end

_outage_key(branch::PSY.ACTransmission) =
    (_branch_type_name(typeof(branch)), PSY.get_name(branch))

_branch_tag(nr::NetworkReductionData, branch::PSY.ACTransmission) =
    string(_resolve_branch_arc(nr, branch)[1])
_branch_tag(::NetworkReductionData, ::PSY.ThreeWindingTransformer) = "3wt"

function _write_row_arrays(g::HDF5.Group, mods::Vector{NetworkModification})
    starts = Int[]
    lens = Int[]
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

function _write_branch_outages(
    f::HDF5.File,
    mat::Union{VirtualPTDF, VirtualMODF},
    sys::PSY.System,
)
    nr = get_network_reduction_data(mat)
    keys_ = Tuple{String, String}[]
    tags = String[]
    mods = NetworkModification[]
    skipped = 0
    first_reason = ""
    branches = Iterators.flatten((
        PSY.get_components(PSY.ACTransmission, sys),
        PSY.get_components(PSY.ThreeWindingTransformer, sys),
    ))
    for branch in branches
        tag = _branch_tag(nr, branch)
        reason = ""
        if tag == "not_found"
            reason = "a network reduction absorbed it"
        else
            try
                push!(mods, NetworkModification(mat, branch))
            catch e
                reason = sprint(showerror, e)
            end
        end
        if isempty(reason)
            push!(keys_, _outage_key(branch))
            push!(tags, tag)
        else
            skipped += 1
            if isempty(first_reason)
                first_reason = "$(PSY.get_name(branch)): $reason"
            end
        end
    end
    if !iszero(skipped)
        @warn "$skipped branches have no outage row because NetworkModification failed on them. First: $first_reason"
    end
    g = HDF5.create_group(f, "branch_outages")
    g["type"] = first.(keys_)
    g["name"] = last.(keys_)
    g["tag"] = tags
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods)
    return
end

_write_registered_outages(::HDF5.File, ::VirtualPTDF) = nothing

function _write_registered_outages(f::HDF5.File, mat::VirtualMODF)
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
    _write_row_arrays(g, mods)
    return
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename, sys::PSY.System; force = false)

Write the factor data of `mat` as the method without `sys` does. Also write the group
`branch_outages` with one row for each branch of `sys` that `NetworkModification(mat, branch)`
accepts, so `from_hdf5(BranchOutageTable, filename)` can build outages by name. A branch that
a network reduction absorbed, or that `NetworkModification` rejects, has no row; the function
logs one warning with the count and the first reason. For a `VirtualMODF`, the function also
writes the group `outages` with the registered contingencies.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString,
    sys::PSY.System;
    force::Bool = false,
)
    return _write_factor_file(mat, filename, force) do f
        _write_branch_outages(f, mat, sys)
        _write_registered_outages(f, mat)
    end
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

function _read_row_arrays(g::HDF5.Group)
    starts = _read(g, "start", Vector{Int})
    lens = _read(g, "len", Vector{Int})
    arc_index = _read(g, "arc_index", Vector{Int})
    delta_b = _read(g, "delta_b", Vector{Float64})
    delta_shift = _read(g, "delta_shift", Vector{Float64})
    opened = _read(g, "opened", Vector{Int})
    _check_same_length("rows", "start" => starts, "len" => lens)
    _check_same_length(
        "arc rows",
        "arc_index" => arc_index,
        "delta_b" => delta_b,
        "delta_shift" => delta_shift,
        "opened" => opened,
    )
    ranges = [s:(s + l - 1) for (s, l) in zip(starts, lens)]
    for range in ranges
        (isempty(range) || checkbounds(Bool, arc_index, range)) ||
            error(
                "Factor file row range $range is outside the $(length(arc_index)) arc rows.",
            )
    end
    table = BranchOutageTable(
        Dict{Tuple{String, String}, Tuple{String, Bool, UnitRange{Int}}}(),
        arc_index,
        delta_b,
        delta_shift,
        opened,
    )
    return ranges, table
end

function _arc_modifications(table::BranchOutageTable, range::UnitRange{Int})
    z = zero(YBUS_ELTYPE)
    return ArcModification[
        ArcModification(
            table.arc_index[i],
            table.delta_b[i],
            table.delta_shift[i],
            z,
            z,
            z,
            z,
            table.opened[i],
        ) for i in range
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
        ranges, table = _read_row_arrays(g)
        types = _read(g, "type", Vector{String})
        names = _read(g, "name", Vector{String})
        tags = _read(g, "tag", Vector{String})
        is_islanding = _read(g, "is_islanding", Vector{Bool})
        _check_same_length(
            "branch_outages",
            "start" => ranges,
            "type" => types,
            "name" => names,
            "tag" => tags,
            "is_islanding" => is_islanding,
        )
        for (t, n, tag, islanding, range) in zip(types, names, tags, is_islanding, ranges)
            table.rows[(t, n)] = (tag, islanding, range)
        end
        return table
    end
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
    if isempty(keys)
        error("An outage needs at least one branch key.")
    end
    mods = ArcModification[]
    seen_series = Set{Int}()
    islanding = false
    for key in keys
        if !haskey(table.rows, key)
            error("Branch $(key[1]) $(key[2]) has no row in the outage table.")
        end
        tag, is_islanding, range = table.rows[key]
        islanding |= is_islanding
        is_series = tag == "series"
        for m in _arc_modifications(table, range)
            if is_series
                if m.arc_index in seen_series
                    continue
                end
                push!(seen_series, m.arc_index)
            end
            push!(mods, m)
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

function _read_registered_outages!(f::HDF5.File, cache::Dict{Int, ContingencySpec})
    g = f["outages"]
    ranges, table = _read_row_arrays(g)
    ids = _read(g, "id", Vector{Int})
    labels = _read(g, "label", Vector{String})
    is_islanding = _read(g, "is_islanding", Vector{Bool})
    _check_same_length(
        "outages",
        "start" => ranges,
        "id" => ids,
        "label" => labels,
        "is_islanding" => is_islanding,
    )
    for (id, label, islanding, range) in zip(ids, labels, is_islanding, ranges)
        mods = _arc_modifications(table, range)
        cache[id] = ContingencySpec(
            id,
            NetworkModification(label, mods, ShuntModification[], islanding),
        )
    end
    return
end

function _read_factor_core(f::HDF5.File; linear_solver::String)
    a = HDF5.attributes(f)
    version = _read(a, "format_version", Int)
    if version != FACTOR_FILE_FORMAT_VERSION
        error(
            "Unsupported factor file format version $version; " *
            "this version of PowerNetworkMatrices reads version $FACTOR_FILE_FORMAT_VERSION.",
        )
    end
    bus_ax = _read(f, "bus_axis", Vector{Int})
    arc_from = _read(f, "arc_from", Vector{Int})
    arc_to = _read(f, "arc_to", Vector{Int})
    _check_same_length("arcs", "arc_from" => arc_from, "arc_to" => arc_to)
    arc_ax = collect(zip(arc_from, arc_to))

    reduced_from = _read(f, "reduced_bus_from", Vector{Int})
    reduced_to = _read(f, "reduced_bus_to", Vector{Int})
    _check_same_length(
        "reduced buses",
        "reduced_bus_from" => reduced_from,
        "reduced_bus_to" => reduced_to,
    )
    nrd = NetworkReductionData(;
        reverse_bus_search_map = Dict{Int, Int}(zip(reduced_from, reduced_to)),
    )
    name_to_arc = NAME_TO_ARC()
    g = f["names"]
    types = _read(g, "type", Vector{String})
    names = _read(g, "name", Vector{String})
    name_from = _read(g, "from", Vector{Int})
    name_to = _read(g, "to", Vector{Int})
    _check_same_length(
        "names",
        "type" => types,
        "name" => names,
        "from" => name_from,
        "to" => name_to,
    )
    for (t, n, from, to) in zip(types, names, name_from, name_to)
        T = _branch_type(t)
        entries = get!(name_to_arc, T) do
            DataStructures.SortedDict{String, ARC_ENTRY}()
        end
        entries[n] = (from, to)
    end

    subnetwork_axes = Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}}()
    for k in keys(f["subnetworks"])
        sg = f["subnetworks"][k]
        sub_from = _read(sg, "arc_from", Vector{Int})
        sub_to = _read(sg, "arc_to", Vector{Int})
        _check_same_length("subnetwork $k", "arc_from" => sub_from, "arc_to" => sub_to)
        subnetwork_axes[_read(sg, "ref", Int)] =
            (collect(zip(sub_from, sub_to)), _read(sg, "bus", Vector{Int}))
    end

    return VirtualFactorCore(
        _read(f, "arc_susceptances", Vector{Float64}),
        Set{Int}(_read(f, "ref_bus_positions", Vector{Int})),
        (arc_ax, bus_ax),
        subnetwork_axes,
        _read(f, "arc_live_members", Vector{Int}),
        BranchCatalog(nrd, name_to_arc);
        linear_solver = linear_solver,
        tol = _read_tol(a),
        system_uuid = _restore_system_uuid(_read(a, "system_uuid", String)),
    )
end

"""
    from_hdf5(::Type{VirtualPTDF}, filename; linear_solver, max_cache_size) -> VirtualPTDF

Read a `VirtualPTDF` from a file that [`to_hdf5`](@ref) wrote. The function does not need a
`System`. It factorizes the ABA matrix again with `linear_solver` and resolves the stored
`tol` as a fresh build does.

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
        bus_ax = core.axes[2]
        weights = read(f["dist_slack"])
        dist_slack =
            Dict{Int, Float64}(
                bus_ax[i] => w for (i, w) in enumerate(weights) if !iszero(w)
            )
        return VirtualPTDF(core; dist_slack = dist_slack, max_cache_size = max_cache_size)
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
    return HDF5.h5open(filename, "r") do f
        core = _read_factor_core(f; linear_solver = linear_solver)
        vmodf = VirtualMODF(core; max_cache_size = max_cache_size)
        if haskey(f, "outages")
            _read_registered_outages!(f, get_registered_contingencies(vmodf))
        end
        return vmodf
    end
end
