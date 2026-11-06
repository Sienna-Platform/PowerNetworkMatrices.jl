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
        PSY.SeriesCompensator,
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

function _write_factor_file_atomically(body, filename::AbstractString, force::Bool)
    if isfile(filename) && !force
        error("File $filename exists. Pass `force = true` to overwrite it.")
    end
    tmp = tempname(dirname(abspath(filename)); cleanup = false)
    try
        HDF5.h5open(body, tmp, "w")
        mv(tmp, filename; force = true)
    catch
        rm(tmp; force = true)
        rethrow()
    end
    return filename
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename; force = false)

Write the factor data of `mat` to the HDF5 file `filename`. The file holds the arrays that
[`from_hdf5`](@ref) needs to rebuild the matrix without a `System`. The function does not
write cached rows. A `VirtualMODF` writes an empty `dist_slack` dataset. A `VirtualMODF` also
writes its registered contingencies. The function writes a temporary file and moves it to
`filename` when the write succeeds, so a failed write does not change an existing file. If
the file exists and `force` is `false`, the function throws an error.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString;
    force::Bool = false,
)
    return _write_factor_file_atomically(filename, force) do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
        _write_registered_outages(f, mat)
    end
end

_outage_key(branch::PSY.ACTransmission) =
    (_branch_type_name(typeof(branch)), PSY.get_name(branch))

# The members that `NetworkModification(mat, branch)` classifies through the reverse maps.
_outage_members(branch::PSY.ACTransmission) = PSY.ACTransmission[branch]

function _outage_members(trf::PSY.ThreeWindingTransformer)
    windings = PSY.ACTransmission[
        ThreeWindingTransformerCircuit(trf, circuit, i) for
        (i, circuit) in enumerate(PSY.get_circuits(trf))
    ]
    return filter(get_equivalent_available, windings)
end

function _is_series(nr::NetworkReductionData, branch::PSY.ACTransmission)
    return any(
        member -> first(_resolve_branch_arc(nr, member)) === :series,
        _outage_members(branch),
    )
end

_opens_arc(::Val{:direct}, ::NetworkReductionData, ::Tuple{Int, Int}, _) = true
_opens_arc(::Val{:parallel}, ::NetworkReductionData, ::Tuple{Int, Int}, _) = false

# The same predicate that makes `_compute_series_outage_delta_b` return `-b_old`.
function _opens_arc(
    ::Val{:series},
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
    tripped::Vector{PSY.ACTransmission},
)
    tripped_set = Set{PSY.ACTransmission}(tripped)
    return any(
        segment -> iszero(_segment_susceptance_after_outage(segment, tripped_set, nr)),
        nr.series_branch_map[arc],
    )
end

# One flag for each arc row of `mod`: `true` when the outage of `branch` removes all
# admittance of that arc.
function _opens_arc_flags(
    mat::Union{VirtualPTDF, VirtualMODF},
    branch::PSY.ACTransmission,
    mod::NetworkModification,
)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    by_arc = Dict{Int, Tuple{Symbol, Tuple{Int, Int}, Vector{PSY.ACTransmission}}}()
    for member in _outage_members(branch)
        tag, arc = _resolve_branch_arc(nr, member)
        if tag === :not_found
            continue
        end
        entry = get!(by_arc, arc_lookup[arc], (tag, arc, PSY.ACTransmission[]))
        push!(entry[3], member)
    end
    return map(collect(mod.arc_modifications)) do m
        tag, arc, members = by_arc[m.arc_index]
        return _opens_arc(Val(tag), nr, arc, members)
    end
end

function _write_row_arrays(
    g::HDF5.Group,
    mods::Vector{NetworkModification},
    n_arc::Int,
)
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
    _check_row_values(arc_index, delta_b, delta_shift, opened, n_arc)
    g["start"] = starts
    g["len"] = lens
    g["arc_index"] = arc_index
    g["delta_b"] = delta_b
    g["delta_shift"] = delta_shift
    g["opened"] = opened
    return
end

_is_absorbed(nr::NetworkReductionData, branch::PSY.ACTransmission) =
    _resolve_branch_arc(nr, branch)[1] === :not_found

_is_absorbed(::NetworkReductionData, ::PSY.ThreeWindingTransformer) = false

_skippable(::ErrorException) = true
_skippable(::Any) = false

function _write_branch_outages(
    f::HDF5.File,
    mat::Union{VirtualPTDF, VirtualMODF},
    sys::PSY.System,
)
    nr = get_network_reduction_data(mat)
    keys_ = Tuple{String, String}[]
    is_series = Bool[]
    opens_arc = Bool[]
    mods = NetworkModification[]
    absorbed = String[]
    rejected = String[]
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        if _is_absorbed(nr, branch)
            push!(absorbed, PSY.get_name(branch))
            continue
        end
        mod = try
            NetworkModification(mat, branch)
        catch e
            _skippable(e) || rethrow()
            push!(rejected, "$(PSY.get_name(branch)): $(sprint(showerror, e))")
            continue
        end
        push!(mods, mod)
        push!(keys_, _outage_key(branch))
        push!(is_series, _is_series(nr, branch))
        append!(opens_arc, _opens_arc_flags(mat, branch, mod))
    end
    if !isempty(absorbed)
        @warn "$(length(absorbed)) branches are in no reduction map of the matrix (a network reduction absorbed them, or they are not in service). First: $(first(absorbed))"
    end
    if !isempty(rejected)
        @warn "$(length(rejected)) branches have no outage row because NetworkModification rejected them. First: $(first(rejected))"
    end
    g = HDF5.create_group(f, "branch_outages")
    g["type"] = first.(keys_)
    g["name"] = last.(keys_)
    g["is_series"] = is_series
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods, length(first(get_core(mat).axes)))
    g["opens_arc"] = opens_arc
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
    _write_row_arrays(g, mods, length(first(get_core(mat).axes)))
    return
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename, sys::PSY.System; force = false)

Write the factor data of `mat` as the method without `sys` does. Also write the group
`branch_outages` with one row for each branch of `sys` that `NetworkModification(mat, branch)`
accepts, so `from_hdf5(BranchOutageTable, filename)` can build outages by name. A branch that
is in no reduction map of the matrix has no row (a network reduction absorbed it, or it is not
in service). A branch that `NetworkModification` rejects with an `ErrorException` has no row. The function logs one warning for each case with the count and
the first branch. Any other exception propagates. A `VirtualMODF` also writes the group
`outages` with the registered contingencies. The function writes the file atomically.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString,
    sys::PSY.System;
    force::Bool = false,
)
    _validate_system_uuid(mat, sys)
    return _write_factor_file_atomically(filename, force) do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
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
- `rows`: for the key `(type, name)`, the series flag, the islanding flag and the row range.
  The series flag is `true` when the branch is a member of a series chain.
- `arc_index`, `delta_b`, `delta_shift`, `opened`: the arrays that the row ranges index.
- `opens_arc`: for each arc row, `true` when the outage of that one branch removes all
  admittance of the arc.
- `n_arc`: the number of arcs in the file. Each `arc_index` is at most `n_arc`.
"""
struct BranchOutageTable
    rows::Dict{Tuple{String, String}, Tuple{Bool, Bool, UnitRange{Int}}}
    arc_index::Vector{Int}
    delta_b::Vector{Float64}
    delta_shift::Vector{Float64}
    opened::Vector{Int}
    opens_arc::Vector{Bool}
    n_arc::Int
end

function _check_row_values(
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    n_arc::Int,
)
    all(i -> 1 <= i <= n_arc, arc_index) ||
        error("Factor file entry arc_index has a value outside 1:$n_arc.")
    all(b -> isfinite(b) && !iszero(b), delta_b) ||
        error("Factor file entry delta_b has a value that is zero or not finite.")
    all(isfinite, delta_shift) ||
        error("Factor file entry delta_shift has a value that is not finite.")
    all(>=(0), opened) || error("Factor file entry opened has a negative value.")
    return
end

# `extra` holds the arc-row columns that only one group stores. They join the length check.
function _read_row_arrays(
    g::HDF5.Group,
    n_arc::Int,
    extra::Pair{String, <:AbstractVector}...,
)
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
        extra...,
    )
    _check_row_values(arc_index, delta_b, delta_shift, opened, n_arc)
    ranges = [s:(s + l - 1) for (s, l) in zip(starts, lens)]
    for range in ranges
        (isempty(range) || checkbounds(Bool, arc_index, range)) ||
            error(
                "Factor file row range $range is outside the $(length(arc_index)) arc rows.",
            )
    end
    return ranges, (; arc_index, delta_b, delta_shift, opened)
end

# `arrays` is a `BranchOutageTable` or the named tuple from `_read_row_arrays`.
function _arc_modifications(arrays, rows::AbstractVector{Int})
    z = zero(YBUS_ELTYPE)
    return ArcModification[
        ArcModification(
            arrays.arc_index[i],
            arrays.delta_b[i],
            arrays.delta_shift[i],
            z,
            z,
            z,
            z,
            arrays.opened[i],
        ) for i in rows
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
        n_arc = length(_read(f, "arc_from", Vector{Int}))
        opens_arc = _read(g, "opens_arc", Vector{Bool})
        ranges, arrays = _read_row_arrays(g, n_arc, "opens_arc" => opens_arc)
        types = _read(g, "type", Vector{String})
        names = _read(g, "name", Vector{String})
        is_series = _read(g, "is_series", Vector{Bool})
        is_islanding = _read(g, "is_islanding", Vector{Bool})
        _check_same_length(
            "branch_outages",
            "start" => ranges,
            "type" => types,
            "name" => names,
            "is_series" => is_series,
            "is_islanding" => is_islanding,
        )
        rows = Dict{Tuple{String, String}, Tuple{Bool, Bool, UnitRange{Int}}}()
        for (t, n, series, islanding, range) in
            zip(types, names, is_series, is_islanding, ranges)
            rows[(t, n)] = (series, islanding, range)
        end
        return BranchOutageTable(
            rows,
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            opens_arc,
            n_arc,
        )
    end
end

"""
    NetworkModification(table::BranchOutageTable, type, name)
    NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})

Build the outage of one branch, or of several branches at once, from a
[`BranchOutageTable`](@ref). `type` is a `String`, for example
`"Line"`. The function groups the arc rows of all keys by arc:

  - If a row opens the whole arc, the arc gets the first such row only.
  - If no row opens the arc and two or more rows come from members of a series chain, the
    function throws an error. The table cannot combine partial outages of one chain. Build
    that outage from the `System` with `NetworkModification(mat, sys, outage)`.
  - Otherwise the arc gets every row. Partial outages of parallel members add.

The modification has no Ybus deltas and no shunt modifications. It serves the DC row
kernels only.
"""
function NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})
    if isempty(keys)
        error("An outage needs at least one branch key.")
    end
    allunique(keys) || error(
        "Duplicate branch key in outage: $(collect(Set(k for k in keys if count(==(k), keys) > 1))).",
    )
    by_arc = DataStructures.OrderedDict{Int, Vector{Tuple{Int, Bool, String}}}()
    islanding = false
    for key in keys
        if !haskey(table.rows, key)
            error("Branch $(key[1]) $(key[2]) has no row in the outage table.")
        end
        is_series, is_islanding, range = table.rows[key]
        islanding |= is_islanding
        for i in range
            rows = get!(by_arc, table.arc_index[i], Tuple{Int, Bool, String}[])
            push!(rows, (i, is_series, "$(key[1]) $(key[2])"))
        end
    end
    picked = Int[]
    for rows in values(by_arc)
        append!(picked, _combined_rows(table, rows))
    end
    return NetworkModification(
        join((key[2] for key in keys), "+"),
        _arc_modifications(table, picked),
        ShuntModification[],
        islanding,
    )
end

# `rows` holds `(row, is_series, branch)` for each arc row of one arc.
function _combined_rows(table::BranchOutageTable, rows::Vector{Tuple{Int, Bool, String}})
    for (i, _, _) in rows
        if table.opens_arc[i]
            return [i]
        end
    end
    series = [branch for (_, is_series, branch) in rows if is_series]
    if length(series) >= 2
        error(
            "Branches $(join(series, ", ")) are partial outages of one series chain. " *
            "The outage table cannot combine them. Build the outage from the System with " *
            "NetworkModification(mat, sys, outage).",
        )
    end
    return first.(rows)
end

function NetworkModification(table::BranchOutageTable, type::String, name::String)
    return NetworkModification(table, [(type, name)])
end

function _group(f::HDF5.File, name::String)
    haskey(f, name) || error("Factor file has no entry $name.")
    return f[name]
end

function _read_registered_outages!(
    f::HDF5.File,
    cache::Dict{Int, ContingencySpec},
    n_arc::Int,
)
    g = _group(f, "outages")
    ranges, arrays = _read_row_arrays(g, n_arc)
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
        mods = _arc_modifications(arrays, range)
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
    g = _group(f, "names")
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
    subnetworks = _group(f, "subnetworks")
    for k in keys(subnetworks)
        sg = subnetworks[k]
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
        weights = _read(f, "dist_slack", Vector{Float64})
        isempty(weights) || _check_same_length(
            "dist_slack",
            "bus_axis" => bus_ax,
            "dist_slack" => weights,
        )
        all(isfinite, weights) ||
            error("Factor file entry dist_slack has a value that is not finite.")
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
            _read_registered_outages!(
                f,
                get_registered_contingencies(vmodf),
                length(core.arc_susceptances),
            )
        end
        return vmodf
    end
end
