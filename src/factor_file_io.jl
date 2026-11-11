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

function _branch_type(name::String, path::String)
    haskey(FACTOR_FILE_BRANCH_TYPES, name) ||
        error("Factor file entry $path holds the unknown branch type \"$name\".")
    return FACTOR_FILE_BRANCH_TYPES[name]
end

function _branch_type_name(::Type{T}) where {T}
    name = string(nameof(T))
    haskey(FACTOR_FILE_BRANCH_TYPES, name) ||
        error("Branch type $T has no entry in FACTOR_FILE_BRANCH_TYPES.")
    return name
end

function _write_branch_map!(f::HDF5.File, table::BranchArcTable)
    g = HDF5.create_group(f, "branch_map")
    g["kind"] = table.kind
    g["id"] = table.id
    g["parent"] = table.parent
    g["sign"] = table.sign
    g["b"] = table.b
    g["alpha"] = table.alpha
    return
end

function _write_reduction!(
    f::HDF5.File,
    nr::NetworkReductionData,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
)
    rec = _reduction_records(nr)
    _check_reduction_records(rec, bus_ax, arc_ax, get_reverse_bus_search_map(nr))
    g = HDF5.create_group(f, "reduction")
    for name in _REDUCTION_COLUMNS
        g[string(name)] = rec[name]
    end
    return
end

function _read_reduction(
    f::HDF5.File,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
    budget::ReadBudget,
)
    g = _group(f, "reduction")
    rec = NamedTuple{_REDUCTION_COLUMNS}(
        map(name -> _read(g, string(name), Vector{Int}, budget), _REDUCTION_COLUMNS),
    )
    _check_reduction_records(rec, bus_ax, arc_ax, reduced)
    bus_reduction_map = Dict{Int, Set{Int}}(b => Set{Int}() for b in bus_ax)
    for (from, root) in reduced
        push!(get!(bus_reduction_map, root, Set{Int}()), from)
    end
    for bus in rec.removed_buses
        delete!(bus_reduction_map, bus)
    end
    boundary = Dict{Int, Set{Tuple{Int, Int}}}()
    for (bus, from, to) in zip(rec.boundary_bus, rec.boundary_from, rec.boundary_to)
        push!(get!(boundary, bus, Set{Tuple{Int, Int}}()), (from, to))
    end
    removed_arcs = collect(zip(rec.removed_arc_from, rec.removed_arc_to))
    return NetworkReductionData(;
        irreducible_buses = Set(rec.irreducible_buses),
        bus_reduction_map = bus_reduction_map,
        reverse_bus_search_map = reduced,
        removed_buses = Set(rec.removed_buses),
        removed_arcs = Set(removed_arcs),
        merged_bus_pairs = Dict{Int, Int}(zip(rec.merged_from, rec.merged_to)),
        removed_arc_to_surviving_bus = Dict{Tuple{Int, Int}, Int}(
            zip(zip(rec.survivor_from, rec.survivor_to), rec.survivor_bus),
        ),
        boundary_bus_to_removed_arcs = boundary,
        removed_arc_reduction = Dict{Tuple{Int, Int}, Int}(
            zip(removed_arcs, rec.removed_arc_code),
        ),
        removed_arc_composite = Dict{Tuple{Int, Int}, Tuple{Int, Int}}(
            a => (rec.composite_from[i], rec.composite_to[i]) for
            (i, a) in enumerate(removed_arcs) if
            rec.removed_arc_code[i] == REDUCTION_CODE_DEGREE_TWO
        ),
        reduction_codes = copy(rec.applied),
    )
end

function _write_factor_core(
    f::HDF5.File,
    core::VirtualFactorCore,
    dist_slack::Vector{Float64},
)
    holds_components(core) || error(
        "Cannot write a matrix that was restored from a factor file: it has no " *
        "reduction maps. Write the factor file from the matrix built on the system.",
    )
    nr = get_network_reduction_data(core)
    arc_ax, bus_ax = core.axes
    reduced = get_reverse_bus_search_map(nr)
    write_factor_core(
        f,
        core;
        shift_injection = Float64[arc_dc_shift_injection(nr, arc) for arc in arc_ax],
        dist_slack = dist_slack,
        ref_bus_positions = get_ref_bus_position(core),
        reverse_bus_map = reduced,
    )
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
    _write_strings!(g, "type", types)
    _write_strings!(g, "name", names)
    g["from"] = from
    g["to"] = to

    _write_branch_map!(f, _build_branch_arc_table(nr, arc_ax))
    _write_reduction!(f, nr, bus_ax, arc_ax)
    return
end

_dist_slack(mat::VirtualPTDF) = get_dist_slack(mat)
_dist_slack(::VirtualMODF) = Float64[]

function _write_factor_file_atomically(body, filename::AbstractString, force::Bool)
    if ispath(filename) && !isfile(filename)
        error("Cannot write $filename: the path exists and is not a regular file.")
    end
    if isfile(filename) && !force
        error("File $filename exists. Pass `force = true` to overwrite it.")
    end
    tmp = tempname(dirname(abspath(filename)); cleanup = false)
    try
        HDF5.h5open(body, tmp, "w")
        _publish_factor_file(tmp, filename, force)
    finally
        rm(tmp; force = true)
    end
    return filename
end

function _publish_factor_file(tmp::AbstractString, filename::AbstractString, force::Bool)
    if force
        Base.Filesystem.rename(tmp, filename)
        return
    end
    try
        hardlink(tmp, filename)
    catch e
        if _is_exists_error(e)
            error("File $filename exists. Pass `force = true` to overwrite it.")
        end
        error(_hardlink_failure_message(filename, e))
    end
    return
end

function _hardlink_failure_message(filename::AbstractString, e::Exception)
    return "Cannot publish $filename with a hard link: $(sprint(showerror, e)). " *
           "The target file system may not support hard links. " *
           "Pass `force = true` to write with a move."
end

_is_exists_error(e::Base.IOError) = e.code == Base.UV_EEXIST
_is_exists_error(::Exception) = false

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename; force = false)

Write the factor data of `mat` to the HDF5 file `filename`. The file holds the arrays that
[`from_hdf5`](@ref) needs to rebuild the matrix without a `System`. The function does not
write cached rows. A `VirtualMODF` writes an empty `dist_slack` dataset. A `VirtualMODF` also
writes its registered contingencies. The function writes a temporary file and moves it to
`filename` when the write succeeds, so a failed write does not change an existing file. If
the file exists and `force` is `false`, the function throws an error and never replaces the
file. If `force` is `true` and `filename` is a symbolic link, the function replaces the link
and does not write through it.
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

# The live path rejects partial series outages, so every series or direct row removes its arc.
_opens_arc(::Val{:parallel}) = false
_opens_arc(::Val) = true

# The position of the tripped member in its parallel group. Zero on an arc that is not a
# parallel group.
_group_member(::NetworkReductionData, ::Val, ::Tuple{Int, Int}, ::PSY.ACTransmission) = 0
function _group_member(
    nr::NetworkReductionData,
    ::Val{:parallel},
    arc::Tuple{Int, Int},
    branch::PSY.ACTransmission,
)
    bp = nr.parallel_branch_map[arc]
    member = _outaged_group_member(bp, branch)
    return findfirst(m -> m === member, bp.branches)::Int
end

# The number of members of the parallel group on the arc. Zero on an arc that is not a
# parallel group.
_group_size(::NetworkReductionData, ::Val, ::Tuple{Int, Int}) = 0
_group_size(nr::NetworkReductionData, ::Val{:parallel}, arc::Tuple{Int, Int}) =
    length(nr.parallel_branch_map[arc].branches)

# For each arc row of `mod`: `true` when the outage of `branch` removes all admittance of
# that arc, and the position of the tripped member in the parallel group of that arc. An
# outage that trips every member of a group removes the whole arc.
function _arc_row_flags(
    mat::Union{VirtualPTDF, VirtualMODF},
    branch::PSY.ACTransmission,
    mod::NetworkModification,
)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    by_arc = Dict{Int, Tuple{Bool, Int}}()
    tripped = Dict{Int, Set{Int}}()
    sizes = Dict{Int, Int}()
    for member in _outage_members(branch)
        tag, arc = _resolve_branch_arc(nr, member)
        if tag === :not_found
            continue
        end
        ix = arc_lookup[arc]
        position = _group_member(nr, Val(tag), arc, member)
        get!(by_arc, ix) do
            (_opens_arc(Val(tag)), position)
        end
        push!(get!(Set{Int}, tripped, ix), position)
        sizes[ix] = _group_size(nr, Val(tag), arc)
    end
    for (ix, positions) in tripped
        if length(positions) == sizes[ix]
            by_arc[ix] = (true, last(by_arc[ix]))
        end
    end
    flags = [by_arc[m.arc_index] for m in mod.arc_modifications]
    return first.(flags), last.(flags)
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
    _check_row_values(HDF5.name(g), arc_index, delta_b, delta_shift, opened, n_arc)
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

function _is_absorbed(nr::NetworkReductionData, trf::PSY.ThreeWindingTransformer)
    return all(m -> _resolve_branch_arc(nr, m)[1] === :not_found, _outage_members(trf))
end

_skippable(::UnsupportedOutageError) = true
_skippable(::Any) = false

# An arc with zero susceptance has no DC effect, so its outage has no arc row.
function _drop_zero_susceptance_arcs(mod::NetworkModification)
    kept = ArcModification[m for m in mod.arc_modifications if !iszero(m.delta_b)]
    if length(kept) == length(mod.arc_modifications)
        return mod
    end
    return NetworkModification(
        mod.label,
        kept,
        ShuntModification[m for m in mod.shunt_modifications],
        mod.is_islanding,
    )
end

const _MAX_REJECTION_LINES = 10

function _rejection_summary(rejected::Vector{Pair{String, String}})
    counts = DataStructures.counter(last.(rejected))
    ranked = sort!(collect(counts); by = p -> (-last(p), first(p)))
    lines = ["$(n) x $(msg)" for (msg, n) in first(ranked, _MAX_REJECTION_LINES)]
    if length(ranked) > _MAX_REJECTION_LINES
        push!(lines, "and $(length(ranked) - _MAX_REJECTION_LINES) more")
    end
    return join(lines, "\n")
end

function _write_branch_outages(
    f::HDF5.File,
    mat::Union{VirtualPTDF, VirtualMODF},
    sys::PSY.System,
)
    nr = get_network_reduction_data(mat)
    keys_ = Tuple{String, String}[]
    ids = Int[]
    opens_arc = Bool[]
    members = Int[]
    mods = NetworkModification[]
    absorbed = String[]
    rejected = Pair{String, String}[]
    zero_susceptance = String[]
    partial_zero = String[]
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        if _is_absorbed(nr, branch)
            push!(absorbed, PSY.get_name(branch))
            continue
        end
        mod = try
            NetworkModification(mat, branch)
        catch e
            _skippable(e) || rethrow()
            push!(rejected, PSY.get_name(branch) => sprint(showerror, e))
            continue
        end
        full_mod = mod
        mod = _drop_zero_susceptance_arcs(full_mod)
        if isempty(mod.arc_modifications)
            push!(zero_susceptance, PSY.get_name(branch))
            continue
        end
        if length(mod.arc_modifications) < length(full_mod.arc_modifications)
            push!(partial_zero, PSY.get_name(branch))
        end
        push!(mods, mod)
        push!(keys_, _outage_key(branch))
        push!(ids, IS.get_id(branch))
        flags, positions = _arc_row_flags(mat, branch, mod)
        append!(opens_arc, flags)
        append!(members, positions)
    end
    if !isempty(absorbed)
        @warn "$(length(absorbed)) branches are in no reduction map of the matrix (a network reduction absorbed them, or they are not in service). First: $(first(absorbed))"
    end
    if !isempty(rejected)
        @warn "$(length(rejected)) branches have no outage row because NetworkModification rejected them. First: $(first(first(rejected))).\n$(_rejection_summary(rejected))"
    end
    if !isempty(zero_susceptance)
        @warn "$(length(zero_susceptance)) branches have no outage row because their susceptance is zero, so an outage has no DC effect: $(join(first(zero_susceptance, _MAX_REJECTION_LINES), ", "))"
    end
    if !isempty(partial_zero)
        @warn "$(length(partial_zero)) branches have no row for their zero-susceptance arcs, which have no DC effect. The other arcs keep their rows: $(join(first(partial_zero, _MAX_REJECTION_LINES), ", "))"
    end
    g = HDF5.create_group(f, "branch_outages")
    _write_strings!(g, "type", first.(keys_))
    _write_strings!(g, "name", last.(keys_))
    g["id"] = ids
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods, length(first(get_core(mat).axes)))
    g["opens_arc"] = opens_arc
    g["member"] = members
    return
end

_write_registered_outages(::HDF5.File, ::VirtualPTDF) = nothing

function _write_registered_outages(f::HDF5.File, mat::VirtualMODF)
    registered = @lock get_core(mat).solver_lock copy(get_registered_contingencies(mat))
    if isempty(registered)
        return
    end
    ids = sort!(collect(keys(registered)))
    mods = NetworkModification[registered[id].modification for id in ids]
    g = HDF5.create_group(f, "outages")
    g["id"] = ids
    _write_strings!(g, "label", String[m.label for m in mods])
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods, length(first(get_core(mat).axes)))
    return
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename, sys::PSY.System; force = false, outages = true)

Write the factor data of `mat` as the method without `sys` does. Also write the group
`branch_reason` with the [`BranchReasonTable`](@ref) of every AC branch of `sys`. If `outages`
is `true`, also write the group
`branch_outages` with one row for each branch of `sys` that `NetworkModification(mat, branch)`
accepts, so `from_hdf5(BranchOutageTable, filename)` can build outages by name. A branch that
is in no reduction map of the matrix has no row (a network reduction absorbed it, or it is not
in service). A branch that `NetworkModification` rejects with an `UnsupportedOutageError` has no row.
A branch whose arcs all have zero susceptance has no row either, because its outage has no DC effect.
A branch with some zero-susceptance arcs (a three-winding transformer) has rows for the other arcs only.
The function logs one warning for each case with the count. The warning for rejected branches
also lists the first branch and each distinct message with its count, at most 10 lines. Any other
exception propagates. A `VirtualMODF` also writes the group `outages` with the registered
contingencies. The function writes the file atomically. If `force` is `true` and `filename` is
a symbolic link, the function replaces the link and does not write through it.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString,
    sys::PSY.System;
    force::Bool = false,
    outages::Bool = true,
)
    return _write_factor_file_atomically(filename, force) do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
        _write_branch_reason!(f, BranchReasonTable(mat, sys))
        if outages
            _write_branch_outages(f, mat, sys)
        end
        _write_registered_outages(f, mat)
    end
end

# `extra` holds the arc-row columns that only one group stores. They join the length check.
function _read_row_arrays(
    g::HDF5.Group,
    n_arc::Int,
    budget::ReadBudget,
    extra::Pair{String, <:AbstractVector}...,
)
    starts = _read(g, "start", Vector{Int}, budget)
    lens = _read(g, "len", Vector{Int}, budget)
    arc_index = _read(g, "arc_index", Vector{Int}, budget)
    delta_b = _read(g, "delta_b", Vector{Float64}, budget)
    delta_shift = _read(g, "delta_shift", Vector{Float64}, budget)
    opened = _read(g, "opened", Vector{Int}, budget)
    _check_same_length("rows", "start" => starts, "len" => lens)
    _check_same_length(
        "arc rows",
        "arc_index" => arc_index,
        "delta_b" => delta_b,
        "delta_shift" => delta_shift,
        "opened" => opened,
        extra...,
    )
    _check_row_values(HDF5.name(g), arc_index, delta_b, delta_shift, opened, n_arc)
    n_rows = length(arc_index)
    _check_row_ranges(HDF5.name(g), starts, lens, n_rows)
    ranges = [s:(s + l - 1) for (s, l) in zip(starts, lens)]
    for (k, range) in enumerate(ranges)
        (isempty(range) || checkbounds(Bool, arc_index, range)) ||
            error(
                "Factor file row range $k ($range) is outside the $(length(arc_index)) arc rows.",
            )
    end
    # Overlapping ranges would let a small file allocate one vector per range and row.
    next = 1
    for k in eachindex(ranges)
        starts[k] == next || error(
            "Factor file row range $k starts at $(starts[k]); " *
            "ranges must be contiguous and ordered, so it must start at $next.",
        )
        next += lens[k]
    end
    next == n_rows + 1 || error(
        "Factor file row ranges cover $(next - 1) of the $n_rows arc rows.",
    )
    for (k, range) in enumerate(ranges)
        seen = Set{Int}()
        for row in range
            arc_index[row] in seen && error(
                "Factor file row range $k has the arc_index $(arc_index[row]) more than once, at row $row.",
            )
            push!(seen, arc_index[row])
        end
    end
    return ranges, (; arc_index, delta_b, delta_shift, opened)
end

"""
    from_hdf5(::Type{BranchOutageTable}, filename) -> BranchOutageTable

Read the branch outage table from a file that `to_hdf5(mat, filename, sys)` wrote. The
function throws an error if the file has no such table.
"""
function from_hdf5(::Type{BranchOutageTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        if !haskey(f, "branch_outages")
            error(
                "File $filename has no branch outage table. " *
                "The file was written without a system.",
            )
        end
        g = _group(f, "branch_outages")
        arc_susceptances = _read(f, "arc_susceptances", Vector{Float64}, budget)
        n_arc = length(_read(f, "arc_from", Vector{Int}, budget))
        length(arc_susceptances) == n_arc || error(
            "Factor file entry arc_susceptances has $(length(arc_susceptances)) values for $n_arc arcs.",
        )
        opens_arc = _read(g, "opens_arc", Vector{Bool}, budget)
        member = _read(g, "member", Vector{Int}, budget)
        ranges, arrays = _read_row_arrays(
            g,
            n_arc,
            budget,
            "opens_arc" => opens_arc,
            "member" => member,
        )
        _check_outage_arcs(arrays.arc_index, arc_susceptances)
        # A row that does not open its arc removes one group member, and the reader keys it
        # by that member.
        for (row, m) in enumerate(member)
            opens_arc[row] || m >= 1 ||
                error(
                    "Factor file entry branch_outages/member has the value $m at row $row, " *
                    "a partial row; expected 1 or more.",
                )
        end
        types = _read(g, "type", Vector{String}, budget)
        names = _read(g, "name", Vector{String}, budget)
        for t in types
            _branch_type(t, "branch_outages/type")
        end
        ids = _read(g, "id", Vector{Int}, budget)
        is_islanding = _read(g, "is_islanding", Vector{Bool}, budget)
        _check_same_length(
            "branch_outages",
            "start" => ranges,
            "type" => types,
            "name" => names,
            "id" => ids,
            "is_islanding" => is_islanding,
        )
        _check_unique("branch_outages/id", ids)
        _check_unique("branch_outages/(type, name)", collect(zip(types, names)))
        rows = Dict(zip(zip(types, names), zip(is_islanding, ranges)))
        id_keys = Dict(id => (t, n) for (id, t, n) in zip(ids, types, names))
        return BranchOutageTable(
            rows,
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            opens_arc,
            member,
            id_keys,
            n_arc,
        )
    end
end

function _read_registered_outages!(
    f::HDF5.File,
    cache::Dict{Int, ContingencySpec},
    arc_susceptances::Vector{Float64},
    budget::ReadBudget,
)
    g = _group(f, "outages")
    ranges, arrays = _read_row_arrays(g, length(arc_susceptances), budget)
    _check_outage_arcs(arrays.arc_index, arc_susceptances)
    ids = _read(g, "id", Vector{Int}, budget)
    labels = _read(g, "label", Vector{String}, budget)
    is_islanding = _read(g, "is_islanding", Vector{Bool}, budget)
    _check_same_length(
        "outages",
        "start" => ranges,
        "id" => ids,
        "label" => labels,
        "is_islanding" => is_islanding,
    )
    _check_unique("outages/id", ids)
    for (id, label, islanding, range) in zip(ids, labels, is_islanding, ranges)
        mods = _arc_modifications(
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            range,
        )
        cache[id] = ContingencySpec(
            id,
            NetworkModification(label, mods, ShuntModification[], islanding),
        )
    end
    return
end

"""
    from_hdf5(::Type{BranchArcTable}, filename) -> BranchArcTable

Read the `branch_map` group from a file that [`to_hdf5`](@ref) wrote. The function throws an
error if the format version is not 1, the group is missing, or a column breaks a rule of
the table.
"""
function from_hdf5(::Type{BranchArcTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        arc_from = _read(f, "arc_from", Vector{Int}, budget)
        arc_to = _read(f, "arc_to", Vector{Int}, budget)
        _check_same_length("arcs", "arc_from" => arc_from, "arc_to" => arc_to)
        g = _group(f, "branch_map")
        kind = _read(g, "kind", Vector{Int}, budget)
        id = _read(g, "id", Vector{Int}, budget)
        parent = _read(g, "parent", Vector{Int}, budget)
        sign = _read(g, "sign", Vector{Int}, budget)
        b = _read(g, "b", Vector{Float64}, budget)
        alpha = _read(g, "alpha", Vector{Float64}, budget)
        n_arc = length(arc_from)
        table = BranchArcTable(kind, id, parent, sign, b, alpha, n_arc)
        _check_branch_map(table)
        return table
    end
end

function _read_factor_core(f::HDF5.File, budget::ReadBudget; linear_solver::String)
    arrays = read_factor_arrays(f, budget)
    arc_ax, bus_ax = arrays.axes

    reduced_from = _read(f, "reduced_bus_from", Vector{Int}, budget)
    reduced_to = _read(f, "reduced_bus_to", Vector{Int}, budget)
    _check_same_length(
        "reduced buses",
        "reduced_bus_from" => reduced_from,
        "reduced_bus_to" => reduced_to,
    )
    _check_unique("reduced_bus_from", reduced_from)
    bus_set = Set(bus_ax)
    for from in reduced_from
        from in bus_set &&
            error(
                "Factor file entry reduced_bus_from holds bus $from, which is on the bus axis.",
            )
    end
    nrd = _read_reduction(
        f,
        bus_ax,
        arc_ax,
        Dict{Int, Int}(zip(reduced_from, reduced_to)),
        budget,
    )
    name_to_arc = NAME_TO_ARC()
    g = _group(f, "names")
    types = _read(g, "type", Vector{String}, budget)
    names = _read(g, "name", Vector{String}, budget)
    name_from = _read(g, "from", Vector{Int}, budget)
    name_to = _read(g, "to", Vector{Int}, budget)
    _check_same_length(
        "names",
        "type" => types,
        "name" => names,
        "from" => name_from,
        "to" => name_to,
    )
    _check_unique("names/(type, name)", collect(zip(types, names)))
    arc_set = Set(arc_ax)
    for (row, (n, from, to)) in enumerate(zip(names, name_from, name_to))
        (from, to) in arc_set || error(
            "Factor file group names maps $n to ($from, $to) at row $row, which is not an arc.",
        )
    end
    for (t, n, from, to) in zip(types, names, name_from, name_to)
        T = _branch_type(t, "names/type")
        entries = get!(name_to_arc, T) do
            DataStructures.SortedDict{String, ARC_ENTRY}()
        end
        entries[n] = (from, to)
    end

    return VirtualFactorCore(
        arrays.arc_susceptances,
        arrays.ref_bus_positions,
        arrays.axes,
        arrays.subnetwork_axes,
        arrays.arc_live_members,
        BranchCatalog(nrd, name_to_arc);
        linear_solver = linear_solver,
        tol = arrays.tol,
        system_uuid = arrays.system_uuid,
    )
end

"""
    from_hdf5(::Type{VirtualPTDF}, filename; linear_solver, max_cache_size, persistent_arcs) -> VirtualPTDF

Read a `VirtualPTDF` from a file that [`to_hdf5`](@ref) wrote. The function does not need a
`System`. It factorizes the ABA matrix again with `linear_solver` and resolves the stored
`tol` as a fresh build does. The file holds no cache settings. `max_cache_size` and
`persistent_arcs` have the same meaning as in the `VirtualPTDF` constructor.

The restored matrix has no components and no reduction maps. `NetworkModification(mat, branch)`,
`NetworkModification(mat, arc)` and `NetworkModification(mat, sys, outage)` throw an error.
`get_branch_susceptances_by_arc` and `to_hdf5` throw an error. `mat[name, bus]` does not work.
The function throws an error if the format version of the file is not 1.
"""
function from_hdf5(
    ::Type{VirtualPTDF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        core = _read_factor_core(f, budget; linear_solver = linear_solver)
        bus_ax = core.axes[2]
        weights = read_dist_slack(f, core, budget)
        dist_slack =
            Dict{Int, Float64}(
                bus_ax[i] => w for (i, w) in enumerate(weights) if !iszero(w)
            )
        return VirtualPTDF(
            core;
            dist_slack = dist_slack,
            max_cache_size = max_cache_size,
            persistent_arcs = persistent_arcs,
        )
    end
end

"""
    from_hdf5(::Type{VirtualMODF}, filename; linear_solver, max_cache_size) -> VirtualMODF

Read a `VirtualMODF` from a file that [`to_hdf5`](@ref) wrote. If the file has the group
`outages`, the restored matrix has those registered contingencies. Their modifications serve
the DC row kernels only. The restored matrix has the same limits as a restored `VirtualPTDF`:
`NetworkModification` from a branch or outage, `get_branch_susceptances_by_arc` and `to_hdf5`
throw an error.
"""
function from_hdf5(
    ::Type{VirtualMODF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        core = _read_factor_core(f, budget; linear_solver = linear_solver)
        vmodf = VirtualMODF(core; max_cache_size = max_cache_size)
        if haskey(f, "outages")
            _read_registered_outages!(
                f,
                get_registered_contingencies(vmodf),
                core.arc_susceptances,
                budget,
            )
        end
        return vmodf
    end
end

function _write_branch_reason!(f::HDF5.File, t::BranchReasonTable)
    rows = sort!(collect(t.rows); by = first)
    id = [first(first(r)) for r in rows]
    winding = [last(first(r)) for r in rows]
    reason = [first(last(r)) for r in rows]
    surviving_bus = [last(last(r)) for r in rows]
    _check_branch_reason(id, winding, reason, surviving_bus)
    g = HDF5.create_group(f, "branch_reason")
    g["id"] = id
    g["winding"] = winding
    g["reason"] = reason
    g["surviving_bus"] = surviving_bus
    return
end

"""
    from_hdf5(::Type{BranchReasonTable}, filename) -> BranchReasonTable

Read the `branch_reason` group that `to_hdf5(mat, filename, sys)` wrote. The function throws an
error if the file has no such group, because a file written without a `System` cannot name
its branches. It also throws an error if the format version is not 1 or a column breaks a
rule of the table.
"""
function from_hdf5(::Type{BranchReasonTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        haskey(f, "branch_reason") || error(
            "Factor file $filename has no group branch_reason: it was written without a " *
            "System. Write it with `to_hdf5(mat, filename, sys)`.",
        )
        g = _group(f, "branch_reason")
        id = _read(g, "id", Vector{Int}, budget)
        winding = _read(g, "winding", Vector{Int}, budget)
        reason = _read(g, "reason", Vector{Int}, budget)
        surviving_bus = _read(g, "surviving_bus", Vector{Int}, budget)
        _check_branch_reason(id, winding, reason, surviving_bus)
        return BranchReasonTable(
            Dict(zip(zip(id, winding), zip(reason, surviving_bus))),
        )
    end
end
