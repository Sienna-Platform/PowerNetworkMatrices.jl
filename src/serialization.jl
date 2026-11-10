# Columns of one table must agree in length, or `zip` would drop rows in silence.
function _check_same_length(table::String, columns::Pair{String, <:AbstractVector}...)
    lengths = map(column -> first(column) => length(last(column)), columns)
    allequal(map(last, lengths)) ||
        error("Factor file table $table has columns of unequal length: $lengths.")
    return
end

function _check_dist_slack(weights::Vector{Float64}, bus_ax::Vector{Int})
    isempty(weights) && return
    _check_same_length("dist_slack", "bus_axis" => bus_ax, "dist_slack" => weights)
    for (i, w) in enumerate(weights)
        isfinite(w) || error(
            "Factor file entry dist_slack has the value $w at position $i; " *
            "the value must be finite.",
        )
        w >= 0 || error(
            "Factor file entry dist_slack has the weight $w at position $i; " *
            "the weight must not be negative.",
        )
    end
    total = sum(weights)
    (isfinite(total) && total > 0) || error(
        "Factor file entry dist_slack has the sum $total; the sum must be finite and positive.",
    )
    return
end

"""
    BranchArcTable

Columns of the `branch_map` group of a factor file: one row per node of the reduction tree
of each arc, in depth-first preorder. `n_arc` is the number of arcs. See the factor file
guide for the meaning of each column.
"""
struct BranchArcTable
    kind::Vector{Int}
    id::Vector{Int}
    parent::Vector{Int}
    sign::Vector{Int}
    b::Vector{Float64}
    alpha::Vector{Float64}
    n_arc::Int
end

"Return the `kind` column of `t`: the node kind of each row."
get_branch_kind(t::BranchArcTable) = t.kind

"Return the `id` column of `t`: the `IS.get_id` of the branch of each row, or 0."
get_branch_id(t::BranchArcTable) = t.id

"Return the `parent` column of `t`: the parent row, or minus the arc index for a top node."
get_branch_parent(t::BranchArcTable) = t.parent

"Return the `sign` column of `t`: the orientation of each row against its parent."
get_branch_sign(t::BranchArcTable) = t.sign

"Return the `b` column of `t`: the `BA_Matrix` susceptance of each leaf or Ward row, in system base."
get_branch_susceptance(t::BranchArcTable) = t.b

"Return the `alpha` column of `t`: the phase shift of each leaf row, in radians."
get_branch_alpha(t::BranchArcTable) = t.alpha

"Return the number of arcs of `t`."
get_branch_arc_count(t::BranchArcTable) = t.n_arc

# A magnitude outside this range makes the series and parallel sums overflow or vanish.
_in_susceptance_range(b::Float64) = 1 / MAX_ARC_SUSCEPTANCE <= abs(b) <= MAX_ARC_SUSCEPTANCE

function _check_branch_map(t::BranchArcTable)
    (; kind, id, parent, sign, b, alpha, n_arc) = t
    _check_same_length(
        "branch_map",
        "kind" => kind, "id" => id, "parent" => parent,
        "sign" => sign, "b" => b, "alpha" => alpha,
    )
    n = length(kind)
    has_child = falses(n)
    has_top = falses(n_arc)
    leaf_keys = Set{Tuple{Int, Int}}()
    for r in 1:n
        k = kind[r]
        p = parent[r]
        fail(what) = error("Factor file table branch_map row $r (kind $k): $what.")
        1 <= k <= 11 || fail("kind is not in 1:11")
        iszero(p) && fail("parent is 0")
        p >= -n_arc || fail("parent $p is not an arc of the $n_arc arcs")
        p < r || fail("parent $p is not below the row")
        if p > 0
            (kind[p] == 9 || kind[p] == 10) ||
                fail("parent row $p has kind $(kind[p]), not an aggregate")
            has_child[p] = true
        else
            has_top[-p] = true
        end
        (k == 11 && p > 0) && fail("a Ward equivalent must be a top node")
        (sign[r] == 1 || sign[r] == -1) || fail("sign is not 1 or -1")
        if k <= 8
            id[r] > 0 || fail("id is not positive")
            # Exact zero is BA's "no DC coupling", for example a purely resistive branch. BA
            # uses component values inside a series chain, which are never zero.
            in_chain = p > 0 && kind[p] == 10
            ((iszero(b[r]) && !in_chain) || _in_susceptance_range(b[r])) ||
                fail(
                    "b is $(b[r]); expected 0 outside a series chain or a magnitude from $(1 / MAX_ARC_SUSCEPTANCE) to $MAX_ARC_SUSCEPTANCE",
                )
            isfinite(alpha[r]) || fail("alpha is $(alpha[r]); expected a finite value")
            key = (id[r], max(k - 5, 0))
            key in leaf_keys && fail("id and winding $key repeat")
            push!(leaf_keys, key)
        elseif k == 11
            iszero(id[r]) || fail("id is not 0")
            # A Ward row is a top node, so no flow inverts its b: only the upper bound applies.
            (isfinite(b[r]) && abs(b[r]) <= MAX_ARC_SUSCEPTANCE) ||
                fail(
                    "b is $(b[r]); expected a finite magnitude of at most $MAX_ARC_SUSCEPTANCE",
                )
            iszero(alpha[r]) || fail("alpha is not 0")
        else
            iszero(id[r]) || fail("id is not 0")
            iszero(b[r]) || fail("b is not 0")
            iszero(alpha[r]) || fail("alpha is not 0")
        end
    end
    for r in 1:n
        (kind[r] == 9 || kind[r] == 10) && !has_child[r] &&
            error("Factor file table branch_map row $r (kind $(kind[r])): no child.")
    end
    for k in 1:n_arc
        has_top[k] || error("Factor file table branch_map: arc $k has no top node.")
    end
    return
end

_branch_map_kind(::PSY.Line) = 1
_branch_map_kind(::PSY.TwoWindingTransformer) = 2
_branch_map_kind(::PSY.DiscreteControlledACBranch) = 3
_branch_map_kind(::PSY.GenericArcImpedance) = 4
_branch_map_kind(::PSY.SeriesCompensator) = 5
_branch_map_kind(w::ThreeWindingTransformerCircuit) = 5 + get_winding_number(w)
_branch_map_kind(::AbstractBranchesParallel) = 9
_branch_map_kind(::BranchesSeries) = 10

_branch_map_id(x::PSY.ACTransmission) = IS.get_id(x)
_branch_map_id(w::ThreeWindingTransformerCircuit) = IS.get_id(get_transformer(w))
_branch_map_id(::AbstractReductionAggregate) = 0

function _frame_sign(frame::Tuple{Int, Int}, own::Tuple{Int, Int})
    own == frame && return 1
    own == (frame[2], frame[1]) && return -1
    return error("Arc $own is not arc $frame in either orientation.")
end

function _push_row!(
    rows::BranchArcTable,
    kind::Int,
    id::Int,
    parent::Int,
    sign::Int,
    b::Float64,
    alpha::Float64,
)
    push!(rows.kind, kind)
    push!(rows.id, id)
    push!(rows.parent, parent)
    push!(rows.sign, sign)
    push!(rows.b, b)
    push!(rows.alpha, alpha)
    return length(rows.kind)
end

# `b_of` gives the susceptance that `BA_Matrix` uses for the node: the two-port value of the
# node itself, or the component value inside a series chain, because BA sums the components
# of a chain (`_ba_arc_susceptance(::BranchesSeries, nr)`).
function _push_node!(
    rows::BranchArcTable,
    leaf::PSY.ACTransmission,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
    b_of::F,
) where {F <: Function}
    _push_row!(
        rows, _branch_map_kind(leaf), _branch_map_id(leaf), parent, sign,
        b_of(leaf, nr), _segment_phase_shift(leaf, nr),
    )
    return
end

function _push_node!(
    rows::BranchArcTable,
    bp::AbstractBranchesParallel,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
    b_of::F,
) where {F <: Function}
    row = _push_row!(rows, 9, 0, parent, sign, 0.0, 0.0)
    frame = get_arc_tuple(bp, nr)
    for member in bp
        _push_node!(
            rows, member, row, _frame_sign(frame, get_arc_tuple(member, nr)), nr, b_of,
        )
    end
    return
end

function _push_node!(
    rows::BranchArcTable,
    bs::BranchesSeries,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
    ::F,
) where {F <: Function}
    row = _push_row!(rows, 10, 0, parent, sign, 0.0, 0.0)
    orientations = get_segment_orientations(bs)
    length(orientations) == length(bs) || error(
        "Series chain on arc $(get_arc_tuple(bs, nr)) has $(length(bs)) segments " *
        "and $(length(orientations)) orientations.",
    )
    for (segment, orientation) in zip(bs, orientations)
        if orientation == :FromTo
            seg_sign = 1
        elseif orientation == :ToFrom
            seg_sign = -1
        else
            error(
                "Series chain segment orientation $orientation is not :FromTo or :ToFrom.",
            )
        end
        _push_node!(rows, segment, row, seg_sign, nr, _finite_series_susceptance)
    end
    return
end

function _push_ward!(
    rows::BranchArcTable,
    ward::PSY.GenericArcImpedance,
    parent::Int,
    nr::NetworkReductionData,
)
    _push_row!(rows, 11, 0, parent, 1, _ba_arc_susceptance(ward, nr), 0.0)
    return
end

function _build_branch_arc_table(nr::NetworkReductionData, arc_ax::Vector{Tuple{Int, Int}})
    table = BranchArcTable(Int[], Int[], Int[], Int[], Float64[], Float64[], length(arc_ax))
    added = get_added_arc_impedance_map(nr)
    for (k, arc) in enumerate(arc_ax)
        entry, _ = _resolve_arc_entry(nr, arc)
        has_ward = haskey(added, arc)
        if has_ward && added[arc] === entry
            _push_ward!(table, entry, -k, nr)
        else
            _push_node!(
                table, entry, -k, _frame_sign(arc, get_arc_tuple(entry, nr)), nr,
                _ba_arc_susceptance,
            )
            has_ward && _push_ward!(table, added[arc], -k, nr)
        end
    end
    _check_branch_map(table)
    return table
end

const _REDUCTION_COLUMNS = (
    :applied,
    :removed_arc_from, :removed_arc_to, :removed_arc_code,
    :composite_from, :composite_to,
    :survivor_from, :survivor_to, :survivor_bus,
    :boundary_bus, :boundary_from, :boundary_to,
    :removed_buses,
    :merged_from, :merged_to,
    :irreducible_buses,
)

const _ReductionRecords = NamedTuple{_REDUCTION_COLUMNS, NTuple{16, Vector{Int}}}

function _reduction_records(nr::NetworkReductionData)::_ReductionRecords
    removed = sort!(collect(keys(nr.removed_arc_reduction)))
    survivors = sort!(collect(keys(nr.removed_arc_to_surviving_bus)))
    boundary = sort!([
        (bus, arc) for (bus, arcs) in nr.boundary_bus_to_removed_arcs for arc in arcs
    ])
    merged = sort!(collect(nr.merged_bus_pairs))
    return (
        applied = copy(nr.reduction_codes),
        removed_arc_from = first.(removed),
        removed_arc_to = last.(removed),
        removed_arc_code = [nr.removed_arc_reduction[a] for a in removed],
        composite_from = [first(get(nr.removed_arc_composite, a, (0, 0))) for a in removed],
        composite_to = [last(get(nr.removed_arc_composite, a, (0, 0))) for a in removed],
        survivor_from = first.(survivors),
        survivor_to = last.(survivors),
        survivor_bus = [nr.removed_arc_to_surviving_bus[a] for a in survivors],
        boundary_bus = first.(boundary),
        boundary_from = [first(last(x)) for x in boundary],
        boundary_to = [last(last(x)) for x in boundary],
        removed_buses = sort!(collect(nr.removed_buses)),
        merged_from = first.(merged),
        merged_to = last.(merged),
        irreducible_buses = sort!(collect(nr.irreducible_buses)),
    )
end

function _check_unique(path::String, values::Vector{T}) where {T}
    seen = Set{T}()
    for (i, v) in enumerate(values)
        v in seen &&
            error(
                "Factor file entry $path has the value $v more than once, at row $i.",
            )
        push!(seen, v)
    end
    return
end

function _check_known_bus(name::String, bus::Int, i::Int, known::Set{Int})
    bus in known || error(
        "Factor file entry reduction/$name holds bus $bus at row $i, which is not on the bus " *
        "axis, in the reduced-bus map or in removed_buses.",
    )
    return
end

function _check_known_buses(name::String, buses::Vector{Int}, known::Set{Int})
    for (i, bus) in enumerate(buses)
        _check_known_bus(name, bus, i, known)
    end
    return
end

# One validator serves the writer and the reader.
function _check_reduction_records(
    rec::_ReductionRecords,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
)
    _check_same_length(
        "reduction removed_arc",
        "removed_arc_from" => rec.removed_arc_from,
        "removed_arc_to" => rec.removed_arc_to,
        "removed_arc_code" => rec.removed_arc_code,
        "composite_from" => rec.composite_from,
        "composite_to" => rec.composite_to,
    )
    _check_same_length(
        "reduction survivor",
        "survivor_from" => rec.survivor_from,
        "survivor_to" => rec.survivor_to,
        "survivor_bus" => rec.survivor_bus,
    )
    _check_same_length(
        "reduction boundary",
        "boundary_bus" => rec.boundary_bus,
        "boundary_from" => rec.boundary_from,
        "boundary_to" => rec.boundary_to,
    )
    _check_same_length(
        "reduction merged",
        "merged_from" => rec.merged_from,
        "merged_to" => rec.merged_to,
    )
    for code in rec.applied
        1 <= code <= 4 ||
            error("Factor file entry reduction/applied holds code $code; expected 1 to 4.")
    end
    _check_unique("reduction/applied", rec.applied)
    iszero(count(==(REDUCTION_CODE_WARD), rec.applied)) ||
        last(rec.applied) == REDUCTION_CODE_WARD ||
        error("Factor file entry reduction/applied has the Ward reduction before the end.")
    for (i, code) in enumerate(rec.removed_arc_code)
        code in rec.applied || error(
            "Factor file entry reduction/removed_arc_code row $i holds code $code, " *
            "which is not in applied.",
        )
    end
    removed_buses = Set(rec.removed_buses)
    _check_unique("reduction/removed_buses", rec.removed_buses)
    bus_set = Set(bus_ax)
    for bus in rec.removed_buses
        bus in bus_set && error(
            "Factor file entry reduction/removed_buses holds bus $bus, which is on the bus axis.",
        )
    end
    known = union(bus_set, keys(reduced), removed_buses)
    for root in values(reduced)
        root in bus_set || root in removed_buses ||
            error(
                "Factor file entry reduced_bus_to holds bus $root, which is not on the bus axis or in removed_buses.",
            )
    end
    removed_arcs = collect(zip(rec.removed_arc_from, rec.removed_arc_to))
    _check_unique("reduction/removed_arc", removed_arcs)
    _check_known_buses("removed_arc_from", rec.removed_arc_from, known)
    _check_known_buses("removed_arc_to", rec.removed_arc_to, known)
    removed_set = Set(removed_arcs)
    arc_set = Set(arc_ax)
    for (i, arc) in enumerate(removed_arcs)
        (arc in arc_set || (arc[2], arc[1]) in arc_set) && error(
            "Factor file entry reduction/removed_arc row $i holds arc $arc, which is on the arc axis.",
        )
    end
    code_of_arc = Dict(zip(removed_arcs, rec.removed_arc_code))
    for (i, code) in enumerate(rec.removed_arc_code)
        pair = (rec.composite_from[i], rec.composite_to[i])
        if code == REDUCTION_CODE_DEGREE_TWO
            (iszero(pair[1]) || iszero(pair[2])) && error(
                "Factor file entry reduction/composite row $i is empty for a degree-two arc.",
            )
            _check_known_bus("composite_from", pair[1], i, known)
            _check_known_bus("composite_to", pair[2], i, known)
            any(key -> key in arc_set || key in removed_set, (pair, (pair[2], pair[1]))) ||
                error(
                    "Factor file entry reduction/composite row $i holds arc $pair, which " *
                    "is neither on the arc axis nor a removed arc.",
                )
            # One degree-two step exists, so a composite arc is never a degree-two member.
            # This also stops the reason lookup from following a loop.
            for key in (pair, (pair[2], pair[1]))
                get(code_of_arc, key, 0) == REDUCTION_CODE_DEGREE_TWO && error(
                    "Factor file entry reduction/composite row $i holds the degree-two arc $key.",
                )
            end
        else
            pair == (0, 0) || error(
                "Factor file entry reduction/composite row $i holds an arc, but code $code is not the degree-two code.",
            )
        end
    end
    survivor_arcs = collect(zip(rec.survivor_from, rec.survivor_to))
    _check_unique("reduction/survivor", survivor_arcs)
    _check_known_buses("survivor_bus", rec.survivor_bus, known)
    for arc in survivor_arcs
        arc in removed_set || error(
            "Factor file entry reduction/survivor holds arc $arc, which is not a removed arc.",
        )
    end
    boundary = collect(zip(rec.boundary_bus, rec.boundary_from, rec.boundary_to))
    _check_unique("reduction/boundary", boundary)
    _check_known_buses("boundary_bus", rec.boundary_bus, known)
    for (bus, from, to) in boundary
        (from, to) in removed_set || error(
            "Factor file entry reduction/boundary holds arc $((from, to)), which is not a removed arc.",
        )
    end
    _check_unique("reduction/merged_from", rec.merged_from)
    for bus in rec.merged_from
        bus in bus_set && error(
            "Factor file entry reduction/merged_from holds bus $bus, which is on the bus axis.",
        )
    end
    _check_known_buses("merged_from", rec.merged_from, known)
    _check_known_buses("merged_to", rec.merged_to, known)
    merged_from_set = Set(rec.merged_from)
    for (i, bus) in enumerate(rec.merged_to)
        bus in merged_from_set && error(
            "Factor file entry reduction/merged_to holds bus $bus at row $i, which is also in merged_from.",
        )
    end
    _check_unique("reduction/irreducible_buses", rec.irreducible_buses)
    _check_known_buses("irreducible_buses", rec.irreducible_buses, known)
    return
end

"""
    BranchOutageTable

Branch outages that `to_hdf5(mat, filename, sys)` stored. Read it with
`from_hdf5(BranchOutageTable, filename)`. Build a `NetworkModification` from it with
`NetworkModification(table, type, name)`.

# Fields
- `rows`: for the key `(type, name)`, the islanding flag and the row range.
- `arc_index`, `delta_b`, `delta_shift`, `opened`: the arrays that the row ranges index.
- `opens_arc`: for each arc row, `true` when the outage of that one branch removes all
  admittance of the arc.
- `member`: for each arc row, the position of the tripped member in the parallel group of
  the arc. Zero when the arc is not a parallel group.
- `ids`: the `IS.get_id` of each branch, mapped to its key `(type, name)`. Build a
  modification with `NetworkModification(table, id::Int)`.
- `n_arc`: the number of arcs in the file. Each `arc_index` is at most `n_arc`.
"""
struct BranchOutageTable
    rows::Dict{Tuple{String, String}, Tuple{Bool, UnitRange{Int}}}
    arc_index::Vector{Int}
    delta_b::Vector{Float64}
    delta_shift::Vector{Float64}
    opened::Vector{Int}
    opens_arc::Vector{Bool}
    member::Vector{Int}
    ids::Dict{Int, Tuple{String, String}}
    n_arc::Int
end

"Return the map from the key `(type, name)` to the islanding flag and the row range of `t`."
get_outage_rows(t::BranchOutageTable) = t.rows

"Return the map from `IS.get_id` of a branch to its key `(type, name)` in `t`."
get_outage_ids(t::BranchOutageTable) = t.ids

"Return the `opens_arc` column of `t`: `true` when the outage removes all admittance of the arc."
get_outage_opens_arc(t::BranchOutageTable) = t.opens_arc

"Return the `member` column of `t`: the position of the tripped member in the parallel group."
get_outage_member(t::BranchOutageTable) = t.member

function _check_row_values(
    group::String,
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    n_arc::Int,
)
    for (row, i) in enumerate(arc_index)
        1 <= i <= n_arc || error(
            "Factor file entry $group/arc_index has the value $i at row $row; " *
            "expected a value in 1:$n_arc.",
        )
    end
    for (row, b) in enumerate(delta_b)
        (isfinite(b) && !iszero(b) && abs(b) <= MAX_ARC_SUSCEPTANCE) || error(
            "Factor file entry $group/delta_b has the value $b in arc row $row. The value " *
            "must be finite, not zero, and at most $MAX_ARC_SUSCEPTANCE in magnitude.",
        )
    end
    for (row, x) in enumerate(delta_shift)
        isfinite(x) || error(
            "Factor file entry $group/delta_shift has the value $x at row $row; " *
            "the value must be finite.",
        )
    end
    for (row, x) in enumerate(opened)
        x >= 0 || error(
            "Factor file entry $group/opened has the value $x at row $row; " *
            "the value must not be negative.",
        )
    end
    return
end

function _check_row_ranges(
    group::String,
    starts::Vector{Int},
    lens::Vector{Int},
    n_rows::Int,
)
    for (k, s) in enumerate(starts)
        s >= 1 || error(
            "Factor file entry $group/start has the value $s at row $k; expected 1 or more.",
        )
        s <= n_rows + 1 || error(
            "Factor file entry $group/start has the value $s at row $k; " *
            "the group has $n_rows arc rows.",
        )
    end
    for (k, l) in enumerate(lens)
        l >= 0 || error(
            "Factor file entry $group/len has the value $l at row $k; " *
            "the value must not be negative.",
        )
        l <= n_rows || error(
            "Factor file entry $group/len has the value $l at row $k; " *
            "the group has $n_rows arc rows.",
        )
    end
    return
end

# An outage on an arc with zero susceptance divides by zero in the Woodbury factors.
function _check_outage_arcs(
    arc_index::Vector{Int},
    arc_susceptances::Vector{Float64},
)
    for (row, i) in enumerate(arc_index)
        iszero(arc_susceptances[i]) && error(
            "Factor file outage row $row is on arc $i, whose arc_susceptances entry is zero.",
        )
    end
    return
end

function _arc_modifications(
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    rows::Union{Vector{Int}, UnitRange{Int}},
)
    z = zero(YBUS_ELTYPE)
    return ArcModification[
        ArcModification(arc_index[i], delta_b[i], delta_shift[i], z, z, z, z, opened[i])
        for i in rows
    ]
end

"""
    NetworkModification(table::BranchOutageTable, type, name)
    NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})

Build the outage of one branch, or of several branches at once, from a
[`BranchOutageTable`](@ref). `type` is a `String`, for example
`"Line"`. The function groups the arc rows of all keys by arc:

  - If a row opens the whole arc, the arc gets the first such row only.
  - Otherwise the arc gets the first row of each tripped group member. Partial outages of
    different members add. Two lines of one chain in a parallel group remove that chain one
    time.

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
    by_arc = DataStructures.OrderedDict{Int, Vector{Int}}()
    islanding = false
    for key in keys
        if !haskey(table.rows, key)
            error("Branch $(key[1]) $(key[2]) has no row in the outage table.")
        end
        is_islanding, range = table.rows[key]
        islanding |= is_islanding
        for i in range
            push!(get!(by_arc, table.arc_index[i], Int[]), i)
        end
    end
    picked = Int[]
    for rows in values(by_arc)
        append!(picked, _combined_rows(table, rows))
    end
    return NetworkModification(
        join((key[2] for key in keys), "+"),
        _arc_modifications(
            table.arc_index, table.delta_b, table.delta_shift, table.opened, picked,
        ),
        ShuntModification[],
        islanding,
    )
end

# `rows` holds the rows of one arc.
function _combined_rows(table::BranchOutageTable, rows::Vector{Int})
    for i in rows
        if table.opens_arc[i]
            return [i]
        end
    end
    return unique(i -> table.member[i], rows)
end

function NetworkModification(table::BranchOutageTable, type::String, name::String)
    return NetworkModification(table, [(type, name)])
end

"""
    NetworkModification(table::BranchOutageTable, id::Int)

Build the outage of the branch with `IS.get_id(branch) == id`. The function throws an error
if the table has no row for `id`.
"""
function NetworkModification(table::BranchOutageTable, id::Int)
    haskey(table.ids, id) || error("Branch with id $id has no row in the outage table.")
    return NetworkModification(table, [table.ids[id]])
end

"""
    branch_flows(table::BranchArcTable, arc_flows::Vector{Float64}, outaged_rows::Vector{Int} = Int[]) -> Vector{Float64}

Split the arc flows into one flow for each row of `table`. A flow is in system base per
unit, positive from the first to the second bus of the frame of the row. A row in
`outaged_rows` carries no flow, and neither do the rows below an outaged parallel group.
A parallel group without a live member carries no flow. A series chain with an outaged
segment carries no flow, and its live segments carry none. A live parallel group in such a
chain still splits a circulating flow between its members when a member has a phase shift.
The function throws an error if `arc_flows` does not have one value per arc, a row of
`outaged_rows` is out of range, or a series sum is zero. A row with `b = 0` has no DC
coupling and carries no flow. The function also throws an error if a parallel or arc sum is
zero while the flow to split is not zero.
"""
function branch_flows(
    table::BranchArcTable,
    arc_flows::Vector{Float64},
    outaged_rows::Vector{Int} = Int[],
)
    n = length(table.kind)
    n_arc = table.n_arc
    length(arc_flows) == n_arc ||
        error("branch_flows: arc_flows has $(length(arc_flows)) values; expected $n_arc.")
    dead = falses(n)
    for r in outaged_rows
        1 <= r <= n ||
            error("branch_flows: outaged row $r is not in 1:$n.")
        dead[r] = true
    end
    outaged = copy(dead)
    kind = table.kind
    parent = table.parent
    sign = table.sign
    b = copy(table.b)
    alpha = copy(table.alpha)
    sb = zeros(n)
    ss = zeros(n)
    sr = zeros(n)
    sa = zeros(n)
    live_children = zeros(Int, n)
    dead_children = zeros(Int, n)
    arc_sb = zeros(n_arc)
    arc_ss = zeros(n_arc)
    for r in n:-1:1
        if dead[r]
            # An outaged row carries no flow, so no sum is checked.
        elseif kind[r] == 9
            if iszero(live_children[r])
                dead[r] = true
            elseif iszero(sb[r])
                # No member has DC coupling, so the group has none.
                b[r] = 0.0
                alpha[r] = 0.0
            else
                b[r] = sb[r]
                alpha[r] = ss[r] / sb[r]
            end
        elseif kind[r] == 10
            if !iszero(dead_children[r])
                dead[r] = true
            else
                iszero(sr[r]) &&
                    error("branch_flows: row $r (series chain) has a zero impedance sum.")
                b[r] = 1 / sr[r]
                alpha[r] = sa[r]
            end
        end
        p = parent[r]
        if p > 0 && kind[p] == 10
            if dead[r]
                dead_children[p] += 1
            else
                sr[p] += 1 / b[r]
                sa[p] += sign[r] * alpha[r]
            end
        elseif !dead[r]
            if p < 0
                arc_sb[-p] += b[r]
                arc_ss[-p] += b[r] * sign[r] * alpha[r]
            else
                live_children[p] += 1
                sb[p] += b[r]
                ss[p] += b[r] * sign[r] * alpha[r]
            end
        end
    end
    flows = zeros(n)
    inherited = falses(n)
    for r in 1:n
        p = parent[r]
        # A group that is dead only because no member is live does not stop a loop flow
        # in a live group below it.
        if p > 0 && (inherited[p] || (outaged[p] && kind[p] == 9))
            dead[r] = true
            inherited[r] = true
        end
        dead[r] && continue
        if p < 0
            flows[r] = _split_flow(
                r,
                sign[r],
                b[r],
                alpha[r],
                arc_sb[-p],
                arc_flows[-p] + arc_ss[-p],
            )
        elseif kind[p] == 9
            flows[r] = _split_flow(r, sign[r], b[r], alpha[r], sb[p], flows[p] + ss[p])
        else
            flows[r] = sign[r] * flows[p]
        end
        isfinite(flows[r]) || error(
            "branch_flows: the flow of row $r (kind $(kind[r])) is $(flows[r]); " *
            "an arc flow or a susceptance is out of range.",
        )
    end
    return flows
end

# A zero susceptance sum is BA's "no DC coupling": the rows carry no flow, and the flow to
# split must be zero.
function _split_flow(
    r::Int,
    sign::Int,
    b::Float64,
    alpha::Float64,
    b_sum::Float64,
    flow::Float64,
)
    if iszero(b_sum)
        iszero(flow) || error(
            "branch_flows: row $r has no DC coupling (zero susceptance sum) but its parent carries the flow $flow.",
        )
        return 0.0
    end
    return sign * (b / b_sum) * flow - b * alpha
end

const NO_ARC_HAS_ARC = 0
const NO_ARC_RADIAL = 1
const NO_ARC_ZERO_IMPEDANCE = 2
const NO_ARC_WARD = 3
const NO_ARC_NOT_IN_NETWORK = 4

# Reason a removed arc has no arc of its own. `_reason_of_arc` handles the degree-two code.
const _REASON_OF_REDUCTION = Dict{Int, Int}(
    REDUCTION_CODE_ZERO_IMPEDANCE => NO_ARC_ZERO_IMPEDANCE,
    REDUCTION_CODE_RADIAL => NO_ARC_RADIAL,
    REDUCTION_CODE_WARD => NO_ARC_WARD,
)

# The key of a removed arc is in the numbering of its step: the original numbers, or the
# numbers after the zero-impedance merge (that reduction runs first).
function _candidate_arcs(nr::NetworkReductionData, from::Int, to::Int)
    zf = get(nr.merged_bus_pairs, from, from)
    zt = get(nr.merged_bus_pairs, to, to)
    return ((from, to), (to, from), (zf, zt), (zt, zf))
end

# The removed arcs that match the buses, in the order of `_candidate_arcs`.
function _removed_arcs(nr::NetworkReductionData, from::Int, to::Int)
    return Tuple{Int, Int}[
        a for a in _candidate_arcs(nr, from, to) if haskey(nr.removed_arc_reduction, a)
    ]
end

# The reason for the in-service branch between the bus numbers `from` and `to`, as the system
# document has them. In-service branches between one pair of buses share their fate, so the
# pair identifies them. An out-of-service branch needs its id: see `BranchReasonTable`.
function _bus_pair_reason(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    from::Int,
    to::Int,
)
    removed = _removed_arcs(nr, from, to)
    if !isempty(removed)
        return _reason_of_arc(nr, arc_lookup, first(removed))
    end
    zf = get(nr.merged_bus_pairs, from, from)
    zt = get(nr.merged_bus_pairs, to, to)
    # A branch with impedance between two merged buses is a self-loop: no record names it.
    zf == zt && return NO_ARC_ZERO_IMPEDANCE
    if haskey(arc_lookup, (zf, zt)) || haskey(arc_lookup, (zt, zf))
        return NO_ARC_HAS_ARC
    end
    return NO_ARC_NOT_IN_NETWORK
end

function _is_query_bus(nr::NetworkReductionData, bus::Int)
    return haskey(nr.bus_reduction_map, bus) || haskey(nr.reverse_bus_search_map, bus) ||
           bus in nr.removed_buses || haskey(nr.merged_bus_pairs, bus)
end

# A degree-two member has an arc while its composite arc survives. A later reduction can
# remove the composite arc, and then that record gives the reason.
function _reason_of_arc(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    arc::Tuple{Int, Int},
)
    code = nr.removed_arc_reduction[arc]
    code == REDUCTION_CODE_DEGREE_TWO || return _REASON_OF_REDUCTION[code]
    composite = nr.removed_arc_composite[arc]
    reverse_composite = reverse(composite)
    if haskey(arc_lookup, composite) || haskey(arc_lookup, reverse_composite)
        return NO_ARC_HAS_ARC
    end
    for key in (composite, reverse_composite)
        haskey(nr.removed_arc_reduction, key) &&
            return _reason_of_arc(nr, arc_lookup, key)
    end
    error(
        "The degree-two arc $arc has the composite arc $composite, which no record explains.",
    )
end

# The record that last removed the arc: a degree-two member leads to its composite arc when a
# later reduction removed that arc.
function _terminal_arc(nr::NetworkReductionData, arc::Tuple{Int, Int})
    nr.removed_arc_reduction[arc] == REDUCTION_CODE_DEGREE_TWO || return arc
    composite = nr.removed_arc_composite[arc]
    for key in (composite, reverse(composite))
        haskey(nr.removed_arc_reduction, key) && return _terminal_arc(nr, key)
    end
    return arc
end

# The bus that kept the branch between `from` and `to` when the radial reduction absorbed it, in
# the numbering after the later reductions. A later degree-two reduction can remove that bus.
function _bus_pair_surviving_bus(nr::NetworkReductionData, from::Int, to::Int)
    removed = _removed_arcs(nr, from, to)
    isempty(removed) &&
        error("No reduction removed a branch between buses ($from, $to).")
    arc = _terminal_arc(nr, first(removed))
    nr.removed_arc_reduction[arc] == REDUCTION_CODE_RADIAL ||
        error("The radial reduction did not remove the branch between buses ($from, $to).")
    # A cascaded stub arc, where the peel removed both ends, has no stored survivor.
    return get_mapped_bus_number(nr, get(nr.removed_arc_to_surviving_bus, arc, first(arc)))
end

"""
    BranchReasonTable

For each branch of a `System`, the code that tells whether the branch has an arc in a matrix,
and the surviving bus of a branch that the radial reduction absorbed. The key is
(`id`, winding), the key of a leaf row of [`BranchArcTable`](@ref): `IS.get_id` of the
branch, and winding 0 for a branch or 1 to 3 for a circuit of a three-winding transformer.

Build the table with `BranchReasonTable(mat, sys)`, or read it with
`from_hdf5(BranchReasonTable, filename)`. Query it with [`no_arc_reason`](@ref) and
[`no_arc_surviving_bus`](@ref).
"""
struct BranchReasonTable
    rows::Dict{Tuple{Int, Int}, Tuple{Int, Int}}
end

"""
    BranchReasonTable(mat::Union{VirtualPTDF, VirtualMODF}, sys::PSY.System) -> BranchReasonTable

Give each AC branch of `sys` its code in `mat`. A branch or circuit that the Ybus build skips
(not available, or an open breaker), or that touches a bus the matrix excluded, has the code
`NO_ARC_NOT_IN_NETWORK`. The function
throws an error if `mat` was built from another system.
"""
function BranchReasonTable(mat::Union{VirtualPTDF, VirtualMODF}, sys::PSY.System)
    _validate_system_uuid(mat, sys)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    rows = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        for (w, part) in _windings(branch)
            rows[(IS.get_id(branch), w)] =
                _reason_row(nr, arc_lookup, _in_ybus(part), PSY.get_arc(part))
        end
    end
    return BranchReasonTable(rows)
end

# Each (winding, part) of a branch: winding 0 for a branch, 1 to 3 for a circuit.
_windings(branch::PSY.ACTransmission) = ((0, branch),)
_windings(trf::PSY.ThreeWindingTransformer) = enumerate(PSY.get_circuits(trf))

function _reason_row(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    in_ybus::Bool,
    arc::PSY.Arc,
)
    from, to = get_arc_tuple(arc)
    # Ybus also skips every branch at a bus that it excluded.
    if !in_ybus || !_is_query_bus(nr, from) || !_is_query_bus(nr, to)
        return (NO_ARC_NOT_IN_NETWORK, 0)
    end
    code = _bus_pair_reason(nr, arc_lookup, from, to)
    if code == NO_ARC_RADIAL
        return (code, _bus_pair_surviving_bus(nr, from, to))
    end
    return (code, 0)
end

function _branch_reason(t::BranchReasonTable, id::Int, winding::Int)
    haskey(t.rows, (id, winding)) || error(
        "The branch reason table has no branch with id $id and winding $winding. " *
        "It holds the branches of the system that built it.",
    )
    return t.rows[(id, winding)]
end

"""
    no_arc_reason(table::BranchReasonTable, id::Int, winding::Int = 0) -> Int

Tell why the branch with `IS.get_id` equal to `id` has or lacks an arc. Use `winding` 1 to 3
for a circuit of a three-winding transformer. The function returns one of these codes:

  - `NO_ARC_HAS_ARC` (0): the branch has an arc, alone or inside a parallel group or series chain.
  - `NO_ARC_RADIAL` (1): the radial reduction absorbed the branch. Call [`no_arc_surviving_bus`](@ref) for the bus that kept it.
  - `NO_ARC_ZERO_IMPEDANCE` (2): the zero-impedance reduction merged both ends into one bus.
  - `NO_ARC_WARD` (3): the Ward reduction eliminated the branch.
  - `NO_ARC_NOT_IN_NETWORK` (4): the branch is out of service, or it touches a bus that the matrix excluded.

The function throws an error if the table has no row for (`id`, `winding`).
"""
no_arc_reason(t::BranchReasonTable, id::Int, winding::Int = 0) =
    first(_branch_reason(t, id, winding))

"""
    no_arc_surviving_bus(table::BranchReasonTable, id::Int, winding::Int = 0) -> Int

The bus that kept the branch when the radial reduction absorbed it, in the numbering after
the later reductions. A later degree-two reduction can remove that bus. The function throws
an error if the table has no row for (`id`, `winding`), or if the radial reduction did not
absorb the branch.
"""
function no_arc_surviving_bus(t::BranchReasonTable, id::Int, winding::Int = 0)
    code, bus = _branch_reason(t, id, winding)
    code == NO_ARC_RADIAL || error(
        "The radial reduction did not remove the branch with id $id and winding $winding; " *
        "its code is $code.",
    )
    return bus
end

function _check_branch_reason(
    id::Vector{Int},
    winding::Vector{Int},
    reason::Vector{Int},
    surviving_bus::Vector{Int},
)
    _check_same_length(
        "branch_reason",
        "id" => id, "winding" => winding, "reason" => reason,
        "surviving_bus" => surviving_bus,
    )
    _check_unique("branch_reason/(id, winding)", collect(zip(id, winding)))
    for r in eachindex(id)
        fail(what) = error("Factor file table branch_reason row $r: $what.")
        id[r] > 0 || fail("id $(id[r]) is not positive")
        0 <= winding[r] <= 3 || fail("winding $(winding[r]) is not in 0:3")
        NO_ARC_HAS_ARC <= reason[r] <= NO_ARC_NOT_IN_NETWORK ||
            fail("reason $(reason[r]) is not in 0:4")
        if reason[r] == NO_ARC_RADIAL
            surviving_bus[r] > 0 ||
                fail("surviving_bus is not positive for a radial branch")
        else
            iszero(surviving_bus[r]) ||
                fail(
                    "surviving_bus is $(surviving_bus[r]); expected 0 for reason $(reason[r])",
                )
        end
    end
    return
end
