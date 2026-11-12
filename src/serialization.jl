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
            # This also stops the reduction lookup from following a loop.
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

# The code of a removed arc that has no arc of its own. `_reduction_of_arc` handles the degree-two code.
const _BRANCH_REDUCTION_OF_CODE = Dict{Int, Int}(
    REDUCTION_CODE_ZERO_IMPEDANCE => BRANCH_REDUCTION_ZERO_IMPEDANCE,
    REDUCTION_CODE_RADIAL => BRANCH_REDUCTION_RADIAL,
    REDUCTION_CODE_WARD => BRANCH_REDUCTION_WARD,
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

# The code for the in-service branch between the bus numbers `from` and `to`, as the system
# document has them. In-service branches between one pair of buses share their fate, so the
# pair identifies them. An out-of-service branch needs its id: see `BranchReductionTable`.
function _bus_pair_reduction(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    from::Int,
    to::Int,
)
    removed = _removed_arcs(nr, from, to)
    if !isempty(removed)
        return _reduction_of_arc(nr, arc_lookup, first(removed))
    end
    zf = get(nr.merged_bus_pairs, from, from)
    zt = get(nr.merged_bus_pairs, to, to)
    # A branch with impedance between two merged buses is a self-loop: no record names it.
    zf == zt && return BRANCH_REDUCTION_ZERO_IMPEDANCE
    if haskey(arc_lookup, (zf, zt)) || haskey(arc_lookup, (zt, zf))
        return BRANCH_REDUCTION_HAS_ARC
    end
    return BRANCH_REDUCTION_NOT_IN_NETWORK
end

function _is_query_bus(nr::NetworkReductionData, bus::Int)
    return haskey(nr.bus_reduction_map, bus) || haskey(nr.reverse_bus_search_map, bus) ||
           bus in nr.removed_buses || haskey(nr.merged_bus_pairs, bus)
end

# A degree-two member has an arc while its composite arc survives. A later reduction can
# remove the composite arc, and then that record gives the code.
function _reduction_of_arc(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    arc::Tuple{Int, Int},
)
    code = nr.removed_arc_reduction[arc]
    code == REDUCTION_CODE_DEGREE_TWO || return _BRANCH_REDUCTION_OF_CODE[code]
    composite = nr.removed_arc_composite[arc]
    reverse_composite = reverse(composite)
    if haskey(arc_lookup, composite) || haskey(arc_lookup, reverse_composite)
        return BRANCH_REDUCTION_HAS_ARC
    end
    for key in (composite, reverse_composite)
        haskey(nr.removed_arc_reduction, key) &&
            return _reduction_of_arc(nr, arc_lookup, key)
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
    BranchReductionTable(mat::Union{VirtualPTDF, VirtualMODF}, sys::PSY.System) -> BranchReductionTable

Give each AC branch of `sys` its code in `mat`. A branch or circuit that the Ybus build skips
(not available, or an open breaker), or that touches a bus the matrix excluded, has the code
`BRANCH_REDUCTION_NOT_IN_NETWORK`. The function
throws an error if `mat` was built from another system.
"""
function BranchReductionTable(mat::Union{VirtualPTDF, VirtualMODF}, sys::PSY.System)
    _validate_system_uuid(mat, sys)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    rows = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        for (w, part) in _windings(branch)
            rows[(IS.get_id(branch), w)] =
                _reduction_row(nr, arc_lookup, _in_ybus(part), PSY.get_arc(part))
        end
    end
    return BranchReductionTable(rows)
end

# Each (winding, part) of a branch: winding 0 for a branch, 1 to 3 for a circuit.
_windings(branch::PSY.ACTransmission) = ((0, branch),)
_windings(trf::PSY.ThreeWindingTransformer) = enumerate(PSY.get_circuits(trf))

function _reduction_row(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    in_ybus::Bool,
    arc::PSY.Arc,
)
    from, to = get_arc_tuple(arc)
    # Ybus also skips every branch at a bus that it excluded.
    if !in_ybus || !_is_query_bus(nr, from) || !_is_query_bus(nr, to)
        return (BRANCH_REDUCTION_NOT_IN_NETWORK, 0)
    end
    code = _bus_pair_reduction(nr, arc_lookup, from, to)
    if code == BRANCH_REDUCTION_RADIAL
        return (code, _bus_pair_surviving_bus(nr, from, to))
    end
    return (code, 0)
end
