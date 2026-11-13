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
