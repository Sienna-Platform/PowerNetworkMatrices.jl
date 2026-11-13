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

# `ctx` is a `_TwoPortNode` or a `_ChainNode`. It selects the susceptance that `BA_Matrix` uses.
function _push_node!(
    rows::BranchArcTable,
    leaf::PSY.ACTransmission,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
    ctx::Union{_TwoPortNode, _ChainNode},
)
    _push_row!(
        rows, _branch_map_kind(leaf), _branch_map_id(leaf), parent, sign,
        _ba_node_susceptance(ctx, leaf, nr), _segment_phase_shift(leaf, nr),
    )
    return
end

function _push_node!(
    rows::BranchArcTable,
    bp::AbstractBranchesParallel,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
    ctx::Union{_TwoPortNode, _ChainNode},
)
    row = _push_row!(rows, 9, 0, parent, sign, 0.0, 0.0)
    frame = get_arc_tuple(bp, nr)
    for member in bp
        _push_node!(
            rows, member, row, _frame_sign(frame, get_arc_tuple(member, nr)), nr, ctx,
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
    ::Union{_TwoPortNode, _ChainNode},
)
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
        _push_node!(rows, segment, row, seg_sign, nr, _ChainNode())
    end
    return
end

function _push_ward!(
    rows::BranchArcTable,
    ward::PSY.GenericArcImpedance,
    parent::Int,
    nr::NetworkReductionData,
)
    _push_row!(rows, 11, 0, parent, 1, _ba_node_susceptance(_TwoPortNode(), ward, nr), 0.0)
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
                _TwoPortNode(),
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

function _branch_reduction_rows(nr::NetworkReductionData)
    rows = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for branch_map in (nr.direct_branch_map, nr.parallel_branch_map, nr.series_branch_map)
        for node in values(branch_map)
            _record_has_arc!(rows, node)
        end
    end
    for (key, (code, survivor)) in nr.removed_branch_reduction
        haskey(rows, key) && continue
        if code == BRANCH_REDUCTION_RADIAL
            survivor = get_mapped_bus_number(nr, survivor)
        end
        rows[key] = (code, survivor)
    end
    return rows
end

function _record_has_arc!(rows, node::PSY.ACTransmission)
    for leaf in leaf_components(node)
        rows[_branch_reduction_key(leaf)] = (BRANCH_REDUCTION_HAS_ARC, 0)
    end
    return
end

"""
    BranchReductionTable(mat::Union{VirtualPTDF, VirtualMODF}, sys::PSY.System) -> BranchReductionTable

Give each AC branch of `sys` its code in `mat`. A branch or circuit that the Ybus build skips
(not available, or an open breaker), or that touches a bus the matrix excluded, has the code
`BRANCH_REDUCTION_NOT_IN_NETWORK`. The function
throws an error if `mat` was built from another system, or if `mat` was restored from a
factor file. A restored matrix has no branch record: read the table with
`from_hdf5(BranchReductionTable, filename)`.
"""
function BranchReductionTable(mat::Union{VirtualPTDF, VirtualMODF}, sys::PSY.System)
    _validate_system_uuid(mat, sys)
    holds_components(get_core(mat)) || error(
        "Cannot build a BranchReductionTable from a matrix that was restored from a factor file: it has no branch record. Read the table with from_hdf5(BranchReductionTable, filename).",
    )
    rec = _branch_reduction_rows(get_network_reduction_data(mat))
    rows = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        for (w, _) in _windings(branch)
            key = (IS.get_id(branch), w)
            rows[key] = get(rec, key, (BRANCH_REDUCTION_NOT_IN_NETWORK, 0))
        end
    end
    return BranchReductionTable(rows)
end

# Each (winding, part) of a branch: winding 0 for a branch, 1 to 3 for a circuit.
_windings(branch::PSY.ACTransmission) = ((0, branch),)
_windings(trf::PSY.ThreeWindingTransformer) = enumerate(PSY.get_circuits(trf))
