"""
    NetworkModification

Lightweight description of topology changes to a power network.
Wraps a vector of [`ArcModification`](@ref) plus a label.
No dependency on `PSY.System` after construction.

# Fields
- `label::String`: Human-readable identifier for the modification.
- `modifications::Vector{ArcModification}`: One entry per affected arc.
"""
struct NetworkModification
    label::String
    modifications::Vector{ArcModification}
    function NetworkModification(label::String, mods::Vector{ArcModification})
        return new(label, _merge_modifications(mods))
    end
end

"""
$(TYPEDSIGNATURES)

Construct a full arc outage `NetworkModification` by bus-pair tuple.
Looks up arc susceptance from the matrix and sets `Δb = -b_arc`.
"""
function NetworkModification(mat::PowerNetworkMatrix, arc::Tuple{Int, Int})
    arc_lookup = get_arc_lookup(mat)
    arc_idx = arc_lookup[arc]
    b = _get_arc_susceptances(mat)[arc_idx]
    return NetworkModification(
        "outage_$(arc[1])_$(arc[2])",
        [ArcModification(arc_idx, -b)],
    )
end

"""
$(TYPEDSIGNATURES)

Construct a `NetworkModification` from a branch component using network
reduction reverse maps to classify the branch as direct, parallel, or series.
"""
function NetworkModification(mat::PowerNetworkMatrix, branch::PSY.ACTransmission)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    arc_sus = _get_arc_susceptances(mat)
    mods = _classify_branch_modification(nr, arc_lookup, arc_sus, branch)
    return NetworkModification(
        PSY.get_name(branch),
        mods,
    )
end

"""
$(TYPEDSIGNATURES)

Construct a `NetworkModification` from an existing [`ContingencySpec`](@ref).
"""
function NetworkModification(ctg::ContingencySpec)
    return NetworkModification(ctg.name, ctg.modifications)
end

"""
    _classify_branch_modification(nr, arc_lookup, arc_susceptances, branch) -> Vector{ArcModification}

Classify a branch component into the appropriate arc modification using
the network reduction reverse maps.
"""
function _classify_branch_modification(
    ::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    branch::PSY.PhaseShiftingTransformer,
)
    error(
        "Modifications on PhaseShiftingTransformer are not supported. " *
        "Component: $(PSY.get_name(branch)).",
    )
end

function _classify_branch_modification(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    branch::PSY.ACTransmission,
)::Vector{ArcModification}
    if haskey(nr.reverse_direct_branch_map, branch)
        arc_tuple = nr.reverse_direct_branch_map[branch]
        arc_idx = arc_lookup[arc_tuple]
        b_arc = arc_susceptances[arc_idx]
        return [ArcModification(arc_idx, -b_arc)]

    elseif haskey(nr.reverse_parallel_branch_map, branch)
        arc_tuple = nr.reverse_parallel_branch_map[branch]
        arc_idx = arc_lookup[arc_tuple]
        b_circuit = PSY.get_series_susceptance(branch)
        return [ArcModification(arc_idx, -b_circuit)]

    elseif haskey(nr.reverse_series_branch_map, branch)
        arc_tuple = nr.reverse_series_branch_map[branch]
        arc_idx = arc_lookup[arc_tuple]
        series_chain = nr.series_branch_map[arc_tuple]
        delta_b = _compute_series_outage_delta_b(series_chain, branch)
        return [ArcModification(arc_idx, delta_b)]

    else
        @warn "Branch $(PSY.get_name(branch)) not found in any reduction map. " *
              "The component may have been eliminated by a radial reduction."
        return ArcModification[]
    end
end
