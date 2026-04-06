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
    return ctg.modification
end

"""
$(TYPEDSIGNATURES)

Construct a `NetworkModification` from a `PSY.Outage` supplemental attribute.
Resolves the outage's associated `ACTransmission` components through the system,
classifies each by the matrix's network reduction maps, and builds the
modification. Handles multi-component outages with series-chain grouping.
"""
function NetworkModification(
    mat::PowerNetworkMatrix,
    sys::PSY.System,
    outage::PSY.Outage,
)
    _validate_system_uuid(mat, sys)

    associated_components = collect(
        PSY.get_associated_components(sys, outage;
            component_type = PSY.ACTransmission),
    )

    if isempty(associated_components)
        error("Outage has no associated ACTransmission components.")
    end

    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    arc_sus = _get_arc_susceptances(mat)

    # Pass 1: classify components. Series branches on the same arc must be
    # grouped so their combined Δb is computed correctly.
    direct_mods = ArcModification[]
    parallel_mods = ArcModification[]
    series_components_by_arc = Dict{Int, Vector{PSY.ACTransmission}}()
    series_arc_tuples = Dict{Int, Tuple{Int, Int}}()
    component_names = String[]

    for component in associated_components
        push!(component_names, PSY.get_name(component))
        _classify_outage_component!(
            nr, arc_lookup, arc_sus, component,
            direct_mods, parallel_mods,
            series_components_by_arc, series_arc_tuples,
        )
    end

    # Pass 2: compute series Δb with all tripped components grouped
    series_mods = ArcModification[]
    for (arc_idx, tripped) in series_components_by_arc
        arc_tuple = series_arc_tuples[arc_idx]
        series_chain = nr.series_branch_map[arc_tuple]
        delta_b = _compute_series_outage_delta_b(series_chain, tripped)
        push!(series_mods, ArcModification(arc_idx, delta_b))
    end

    mods = vcat(direct_mods, parallel_mods, series_mods)

    if isempty(mods)
        error("No valid arc modifications found for outage.")
    end

    outage_uuid = IS.get_uuid(outage)
    ctg_name = isempty(component_names) ? string(outage_uuid) :
               join(component_names, "+")
    return NetworkModification(ctg_name, mods)
end

"""
    _classify_outage_component!(nr, arc_lookup, arc_sus, component, ...) -> nothing

Classify a single outage component into direct, parallel, or series accumulators.
Series components on the same arc are grouped for combined Δb computation.
"""
function _classify_outage_component!(
    ::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    component::PSY.PhaseShiftingTransformer,
    ::Vector{ArcModification},
    ::Vector{ArcModification},
    ::Dict{Int, Vector{PSY.ACTransmission}},
    ::Dict{Int, Tuple{Int, Int}},
)
    error(
        "Contingencies on PhaseShiftingTransformer are not supported. " *
        "Component: $(PSY.get_name(component)).",
    )
end

function _classify_outage_component!(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    component::PSY.ACTransmission,
    direct_mods::Vector{ArcModification},
    parallel_mods::Vector{ArcModification},
    series_components_by_arc::Dict{Int, Vector{PSY.ACTransmission}},
    series_arc_tuples::Dict{Int, Tuple{Int, Int}},
)
    if haskey(nr.reverse_direct_branch_map, component)
        arc_tuple = nr.reverse_direct_branch_map[component]
        arc_idx = arc_lookup[arc_tuple]
        b_arc = arc_susceptances[arc_idx]
        push!(direct_mods, ArcModification(arc_idx, -b_arc))

    elseif haskey(nr.reverse_parallel_branch_map, component)
        arc_tuple = nr.reverse_parallel_branch_map[component]
        arc_idx = arc_lookup[arc_tuple]
        b_circuit = PSY.get_series_susceptance(component)
        push!(parallel_mods, ArcModification(arc_idx, -b_circuit))

    elseif haskey(nr.reverse_series_branch_map, component)
        arc_tuple = nr.reverse_series_branch_map[component]
        arc_idx = arc_lookup[arc_tuple]
        if !haskey(series_components_by_arc, arc_idx)
            series_components_by_arc[arc_idx] = PSY.ACTransmission[]
            series_arc_tuples[arc_idx] = arc_tuple
        end
        push!(series_components_by_arc[arc_idx], component)

    else
        @warn "Branch $(PSY.get_name(component)) not found in any reduction map. " *
              "The component may have been eliminated by a radial reduction."
    end
    return
end

"""
    _classify_branch_modification(nr, arc_lookup, arc_susceptances, branch) -> Vector{ArcModification}

Classify a single branch component into the appropriate arc modification using
the network reduction reverse maps. For single-branch modifications only;
use `_classify_outage_component!` for multi-component outages with series grouping.
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
