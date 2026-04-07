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

    # Single query for all associated components (avoids repeated PSY lookups)
    all_components = collect(PSY.get_associated_components(sys, outage))

    if isempty(all_components)
        error("No valid arc or shunt modifications found for outage.")
    end

    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    arc_sus = _get_arc_susceptances(mat)
    bus_lookup = get_bus_lookup(mat)

    # Pass 1: classify components. Series branches on the same arc must be
    # grouped so their combined Δb is computed correctly.
    direct_mods = ArcModification[]
    parallel_mods = ArcModification[]
    series_components_by_arc = Dict{Int, Vector{PSY.ACTransmission}}()
    series_arc_tuples = Dict{Int, Tuple{Int, Int}}()
    component_names = String[]
    shunt_mods = ShuntModification[]

    for component in all_components
        _classify_outage_component!(
            nr, arc_lookup, arc_sus, bus_lookup, component,
            direct_mods, parallel_mods,
            series_components_by_arc, series_arc_tuples,
            shunt_mods, component_names,
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

    if isempty(mods) && isempty(shunt_mods)
        error("No valid arc or shunt modifications found for outage.")
    end

    outage_uuid = IS.get_uuid(outage)
    ctg_name = isempty(component_names) ? string(outage_uuid) :
               join(component_names, "+")
    return NetworkModification(ctg_name, mods, shunt_mods, false)
end

"""
    _classify_outage_component!(nr, arc_lookup, arc_sus, bus_lookup, component, ...) -> nothing

Classify a single outage component via multiple dispatch. ACTransmission branches are
classified into direct/parallel/series arc modifications. Shunt components produce
diagonal admittance changes. Unsupported component types are silently ignored.
"""
function _classify_outage_component!(
    ::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    ::Dict{Int, Int},
    component::PSY.PhaseShiftingTransformer,
    ::Vector{ArcModification},
    ::Vector{ArcModification},
    ::Dict{Int, Vector{PSY.ACTransmission}},
    ::Dict{Int, Tuple{Int, Int}},
    ::Vector{ShuntModification},
    ::Vector{String},
)
    _assert_not_phase_shifting(component)
end

function _classify_outage_component!(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    ::Dict{Int, Int},
    component::PSY.ACTransmission,
    direct_mods::Vector{ArcModification},
    parallel_mods::Vector{ArcModification},
    series_components_by_arc::Dict{Int, Vector{PSY.ACTransmission}},
    series_arc_tuples::Dict{Int, Tuple{Int, Int}},
    ::Vector{ShuntModification},
    component_names::Vector{String},
)
    tag, arc_tuple = _resolve_branch_arc(nr, component)

    if tag === :direct || tag === :transformer3w
        arc_idx = arc_lookup[arc_tuple]
        b_arc = arc_susceptances[arc_idx]
        push!(direct_mods, ArcModification(arc_idx, -b_arc))
    elseif tag === :parallel
        arc_idx = arc_lookup[arc_tuple]
        b_circuit = PSY.get_series_susceptance(component)
        push!(parallel_mods, ArcModification(arc_idx, -b_circuit))
    elseif tag === :series
        arc_idx = arc_lookup[arc_tuple]
        if !haskey(series_components_by_arc, arc_idx)
            series_components_by_arc[arc_idx] = PSY.ACTransmission[]
            series_arc_tuples[arc_idx] = arc_tuple
        end
        push!(series_components_by_arc[arc_idx], component)
    else
        @warn "Branch $(PSY.get_name(component)) not found in any reduction map. " *
              "The component may have been eliminated by a radial reduction."
        return
    end
    push!(component_names, PSY.get_name(component))
    return
end

function _classify_outage_component!(
    nr::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    bus_lookup::Dict{Int, Int},
    component::Union{PSY.FixedAdmittance, PSY.SwitchedAdmittance},
    ::Vector{ArcModification},
    ::Vector{ArcModification},
    ::Dict{Int, Vector{PSY.ACTransmission}},
    ::Dict{Int, Tuple{Int, Int}},
    shunt_mods::Vector{ShuntModification},
    component_names::Vector{String},
)
    bus_ix = get_bus_index(component, bus_lookup, nr)
    Y = PSY.get_Y(component)
    push!(shunt_mods, ShuntModification(bus_ix, YBUS_ELTYPE(-Y)))
    push!(component_names, PSY.get_name(component))
    return
end

function _classify_outage_component!(
    nr::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    bus_lookup::Dict{Int, Int},
    component::PSY.StandardLoad,
    ::Vector{ArcModification},
    ::Vector{ArcModification},
    ::Dict{Int, Vector{PSY.ACTransmission}},
    ::Dict{Int, Tuple{Int, Int}},
    shunt_mods::Vector{ShuntModification},
    component_names::Vector{String},
)
    bus_ix = get_bus_index(component, bus_lookup, nr)
    Y =
        PSY.get_impedance_active_power(component) -
        im * PSY.get_impedance_reactive_power(component)
    push!(shunt_mods, ShuntModification(bus_ix, YBUS_ELTYPE(-Y)))
    push!(component_names, PSY.get_name(component))
    return
end

function _classify_outage_component!(
    ::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    ::Dict{Int, Int},
    component::PSY.Component,
    ::Vector{ArcModification},
    ::Vector{ArcModification},
    ::Dict{Int, Vector{PSY.ACTransmission}},
    ::Dict{Int, Tuple{Int, Int}},
    ::Vector{ShuntModification},
    ::Vector{String},
)
    @info "Component $(PSY.get_name(component)) ($(typeof(component))) " *
          "is not supported for outage classification. Skipping."
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
    _assert_not_phase_shifting(branch)
end

function _classify_branch_modification(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    branch::PSY.ACTransmission,
)::Vector{ArcModification}
    tag, arc_tuple = _resolve_branch_arc(nr, branch)

    if tag === :direct || tag === :transformer3w
        arc_idx = arc_lookup[arc_tuple]
        b_arc = arc_susceptances[arc_idx]
        return [ArcModification(arc_idx, -b_arc)]
    elseif tag === :parallel
        arc_idx = arc_lookup[arc_tuple]
        b_circuit = PSY.get_series_susceptance(branch)
        return [ArcModification(arc_idx, -b_circuit)]
    elseif tag === :series
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

# --- Bridge from NetworkModification to Ybus domain ---

"""
    compute_ybus_delta(ybus::Ybus, mod::NetworkModification) -> SparseMatrixCSC{YBUS_ELTYPE, Int}

Compute the sparse ΔYbus matrix from a canonical `NetworkModification`.
Combines arc modifications (branch outages producing Pi-model deltas) and
shunt modifications (diagonal admittance changes) into a single sparse delta.

This is the bridge between the DC sensitivity path (`NetworkModification`) and the
AC admittance path (`Ybus`). The `NetworkModification` is the canonical representation;
this function converts it to the Ybus domain.
"""
function compute_ybus_delta(
    ybus::Ybus,
    mod::NetworkModification,
)::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int}
    bus_lookup = get_bus_lookup(ybus)
    nr = get_network_reduction_data(ybus)
    n = length(bus_lookup)
    arc_ax = get_arc_axis(nr)

    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{YBUS_ELTYPE}()
    expected = 4 * length(mod.arc_modifications) + length(mod.shunt_modifications)
    sizehint!(I, expected)
    sizehint!(J, expected)
    sizehint!(V, expected)

    # Arc modifications -> full Pi-model delta Y entries
    for arc_mod in mod.arc_modifications
        arc_tuple = arc_ax[arc_mod.arc_index]
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]

        if haskey(nr.direct_branch_map, arc_tuple)
            br = nr.direct_branch_map[arc_tuple]
            b_arc = get_series_susceptance(br)
            if isapprox(arc_mod.delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
                # Full outage: negate all Pi-model entries
                _accumulate_branch_outage!(I, J, V, br, fb_ix, tb_ix)
            else
                # Partial: scale Pi-model entries by delta_b / b_arc
                Y11, Y12, Y21, Y22 = ybus_branch_entries(br)
                scale = arc_mod.delta_b / b_arc
                _accumulate_arc_delta!(
                    I, J, V, fb_ix, tb_ix,
                    YBUS_ELTYPE(scale * Y11), YBUS_ELTYPE(scale * Y12),
                    YBUS_ELTYPE(scale * Y21), YBUS_ELTYPE(scale * Y22),
                )
            end
        elseif haskey(nr.parallel_branch_map, arc_tuple)
            bp = nr.parallel_branch_map[arc_tuple]
            b_arc = get_series_susceptance(bp)
            if isapprox(arc_mod.delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
                _accumulate_branch_outage!(I, J, V, bp, fb_ix, tb_ix)
            else
                # Partial outage on parallel group: find and negate matching circuit(s)
                _accumulate_parallel_partial_outage!(
                    I, J, V, bp, fb_ix, tb_ix, arc_mod.delta_b,
                )
            end
        elseif haskey(nr.series_branch_map, arc_tuple)
            series_chain = nr.series_branch_map[arc_tuple]
            b_arc = get_series_susceptance(series_chain)
            if isapprox(arc_mod.delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
                _accumulate_branch_outage!(I, J, V, series_chain, fb_ix, tb_ix)
            else
                error(
                    "compute_ybus_delta does not support partial modifications on " *
                    "series-reduced arcs. Arc $(arc_tuple), Δb=$(arc_mod.delta_b). " *
                    "Per-component Pi-model data is required for partial Ybus deltas.",
                )
            end
        elseif haskey(nr.transformer3W_map, arc_tuple)
            tr = nr.transformer3W_map[arc_tuple]
            _accumulate_branch_outage!(I, J, V, tr, fb_ix, tb_ix)
        else
            error(
                "ArcModification for arc $(arc_tuple) (index=$(arc_mod.arc_index)) " *
                "could not be resolved to any Ybus-relevant component. " *
                "The NetworkModification and Ybus may have incompatible reduction contexts.",
            )
        end
    end

    # Shunt modifications -> diagonal delta Y entries
    for smod in mod.shunt_modifications
        push!(I, smod.bus_index)
        push!(J, smod.bus_index)
        push!(V, smod.delta_y)
    end

    return SparseArrays.sparse(I, J, V, n, n)
end

"""
    apply_ybus_modification(ybus::Ybus, mod::NetworkModification) -> SparseMatrixCSC

Apply a canonical NetworkModification to a Ybus, returning the modified sparse matrix.
Convenience wrapper around `compute_ybus_delta`.
"""
function apply_ybus_modification(
    ybus::Ybus,
    mod::NetworkModification,
)
    delta = compute_ybus_delta(ybus, mod)
    return ybus.data + delta
end
