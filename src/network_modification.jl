# `delta_b` removes the arc's entire series susceptance (a full outage).
_is_full_outage(delta_b::Float64, b_arc::Float64) =
    isapprox(delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)

# Negated Pi-model entries: the delta that cancels the arc's contribution (full outage).
function _negated_pi_model(entries::NTuple{4, <:Complex})::NTuple{4, YBUS_ELTYPE}
    return (
        YBUS_ELTYPE(-entries[1]),
        YBUS_ELTYPE(-entries[2]),
        YBUS_ELTYPE(-entries[3]),
        YBUS_ELTYPE(-entries[4]),
    )
end

# Scaled Pi-model entries: the delta for a partial susceptance change on a direct arc.
function _scaled_pi_model(
    entries::NTuple{4, <:Complex},
    scale::Float64,
)::NTuple{4, YBUS_ELTYPE}
    return (
        YBUS_ELTYPE(scale * entries[1]),
        YBUS_ELTYPE(scale * entries[2]),
        YBUS_ELTYPE(scale * entries[3]),
        YBUS_ELTYPE(scale * entries[4]),
    )
end

"""
    _member_outage_ybus_delta(bp, nr, component) -> NTuple{4, YBUS_ELTYPE}

π-model Ybus delta for tripping `component`, a specific member of the parallel group `bp`,
resolved by object identity. Anti-parallel members are swapped into the group's key frame,
mirroring `ybus_branch_entries(bp, nr)`.
"""
function _member_outage_ybus_delta(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    component::PSY.ACTransmission,
)::NTuple{4, YBUS_ELTYPE}
    # `===` relies on `ThreeWindingTransformerCircuit` being an immutable wrapper over the
    # parent's stored circuit, so reconstructed wrappers stay egal; if that changes,
    # membership must key on `{parent, winding_number}` instead.
    if !any(br === component for br in bp.branches)
        error(
            "Component $(get_name(component)) is not a member of the parallel group " *
            "$(get_name(bp)); cannot compute its outage delta.",
        )
    end
    entries = ybus_branch_entries(component)
    if get_arc_tuple(component, nr) != get_arc_tuple(bp, nr)
        entries = (entries[4], entries[3], entries[2], entries[1])
    end
    return _negated_pi_model(entries)
end

# Direct arc: full outage negates the Pi-model; otherwise scale it by `delta_b / b_arc`.
function _direct_arc_ybus_delta(
    br::PSY.ACTransmission,
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    b_arc = get_series_susceptance(br, PSY.SU)
    entries = ybus_branch_entries(br)
    if _is_full_outage(delta_b, b_arc)
        return _negated_pi_model(entries)
    end
    return _scaled_pi_model(entries, delta_b / b_arc)
end

# A three-winding transformer winding is a one-to-one arc, but its stored arc susceptance
# (the tap-divided star-leg convention) need not equal the imaginary Pi-model term that
# `ybus_branch_entries` builds, so the generic susceptance-matched full/partial test above is
# unreliable here. Only a full outage of the winding is meaningful, so cancel its whole
# Pi-model directly.
function _direct_arc_ybus_delta(
    tr::ThreeWindingTransformerCircuit,
    ::Float64,
)::NTuple{4, YBUS_ELTYPE}
    return _negated_pi_model(ybus_branch_entries(tr))
end

# Parallel group: full outage negates the equivalent; a partial outage needs the tripped
# member's identity (see `_member_outage_ybus_delta`) — value-matching susceptances picks
# the wrong member when two members share a susceptance.
function _parallel_arc_ybus_delta(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    if _is_full_outage(delta_b, get_series_susceptance(bp, PSY.SU))
        return _negated_pi_model(ybus_branch_entries(bp, nr))
    end
    error(
        "Partial outage on parallel group $(get_name(bp)) requires the tripped " *
        "component's identity; construct the modification from the branch component " *
        "instead of the arc tuple. Δb=$(delta_b), group b=" *
        "$(get_series_susceptance(bp, PSY.SU)).",
    )
end

# Series-reduced arc: only full outage of the equivalent chain is supported.
function _series_arc_ybus_delta(
    series_chain::BranchesSeries,
    nr::NetworkReductionData,
    arc_tuple::Tuple{Int, Int},
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    if !_is_full_outage(delta_b, get_series_susceptance(series_chain, PSY.SU))
        error(
            "Partial Ybus delta is not supported on series-reduced arcs. " *
            "Arc $(arc_tuple), Δb=$(delta_b).",
        )
    end
    return _negated_pi_model(ybus_branch_entries(series_chain, nr))
end

"""
    _compute_arc_ybus_delta(nr, arc_tuple, delta_b) -> NTuple{4, YBUS_ELTYPE}

Compute the Pi-model Ybus delta `(ΔY11, ΔY12, ΔY21, ΔY22)` for an arc modification by
dispatching to the per-map handler that owns `arc_tuple`.
"""
function _compute_arc_ybus_delta(
    nr::NetworkReductionData,
    arc_tuple::Tuple{Int, Int},
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    if haskey(nr.direct_branch_map, arc_tuple)
        return _direct_arc_ybus_delta(nr.direct_branch_map[arc_tuple], delta_b)
    elseif haskey(nr.parallel_branch_map, arc_tuple)
        return _parallel_arc_ybus_delta(nr.parallel_branch_map[arc_tuple], nr, delta_b)
    elseif haskey(nr.series_branch_map, arc_tuple)
        return _series_arc_ybus_delta(
            nr.series_branch_map[arc_tuple],
            nr,
            arc_tuple,
            delta_b,
        )
    end
    error(
        "Arc $(arc_tuple) not found in any network reduction map. " *
        "Cannot compute Ybus Pi-model delta.",
    )
end

"""
    _compute_arc_ybus_delta(nr, arc_tuple, delta_b, component) -> NTuple{4, YBUS_ELTYPE}

Component-aware variant: on a parallel-map arc the tripped member is resolved by object
identity instead of susceptance value. Direct and series arcs delegate to the arc-tuple
handlers (the direct-map entry *is* the component; series arcs carry aggregate deltas).
"""
function _compute_arc_ybus_delta(
    nr::NetworkReductionData,
    arc_tuple::Tuple{Int, Int},
    delta_b::Float64,
    component::PSY.ACTransmission,
)::NTuple{4, YBUS_ELTYPE}
    if haskey(nr.parallel_branch_map, arc_tuple)
        return _member_outage_ybus_delta(
            nr.parallel_branch_map[arc_tuple],
            nr,
            component,
        )
    end
    return _compute_arc_ybus_delta(nr, arc_tuple, delta_b)
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
    nr = get_network_reduction_data(mat)
    dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc, -b)
    return NetworkModification(
        "outage_$(arc[1])_$(arc[2])",
        [ArcModification(arc_idx, -b, dy11, dy12, dy21, dy22)],
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
    return NetworkModification(get_name(branch), mods)
end

"""
$(TYPEDSIGNATURES)

Construct a `NetworkModification` from a `ThreeWindingTransformer` component.
Automatically decomposes the transformer into its three winding arcs and classifies
each one. For a partial outage (single winding trip), use a
`ThreeWindingTransformerCircuit` instead.
"""
function NetworkModification(mat::PowerNetworkMatrix, branch::PSY.ThreeWindingTransformer)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    arc_sus = _get_arc_susceptances(mat)
    mods = _classify_branch_modification(nr, arc_lookup, arc_sus, branch)
    return NetworkModification(
        PSY.get_name(branch),
        mods,
        ShuntModification[],
        _3wt_real_bus_islanding(mat, mods),
    )
end

"""
    _3wt_real_bus_islanding(mat, mods) -> Bool

True if outaging the 3WT winding arcs `mods` disconnects a real (non-star) bus
from its reference. A full 3WT outage always isolates the fictitious star bus,
which is benign (it carries no injection); this flags only the case where a
load/gen-bearing terminal also disconnects. The star bus is kept in the
union-find so it connects its terminals in the baseline, but is excluded from
the component count — a higher post-outage count means a real bus was islanded.
"""
function _3wt_real_bus_islanding(mat::PowerNetworkMatrix, mods::Vector{ArcModification})
    isempty(mods) && return false
    arc_ax = get_arc_axis(mat)
    bus_lookup = get_bus_lookup(mat)
    nbus = length(get_bus_axis(mat))
    star_idx = bus_lookup[arc_ax[mods[1].arc_index][2]]
    removed = Set(m.arc_index for m in mods)

    uf_before = collect(1:nbus)
    uf_after = collect(1:nbus)
    for (e, arc) in enumerate(arc_ax)
        f = bus_lookup[arc[1]]
        t = bus_lookup[arc[2]]
        union_sets!(uf_before, f, t)
        e in removed || union_sets!(uf_after, f, t)
    end

    _count(uf) = length(Set(get_representative(uf, b) for b in 1:nbus if b != star_idx))
    return _count(uf_after) > _count(uf_before)
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
function NetworkModification(mat::PowerNetworkMatrix, sys::PSY.System, outage::PSY.Outage)
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
            nr,
            arc_lookup,
            arc_sus,
            bus_lookup,
            component,
            direct_mods,
            parallel_mods,
            series_components_by_arc,
            series_arc_tuples,
            shunt_mods,
            component_names,
        )
    end

    # Pass 2: compute series Δb with all tripped components grouped
    series_mods = ArcModification[]
    for (arc_idx, tripped) in series_components_by_arc
        arc_tuple = series_arc_tuples[arc_idx]
        series_chain = nr.series_branch_map[arc_tuple]
        delta_b = _compute_series_outage_delta_b(series_chain, tripped)
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, delta_b)
        push!(series_mods, ArcModification(arc_idx, delta_b, dy11, dy12, dy21, dy22))
    end

    mods = vcat(direct_mods, parallel_mods, series_mods)

    if isempty(mods) && isempty(shunt_mods)
        @info "No valid arc or shunt modifications found for outage. " *
              "The outage may only affect non-network components (e.g., generators)."
    end

    outage_uuid = IS.get_uuid(outage)
    ctg_name = isempty(component_names) ? string(outage_uuid) : join(component_names, "+")

    # A fully-outaged ThreeWindingTransformer isolates its star bus and may
    # island a real terminal bus; flag that on `is_islanding`.
    is_island = false
    arc_ax = get_arc_axis(mat)
    for component in all_components
        _is_three_winding_transformer(component) || continue
        # Every circuit's arc ends at the star bus, so read it off the first one.
        star_num =
            PSY.get_number(PSY.get_to(PSY.get_arc(first(PSY.get_circuits(component)))))
        t3w_mods = [m for m in direct_mods if arc_ax[m.arc_index][2] == star_num]
        is_island = is_island || _3wt_real_bus_islanding(mat, t3w_mods)
    end
    return NetworkModification(ctg_name, mods, shunt_mods, is_island)
end

_is_three_winding_transformer(::Any) = false
_is_three_winding_transformer(::PSY.ThreeWindingTransformer) = true

"""
    _classify_outage_component!(nr, arc_lookup, arc_sus, bus_lookup, component, ...) -> nothing

Classify a single outage component via multiple dispatch. ACTransmission branches are
classified into direct/parallel/series arc modifications. Shunt components produce
diagonal admittance changes. Unsupported component types are silently ignored.
"""
# Phase-shifting transformers are unsupported for contingency classification. The guard is
# data-driven (`_assert_not_phase_shifting` tests `PSY.is_phase_shifting`) and lives at the
# top of the generic `ACTransmission` method below, which also handles all non-phase-shifting
# two-winding transformers.
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
    _assert_not_phase_shifting(component)
    tag, arc_tuple = _resolve_branch_arc(nr, component)

    if tag === :direct
        arc_idx = arc_lookup[arc_tuple]
        b_arc = arc_susceptances[arc_idx]
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_arc, component)
        push!(direct_mods, ArcModification(arc_idx, -b_arc, dy11, dy12, dy21, dy22))
    elseif tag === :parallel
        arc_idx = arc_lookup[arc_tuple]
        # `get_series_susceptance` divides a two-winding transformer's susceptance by
        # its winding tap and has a `ThreeWindingTransformerCircuit` method.
        b_circuit = get_series_susceptance(component, PSY.SU)
        dy11, dy12, dy21, dy22 =
            _compute_arc_ybus_delta(nr, arc_tuple, -b_circuit, component)
        push!(parallel_mods, ArcModification(arc_idx, -b_circuit, dy11, dy12, dy21, dy22))
    elseif tag === :series
        arc_idx = arc_lookup[arc_tuple]
        if !haskey(series_components_by_arc, arc_idx)
            series_components_by_arc[arc_idx] = PSY.ACTransmission[]
            series_arc_tuples[arc_idx] = arc_tuple
        end
        push!(series_components_by_arc[arc_idx], component)
    else
        @info "Branch $(get_name(component)) not found in any reduction map. " *
              "The component may have been eliminated by a radial reduction."
        return
    end
    push!(component_names, get_name(component))
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
        PSY.get_impedance_active_power(component, PSY.SU) -
        im * PSY.get_impedance_reactive_power(component, PSY.SU)
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

function _classify_outage_component!(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    bus_lookup::Dict{Int, Int},
    component::PSY.ThreeWindingTransformer,
    direct_mods::Vector{ArcModification},
    parallel_mods::Vector{ArcModification},
    series_components_by_arc::Dict{Int, Vector{PSY.ACTransmission}},
    series_arc_tuples::Dict{Int, Tuple{Int, Int}},
    shunt_mods::Vector{ShuntModification},
    component_names::Vector{String},
)
    _assert_not_phase_shifting(component)
    # An unavailable parent transformer is already out of service, so it cannot be
    # outaged; skip it regardless of the per-winding availability flags. This mirrors
    # the parent-then-winding gating used when building the Ybus.
    if !PSY.get_available(component)
        return
    end
    for (winding_num, circuit) in enumerate(PSY.get_circuits(component))
        winding = ThreeWindingTransformerCircuit(component, circuit, winding_num)
        if !get_equivalent_available(winding)
            continue
        end
        _classify_outage_component!(
            nr,
            arc_lookup,
            arc_susceptances,
            bus_lookup,
            winding,
            direct_mods,
            parallel_mods,
            series_components_by_arc,
            series_arc_tuples,
            shunt_mods,
            component_names,
        )
    end
    return
end

"""
    _classify_branch_modification(nr, arc_lookup, arc_susceptances, branch) -> Vector{ArcModification}

Classify a single branch component into the appropriate arc modification using
the network reduction reverse maps. For single-branch modifications only;
use `_classify_outage_component!` for multi-component outages with series grouping.
"""
# Phase-shifting transformers are unsupported here too; the data-driven guard
# (`_assert_not_phase_shifting`) sits at the top of the generic `ACTransmission` method
# below rather than in a type-specific method.

"""
    _classify_branch_modification(nr, arc_lookup, arc_susceptances, branch::PSY.ThreeWindingTransformer) -> Vector{ArcModification}

Classify a `ThreeWindingTransformer` by decomposing it into its three winding arcs
and classifying each one individually. Returns arc modifications for all windings
present in the network.
"""
function _classify_branch_modification(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    branch::PSY.ThreeWindingTransformer,
)::Vector{ArcModification}
    _assert_not_phase_shifting(branch)
    # An unavailable parent transformer is already out of service and produces no
    # modifications, irrespective of the per-winding availability flags.
    if !PSY.get_available(branch)
        return ArcModification[]
    end
    mods = ArcModification[]
    for (winding_num, circuit) in enumerate(PSY.get_circuits(branch))
        winding = ThreeWindingTransformerCircuit(branch, circuit, winding_num)
        if !get_equivalent_available(winding)
            continue
        end
        append!(
            mods,
            _classify_branch_modification(nr, arc_lookup, arc_susceptances, winding),
        )
    end
    return mods
end

function _classify_branch_modification(
    nr::NetworkReductionData,
    arc_lookup::Dict,
    arc_susceptances::Vector{Float64},
    branch::PSY.ACTransmission,
)::Vector{ArcModification}
    _assert_not_phase_shifting(branch)
    tag, arc_tuple = _resolve_branch_arc(nr, branch)

    if tag === :direct
        arc_idx = arc_lookup[arc_tuple]
        b_arc = arc_susceptances[arc_idx]
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_arc, branch)
        return [ArcModification(arc_idx, -b_arc, dy11, dy12, dy21, dy22)]
    elseif tag === :parallel
        arc_idx = arc_lookup[arc_tuple]
        # `get_series_susceptance` is tap-aware for two-winding transformers and
        # dispatches the winding wrapper — see the note in `_classify_outage_component!`.
        b_circuit = get_series_susceptance(branch, PSY.SU)
        dy11, dy12, dy21, dy22 =
            _compute_arc_ybus_delta(nr, arc_tuple, -b_circuit, branch)
        return [ArcModification(arc_idx, -b_circuit, dy11, dy12, dy21, dy22)]
    elseif tag === :series
        arc_idx = arc_lookup[arc_tuple]
        series_chain = nr.series_branch_map[arc_tuple]
        delta_b = _compute_series_outage_delta_b(series_chain, branch)
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, delta_b)
        return [ArcModification(arc_idx, delta_b, dy11, dy12, dy21, dy22)]
    else
        @info "Branch $(get_name(branch)) not found in any reduction map. " *
              "The component may have been eliminated by a radial reduction."
        return ArcModification[]
    end
end

# --- Accumulation helpers for Ybus deltas ---

function _accumulate_arc_delta!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    fb_ix::Int,
    tb_ix::Int,
    delta_y11::YBUS_ELTYPE,
    delta_y12::YBUS_ELTYPE,
    delta_y21::YBUS_ELTYPE,
    delta_y22::YBUS_ELTYPE,
)
    push!(I, fb_ix, fb_ix, tb_ix, tb_ix)
    push!(J, fb_ix, tb_ix, fb_ix, tb_ix)
    push!(V, delta_y11, delta_y12, delta_y21, delta_y22)
    return
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

    for arc_mod in mod.arc_modifications
        arc_tuple = arc_ax[arc_mod.arc_index]
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]
        _accumulate_arc_delta!(
            I,
            J,
            V,
            fb_ix,
            tb_ix,
            arc_mod.delta_y11,
            arc_mod.delta_y12,
            arc_mod.delta_y21,
            arc_mod.delta_y22,
        )
    end

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
function apply_ybus_modification(ybus::Ybus, mod::NetworkModification)
    delta = compute_ybus_delta(ybus, mod)
    return ybus.data + delta
end
