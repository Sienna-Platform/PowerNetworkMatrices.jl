"""
    _compute_parallel_partial_ybus_delta(bp, delta_b) -> NTuple{4, YBUS_ELTYPE}

Compute the Pi-model Ybus delta for a partial outage on a parallel branch group.
Finds the circuit whose susceptance matches `delta_b` and returns its negated Pi-model entries.
"""
function _compute_parallel_partial_ybus_delta(
    bp::BranchesParallel,
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    for br in bp.branches
        b_circuit = get_series_susceptance(br)
        if isapprox(-b_circuit, delta_b; atol = YBUS_DELTA_TOL, rtol = 0)
            Y11, Y12, Y21, Y22 = ybus_branch_entries(br)
            return (YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
                YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22))
        end
    end
    error(
        "Could not resolve partial parallel outage to individual circuit Pi-models. " *
        "No circuit in parallel group matches delta_b=$(delta_b).",
    )
end

"""
    _compute_arc_ybus_delta(nr, arc_tuple, delta_b) -> NTuple{4, YBUS_ELTYPE}

Compute the Pi-model Ybus delta `(ΔY11, ΔY12, ΔY21, ΔY22)` for an arc modification.
Dispatches on the branch map that contains `arc_tuple` and handles full vs partial outages.
"""
function _compute_arc_ybus_delta(
    nr::NetworkReductionData,
    arc_tuple::Tuple{Int, Int},
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    if haskey(nr.direct_branch_map, arc_tuple)
        br = nr.direct_branch_map[arc_tuple]
        b_arc = get_series_susceptance(br)
        Y11, Y12, Y21, Y22 = ybus_branch_entries(br)
        if isapprox(delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
            return (YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
                YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22))
        else
            scale = delta_b / b_arc
            return (YBUS_ELTYPE(scale * Y11), YBUS_ELTYPE(scale * Y12),
                YBUS_ELTYPE(scale * Y21), YBUS_ELTYPE(scale * Y22))
        end
    elseif haskey(nr.parallel_branch_map, arc_tuple)
        bp = nr.parallel_branch_map[arc_tuple]
        b_arc = get_series_susceptance(bp)
        if isapprox(delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
            Y11, Y12, Y21, Y22 = ybus_branch_entries(bp)
            return (YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
                YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22))
        else
            return _compute_parallel_partial_ybus_delta(bp, delta_b)
        end
    elseif haskey(nr.series_branch_map, arc_tuple)
        series_chain = nr.series_branch_map[arc_tuple]
        b_arc = get_series_susceptance(series_chain)
        if isapprox(delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
            Y11, Y12, Y21, Y22 = ybus_branch_entries(series_chain)
            return (YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
                YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22))
        else
            error(
                "Partial Ybus delta is not supported on series-reduced arcs. " *
                "Arc $(arc_tuple), Δb=$(delta_b).",
            )
        end
    elseif haskey(nr.transformer3W_map, arc_tuple)
        tr = nr.transformer3W_map[arc_tuple]
        b_arc = get_series_susceptance(tr)
        if !isapprox(delta_b, -b_arc; atol = YBUS_DELTA_TOL, rtol = 0)
            error(
                "Partial Ybus delta is not supported on 3-winding transformer arcs. " *
                "Arc $(arc_tuple), Δb=$(delta_b).",
            )
        end
        Y11, Y12, Y21, Y22 = ybus_branch_entries(tr)
        return (YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
            YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22))
    else
        error(
            "Arc $(arc_tuple) not found in any network reduction map. " *
            "Cannot compute Ybus Pi-model delta.",
        )
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
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, delta_b)
        push!(series_mods, ArcModification(arc_idx, delta_b, dy11, dy12, dy21, dy22))
    end

    mods = vcat(direct_mods, parallel_mods, series_mods)

    if isempty(mods) && isempty(shunt_mods)
        @info "No valid arc or shunt modifications found for outage. " *
              "The outage may only affect non-network components (e.g., generators)."
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
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_arc)
        push!(direct_mods, ArcModification(arc_idx, -b_arc, dy11, dy12, dy21, dy22))
    elseif tag === :parallel
        arc_idx = arc_lookup[arc_tuple]
        b_circuit = PSY.get_series_susceptance(component)
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_circuit)
        push!(parallel_mods, ArcModification(arc_idx, -b_circuit, dy11, dy12, dy21, dy22))
    elseif tag === :series
        arc_idx = arc_lookup[arc_tuple]
        if !haskey(series_components_by_arc, arc_idx)
            series_components_by_arc[arc_idx] = PSY.ACTransmission[]
            series_arc_tuples[arc_idx] = arc_tuple
        end
        push!(series_components_by_arc[arc_idx], component)
    else
        @info "Branch $(PSY.get_name(component)) not found in any reduction map. " *
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

function _classify_outage_component!(
    ::NetworkReductionData,
    ::Dict,
    ::Vector{Float64},
    ::Dict{Int, Int},
    component::PSY.ThreeWindingTransformer,
    ::Vector{ArcModification},
    ::Vector{ArcModification},
    ::Dict{Int, Vector{PSY.ACTransmission}},
    ::Dict{Int, Tuple{Int, Int}},
    ::Vector{ShuntModification},
    ::Vector{String},
)
    error(
        "Outages on ThreeWindingTransformer components are not yet supported. " *
        "Component: $(PSY.get_name(component)). " *
        "Use individual ThreeWindingTransformerWinding arcs instead.",
    )
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
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_arc)
        return [ArcModification(arc_idx, -b_arc, dy11, dy12, dy21, dy22)]
    elseif tag === :parallel
        arc_idx = arc_lookup[arc_tuple]
        b_circuit = PSY.get_series_susceptance(branch)
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_circuit)
        return [ArcModification(arc_idx, -b_circuit, dy11, dy12, dy21, dy22)]
    elseif tag === :series
        arc_idx = arc_lookup[arc_tuple]
        series_chain = nr.series_branch_map[arc_tuple]
        delta_b = _compute_series_outage_delta_b(series_chain, branch)
        dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, delta_b)
        return [ArcModification(arc_idx, delta_b, dy11, dy12, dy21, dy22)]
    else
        @info "Branch $(PSY.get_name(branch)) not found in any reduction map. " *
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
            I, J, V, fb_ix, tb_ix,
            arc_mod.delta_y11, arc_mod.delta_y12,
            arc_mod.delta_y21, arc_mod.delta_y22,
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
function apply_ybus_modification(
    ybus::Ybus,
    mod::NetworkModification,
)
    delta = compute_ybus_delta(ybus, mod)
    return ybus.data + delta
end

# --- Accessors so BA_Matrix and ABA_Matrix can be used directly with
#     NetworkModification(mat, sys, outage). Previously only VirtualPTDF /
#     VirtualMODF defined these.

"""
    _get_arc_susceptances(ba::BA_Matrix) -> Vector{Float64}

Extract per-arc series susceptance from a `BA_Matrix`. Each column of `ba.data`
has `+b` at the from-bus row and `-b` at the to-bus row; the from-bus entry is
returned for each arc. Aligned with `get_arc_axis(ba)`.
"""
function _get_arc_susceptances(ba::BA_Matrix)
    arc_ax = get_arc_axis(ba)
    bus_lookup = get_bus_lookup(ba)
    nr = get_network_reduction_data(ba)
    n_arcs = length(arc_ax)
    sus = Vector{Float64}(undef, n_arcs)
    rows = SparseArrays.rowvals(ba.data)
    nzs = SparseArrays.nonzeros(ba.data)
    for (i, arc) in enumerate(arc_ax)
        fb = get_bus_index(arc[1], bus_lookup, nr)
        found = 0.0
        for nz in SparseArrays.nzrange(ba.data, i)
            if rows[nz] == fb
                found = nzs[nz]
                break
            end
        end
        sus[i] = found
    end
    return sus
end

# --- Tap-ratio guard for ABA delta -------------------------------------
#
# Ybus entries for transformers with off-nominal taps are asymmetric
# (`Y11 = Y_t/abs2(tap)`, `Y22 = Y_t`, `Y12 = -Y_t/c_tap`, `Y21 = -Y_t/tap`).
# `compute_aba_delta`'s symmetric `ΔABA = Uᵀ·diag(Δb)·U` only holds when
# tap=1 and α=0; asymmetric Woodbury for tapped branches is not implemented.

"""Return the tap ratio of `branch`, defaulting to `1.0` for types that do
not carry one (lines, non-tap two-winding transformers)."""
_branch_tap(::PSY.ACTransmission) = 1.0
_branch_tap(br::PSY.TwoWindingTransformer) = PSY.get_tap(br)

function _branch_tap(tw::ThreeWindingTransformerWinding)
    tfw = get_transformer(tw)
    winding_num = get_winding_number(tw)
    if winding_num == 1
        return PSY.get_primary_turns_ratio(tfw)
    elseif winding_num == 2
        return PSY.get_secondary_turns_ratio(tfw)
    elseif winding_num == 3
        return PSY.get_tertiary_turns_ratio(tfw)
    else
        throw(ArgumentError("Invalid winding number: $winding_num"))
    end
end

"""Phase-shift angle (radians), default zero."""
_branch_alpha(::PSY.ACTransmission) = 0.0
_branch_alpha(br::PSY.PhaseShiftingTransformer) = PSY.get_α(br)

function _branch_alpha(
    tw::ThreeWindingTransformerWinding{PSY.PhaseShiftingTransformer3W},
)
    tfw = get_transformer(tw)
    winding_num = get_winding_number(tw)
    if winding_num == 1
        return PSY.get_α_primary(tfw)
    elseif winding_num == 2
        return PSY.get_α_secondary(tfw)
    elseif winding_num == 3
        return PSY.get_α_tertiary(tfw)
    else
        throw(ArgumentError("Invalid winding number: $winding_num"))
    end
end

"""Check whether a single branch has an off-nominal tap or a non-zero phase
shift. Raises `ArgumentError` with a specific message if so."""
function _assert_nominal_branch_for_aba_delta(branch::PSY.ACTransmission)
    tap = _branch_tap(branch)
    if !isapprox(tap, 1.0; atol = 1e-9)
        throw(
            ArgumentError(
                "compute_aba_delta does not yet support off-nominal tap ratios: " *
                "branch '$(PSY.get_name(branch))' has tap=$tap. The symmetric " *
                "Woodbury decomposition requires tap=1.0; asymmetric Woodbury " *
                "for tapped transformers is a planned follow-on (see the " *
                "PF-DC-Contingency-Analysis project note). For now, exclude " *
                "off-nominal-tap branches from the contingency set.",
            ),
        )
    end
    α = _branch_alpha(branch)
    if !isapprox(α, 0.0; atol = 1e-9)
        throw(
            ArgumentError(
                "compute_aba_delta does not support phase-shifting transformers: " *
                "branch '$(PSY.get_name(branch))' has α=$α rad. Exclude " *
                "phase-shifters from the contingency set.",
            ),
        )
    end
    return nothing
end

"""Validate that no branch(es) associated with `arc_tuple` have off-nominal
tap or phase shift. Resolves through direct / parallel / series maps."""
function _assert_nominal_arc_for_aba_delta(
    nr::NetworkReductionData,
    arc_tuple::Tuple{Int, Int},
)
    if haskey(nr.direct_branch_map, arc_tuple)
        _assert_nominal_branch_for_aba_delta(nr.direct_branch_map[arc_tuple])
    elseif haskey(nr.parallel_branch_map, arc_tuple)
        bp = nr.parallel_branch_map[arc_tuple]
        for br in bp.branches
            _assert_nominal_branch_for_aba_delta(br)
        end
    elseif haskey(nr.series_branch_map, arc_tuple)
        chain = nr.series_branch_map[arc_tuple]
        # BranchesSeries.branches is a Dict{DataType, Vector{<:ACTransmission}}
        for (_, brs) in chain.branches
            for br in brs
                _assert_nominal_branch_for_aba_delta(br)
            end
        end
    end
    return nothing
end

# --- Bridge from NetworkModification to ABA (DC) domain ---

"""
    ABADelta

Bundle returned by [`compute_aba_delta`](@ref). Carries both the assembled sparse
ΔABA matrix (for generic use or convenience add) and the Woodbury-ready pair
`(U, delta_b)` so callers doing Sherman-Morrison-Woodbury updates do not have to
reconstruct the incidence columns.

# Fields
- `delta::SparseMatrixCSC{Float64, Int}` — the sparse delta, same size as `aba.data`.
  Ref-bus rows/columns are already dropped.
- `U::SparseMatrixCSC{Float64, Int}` — reduced arc incidence matrix for the modified
  arcs. Size `n_valid × M` where `M = length(mod.arc_modifications)`. Each column has
  entries `+1` at the from-bus row and `-1` at the to-bus row (either may be absent
  if the corresponding endpoint is a reference bus).
- `delta_b::Vector{Float64}` — susceptance deltas, length `M`. `Δb = -b_arc` for a
  full outage.
- `arc_indices::Vector{Int}` — arc indices of the modified arcs, length `M`,
  aligned with `delta_b` and the columns of `U`.
- `skipped_shunts::Int` — count of `ShuntModification`s present in the input `mod`
  that were skipped (DC ABA is susceptance-only; shunt admittance deltas do not
  contribute to susceptance).
"""
struct ABADelta
    delta::SparseArrays.SparseMatrixCSC{Float64, Int}
    U::SparseArrays.SparseMatrixCSC{Float64, Int}
    delta_b::Vector{Float64}
    arc_indices::Vector{Int}
    skipped_shunts::Int
end

"""
    compute_aba_delta(aba::ABA_Matrix, mod::NetworkModification) -> ABADelta

Compute the DC susceptance-domain ΔABA for a canonical `NetworkModification`, plus
the Woodbury-ready `(U, delta_b)` pair.

The ABA matrix is `Aᵀ diag(b) A` on the reduced bus basis (reference buses removed).
For a set of arc modifications with susceptance deltas `Δb`, the delta matrix is

    ΔABA = Aᵤᵀ diag(Δb) Aᵤ

where `Aᵤ` is the reduced arc incidence matrix restricted to the modified arcs.

This is the DC companion to [`compute_ybus_delta`](@ref). Arc modifications
contribute via `delta_b`; shunt modifications are silently skipped (they are
purely complex admittance changes and do not affect the DC susceptance B matrix).

# Notes
- If an arc touches a reference bus, the corresponding row in that column of `U`
  is omitted (only one of the two endpoint entries is present).
- The returned `delta` shares the sparsity pattern footprint of the original ABA
  for typical branch-outage modifications, so `aba.data + delta` preserves the
  structural pattern. For in-place numeric refactor paths that rely on
  `check_pattern=true`, callers should mutate an `nzval` copy at known positions
  rather than use Julia's sparse `+` (which canonicalizes explicit zeros).
"""
function compute_aba_delta(
    aba::ABA_Matrix,
    mod::NetworkModification,
)::ABADelta
    bus_lookup = get_bus_lookup(aba)
    n = size(aba.data, 1)
    nr = get_network_reduction_data(aba)
    arc_ax = get_arc_axis(nr)
    n_arcs_mod = length(mod.arc_modifications)

    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{Float64}()
    sizehint!(I, 4 * n_arcs_mod)
    sizehint!(J, 4 * n_arcs_mod)
    sizehint!(V, 4 * n_arcs_mod)

    U_I = Vector{Int}()
    U_J = Vector{Int}()
    U_V = Vector{Float64}()
    sizehint!(U_I, 2 * n_arcs_mod)
    sizehint!(U_J, 2 * n_arcs_mod)
    sizehint!(U_V, 2 * n_arcs_mod)

    delta_b = Vector{Float64}()
    arc_indices = Vector{Int}()
    sizehint!(delta_b, n_arcs_mod)
    sizehint!(arc_indices, n_arcs_mod)

    kept = 0
    for arc_mod in mod.arc_modifications
        arc_tuple = arc_ax[arc_mod.arc_index]
        _assert_nominal_arc_for_aba_delta(nr, arc_tuple)
        fb_ix = get(bus_lookup, arc_tuple[1], 0)
        tb_ix = get(bus_lookup, arc_tuple[2], 0)
        if fb_ix == 0 && tb_ix == 0
            throw(
                ArgumentError(
                    "Modified arc $(arc_tuple) has both endpoints at reference buses. " *
                    "This is a degenerate topology for DC contingency analysis; " *
                    "the arc contributes nothing to the reduced ABA basis and " *
                    "indicates either a malformed system or an accidentally " *
                    "double-ref subnetwork.",
                ),
            )
        end
        kept += 1
        Δb = arc_mod.delta_b
        push!(delta_b, Δb)
        push!(arc_indices, arc_mod.arc_index)

        # U column: +1 at fb, -1 at tb (drop ref entries)
        if fb_ix != 0
            push!(U_I, fb_ix)
            push!(U_J, kept)
            push!(U_V, 1.0)
        end
        if tb_ix != 0
            push!(U_I, tb_ix)
            push!(U_J, kept)
            push!(U_V, -1.0)
        end

        # ΔABA = A_colᵀ Δb A_col expansion, ref entries dropped
        if fb_ix != 0
            push!(I, fb_ix)
            push!(J, fb_ix)
            push!(V, Δb)
        end
        if fb_ix != 0 && tb_ix != 0
            push!(I, fb_ix)
            push!(J, tb_ix)
            push!(V, -Δb)
            push!(I, tb_ix)
            push!(J, fb_ix)
            push!(V, -Δb)
        end
        if tb_ix != 0
            push!(I, tb_ix)
            push!(J, tb_ix)
            push!(V, Δb)
        end
    end

    delta = SparseArrays.sparse(I, J, V, n, n)
    U = SparseArrays.sparse(U_I, U_J, U_V, n, kept)
    return ABADelta(delta, U, delta_b, arc_indices, length(mod.shunt_modifications))
end

"""
    apply_aba_modification(aba::ABA_Matrix, mod::NetworkModification) -> SparseMatrixCSC

Apply a canonical NetworkModification to an ABA matrix and return the modified
sparse matrix. Convenience wrapper around [`compute_aba_delta`](@ref) for
correctness reference; performance-critical paths should mutate an `nzval` copy
in place to preserve exact sparsity pattern (see notes on `compute_aba_delta`).
"""
function apply_aba_modification(
    aba::ABA_Matrix,
    mod::NetworkModification,
)
    d = compute_aba_delta(aba, mod)
    return aba.data + d.delta
end
