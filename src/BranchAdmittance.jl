# ── Branch π-model admittance ────────────────────────────────────────────────
# Compute a branch's π-model admittance as a
# `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` NamedTuple following the PowerModels
# convention: `g + im*b == 1/(r + im*x)` is the series admittance, `*_fr`/`*_to` are the
# from/to shunts, and `tap`/`shift` are the transformer ratio / phase shift (radians).

# Split a two-winding transformer's complex magnetizing shunt onto the π-model from/to shunt
# slots per its `PSY.TwoWindingTransformerShuntLocation`: PRIMARY places the full value on
# the from side, SECONDARY on the to side, SPLIT the full value on both sides.
function _magnetizing_shunt_split(
    y_shunt::ComplexF64,
    location::PSY.TwoWindingTransformerShuntLocation.Value,
)
    g, b = real(y_shunt), imag(y_shunt)
    if location == PSY.TwoWindingTransformerShuntLocation.PRIMARY
        return (g_fr = g, b_fr = b, g_to = 0.0, b_to = 0.0)
    elseif location == PSY.TwoWindingTransformerShuntLocation.SECONDARY
        return (g_fr = 0.0, b_fr = 0.0, g_to = g, b_to = b)
    elseif location == PSY.TwoWindingTransformerShuntLocation.SPLIT
        return (g_fr = g, b_fr = b, g_to = g, b_to = b)
    end
    error(
        "Unhandled TwoWindingTransformerShuntLocation: $location. " *
        "Expected PRIMARY, SECONDARY, or SPLIT.",
    )
end

# Split a three-winding transformer's complex magnetizing shunt for one circuit's π-model.
# The shunt lives on the parent transformer and lands on circuit 1 only, per its
# `PSY.ThreeWindingTransformerShuntLocation`: PRIMARY on the terminal (from) side, STAR on
# the star-node (to) side. Circuits 2 and 3 carry no shunt.
function _three_winding_shunt_split(
    y_shunt::ComplexF64,
    location::PSY.ThreeWindingTransformerShuntLocation.Value,
    winding_number::Int,
)
    winding_number == 1 || return (g_fr = 0.0, b_fr = 0.0, g_to = 0.0, b_to = 0.0)
    g, b = real(y_shunt), imag(y_shunt)
    if location == PSY.ThreeWindingTransformerShuntLocation.PRIMARY
        return (g_fr = g, b_fr = b, g_to = 0.0, b_to = 0.0)
    elseif location == PSY.ThreeWindingTransformerShuntLocation.STAR
        return (g_fr = 0.0, b_fr = 0.0, g_to = g, b_to = b)
    end
    error(
        "Unhandled ThreeWindingTransformerShuntLocation: $location. " *
        "Expected PRIMARY or STAR.",
    )
end

# Kept out of line so the error body does not count against the caller's inlining budget.
@noinline function _throw_non_finite_susceptance(segment::PSY.ACTransmission, b::Float64)
    error(
        "Series susceptance of $(get_name(segment)) is $(b): the branch has x == 0. " *
        "Ybus assembly substitutes the reduction's minimum retained impedance for such a " *
        "branch, so a consumer that needs the value the matrices use should call " *
        "`get_effective_series_susceptance(segment, nr)` instead.",
    )
end

"""
    get_series_susceptance(b::PSY.ACTransmission, units::IS.AbstractUnitSystem)

Series susceptance `1/x` of an `PSY.ACTransmission` branch, from
the stored series reactance alone. `PSY.TwoWindingTransformer` has a more specific
method (below) that additionally divides by the winding tap ratio
(`PSY.get_tap(PSY.get_circuit(t))`). This is a deliberate asymmetry: only the susceptance
form is tap-divided; Ybus/PTDF/LODF assembly needs the tap-divided value, while callers that
need the untapped complex admittance should build it directly from `PSY.get_r`/`PSY.get_x`.

Throws if the branch has `r == x == 0` (susceptance is non-finite). A consumer that needs
the value the matrices actually use should call `get_effective_series_susceptance` instead.
"""
function get_series_susceptance(b::PSY.ACTransmission, units::IS.AbstractUnitSystem)
    v = _series_susceptance_raw(b, units)
    isfinite(v) || _throw_non_finite_susceptance(b, v)
    return v
end

"""
    get_series_susceptance(t::PSY.TwoWindingTransformer, units::IS.AbstractUnitSystem)

Series susceptance of a `PSY.TwoWindingTransformer`: the generic `ACTransmission`
value (`1/x`) divided by the winding tap ratio `PSY.get_tap(PSY.get_circuit(t))`. A
fixed-ratio transformer has `tap = 1.0`, so this is a no-op for it and matches the plain
`ACTransmission` value.

Throws if the branch has `r == x == 0` (susceptance is non-finite). A consumer that needs
the value the matrices actually use should call `get_effective_series_susceptance` instead.
"""
function get_series_susceptance(t::PSY.TwoWindingTransformer, units::IS.AbstractUnitSystem)
    v = _series_susceptance_raw(t, units)
    isfinite(v) || _throw_non_finite_susceptance(t, v)
    return v
end

# The single home for the `1/x` arithmetic. Returns `Inf` when `r == x == 0`:
# `get_series_susceptance` rejects that, `_finite_series_susceptance` substitutes.
_series_susceptance_raw(b::PSY.ACTransmission, units::IS.AbstractUnitSystem) =
    1 / PSY.get_x(b, units)
# A Ward equivalent is detached, so it cannot resolve the system base and any other unit
# system throws. Its r/x are already system-base values, which `PSY.CU` returns unchanged —
# the same reading `equivalent_branch` takes for this type.
_series_susceptance_raw(b::PSY.GenericArcImpedance, ::IS.AbstractUnitSystem) =
    1 / PSY.get_x(b, PSY.CU)
_series_susceptance_raw(t::PSY.TwoWindingTransformer, units::IS.AbstractUnitSystem) =
    _series_susceptance_raw(PSY.get_circuit(t), units)
_series_susceptance_raw(c::PSY.TransformerCircuit, units::IS.AbstractUnitSystem) =
    (1 / PSY.get_x(c, units)) / PSY.get_tap(c)

"""
    get_series_phase_shift(br) -> Float64

Phase-shift angle α (radians) of the series element of a branch's π-model, in the branch's
own from → to orientation. Non-transformer branches never shift. This is the exact stored
angle for the DC injection model (`f = b·(θ_f − θ_t − α)`), unlike the numerically recovered
`get_equivalent_shift` of an aggregate's `EquivalentBranch`.
"""
function get_series_phase_shift(::PSY.ACTransmission)
    return 0.0
end

# A circuit's series shift is its stored α. Internal: a circuit reaches the public accessor
# through its transformer, or through `ThreeWindingTransformerCircuit` at three windings.
_circuit_phase_shift(c::PSY.TransformerCircuit) = PSY.get_α(c)

function get_series_phase_shift(t::PSY.TwoWindingTransformer)
    return _circuit_phase_shift(PSY.get_circuit(t))
end

# Both π shunts in one pass: the explicit-units getters convert the whole from/to pair, so
# reading each once and taking both fields statically halves the unit-conversion work.
# ComplexF64, not YBUS_ELTYPE — the π layer is Float64; narrowing happens at Ybus storage.
function _get_shunts(br::PSY.ACTransmission)
    g = PSY.get_g(br, PSY.SU)
    b = PSY.get_b(br, PSY.SU)
    return (complex(g.from, b.from), complex(g.to, b.to))
end

_get_shunts(::PSY.DiscreteControlledACBranch) = (zero(ComplexF64), zero(ComplexF64))

# Kept out of line so the `@warn` body does not count against the caller's inlining budget.
# ZIR merges a zero-impedance line's endpoints but excludes transformer arcs, so the
# substitution is transient for a line and permanent for a transformer.
@noinline function _warn_zero_impedance(b::PSY.ACTransmission, min_x_eps::Float64)
    fate = if _is_transformer(b)
        "ZeroImpedanceBranchReduction excludes transformer arcs, so the substituted reactance is retained."
    else
        "This branch will be reduced by ZeroImpedanceBranchReduction unless its endpoints are irreducible."
    end
    @warn "Branch $(get_name(b)) has r=0.0 and x=0.0; substituting x=$(min_x_eps) to avoid division by zero. $fate"
    return
end

# `@inline` because the explicit-units getters expand past the inliner's cost model (~35 IR
# statements each, from the `base_value` check and its error branch), so without it
# `equivalent_branch` stays an out-of-line call in `branch_admittance` on the line path. The
# transformer methods below are deliberately unannotated: forcing them inline inflates the
# caller and measures no faster. Keep nothing between the docstring and the definition — a
# comment there silently detaches the docstring.
"""
    equivalent_branch(b; min_x_eps) -> EquivalentBranch

The π-model of a single branch in **impedance** form: `(r, x, g_from, b_from, g_to, b_to,
tap, shift)`. This is PNM's single source of truth for branch electrical parameters — both
[`branch_admittance`](@ref) (admittance form) and `ybus_branch_entries` (Ybus 2×2) are
derived from it, and [`arc_equivalent_branch`](@ref) resolves any arc to one.

This method is the component's **own** π-model: no impedance correction, and `min_x_eps`
rather than a reduction's configured substitute reactance. `branch_admittance(b)` and
`ybus_branch_entries(b)` are the admittance and Ybus views of exactly this, for consumers
holding a branch with no reduction in play. The `nr` forms of all three report what the AC
matrices carry instead, and the two families must not be mixed over one network. (The DC side is a separate contract, and a mixed one: phase-shifting arcs
and standalone series chains read component reactances, which the correction does not touch,
while every other arc inherits it through Ybus. It warns when a correction is active, see
`_warn_impedance_correction_in_dc`.)

Methods exist for lines, `GenericArcImpedance` Ward equivalents, and transformer circuits of
either arity — a transformer's series data lives on its `PSY.TransformerCircuit`, so 2W and
3W differ only in which shunt-placement enum applies.

The from/to shunts carry the real `PSY.get_g` conductance. A caller wanting PowerModels'
`get_branch_to_pm` convention (which drops `g`) must zero it at that boundary.

`min_x_eps` substitutes for `x` when `r == x == 0`, for every branch type.
"""
@inline function equivalent_branch(
    b::PSY.ACTransmission;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    r = PSY.get_r(b, PSY.SU)
    x = PSY.get_x(b, PSY.SU)
    if iszero(r) && iszero(x)
        _warn_zero_impedance(b, min_x_eps)
        x = min_x_eps
    end
    y_fr, y_to = _get_shunts(b)
    return EquivalentBranch(
        r, x,
        real(y_fr), imag(y_fr), real(y_to), imag(y_to),
        1.0, 0.0,
    )
end

# A detached Ward equivalent's r/x are already system-base values, so they are read on the
# device base (`PSY.CU`), which returns them unchanged — a detached component cannot resolve
# the system base power. It carries no shunts.
function equivalent_branch(
    b::PSY.GenericArcImpedance;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    r = PSY.get_r(b, PSY.CU)
    x = PSY.get_x(b, PSY.CU)
    if iszero(r) && iszero(x)
        _warn_zero_impedance(b, min_x_eps)
        x = min_x_eps
    end
    return EquivalentBranch(
        r, x,
        0.0, 0.0, 0.0, 0.0,
        1.0, 0.0,
    )
end

function equivalent_branch(
    b::PSY.TwoWindingTransformer;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    sh = _magnetizing_shunt_split(
        PSY.get_magnetizing_shunt(b, PSY.SU),
        PSY.get_shunt_location(b),
    )
    return _circuit_equivalent_branch(PSY.get_circuit(b), sh, b, min_x_eps)
end

function equivalent_branch(
    w::ThreeWindingTransformerCircuit;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    transformer = get_transformer(w)
    sh = _three_winding_shunt_split(
        PSY.get_magnetizing_shunt(transformer, PSY.SU),
        PSY.get_shunt_location(transformer),
        get_winding_number(w),
    )
    return _circuit_equivalent_branch(w.circuit, sh, w, min_x_eps)
end

# The whole of "a transformer is a circuit": the circuit owns r/x/tap/α for both arities, so
# 2W and 3W differ only in which shunt-split rule produced `sh`. `br` is the nameable owner —
# a bare `PSY.TransformerCircuit` has no `get_name`, so the warning cannot be raised from here
# without it.
function _circuit_equivalent_branch(
    circuit::PSY.TransformerCircuit,
    sh::@NamedTuple{g_fr::Float64, b_fr::Float64, g_to::Float64, b_to::Float64},
    br::PSY.ACTransmission,
    min_x_eps::Float64,
)
    r = PSY.get_r(circuit, PSY.SU)
    x = PSY.get_x(circuit, PSY.SU)
    if iszero(r) && iszero(x)
        _warn_zero_impedance(br, min_x_eps)
        x = min_x_eps
    end
    return EquivalentBranch(
        r, x,
        sh.g_fr, sh.b_fr, sh.g_to, sh.b_to,
        PSY.get_tap(circuit), PSY.get_α(circuit),
    )
end

"""
    branch_admittance(b; min_x_eps) -> NamedTuple

π-model admittance `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` of a branch on its own terms,
where `g + im*b == 1 / (r + im*x)` is the series admittance. The admittance-form view of
[`equivalent_branch`](@ref) and carries that method's meaning exactly: no impedance
correction, and `min_x_eps` rather than a reduction's configured substitute reactance.

Use `branch_admittance(b, nr)` whenever an `nr` is available. This form is for consumers
holding a branch with no reduction in play, and it must never be mixed with the corrected
form for the same network.
"""
function branch_admittance(
    b::PSY.ACTransmission;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    return _to_admittance(equivalent_branch(b; min_x_eps = min_x_eps))
end

function _to_admittance(eb::EquivalentBranch)
    ys = 1.0 / (get_equivalent_r(eb) + get_equivalent_x(eb) * im)
    return (
        g = real(ys), b = imag(ys),
        g_fr = get_equivalent_g_from(eb), b_fr = get_equivalent_b_from(eb),
        g_to = get_equivalent_g_to(eb), b_to = get_equivalent_b_to(eb),
        tap = get_equivalent_tap(eb), shift = get_equivalent_shift(eb),
    )
end

# ── Impedance correction (PSS/E transformer impedance correction tables) ─────

"""
Linearly interpolate `curve` at `x`, holding the end values outside the tabulated range.

PSS/E impedance correction tables are sparse and the operating point regularly sits outside
them; clamping matches PSS/E, which applies the nearest tabulated factor rather than
extrapolating.
"""
function _interpolate_correction_factor(curve::IS.PiecewiseLinearData, x::Real)
    points = IS.get_points(curve)
    x = clamp(x, points[1].x, points[end].x)
    for i in 1:(length(points) - 1)
        if x <= points[i + 1].x
            dx = points[i + 1].x - points[i].x
            iszero(dx) && return points[i].y
            t = (x - points[i].x) / dx
            return points[i].y + t * (points[i + 1].y - points[i].y)
        end
    end
    return points[end].y
end

"""
The series-impedance multiplier `ict` prescribes for `circuit` at its present operating point.

One method covers both arities: a transformer's tap and phase shift live on its
`PSY.TransformerCircuit` regardless of winding count.
"""
function _evaluate_correction_table(
    circuit::PSY.TransformerCircuit,
    ict::PSY.ImpedanceCorrectionData,
)
    mode = PSY.get_transformer_control_mode(ict)
    if mode == PSY.ImpedanceCorrectionTransformerControlMode.TAP_RATIO
        x = abs(PSY.get_tap(circuit))
    else
        # The table's x-values are degrees; `α` is stored in radians.
        x = rad2deg(PSY.get_α(circuit))
    end
    return _interpolate_correction_factor(PSY.get_impedance_correction_curve(ict), x)
end

# `WindingCategory` encodes the winding position directly (`TR2W_WINDING = 0`, then
# `PRIMARY_WINDING`/`SECONDARY_WINDING`/`TERTIARY_WINDING` = 1/2/3, matching
# `PSY.get_circuits` order), so the enum
# value doubles as the map key and the 3W circuit index.
_winding_index(category::PSY.WindingCategory.Value) = Int(category)

"""
    build_impedance_correction_factors!(nr::NetworkReductionData, sys::PSY.System)

Evaluate every `PSY.ImpedanceCorrectionData` attached to a transformer in `sys` and cache the
resulting series-impedance multipliers on `nr`, keyed by `(transformer id, winding)`.

Evaluated once per `Ybus` build rather than per branch: resolving a component's supplemental
attributes queries the association store, which is far too costly for the assembly loop.
"""
function build_impedance_correction_factors!(nr::NetworkReductionData, sys::PSY.System)
    # In-memory type check; the pair queries below each round-trip the association store.
    isempty(PSY.get_supplemental_attributes(PSY.ImpedanceCorrectionData, sys)) && return
    factors = nr.impedance_correction_factors
    # A 2W transformer has one circuit, so its table's winding tag carries no information and
    # is not validated by PSY; key on the 2W code regardless of the tag.
    for pair in PSY.get_component_supplemental_attribute_pairs(
        PSY.TwoWindingTransformer,
        PSY.ImpedanceCorrectionData,
        sys,
    )
        transformer = pair.component
        factors[(
            IS.get_id(transformer),
            _winding_index(PSY.WindingCategory.TR2W_WINDING),
        )] =
            _evaluate_correction_table(
                PSY.get_circuit(transformer),
                pair.supplemental_attribute,
            )
    end
    for pair in PSY.get_component_supplemental_attribute_pairs(
        PSY.ThreeWindingTransformer,
        PSY.ImpedanceCorrectionData,
        sys,
    )
        transformer = pair.component
        ict = pair.supplemental_attribute
        winding = _winding_index(PSY.get_transformer_winding(ict))
        if !(winding in 1:3)
            error(
                "ImpedanceCorrectionData table $(PSY.get_table_number(ict)) on " *
                "ThreeWindingTransformer $(PSY.get_name(transformer)) is tagged " *
                "$(PSY.get_transformer_winding(ict)); expected PRIMARY_, SECONDARY_ or " *
                "TERTIARY_WINDING.",
            )
        end
        factors[(IS.get_id(transformer), winding)] =
            _evaluate_correction_table(PSY.get_circuits(transformer)[winding], ict)
    end
    return
end

# Only transformers carry correction tables; every other branch kind is uncorrected.
_impedance_correction_factor(::PSY.ACTransmission, ::NetworkReductionData) = 1.0

_impedance_correction_factor(br::PSY.TwoWindingTransformer, nr::NetworkReductionData) =
    get(
        nr.impedance_correction_factors,
        (IS.get_id(br), _winding_index(PSY.WindingCategory.TR2W_WINDING)),
        1.0,
    )

_impedance_correction_factor(w::ThreeWindingTransformerCircuit, nr::NetworkReductionData) =
    get(
        nr.impedance_correction_factors,
        (IS.get_id(get_transformer(w)), get_winding_number(w)),
        1.0,
    )

"""
    equivalent_branch(b, nr::NetworkReductionData) -> EquivalentBranch

The π-model of any arc entry as the assembled matrices see it: [`equivalent_branch`](@ref) of
a single branch with the impedance correction `nr` caches for it applied to the series
impedance, or the reduction-aware equivalent of an aggregate. The `nr`-less method is the
uncorrected component value on the default epsilon; here the zero-impedance substitute is
the reduction's configured `minimum_retained_impedance`. Correction scales the impedance
rather than the admittance so it composes with the tap, shift and shunt terms as PSS/E
defines it.
"""
function equivalent_branch(b::PSY.ACTransmission, nr::NetworkReductionData)
    eb = equivalent_branch(b; min_x_eps = _minimum_retained_impedance(nr))
    factor = _impedance_correction_factor(b, nr)
    isone(factor) && return eb
    return EquivalentBranch(
        get_equivalent_r(eb) * factor,
        get_equivalent_x(eb) * factor,
        get_equivalent_g_from(eb),
        get_equivalent_b_from(eb),
        get_equivalent_g_to(eb),
        get_equivalent_b_to(eb),
        get_equivalent_tap(eb),
        get_equivalent_shift(eb),
    )
end

function equivalent_branch(group::AbstractReductionAggregate, nr::NetworkReductionData)
    return get_equivalent_physical_branch_parameters(group, nr)
end

"""
    branch_admittance(b::PSY.ACTransmission, nr::NetworkReductionData) -> NamedTuple

π-model admittance `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` of a single branch as the
assembled matrices carry it, where `g + im*b == 1 / (r + im*x)` is the series admittance.
The admittance-form view of [`equivalent_branch`](@ref); see it for the shunt and unit
conventions, and for what `nr` contributes.

Prefer this form wherever an `nr` is in hand: an admittance that skips `nr`'s impedance
correction does not match what was stamped. The `nr`-less method above is the component's
own π-model, for callers with no reduction.
"""
function branch_admittance(b::PSY.ACTransmission, nr::NetworkReductionData)
    return _to_admittance(equivalent_branch(b, nr))
end

"""
    branch_admittance(segment::AbstractReductionAggregate, nr::NetworkReductionData) -> NamedTuple

π-model admittance of a reduction-aggregated arc — a `BranchesSeries` chain or a
`BranchesParallel` group — from PNM's reduction-aware equivalent physical branch
parameters. Series/parallel equivalents of lines carry `tap == 1`.
"""
function branch_admittance(segment::AbstractReductionAggregate, nr::NetworkReductionData)
    return _to_admittance(equivalent_branch(segment, nr))
end

"""
    reduced_arc_admittance(nr::NetworkReductionData, from_no::Int, to_no::Int) -> Union{NamedTuple, Nothing}

Reduction-aware admittance for the retained arc `from_no -> to_no`. Returns the series/parallel
equivalent π-tuple (oriented from->to) when the arc was aggregated by a network reduction, or
`nothing` when the arc is direct (the caller falls back to the branch's own
[`branch_admittance`](@ref)).
"""
function reduced_arc_admittance(nr::NetworkReductionData, from_no::Int, to_no::Int)
    eb = _reduced_arc_equivalent_branch(nr, (from_no, to_no))
    if isnothing(eb)
        return nothing
    end
    return _to_admittance(eb)
end

# Reorient an `EquivalentBranch` from<->to: series r/x are symmetric, the from/to shunts swap,
# and the phase shift negates.
function _reverse_equivalent_branch(eb::EquivalentBranch)
    @assert isone(get_equivalent_tap(eb)) "Cannot reorient a reduced arc with a non-unit tap ($(get_equivalent_tap(eb)))."
    return EquivalentBranch(
        get_equivalent_r(eb), get_equivalent_x(eb),
        get_equivalent_g_to(eb), get_equivalent_b_to(eb),
        get_equivalent_g_from(eb), get_equivalent_b_from(eb),
        get_equivalent_tap(eb), -get_equivalent_shift(eb),
    )
end

"""
    arc_equivalent_branch(nr::NetworkReductionData, arc::Tuple{Int, Int}) -> EquivalentBranch

Electrical parameters of the retained `arc`, resolved through whichever reduction map owns it:
a direct branch (including a three-winding transformer circuit on its star-point arc), a
parallel group, a series chain, or an added Ward-equivalent impedance. The result is always
oriented `from -> to` to match `arc`.

This is the accessor consumers should use instead of walking
`get_direct_branch_map`/`get_parallel_branch_map`/`get_series_branch_map` themselves — PNM owns
the reduction bookkeeping, so an arc's parameters resolve here.

Throws if `arc` is in no map.

Throws for a parallel group that mixes phase-shift angles with impedance angles — that group
needs more than one π branch. Use [`arc_equivalent_branches`](@ref) for the total accessor.
"""
function arc_equivalent_branch(nr::NetworkReductionData, arc::Tuple{Int, Int})
    entry, reversed = _resolve_arc_entry(nr, arc)
    equivalent = equivalent_branch(entry, nr)
    if reversed
        return _reverse_equivalent_branch(equivalent)
    end
    return equivalent
end

# Parallel/series equivalent for `arc`, oriented to match it. A group may be keyed by the
# opposite orientation to the one asked for, so probe both and reorient on a reverse hit.
# Returns `nothing` when the arc is not aggregated. `get` is a single probe per key, unlike
# haskey-then-index.
function _reduced_arc_equivalent_branch(nr::NetworkReductionData, arc::Tuple{Int, Int})
    haskey(get_direct_branch_map(nr), arc) && return nothing
    rev = (arc[2], arc[1])
    for map in (get_series_branch_map(nr), get_parallel_branch_map(nr))
        forward = get(map, arc, nothing)
        if !isnothing(forward)
            return get_equivalent_physical_branch_parameters(forward, nr)
        end
        reversed = get(map, rev, nothing)
        if !isnothing(reversed)
            return _reverse_equivalent_branch(
                get_equivalent_physical_branch_parameters(reversed, nr),
            )
        end
    end
    return nothing
end

# ── Three-winding transformer admittance ─────────────────────────────────────

"""
    three_winding_arcs(d::PSY.ThreeWindingTransformer) -> Vector{<:NamedTuple}

Decompose a `ThreeWindingTransformer` into its wye-model circuits via
[`ThreeWindingTransformerCircuit`](@ref), returning per-circuit data: a naming `suffix`, the
star-point `arc` (for reduction-aware bus mapping), the circuit `rating`, and the `circuit`
object itself (for [`branch_admittance`](@ref)).
"""
function three_winding_arcs(d::PSY.ThreeWindingTransformer)
    return [
        (
            suffix = "winding_$i",
            arc = PSY.get_arc(c),
            rating = get_equivalent_rating(ThreeWindingTransformerCircuit(d, c, i)),
            circuit = ThreeWindingTransformerCircuit(d, c, i),
        )
        for (i, c) in enumerate(PSY.get_circuits(d))
    ]
end

# ── π-model → Ybus 2x2 ───────────────────────────────────────────────────────

"""
    _pi_to_ybus(adm) -> (Y11, Y12, Y21, Y22)

Ybus 2x2 for a π-model tuple. The unit-tap case is split out as an optimisation, not for
correctness: a line's π-model is built with literal `tap`/`shift`, so inference constant-folds
the `tap == 1` test and the `exp` and three complex divisions vanish from the emitted code.
(Division by `1.0 + 0.0im` *is* bit-exact for finite values — only the sign of a zero can
differ, which no downstream comparison observes.)
"""
function _pi_to_ybus(
    adm::@NamedTuple{
        g::Float64, b::Float64,
        g_fr::Float64, b_fr::Float64,
        g_to::Float64, b_to::Float64,
        tap::Float64, shift::Float64,
    },
)
    Y_l = complex(adm.g, adm.b)
    y_fr = complex(adm.g_fr, adm.b_fr)
    y_to = complex(adm.g_to, adm.b_to)
    if isone(adm.tap) && iszero(adm.shift)
        return (Y_l + y_fr, -Y_l, -Y_l, Y_l + y_to)
    end
    tap = adm.tap * exp(adm.shift * 1im)
    return (
        Y_l / abs2(tap) + y_fr,
        -Y_l / conj(tap),
        -Y_l / tap,
        Y_l + y_to,
    )
end

# ── Branch flow limits ───────────────────────────────────────────────────────

"""
    branch_flow_limits(branch) -> NamedTuple

Directional flow limits in MVA (device units, `PSY.CU`): `(from_to, to_from)`. For symmetric
branches both fields equal the branch's [`get_equivalent_rating`](@ref); `MonitoredLine`
carries asymmetric limits and has its own method. Branches whose rating lives on a
transformer circuit — and reduction groups containing them — may carry `nothing` in both
fields when no rating is known; `Line`/`MonitoredLine` limits are always `Float64`.

A reduction aggregate answers with its equivalent rating in both directions, and throws when
any member carries asymmetric limits.
"""
function branch_flow_limits(b::PSY.ACTransmission)
    r = get_equivalent_rating(b)
    return (from_to = r, to_from = r)
end

function branch_flow_limits(b::PSY.MonitoredLine)
    fl = PSY.get_flow_limits(b, PSY.CU)
    return (from_to = fl.from_to, to_from = fl.to_from)
end

_has_asymmetric_flow_limits(::PSY.ACTransmission) = false

function _has_asymmetric_flow_limits(b::PSY.MonitoredLine)
    fl = PSY.get_flow_limits(b, PSY.CU)
    return fl.from_to != fl.to_from
end

_has_asymmetric_flow_limits(seg::AbstractReductionAggregate) =
    any(_has_asymmetric_flow_limits, seg)

# Aggregates subtype `PSY.ACTransmission`, so without this arm a reduction group takes the
# blanket method and reports its equivalent rating in both directions — right only while every
# member is symmetric, and wrong silently otherwise. Orienting an asymmetric member against the
# group's arc frame needs the `NetworkReductionData` (see `_subset_two_port`), which this
# accessor does not take, so such a group is rejected rather than guessed at.
function branch_flow_limits(seg::AbstractReductionAggregate)
    if _has_asymmetric_flow_limits(seg)
        error(
            "Reduction aggregate $(get_name(seg)) holds a member with asymmetric flow " *
            "limits, so its directional limits depend on each member's orientation in the " *
            "group's arc frame, which this accessor cannot resolve. Read the members' own " *
            "branch_flow_limits and orient them with get_arc_tuple(member, nr).",
        )
    end
    r = get_equivalent_rating(seg)
    return (from_to = r, to_from = r)
end
