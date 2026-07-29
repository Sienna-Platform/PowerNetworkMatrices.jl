# ── Branch π-model admittance ────────────────────────────────────────────────
# Compute a branch's π-model admittance as a
# `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` NamedTuple following the PowerModels
# convention: `g + im*b == 1/(r + im*x)` is the series admittance, `*_fr`/`*_to` are the
# from/to shunts, and `tap`/`shift` are the transformer ratio / phase shift (radians).

# Split a two-winding transformer's complex magnetizing shunt onto the π-model from/to shunt
# slots per its `PSY.TwoWindingTransformerShuntLocation`: PRIMARY places the full value on
# the from side, SECONDARY on the to side, SPLIT the full value on both sides.
function _magnetizing_shunt_split(y_shunt::Complex, location)
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
function _three_winding_shunt_split(y_shunt::Complex, location, winding_number::Int)
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

"""
    get_series_susceptance(b::PSY.ACTransmission, units::IS.AbstractUnitSystem)

Series susceptance `1/x` of an [`ACTransmission`](@ref PSY.ACTransmission) branch, from
the stored series reactance alone. [`PSY.TwoWindingTransformer`](@ref) has a more specific
method (below) that additionally divides by the winding tap ratio
(`PSY.get_tap(PSY.get_circuit(t))`). This is a deliberate asymmetry: only the susceptance
form is tap-divided; Ybus/PTDF/LODF assembly needs the tap-divided value, while callers that
need the untapped complex admittance should build it directly from `PSY.get_r`/`PSY.get_x`.
"""
get_series_susceptance(b::PSY.ACTransmission, units::IS.AbstractUnitSystem) =
    1 / PSY.get_x(b, units)

"""
    get_series_susceptance(t::PSY.TwoWindingTransformer, units::IS.AbstractUnitSystem)

Series susceptance of a [`PSY.TwoWindingTransformer`](@ref): the generic `ACTransmission`
value (`1/x`) divided by the winding tap ratio `PSY.get_tap(PSY.get_circuit(t))`. A
fixed-ratio transformer has `tap = 1.0`, so this is a no-op for it and matches the plain
`ACTransmission` value.
"""
get_series_susceptance(t::PSY.TwoWindingTransformer, units::IS.AbstractUnitSystem) =
    get_series_susceptance(PSY.get_circuit(t), units)

"""
    get_series_susceptance(c::PSY.TransformerCircuit, units::IS.AbstractUnitSystem)

Tap-divided series susceptance of a transformer circuit. Both arities route here: the circuit
owns `x` and `tap`, so the convention is stated once.
"""
get_series_susceptance(c::PSY.TransformerCircuit, units::IS.AbstractUnitSystem) =
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

function get_series_phase_shift(c::PSY.TransformerCircuit)
    return PSY.get_α(c)
end

function get_series_phase_shift(t::PSY.TwoWindingTransformer)
    return get_series_phase_shift(PSY.get_circuit(t))
end

"""
    equivalent_branch(b; min_x_eps) -> EquivalentBranch

The π-model of a single branch in **impedance** form: `(r, x, g_from, b_from, g_to, b_to,
tap, shift)`. This is PNM's single source of truth for branch electrical parameters — both
[`branch_admittance`](@ref) (admittance form) and [`ybus_branch_entries`](@ref) (Ybus 2×2)
are derived from it, and [`arc_equivalent_branch`](@ref) resolves any arc to one.

Methods exist for lines, `GenericArcImpedance` Ward equivalents, and transformer circuits of
either arity — a transformer's series data lives on its `PSY.TransformerCircuit`, so 2W and
3W differ only in which shunt-placement enum applies.

The from/to shunts carry the real `PSY.get_g` conductance. A caller wanting PowerModels'
`get_branch_to_pm` convention (which drops `g`) must zero it at that boundary.

`min_x_eps` substitutes for `x` when `r == x == 0`, matching Ybus assembly.
"""
# Both π shunts in one pass: the explicit-units getters convert the whole from/to pair, so
# reading each once and taking both fields statically halves the unit-conversion work.
# ComplexF64, not YBUS_ELTYPE — the π layer is Float64; narrowing happens at Ybus storage.
function _get_shunts(br::PSY.ACTransmission)
    g = PSY.get_g(br, PSY.SU)
    b = PSY.get_b(br, PSY.SU)
    return (complex(g.from, b.from), complex(g.to, b.to))
end

_get_shunts(::PSY.DiscreteControlledACBranch) = (zero(ComplexF64), zero(ComplexF64))

# Kept out of line so the `@warn` body does not blow `equivalent_branch`'s inlining budget:
# without this, the whole π chain stops inlining into `ybus_branch_entries` and the unit-tap
# test in `_pi_to_ybus` can no longer constant-fold for lines.
@noinline function _warn_zero_impedance(b, min_x_eps)
    @warn "Branch $(PSY.get_name(b)) has r=0.0 and x=0.0; substituting x=$(min_x_eps) to avoid division by zero. This branch will be reduced by ZeroImpedanceBranchReduction unless its endpoints are irreducible."
    return
end

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
# device base (`PSY.DU`), which returns them unchanged — a detached component cannot resolve
# the system base power. It carries no shunts.
function equivalent_branch(
    b::PSY.GenericArcImpedance;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    return EquivalentBranch(
        PSY.get_r(b, PSY.DU), PSY.get_x(b, PSY.DU),
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
    return _circuit_equivalent_branch(PSY.get_circuit(b), sh)
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
    return _circuit_equivalent_branch(w.circuit, sh)
end

# The whole of "a transformer is a circuit": the circuit owns r/x/tap/α for both arities, so
# 2W and 3W differ only in which shunt-split rule produced `sh`.
function _circuit_equivalent_branch(circuit::PSY.TransformerCircuit, sh)
    return EquivalentBranch(
        PSY.get_r(circuit, PSY.SU), PSY.get_x(circuit, PSY.SU),
        sh.g_fr, sh.b_fr, sh.g_to, sh.b_to,
        PSY.get_tap(circuit), PSY.get_α(circuit),
    )
end

"""
    branch_admittance(b; min_x_eps) -> NamedTuple

π-model admittance `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` for any single branch, where
`g + im*b == 1 / (r + im*x)` is the series admittance. The admittance-form view of
[`equivalent_branch`](@ref); see it for the shunt and unit conventions.
"""
function branch_admittance(b; min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON)
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

"""
    branch_admittance(segment, nr::NetworkReductionData) -> NamedTuple

π-model admittance for a reduction-aggregated arc (a `BranchesSeries` chain or
`BranchesParallel` group), built from PNM's reduction-aware equivalent physical branch
parameters. Series/parallel equivalents of lines carry `tap == 1`.
"""
function branch_admittance(segment, nr::NetworkReductionData)
    return _to_admittance(get_equivalent_physical_branch_parameters(segment, nr))
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
[`get_direct_branch_map`](@ref)/[`get_parallel_branch_map`](@ref)/[`get_series_branch_map`](@ref)
themselves — PNM owns the reduction bookkeeping, so an arc's parameters resolve here.

Throws if `arc` is in no map.

Throws for a parallel group that mixes phase-shift angles with impedance angles — that group
needs more than one π branch. Use [`arc_equivalent_branches`](@ref) for the total accessor.
"""
# Single branches (direct and added-Ward alike) carry their own equivalent; aggregates go through
# the reduction-aware recovery, which throws when no single π exists.
_single_arc_equivalent(br::PSY.ACTransmission, ::NetworkReductionData) =
    equivalent_branch(br)
_single_arc_equivalent(group::AbstractBranchesParallel, nr::NetworkReductionData) =
    get_equivalent_physical_branch_parameters(group, nr)
_single_arc_equivalent(group::BranchesSeries, nr::NetworkReductionData) =
    get_equivalent_physical_branch_parameters(group, nr)

function arc_equivalent_branch(nr::NetworkReductionData, arc::Tuple{Int, Int})
    entry, reversed = _resolve_arc_entry(nr, arc)
    equivalent = _single_arc_equivalent(entry, nr)
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
correctness: with `equivalent_branch` inlined the `tap == 1` test constant-folds for lines, so
the `exp` and three complex divisions vanish from the emitted code. (Division by `1.0 + 0.0im`
*is* bit-exact for finite values — only the sign of a zero can differ, which no downstream
comparison observes.)
"""
function _pi_to_ybus(adm)
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

Directional flow limits in MVA (device units, `PSY.DU`): `(from_to, to_from)`. For symmetric
branches both fields equal the branch's [`get_equivalent_rating`](@ref); `MonitoredLine`
carries asymmetric limits and has its own method. Branches whose rating lives on a
transformer circuit — and reduction groups containing them — may carry `nothing` in both
fields when no rating is known; `Line`/`MonitoredLine` limits are always `Float64`.
"""
function branch_flow_limits(b::PSY.ACTransmission)
    r = get_equivalent_rating(b)
    return (from_to = r, to_from = r)
end

function branch_flow_limits(b::PSY.MonitoredLine)
    fl = PSY.get_flow_limits(b, PSY.DU)
    return (from_to = fl.from_to, to_from = fl.to_from)
end
