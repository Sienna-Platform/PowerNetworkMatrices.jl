# ── Branch π-model admittance ────────────────────────────────────────────────
# Compute a branch's π-model admittance as a
# `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` NamedTuple following the PowerModels
# convention: `g + im*b == 1/(r + im*x)` is the series admittance, `*_fr`/`*_to` are the
# from/to shunts, and `tap`/`shift` are the transformer ratio / phase shift (radians).

# Transformer ratio / phase-shift accessors. Default to a unit, shift-free branch; a
# two-winding transformer reads tap/shift off its single winding (tap defaults to 1, shift
# to 0 for non-shifting windings).
_branch_tap(::PSY.ACTransmission) = 1.0
_branch_tap(b::PSY.TwoWindingTransformer) = PSY.get_tap(PSY.get_circuit(b))
_branch_shift(::PSY.ACTransmission) = 0.0
_branch_shift(b::PSY.TwoWindingTransformer) = PSY.get_α(PSY.get_circuit(b))

# Split a two-winding transformer's complex magnetizing shunt onto the π-model from/to shunt
# slots per its `PSY.TwoWindingTransformerShuntLocation`: PRIMARY places the full value on
# the from side, SECONDARY on the to side, SPLIT the full value on both sides.
function _magnetizing_shunt_split(y_shunt::Complex, location)
    g, b = real(y_shunt), imag(y_shunt)
    if location == PSY.TwoWindingTransformerShuntLocation.SECONDARY
        return (g_fr = 0.0, b_fr = 0.0, g_to = g, b_to = b)
    elseif location == PSY.TwoWindingTransformerShuntLocation.SPLIT
        return (g_fr = g, b_fr = b, g_to = g, b_to = b)
    else
        return (g_fr = g, b_fr = b, g_to = 0.0, b_to = 0.0)
    end
end

# Split a three-winding transformer's complex magnetizing shunt for one circuit's π-model.
# The shunt lives on the parent transformer and lands on circuit 1 only, per its
# `PSY.ThreeWindingTransformerShuntLocation`: PRIMARY on the terminal (from) side, STAR on
# the star-node (to) side. Circuits 2 and 3 carry no shunt.
function _three_winding_shunt_split(y_shunt::Complex, location, winding_number::Int)
    winding_number == 1 || return (g_fr = 0.0, b_fr = 0.0, g_to = 0.0, b_to = 0.0)
    g, b = real(y_shunt), imag(y_shunt)
    if location == PSY.ThreeWindingTransformerShuntLocation.STAR
        return (g_fr = 0.0, b_fr = 0.0, g_to = g, b_to = b)
    else
        return (g_fr = g, b_fr = b, g_to = 0.0, b_to = 0.0)
    end
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
    (1 / PSY.get_x(t, units)) / PSY.get_tap(PSY.get_circuit(t))

"""
    branch_admittance(b::PSY.ACTransmission) -> NamedTuple

π-model admittance `(g, b, g_fr, b_fr, g_to, b_to, tap, shift)` for a line or symmetric
branch, where `g + im*b == 1 / (r + im*x)` is the series admittance and the line charging
susceptance is split between the from/to shunts.
"""
function branch_admittance(b::PSY.ACTransmission)
    ys = 1.0 / (PSY.get_r(b, PSY.SU) + PSY.get_x(b, PSY.SU) * im)
    b_sh = PSY.get_b(b, PSY.SU)
    return (
        g = real(ys), b = imag(ys),
        g_fr = 0.0, b_fr = b_sh.from, g_to = 0.0, b_to = b_sh.to,
        tap = 1.0, shift = 0.0,
    )
end

"""
    branch_admittance(b::PSY.TwoWindingTransformer) -> NamedTuple

π-model admittance for a two-winding transformer. The magnetizing shunt is allocated per the
transformer's [`PSY.TwoWindingTransformerShuntLocation`](@ref); `tap`/`shift` carry the
transformer ratio and phase shift.
"""
function branch_admittance(
    b::PSY.TwoWindingTransformer,
)
    ys = 1.0 / (PSY.get_r(b, PSY.SU) + PSY.get_x(b, PSY.SU) * im)
    sh = _magnetizing_shunt_split(
        PSY.get_magnetizing_shunt(b, PSY.SU),
        PSY.get_shunt_location(b),
    )
    return (
        g = real(ys), b = imag(ys),
        g_fr = sh.g_fr, b_fr = sh.b_fr, g_to = sh.g_to, b_to = sh.b_to,
        tap = _branch_tap(b), shift = _branch_shift(b),
    )
end

"""
    branch_admittance(segment, nr::NetworkReductionData) -> NamedTuple

π-model admittance for a reduction-aggregated arc (a `BranchesSeries` chain or
`BranchesParallel` group), built from PNM's reduction-aware equivalent physical branch
parameters. Series/parallel equivalents of lines carry `tap == 1`.
"""
function branch_admittance(segment, nr::NetworkReductionData)
    eb = get_equivalent_physical_branch_parameters(segment, nr)
    ys = 1.0 / (get_equivalent_r(eb) + get_equivalent_x(eb) * im)
    return (
        g = real(ys), b = imag(ys),
        g_fr = get_equivalent_g_from(eb), b_fr = get_equivalent_b_from(eb),
        g_to = get_equivalent_g_to(eb), b_to = get_equivalent_b_to(eb),
        tap = get_equivalent_tap(eb), shift = get_equivalent_shift(eb),
    )
end

# Flip a π-admittance tuple to the opposite orientation (from<->to). Reduced equivalents may
# be keyed by an arc whose orientation is reversed vs. the surviving branch's retained
# from->to; reorient so coefficients match (from_bus, to_bus). g/b are symmetric; from/to
# shunts swap; phase shift negates. Reduced line equivalents have tap == 1.
function _reverse_admittance(adm)
    @assert adm.tap == 1.0 "Cannot reorient a reduced arc with a non-unit tap ($(adm.tap))."
    return (
        g = adm.g,
        b = adm.b,
        g_fr = adm.g_to,
        b_fr = adm.b_to,
        g_to = adm.g_fr,
        b_to = adm.b_fr,
        tap = adm.tap,
        shift = -adm.shift,
    )
end

"""
    reduced_arc_admittance(nr::NetworkReductionData, from_no::Int, to_no::Int) -> Union{NamedTuple, Nothing}

Reduction-aware admittance for the retained arc `from_no -> to_no`. Returns the series/parallel
equivalent π-tuple (oriented from->to) when the arc was aggregated by a network reduction, or
`nothing` when the arc is direct (the caller falls back to the branch's own
[`branch_admittance`](@ref)).
"""
function reduced_arc_admittance(nr::NetworkReductionData, from_no::Int, to_no::Int)
    series_map = get_series_branch_map(nr)
    parallel_map = get_parallel_branch_map(nr)
    arc = (from_no, to_no)
    rev = (to_no, from_no)
    if haskey(series_map, arc)
        return branch_admittance(series_map[arc], nr)
    elseif haskey(series_map, rev)
        return _reverse_admittance(branch_admittance(series_map[rev], nr))
    elseif haskey(parallel_map, arc)
        return branch_admittance(parallel_map[arc], nr)
    elseif haskey(parallel_map, rev)
        return _reverse_admittance(branch_admittance(parallel_map[rev], nr))
    end
    return nothing
end

# ── Three-winding transformer admittance ─────────────────────────────────────

"""
    three_winding_arcs(d::PSY.ThreeWindingTransformer) -> Vector{<:NamedTuple}

Decompose a `ThreeWindingTransformer` into its three wye-model circuits via
[`ThreeWindingTransformerCircuit`](@ref), returning per-circuit data: a naming `suffix`, the
star-point `arc` (for reduction-aware bus mapping), the circuit `rating`, and the `circuit`
object itself (for [`winding_admittance`](@ref)).
"""
function three_winding_arcs(d::PSY.ThreeWindingTransformer)
    star_arcs = PSY.get_arc.(PSY.get_circuits(d))
    out = NamedTuple[]
    for i in 1:3
        w = ThreeWindingTransformerCircuit(d, i)
        push!(
            out,
            (
                suffix = "winding_$i",
                arc = star_arcs[i],
                rating = get_equivalent_rating(w),
                circuit = w,
            ),
        )
    end
    return out
end

# Only the circuit tap matters for the π-model here (phase shift is not carried by
# `winding_admittance`); a circuit with no tap reports 1.0.
_winding_tap(w::ThreeWindingTransformerCircuit) = get_equivalent_tap(w)

"""
    winding_admittance(w::ThreeWindingTransformerCircuit) -> NamedTuple

Per-circuit π-model admittance `(g, b, g_fr, b_fr, g_to, b_to, tap)` for one circuit of a
three-winding transformer (shunt split from/to, no phase shift). The parent transformer's
magnetizing shunt is placed per its [`PSY.ThreeWindingTransformerShuntLocation`](@ref) and
lands on circuit 1 only.
"""
function winding_admittance(w::ThreeWindingTransformerCircuit)
    ys = 1.0 / (get_equivalent_r(w) + get_equivalent_x(w) * im)
    sh = _three_winding_shunt_split(
        PSY.get_magnetizing_shunt(get_transformer(w), PSY.SU),
        PSY.get_shunt_location(get_transformer(w)),
        get_winding_number(w),
    )
    return (
        g = real(ys), b = imag(ys),
        g_fr = sh.g_fr, b_fr = sh.b_fr, g_to = sh.g_to, b_to = sh.b_to,
        tap = _winding_tap(w),
    )
end

# ── Branch flow limits ───────────────────────────────────────────────────────

"""
    branch_flow_limits(branch) -> NamedTuple

Directional flow limits in MVA (device units, `PSY.DU`): `(from_to, to_from)`. For symmetric
branches both fields equal the branch rating; `MonitoredLine` carries asymmetric limits.
Branches whose rating lives on a transformer winding (`TwoWindingTransformer`,
`ThreeWindingTransformerCircuit`) — and reduction groups containing them — may carry
`nothing` in both fields when no rating is known; `Line`/`MonitoredLine` limits are always
`Float64`.
"""
function branch_flow_limits end

function branch_flow_limits(b::PSY.MonitoredLine)
    fl = PSY.get_flow_limits(b, PSY.DU)
    return (from_to = fl.from_to, to_from = fl.to_from)
end

function branch_flow_limits(b::PSY.Line)
    r = PSY.get_rating(b, PSY.DU)
    return (from_to = r, to_from = r)
end

# A `TwoWindingTransformer` has no parent rating; the rating lives on its single winding.
function branch_flow_limits(b::PSY.TwoWindingTransformer)
    r = PSY.get_rating(PSY.get_circuit(b), PSY.DU)
    return (from_to = r, to_from = r)
end

function branch_flow_limits(b::BranchesParallel)
    r = get_equivalent_rating(b)
    return (from_to = r, to_from = r)
end

function branch_flow_limits(b::BranchesSeries)
    r = get_equivalent_rating(b)
    return (from_to = r, to_from = r)
end

function branch_flow_limits(w::ThreeWindingTransformerCircuit)
    r = get_equivalent_rating(w)
    return (from_to = r, to_from = r)
end
