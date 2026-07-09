"""
    ThreeWindingTransformerWinding <: PSY.ACTransmission

Internal object representing a single winding of a [`PSY.ThreeWindingTransformer`](@ref).
Do not export.

This structure decomposes a three-winding transformer into individual winding components
for network-matrix construction and analysis. The delta→star derivation (single home in
PNM) is performed once at construction: the three pairwise measured impedances are read on
a common base (system base, via `PSY.get_r_12(t, PSY.SU)` etc.), converted to the three
star-leg impedances by the standard identity, and the reactance floored away from zero.

# Fields
- `transformer::PSY.ThreeWindingTransformer`: The parent three-winding transformer
- `winding::PSY.TransformerWinding`: The parent's winding this object represents
- `winding_number::Int`: The winding number (1 = primary, 2 = secondary, 3 = tertiary)
- `r::Float64`: Derived star-leg resistance (pu, system base)
- `x::Float64`: Derived star-leg reactance (pu, system base), floored at
  [`STAR_LEG_REACTANCE_FLOOR`](@ref)

# Note
This is an internal object and should not be constructed directly by users or added to a
system.
"""
struct ThreeWindingTransformerWinding <: PSY.ACTransmission
    transformer::PSY.ThreeWindingTransformer
    winding::PSY.TransformerWinding
    winding_number::Int
    r::Float64
    x::Float64
end

# Standard delta→star identity on the common-base (SU) pairwise impedances; the star-leg
# reactance is floored (see `STAR_LEG_REACTANCE_FLOOR`) so a measured-zero leg does not blow
# up the admittance.
function _star_leg_impedance(t::PSY.ThreeWindingTransformer, winding_number::Int)
    z12 = complex(PSY.get_r_12(t, PSY.SU), PSY.get_x_12(t, PSY.SU))
    z23 = complex(PSY.get_r_23(t, PSY.SU), PSY.get_x_23(t, PSY.SU))
    z13 = complex(PSY.get_r_13(t, PSY.SU), PSY.get_x_13(t, PSY.SU))
    z = if winding_number == 1
        (z12 + z13 - z23) / 2
    elseif winding_number == 2
        (z12 + z23 - z13) / 2
    elseif winding_number == 3
        (z13 + z23 - z12) / 2
    else
        throw(ArgumentError("Invalid winding number: $winding_number"))
    end
    return (real(z), _floor_star_leg_reactance(imag(z), t, winding_number))
end

function _floor_star_leg_reactance(
    x::Float64,
    t::PSY.ThreeWindingTransformer,
    winding_number::Int,
)
    if isapprox(x, 0.0; atol = STAR_LEG_ZERO_REACTANCE_ATOL)
        # Constructed repeatedly on hot paths (Ybus assembly, reductions); @debug avoids
        # log spam while preserving traceability of the flooring event.
        @debug "Zero star-leg reactance in ThreeWindingTransformer $(PSY.get_name(t)) " *
               "winding $(winding_number); flooring to $(STAR_LEG_REACTANCE_FLOOR)."
        return STAR_LEG_REACTANCE_FLOOR
    end
    return x
end

"""
    ThreeWindingTransformerWinding(t::PSY.ThreeWindingTransformer, winding_number::Int)

Construct the wrapper for `winding_number` (1/2/3) of `t`, deriving and flooring the
star-leg impedance from the parent's pairwise data.
"""
function ThreeWindingTransformerWinding(
    t::PSY.ThreeWindingTransformer,
    winding_number::Int,
)
    (winding_number in 1:3) ||
        throw(ArgumentError("Invalid winding number: $winding_number"))
    r, x = _star_leg_impedance(t, winding_number)
    winding = PSY.get_windings(t)[winding_number]
    return ThreeWindingTransformerWinding(t, winding, winding_number, r, x)
end

# Lookup identity must be `{parent, winding_number}`, NOT full field-egal equality: `r`/`x`
# are derived at construction time from the parent's pairwise data (a build-time snapshot),
# so two wrappers for the same winding built before/after a pairwise-impedance mutation would
# otherwise compare unequal and hash differently, breaking Dict/Set-keyed lookups that
# reconstruct a fresh wrapper to query a map keyed by an earlier-built one. `transformer` is
# compared by identity (`===`).
function Base.:(==)(a::ThreeWindingTransformerWinding, b::ThreeWindingTransformerWinding)
    return a.transformer === b.transformer && a.winding_number == b.winding_number
end

function Base.hash(tw::ThreeWindingTransformerWinding, h::UInt)
    return hash(tw.winding_number, hash(objectid(tw.transformer), h))
end

get_transformer(tw::ThreeWindingTransformerWinding) = tw.transformer
get_winding_number(tw::ThreeWindingTransformerWinding) = tw.winding_number
# Lets callers key reduction maps by the parent transformer type.
get_transformer_type(tw::ThreeWindingTransformerWinding) = typeof(tw.transformer)

function get_name(three_wt_winding::ThreeWindingTransformerWinding)
    transformer = get_transformer(three_wt_winding)
    winding = get_winding_number(three_wt_winding)
    return PSY.get_name(transformer) * "_winding_$winding"
end

"""
    get_series_susceptance(segment::ThreeWindingTransformerWinding, units)

Series susceptance `imag(1 / (r + im*x))` of the star leg (system base). `units` is
accepted for interface symmetry; the stored star-leg impedance is always system base.

Method of PNM's own `get_series_susceptance` generic (see `BranchAdmittance.jl` for the
`ACTransmission`/`TwoWindingTransformer` methods) for the `ThreeWindingTransformerWinding`
wrapper type, so the whole codebase resolves `get_series_susceptance` calls to a single
generic-function family.

!!! note "Mixed susceptance conventions"
    This returns `imag(1/(r + im*x))` (r-aware, **negative-signed**), while `Line`
    susceptance is `+1/x` (positive, r-free). Both conventions are consumed together in the
    reduction sums (`BranchesSeries`/`BranchesParallel`, `virtual_factor_helpers`,
    `network_modification`). They are physically inconsistent, so changing the sign/model on
    either side requires revisiting both; the Ybus equivalence gate arbitrates any change. Do
    not change it unilaterally.
"""
function get_series_susceptance(
    segment::ThreeWindingTransformerWinding,
    ::IS.AbstractUnitSystem,
)
    return imag(1 / (segment.r + segment.x * im))
end

"""
    get_equivalent_r(tw::ThreeWindingTransformerWinding)

Derived star-leg resistance (pu, system base) of this winding.
"""
get_equivalent_r(tw::ThreeWindingTransformerWinding) = tw.r

"""
    get_equivalent_x(tw::ThreeWindingTransformerWinding)

Derived star-leg reactance (pu, system base) of this winding, floored at
[`STAR_LEG_REACTANCE_FLOOR`](@ref).
"""
get_equivalent_x(tw::ThreeWindingTransformerWinding) = tw.x

"""
    get_equivalent_b(tw::ThreeWindingTransformerWinding)

Shunt susceptance split. Only the primary winding (winding 1) carries the parent's
magnetizing shunt (its imaginary part); secondary/tertiary carry none.
"""
function get_equivalent_b(tw::ThreeWindingTransformerWinding)
    if tw.winding_number == 1
        return (from = imag(PSY.get_magnetizing_shunt(tw.transformer, PSY.SU)), to = 0.0)
    else
        return (from = 0.0, to = 0.0)
    end
end

"""
    get_equivalent_rating(tw::ThreeWindingTransformerWinding)

The winding's own rating (MVA, device base). May be `nothing` when unset, mirroring how a
[`PSY.Line`](@ref)'s rating is surfaced; there is no parent-level rating to fall back to.
"""
get_equivalent_rating(tw::ThreeWindingTransformerWinding) =
    PSY.get_rating(tw.winding, PSY.DU)

"""
    get_equivalent_emergency_rating(tw::ThreeWindingTransformerWinding)

Emergency rating for this winding. No separate `rating_b` is modeled per winding, so this
mirrors [`get_equivalent_rating`](@ref).
"""
get_equivalent_emergency_rating(tw::ThreeWindingTransformerWinding) =
    get_equivalent_rating(tw)

"""
    get_equivalent_available(tw::ThreeWindingTransformerWinding)

Per-winding availability, the single source of truth for availability.
"""
get_equivalent_available(tw::ThreeWindingTransformerWinding) =
    PSY.get_available(tw.winding)

PSY.get_available(tw::ThreeWindingTransformerWinding) = get_equivalent_available(tw)

function get_arc_tuple(tr::ThreeWindingTransformerWinding)
    arc = PSY.get_arc(tr.winding)
    return (
        PSY.get_number(PSY.get_from(arc)),
        PSY.get_number(PSY.get_to(arc)),
    )
end

"""
    get_equivalent_tap(tw::ThreeWindingTransformerWinding)

The winding's tap (turns ratio). Defaults to `1.0` for windings with no tap.
"""
get_equivalent_tap(tw::ThreeWindingTransformerWinding) = PSY.get_tap(tw.winding)

"""
    get_equivalent_α(tw::ThreeWindingTransformerWinding)

The winding's phase-shift angle (radians). `0.0` for non-shifting windings.
"""
get_equivalent_α(tw::ThreeWindingTransformerWinding) = PSY.get_α(tw.winding)

function add_to_map(device::ThreeWindingTransformerWinding, filters::Dict)
    isempty(filters) && return true
    return add_to_map(get_transformer(device), filters)
end

is_a_reduction(::ThreeWindingTransformerWinding) = true

function has_time_series(
    device::ThreeWindingTransformerWinding,
    ts_type::Type{T},
    ts_name::String,
) where {
    T <: PSY.TimeSeriesData,
}
    return PSY.has_time_series(get_transformer(device), ts_type, ts_name)
end

function get_device_with_time_series(
    device::ThreeWindingTransformerWinding,
    ts_type::Type{T},
    ts_name::String,
) where {
    T <: PSY.TimeSeriesData,
}
    transformer = get_transformer(device)
    if PSY.has_time_series(transformer, ts_type, ts_name)
        return transformer
    end
    return nothing
end
