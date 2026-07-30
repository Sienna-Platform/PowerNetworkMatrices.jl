"""
    ThreeWindingTransformerCircuit <: PSY.ACTransmission

Internal object representing a single circuit of a `PSY.ThreeWindingTransformer`.
Do not export.

This structure decomposes a three-winding transformer into individual circuit components
for network-matrix construction and analysis. The star-leg series impedance is stored on
the circuit (`PSY.get_r`/`PSY.get_x`); this wrapper reads it directly and exposes it on the
system base through [`get_equivalent_r`](@ref)/[`get_equivalent_x`](@ref).

# Fields
- `transformer::PSY.ThreeWindingTransformer`: The parent three-winding transformer
- `circuit::PSY.TransformerCircuit`: The parent's circuit this object represents
- `winding_number::Int`: The PSSE winding position (1 = primary, 2 = secondary, 3 = tertiary)

# Note
This is an internal object and should not be constructed directly by users or added to a
system.
"""
struct ThreeWindingTransformerCircuit <: PSY.ACTransmission
    transformer::PSY.ThreeWindingTransformer
    circuit::PSY.TransformerCircuit
    winding_number::Int
end

"""
    ThreeWindingTransformerCircuit(t::PSY.ThreeWindingTransformer, winding_number::Int)

Construct the wrapper for `winding_number` (1/2/3) of `t`.
"""
function ThreeWindingTransformerCircuit(
    t::PSY.ThreeWindingTransformer,
    winding_number::Int,
)
    (winding_number in 1:3) ||
        throw(ArgumentError("Invalid winding number: $winding_number"))
    circuit = PSY.get_circuits(t)[winding_number]
    return ThreeWindingTransformerCircuit(t, circuit, winding_number)
end

# Lookup identity is `{parent, winding_number}`, NOT full field-egal equality: the circuit
# object is compared by the parent transformer's identity and the winding number, so a fresh
# wrapper reconstructed to query a Dict/Set keyed by an earlier-built one resolves the same
# entry. `transformer` is compared by identity (`===`).
function Base.:(==)(a::ThreeWindingTransformerCircuit, b::ThreeWindingTransformerCircuit)
    return a.transformer === b.transformer && a.winding_number == b.winding_number
end

function Base.hash(tw::ThreeWindingTransformerCircuit, h::UInt)
    return hash(tw.winding_number, hash(objectid(tw.transformer), h))
end

get_transformer(tw::ThreeWindingTransformerCircuit) = tw.transformer
get_winding_number(tw::ThreeWindingTransformerCircuit) = tw.winding_number
# Extends the PSY getter rather than adding a PNM-local one: `Base.show(::Component)` derives
# `get_$field` and calls it whenever PSY exposes that name, so a wrapper with a `circuit` field
# and no `PSY.get_circuit` method throws a MethodError on display.
PSY.get_circuit(tw::ThreeWindingTransformerCircuit) = tw.circuit
# Lets callers key reduction maps by the parent transformer type.
get_transformer_type(tw::ThreeWindingTransformerCircuit) = typeof(tw.transformer)

function get_name(three_wt_circuit::ThreeWindingTransformerCircuit)
    transformer = get_transformer(three_wt_circuit)
    winding = get_winding_number(three_wt_circuit)
    return PSY.get_name(transformer) * "_winding_$winding"
end

"""
    get_series_susceptance(segment::ThreeWindingTransformerCircuit, units)

Series susceptance of the star leg for the DC/reduction model: `(1/x)/tap`, computed from
the circuit's star-leg reactance alone (r-free) and divided by the circuit tap ratio — the
same convention as the `PSY.TwoWindingTransformer` method in `BranchAdmittance.jl`,
and reactance-additive like the generic `ACTransmission` `1/x` method, so all branch kinds
combine consistently in the reduction sums (`BranchesSeries`/`BranchesParallel`,
`virtual_factor_helpers`, `network_modification`) and in `BA_Matrix` assembly. The sign
follows `x`: a star-leg reactance can legitimately be negative, giving a negative
susceptance. `units` selects the reactance base, as in the sibling methods — unlike
[`get_equivalent_x`](@ref), which is pinned to the system base because reduction
aggregation must sum on a common base.
"""
get_series_susceptance(
    segment::ThreeWindingTransformerCircuit,
    units::IS.AbstractUnitSystem,
) = get_series_susceptance(segment.circuit, units)

function get_series_phase_shift(tw::ThreeWindingTransformerCircuit)
    return get_series_phase_shift(tw.circuit)
end

"""
    get_equivalent_r(tw::ThreeWindingTransformerCircuit)

Star-leg resistance (pu, system base) of this circuit.
"""
get_equivalent_r(tw::ThreeWindingTransformerCircuit) = PSY.get_r(tw.circuit, PSY.SU)

"""
    get_equivalent_x(tw::ThreeWindingTransformerCircuit)

Star-leg reactance (pu, system base) of this circuit.
"""
get_equivalent_x(tw::ThreeWindingTransformerCircuit) = PSY.get_x(tw.circuit, PSY.SU)

"""
    get_equivalent_rating(tw::ThreeWindingTransformerCircuit)

The circuit's own rating (MVA, device base). May be `nothing` when unset, mirroring how a
`PSY.Line`'s rating is surfaced; there is no parent-level rating to fall back to.
"""
get_equivalent_rating(tw::ThreeWindingTransformerCircuit) =
    PSY.get_rating(tw.circuit, PSY.DU)

"""
    get_equivalent_emergency_rating(tw::ThreeWindingTransformerCircuit)

Emergency rating for this circuit. No separate `rating_b` is modeled per circuit, so this
mirrors [`get_equivalent_rating`](@ref).
"""
get_equivalent_emergency_rating(tw::ThreeWindingTransformerCircuit) =
    get_equivalent_rating(tw)

"""
    get_equivalent_available(tw::ThreeWindingTransformerCircuit)

Per-circuit availability, the single source of truth for availability.
"""
get_equivalent_available(tw::ThreeWindingTransformerCircuit) =
    PSY.get_available(tw.circuit)

PSY.get_available(tw::ThreeWindingTransformerCircuit) = get_equivalent_available(tw)

# Delegating these to the circuit lets the generic `PSY.ACTransmission` methods for
# `get_arc_tuple` (common.jl) and the phase-shift guards cover the wrapper unchanged.
PSY.get_arc(tw::ThreeWindingTransformerCircuit) = PSY.get_arc(tw.circuit)
PSY.is_phase_shifting(tw::ThreeWindingTransformerCircuit) =
    PSY.is_phase_shifting(tw.circuit)

"""
    get_equivalent_tap(tw::ThreeWindingTransformerCircuit)

The circuit's tap (turns ratio). Defaults to `1.0` for circuits with no tap.
"""
get_equivalent_tap(tw::ThreeWindingTransformerCircuit) = PSY.get_tap(tw.circuit)

function add_to_map(device::ThreeWindingTransformerCircuit, filters::Dict)
    isempty(filters) && return true
    return add_to_map(get_transformer(device), filters)
end

function has_time_series(
    device::ThreeWindingTransformerCircuit,
    ts_type::Type{T},
    ts_name::String,
) where {
    T <: PSY.TimeSeriesData,
}
    return PSY.has_time_series(get_transformer(device), ts_type, ts_name)
end

function get_device_with_time_series(
    device::ThreeWindingTransformerCircuit,
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
