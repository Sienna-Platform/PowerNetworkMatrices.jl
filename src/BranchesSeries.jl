mutable struct BranchesSeries <: AbstractReductionAggregate
    branches::Vector{PSY.ACTransmission}
    segment_orientations::Vector{Symbol}
    # The chain's endpoints in original bus numbers, remapped with `nr` on read. A chain can be
    # a member of a parallel group, where orientation is resolved against the group's frame.
    arc_key::Tuple{Int, Int}
    equivalent_ybus::CACHED_TWO_PORT
    equivalent_ybus_populated::Bool
end

# More than one type means a chain mixing branch types.
_has_mixed_types(bs::BranchesSeries) = !allequal(typeof, bs.branches)

BranchesSeries(arc_key::Tuple{Int, Int}) = BranchesSeries(
    Vector{PSY.ACTransmission}(),
    Vector{Symbol}(),
    arc_key,
    EMPTY_TWO_PORT,
    false,
)

function add_branch!(bs::BranchesSeries, branch::PSY.ACTransmission, orientation)
    invalidate_equivalent_ybus!(bs)
    push!(bs.segment_orientations, orientation)
    push!(bs.branches, branch)
    return
end

Base.iterate(bs::BranchesSeries, state...) = iterate(bs.branches, state...)

Base.length(bs::BranchesSeries) = length(bs.branches)

Base.eltype(::Type{BranchesSeries}) = PSY.ACTransmission

# Chain segments can themselves be parallel groups, so this recurses through
# `_is_phase_shifting(::AbstractBranchesParallel)` (BranchesParallel.jl).
function _is_phase_shifting(bs::BranchesSeries)
    return any(_is_phase_shifting, bs)
end

get_arc_key(bs::BranchesSeries) = bs.arc_key

"""
Per-segment orientation, in the chain's own iteration order, relative to its `arc_key`:
`:FromTo` when the segment's arc runs along the chain's traversal direction, `:ToFrom` when
it runs against it.

Recorded by `add_branch!` while `_build_chain_segments!` walks the chain from `arc_key[1]` to
`arc_key[2]`, so the vector is only meaningful in that frame. A caller holding an equivalent
arc from elsewhere should use the two-argument method, which checks the frame.
"""
get_segment_orientations(bs::BranchesSeries) = bs.segment_orientations

function _reverse_orientation(orientation::Symbol)
    if orientation === :FromTo
        return :ToFrom
    elseif orientation === :ToFrom
        return :FromTo
    end
    return error(
        "Unknown segment orientation $orientation; expected :FromTo or :ToFrom.",
    )
end

"""
Per-segment orientation of `bs` expressed relative to `equivalent_arc`, in the chain's own
iteration order.

`equivalent_arc` may be the chain's `arc_key` or its reverse. The reverse is a routine
request, not an error: `DegreeTwoReduction` groups sibling chains that resolve to the same
*unordered* endpoint pair into one `BranchesParallel` framed on the seed chain's key, so a
sibling legitimately keeps the opposite key while consumers reach it through the group and
hold only the group's frame. Reframing is well defined — traversing from the other endpoint
flips every segment's relation to the traversal, so each orientation negates while the
segment order is preserved, which is what callers zipping this against the chain's members
require. This mirrors `_subset_two_port`, which transposes an anti-frame member rather than
refusing it.

Returns a fresh vector; the one-argument method exposes the stored field and must be treated
as read-only.
"""
function get_segment_orientations(bs::BranchesSeries, equivalent_arc::Tuple{Int, Int})
    key = get_arc_key(bs)
    orientations = get_segment_orientations(bs)
    equivalent_arc == key && return copy(orientations)
    if equivalent_arc == reverse(key)
        return [_reverse_orientation(o) for o in orientations]
    end
    return error(
        "Chain orientations are recorded against arc $key, but were requested against " *
        "$equivalent_arc, which is neither that arc nor its reverse.",
    )
end

"""
A chain's name is its arc, spelled `series_<from>_<to>`. See the `AbstractBranchesParallel`
method (BranchesParallel.jl) for why the aggregate spells its own key and the catalog spells
the indexed one.

A nested chain keeps its own frame, so two siblings in one group can name themselves from
opposite endpoint orders -- they are not indexed, and a nested chain's `arc_key` is a traversal
frame rather than an identity.
"""
get_name(bs::BranchesSeries) = "series_$(bs.arc_key[1])_$(bs.arc_key[2])"

# Series segments add impedance. Reading a leaf's `tap * x` directly lets a zero-impedance
# segment contribute exactly 0.0, with no transient `Inf` for the sum to absorb.
_series_reactance(b::PSY.ACTransmission, units) =
    PSY.get_x(b, units)
_series_reactance(t::PSY.TwoWindingTransformer, units) =
    _series_reactance(PSY.get_circuit(t), units)
_series_reactance(w::ThreeWindingTransformerCircuit, units) =
    _series_reactance(w.circuit, units)
_series_reactance(c::PSY.TransformerCircuit, units) =
    PSY.get_x(c, units) * PSY.get_tap(c)
# A parallel group has no single reactance, so invert its susceptance sum; an all-zero
# group gives `Inf` there and `inv(Inf) = 0.0` is the correct contribution.
_series_reactance(seg::AbstractReductionAggregate, units) =
    inv(_series_susceptance_raw(seg, units))

function _series_susceptance_raw(
    series_chain::BranchesSeries,
    units,
)::Float64
    return 1 / sum(_series_reactance(x, units) for x in series_chain)
end

"""
    get_equivalent_rating(bs::BranchesSeries) -> Union{Nothing, Float64}

Calculate the rating for branches in series.
Series chains can be composed of PSY.ACTransmission branches and parallel groups.
For series circuits, the rating is limited by the weakest link: Rating_total = min(Rating1, Rating2, ..., Ratingn).
Parallel members contribute their N-1 single-element-contingency rating.

Members with no known rating (transformer circuits carry `rating::Union{Nothing, Float64}`)
do not bind the minimum and are skipped; returns `nothing` only when no member has a known
rating.
"""
function get_equivalent_rating(bs::BranchesSeries)
    return _aggregate_known_ratings(minimum, _series_member_rating, bs)
end

_series_member_rating(branch::PSY.ACTransmission) = get_equivalent_rating(branch)

"""
    get_equivalent_rating(bs<:PSY.ACTransmission)

Return the rating for PSY.ACTransmission branches, per unit on the system base (`u"SU"`).
Every equivalent rating is on the system base, so a series minimum or a parallel sum can
combine members whose own base powers differ.
"""
function get_equivalent_rating(bs::PSY.ACTransmission)
    return PSY.get_rating(bs, u"SU")
end

"""
    get_equivalent_rating(bs::PSY.TwoWindingTransformer) -> Union{Nothing, Float64}

A `TwoWindingTransformer` has no parent rating (there is no `get_rating(::TwoWindingTransformer)`);
the rating lives on its single winding and may be `nothing`. Mirrors `branch_flow_limits`.
The winding stores it per unit on its own `base_power`; it is returned on the system base.
"""
function get_equivalent_rating(bs::PSY.TwoWindingTransformer)
    return PSY.get_rating(PSY.get_circuit(bs), u"SU")
end

"""
    get_equivalent_rating(bs::PSY.GenericArcImpedance)

The largest directional maximum of its `operational_flow_limit`, per unit on the system base.
A generic arc without an `operational_flow_limit` is unbounded and returns `Inf`.
"""
function get_equivalent_rating(bs::PSY.GenericArcImpedance)
    return _largest_flow_limit(PSY.get_operational_flow_limit(bs, u"SU"))
end

_largest_flow_limit(::Nothing) = Inf
_largest_flow_limit(ofl::NamedTuple) = max(ofl.from_to.max, ofl.to_from.max)

"""
    get_equivalent_emergency_rating(bs::BranchesSeries) -> Union{Nothing, Float64}

Calculate the emergency rating for branches in series.
For series circuits, the emergency rating is limited by the weakest link: Rating_total = min(Rating1, Rating2, ..., Ratingn)

Members with no known rating do not bind the minimum and are skipped; returns `nothing` only
when no member has a known rating (see [`get_equivalent_rating`](@ref)).
"""
function get_equivalent_emergency_rating(bs::BranchesSeries)
    return _aggregate_known_ratings(minimum, get_equivalent_emergency_rating, bs)
end

"""
    get_equivalent_emergency_rating(bs<:PSY.ACTransmission)

Return the emergency rating for PSY.ACTransmission branches, per unit on the system base.
"""
function get_equivalent_emergency_rating(branch::PSY.ACTransmission)
    if isnothing(PSY.get_rating_b(branch, u"SU"))
        @debug "Branch $(get_name(branch)) has no 'rating_b' defined. Post-contingency limit is going to be set using normal-operation rating.
            \n Consider including post-contingency limits using set_rating_b!()."
        return PSY.get_rating(branch, u"SU")
    end
    return PSY.get_rating_b(branch, u"SU")
end

"""
    get_equivalent_emergency_rating(branch::PSY.TwoWindingTransformer) -> Union{Nothing, Float64}

`TwoWindingTransformer` carries its ratings on the winding (no parent
`get_rating`/`get_rating_b`); falls back to the winding's normal-operation rating when
`rating_b` is unset. May return `nothing` when the winding has neither rating. Returned per
unit on the system base, like [`get_equivalent_rating`](@ref).
"""
get_equivalent_emergency_rating(branch::PSY.TwoWindingTransformer) =
    _circuit_emergency_rating(PSY.get_circuit(branch), "Winding of $(PSY.get_name(branch))")

"""
    get_equivalent_emergency_rating(bs<:PSY.ACTransmission)

Return the emergency rating for PSY.GenericArcImpedance.
"""
function get_equivalent_emergency_rating(branch::PSY.GenericArcImpedance)
    @debug "GenericArcImpedance $(get_name(branch)) has no emergency rating. Using its flow limit as a proxy instead."
    return get_equivalent_rating(branch)
end

# Indexed only when EVERY segment is: a chain missing one is not a valid representation of
# the path between its endpoints.
# Recursive: might be nested, have BranchesParallel as link in degree 2 chain.
function _entry_matches(chain::BranchesSeries, predicate)
    if _has_mixed_types(chain) && !_is_unfiltered(predicate)
        _warn_mixed_group("Series circuit", leaf_components(chain))
    end
    return all(_entry_matches(segment, predicate)::Bool for segment in chain)
end

function Base.:(==)(a::BranchesSeries, b::BranchesSeries)
    return a.branches == b.branches
end

function Base.show(io::IO, x::MIME{Symbol("text/plain")}, y::BranchesSeries)
    show(io, x, y.branches)
end
