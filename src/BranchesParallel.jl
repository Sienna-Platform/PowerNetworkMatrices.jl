abstract type AbstractBranchesParallel <: AbstractReductionAggregate end

# `arc_key` is the group's canonical arc in original bus numbers (the seed member's
# orientation). It is remapped with `nr` on read, so orientation no longer depends on the
# order of `branches`.
mutable struct BranchesParallel{T <: PSY.ACTransmission} <: AbstractBranchesParallel
    branches::Vector{T}
    arc_key::Tuple{Int, Int}
    equivalent_ybus::CACHED_TWO_PORT
    equivalent_ybus_populated::Bool

    function BranchesParallel{T}(
        branches::Vector{T},
        arc_key::Tuple{Int, Int},
        equivalent_ybus::CACHED_TWO_PORT,
        equivalent_ybus_populated::Bool,
    ) where {T <: PSY.ACTransmission}
        if !isconcretetype(T)
            error(
                "BranchesParallel{T} requires a concrete branch type T. " *
                "Use MixedBranchesParallel for groups with mixed branch types. Got T=$T.",
            )
        end
        return new{T}(branches, arc_key, equivalent_ybus, equivalent_ybus_populated)
    end
end

function BranchesParallel(branches::Vector{T}) where {T <: PSY.ACTransmission}
    return BranchesParallel{T}(
        branches,
        get_arc_tuple(first(branches)),
        EMPTY_TWO_PORT,
        false,
    )
end

mutable struct MixedBranchesParallel <: AbstractBranchesParallel
    branches::Vector{PSY.ACTransmission}
    arc_key::Tuple{Int, Int}
    equivalent_ybus::CACHED_TWO_PORT
    equivalent_ybus_populated::Bool
end

function MixedBranchesParallel(branches::Vector{<:PSY.ACTransmission})
    typed = Vector{PSY.ACTransmission}(branches)
    return MixedBranchesParallel(
        typed,
        get_arc_tuple(first(typed)),
        EMPTY_TWO_PORT,
        false,
    )
end

# The cached two-port is a function of the member set and the group's arc frame, so any change
# to either must clear it or later queries return a stale equivalent.
function invalidate_equivalent_ybus!(segment)
    segment.equivalent_ybus_populated = false
    return
end

function add_branch!(bp::BranchesParallel{T}, branch::T) where {T <: PSY.ACTransmission}
    push!(bp.branches, branch)
    invalidate_equivalent_ybus!(bp)
    return
end

function add_branch!(mbp::MixedBranchesParallel, branch::PSY.ACTransmission)
    push!(mbp.branches, branch)
    invalidate_equivalent_ybus!(mbp)
    return
end

# The blanket `_is_phase_shifting(::PSY.ACTransmission) = false` in definitions.jl would
# silently answer for groups; a group shifts when any member does.
function _is_phase_shifting(bp::AbstractBranchesParallel)
    return any(_is_phase_shifting, bp.branches)
end

"""
An aggregate's name is its arc, spelled `<from>_<to>_double_circuit`.

`double_circuit` is kept from the previous scheme: it is what tells a reader of results that
the row is a total across the parallel members rather than one member's own flow. Only the
stem changed, from the members' longest common prefix to the arc.

The bus numbers are the group's own `arc_key`, i.e. the endpoints as they stood when the
group was formed; a later reduction can remap them, so this is an identity for reading, not a
key to look the group up with. `BranchCatalog._entry_name` derives the *indexed* name from
the arc the catalog files the entry under, which is the remapped one.

The prefix stem was not injective -- `La` ∥ `Lb` and `Lc` ∥ `Ld` both produced
`Ldouble_circuit` -- and it moved whenever membership did, so a corridor was renamed by the
act of adding a circuit to it. The arc stem is injective by construction.
"""
get_name(bp::AbstractBranchesParallel) =
    "$(bp.arc_key[1])_$(bp.arc_key[2])_double_circuit"

"""
    compute_parallel_multiplier(parallel_branch_set, branch) -> Float64

Susceptance fraction `b_branch / b_total` of one member of a parallel group. The member is
resolved by object identity; passing a component that is not in the group is an error.
"""
function compute_parallel_multiplier(
    parallel_branch_set::AbstractBranchesParallel,
    branch::PSY.ACTransmission,
)
    b_total = 0.0
    b_branch = 0.0
    found = false
    for br in parallel_branch_set
        # `get_series_susceptance` (see BranchAdmittance.jl) is tap-aware for
        # two-winding transformers and dispatches PNM's three-winding winding wrapper.
        if br === branch
            b_branch = get_series_susceptance(br, PSY.SU)
            found = true
        end
        b_total += get_series_susceptance(br, PSY.SU)
    end
    if !found
        error(
            "Branch $(get_name(branch)) is not a member of parallel group " *
            "$(get_name(parallel_branch_set)).",
        )
    end
    return b_branch / b_total
end

# Name-based lookup kept for callers that only hold a name (PTDF row API, PowerFlows).
# PSY names are unique per concrete type only, so a name may match several members of a
# mixed group; that was silently double-counted before — now it must resolve to exactly one.
function compute_parallel_multiplier(
    parallel_branch_set::AbstractBranchesParallel,
    branch_name::String,
)
    matches = PSY.ACTransmission[]
    for br in parallel_branch_set
        if get_name(br) == branch_name
            push!(matches, br)
        end
    end
    if length(matches) != 1
        error(
            "Branch name $(branch_name) matches $(length(matches)) members of parallel " *
            "group $(get_name(parallel_branch_set)); resolve by component identity.",
        )
    end
    return compute_parallel_multiplier(parallel_branch_set, first(matches))
end

function get_series_susceptance(
    segment::AbstractBranchesParallel,
    units::IS.AbstractUnitSystem,
)
    return sum(get_series_susceptance(branch, units) for branch in segment.branches)
end

# `get_equivalent_physical_branch_parameters` / `populate_equivalent_ybus!` for parallel and
# series groups live in common.jl, which is included after NetworkReductionData so `nr` can be
# typed.

# The single home for the group-rating `nothing` policy: a `PSY.TransformerCircuit` rating may
# be `nothing` (unlike a `Line`, whose rating is always a `Float64`), so aggregate only over
# the members with a known rating and propagate `nothing` when none is known. The `Float64`
# accumulator keeps the eltype concrete — `filter(!isnothing, …)` would retain the union.
function _aggregate_known_ratings(agg, rating_of, branches)
    known = Float64[]
    for br in branches
        r = rating_of(br)
        if !isnothing(r)
            push!(known, r)
        end
    end
    if isempty(known)
        return nothing
    end
    return agg(known)
end

"""
    get_sum_of_max_rating(bp::AbstractBranchesParallel) -> Union{Nothing, Float64}

Sum of the individual branch ratings, treating each circuit as independently loadable
up to its own thermal limit. This is the least conservative aggregate and assumes
unconstrained flow steering across the parallel group.

Members with no known rating (transformer circuits carry `rating::Union{Nothing, Float64}`)
are skipped; returns `nothing` only when no member has a known rating.
"""
function get_sum_of_max_rating(bp::AbstractBranchesParallel)
    return _aggregate_known_ratings(sum, get_equivalent_rating, bp.branches)
end

"""
    get_single_element_contingency_rating(bp::AbstractBranchesParallel) -> Union{Nothing, Float64}

N-1 rating for the parallel group: the surviving capacity after the largest-rated
circuit trips, ``\\sum_i S_i - \\max_i S_i``. For a group of one branch this is zero.

Members with no known rating are skipped; returns `nothing` only when no member has a known
rating (see [`get_sum_of_max_rating`](@ref)).
"""
function get_single_element_contingency_rating(bp::AbstractBranchesParallel)
    return _aggregate_known_ratings(
        r -> sum(r) - maximum(r),
        get_equivalent_rating,
        bp.branches,
    )
end

"""
    get_impedance_averaged_rating(bp::AbstractBranchesParallel) -> Union{Nothing, Float64}

Susceptance-weighted average of individual branch ratings,
``\\sum_i f_i \\cdot S_i`` with ``f_i = b_i / \\sum_k b_k``. Reflects how DC flow
physically splits across a parallel group. Throws `ArgumentError` if the total
series susceptance is zero or non-finite.

Members with no known rating are skipped (their susceptance still contributes to the
weighting denominator); returns `nothing` only when no member has a known rating (see
[`get_sum_of_max_rating`](@ref)).
"""
function get_impedance_averaged_rating(bp::AbstractBranchesParallel)
    # The susceptance weights must share a consistent impedance base across the group, so use
    # system base (SU) like the sibling `compute_parallel_multiplier`. Within a parallel group
    # (a single bus pair) this equals the natural-units weighting; device base would mix bases
    # when the branches differ in base power. Requires the branches to be attached to a system.
    # Σᵢ (bᵢ/b_total)·rᵢ == (Σᵢ bᵢ·rᵢ)/b_total, so one pass and no stored per-member state.
    b_total = 0.0
    numerator = 0.0
    any_known = false
    for br in bp.branches
        b = get_series_susceptance(br, PSY.SU)
        b_total += b
        r = get_equivalent_rating(br)
        if !isnothing(r)
            numerator += b * r
            any_known = true
        end
    end
    if !isfinite(b_total) || iszero(b_total)
        throw(
            ArgumentError(
                "Cannot compute impedance-averaged rating: total series susceptance across the parallel group must be finite and non-zero.",
            ),
        )
    end
    if !any_known
        return nothing
    end
    return numerator / b_total
end

# Series-chain rating contribution for a parallel block: dispatch arm for
# `get_equivalent_rating(::BranchesSeries)` defined in BranchesSeries.jl. A chain is only as
# strong as its weakest link *after* a contingency, so a parallel block inside a chain
# contributes its N-1 rating rather than its normal-operation total.
_series_member_rating(bp::AbstractBranchesParallel) =
    get_single_element_contingency_rating(bp)

"""
    get_equivalent_rating(bp::AbstractBranchesParallel) -> Union{Nothing, Float64}

Normal-operation rating of the equivalent arc: the sum of the members' ratings, since every
circuit on the arc carries flow simultaneously. This is [`get_sum_of_max_rating`](@ref), and it
is the unexported generic fallback used inside PNM by [`branch_flow_limits`](@ref) and
`get_partition_rating`.

It is **not** the aggregate a consumer gets by default. A series chain applies the N-1 variant
[`get_single_element_contingency_rating`](@ref) to an embedded parallel block (see
`_series_member_rating`), and POM selects the aggregate per `DeviceModel` — also defaulting to
N-1. Pick the named aggregator explicitly rather than relying on this fallback.

Members with no known rating are skipped; returns `nothing` only when no member has a known
rating.
"""
get_equivalent_rating(bp::AbstractBranchesParallel) = get_sum_of_max_rating(bp)

"""
    get_equivalent_emergency_rating(bp::AbstractBranchesParallel) -> Union{Nothing, Float64}

Sum of the members' emergency ratings, matching `get_equivalent_rating`'s normal-operation
convention: every circuit on the arc carries flow simultaneously.

Members with no known rating are skipped; returns `nothing` only when no member has a known
rating (see [`get_sum_of_max_rating`](@ref)).
"""
function get_equivalent_emergency_rating(bp::AbstractBranchesParallel)
    return _aggregate_known_ratings(sum, get_equivalent_emergency_rating, bp.branches)
end

function Base.iterate(bp::AbstractBranchesParallel)
    return iterate(bp.branches)
end

function Base.iterate(bp::AbstractBranchesParallel, state)
    return iterate(bp.branches, state)
end

function Base.length(bp::AbstractBranchesParallel)
    return length(bp.branches)
end

# Indexed when ANY member is: the arc is modeled, so its group must be reachable.
# Recursive: consider series-in-parallel.
_entry_matches(group::BranchesParallel, predicate) =
    any(_entry_matches(member, predicate)::Bool for member in group)

# Indexed only when EVERY member is: a partially-filtered mixed group is not a complete
# representation of its arc.
function _entry_matches(group::MixedBranchesParallel, predicate)
    _is_unfiltered(predicate) ||
        _warn_mixed_group("Parallel circuit", _get_segment_components(group))
    return all(_entry_matches(member, predicate)::Bool for member in group)
end

function Base.:(==)(a::AbstractBranchesParallel, b::AbstractBranchesParallel)
    return a.branches == b.branches
end

function Base.show(io::IO, x::MIME{Symbol("text/plain")}, y::AbstractBranchesParallel)
    show(io, x, y.branches)
end
