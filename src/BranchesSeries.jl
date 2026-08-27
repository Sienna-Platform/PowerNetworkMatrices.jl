mutable struct BranchesSeries <: AbstractReductionAggregate
    branches::Dict{DataType, Vector{<:PSY.ACTransmission}}
    needs_insertion_order::Bool
    insertion_order::Vector{Tuple{DataType, Int}}
    segment_orientations::Vector{Symbol}
    # The chain's endpoints in original bus numbers, remapped with `nr` on read. A chain can be
    # a member of a parallel group, where orientation is resolved against the group's frame.
    arc_key::Tuple{Int, Int}
    equivalent_ybus::CACHED_TWO_PORT
    equivalent_ybus_populated::Bool
end

BranchesSeries(arc_key::Tuple{Int, Int}) = BranchesSeries(
    Dict{DataType, Vector{<:PSY.ACTransmission}}(),
    false,
    Vector{Tuple{DataType, Int}}(),
    Vector{Symbol}(),
    arc_key,
    EMPTY_TWO_PORT,
    false,
)

function add_branch!(
    bs::BranchesSeries,
    branch::T,
    orientation,
) where {T <: PSY.ACTransmission}
    # Clear the cached two-port up front so every early return below is covered.
    invalidate_equivalent_ybus!(bs)
    push!(bs.segment_orientations, orientation)
    if isempty(bs.branches)
        # add the branch just once and return
        push!(get!(bs.branches, T, Vector{T}()), branch)
        return
    end

    if haskey(bs.branches, T) && !bs.needs_insertion_order
        push!(bs.branches[T], branch)
    elseif !haskey(bs.branches, T) && isempty(bs.insertion_order)
        bs.needs_insertion_order = true
        @assert length(keys(bs.branches)) == 1
        for (existing_type, existing_branches) in bs.branches
            for i in eachindex(existing_branches)
                push!(bs.insertion_order, (existing_type, i))
            end
        end
        push!(get!(bs.branches, T, Vector{T}()), branch)
        push!(bs.insertion_order, (T, 1))
    elseif !haskey(bs.branches, T) && !isempty(bs.insertion_order)
        push!(get!(bs.branches, T, Vector{T}()), branch)
        push!(bs.insertion_order, (T, length(bs.branches[T])))
    else
        push!(bs.branches[T], branch)
        push!(bs.insertion_order, (T, length(bs.branches[T])))
    end
    return
end

# Iteration support
function Base.iterate(bs::BranchesSeries)
    if isempty(bs.branches)
        return nothing
    end

    # Single key case - iterate over the single vector directly
    if bs.needs_insertion_order
        # Multi-key case - use insertion_order
        if isempty(bs.insertion_order)
            return nothing
        end

        type, idx = bs.insertion_order[1]
        branch = bs.branches[type][idx]
        return (branch, (1, nothing))
    else
        single_vector = first(values(bs.branches))
        if isempty(single_vector)
            return nothing
        end
        return (single_vector[1], (1, single_vector))
    end
end

function Base.iterate(bs::BranchesSeries, state)
    position, vector_cache = state

    if bs.needs_insertion_order
        # Multi-key iteration using insertion_order
        next_position = position + 1
        if next_position > length(bs.insertion_order)
            return nothing
        end
        type, idx = bs.insertion_order[next_position]
        branch = bs.branches[type][idx]
        return (branch, (next_position, nothing))
    else
        # Single key iteration
        next_idx = position + 1
        if next_idx > length(vector_cache)
            return nothing
        end
        return (vector_cache[next_idx], (next_idx, vector_cache))
    end
end

Base.length(bs::BranchesSeries) =
    if bs.needs_insertion_order
        length(bs.insertion_order)
    else
        sum(length(v) for v in values(bs.branches))
    end

Base.eltype(::Type{BranchesSeries}) = PSY.ACTransmission

# Chain segments can themselves be parallel groups, so this recurses through
# `_is_phase_shifting(::AbstractBranchesParallel)` (BranchesParallel.jl).
function _is_phase_shifting(bs::BranchesSeries)
    return any(_is_phase_shifting, bs)
end

# `BranchesSeries` has no `name` field, so the generic `PSY.ACTransmission` fallback
# (`get_name(device::T) where {T <: PSY.ACTransmission}`, NetworkReductionData.jl) errors on
# it. Unqualified `get_name` on each segment recurses through nested parallel/series segments
# the same way `_is_phase_shifting` does.
function get_name(bs::BranchesSeries)
    names = [get_name(br) for br in bs]
    base_string = _longest_starting_substring(names...)
    if isempty(base_string)
        base_string = join(names, "_") * "_"
    end
    return base_string *= "series_chain"
end

function get_series_susceptance(
    series_chain::BranchesSeries,
    units::IS.AbstractUnitSystem,
)
    series_susceptances_sum =
        sum(inv(get_series_susceptance(x, units)) for x in series_chain)
    total_susceptance = 1 / series_susceptances_sum
    return total_susceptance
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

Return the rating for PSY.ACTransmission branches.
"""
function get_equivalent_rating(bs::PSY.ACTransmission)
    return PSY.get_rating(bs, PSY.DU)
end

"""
    get_equivalent_rating(bs::PSY.TwoWindingTransformer) -> Union{Nothing, Float64}

A `TwoWindingTransformer` has no parent rating (there is no `get_rating(::TwoWindingTransformer)`);
the rating lives on its single winding and may be `nothing`. Mirrors `branch_flow_limits`.
"""
function get_equivalent_rating(bs::PSY.TwoWindingTransformer)
    return PSY.get_rating(PSY.get_circuit(bs), PSY.DU)
end

"""
    get_equivalent_rating(bs::PSY.GenericArcImpedance)

Rating is assumed to be max_flow for GenericArcImpedance.
"""
function get_equivalent_rating(bs::PSY.GenericArcImpedance)
    # Detached synthetic ward equivalent: read the stored value with device base.
    return PSY.get_max_flow(bs, PSY.DU)
end

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

Return the emergency rating for PSY.ACTransmission branches.
"""
function get_equivalent_emergency_rating(branch::PSY.ACTransmission)
    if isnothing(PSY.get_rating_b(branch, PSY.DU))
        @debug "Branch $(get_name(branch)) has no 'rating_b' defined. Post-contingency limit is going to be set using normal-operation rating.
            \n Consider including post-contingency limits using set_rating_b!()."
        return PSY.get_rating(branch, PSY.DU)
    end
    return PSY.get_rating_b(branch, PSY.DU)
end

"""
    get_equivalent_emergency_rating(branch::PSY.TwoWindingTransformer) -> Union{Nothing, Float64}

`TwoWindingTransformer` carries its ratings on the winding (no parent
`get_rating`/`get_rating_b`); falls back to the winding's normal-operation rating when
`rating_b` is unset. May return `nothing` when the winding has neither rating.
"""
function get_equivalent_emergency_rating(branch::PSY.TwoWindingTransformer)
    w = PSY.get_circuit(branch)
    if isnothing(PSY.get_rating_b(w, PSY.DU))
        @debug "Winding of $(PSY.get_name(branch)) has no 'rating_b' defined; using normal-operation rating."
        return PSY.get_rating(w, PSY.DU)
    end
    return PSY.get_rating_b(w, PSY.DU)
end

"""
    get_equivalent_emergency_rating(bs<:PSY.ACTransmission)

Return the emergency rating for PSY.GenericArcImpedance.
"""
function get_equivalent_emergency_rating(branch::PSY.GenericArcImpedance)
    @debug "GenericArcImpedance $(get_name(branch)) has no emergency rating. Using max_flow as a proxy instead."
    return PSY.get_max_flow(branch, PSY.DU)
end

# Indexed only when every segment is: a chain missing one is not a valid representation of
# the path between its endpoints.
function _entry_matches(chain::BranchesSeries, predicate)
    if chain.needs_insertion_order && !_is_unfiltered(predicate)
        _warn_mixed_group("Series circuit", _chain_branches(chain))
    end
    for (branch_type, branch_list) in chain.branches
        for device in branch_list
            predicate(branch_type, device) || return false
        end
    end
    return true
end

_chain_branches(chain::BranchesSeries) =
    reduce(vcat, values(chain.branches); init = PSY.ACTransmission[])

function Base.:(==)(a::BranchesSeries, b::BranchesSeries)
    return a.branches == b.branches
end

function Base.show(io::IO, x::MIME{Symbol("text/plain")}, y::BranchesSeries)
    show(io, x, y.branches)
end
