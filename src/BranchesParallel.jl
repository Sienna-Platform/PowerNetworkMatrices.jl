abstract type AbstractBranchesParallel <: PSY.ACTransmission end

mutable struct BranchesParallel{T <: PSY.ACTransmission} <: AbstractBranchesParallel
    branches::Vector{T}
    equivalent_ybus::Union{Matrix{YBUS_ELTYPE}, Nothing}

    function BranchesParallel{T}(
        branches::Vector{T},
        equivalent_ybus::Union{Matrix{YBUS_ELTYPE}, Nothing},
    ) where {T <: PSY.ACTransmission}
        if !isconcretetype(T)
            error(
                "BranchesParallel{T} requires a concrete branch type T. " *
                "Use MixedBranchesParallel for groups with mixed branch types. Got T=$T.",
            )
        end
        return new{T}(branches, equivalent_ybus)
    end
end

function BranchesParallel(branches::Vector{T}) where {T <: PSY.ACTransmission}
    return BranchesParallel{T}(branches, nothing)
end

mutable struct MixedBranchesParallel <: AbstractBranchesParallel
    branches::Vector{PSY.ACTransmission}
    equivalent_ybus::Union{Matrix{YBUS_ELTYPE}, Nothing}
end

function MixedBranchesParallel(branches::Vector{<:PSY.ACTransmission})
    return MixedBranchesParallel(Vector{PSY.ACTransmission}(branches), nothing)
end

function add_branch!(bp::BranchesParallel{T}, branch::T) where {T <: PSY.ACTransmission}
    push!(bp.branches, branch)
end

function add_branch!(mbp::MixedBranchesParallel, branch::PSY.ACTransmission)
    push!(mbp.branches, branch)
end

function get_name(bp::AbstractBranchesParallel)
    base_string = _longest_starting_substring(PSY.get_name.(bp.branches)...)
    if isempty(base_string)
        base_string = join(PSY.get_name.(bp.branches), "_") * "_"
    end
    return base_string *= "double_circuit"
end

function _longest_starting_substring(branch_names...)
    first_name = first(branch_names)
    n_chars = minimum(length.(branch_names))
    n_branches = length(branch_names)
    for ix in 1:n_chars
        for jx in 2:n_branches
            if branch_names[jx][ix] != first_name[ix]
                if ix == 1
                    return ""
                else
                    return first_name[1:(ix - 1)]
                end
            end
        end
    end
    return first_name[1:n_chars]
end

function compute_parallel_multiplier(
    parallel_branch_set::AbstractBranchesParallel,
    branch_name::String,
)
    b_total = 0.0
    b_branch = 0.0
    for br in parallel_branch_set
        if PSY.get_name(br) == branch_name
            b_branch += PSY.get_series_susceptance(br, PSY.SU)
        end
        b_total += PSY.get_series_susceptance(br, PSY.SU)
    end
    return b_branch / b_total
end

function get_series_susceptance(
    segment::AbstractBranchesParallel,
    units::IS.AbstractUnitSystem,
)
    return sum(get_series_susceptance(branch, units) for branch in segment.branches)
end

function get_equivalent_physical_branch_parameters(bp::AbstractBranchesParallel)
    if isnothing(bp.equivalent_ybus)
        populate_equivalent_ybus!(bp)
    end
    equivalent_ybus = bp.equivalent_ybus
    return _get_equivalent_physical_branch_parameters(equivalent_ybus)
end

function populate_equivalent_ybus!(bp::AbstractBranchesParallel)
    Y11, Y12, Y21, Y22 = ybus_branch_entries(bp)
    bp.equivalent_ybus = YBUS_ELTYPE[Y11 Y12; Y21 Y22]
    return
end

"""
    get_sum_of_max_rating(bp::AbstractBranchesParallel)

Sum of the individual branch ratings, treating each circuit as independently loadable
up to its own thermal limit. This is the least conservative aggregate and assumes
unconstrained flow steering across the parallel group.
"""
function get_sum_of_max_rating(bp::AbstractBranchesParallel)
    return sum(get_equivalent_rating(branch) for branch in bp.branches)
end

"""
    get_single_element_contingency_rating(bp::AbstractBranchesParallel)

N-1 rating for the parallel group: the surviving capacity after the largest-rated
circuit trips, ``\\sum_i S_i - \\max_i S_i``. For a group of one branch this is zero.
"""
function get_single_element_contingency_rating(bp::AbstractBranchesParallel)
    ratings = get_equivalent_rating.(bp.branches)
    return sum(ratings) - maximum(ratings)
end

"""
    get_impedance_averaged_rating(bp::AbstractBranchesParallel)

Susceptance-weighted average of individual branch ratings,
``\\sum_i f_i \\cdot S_i`` with ``f_i = b_i / \\sum_k b_k``. Reflects how DC flow
physically splits across a parallel group. Throws `ArgumentError` if the total
series susceptance is zero or non-finite.
"""
function get_impedance_averaged_rating(bp::AbstractBranchesParallel)
    # The susceptance weights must share a consistent impedance base across the
    # group, so use system base (SU) like the sibling `compute_parallel_multiplier`.
    # Within a parallel group (a single bus pair) this equals the natural-units
    # weighting; device base would mix bases when the branches differ in base power.
    # Requires the branches to be attached to a system.
    b_total = sum(PSY.get_series_susceptance(br, PSY.SU) for br in bp.branches)
    if !isfinite(b_total) || iszero(b_total)
        throw(
            ArgumentError(
                "Cannot compute impedance-averaged rating: total series susceptance across the parallel group must be finite and non-zero.",
            ),
        )
    end
    return sum(
        PSY.get_series_susceptance(br, PSY.SU) / b_total * get_equivalent_rating(br)
        for br in bp.branches
    )
end

# Series-chain rating contribution for a parallel block: dispatch arm for
# `get_equivalent_rating(::BranchesSeries)` defined in BranchesSeries.jl.
_series_member_rating(bp::AbstractBranchesParallel) =
    get_single_element_contingency_rating(bp)

"""
    get_equivalent_emergency_rating(bp::AbstractBranchesParallel)

Calculate the total emergency rating for branches in parallel.
For parallel circuits, the emergency rating is the sum of individual emergency ratings divided by the number of circuits.
This provides a conservative estimate that accounts for potential overestimation of total capacity.
"""
function get_equivalent_emergency_rating(bp::AbstractBranchesParallel)
    equivalent_rating = 0.0
    for branch in bp.branches
        rating_b = get_equivalent_emergency_rating(branch)
        equivalent_rating += rating_b
    end
    return equivalent_rating # In Emergency conditions, we should consider the full capacity
end

"""
    get_equivalent_available(bp::AbstractBranchesParallel)

Get the availability status for parallel branches.
All branches in parallel must be available for the parallel circuit to be available.
"""
function get_equivalent_available(bp::AbstractBranchesParallel)
    # All branches must be available
    return all(PSY.get_available(branch) for branch in bp.branches)
end

PSY.get_available(bp::AbstractBranchesParallel) = get_equivalent_available(bp)

"""
    get_equivalent_α(bp::AbstractBranchesParallel)

Get the phase angle shift for parallel branches.
Returns the average phase angle shift across all parallel branches.
Returns 0.0 if branches don't support phase angle shift (e.g., lines).
"""
function get_equivalent_α(bp::AbstractBranchesParallel)
    # Need to check the PS books
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

function add_to_map(
    double_circuit::BranchesParallel{T},
    filters::Dict,
) where {T <: PSY.ACTransmission}
    isempty(filters) && return true
    if !haskey(filters, T)
        return true
    end
    return any(filters[T](device) for device in double_circuit)
end

function add_to_map(double_circuit::MixedBranchesParallel, filters::Dict)
    isempty(filters) && return true
    @warn "Parallel circuit contains mixed branch types, filters might be applied to more components than intended. Use Logging.Debug for additional information."
    @debug "Parallel circuit branch types: $(typeof.(double_circuit.branches))"
    @debug "Parallel circuit branch names: $(PSY.get_name.(double_circuit.branches))"
    for branch in double_circuit.branches
        filter = get(filters, typeof(branch), x -> true)
        if !filter(branch)
            return false
        end
    end
    return true
end

function Base.:(==)(a::AbstractBranchesParallel, b::AbstractBranchesParallel)
    return a.branches == b.branches
end

function Base.show(io::IO, x::MIME{Symbol("text/plain")}, y::AbstractBranchesParallel)
    show(io, x, y.branches)
end

is_a_reduction(::AbstractBranchesParallel) = true
