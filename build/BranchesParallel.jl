mutable struct BranchesParallel{T <: PSY.ACTransmission} <: PSY.ACTransmission
    branches::Vector{T}
    equivalent_ybus::Union{Matrix{ComplexF32}, Nothing}
end

function BranchesParallel(branches::Vector{T}) where {T <: PSY.ACTransmission}
    BranchesParallel(branches, nothing)
end
# Constructor for the mixed types
function BranchesParallel(branches::Vector{PSY.ACTransmission})
    return BranchesParallel{PSY.ACTransmission}(branches, nothing)
end

function add_branch!(bp::BranchesParallel{T}, branch::T) where {T <: PSY.ACTransmission}
    push!(bp.branches, branch)
end

get_branch_type(::BranchesParallel{T}) where {T <: PSY.ACTransmission} = T

function get_name(bp::BranchesParallel{T}) where {T <: PSY.ACTransmission}
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
    parallel_branch_set::BranchesParallel,
    branch_name::String,
)
    b_total = 0.0
    b_branch = 0.0
    for br in parallel_branch_set
        if PSY.get_name(br) == branch_name
            b_branch += PSY.get_series_susceptance(br)
        end
        b_total += PSY.get_series_susceptance(br)
    end
    return b_branch / b_total
end

function get_series_susceptance(segment::BranchesParallel)
    return sum(get_series_susceptance(branch) for branch in segment.branches)
end

function get_equivalent_physical_branch_parameters(bp::BranchesParallel)
    if isnothing(bp.equivalent_ybus)
        populate_equivalent_ybus!(bp)
    end
    equivalent_ybus = bp.equivalent_ybus
    return _get_equivalent_physical_branch_parameters(equivalent_ybus)
end

function populate_equivalent_ybus!(bp::BranchesParallel)
    Y11, Y12, Y21, Y22 = ybus_branch_entries(bp)
    bp.equivalent_ybus = ComplexF32[Y11 Y12; Y21 Y22]
    return
end

"""
    get_equivalent_rating(bp::BranchesParallel)

Calculate the total rating for branches in parallel.
For parallel circuits, the rating is the sum of individual ratings divided by the number of circuits.
This provides a conservative estimate that accounts for potential overestimation of total capacity.
"""
function get_equivalent_rating(bp::BranchesParallel)
    # Sum of ratings divided by number of circuits
    return sum(PSY.get_rating(branch) for branch in bp.branches) / length(bp.branches)
end

"""
    get_equivalent_emergency_rating(bp::BranchesParallel)

Calculate the total emergency rating for branches in parallel.
For parallel circuits, the emergency rating is the sum of individual emergency ratings divided by the number of circuits.
This provides a conservative estimate that accounts for potential overestimation of total capacity.
"""
function get_equivalent_emergency_rating(bp::BranchesParallel)
    equivalent_rating = 0.0
    for branch in bp.branches
        rating_b = get_equivalent_emergency_rating(branch)
        equivalent_rating += rating_b
    end
    return equivalent_rating # In Emergency conditions, we should consider the full capacity
end

"""
    get_equivalent_available(bp::BranchesParallel)

Get the availability status for parallel branches.
All branches in parallel must be available for the parallel circuit to be available.
"""
function get_equivalent_available(bp::BranchesParallel)
    # All branches must be available
    return all(PSY.get_available(branch) for branch in bp.branches)
end

"""
    get_equivalent_α(bp::BranchesParallel)

Get the phase angle shift for parallel branches.
Returns the average phase angle shift across all parallel branches.
Returns 0.0 if branches don't support phase angle shift (e.g., lines).
"""
function get_equivalent_α(bp::BranchesParallel)
    # Need to check the PS books
end

function Base.iterate(bp::BranchesParallel)
    return iterate(bp.branches)
end

function Base.iterate(bp::BranchesParallel, state)
    return iterate(bp.branches, state)
end

function Base.length(bp::BranchesParallel)
    return length(bp.branches)
end

function add_to_map(
    double_circuit::BranchesParallel{T},
    filters::Dict,
) where {T <: PSY.ACTransmission}
    isempty(filters) && return true
    if isabstracttype(T)
        @warn "Parallel circuit contains mixed branch types, filters might be applied to more components than intended. Use Logging.Debug for additional information."
        @debug "Parallel circuit branch types: $(keys(double_circuit.branches))"
        @debug "Parallel circuit branch names: $(vcat([PSY.get_name.(v) for (k , v) in double_circuit.branches]))"
        for branch in double_circuit.branches
            filter = get(filters, typeof(branch), x -> true)
            if !filter(branch)
                return false
            end
        end
        return true
    end

    if !haskey(filters, T)
        return true
    else
        return any([filters[T](device) for device in double_circuit])
    end
    error("Invalid condition reached in add_to_map for BranchesParallel")
end

function Base.:(==)(a::BranchesParallel, b::BranchesParallel)
    return a.branches == b.branches
end

function Base.show(io::IO, x::MIME{Symbol("text/plain")}, y::BranchesParallel)
    show(io, x, y.branches)
end
