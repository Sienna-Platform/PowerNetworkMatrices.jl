@kwdef mutable struct ReductionContainer
    radial_reduction::Union{Nothing, RadialReduction} = nothing
    degree_two_reduction::Union{Nothing, DegreeTwoReduction} = nothing
    ward_reduction::Union{Nothing, WardReduction} = nothing
end

has_radial_reduction(rb::ReductionContainer) = !isnothing(rb.radial_reduction)
has_degree_two_reduction(rb::ReductionContainer) = !isnothing(rb.degree_two_reduction)
has_ward_reduction(rb::ReductionContainer) = !isnothing(rb.ward_reduction)

function validate_reduction_type(
    new_reduction::NetworkReduction,
    prior_reductions::ReductionContainer,
)
    if typeof(new_reduction) == RadialReduction && has_radial_reduction(prior_reductions)
        throw(IS.DataFormatError("Radial reduction is applied twice to the same system"))
    end
    if typeof(new_reduction) == DegreeTwoReduction &&
       has_degree_two_reduction(prior_reductions)
        throw(
            IS.DataFormatError("Degree two reduction is applied twice to the same system"),
        )
    end
    if typeof(new_reduction) == WardReduction && has_ward_reduction(prior_reductions)
        throw(IS.DataFormatError("Ward reduction is applied twice to the same system"))
    end
    if has_ward_reduction(prior_reductions)
        throw(IS.DataFormatError("Ward reduction must be the last applied reduction"))
    end
    if typeof(new_reduction) == RadialReduction &&
       has_degree_two_reduction(prior_reductions)
        @warn "When applying both RadialReduction and DegreeTwoReduction, it is likely beneficial to apply RadialReduction first."
    end
    return
end

function add_reduction!(r1::ReductionContainer, r2::ReductionContainer)
    if has_radial_reduction(r2)
        r1.radial_reduction = r2.radial_reduction
    end
    if has_degree_two_reduction(r2)
        r1.degree_two_reduction = r2.degree_two_reduction
    end
    if has_ward_reduction(r2)
        r1.ward_reduction = r2.ward_reduction
    end
end

function Base.:(==)(x::ReductionContainer, y::ReductionContainer)
    for field in fieldnames(ReductionContainer)
        if getfield(x, field) != getfield(y, field)
            return false
        end
    end
    return true
end

function Base.isempty(rb::ReductionContainer)
    # Verbose on purpose
    if !isnothing(rb.radial_reduction)
        return false
    end
    if !isnothing(rb.degree_two_reduction)
        return false
    end
    if !isnothing(rb.ward_reduction)
        return false
    end
    return true
end

function Base.empty!(rb::ReductionContainer)
    for field in fieldnames(ReductionContainer)
        setfield!(rb, field, nothing)
    end
end
