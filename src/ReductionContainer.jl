"""
Holds the user-supplied irreducible bus set and the applied reduction specs.
`user_irreducible_buses` is seeded once by `Ybus(sys; irreducible_buses=...)` and
read by every `get_reduction` so the user portion is shared across reductions;
per-step `NetworkReductionData.irreducible_buses` adds each reduction's
system-derived complement on top.
"""
@kwdef mutable struct ReductionContainer
    user_irreducible_buses::Set{Int} = Set{Int}()
    zero_impedance_reduction::Union{Nothing, ZeroImpedanceBranchReduction} = nothing
    radial_reduction::Union{Nothing, RadialReduction} = nothing
    degree_two_reduction::Union{Nothing, DegreeTwoReduction} = nothing
    ward_reduction::Union{Nothing, WardReduction} = nothing
end

get_user_irreducible_buses(rb::ReductionContainer) = rb.user_irreducible_buses
get_zero_impedance_reduction(rb::ReductionContainer) = rb.zero_impedance_reduction
has_zero_impedance_reduction(rb::ReductionContainer) =
    !isnothing(rb.zero_impedance_reduction)
has_radial_reduction(rb::ReductionContainer) = !isnothing(rb.radial_reduction)
has_degree_two_reduction(rb::ReductionContainer) = !isnothing(rb.degree_two_reduction)
has_ward_reduction(rb::ReductionContainer) = !isnothing(rb.ward_reduction)

function _reject_after_ward(prior_reductions::ReductionContainer)
    has_ward_reduction(prior_reductions) &&
        throw(IS.DataFormatError("Ward reduction must be the last applied reduction"))
    return
end

# ZIBR is auto-applied by `Ybus(sys; ...)`; a user-supplied one would double-apply.
_reject_zibr_in_user_reductions(::NetworkReduction) = nothing
function _reject_zibr_in_user_reductions(::ZeroImpedanceBranchReduction)
    throw(
        IS.DataFormatError(
            "ZeroImpedanceBranchReduction is auto-applied during Ybus construction; " *
            "do not include it in `network_reductions`. To customize its detection " *
            "threshold or substituted reactance, pass `zero_impedance_reduction = " *
            "ZeroImpedanceBranchReduction(susceptance_threshold=..., " *
            "minimum_retained_impedance=...)` to `Ybus(sys; ...)`.",
        ),
    )
end

# Dispatched on the new reduction's concrete type: reject duplicates and reductions after Ward.
function validate_reduction_type(
    ::ZeroImpedanceBranchReduction,
    prior_reductions::ReductionContainer,
)
    _reject_after_ward(prior_reductions)
    return
end

function validate_reduction_type(
    ::RadialReduction,
    prior_reductions::ReductionContainer,
)
    _reject_after_ward(prior_reductions)
    has_radial_reduction(prior_reductions) &&
        throw(IS.DataFormatError("Radial reduction is applied twice to the same system"))
    has_degree_two_reduction(prior_reductions) &&
        @warn "When applying both RadialReduction and DegreeTwoReduction, it is likely beneficial to apply RadialReduction first."
    return
end

function validate_reduction_type(
    ::DegreeTwoReduction,
    prior_reductions::ReductionContainer,
)
    _reject_after_ward(prior_reductions)
    has_degree_two_reduction(prior_reductions) &&
        throw(
            IS.DataFormatError("Degree two reduction is applied twice to the same system"),
        )
    return
end

function validate_reduction_type(
    ::WardReduction,
    prior_reductions::ReductionContainer,
)
    has_ward_reduction(prior_reductions) &&
        throw(IS.DataFormatError("Ward reduction is applied twice to the same system"))
    return
end

# `user_irreducible_buses` and `zero_impedance_reduction` are configuration, seeded once
# at Ybus construction; per-reduction steps never update them.
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

# Configuration fields (user_irreducible_buses, zero_impedance_reduction) don't count
# toward emptiness; only applied-reduction slots do.
function Base.isempty(rb::ReductionContainer)
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
    empty!(rb.user_irreducible_buses)
    rb.zero_impedance_reduction = nothing
    rb.radial_reduction = nothing
    rb.degree_two_reduction = nothing
    rb.ward_reduction = nothing
    return rb
end
