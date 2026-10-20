"""
Holds the Ybus build configuration and the applied reduction specs.

- `user_irreducible_buses`: the buses the user passed as `Ybus(sys; irreducible_buses)`
- `schedule_pinned_buses`: the buses passed as `Ybus(sys; schedule_pinned_buses)`, kept out
  of every reduction because a scheduled topology change acts on them
- `closed_switches`: names of the available switches passed as `Ybus(sys; closed_switches)`,
  included in the network as closed regardless of their `branch_status`

Every `get_reduction` exempts the union of the user and schedule-pinned sets (see
`get_reduction_exempt_buses`); per-step `NetworkReductionData.irreducible_buses`
adds each reduction's system-derived complement on top.
"""
@kwdef mutable struct ReductionContainer
    user_irreducible_buses::Set{Int} = Set{Int}()
    schedule_pinned_buses::Set{Int} = Set{Int}()
    closed_switches::Set{String} = Set{String}()
    zero_impedance_reduction::Union{Nothing, ZeroImpedanceBranchReduction} = nothing
    radial_reduction::Union{Nothing, RadialReduction} = nothing
    degree_two_reduction::Union{Nothing, DegreeTwoReduction} = nothing
    ward_reduction::Union{Nothing, WardReduction} = nothing
end

get_user_irreducible_buses(rb::ReductionContainer) = rb.user_irreducible_buses
get_schedule_pinned_buses(rb::ReductionContainer) = rb.schedule_pinned_buses
get_closed_switches(rb::ReductionContainer) = rb.closed_switches

"""
    get_reduction_exempt_buses(rb::ReductionContainer) -> Set{Int}

The buses every reduction keeps: the user's irreducible buses and the schedule-pinned buses.
"""
get_reduction_exempt_buses(rb::ReductionContainer) =
    union(rb.user_irreducible_buses, rb.schedule_pinned_buses)
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

# `user_irreducible_buses`, `schedule_pinned_buses`, `closed_switches` and
# `zero_impedance_reduction` are configuration, seeded once at Ybus construction;
# per-reduction steps never update them.
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

Base.:(==)(x::ReductionContainer, y::ReductionContainer) = _fieldwise_equal(x, y)

# Configuration fields (user_irreducible_buses, schedule_pinned_buses, closed_switches,
# zero_impedance_reduction) don't count toward emptiness; only applied-reduction slots do.
function Base.isempty(rb::ReductionContainer)
    return all(isnothing, (rb.radial_reduction, rb.degree_two_reduction, rb.ward_reduction))
end

function Base.empty!(rb::ReductionContainer)
    empty!(rb.user_irreducible_buses)
    empty!(rb.schedule_pinned_buses)
    empty!(rb.closed_switches)
    rb.zero_impedance_reduction = nothing
    rb.radial_reduction = nothing
    rb.degree_two_reduction = nothing
    rb.ward_reduction = nothing
    return rb
end
