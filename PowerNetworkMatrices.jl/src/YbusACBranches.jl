struct YbusACBranches
    lines::Vector{PSY.Line}
    generic_arc_impedances::Vector{PSY.GenericArcImpedance}
    two_winding_transformers::Vector{PSY.TwoWindingTransformer}
    breaker_switches::Vector{PSY.DiscreteControlledACBranch}
    series_compensators::Vector{PSY.SeriesCompensator}
end

function Base.length(b::YbusACBranches)::Int
    return length(b.lines) +
           length(b.generic_arc_impedances) +
           length(b.two_winding_transformers) +
           length(b.breaker_switches) +
           length(b.series_compensators)
end

# Ybus stamps a branch or a circuit only when this is true.
_in_ybus(br::PSY.ACTransmission) = PSY.get_available(br)
_in_ybus(circuit::PSY.TransformerCircuit) = PSY.get_available(circuit)

function _in_ybus(br::PSY.DiscreteControlledACBranch)
    return PSY.get_available(br) &&
           PSY.get_branch_status(br) == PSY.DiscreteControlledBranchStatus.CLOSED
end

function _populate_ybus_branch_vector!(
    vec::Vector{T},
    sys::PSY.System,
) where {T <: PSY.ACTransmission}
    iter = PSY.get_components(T, sys)
    sizehint!(vec, length(iter))
    for br in iter
        _in_ybus(br) && push!(vec, br)
    end
    return
end

function _get_ybus_two_terminal_ac_branches(sys::PSY.System)::YbusACBranches
    branches = YbusACBranches(
        Vector{PSY.Line}(),
        Vector{PSY.GenericArcImpedance}(),
        Vector{PSY.TwoWindingTransformer}(),
        Vector{PSY.DiscreteControlledACBranch}(),
        Vector{PSY.SeriesCompensator}(),
    )
    _populate_ybus_branch_vector!(branches.lines, sys)
    _populate_ybus_branch_vector!(branches.generic_arc_impedances, sys)
    _populate_ybus_branch_vector!(branches.two_winding_transformers, sys)
    _populate_ybus_branch_vector!(branches.breaker_switches, sys)
    _populate_ybus_branch_vector!(branches.series_compensators, sys)
    return branches
end

function _foreach_ybus_branch(
    f::F,
    branches::YbusACBranches,
) where {F <: Function}
    ix = _foreach_typed_branches(f, branches.lines, 0)
    ix = _foreach_typed_branches(f, branches.generic_arc_impedances, ix)
    ix = _foreach_typed_branches(f, branches.two_winding_transformers, ix)
    ix = _foreach_typed_branches(f, branches.breaker_switches, ix)
    ix = _foreach_typed_branches(f, branches.series_compensators, ix)
    return ix
end

function _foreach_typed_branches(
    f::F,
    vec::Vector{T},
    offset::Int,
) where {F <: Function, T <: PSY.ACTransmission}
    for (i, br) in enumerate(vec)
        f(br, offset + i)
    end
    return offset + length(vec)
end
