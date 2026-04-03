struct YbusACBranches
    lines::Vector{PSY.Line}
    monitored_lines::Vector{PSY.MonitoredLine}
    generic_arc_impedances::Vector{PSY.GenericArcImpedance}
    tap_transformers::Vector{PSY.TapTransformer}
    phase_shifting_transformers::Vector{PSY.PhaseShiftingTransformer}
    transformer2w::Vector{PSY.Transformer2W}
    dynamic_branches::Vector{PSY.DynamicBranch}
    breaker_switches::Vector{PSY.DiscreteControlledACBranch}
end

function Base.length(b::YbusACBranches)::Int
    return length(b.lines) +
           length(b.monitored_lines) +
           length(b.generic_arc_impedances) +
           length(b.tap_transformers) +
           length(b.phase_shifting_transformers) +
           length(b.transformer2w) +
           length(b.dynamic_branches) +
           length(b.breaker_switches)
end

function _populate_ybus_branch_vector!(
    vec::Vector{T},
    sys::PSY.System,
) where {T <: PSY.ACTransmission}
    iter = PSY.get_components(T, sys)
    sizehint!(vec, size(iter))
    for br in iter
        PSY.get_available(br) && push!(vec, br)
    end
    return
end

function _get_ybus_ac_branches(sys::PSY.System)::YbusACBranches
    branches = YbusACBranches(
        Vector{PSY.Line}(),
        Vector{PSY.MonitoredLine}(),
        Vector{PSY.GenericArcImpedance}(),
        Vector{PSY.TapTransformer}(),
        Vector{PSY.PhaseShiftingTransformer}(),
        Vector{PSY.Transformer2W}(),
        Vector{PSY.DynamicBranch}(),
        Vector{PSY.DiscreteControlledACBranch}(),
    )
    _populate_ybus_branch_vector!(branches.lines, sys)
    _populate_ybus_branch_vector!(branches.monitored_lines, sys)
    _populate_ybus_branch_vector!(branches.generic_arc_impedances, sys)
    _populate_ybus_branch_vector!(branches.tap_transformers, sys)
    _populate_ybus_branch_vector!(branches.phase_shifting_transformers, sys)
    _populate_ybus_branch_vector!(branches.transformer2w, sys)
    _populate_ybus_branch_vector!(branches.dynamic_branches, sys)
    return branches
end

function _foreach_ybus_branch(
    f::F,
    branches::YbusACBranches,
) where {F <: Function}
    ix = _foreach_typed_branches(f, branches.lines, 0)
    ix = _foreach_typed_branches(f, branches.monitored_lines, ix)
    ix = _foreach_typed_branches(f, branches.generic_arc_impedances, ix)
    ix = _foreach_typed_branches(f, branches.tap_transformers, ix)
    ix = _foreach_typed_branches(f, branches.phase_shifting_transformers, ix)
    ix = _foreach_typed_branches(f, branches.transformer2w, ix)
    ix = _foreach_typed_branches(f, branches.dynamic_branches, ix)
    ix = _foreach_typed_branches(f, branches.breaker_switches, ix)
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
