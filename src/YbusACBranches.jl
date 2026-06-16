struct ICDCorrectionMap
    map_2w::Dict{Base.UUID, Float64}
    map_3w::Dict{Tuple{Base.UUID, Int}, Float64}
end

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
    sizehint!(vec, length(iter))
    for br in iter
        PSY.get_available(br) && push!(vec, br)
    end
    return
end

_get_correction(::Nothing, ::PSY.ACTransmission) = 1.0
_get_correction(::Nothing, ::PSY.DynamicBranch) = 1.0
_get_correction(::ICDCorrectionMap, ::PSY.ACTransmission) = 1.0
_get_correction(map::ICDCorrectionMap, br::PSY.TwoWindingTransformer) =
    get(map.map_2w, IS.get_uuid(br), 1.0)
_get_correction(map::ICDCorrectionMap, br::PSY.DynamicBranch) =
    get(map.map_2w, IS.get_uuid(br.branch), 1.0)
function _populate_ybus_branch_vector!(
    vec::Vector{T},
    sys::PSY.System,
) where {T <: PSY.DiscreteControlledACBranch}
    iter = PSY.get_components(T, sys)
    sizehint!(vec, length(iter))
    for br in iter
        PSY.get_available(br) &&
            PSY.get_branch_status(br) == PSY.DiscreteControlledBranchStatus.CLOSED &&
            push!(vec, br)
    end
    return
end

function _get_ybus_two_terminal_ac_branches(sys::PSY.System)::YbusACBranches
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
    _populate_ybus_branch_vector!(branches.breaker_switches, sys)
    return branches
end

function _foreach_ybus_branch(
    f::F,
    branches::YbusACBranches,
    icd_map::Union{Nothing, ICDCorrectionMap},
) where {F <: Function}
    ix = _foreach_typed_branches(f, branches.lines, icd_map, 0)
    ix = _foreach_typed_branches(f, branches.monitored_lines, icd_map, ix)
    ix = _foreach_typed_branches(f, branches.generic_arc_impedances, icd_map, ix)
    ix = _foreach_typed_branches(f, branches.tap_transformers, icd_map, ix)
    ix = _foreach_typed_branches(f, branches.phase_shifting_transformers, icd_map, ix)
    ix = _foreach_typed_branches(f, branches.transformer2w, icd_map, ix)
    ix = _foreach_typed_branches(f, branches.dynamic_branches, icd_map, ix)
    ix = _foreach_typed_branches(f, branches.breaker_switches, icd_map, ix)
    return ix
end

function _foreach_typed_branches(
    f::F,
    vec::Vector{T},
    icd_map::Union{Nothing, ICDCorrectionMap},
    offset::Int,
) where {F <: Function, T <: PSY.ACTransmission}
    for (i, br) in enumerate(vec)
        f(br, offset + i, _get_correction(icd_map, br))
    end
    return offset + length(vec)
end
