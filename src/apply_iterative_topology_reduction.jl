# The loop's entry point. Separated from the spec file so the struct can precede
# `ReductionContainer` in the include order while this dispatch, which needs `Ybus`, follows it.

# Applies one primitive without the duplicate-application guard, which is a user-input check that
# the loop deliberately bypasses. Returns the same `ybus` object, unchanged, when the primitive
# eliminates nothing, so identity comparison is a sound convergence test for the caller.
#
# `isempty(nrd)` alone is not that signal here: `RadialReduction`'s and `DegreeTwoReduction`'s
# `get_reduction` unconditionally stamp their own spec into `nrd.reductions` (so
# `isempty(nrd.reductions)` is always `false`), and `RadialReduction` also always populates
# `bus_reduction_map` with one entry per surviving bus even when none of them has anything
# reduced onto it. Both make `isempty(nrd)` `false` on a genuine no-op round, unlike
# `ZeroImpedanceBranchReduction`, whose `get_reduction` returns a bare default
# `NetworkReductionData` when it finds nothing. The bus-count check below is what actually
# distinguishes a productive round from a no-op one for the two primitives this loop alternates.
function _apply_primitive_reduction(
    ybus::Ybus,
    sys::PSY.System,
    reduction::NetworkReduction,
)
    nrd = get_reduction(ybus, sys, reduction)
    isempty(nrd) && return ybus
    reduced = _apply_reduction(ybus, nrd)
    length(get_bus_axis(reduced)) == length(get_bus_axis(ybus)) && return ybus
    return reduced
end

"""
    build_reduced_ybus(ybus::Ybus, sys::PSY.System, reduction::IterativeTopologyReduction) -> Ybus

Alternate [`RadialReduction`](@ref) and [`DegreeTwoReduction`](@ref), each applied via
[`get_reduction`](@ref)/`_apply_reduction` directly rather than through `build_reduced_ybus`, until
one full round eliminates no bus. A composite reduction cannot itself satisfy the single-step
`get_reduction` contract, since it produces several [`NetworkReductionData`](@ref) applied in
sequence; this method is therefore the reduction's only entry point.
"""
function build_reduced_ybus(
    ybus::Ybus,
    sys::PSY.System,
    reduction::IterativeTopologyReduction,
)
    validate_reduction_type(reduction, get_reductions(get_network_reduction_data(ybus)))
    radial = get_radial_reduction(reduction)
    degree_two = get_degree_two_reduction(reduction)
    rounds = 0
    while true
        rounds += 1
        if rounds > get_max_rounds(reduction)
            error(
                "IterativeTopologyReduction did not converge within " *
                "$(get_max_rounds(reduction)) rounds; each round eliminates at least one bus " *
                "or ends the loop, so this indicates a defect in a primitive reduction.",
            )
        end
        before = ybus
        ybus = _apply_primitive_reduction(ybus, sys, radial)
        ybus = _apply_primitive_reduction(ybus, sys, degree_two)
        ybus === before && break
    end
    add_reduction!(
        get_reductions(get_network_reduction_data(ybus)),
        ReductionContainer(; iterative_topology_reduction = reduction),
    )
    # The last round is always a no-op (it is what ends the loop), so it does not correspond to a
    # productive elimination pass.
    @info "IterativeTopologyReduction converged after $(rounds - 1) productive round(s) " *
          "(round $rounds was a no-op that confirmed convergence)."
    return ybus
end
