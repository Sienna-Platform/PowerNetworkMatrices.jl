# The loop's entry point. Separated from the spec file so the struct can precede
# `ReductionContainer` in the include order while this dispatch, which needs `Ybus`, follows it.

# Applies one primitive, or returns `ybus` completely untouched (no `_apply_reduction` call at
# all) when `nrd` eliminates no bus, so identity comparison is a sound convergence test for the
# caller. The no-op check must happen on `nrd` *before* calling `_apply_reduction`, not on its
# result afterward: `_apply_reduction` mutates the input ybus's shared `NetworkReductionData` in
# place (it stamps the applied spec into `reductions`, unions into `irreducible_buses`, and
# overwrites `boundary_bus_to_removed_arcs`) and returns a new `Ybus` that shares that same
# mutated `nr`. Discarding a post-hoc "no progress" result would still leave those mutations
# applied to state the caller keeps using on every subsequent round.
#
# `isempty(nrd)` is not the right no-op test: `RadialReduction`'s and `DegreeTwoReduction`'s
# `get_reduction` unconditionally stamp their own spec into `nrd.reductions` (so
# `isempty(nrd.reductions)` is always `false`), and `RadialReduction` also always populates
# `bus_reduction_map` with one entry per surviving bus even when none of them has anything
# reduced onto it. `_apply_bus_reductions!` derives every bus it removes solely from
# `nrd.reverse_bus_search_map`'s keys and `nrd.removed_buses`, so checking those two fields
# directly is both necessary and sufficient for "this round eliminates no bus."
function _apply_primitive_reduction(
    ybus::Ybus,
    sys::PSY.System,
    reduction::NetworkReduction,
)
    nrd = get_reduction(ybus, sys, reduction)
    isempty(nrd.removed_buses) && isempty(nrd.reverse_bus_search_map) && return ybus
    return _apply_reduction(ybus, nrd)
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
        # `max_rounds` bounds *productive* rounds; the terminal no-op round that confirms
        # convergence is not one of them, so the bound is on `rounds - 1`, not `rounds`.
        if rounds - 1 > get_max_rounds(reduction)
            error(
                "IterativeTopologyReduction did not converge within " *
                "$(get_max_rounds(reduction)) productive rounds; each round eliminates at " *
                "least one bus or ends the loop, so this indicates either a topology deeper " *
                "than `max_rounds`, a `max_rounds` set too low, or a defect in a primitive " *
                "reduction.",
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
