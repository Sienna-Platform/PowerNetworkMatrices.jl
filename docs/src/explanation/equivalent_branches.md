# Equivalent representation of reduced branches

When a network reduction collapses a group of branches into one, the reduced
network must carry a single **equivalent branch** in place of the group. This
page explains how the equivalent's electrical parameters and ratings are formed,
and — the subtle part — **why the rating policies differ** while the impedance
aggregation does not.

The aggregated-branch types themselves (`BranchesParallel`,
`MixedBranchesParallel`, `BranchesSeries`, the internal
`ThreeWindingTransformerWinding`, and the resulting `EquivalentBranch`) are
documented by their docstrings in the
[full public API](../reference/public.md) under the internal (non-exported) symbols.

## Two kinds of group

Reductions produce two topological groupings:

  - **Parallel groups** (`AbstractBranchesParallel`): several branches sharing the
    same pair of buses. Physically they are alternative paths between the same two
    nodes.
  - **Series chains** (`BranchesSeries`): a run of branches through intermediate
    degree-two buses that carry no injection, so the chain behaves as a single
    branch between its endpoints.

A three-winding transformer contributes its own wye-to-star structure, handled
by the internal `ThreeWindingTransformerWinding`.

## Impedance aggregation is physics — one answer

The equivalent electrical parameters are not chosen; they are *derived* from the
requirement that the reduced branch present the same terminal behavior as the
group it replaces. PNM does this by building the group's equivalent admittance
(`populate_equivalent_ybus!`) and reading the physical parameters back off it
(`get_equivalent_physical_branch_parameters`, `src/common.jl`), yielding an
`EquivalentBranch` with series `r`/`x`, shunt `g`/`b` at each end, tap,
and phase shift.

The combining rules follow directly:

  - **Parallel:** admittances add. The group's series susceptance is the sum of the
    members', `b = Σ bᵢ` — more parallel paths means a stronger (lower-impedance)
    connection.
  - **Series:** impedances add, so susceptances combine reciprocally,
    `b = 1 / Σ (1/bᵢ)` — a chain is weaker than its strongest link.

There is a single correct answer here because the electrical behavior is fixed by
Kirchhoff's laws.[^circuits] The equivalent is exact for the linear (DC) model.

## Rating aggregation is policy — several answers

A **rating** is not an electrical quantity the way impedance is. It is a limit
imposed by the study, and "the limit of the group" is a genuinely ambiguous
question whose answer depends on what the study is protecting against. That is
why PNM exposes several rating strategies for a parallel group rather than one
(`src/BranchesParallel.jl`):

  - **[`get_sum_of_max_rating`](@ref) — nominal capacity.** `Σ Sᵢ`, treating every
    circuit as independently loadable to its own thermal limit. This is the least
    conservative aggregate; it assumes flow can be steered freely across the group
    so that each circuit reaches its limit at once. It answers "how much could this
    corridor carry in the best case?"
  - **[`get_single_element_contingency_rating`](@ref) — N-1 security.** `Σ Sᵢ − maxᵢ Sᵢ`, the capacity that survives when the largest circuit in the group
    trips. It answers "what can I still rely on after losing one element?" For a
    group of one it is zero, correctly, because there is nothing left after the sole
    circuit trips.
  - **[`get_impedance_averaged_rating`](@ref) — realistic DC loading.** The
    susceptance-weighted average `Σ fᵢ Sᵢ` with `fᵢ = bᵢ / Σ bₖ`. This reflects how
    DC flow *actually* divides across a parallel group: current follows the path of
    least impedance, so the low-impedance (high-susceptance) circuit carries the
    larger share and reaches its limit first. Because flow cannot in fact be steered
    arbitrarily, the sum-of-max is optimistic and this weighted figure is closer to
    the binding constraint. It requires a finite, non-zero total susceptance and
    throws an `ArgumentError` otherwise.

The three are ordered from least to most physically constrained:
`sum_of_max ≥ impedance_averaged`, and the N-1 figure answers a different
(security) question entirely. Which one is "right" is a modeling decision, not a
computation the package can make for you — hence three named policies instead of
a silent default.

### Series chains and emergency ratings

Ratings propagate through the two group types differently, again for physical
reasons:

  - **Series chain rating** is the **weakest link**: `min` over the chain's
    members, because a series path can carry no more than its most limited segment
    (`get_equivalent_rating` on a `BranchesSeries`). When a member of the chain
    is itself a parallel group, it contributes its *single-element-contingency*
    rating — the conservative N-1 figure — rather than its optimistic sum.
  - **Emergency ratings** follow the same shapes: a parallel group sums the
    members' emergency ratings, a series chain takes the minimum (weakest link).
    Where a branch has no distinct emergency (`rating_b`) value, its normal rating
    is used as the post-contingency limit.

### Availability

A group is available only if all of its members are: parallel and series groups
both require every branch present and in service. Losing any one member makes the
equivalent unavailable, which keeps the reduced model consistent with outages of
the underlying branches.

## Why this separation matters

The clean split — **impedance is derived, rating is chosen** — is the key idea.
Impedance aggregation has a unique physical answer and PNM computes it once.
Rating aggregation encodes an operator's risk posture, so PNM refuses to pick for
you and instead names the policies (`sum_of_max`, `single_element_contingency`,
`impedance_averaged`) so a study selects the one matching its purpose: raw
capacity, N-1 security, or realistic DC loading.

## References

[^circuits]: Series and parallel combination of admittances/impedances is
    elementary circuit theory; see
    [Series and parallel circuits](https://en.wikipedia.org/wiki/Series_and_parallel_circuits).
