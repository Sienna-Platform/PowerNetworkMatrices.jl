# Slack distribution & reference-bus conventions

A PTDF answers "if I inject one unit of power at bus *j*, how much flows on branch
*i*?" But power is conserved: an injection somewhere must be balanced by a
withdrawal somewhere else. **Where that balancing withdrawal goes is the slack
convention**, and it changes the PTDF. This page explains single versus
distributed slack, the role of the reference bus, and how the choice shows up in
the sensitivities.

For the mechanics of configuring a distributed slack, see the
[distributed-slack how-to](../how_to_guides/generated_distributed_slack.md).

## Why a reference is needed at all

The DC power flow solves `θ = ABA⁻¹ · p` for bus angles from injections. `ABA` is
a grounded graph Laplacian: without grounding it is singular (angles are only
defined up to a constant, and injections must sum to zero). Grounding it means
designating a **reference bus** whose angle is fixed and whose row/column is
removed from the system. That reference bus is also, by construction, the bus that
absorbs the network's power imbalance — the **single slack**.

So every PTDF is implicitly *relative to a slack*. `PTDF[i, j]` is the flow on
branch *i* caused by injecting at bus *j* **and withdrawing at the slack**. There
is no such thing as a slack-free PTDF.

## Single slack (the default)

By default `dist_slack` is empty, and the matrix uses a **single reference bus**
as the slack. This is the standard convention and is what you want when the model
has one designated slack generator absorbing imbalance.

The reference bus is excluded from the solve, so its own column in the PTDF is
zero — injecting at the slack and withdrawing at the slack moves nothing.

**Effect of the reference-bus choice.** Moving the slack to a different bus shifts
each PTDF row by a constant: every sensitivity is measured against a different
balancing point. The choice is therefore not arbitrary for interpreting a single
`PTDF[i, j]` value. It does *not*, however, change physically meaningful flows: if
the injections you apply already sum to zero (a real transfer, generation minus
load), the resulting branch flows are independent of which bus was chosen as
slack. LODF values are likewise invariant to the slack choice.

## Distributed slack

A single slack is an idealization — real imbalance is picked up by many
generators according to participation factors, not one bus. A **distributed
slack** spreads the balancing withdrawal across several buses by weight. Instead
of the whole compensating withdrawal landing on one reference bus, each
participating bus *k* absorbs a share `wₖ / Σ w`.

Concretely, the distributed-slack PTDF is the single-slack PTDF with each row's
weighted average subtracted:

```
PTDF_dist[i, j] = PTDF[i, j] − Σₖ (wₖ / Σ w) · PTDF[i, k]
```

so the sensitivities are now measured relative to the *weighted set* of slack
buses rather than one. The weights are normalized internally, so only their
ratios matter.

Two structural requirements come with distributed slack (enforced by the
constructors):

  - There must be exactly **one reference bus** in the system when a non-empty
    `dist_slack` is supplied — distributed slack redistributes the imbalance, but
    the grounding of `ABA` is still a single reference.
  - The weight vector must cover **every bus** (length equal to the bus count);
    buses that do not participate simply carry weight zero.

## The type differs by matrix

The way you pass the weights depends on the matrix, and this catches people out:

| Matrix                                       | `dist_slack` type                                         |
|:-------------------------------------------- |:--------------------------------------------------------- |
| [`PTDF`](@ref), [`VirtualPTDF`](@ref)        | `Dict{Int, Float64}` — bus number → weight                |
| [`VirtualLODF`](@ref), [`VirtualMODF`](@ref) | `Vector{Float64}` — one weight per bus, in bus-axis order |

For [`PTDF`](@ref)/[`VirtualPTDF`](@ref) you supply a **dictionary keyed by bus
number**, which is convenient because you name only the participating buses;
internally it is normalized and expanded to a per-bus vector. For
[`VirtualLODF`](@ref)/[`VirtualMODF`](@ref) you supply the **positional vector**
directly. In both cases the default is empty,
meaning single-reference-bus slack.

Why the LODF/MODF forms take a raw vector rather than a dict is an interface
detail, not a semantic difference — the underlying meaning (weighted balancing
withdrawal) is identical. Just match the type to the matrix you are building.

## Choosing a convention

  - Use the **default single slack** for most analyses, and for validating against
    references that assume one slack bus.
  - Use a **distributed slack** when you want sensitivities consistent with how
    generation actually rebalances — for economic dispatch, participation-factor
    studies, or comparison with an AC solution whose losses and response are spread
    across machines.

The distributed slack does not make the DC model more or less exact; it changes
the *question* the PTDF answers, from "balanced at one bus" to "balanced across a
weighted fleet."
