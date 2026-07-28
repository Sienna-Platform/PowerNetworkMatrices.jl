# DRAFT — PowerFlows.jl follow-up issue (do not file without user approval)

**Title:** DCLF loss injection errors on arcs where a lossy phase-shifting circuit is parallel to another branch

## Description

PNM issue Sienna-Platform/PowerNetworkMatrices.jl#305 fixed a bug where a branch sharing an arc with a phase-shifting circuit was silently dropped from `NetworkReductionData`. Such branches now correctly form a parallel group (`BranchesParallel{T}` / `MixedBranchesParallel`).

This exposes a representation limit in one PowerFlows feature: `_get_arc_branch_params` (`src/dcpf_loss_injection.jl:13-31`) resolves *every* arc through `PNM.arc_equivalent_branch`, which requires a single-π equivalent. A parallel group whose members are **lossy** and have **different phase-shift angles** has no such equivalent (`|Y12| ≠ |Y21|`), and PNM now raises a clear error naming the offending group:

```text
Equivalent parameters for the series or parallel reduction of branches results
in a real part of the phase shift angle. A lossy phase-shifting circuit in parallel
with other branches has no single-π equivalent. Offending group: <name>.
```

Consequence: enabling DCLF-style loss injection on a system containing a lossy Line∥PST arc (e.g. `case6470rte`) throws. Before the PNM fix it did not throw — but only because the NRD had silently dropped one of the branches, so the loss numbers were wrong anyway.

## Scope

  - Only the opt-in DCLF loss-injection path. The DC solve, AC solve, AC branch-flow post-processing (per-member, exact), and PTDF flows are unaffected.
  - Lossless (r = 0) shifted parallel groups still have an exact single-π equivalent and keep working.

## Suggested fix

Compute per-member losses instead of per-arc-equivalent losses for parallel groups: for each member `br`, get its own π parameters via `PNM.ybus_branch_entries(br)` (or `PNM.get_equivalent_r/x/tap/shift` on the member) and sum `P_loss` over the group. `PNM._is_phase_shifting(group)` (new in PNM #305 work) can gate the fallback, or the per-member path can simply be used for every `AbstractBranchesParallel` arc.
