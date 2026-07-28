# RESOLVED — fixed on jd/transformer_refactor (uncommitted work, 2026-07-27): identity-threaded member deltas replaced the susceptance-value scan. Kept for historical context only; do NOT file.
# DRAFT — PNM follow-up issue (do not file without user approval)

**Title:** `_compute_parallel_partial_ybus_delta` resolves the tripped parallel member by susceptance value, not identity

## Description

`_classify_branch_modification` computes the outaged member's susceptance and passes only that scalar down; `_compute_parallel_partial_ybus_delta` (`src/network_modification.jl:34-47`) then scans `bp.branches` for the **first** member whose `get_series_susceptance` matches within `atol = 1e-10` and returns that member's π-model as the Ybus delta.

When two co-arc members have equal series susceptance, the lookup can pick the wrong member. Example: a `Line` with `x = 0.1` in parallel with a `TwoWindingTransformer` (`tap = 1.0, x = 0.1, α = 0.15`) — both give `b = 10.0`. Tripping the Line can return the transformer's **asymmetric** 2×2 (ΔY₁₂ ≠ ΔY₂₁) as the delta for a symmetric line outage. The DC susceptance delta stays correct; the AC contingency Ybus is silently wrong.

Same root cause, lower impact: `compute_parallel_multiplier` (`src/BranchesParallel.jl`) matches members by **name**; PSY names are unique only within a concrete type, so a `Line` "T1" and a transformer "T1" in one `MixedBranchesParallel` double-count.

## Context

The mechanism pre-dates the #305 fix (two co-arc transformers with equal `tap·x` collide identically), but #305 widens exposure: phase-shifting circuits now legitimately live in parallel groups, so an equal-susceptance collision can pair a symmetric member with an asymmetric one.

## Suggested fix

Thread the tripped component itself down from `_classify_branch_modification` instead of `delta_b`, and resolve the member by identity (`===`). Alternatively (or additionally), assert uniqueness of member susceptances at group construction so the collision is loud.
