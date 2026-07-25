# SDD Plan B — PowerNetworkMatrices: green tests on the transformer refactor

Parent analysis: `.claude/plans/2026-07-24-transformer-refactor-distribution.md`
Repo: `/Users/jdlara/cache/psy6/PowerNetworkMatrices.jl`
Goal: PNM test suite passes against PSY `psy6` @ `d19f3244f`, with the transformer work and
the psy6 ZIBR/units work both intact.

Independent of Plan A (PSB) — PSB as a dependency is already correct on `origin/psy6`.

## Context

Two branches must become one:

  - `origin/psy6` — current mainline. ZIBR rework, susceptance-cancellation fix, IS4 stateless
    units, `PSY.SU` threading. **Broken against PSY psy6**: references all five deleted
    transformer types across 7 src and 11 test files.
  - `origin/transformer-refactor` — 5 commits, API-correct against merged PSY (uses
    `TwoWindingTransformer`, `ThreeWindingTransformer`, `TransformerCircuit`, `get_circuit(s)`,
    `is_phase_shifting`, both shunt-location enums). Forked at `8e9579c`, ~20 psy6 commits
    stale. **Owns the functions PSY deleted**: `get_series_susceptance(s)`,
    `get_series_admittance(s)`.

Overlap is only three files — `Ybus.jl`, `apply_zero_impedance_reduction.jl`,
`BA_ABA_matrices.jl` — but `Ybus.jl` was substantially rewritten on both sides. Everything
else is disjoint (see parent analysis §Step 4 for the file-by-file split).

Local `transformer-refactor` does not exist yet:
`git checkout -b transformer-refactor origin/transformer-refactor`.

## Global Constraints

Bind every task. A reviewer treats a violation as a defect.

 1. **Always `julia --project=test`.** Once per clone:
    `julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`
 2. **Subagents never run the full suite.** It is multi-minute and builds PSB systems.
    Implementers use ReTest name filters only:
    `julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("Ybus")'`
    Cap ~4 min, smallest fixture, N=1. The **controller** backgrounds the full suite and feeds
    results back. A subagent that reports "full suite passes" without the controller having run
    it is reporting something it cannot know — treat as a failed report.
 3. **Compile-check after every file edit:**
    `julia --project=test -e 'using PowerNetworkMatrices'`. Fix compile errors immediately
    rather than letting them cascade. PNM has 40+ source files; a broken include order
    surfaces as an unrelated error three files later.
 4. **Aqua runs at module load**, not inside a testset — `test_ambiguities`,
    `test_undefined_exports`, `test_unbound_args`, `test_stale_deps`, `test_deps_compat`. A
    new method ambiguity or an export of a name that no longer exists fails at *include* time,
    before any test runs. Renaming `ThreeWindingTransformerWinding` →
    `ThreeWindingTransformerCircuit` touches exports; check Aqua explicitly.
 5. **Never export or re-export any `KLUWrapper` symbol.**
 6. **Exports live only in the main module file** (`src/PowerNetworkMatrices.jl`).
 7. **No `isa` / `<:Type` runtime checks**, including tests — use multiple dispatch. This
    refactor is a magnet for violations: the deleted types were often dispatched on, and the
    lazy port is an `isa` chain. Use methods on `TransformerCircuit` / the transformer types.
 8. **No big `Union{...}` aliases enumerating concrete types** to drive dispatch. If several
    types share behavior, use an abstract supertype or a trait. A 2-member `Union` in one
    signature is fine; a named alias listing many concretes is not.
 9. **No ternaries**; `iszero(x)` not `x == 0`; explicit `function … end` with explicit
    `return` for any non-trivial body.
10. **Terse comments** — default to none. Keep only a comment documenting a non-obvious WHY
    (a hidden constraint, a subtle invariant, a bug workaround). No comments narrating the
    refactor, and no PR/issue numbers.
11. **Aggregators return system base.** `BranchesSeries`/`BranchesParallel`/`EquivalentBranch`
    equivalents are already in system base; consumers must not re-apply `PSY.SU`.
12. **Do not "fix" legitimate Ybus asymmetry** — see Task 4.
13. **Do not normalize the deliberate mixed bases in Ybus assembly** — see Task 3.
14. **Run the formatter** before any task is done:
    `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`
    It also walks `docs/` with `format_markdown=true`; fenced blocks need a language label.
15. **Reduction fixtures:** `c_sys5`/`c_sys14` reduce nothing — a passing reduction test on
    them proves nothing. Use `case11_network_reductions`, or matpower RTS/case24.
16. **Do not patch PSY.** It is upstream and frozen here. A genuine PSY bug is BLOCKED.
17. **No `@test_logs`** to assert warnings (MethodError on failure) — use a custom
    `AbstractLogger`.

## Tasks

### Task 1 — Perform the merge; resolve to a compiling state

Integration judgment. Most capable model. Single agent, no parallelism.

`git checkout -b transformer-refactor origin/transformer-refactor && git merge origin/psy6`

Resolve conflicts in the three overlapping files. The **only** exit criterion is that
`using PowerNetworkMatrices` succeeds and Aqua passes at include time. Tests are expected to
fail; that is fine and in scope for later tasks.

Resolution rule, stated once because it governs every hunk: **psy6 owns reduction and units
behavior; transformer-refactor owns transformer representation.** Where a hunk changes how
zero-impedance/series/parallel reduction works or which unit system a getter is called with,
psy6's version wins. Where a hunk changes how a transformer's electrical data is reached
(`get_circuit`, `get_circuits`, circuit-level `r`/`x`/`tap`/`α`), transformer-refactor's
version wins. Where a single hunk does both, keep psy6's reduction logic and rewrite its data
access through the circuit API.

Do not resolve by taking one side wholesale. Do not delete a psy6 test to make a merge clean.

Report: every conflicted hunk, which side won, and why — this report is the input to Tasks 3
and 4 and to the reviewer.

### Task 2 — Retire the stateful units API

Mechanical, but numerically load-bearing. Standard model.

`transformer-refactor` still calls the units API that psy6 deleted in the IS4 migration:

  - `src/Ybus.jl:898` — `PSY.get_units_base(sys)`
  - `src/Ybus.jl:901` — `PSY.set_units_base_system!(sys, "SYSTEM_BASE")`
  - `test/test_ybus.jl:100` — `PSY.set_units_base_system!(sys, "NATURAL_UNITS")`

(Line numbers are pre-merge; re-locate them after Task 1.)

Replace with explicit per-call unit arguments, following what psy6 already did in `7134b8f`
("Drop stateful units block in Ybus for IS4 stateless units") and `87dbfaf` ("Pass PSY.SU to
get_r/get_x"). Read both commits first and match their idiom rather than inventing one.

**Constraint 13 applies directly here.** Ybus assembly deliberately mixes bases:
`GenericArcImpedance` reads `PSY.DU` (correct — its impedance is per-unitized on device base
before conversion) while shunts and admittances elsewhere read `PSY.SU`. The block being
removed wrapped code that relied on ambient state; converting it to explicit args is exactly
where someone "tidies" both to one unit system and silently changes the numbers. Preserve the
existing per-call unit choice for every getter. Add a numeric regression test asserting a
`GenericArcImpedance` Ybus entry, so the distinction is pinned by a test rather than a comment.

### Task 3 — Reconcile the two susceptance/admittance changes

Numerics. Most capable model. Depends on Task 1's report.

Both branches independently changed the same numerics:

  - psy6: `35edc0d` "fix susceptance cancellation", `f4275ca` "ZIBR: compute series admittance
    directly", `88005e1` "gate zero-impedance merge on r == 0"
  - transformer-refactor: `be46323` "fix transformer winding susceptance calculation", plus the
    port of `get_series_susceptance` in from PSY —
    `BranchAdmittance.jl:53,64`, `BranchesParallel.jl:92`, `BranchesSeries.jl:107`,
    `ThreeWindingTransformerCircuit.jl:78`

Read all five commits and the merged result. Establish, with a test for each, that:

 1. the psy6 susceptance-cancellation fix still holds for non-transformer branches;
 2. the transformer winding susceptance fix still holds, now via the circuit API;
 3. the ZIBR direct-admittance path is unchanged by the transformer port.

**Neither branch's resolution is presumed a superset of the other.** If the two fixes are
genuinely incompatible on some input, say so and report BLOCKED rather than picking one.

Note the documented hazard this sits next to: branches with `r>0, x=0` have `b=0` and vanish
from BA, so the susceptance graph fragments more than the admittance graph → singular ABA.
Zero-impedance reduction must resolve *both* endpoints to union-find roots before merging.

### Task 4 — Re-key the phase-shifting predicate, and check whether its meaning widened

Design judgment. Most capable model.

PNM's documented rule: the asymmetric Ybus entries `Y[i,j] = −y/t*`, `Y[j,i] = −y/t` for a
phase-shifting transformer are **correct and must not be "fixed"**. That rule was written
against `PSY.PhaseShiftingTransformer`, which no longer exists.

The replacement is `PSY.is_phase_shifting(circuit)`, defined upstream as:

    !iszero(get_α(w)) && return true
    return get_control_objective(w) in _PHASE_SHIFT_OBJECTIVES

where the objectives are `ACTIVE_POWER_FLOW`, `ACTIVE_POWER_FLOW_DISABLED`,
`ASYMMETRIC_ACTIVE_POWER_FLOW`, `ASYMMETRIC_ACTIVE_POWER_FLOW_DISABLED`.

**The second arm is new behavior.** A circuit with `α = 0` but an active-power control
objective is now phase-shifting, where the old type check would have missed it. Two
consequences to establish, with a test for each:

 1. Ybus entries for such a circuit become asymmetric where they previously were symmetric.
 2. `_assert_not_phase_shifting` (`common.jl:520-537`) now rejects contingencies on circuits it
    previously admitted. The old code needed a special case because
    `PhaseShiftingTransformer3W` was not a subtype of `PhaseShiftingTransformer`; with a
    circuit-level predicate that special case should collapse — verify it does, and delete it
    if so rather than leaving it dead.

Also: orientation knowledge is scattered across three sites (Ybus assembly sign-flip at
`Ybus.jl:~487-513`, `get_ptdf_orientation_sign`, POM's area_interchange). When touching signs,
check all three. Do not attempt the "normalize orientation at construction" refactor here —
it is a separate audit candidate and out of scope.

If the widening looks unintended rather than a deliberate upstream decision, report
DONE_WITH_CONCERNS and say so; the controller escalates to the human. Do not narrow the
predicate locally to preserve old behavior.

### Task 5 — Per-circuit availability through the reduction and contingency paths

Design judgment. Most capable model.

Availability moved from one flag per transformer to one flag per circuit:
`get_available(t) = any(get_available, get_circuits(t))`, and `set_available!(t, val)`
cascades to all circuits.

This creates a state PNM never had to represent: **a 3-winding transformer with one or two
circuits out and the rest energized.** Previously availability was all-or-nothing per device.

Establish behavior for each, with tests:

 1. All three circuits out — the documented path. Star bus isolates → singular ABA, handled by
    the pinv islanding route. Confirm still true. Note AppleAccelerate LU silently factorizes
    singular matrices (garbage results) where KLU throws — prefer KLU wherever singularity is
    possible.
 2. **One circuit out, two in** — new. What does Ybus assembly do? What does
    `ThreeWindingTransformerCircuit` decomposition do? Does the reduction map keep the
    surviving circuits and drop only the dead one?
 3. A 2W transformer whose single circuit is out — should behave exactly as the old
    unavailable-transformer case.

Also verify the MODF/reduction consistency invariant still holds: outaged and monitored
branches must survive the reduction or queries silently return the base-case row. Universal
survive-check is membership in `keys(get_bus_reduction_map(nrd))`.

### Task 6 — Reconcile the stale test files

Mechanical, high volume. Standard model. May be split into two dispatches if the diff is large.

`origin/psy6` has 11 test files naming deleted PSY types:
`test_arc_types_and_reductions.jl` (4), `test_branch_admittance.jl` (3),
`test_equivalent_getters.jl` (5), `test_modf_lodf_reductions.jl` (8),
`test_modf_reduction_consistency.jl` (5), `test_virtual_modf.jl` (2),
`test_ybus_contingencies.jl` (2), `test_ybus_matpower.jl` (1), `test_ybus_reductions.jl` (3),
`test_BA_ABA_matrix.jl` (1), `performance/performance_test.jl` (1).

`transformer-refactor` already rewrote 19 test files, including several of these
(`test_branch_admittance.jl` +504, `test_ybus_reductions.jl` +210, new `test_2d_reduction.jl`
+166). After Task 1's merge, some are already correct.

For each remaining reference: port it to the new API. **Do not delete a test to make it
pass** — if a test covered behavior that no longer exists (e.g. a `TapTransformer`-specific
dispatch), the equivalent new-API behavior still needs coverage. If a test is genuinely
obsolete, say which behavior it covered and where that behavior now lives; deleting without
that accounting is a failed task.

Both branches' test sets must pass together, not one at the expense of the other.

### Task 7 — Full-suite green

Controller-run, iterative. Background the suite, feed failures back as scoped fix waves,
one wave per coherent failure cluster (not one agent per failure).

Includes the PF-suite smoke check from PNM's downstream contract: after a PNM change, run the
PowerFlows suite and at least the POM network-constructor tests. **Expect PowerFlows to fail
for unrelated reasons** — `PowerFlows.jl@psy6` still names deleted PSY types in
`dcpf_loss_injection.jl` (5), `post_processing.jl` (2), `psse_export.jl` (16) and will not
precompile against PSY psy6. Record that as a known-external failure; do not fix PF here.

### Task 8 — Update `.claude/CLAUDE.md`

Mechanical. Cheap model. See the separate CLAUDE.md work — the controller handles this
directly rather than dispatching, since it spans repos.

### Task 9 — Rebase `precompile-load`

Mechanical-to-standard. After Task 7 is green and `transformer-refactor` is merged to `psy6`.

`precompile-load` adds a PrecompileTools workload over the power-flow matrix path, which this
plan rewrote. Expect the workload body to need updating for `TransformerCircuit` construction
— a workload that fails to precompile breaks every downstream load, so this is not cosmetic.

## Task dependency graph

```text
Task 1 (merge, compiles)
  ├─→ Task 2 (units)        ─┐
  ├─→ Task 3 (susceptance)  ─┤
  ├─→ Task 4 (phase shift)  ─┼─→ Task 7 (full suite green) ─→ Task 9 (precompile-load)
  ├─→ Task 5 (availability) ─┤
  └─→ Task 6 (stale tests)  ─┘
Task 8 (docs) — anytime after Task 1
```

Tasks 2-6 all touch `Ybus.jl` and its neighbours. **Run them sequentially, not in parallel**
— SDD's no-parallel-implementers rule is not optional here, it is the whole reason this merge
is delicate.

## Test strategy

| Who         | What                                                  | When                                   |
|:----------- |:----------------------------------------------------- |:-------------------------------------- |
| Implementer | `run_tests("<filter>")`, smallest fixture, ~4 min cap | per task                               |
| Implementer | `using PowerNetworkMatrices` (also exercises Aqua)    | after every file edit                  |
| Controller  | full `test/runtests.jl`, backgrounded                 | Task 1 exit, after each of 2-6, Task 7 |
| Controller  | PowerFlows suite + POM network-constructor tests      | Task 7 only                            |

Verify testset registration with `run_tests(dry=true)` before trusting a filter that returns
zero tests — a typo'd filter passes vacuously.

## Open Decision (controller must resolve before Task 1)

**Commits.** SDD's review packages diff `BASE..HEAD` and its ledger records commit ranges as
the compaction-recovery map. The standing rule here is never to commit without explicit
direction. **Plan B cannot avoid commits**: Task 1 *is* a merge commit, and a conflict
resolution that is not committed cannot be handed to a reviewer as a range.

Recommendation: authorize commits on the `transformer-refactor` branch only, for this plan,
with no push and no merge to `psy6` until the human approves. That keeps `psy6` untouched,
keeps the review mechanism working, and stays inside the spirit of the rule — the branch is
scratch until it is merged. Needs explicit sign-off before Task 1.
