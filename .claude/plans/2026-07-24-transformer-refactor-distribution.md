# Distributing the PSY transformer refactor to PSB and PNM

Upstream: `PowerSystems.jl` `origin/psy6` @ `d19f3244f` — "transformer refactor (#1714)".
Targets: `PowerSystemCaseBuilder.jl@psy6`, `PowerNetworkMatrices.jl@psy6`.

## 1. What the upstream change actually is

`TwoWindingTransformer` and `ThreeWindingTransformer` were **abstract supertypes**; they are
now **concrete structs**. Five concrete types were deleted outright:

| Deleted                                                       | Replacement               |
|:------------------------------------------------------------- |:------------------------- |
| `Transformer2W`, `TapTransformer`, `PhaseShiftingTransformer` | `TwoWindingTransformer`   |
| `Transformer3W`, `PhaseShiftingTransformer3W`                 | `ThreeWindingTransformer` |

All series electrical data moved onto a new `TransformerCircuit <: DeviceParameter`:
`available, arc, tap, α, winding_group_number, r, x, control_objective, regulated_bus_number, control_limits, controlled_quantity_limits, number_of_tap_positions, rating, rating_b, rating_c, active_power_flow, reactive_power_flow, base_power, base_voltage_primary, base_voltage_secondary, base_value`.

  - 2W owns one circuit (`get_circuit`); 3W owns three (`get_primary_circuit` /
    `_secondary_` / `_tertiary_`) plus `star_bus`. `get_circuits(t)` returns the tuple.
  - Transformer-level fields shrink to `name, magnetizing_shunt::Complex, shunt_location, services, ext, internal`.
  - **Availability is derived**: `get_available(t) = any(get_available, get_circuits(t))`.
    `set_available!(t, true)` re-energizes *all* circuits (PSS/E STAT semantics).
  - New enums: `TwoWindingTransformerShuntLocation` (`PRIMARY`/`SECONDARY`/`SPLIT`),
    `ThreeWindingTransformerShuntLocation` (`PRIMARY`/`STAR`).
  - New exports: `TransformerCircuit`, `get_circuits`, `is_phase_shifting`, `has_control`,
    `get_limits`, both shunt-location enums.
  - 3W pairwise PSS/E fields (`r_12`,`x_12`,`r_23`,`x_23`,`r_31`,`x_31`,
    `base_power_12/23/31`) are now `Union{Nothing,Float64}`, **all-or-none**.
    `base_power_13` → `base_power_31`; `get_base_power_1?_unitful` no longer exported.
  - `TransformerCircuit` has hand-written `IS.serialize`/`IS.deserialize` (arc as UUID,
    `base_value` never serialized) and is in `_CONTAINS_SHOULD_ENCODE`.
  - Units anchor: `TransformerCircuit.base_value`, propagated by `set_units_setting!` and by
    hand-written `set_circuit!`/`set_*_circuit!`.

**Deleted from PSY, now owned by PNM** (confirmed: zero `get_series_*` remain in PSY):
`get_series_susceptance`, `get_series_susceptances`, `get_series_admittance`,
`get_series_admittances`, and the winding-group `get_α(::TapTransformer/Transformer2W)` /
`get_α_primary/_secondary/_tertiary` derivations. `get_α` is now a plain circuit getter.

## 2. Branch state (verified 2026-07-24)

| Repo | Branch                        | State                                                                                          |
|:---- |:----------------------------- |:---------------------------------------------------------------------------------------------- |
| PSY  | `origin/psy6`                 | ✅ refactor merged @ `d19f3244f`                                                                |
| PSY  | local `psy6`                  | 17 behind origin; local checkout is on `transformer-refactor` (superseded by the squash-merge) |
| PSB  | `origin/psy6`                 | ✅ transformer work already merged (PR #200 `mb/transformer-refactor`) + parser consolidation   |
| PSB  | local `psy6`                  | 10 behind origin                                                                               |
| PNM  | `origin/psy6`                 | ❌ broken — references all 5 deleted PSY types                                                  |
| PNM  | `origin/transformer-refactor` | API-correct, but forked at `8e9579c`, ~20 psy6 commits stale                                   |
| PNM  | local `precompile-load`       | ahead of `psy6`, touches the same matrix path                                                  |
| PFFP | `psy6`                        | ✅ transformer work merged (PR #19); local fast-forwarded to `adf5cb1`                          |

## 3. Execution order

`PSY (done) → PFFP (done) → PSB → PNM`

### Step 1 — Sync local clones (no code change)

 1. PSY: `git checkout psy6 && git pull` — get `d19f3244f`. Keep `transformer-refactor`
    around only as reference; it is squash-merged, not an ancestor.
 2. PSB: `git checkout psy6 && git pull` — fast-forward 10 commits.
 3. PNM: `git fetch` (already done); stay off `precompile-load` for now.
 4. Verify PSY precompiles: `julia --project=test -e 'using PowerSystems'`.

### Step 2 — PowerFlowFileParser ✅ DONE

 5. Local `psy6` fast-forwarded 27 commits `ea022a6` → `adf5cb1`. `origin/psy6` already
    contains PR #19 `mb/transformer-refactor` plus the parser consolidation and substation
    parsing.
 6. No PSY-side risk here: PFFP depends only on `InfrastructureSystems`, not PowerSystems,
    and emits PowerModels dicts. Zero references to the deleted transformer types.
 7. Run PFFP tests to confirm the pull is sound (new: `test_parse_transformers.jl` 205 lines,
    `test_parse_substation.jl`, `test_parse_v30.jl`).

### Step 3 — PowerSystemCaseBuilder (small; code already landed)

 8. Repoint `test/Project.toml`: `PowerSystems rev` `"transformer-refactor"` → `"psy6"`;
    `PowerFlowFileParser rev` `"mb/transformer-refactor"` → `"psy6"` (now unblocked).
 9. Fix the pre-existing typo in root `Project.toml`: `PowerTableDataParser` url says
    `NLR-Sienna`, should be `NREL-Sienna`.
10. `julia --project=test -e 'using Pkg; Pkg.instantiate()'` then
    `julia --project=test -e 'using PowerSystemCaseBuilder'`.
11. Run the PSB suite, with attention to `test/test_transformer_parsing.jl` (new, 213 lines)
    and the CaseData artifact bump to `PowerSystemsTestData 5.0-dev2`.
12. Confirm every serialized case in the artifact round-trips — `TransformerCircuit`'s new
    `serialize`/`deserialize` and the deleted type names are the highest-risk surface for
    stale JSON in the artifact.

### Step 4 — PowerNetworkMatrices (the real work)

Strategy: **merge `psy6` into `transformer-refactor`**, resolve once, then merge back.
Rebasing would replay 5 commits against a `Ybus.jl` that both sides rewrote.

13. `git checkout transformer-refactor && git merge origin/psy6`.

14. Resolve the three overlapping files. Conflict budget, from the fork point `8e9579c`:
    
    | File                                    | psy6 side                 | transformer-refactor side   |
    |:--------------------------------------- |:------------------------- |:--------------------------- |
    | `src/Ybus.jl`                           | +154/−? (ZIBR, IS4 units) | −277 net, 415 lines touched |
    | `src/apply_zero_impedance_reduction.jl` | +133                      | 10 lines                    |
    | `src/BA_ABA_matrices.jl`                | +44                       | 4 lines                     |
    
    Everything else is disjoint: psy6-only (`subnetworks.jl`, `system_utils.jl`,
    `woodbury_kernel.jl`, `zero_impedance_branch_reduction.jl`) vs transformer-only
    (`BranchAdmittance.jl`, `common.jl`, `NetworkReductionData.jl`, `YbusACBranches.jl`,
    `network_modification.jl`, `definitions.jl`, `BranchesParallel.jl`, `BranchesSeries.jl`,
    `AdjacencyMatrix.jl`, `degree_two_reduction.jl`, `reduction_helpers.jl`,
    `PowerNetworkMatrix.jl`, `ThreeWindingTransformerWinding.jl` →
    `ThreeWindingTransformerCircuit.jl`).
15. **Known post-merge breakage — fix explicitly.** `transformer-refactor` still calls the
    stateful units API that psy6 deleted in the IS4 migration:
    
      + `src/Ybus.jl:898` `PSY.get_units_base(sys)`
      + `src/Ybus.jl:901` `PSY.set_units_base_system!(sys, "SYSTEM_BASE")`
      + `test/test_ybus.jl:100` `PSY.set_units_base_system!(sys, "NATURAL_UNITS")`
    
    Replace with explicit-units getter calls (`PSY.get_r(w, PSY.SU)` style), matching what
    psy6's `7134b8f` ("Drop stateful units block in Ybus") and `87dbfaf` ("Pass PSY.SU to
    get_r/get_x") already did elsewhere.
16. **Reconcile the two susceptance changes.** psy6 has `35edc0d` "fix susceptance
    cancellation" and `f4275ca` "ZIBR: compute series admittance directly";
    `transformer-refactor` has `be46323` "fix transformer winding susceptance calculation"
    and moved `get_series_susceptance` in from PSY (`BranchAdmittance.jl:53,64`,
    `BranchesParallel.jl:92`, `BranchesSeries.jl:107`,
    `ThreeWindingTransformerCircuit.jl:78`). These touch the same numerics — verify the
    merged result against both branches' tests, do not assume either resolution is a
    superset.

16b. **Three hazards from PNM's own `.claude/CLAUDE.md` that this refactor touches.**

    - *Legitimate Ybus asymmetry.* The rule "`Y[i,j] = −y/t*`, `Y[j,i] = −y/t` for
      `PhaseShiftingTransformer` is correct — don't fix it" was written against a type that no
      longer exists. The asymmetry must now be keyed off `PSY.is_phase_shifting(circuit)`,
      which is true when α ≠ 0 *or* the control objective is one of the four active-power
      objectives. That second arm is new: circuits with α = 0 but active-power control now
      qualify as phase-shifting, where the old type check would have missed them. Confirm
      that widening is intended before locking in the merge.
    - *Mixed bases in assembly.* `GenericArcImpedance` reads `PSY.DU` while shunts read
      `PSY.SU` (`Ybus.jl:~438-474`) — deliberate and subtle. Step 15 replaces a
      `set_units_base_system!` block with explicit unit args in the same file. Do not let
      that edit normalize these to one unit system; add a numeric regression test.
    - *3W outage isolates the star bus* → singular ABA, handled by the pinv islanding path.
      Availability is now per-circuit, not per-transformer, so "full outage" means all three
      circuits down and `set_available!(t, false)` cascades to all three. A single de-energized
      circuit is a new intermediate state PNM did not previously have to represent — check
      the reduction and contingency paths handle it.

17. Verify: `julia --project=test -e 'using Pkg; Pkg.instantiate()'`, then
    `julia --project=test -e 'using PowerNetworkMatrices'`. PNM already pins PSY `psy6` and
    PSB `psy6` in both `Project.toml` and `test/Project.toml` — no pin changes needed.
    Bump `PowerFlowFileParser rev` from `main` to `psy6` only if Step 2 lands.

18. Run the PNM suite. The transformer branch already rewrote 19 test files (notably
    `test_branch_admittance.jl` +504, `test_ybus_reductions.jl` +210, new
    `test_2d_reduction.jl` +166); psy6 has 11 test files still naming deleted PSY types.
    Both sets must pass together.
19. Merge `transformer-refactor` → `psy6`.
    19b. Update PNM's `.claude/CLAUDE.md`: the source layout still lists
    `ThreeWindingTransformerWinding.jl` (renamed to `ThreeWindingTransformerCircuit.jl`), and
    the Ybus-asymmetry rule still names `PhaseShiftingTransformer`. Both are now wrong.

### Step 5 — Rebase the trailing PNM branch

20. Rebase `precompile-load` onto the new `psy6`. Its workload exercises the power-flow
    matrix path, which Step 4 rewrote — expect the workload body to need updating for
    `TransformerCircuit` construction.

## 4. Test matrix

| Repo | Command                                           | Gate                                         |
|:---- |:------------------------------------------------- |:-------------------------------------------- |
| PSY  | `julia --project=test -e 'using Pkg; Pkg.test()'` | already green on `psy6` — confirm only       |
| PFFP | full suite                                        | Step 2 — pull landed, suite not yet run      |
| PSB  | full suite + artifact round-trip                  | Step 3                                       |
| PNM  | full suite                                        | Step 4, after each conflict-resolution chunk |

Cross-package: PNM's suite is the integration test — it pulls PSY `psy6` and PSB `psy6`.

## 5. Out of scope, but downstream and affected

`PowerFlows.jl@psy6` still names deleted PSY types in `src/dcpf_loss_injection.jl` (5),
`src/post_processing.jl` (2), `src/psse_export.jl` (16). It will not precompile against
PSY `psy6`. Not part of this task; flag for whoever owns PF.

## 6. Version bumps

Deferred. Everything here is on unreleased `psy6` branches pinned by URL+rev, so no
`compat` specifier can distinguish old from new API. Bump at psy6 release time, not now.
PNM `Project.toml` compat `PowerSystems = "^5.10"` stays as-is.
