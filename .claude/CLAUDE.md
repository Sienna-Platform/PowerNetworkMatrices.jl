# PowerNetworkMatrices.jl (PNM) — psy6 branch

The linear-algebra layer of the psy6 stack: Ybus, Incidence/Adjacency, BA/ABA, PTDF, LODF, lazy Virtual{PTDF,LODF,MODF}, contingency machinery (`ContingencySpec`, Woodbury updates), and **network reductions, which PNM owns exclusively**. Purely computational — the data model is PSY's. Layer 2; consumed by PowerFlows and PowerOperationsModels. Platform conventions: `.claude/Sienna.md`; workspace architecture: the psy6 workspace root `CLAUDE.md`.

## The ownership contract (defining psy6 fact)

**PNM owns network reductions.** Parallel branches between one bus pair are merged into an equivalent branch *before* PF/POM ever see them — never add dedupe bookkeeping downstream, and never let a consumer rebuild reduction state. Consumers get reduction state via:

  - `get_network_reduction_data(ybus)` → `NetworkReductionData`
  - map getters: `get_direct_branch_map`, `get_series_branch_map`, `get_bus_reduction_map`, …
  - reduced arcs source equivalent admittance/ratings from the `BranchesSeries`/`BranchesParallel`/`EquivalentBranch` aggregators; **aggregators return system base** (consumers need no extra `PSY.SU`).

**Aggregates dispatch on `AbstractReductionAggregate`** (`definitions.jl`), whose only subtypes are `AbstractBranchesParallel` (→ `BranchesParallel{T}`, `MixedBranchesParallel`) and `BranchesSeries`. It exists because aggregates subtype `PSY.ACTransmission`: without the intermediate layer they match blanket `::PSY.ACTransmission` methods written for a single physical branch — silently, whenever the blanket method returns a value instead of erroring (`common.jl` `_segment_has_single_pi` answering `true` unconditionally was exactly this). Rules:

  - Any method needing the reduction-aware `(segment, nr)` form dispatches on `AbstractReductionAggregate`, never on a `Union{...}` of the concrete types and never on an untyped parameter.
  - Adding a blanket `::PSY.ACTransmission` method? Check whether aggregates falling into it is correct. If not, add the `AbstractReductionAggregate` arm in the same commit.
  - Genuinely different algorithms stay per-type — parallel sums admittances, series sums impedances. Only collapse a pair whose bodies agree.
  - Unexported, like the concrete types; downstream uses `PNM.AbstractReductionAggregate`.

Known weak spots (2026-07-02 audit) — don't extend them:

  - Reduction invariants (forward/reverse map sync) are never checked; islanding during radial reduction only `@warn`s (`radial_reduction.jl:~195`). New reduction code should validate loudly.
  - `populate_branch_maps_by_type!` (`NetworkReductionData.jl`) is lazy with no invalidation — mutating reduction state after first query is unsafe.
  - PF and POM currently iterate `NetworkReductionData` internal maps directly; prefer adding accessor API here over widening that reach.

## Transformers are circuits, not types (PSY `d19f3244f`, PR #1714)

PSY replaced five concrete transformer types with two, and moved all series electrical data one level down. `Transformer2W`, `TapTransformer`, `PhaseShiftingTransformer` → **`TwoWindingTransformer`**; `Transformer3W`, `PhaseShiftingTransformer3W` → **`ThreeWindingTransformer`**. Those two names used to be abstract supertypes and are now concrete structs — old `<: TwoWindingTransformer` dispatch silently changes meaning rather than erroring.

Series data lives on **`PSY.TransformerCircuit <: DeviceParameter`** (not a `Component`): `available, arc, tap, α, r, x, control_objective, regulated_bus_number, control_limits, controlled_quantity_limits, number_of_tap_positions, rating, rating_b, rating_c, active_power_flow, reactive_power_flow, base_power, base_voltage_primary, base_voltage_secondary, base_value`. `winding_group_number` was removed — it duplicated `α`, which is the sole source of connection-group phase shift now.

  - 2W has one circuit (`PSY.get_circuit`); 3W has three (`get_primary_circuit`/`_secondary_`/`_tertiary_`) joining each terminal bus to `star_bus`. `PSY.get_circuits(t)` returns the tuple for either — **prefer it** over per-arity accessors so code works for both.
  - The transformer keeps only `name, magnetizing_shunt::Complex, shunt_location, services, ext, internal`. Shunt placement is an enum, not a convention: `TwoWindingTransformerShuntLocation` (`PRIMARY`/`SECONDARY`/`SPLIT` — SPLIT applies the **full** value on both sides, it does not halve it) and `ThreeWindingTransformerShuntLocation` (`PRIMARY`/`STAR`).
  - **Availability is derived**, not stored: `get_available(t) = any(get_available, get_circuits(t))`, and `set_available!(t, val)` cascades to every circuit — including ones that were individually out beforehand (PSS/E STAT semantics). Reading a transformer's availability and writing it back is therefore lossy.
  - `get_arc` exists for `TwoWindingTransformer` (delegating to its circuit) but **not** for `ThreeWindingTransformer` — it has three arcs. `get_from_bus`/`get_to_bus` inherit that limitation.
  - No `nothing`-sentinel confusion: 3W pairwise PSS/E fields (`r_12`,`x_12`,`r_23`,`x_23`,`r_31`,`x_31`,`base_power_12/23/31`) are legitimately `Union{Nothing,Float64}` and validated **all-or-none**. `base_power_13` was renamed `base_power_31`.

**PNM now owns the series-impedance API PSY deleted.** `get_series_susceptance` moved here, with six methods: the blanket `ACTransmission` `1/x`, the tap-dividing `TransformerCircuit` leaf, two delegating arms (`TwoWindingTransformer`, `ThreeWindingTransformerCircuit`), and the two aggregate reductions (`BranchesParallel.jl` sums, `BranchesSeries.jl` reciprocal-sums). There is no PSY fallback to defer to; a missing method is PNM's bug. The winding-group `get_α` derivations also went away — `get_α(circuit)` is now a plain stored field.

The `ThreeWindingTransformerCircuit` arm is load-bearing: the wrapper subtypes `PSY.ACTransmission`, not `PSY.TransformerCircuit`, so deleting it silently falls through to the **tap-free** blanket method rather than erroring.

`TransformerCircuit` carries its own units anchor in `base_value` (populated on `add_component!` via `set_units_setting!`, never serialized) and has hand-written `IS.serialize`/`IS.deserialize` that encode `arc` as a UUID. A circuit obtained from a detached transformer has `base_value === nothing` and its explicit-units getters will misbehave — build systems through `add_component!` before reading impedances.

**Orientation:** arc tuples are (from, to); anti-parallel members are sign-flipped during Ybus assembly (`Ybus.jl:~487-513`); POM applies `get_ptdf_orientation_sign` only in area_interchange. Orientation knowledge is scattered (audit candidate 6: normalize at construction under one PNM-owned convention) — when touching signs, check all three sites.

**Units in assembly:** Ybus assembly deliberately mixes bases — `GenericArcImpedance` reads `PSY.DU` (`Ybus.jl:~438-474`, correct because impedances are per-unitized on device base before conversion), while shunts/admittances elsewhere read `PSY.SU`. This is intentional but subtle; never change a unit argument here without a numeric regression test.

## Source layout

  - Core: `PowerNetworkMatrix.jl` (abstract `PowerNetworkMatrix{T} <: AbstractArray{T,2}`), `definitions.jl` (`AUTO_TOLERANCE_BUS_LIMIT = 2000`, `ZERO_IMPEDANCE_X_EPSILON = 1e-6`, `abstract type AbstractReductionAggregate`), `linalg_settings.jl`, `serialization.jl` (HDF5; dense PTDF only — virtual matrices are not serialized)
  - Matrices: `Ybus.jl`, `YbusACBranches.jl`, `ArcAdmittanceMatrix.jl`, `IncidenceMatrix.jl`, `AdjacencyMatrix.jl`, `BA_ABA_matrices.jl`, `ptdf_calculations.jl` (N_arcs×N_buses, transposed storage), `lodf_calculations.jl` (diagonal = −1.0), `virtual_{ptdf,lodf,modf}_calculations.jl`, `row_cache.jl` (LRU, default 100 MiB)
  - Modification/contingency: `modf_definitions.jl` (`ArcModification`, `ShuntModification`, `ContingencySpec`, `WoodburyFactors`), `network_modification.jl`, `woodbury_kernel.jl`, `ybus_contingencies.jl` — **mainline in psy6**; POM's branch-side N-1 builds on `VirtualMODF`/`ContingencySpec`
  - Reductions: `NetworkReduction.jl`, `NetworkReductionData.jl`, `ReductionContainer.jl`, `reduction_helpers.jl`, `radial_reduction.jl`, `degree_two_reduction.jl`, `ward_reduction.jl`, `zero_impedance_branch_reduction.jl` (the spec) + `apply_zero_impedance_reduction.jl` (the merge), `BranchesParallel.jl`, `BranchesSeries.jl`, `ThreeWindingTransformerCircuit.jl`, `EquivalentBranch.jl`
  - Connectivity: `connectivity_checks.jl`, `subnetworks.jl`
  - Solvers: `KLUWrapper/` (internal), `AccelerateWrapper/` (macOS built-in), `ext/MKLPardisoExt.jl` (x86_64 weakdep)

Exports live only in the main module file.

## Hard rules

  - **Never export or re-export any `KLUWrapper` symbol** (`KLULinSolveCache`, `solve!`, `klu_factorize`, …). Downstream reaches them qualified: `PowerNetworkMatrices.KLUWrapper.foo`.
  - **Per-arc PTDF/MODF KLU solve cost is inherent** — do not propose RHS batching or thread-parallelizing the build loop; Sienna queries rows incrementally and KLU can't do concurrent solves.
  - Only Int64 KLU caches are built internally; the Int32 variants exist solely for PowerFlows' `J_INDEX_TYPE`. Any downstream cache-type Union must list both.
  - Contingency fixtures: to make a contingency merely *exist*, use `PSY.FixedForcedOutage(; outage_status=1.0)` + `add_supplemental_attribute!` — never fabricate `GeometricDistributionForcedOutage` stochastic parameters.

## Numerics gotchas (hard-won; keep the regression tests green)

  - **AutoTolerance:** relative per-row drop `|x| < α·max|row|` with `α = clamp(safety·δ, 1e-6, 1e-2)`; the condition number κ is diagnostic only — never multiplied into the cutoff (κ-as-absolute-droptol was a real densification bug). The bus-count gate (`AUTO_TOLERANCE_BUS_LIMIT`) keeps small PSB cases exact; the dense path never sparsifies under AutoTolerance (an explicit Float64 `tol` still does).
  - **`min_x_eps` applies to every branch type, transformers included.** `equivalent_branch` substitutes it for `x` when `r == x == 0`; the value comes from `ZeroImpedanceBranchReduction`'s `minimum_retained_impedance` (`Ybus.jl` `_minimum_retained_impedance`), falling back to `ZERO_IMPEDANCE_X_EPSILON`. The transformer arms used to accept the kwarg and drop it, which mattered because `PSY.TransformerCircuit` **defaults `r = x = 0.0`** and validates zero silently (range `(-2, 4)`, action `warn`): the result was `1/(0+0im)` → NaN admittance, surfacing as Ybus assembly's `isfinite` guard throwing "non-finite Ybus entry". Do not reintroduce that by adding a transformer path that bypasses `_circuit_equivalent_branch`. ZIR **excludes transformer arcs**, so unlike a line — whose endpoints get merged — a transformer keeps the substituted reactance permanently; the warning text is dispatched on `_is_transformer` for that reason.
  - **Susceptance vs admittance islands:** branches with r>0, x=0 have b=0 and vanish from BA — the susceptance graph fragments more than the admittance graph → singular ABA. Zero-impedance reduction must resolve *both* endpoints to union-find roots before merging.
  - **Ybus asymmetry is legitimate** for phase-shifting circuits (`Y[i,j] = −y/t*`, `Y[j,i] = −y/t`) — don't "fix" it. Test with `PSY.is_phase_shifting(circuit)`, never a type check: the predicate is true when `α ≠ 0` **or** the control objective is one of the four active-power objectives, so a circuit with `α = 0` under active-power control is phase-shifting even though nothing about its angle says so. But ZIR column-merge asymmetry was a real DC-PF NaN bug; both directions have regression tests.
  - **Parallel groups mixing α with impedance angles have no single π** — recovery needs `|Y12| = |Y21|`, so `get_equivalent_physical_branch_parameters` throws by construction on e.g. a lossy PST beside a lossless line. Such a group is still *exactly* representable as several parallel π branches: `equivalent_partitions` / `arc_equivalent_branches` return one π per impedance-angle bucket, and the invariant is that their π-models sum back to `ybus_branch_entries(bp, nr)` (residuals land at ~1e-16). Uniform α — even with mixed R/X — still collapses to a single π, so the whole-group representability check must run *before* partitioning. A *series* chain containing such a group has no π representation at any count; that error is correct, not a gap.
  - **The recovered AC shift is not the DC α.** `imag(log(Y21/Y12))/2` equals `atan(Σbₘ sin αₘ / Σbₘ cos αₘ)` for a lossless group; the susceptance-weighted average `Σbₘαₘ/Σbₘ` that `arc_dc_phase_shift` returns is only its small-angle limit. For Line(x=.1) ∥ PST(x=.2, α=.15) the AC value is 0.04995825507139971 and the DC value is exactly 0.05. Never cross-use them as test oracles.
  - **Equivalent-parameter recovery runs in ComplexF64**, off `ybus_branch_entries`, not the `YBUS_ELTYPE`(=ComplexF32) `equivalent_ybus` cache — that downcast cost ~7e-8 relative and left the representability test only ~8 Float32 eps wide. The cache field and `populate_equivalent_ybus!` are Ybus-assembly-only and currently have no readers. `_build_chain_ybus` still narrows series chains to F32 upstream of recovery.
  - **Anti-parallel branches cancel in value-based adjacency** (+1/−1 sums to 0) — `_resolve_antiparallel_adjacency!` / `_repair_merged_adjacency!` restore connectivity; keep them in any new adjacency path.
  - **3W circuits register through the merge-aware branch-map path.** `add_to_branch_maps!(::NetworkReductionData, ::PSY.ThreeWindingTransformer)` used to write each winding straight into `direct_branch_map[arc]`, silently overwriting anything already on that star-point arc (a `Line`, another winding, an existing group) and leaving a dangling `reverse_direct_branch_map` entry. It now delegates per circuit to the 3-arg `add_to_branch_maps!`, which promotes to a parallel group like any other branch. This was the 3W survival of the exact defect issue #231 described for 2W.
  - **3-winding transformer full outage isolates the star bus** → singular ABA; the pinv islanding path handles it. "Full" now means *all three circuits* de-energized — availability is per-circuit, so a partial outage (one or two circuits out) is a real intermediate state the old per-device flag could not express. AppleAccelerate LU silently factorizes singular matrices (garbage results) where KLU throws — prefer KLU whenever singularity is possible.
  - **DegreeTwoReduction `reduce_reactive_power_injectors` defaults `true`** — correct for DC, electrically wrong for AC (KeyError on `reverse_bus_search_map`; flows drift). AC consumers must pass `false`; PowerFlows throws `ConflictingInputsError` if not.
  - **MODF/reduction consistency:** outaged and monitored branches must survive the reduction, or queries silently return the base-case row. Universal survive-check: membership in `keys(get_bus_reduction_map(nrd))`.

## Commands

```sh
julia --project=test -e 'using Pkg; Pkg.instantiate()'                         # once per clone
julia --project=test test/runtests.jl                                          # full suite (ReTest)
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("PTDF")'   # name filter
julia --project=docs docs/make.jl                                              # docs must build clean
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

Compile-check: `julia --project=test -e 'using PowerNetworkMatrices'`.

**Never `Pkg.develop` into an environment carrying `[sources]` git pins.** It re-resolves and pulls a mismatched PSY/IS pair — the psy6 envs pin `PowerSystems#psy6` (5.10.0) and `InfrastructureSystems#IS4`, and `develop` will happily swap in registry PSY 5.12.0, which fails to precompile with `UndefVarError: SystemUnitsSettings`. `test/Project.toml` already pins what it needs; plain `Pkg.instantiate()` is the whole setup step.

**The docs build is a required gate — `docs/make.jl` must pass.** Treat a docs failure like a test failure, and fix `missing_docs` by registering the docstring in `@autodocs`/`@docs`, never by silencing with `warnonly`.

Local caveat, not an exemption: `docs/Project.toml` currently carries no `[sources]`, so it resolves registry PSY against `InfrastructureSystems#IS4` and dies precompiling with `UndefVarError: SystemUnitsSettings`. Until that env is pinned to `PowerSystems#psy6`, the gate only runs in CI — so docstring edits land unverified locally and need care:

  - Only **exported** symbols render (`docs/src/reference/public.md` is `@autodocs Public = true`; `internals.md` covers only the `KLUWrapper`/`AccelerateWrapper` submodules). An `@ref` to an unexported symbol is a broken link — prefer plain backticks for internal names.
  - Newly *attaching* a docstring on an exported symbol makes its `@ref`s resolve for the first time, which can surface broken links that were previously dead text. Audit them when you fix an attachment.
  - Pinning `docs/Project.toml` to the psy6 sources is the real fix and would restore the gate locally.

**A comment between a docstring and its definition silently detaches the docstring** — Julia does not bridge it, and you get no warning; the symbol just reports "No documentation found". This bit `equivalent_branch`, whose docstring sat above an intervening comment and was dead text for its whole life. Put explanatory comments *above* the docstring, and verify with `@doc PNM.f`.

ReTest notes: don't use `@test_logs` to assert warnings (MethodError on failure) — use a custom `AbstractLogger`; verify testset registration with `run_tests(dry=true)`. Filters match **testset names, not filenames** — `run_tests("equivalent_getters")` matches nothing and reports a misleading 0-pass green. The formatter also walks `docs/` and `.claude/` with `format_markdown=true` and **aborts the whole run** on the first unparseable markdown file, silently skipping `src/` — if it errors on a plan doc, format the touched files directly with the same options.

## Test fixtures for reductions

`c_sys5`/`c_sys14` reduce **nothing** (empty series/parallel maps) — a passing reduction test on them proves nothing. Use `case10_radial_series_reductions` (real series arcs and a 3W transformer; no forecasts, so build `NetworkReductionData` directly for white-box tests) or matpower RTS/case24. If a PSB-cached system misbehaves after upstream changes, `force_build=true` / clear `data/serialized_system/`.

`c_sys14` is still the right base for *zero-impedance* fixtures — zero out an existing branch's `r`/`x` with `set_r!`/`set_x!` rather than constructing one, since a detached component cannot resolve the system base.

## Downstream blast radius

PowerFlows and POM consume matrices, reduction maps (`arc_ax`, `reverse_bus_search_map`, `get_arc_tuple`), and KLU caches; changes there break them — sometimes latently (past examples: PSS/E exporter `get_lcc_names` KeyError, fixed-admittance KeyError surfaced only after a PF pin). After a PNM change, run the PF suite and at least the POM network-constructor tests.
