# PowerNetworkMatrices.jl (PNM) — psy6 branch

The linear-algebra layer of the psy6 stack: Ybus, Incidence/Adjacency, BA/ABA, PTDF, LODF, lazy Virtual{PTDF,LODF,MODF}, contingency machinery (`ContingencySpec`, Woodbury updates), and **network reductions, which PNM owns exclusively**. Purely computational — the data model is PSY's. Layer 2; consumed by PowerFlows and PowerOperationsModels. Platform conventions: `.claude/Sienna.md`; workspace architecture: `/home/jdlara/Sienna_work/psy6/CLAUDE.md`.

## The ownership contract (defining psy6 fact)

**PNM owns network reductions.** Parallel branches between one bus pair are merged into an equivalent branch *before* PF/POM ever see them — never add dedupe bookkeeping downstream, and never let a consumer rebuild reduction state. Consumers get reduction state via:

  - `get_network_reduction_data(ybus)` → `NetworkReductionData`
  - map getters: `get_direct_branch_map`, `get_series_branch_map`, `get_bus_reduction_map`, …
  - reduced arcs source equivalent admittance/ratings from the `BranchesSeries`/`BranchesParallel`/`EquivalentBranch` aggregators; **aggregators return system base** (consumers need no extra `PSY.SU`).

Known weak spots (2026-07-02 audit) — don't extend them:

  - Reduction invariants (forward/reverse map sync) are never checked; islanding during radial reduction only `@warn`s (`radial_reduction.jl:~195`). New reduction code should validate loudly.
  - `populate_branch_maps_by_type!` (`NetworkReductionData.jl`) is lazy with no invalidation — mutating reduction state after first query is unsafe.
  - PF and POM currently iterate `NetworkReductionData` internal maps directly; prefer adding accessor API here over widening that reach.

## Transformers are circuits, not types (PSY `d19f3244f`, PR #1714)

PSY replaced five concrete transformer types with two, and moved all series electrical data one level down. `Transformer2W`, `TapTransformer`, `PhaseShiftingTransformer` → **`TwoWindingTransformer`**; `Transformer3W`, `PhaseShiftingTransformer3W` → **`ThreeWindingTransformer`**. Those two names used to be abstract supertypes and are now concrete structs — old `<: TwoWindingTransformer` dispatch silently changes meaning rather than erroring.

Series data lives on **`PSY.TransformerCircuit <: DeviceParameter`** (not a `Component`): `available, arc, tap, α, winding_group_number, r, x, control_objective, regulated_bus_number, control_limits, controlled_quantity_limits, number_of_tap_positions, rating, rating_b, rating_c, active_power_flow, reactive_power_flow, base_power, base_voltage_primary, base_voltage_secondary, base_value`.

  - 2W has one circuit (`PSY.get_circuit`); 3W has three (`get_primary_circuit`/`_secondary_`/`_tertiary_`) joining each terminal bus to `star_bus`. `PSY.get_circuits(t)` returns the tuple for either — **prefer it** over per-arity accessors so code works for both.
  - The transformer keeps only `name, magnetizing_shunt::Complex, shunt_location, services, ext, internal`. Shunt placement is an enum, not a convention: `TwoWindingTransformerShuntLocation` (`PRIMARY`/`SECONDARY`/`SPLIT` — SPLIT applies the **full** value on both sides, it does not halve it) and `ThreeWindingTransformerShuntLocation` (`PRIMARY`/`STAR`).
  - **Availability is derived**, not stored: `get_available(t) = any(get_available, get_circuits(t))`, and `set_available!(t, val)` cascades to every circuit — including ones that were individually out beforehand (PSS/E STAT semantics). Reading a transformer's availability and writing it back is therefore lossy.
  - `get_arc` exists for `TwoWindingTransformer` (delegating to its circuit) but **not** for `ThreeWindingTransformer` — it has three arcs. `get_from_bus`/`get_to_bus` inherit that limitation.
  - No `nothing`-sentinel confusion: 3W pairwise PSS/E fields (`r_12`,`x_12`,`r_23`,`x_23`,`r_31`,`x_31`,`base_power_12/23/31`) are legitimately `Union{Nothing,Float64}` and validated **all-or-none**. `base_power_13` was renamed `base_power_31`.

**PNM now owns the series-impedance API PSY deleted.** `get_series_susceptance`, `get_series_susceptances`, `get_series_admittance`, `get_series_admittances` moved here — `BranchAdmittance.jl`, `BranchesParallel.jl`, `BranchesSeries.jl`, `ThreeWindingTransformerCircuit.jl`. There is no PSY fallback to defer to; a missing method is PNM's bug. The winding-group `get_α` derivations also went away — `get_α(circuit)` is now a plain stored field.

`TransformerCircuit` carries its own units anchor in `base_value` (populated on `add_component!` via `set_units_setting!`, never serialized) and has hand-written `IS.serialize`/`IS.deserialize` that encode `arc` as a UUID. A circuit obtained from a detached transformer has `base_value === nothing` and its explicit-units getters will misbehave — build systems through `add_component!` before reading impedances.

**Orientation:** arc tuples are (from, to); anti-parallel members are sign-flipped during Ybus assembly (`Ybus.jl:~487-513`); POM applies `get_ptdf_orientation_sign` only in area_interchange. Orientation knowledge is scattered (audit candidate 6: normalize at construction under one PNM-owned convention) — when touching signs, check all three sites.

**Units in assembly:** Ybus assembly deliberately mixes bases — `GenericArcImpedance` reads `PSY.DU` (`Ybus.jl:~438-474`, correct because impedances are per-unitized on device base before conversion), while shunts/admittances elsewhere read `PSY.SU`. This is intentional but subtle; never change a unit argument here without a numeric regression test.

## Source layout

  - Core: `PowerNetworkMatrix.jl` (abstract `PowerNetworkMatrix{T} <: AbstractArray{T,2}`), `definitions.jl` (`AUTO_TOLERANCE_BUS_LIMIT = 2000`), `linalg_settings.jl`, `serialization.jl` (HDF5; dense PTDF only — virtual matrices are not serialized)
  - Matrices: `Ybus.jl`, `YbusACBranches.jl`, `ArcAdmittanceMatrix.jl`, `IncidenceMatrix.jl`, `AdjacencyMatrix.jl`, `BA_ABA_matrices.jl`, `ptdf_calculations.jl` (N_arcs×N_buses, transposed storage), `lodf_calculations.jl` (diagonal = −1.0), `virtual_{ptdf,lodf,modf}_calculations.jl`, `row_cache.jl` (LRU, default 100 MiB)
  - Modification/contingency: `modf_definitions.jl` (`ArcModification`, `ShuntModification`, `ContingencySpec`, `WoodburyFactors`), `network_modification.jl`, `woodbury_kernel.jl`, `ybus_contingencies.jl` — **mainline in psy6**; POM's branch-side N-1 builds on `VirtualMODF`/`ContingencySpec`
  - Reductions: `NetworkReduction.jl`, `NetworkReductionData.jl`, `ReductionContainer.jl`, `radial_reduction.jl`, `degree_two_reduction.jl`, `ward_reduction.jl`, `BranchesParallel.jl`, `BranchesSeries.jl`, `ThreeWindingTransformerCircuit.jl`, `EquivalentBranch.jl`
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
  - **Susceptance vs admittance islands:** branches with r>0, x=0 have b=0 and vanish from BA — the susceptance graph fragments more than the admittance graph → singular ABA. Zero-impedance reduction must resolve *both* endpoints to union-find roots before merging.
  - **Ybus asymmetry is legitimate** for phase-shifting circuits (`Y[i,j] = −y/t*`, `Y[j,i] = −y/t`) — don't "fix" it. Test with `PSY.is_phase_shifting(circuit)`, never a type check: the predicate is true when `α ≠ 0` **or** the control objective is one of the four active-power objectives, so a circuit with `α = 0` under active-power control is phase-shifting even though nothing about its angle says so. But ZIR column-merge asymmetry was a real DC-PF NaN bug; both directions have regression tests.
  - **Anti-parallel branches cancel in value-based adjacency** (+1/−1 sums to 0) — `_resolve_antiparallel_adjacency!` / `_repair_merged_adjacency!` restore connectivity; keep them in any new adjacency path.
  - **3-winding transformer full outage isolates the star bus** → singular ABA; the pinv islanding path handles it. "Full" now means *all three circuits* de-energized — availability is per-circuit, so a partial outage (one or two circuits out) is a real intermediate state the old per-device flag could not express. AppleAccelerate LU silently factorizes singular matrices (garbage results) where KLU throws — prefer KLU whenever singularity is possible.
  - **DegreeTwoReduction `reduce_reactive_power_injectors` defaults `true`** — correct for DC, electrically wrong for AC (KeyError on `reverse_bus_search_map`; flows drift). AC consumers must pass `false`; PowerFlows throws `ConflictingInputsError` if not.
  - **MODF/reduction consistency:** outaged and monitored branches must survive the reduction, or queries silently return the base-case row. Universal survive-check: membership in `keys(get_bus_reduction_map(nrd))`.

## Commands

```sh
julia --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once per clone
julia --project=test test/runtests.jl                                          # full suite (ReTest)
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("PTDF")'   # name filter
julia --project=docs docs/make.jl
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

Compile-check: `julia --project=/home/jdlara/Sienna_work/psy6 -e 'using PowerNetworkMatrices'`.

ReTest notes: don't use `@test_logs` to assert warnings (MethodError on failure) — use a custom `AbstractLogger`; verify testset registration with `run_tests(dry=true)`. The formatter also walks `docs/` with `format_markdown=true` — fenced blocks must carry a language label (`bash`/`text`/`julia`) or it aborts.

## Test fixtures for reductions

`c_sys5`/`c_sys14` reduce **nothing** (empty series/parallel maps) — a passing reduction test on them proves nothing. Use `case11_network_reductions` (real series arcs; no forecasts, so build `NetworkReductionData` directly for white-box tests) or matpower RTS/case24. If a PSB-cached system misbehaves after upstream changes, `force_build=true` / clear `data/serialized_system/`.

## Downstream blast radius

PowerFlows and POM consume matrices, reduction maps (`arc_ax`, `reverse_bus_search_map`, `get_arc_tuple`), and KLU caches; changes there break them — sometimes latently (past examples: PSS/E exporter `get_lcc_names` KeyError, fixed-admittance KeyError surfaced only after a PF pin). After a PNM change, run the PF suite and at least the POM network-constructor tests.
