# PowerNetworkMatrices.jl — Claude Guide

Platform-wide Sienna conventions (performance, type stability, formatter, environments, code style) live in `.claude/Sienna.md` — read it too. This file is repo-specific and does not restate them.

## Purpose & place in the stack

PNM builds the linear-algebra layer of the Sienna power-systems platform: the network matrices used for DC/AC power flow, sensitivity, and contingency analysis. It depends on **PowerSystems.jl** (PSY, component data) and **InfrastructureSystems.jl** (IS, shared utilities), and is consumed by **PowerFlows.jl**, **PowerSimulations.jl**, and **PowerSystemsInvestmentsPortfolios.jl**. PNM is purely computational — it reads a `System` and produces matrix objects; it does not own the data model.

Matrices provided: `Ybus` (complex nodal admittance), `IncidenceMatrix`, `AdjacencyMatrix`, `BA_Matrix`/`ABA_Matrix` (DC susceptance forms), `ArcAdmittanceMatrix`, `PTDF`, `LODF`, and the lazy/on-demand `VirtualPTDF`/`VirtualLODF`/`VirtualMODF`. Plus network-reduction strategies (`RadialReduction`, `DegreeTwoReduction`, `WardReduction`) and network-modification/contingency tooling.

Current version: **0.24.2**. Deps of note: `PowerSystems ^5.11`, `InfrastructureSystems 3`, `KLU` via `SuiteSparse_jll`, `HDF5 0.17`, `julia ^1.10`. `Pardiso` is a weakdep (MKLPardisoExt). AppleAccelerate is **built into PNM** (no weakdep) via `src/AccelerateWrapper/`.

## Source layout (`src/`)

  - **Entry/core:** `PowerNetworkMatrices.jl` (module + include order + all exports), `PowerNetworkMatrix.jl` (abstract `PowerNetworkMatrix{T} <: AbstractArray{T,2}` — array interface, axes, lookup dicts, subnetwork handling), `definitions.jl` (constants/tolerances, e.g. `AUTO_TOLERANCE_BUS_LIMIT = 2000`), `linalg_settings.jl`, `solver_dispatch.jl`, `common.jl` (utilities, `sparsify`), `system_utils.jl`, `serialization.jl` (HDF5 I/O).
  - **Solver wrappers:** `KLUWrapper/` (internal libklu binding, `KLULinSolveCache{Tv,Ti}`, `iterative_refinement.jl`), `AccelerateWrapper/` (internal libSparse binding, macOS-only, `@static if Sys.isapple()`), `ext/MKLPardisoExt.jl`.
  - **Network matrices:** `Ybus.jl`, `YbusACBranches.jl`, `ArcAdmittanceMatrix.jl`, `IncidenceMatrix.jl`, `AdjacencyMatrix.jl`, `BA_ABA_matrices.jl`, `ptdf_calculations.jl`, `lodf_calculations.jl`, `PowerflowMatrixTypes.jl` (type aliases: `DC_PTDF_Matrix`, `DC_ABA_Matrix_Factorized`, `AC_Ybus_Matrix`, etc.).
  - **Virtual/lazy:** `row_cache.jl` (LRU `RowCache`), `virtual_ptdf_calculations.jl`, `virtual_lodf_calculations.jl`, `virtual_modf_calculations.jl`, `virtual_ptdf_modification.jl`, `auto_tolerance.jl` (per-row sparsification).
  - **Modification & contingencies:** `modf_definitions.jl` (`ArcModification`/`ShuntModification`/`ContingencySpec`/`WoodburyFactors`), `network_modification.jl`, `woodbury_kernel.jl`, `ybus_contingencies.jl`, `modf_reduction_consistency.jl`.
  - **Reduction:** `NetworkReduction.jl`, `NetworkReductionData.jl`, `ReductionContainer.jl`, `reduction_helpers.jl`, `radial_reduction.jl`, `degree_two_reduction.jl`, `ward_reduction.jl`, `zero_impedance_branch_reduction.jl`, `apply_zero_impedance_reduction.jl`, `BranchesParallel.jl`, `BranchesSeries.jl`, `EquivalentBranch.jl`, `ThreeWindingTransformerWinding.jl`.
  - **Connectivity:** `connectivity_checks.jl` (island detection), `subnetworks.jl`.

Note: `flowgates.jl` and `VirtualMODF` partial-LODF live on feature branches (`jd/make_flowgates`, `jd/virtual_modf`); `VirtualMODF` is exported and present here, `flowgates` is not on every branch — check before assuming.

## Public API highlights

Exported types: `Ybus`, `IncidenceMatrix`, `AdjacencyMatrix`, `BA_Matrix`, `ABA_Matrix`, `PTDF`, `LODF`, `VirtualPTDF`, `VirtualLODF`, `VirtualMODF`, `ArcModification`, `ShuntModification`, `ContingencySpec`, `NetworkModification`, `RadialReduction`, `DegreeTwoReduction`, `WardReduction`, `NetworkReduction`, `NetworkReductionData`, `AutoTolerance`. Type aliases `DC_PTDF_Matrix`/`DC_ABA_Matrix_*`/`DC_vPTDF_Matrix`/`DC_BA_Matrix`/`AC_Ybus_Matrix`. Functions: `get_ptdf_data`/`get_lodf_data`, `get_network_reduction_data`, `get_bus_reduction_map`, `find_subnetworks`, `validate_connectivity`, `to_hdf5`/`from_hdf5`, `clear_caches!`/`clear_all_caches!`, `apply_ybus_modification`, `apply_woodbury_correction`, `get_partial_lodf_row`. Full list at the top of `src/PowerNetworkMatrices.jl` — keep all exports there.

Indexing: matrices accept bus numbers / branch (arc) tuples directly (`matrix[bus_num, (from,to)]`) and auto-map to internal indices. Subnetworks (electrical islands) are handled transparently via per-island axes.

## Commands (verified)

```bash
# Run full suite (ReTest runner; auto-discovers test_*.jl)
julia --startup-file=no --project=test test/runtests.jl

# Filter a testset (run_tests forwards args to ReTest's retest())
julia --startup-file=no --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("PTDF")'

# Instantiate test env (Manifest is gitignored/regenerable)
julia --startup-file=no --project=test -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'

# Compile-check after each edit
julia --startup-file=no --project=test -e 'using PowerNetworkMatrices'

# Formatter (run after every task)
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'

# Docs
julia --project=docs docs/make.jl
```

Always pass `--startup-file=no`: the user's `~/.julia/config/startup.jl` does `using AppleAccelerate`, which breaks the test env. The test env uses **ReTest** (`retest()` in `run_tests`), not classic `@testset` discovery. Test data comes from **PowerSystemCaseBuilder** (PSB) — IEEE/Matpower/PSS-E cases; results validated against PSS/E and Matpower references.

## Package-specific conventions & invariants

  - **Performance is the headline goal.** Sparse (`SparseMatrixCSC`) by default; concrete types in hot paths; views/in-place ops; `iszero(x)` not `== 0`. Hot paths are the virtual-matrix row miss (`sparsify`) and the per-arc KLU solve.
  - **KLUWrapper export ban:** never re-export any `KLUWrapper` symbol from the main module (`KLULinSolveCache`, `solve!`, `klu_factorize`, etc.). The submodule exports them internally; PNM brings them in via `import .KLUWrapper: name` and downstream reaches them as `PowerNetworkMatrices.KLUWrapper.foo`. KLUWrapper is an internal binding free to evolve.
  - **Solvers:** KLU (default off-Apple / macOS <15.5), built-in `AppleAccelerateLU` (libSparse `SparseFactorizationLU` + Inf-norm equilibration; default on macOS ≥15.5), `AppleAccelerateLDL`, MKLPardiso (ext). Real Float64 only for AA-LU; complex AC Ybus stays on KLU. PNM builds only `Int64` KLU caches internally; `Int32` exists solely for downstream PowerFlows (`J_INDEX_TYPE`).
  - **Virtual matrices** trade compute for memory: LRU row cache (default ~100 MiB). Use these past `AUTO_TOLERANCE_BUS_LIMIT`/large systems instead of dense. They are NOT serialized (only dense `PTDF` is).
  - **AutoTolerance** sparsifies as a *relative per-row* drop: entry dropped when `|x| < α·max|row|`, `α = clamp(safety·δ, 1e-6, 1e-2)`. Condition number κ is logged as a diagnostic only — never multiplied into the cutoff (the old κ-as-absolute-droptol formula produced *denser* columns on large grids and is a known trap). A bus-count gate (`AUTO_TOLERANCE_BUS_LIMIT = 2000`) makes it a no-op below the limit, so all PSB test cases stay exact. The dense path never sparsifies under AutoTolerance (preserves `Matrix{Float64}` dispatch); a Float64 `tol` still does (back-compat).

## Numerical / conditioning gotchas

  - **Susceptance vs admittance island divergence.** `find_subnetworks` walks the Ybus *admittance* graph; ABA is built from the *susceptance* graph. Branches with r>0, x=0 have b=0 and are absent from BA — they can fragment the susceptance graph into more components than the admittance graph, leaving ref-less blocks → singular ABA. Resolved upstream (zero-impedance reduction must resolve both arc endpoints to union-find roots before merging).
  - **Ybus asymmetry is legitimate for PhaseShiftingTransformers** (`Y[i,j] = −y/t*`, `Y[j,i] = −y/t`) — do not "fix" it. But asymmetry from a *zero-impedance reduction column-merge bug* is real and caused DC-PF NaNs (BA reading the zeroed mutual). Regression tests guard both.
  - **Anti-parallel branches cancel** in value-based adjacency (opposite arcs, same bus pair sum +1/−1 to 0). Connectivity readers that test values (not structure) break; the signed convention is kept and `_resolve_antiparallel_adjacency!` / `_repair_merged_adjacency!` restore degree after initial build and after ZIR merges.
  - **3-winding transformers** are a wye to a zero-injection PSY star bus; full outage isolates the star → singular ABA. The generic pinv islanding path already yields correct PTDF rows. **AppleAccelerateLU silently factorizes singular matrices (garbage) where KLU throws** — prefer KLU when singularity is possible.
  - **DegreeTwoReduction with reactive-only injectors:** `reduce_reactive_power_injectors` defaults `true` (more reduction) and is correct for DC, but electrically wrong for AC (eliminated reactive shunt buses lack a `reverse_bus_search_map` entry → KeyError; flows drift). AC consumers must set it `false` (PowerFlows throws `ConflictingInputsError` on the bad pairing).
  - **MODF/reduction consistency:** outaged + monitored branches must survive network reduction or post-contingency queries silently return the base row / crash. Reduction adjustment is mandatory (not opt-out). Universal "did this bus survive" check is `keys(get_bus_reduction_map(nrd))` (Radial/DegreeTwo land in `irreducible_buses`, Ward retains via `study_buses`).
  - **Per-arc PTDF/MODF solve cost is inherent — never propose RHS batching or thread-parallelizing the build-time solve loop.** Sienna queries rows incrementally (one arc/contingency), so multi-RHS doesn't fit, and KLU cannot do concurrent solves even with per-thread workspaces. The build profile is fully mined; the only realized lever was the Ybus `sparse()` adjacency rewrite.

## Cross-package coupling

  - Reads PSY components; thread PSY unit systems through getters (system base `PSY.SU` for impedances/admittances/shunts, device base `PSY.DU` for ratings). Read PSY source to verify API (`~/.julia/dev/PowerSystems`: `src/outages.jl`, `src/contingencies.jl`, `src/get_components_interface.jl`, `src/base.jl`) — don't guess.
  - Changes to matrix construction, reduction maps (`arc_ax`, `reverse_bus_search_map`, `get_arc_tuple`), or KLU caches can break **PowerFlows** and **PowerSimulations**. Several past breakages were latent until PowerFlows pinned a PNM release (e.g. PSS/E exporter `get_lcc_names` KeyError, fixed-admittance KeyError). Consider downstream impact before changing public matrix/reduction surface.
  - Contingencies/tests: when a contingency only needs to *exist*, use `PSY.FixedForcedOutage(; outage_status=1.0)` + `PSY.add_supplemental_attribute!(sys, branch, outage)`. Do not fabricate stochastic `GeometricDistributionForcedOutage` params.

## Test-env quirks

  - ReTest: do **not** use `@test_logs` to assert warnings (it throws `MethodError: record(::ReTestSet, ::Test.LogTestFailure)` on failure). Use a custom `AbstractLogger` collecting `Warn`. Confirm a testset is registered with `run_tests(dry=true)`.
  - Manifest.toml is gitignored. If tests load a stale PNM, `rm test/Manifest.toml Manifest.toml`, then `Pkg.develop(path="."); Pkg.instantiate()`.
  - On this network, Julia libgit2 hits "self-signed certificate in chain" — run Pkg with `JULIA_PKG_USE_CLI_GIT=true` so `[sources]`/git pins aren't silently ignored.
  - PSB shares state: stale serialized caches vs IS enum renames flake; clear with `PSB.build_system(...; force_build=true)`. A known pre-existing error ("Basic ward reduction") is an upstream PSB/IS `NATURAL_UNITS` deserialize incompatibility, not a PNM bug.
  - Formatter has `format_markdown=true` and walks `docs/`: `.md` fenced blocks must be labeled `bash`/`text` (never bare `or`julia for shell fragments) or it aborts. `docs/superpowers/` is gitignored (local-only specs/plans).

## Docs

Diataxis layout under `docs/src/`: `tutorials/`, `how_to_guides/`, `explanation/`, `reference/` (public API via `@autodocs` `Public=true`). Fix Documenter `missing_docs` by registering docstrings (`@autodocs`/`@docs`), not by silencing with `warnonly`. Docstrings on all public interface, `DocStringExtensions.TYPEDSIGNATURES`.

## Workflow reminders

Never `git commit` — stage only. Run formatter and the full suite before declaring done; compile-check each Julia edit before the next. Respect include order in `src/PowerNetworkMatrices.jl` when adding constants/types. Prefer multiple dispatch over `isa`/`<:`; `if/else` over ternary; `get_*` getters over dot access in public-facing code (watch PSY name collisions — prefix where needed).
