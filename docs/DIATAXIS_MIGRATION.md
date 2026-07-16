# PowerNetworkMatrices.jl — Diátaxis Documentation Overhaul: Execution Plan

**Audience:** AI agents (and humans) executing the docs overhaul. This file is a
self-contained handoff — you do **not** need any prior conversation context to
pick up a work package below and complete it.

**Status:** planning / not started. Update the checkboxes as work lands.

* * *

## 0. How to use this document

 1. Read §1 (principles) and §2 (repo facts) once — they apply to every package.
 2. Read §3 (global correctness gotchas) — these are landmines; violating them
    ships wrong docs. Every drafting agent must honor them.
 3. Pick a work package from §5. Each is self-contained: it names its **target
    file**, its **source material**, the **API symbols to cover**, its
    **guardrails**, its **dependencies**, and its **done-when** criteria.
 4. Respect the dependency graph in §6 when parallelizing.
 5. Verify with the build command in §2.4 before marking done.

**Line-number anchors in this doc may drift.** They were accurate at authoring
time. Always open the file and confirm the symbol before quoting a location in
prose; never paste a line number you haven't re-verified.

* * *

## 1. Guiding principles (Diátaxis, applied to a *library*)

The four modes are distinct and must not blur:

  - **Tutorial** — a *learning journey*. One hand-held path, chosen by the author,
    ending in one meaningful success. Organized around the **learner's arc**, not
    the software's surface.
  - **How-to guide** — a *task recipe*. "To accomplish X, do these steps."
    Assumes competence; solves a real goal.
  - **Reference** — *information*. Dry, exhaustive, structured to match the code.
    Describes the machinery; does not teach.
  - **Explanation** — *understanding*. Discussion of the why, the trade-offs, the
    theory. No step-by-step.

### The key diagnosis driving this overhaul

PNM's current `tutorials/` folder is **over-built and mis-categorized**. Most of
its "tutorials" are **feature tours** organized by matrix type ("here's
Incidence, here's BA, here's ABA…") — that is a *reference* concern wearing a
tutorial costume. A computational-matrix library like PNM naturally has a
**small** tutorial quadrant; its real value lives in **reference** and
**explanation**.

**Consequence — do not manufacture tutorials.** The target has exactly **two**
tutorials. Anything you are tempted to add as a tutorial is almost certainly a
how-to or an explanation. The rebalancing is the point:

| Quadrant    | Before | After |
|:----------- |:------ |:----- |
| Tutorials   | 6      | 2     |
| How-to      | 2      | ~12   |
| Reference   | 3      | 10    |
| Explanation | 4      | 8     |

This is **re-homing, not deletion.** The demoted tutorial `.jl` files are good
material — they move across quadrants (type tours → reference prose, DFAX →
how-to, reduction walkthrough → tutorial + how-to split). Preserve their content.

* * *

## 2. Repo facts & conventions

### 2.1 Docs toolchain

  - **Documenter.jl** + **Literate.jl** + **DocumenterInterLinks**.
  - Entry point: `docs/make.jl`. Helper: `docs/make_tutorials.jl` (defines
    `make_literate_folder`).
  - InterLinks are configured for `PowerSystems` and `PowerSystemCaseBuilder` —
    cross-references to those packages resolve automatically.

### 2.2 Folder layout (`docs/src/`)

  - `tutorials/` and `how_to_guides/` — **Literate `.jl` source files**. Each
    `foo.jl` is rendered to `generated_foo.md` (+ `.ipynb`) by
    `make_literate_folder`. **Edit the `.jl`, never the `generated_*.md`** (they
    are build artifacts; `clean_old_generated_files` deletes stale ones each build).
  - `explanation/` and `reference/` — **plain hand-written `.md`**.
  - `index.md` — landing page (currently thin; expansion candidate).
  - `assets/` — images/CSS.

### 2.3 Moving a page between quadrants

  - **To move a Literate page** (e.g. `tutorial_DFAX.jl` → how-to): move the `.jl`
    into the destination Literate folder. `make_literate_folder("how_to_guides")`
    will pick it up automatically. Delete the old `generated_*` artifacts.
  - **To demote a Literate tutorial to reference/explanation prose:** extract the
    narrative/code into a new hand-written `.md` under `reference/` or
    `explanation/`, then delete the source `.jl` from `tutorials/`.
  - **Every add/move/delete must be reflected in the `pages` OrderedDict in
    `docs/make.jl`** (see §5, package NAV). A page not listed there is invisible.

### 2.4 Build & verify

```sh
julia --project=docs -e 'using Pkg; Pkg.instantiate()'   # first time
julia --project=docs docs/make.jl
```

A clean build with no Documenter `@ref`/cross-reference errors and no missing
docstrings is the bar. Prefer the repo's `run.sh` wrapper if present in the
environment so output is captured.

### 2.5 Source-of-truth for the API

Two inventories informed this plan: a full PNM public-API inventory and a PNM
usage inventory from PowerSimulations.jl (PSI, the primary downstream consumer).
Key modules to read when drafting reference:

  - `src/PowerNetworkMatrices.jl` — module file: **all `export`s** (lines ~3–60)
    and the include order (~78–154). This is the definitive list of the public
    surface.
  - Per-type source files listed in each Reference package below.

* * *

## 3. Global correctness gotchas (READ BEFORE DRAFTING ANYTHING)

These are the facts most likely to be documented **wrong**. Honor them everywhere.

 1. **Reduction is supplied only via `network_reductions::Vector{NetworkReduction}`.**
    There is **no** `reduce_radial_branches` / `reduce_degree_two_branches` kwarg
    on PNM constructors. (Those flags exist in PSI's `NetworkModel` wrapper, which
    *translates* them into a `network_reductions` vector — do not attribute them
    to PNM.) Usage: `network_reductions = [RadialReduction(), DegreeTwoReduction(), WardReduction(study_buses)]`.

 2. **Reduction spec constructors:**
    
      + `RadialReduction()` — no fields.
      + `DegreeTwoReduction(; reduce_reactive_power_injectors = true)`.
      + `WardReduction(study_buses::Vector{Int})` — requires the study-bus set.
 3. **`dist_slack` type differs by matrix:** `Dict{Int,Float64}` (bus→weight) for
    `PTDF` / `VirtualPTDF`; `Vector{Float64}` for `VirtualLODF` / `VirtualMODF`.
    Default is empty (single reference bus).
 4. **`tol` is `Union{Float64, AutoTolerance}`**, default `DEFAULT_AUTO_TOLERANCE`.
    Do not describe it as a plain float default.
 5. **There is no dense `MODF` type.** Only `VirtualMODF` exists. (Contrast with
    PTDF/LODF, which have both dense and virtual forms.)
 6. **Serialization is PTDF-only and lossy for reductions.** `to_hdf5` /
    `from_hdf5` support **only** `PTDF`. Reduction data is **not** persisted — a
    deserialized PTDF is rehydrated with an *empty* `NetworkReductionData`. There
    is no Arrow/JLD/generic `save`/`load`, and no `Base.serialize` overload. State
    this limitation explicitly wherever serialization is mentioned.
 7. **`ThreeWindingTransformerWinding` is internal (not exported).** Reference it
    as an internal type; do not present it as a public constructor users call.
 8. **Contingency registration is automatic; there is no public
    `register_contingency`.** `VirtualMODF` auto-registers `PSY.Outage`
    supplemental attributes from the source system at construction/query time.
    Query with `get_registered_contingencies(vmodf) -> Dict{UUID, ContingencySpec}`.
 9. **Concurrency:** a process-wide `_LIBKLU_LOCK` serializes all libklu activity;
    virtual matrices also hold per-cache `ReentrantLock`s. Concurrent `getindex`
    is *safe* but *serialized* through the solver. Do not claim parallel speedup
    from threading factor solves.
10. **Solver backends:** KLU (`src/KLUWrapper/`) and Apple Accelerate
    (`src/AccelerateWrapper/`) are **always-present submodules** (Accelerate is
    runtime-gated on Apple hardware). **MKL Pardiso is a weak-dependency
    extension** (`ext/MKLPardisoExt.jl`, needs `Pardiso.jl`). Do not call KLU or
    Accelerate "extensions."
11. **`ArcAdmittanceMatrix` is built as part of `Ybus`**, not independently
    constructed by typical users.

* * *

## 4. Target structure (final state)

Legend: ✅ stays put · ➡️ moved in (provenance) · 🔶 expand existing · 🆕 net-new.

### 📘 Tutorials (exactly two)

 1. ✅🔶 Getting started
 2. ✅ Reduce a network and see the effect

### 🔧 How-to guides

 1. ✅ Compute the network matrices
 2. ✅ Choose a linear solver
 3. ➡️ Reproduce industry DFAX values
 4. 🆕 Apply network reductions at construction
 5. 🆕 Set the sparsification tolerance
 6. 🆕 Use a distributed slack
 7. 🆕 Inspect a reduced network
 8. 🆕 Define and apply contingencies (VirtualMODF)
 9. 🆕 Factorize and reuse an ABA matrix
10. 🔶 Persist a PTDF to disk
11. 🆕 Tune and clear the virtual-matrix cache
12. 🆕 Check connectivity / find islands

### 📚 Reference

 1. ✅🔶 Matrix overview & indexing (hub)
 2. 🆕➡️ Matrix type reference
 3. 🆕 Accessor functions
 4. 🆕 Network reduction reference
 5. 🆕 Aggregated-branch types
 6. 🆕 Contingency & modification types
 7. 🆕 Tolerance & solver settings
 8. 🆕 Serialization
 9. ✅🔶 Full public API (curated autodocs)
10. ✅ Internals

### 💡 Explanation

 1. ✅ The DC power flow approximation
 2. ✅ Computational considerations
 3. ✅🔶 Network reduction theory
 4. ✅🔶 Flowgate / post-contingency methodology
 5. 🆕 Virtual vs. materialized matrices
 6. 🆕 Concurrency and the KLU lock
 7. 🆕 Equivalent representation of reduced branches
 8. 🆕 Slack distribution & reference-bus conventions

* * *

## 5. Migration map: where every existing page lands

| Existing page (path under `docs/src/`)            | Now         | New home                    | Action                                                                     |
|:------------------------------------------------- |:----------- |:--------------------------- |:-------------------------------------------------------------------------- |
| `tutorials/getting_started.jl`                    | Tutorial    | Tutorial                    | Keep; tighten to one journey (pull compute-and-read core from DF page)     |
| `tutorials/tutorial_NetworkReduction.jl`          | Tutorial    | Tutorial                    | Keep (only existing page with a real learner's arc)                        |
| `tutorials/network_matrices.jl`                   | Tutorial    | Reference                   | Merge into `reference/network_matrices_overview.md`                        |
| `tutorials/tutorial_Incidence_BA_ABA_matrices.jl` | Tutorial    | Reference + Explanation     | Type prose → matrix type reference; concepts → reduction theory            |
| `tutorials/tutorial_DF_matrices.jl`               | Tutorial    | Reference (+ seed tutorial) | Split: compute-and-read core → Getting Started; rest → type reference      |
| `tutorials/tutorial_DFAX.jl`                      | Tutorial    | How-to                      | Move `.jl` → `how_to_guides/`; reframe intro from "learn" to "to validate" |
| `how_to_guides/compute_network_matrices.jl`       | How-to      | How-to                      | Keep                                                                       |
| `how_to_guides/choose_linear_solver.jl`           | How-to      | How-to                      | Keep                                                                       |
| `explanation/computational_considerations.md`     | Explanation | Explanation                 | Keep                                                                       |
| `explanation/dc_power_flow_approximation.md`      | Explanation | Explanation                 | Keep                                                                       |
| `explanation/network_reduction_theory.md`         | Explanation | Explanation                 | Keep; absorb concepts from Incidence/BA/ABA tour                           |
| `explanation/flowgate_methodology.md`             | Explanation | Explanation                 | Keep; expand                                                               |
| `reference/network_matrices_overview.md`          | Reference   | Reference                   | Keep; becomes the hub                                                      |
| `reference/public.md`                             | Reference   | Reference                   | Keep; curate the autodocs                                                  |
| `reference/internals.md`                          | Reference   | Reference                   | Keep                                                                       |

* * *

## 6. Work packages

Each package is independently assignable. Format: **target file**, **sources**,
**must cover**, **guardrails**, **depends on**, **done when**.

> Anchors below are `file:line` from authoring-time inventory — re-verify by
> opening the file. All paths under `docs/src/` unless noted. All `src/…` paths
> are the PNM package source (`/PowerNetworkMatrices.jl/src/…`).

### Tutorials

  - [ ] **T1 — Rewrite "Getting Started" as one journey**
    
      + Target: `tutorials/getting_started.jl`
      + Sources: existing `getting_started.jl`; lift the compute-and-read-a-value
        core from `tutorial_DF_matrices.jl`.
      + Must cover: load a `PSY.System` (use PowerSystemCaseBuilder, InterLinks are
        set up) → `PTDF(sys)` → index one meaningful sensitivity (e.g.
        `ptdf["branch_name", bus_number]`) → one sentence on what it means. Single
        path, single payoff.
      + Guardrails: NOT a tour of every matrix. No LODF/BA/ABA/Ybus detours. If you
        find yourself introducing a second matrix type, stop — that's reference.
      + Depends on: none (but coordinate with RP2 so the DF content isn't lost when
        that page is deleted).
      + Done when: builds; a newcomer can follow start→success without cross-links to
        reference to proceed.

  - [ ] **T2 — Refine "Reduce a network" tutorial**
    
      + Target: `tutorials/tutorial_NetworkReduction.jl`
      + Sources: existing file.
      + Must cover: a full network → apply `[RadialReduction(), DegreeTwoReduction()]`
        → show the before/after (bus count, removed arcs) → one payoff (smaller PTDF,
        same key sensitivities). Keep it a journey.
      + Guardrails: theory belongs in `explanation/network_reduction_theory.md` —
        link out, don't inline the math. Honor §3.1–3.2.
      + Depends on: none.
      + Done when: builds; reads as a guided arc, not an API dump.
  - [ ] **T-CLEANUP — Remove demoted tutorials from the quadrant**
    
      + Delete (after content has landed elsewhere): `tutorials/network_matrices.jl`,
        `tutorials/tutorial_Incidence_BA_ABA_matrices.jl`,
        `tutorials/tutorial_DF_matrices.jl`, and move `tutorials/tutorial_DFAX.jl`
        (see H3). Delete their stale `generated_*` artifacts.
      + Depends on: **RP1, RP2, H3, E3** (their content must be re-homed first).
      + Done when: `tutorials/` contains only `getting_started.jl` and
        `tutorial_NetworkReduction.jl` (+ their generated artifacts); nav updated (NAV).

### How-to guides

  - [ ] **H1 — Compute the network matrices** — Target `how_to_guides/compute_network_matrices.jl`. Keep as-is; light copy-edit only. Done when still builds.

  - [ ] **H2 — Choose a linear solver** — Target `how_to_guides/choose_linear_solver.jl`. Keep. Ensure §3.10 backend framing is correct (KLU/Accelerate submodules vs. MKL Pardiso extension). Done when accurate + builds.
  - [ ] **H3 — Reproduce industry DFAX values**
    
      + Target: move `tutorials/tutorial_DFAX.jl` → `how_to_guides/tutorial_DFAX.jl` (consider renaming to `reproduce_dfax_values.jl`).
      + Action: reframe the intro from "let's learn" to "to validate PNM against known DFAX numbers, do this." Preserve the validation numbers exactly — this reproduce-known-industry-values recipe is high-trust and rare.
      + Depends on: none. Coordinates with T-CLEANUP + NAV.
      + Done when: renders under How-to; nav updated.
  - [ ] **H4 — Apply network reductions at construction**
    
      + Target: `how_to_guides/apply_network_reductions.jl` (new Literate).
      + Must cover: passing `network_reductions=[…]` to `PTDF`/`Ybus`/etc.; the three
        spec types (§3.2); `WardReduction(study_buses)`; interaction with `tol`.
      + Guardrails: §3.1–3.2. Mechanics only; theory → explanation.
      + Done when: builds; nav updated.
  - [ ] **H5 — Set the sparsification tolerance**
    
      + Target: `how_to_guides/set_tolerance.jl` (new).
      + Must cover: fixed `tol::Float64` vs. `AutoTolerance(; data_precision, safety, quantile)`; `discover_data_precision`; `DEFAULT_AUTO_TOLERANCE`. Source: `src/auto_tolerance.jl` (`AutoTolerance` ~:26/39, `discover_data_precision` ~:126, cutoff types ~:63–71).
      + Done when: builds; nav updated.
  - [ ] **H6 — Use a distributed slack**
    
      + Target: `how_to_guides/distributed_slack.jl` (new).
      + Must cover: `dist_slack` for PTDF (`Dict{Int,Float64}`) and virtual LODF/MODF (`Vector{Float64}`) — §3.3. Show a weighted example.
      + Done when: builds; nav updated.
  - [ ] **H7 — Inspect a reduced network**
    
      + Target: `how_to_guides/inspect_reduced_network.jl` (new).
      + Must cover reading `NetworkReductionData`: `get_bus_reduction_map`,
        `get_mapped_bus_number`, `get_removed_buses`, `get_removed_arcs`,
        `has_radial_reduction`/`has_degree_two_reduction`/`has_ward_reduction`,
        `get_reductions`. Source: `src/NetworkReductionData.jl` (accessors ~:417–484,
        `get_mapped_bus_number` ~:574/585).
      + Done when: builds; nav updated.
  - [ ] **H8 — Define and apply contingencies (VirtualMODF)**
    
      + Target: `how_to_guides/contingencies.jl` (new). **Highest-value how-to.**
      + Must cover: build `VirtualMODF(sys)`; how `PSY.Outage` attributes
        auto-register (§3.8); `get_registered_contingencies`; query
        `vmodf[monitored_arc, spec]` where spec is `ContingencySpec` /
        `NetworkModification` / `PSY.Outage`; manual modifications
        (`ArcModification`, `ShuntModification`, `NetworkModification`);
        `get_post_modification_ptdf_row`. Sources: `src/modf_definitions.jl`,
        `src/virtual_modf_calculations.jl` (~:85, :226, :283),
        `src/virtual_ptdf_modification.jl`.
      + Guardrails: §3.5 (no dense MODF), §3.8 (no `register_contingency`).
      + Done when: builds; nav updated.
  - [ ] **H9 — Factorize and reuse an ABA matrix**
    
      + Target: `how_to_guides/factorize_aba.jl` (new).
      + Must cover: `ABA_Matrix(sys; factorize=true)`, `factorize(aba)`,
        `is_factorized(aba)`. Source: `src/BA_ABA_matrices.jl` (~:274, :367, :391).
      + Done when: builds; nav updated.
  - [ ] **H10 — Persist a PTDF to disk**
    
      + Target: `how_to_guides/serialize_ptdf.jl` (new).
      + Must cover: `to_hdf5(ptdf, filename; compress, compression_level, force)`,
        `from_hdf5(PTDF, filename)`, and the `PTDF(filename)` convenience. Source:
        `src/serialization.jl` (~:11, :54), `src/ptdf_calculations.jl:78`.
      + Guardrails: **§3.6** — PTDF-only; reductions not persisted. Say so loudly.
      + Done when: builds; nav updated.
  - [ ] **H11 — Tune and clear the virtual-matrix cache**
    
      + Target: `how_to_guides/virtual_cache.jl` (new).
      + Must cover: `max_cache_size`, `persistent_lines` construction kwargs;
        `clear_caches!` / `clear_all_caches!`. Source: `src/row_cache.jl`,
        `src/virtual_modf_calculations.jl` (~:615, :634).
      + Done when: builds; nav updated.
  - [ ] **H12 — Check connectivity / find islands**
    
      + Target: `how_to_guides/connectivity.jl` (new).
      + Must cover: `validate_connectivity`, `find_subnetworks`,
        `iterative_union_find`, `depth_first_search`. Source:
        `src/connectivity_checks.jl` (~:42, :149, :195, :250), `src/subnetworks.jl`.
      + Done when: builds; nav updated.

### Reference

  - [ ] **RP1 — Matrix overview & indexing (the hub)**
    
      + Target: `reference/network_matrices_overview.md`
      + Sources: existing overview; merge in `tutorials/network_matrices.jl`.
      + Must cover: the per-type summary table; **how `A[row, col]` resolves** —
        row/col may be bus number, arc tuple `(from,to)`, `PSY.ACBus`, `PSY.Arc`,
        `PSY.ACBranch`, branch-name `String`, `Colon`, or `PowerNetworkMatrixKey`.
        Source: `src/PowerNetworkMatrix.jl` (getindex/`to_index` ~:129–311).
      + Depends on: none. (Frees `network_matrices.jl` for deletion by T-CLEANUP.)
      + Done when: overview is the single entry point that links to RP2–RP8.

  - [ ] **RP2 — Matrix type reference** ⭐ *unblocks tutorial cleanup*
    
      + Target: `reference/matrix_types.md` (new).
      + Sources: absorb the type tours from
        `tutorial_Incidence_BA_ABA_matrices.jl` and `tutorial_DF_matrices.jl`.
      + Must cover, one subsection each, with constructor signatures + kwargs +
        "what it represents": `PTDF` (`src/ptdf_calculations.jl:50`, ctors :352/:422/:518/:78),
        `LODF` (`src/lodf_calculations.jl:49`, ctors :307/:372/:475),
        `VirtualPTDF` (`src/virtual_ptdf_calculations.jl:77`, :144/:218),
        `VirtualLODF` (`src/virtual_lodf_calculations.jl:77`, :276),
        `VirtualMODF` (`src/virtual_modf_calculations.jl:85`, :283),
        `Ybus` (`src/Ybus.jl:50`, :954),
        `ArcAdmittanceMatrix` (`src/ArcAdmittanceMatrix.jl:18`),
        `BA_Matrix` (`src/BA_ABA_matrices.jl:25`, :72/:111),
        `ABA_Matrix` (`src/BA_ABA_matrices.jl:211`, :274/:321),
        `IncidenceMatrix` (`src/IncidenceMatrix.jl:43`, :133/:182),
        `AdjacencyMatrix` (`src/AdjacencyMatrix.jl:42`, :116/:132),
        plus concrete aliases (`src/PowerflowMatrixTypes.jl`:
        `DC_PTDF_Matrix`, `DC_vPTDF_Matrix`, `DC_BA_Matrix`,
        `DC_ABA_Matrix_Factorized/Unfactorized`, `AC_Ybus_Matrix`) and the
        supertype `PowerNetworkMatrix{T} <: AbstractArray{T,2}`
        (`src/PowerNetworkMatrix.jl:11`).
      + Guardrails: §3.3 (`dist_slack` type per matrix), §3.4 (`tol`), §3.5 (no dense
        MODF), §3.11 (ArcAdmittance built via Ybus).
      + Depends on: none.
      + Done when: every exported matrix type has a subsection; T-CLEANUP can proceed.
  - [ ] **RP3 — Accessor functions**
    
      + Target: `reference/accessors.md` (new).
      + Must cover: `get_ptdf_data` (`src/ptdf_calculations.jl:603`), `get_lodf_data`
        (dense `src/lodf_calculations.jl:560`; virtual `src/virtual_lodf_calculations.jl:483`),
        `get_partial_lodf_row`, axes/lookups (`get_axes`, `get_lookup`,
        `get_bus_axis`, `get_arc_axis`, `get_bus_lookup`, `get_arc_lookup`),
        `get_ref_bus`/`get_ref_bus_position`, `get_network_reduction_data`,
        `get_system_uuid`. Source: `src/PowerNetworkMatrix.jl` (~:313–350).
      + Done when: builds; linked from RP1.
  - [ ] **RP4 — Network reduction reference**
    
      + Target: `reference/network_reduction.md` (new).
      + Must cover: spec types `NetworkReduction` (abstract, `src/NetworkReduction.jl:13`),
        `RadialReduction` (`src/radial_reduction.jl:7`), `DegreeTwoReduction`
        (`src/degree_two_reduction.jl:17`), `WardReduction` (`src/ward_reduction.jl:21`);
        `NetworkReductionData` + its ~40 accessors (`src/NetworkReductionData.jl`);
        `ReductionContainer` (`src/ReductionContainer.jl:8`); predicates.
      + Guardrails: §3.1–3.2.
      + Done when: builds; linked from RP1 and H4/H7.
  - [ ] **RP5 — Aggregated-branch types**
    
      + Target: `reference/aggregated_branches.md` (new).
      + Must cover: `BranchesSeries` (`src/BranchesSeries.jl:1`), `BranchesParallel` /
        `MixedBranchesParallel` / `AbstractBranchesParallel` (`src/BranchesParallel.jl`),
        `ThreeWindingTransformerWinding` (internal — `src/ThreeWindingTransformerWinding.jl:16`),
        `EquivalentBranch` (`src/EquivalentBranch.jl:16`); equivalent-parameter
        accessors (`get_equivalent_r/x/b/tap/rating/emergency_rating/available/α`) and
        rating strategies `get_sum_of_max_rating`, `get_single_element_contingency_rating`,
        `get_impedance_averaged_rating`; `get_equivalent_physical_branch_parameters`
        (`src/common.jl:381`).
      + Guardrails: §3.7 (mark `ThreeWindingTransformerWinding` internal).
      + Done when: builds; linked from E7.
  - [ ] **RP6 — Contingency & modification types**
    
      + Target: `reference/contingencies.md` (new).
      + Must cover: `ArcModification`, `ShuntModification`, `NetworkModification`,
        `ContingencySpec`, `WoodburyFactors` (`src/modf_definitions.jl`);
        `apply_ybus_modification`, `compute_ybus_delta` (`src/network_modification.jl`);
        `get_post_modification_ptdf_row`, `apply_woodbury_correction`,
        `compute_woodbury_factors` (`src/virtual_ptdf_modification.jl`);
        `get_registered_contingencies` (`src/virtual_modf_calculations.jl:226`).
      + Guardrails: §3.8.
      + Done when: builds; linked from H8 and E4.
  - [ ] **RP7 — Tolerance & solver settings**
    
      + Target: `reference/tolerance_and_solvers.md` (new).
      + Must cover: `AutoTolerance`, cutoff types (`AbsoluteCutoff`/`RelativeCutoff`/
        `SparsificationCutoff`), `discover_data_precision`, `DEFAULT_AUTO_TOLERANCE`
        (`src/auto_tolerance.jl`); `LinearSolverType` variants `KLUSolver`/
        `DenseSolver`/`MKLPardisoSolver`/`AppleAccelerateLUSolver`
        (`src/definitions.jl:26–30`); backend preference helpers
        (`src/linalg_settings.jl`).
      + Guardrails: §3.10 (backend classification).
      + Done when: builds; linked from H2/H5.
  - [ ] **RP8 — Serialization reference**
    
      + Target: `reference/serialization.md` (new).
      + Must cover: `to_hdf5` / `from_hdf5` (`src/serialization.jl:11/:54`).
      + Guardrails: §3.6 — PTDF-only, reductions not persisted, no other formats.
      + Done when: builds; linked from H10.
  - [ ] **RP9 — Curate the public API autodocs**
    
      + Target: `reference/public.md`
      + Action: today this is a single `@autodocs Public=true` block carrying the
        entire API with no curation. Split/group it (matrix types, reductions,
        aggregated branches, contingencies, solvers, serialization) so it complements
        RP2–RP8 rather than duplicating them raw. Keep `@autodocs` sourcing so
        docstrings stay authoritative.
      + Depends on: soft dependency on RP2–RP8 (mirror their grouping).
      + Done when: grouped, builds, no missing-docstring warnings.
  - [ ] **RP10 — Internals** — Target `reference/internals.md`. Keep (KLUWrapper / AccelerateWrapper autodocs). Verify it still builds. Done when builds.

### Explanation

  - [ ] **E1 — DC power flow approximation** — `explanation/dc_power_flow_approximation.md`. Keep; copy-edit. Done when builds.
  - [ ] **E2 — Computational considerations** — `explanation/computational_considerations.md`. Keep. Done when builds.
  - [ ] **E3 — Network reduction theory** — `explanation/network_reduction_theory.md`. Keep; **absorb the conceptual material** from `tutorial_Incidence_BA_ABA_matrices.jl` (what Incidence/BA/ABA *mean* and why they underpin reduction). Frees that tour for deletion (with RP2). Done when builds; concepts re-homed.
  - [ ] **E4 — Flowgate / post-contingency methodology** — `explanation/flowgate_methodology.md`. Keep; expand the `VirtualMODF` + Woodbury narrative. Link to RP6/H8. Done when builds.
  - [ ] **E5 — Virtual vs. materialized matrices** — `explanation/virtual_vs_materialized.md` (new). The lazy/LRU trade-off, `persistent_lines`, when each wins. Source: `src/row_cache.jl`. Done when builds.
  - [ ] **E6 — Concurrency and the KLU lock** — `explanation/concurrency.md` (new). Why solves serialize (`_LIBKLU_LOCK`, `src/solver_dispatch.jl`), what threading is/ isn't safe. Guardrail §3.9. Done when builds.
  - [ ] **E7 — Equivalent representation of reduced branches** — `explanation/equivalent_branches.md` (new). How ratings/impedances aggregate across parallel/series/3-winding groups and *why the rating policies differ* (sum-of-max vs. single-element-contingency vs. impedance-averaged). Links to RP5. Done when builds.
  - [ ] **E8 — Slack distribution & reference-bus conventions** — `explanation/slack_conventions.md` (new). Single vs. distributed slack, reference-bus choice, effect on PTDF. Links to H6. Done when builds.

### Cross-cutting

  - [ ] **NAV — Update `docs/make.jl` nav + landing page**
    
      + Rewrite the `pages` OrderedDict to the §4 target structure: 2 tutorials, the
        how-to list, the reference list (overview → type ref → accessors → reduction →
        aggregated branches → contingencies → tolerance/solvers → serialization →
        public API → internals), the explanation list. Fix the existing "Considertaions"
        typo in the label while you're there.
      + Expand `index.md` into a real landing page (what PNM is, the matrix taxonomy,
        where to start per Diátaxis quadrant).
      + Depends on: **all** page-creating packages (a nav entry with no target fails
        the build). Do this **last**, or add entries incrementally as pages land.
      + Done when: `julia --project=docs docs/make.jl` is clean end-to-end.

* * *

## 7. Dependency graph & parallelization

Most packages are **independent** and can run in parallel. The ordering
constraints:

```
RP2 (matrix type ref) ─┐
RP1 (overview hub)     ├─► T-CLEANUP (delete demoted tutorial .jl) ─► NAV (last)
E3  (reduction theory) ┤
H3  (DFAX move)        ┘

All other RP*, H*, E* packages ──────────────────────────────────► NAV (last)
```

  - **Wave 1 (max fan-out, no deps):** T1, T2, H1, H2, H4–H12, RP1, RP2, RP3, RP4,
    RP5, RP6, RP7, RP8, RP10, E1, E2, E3, E4, E5, E6, E7, E8. (RP9 soft-depends on
    RP2–RP8 grouping — can start last in the wave.)
    
      + **Scheduling priority within Wave 1 — start the Wave 2 blockers first.**
        Wave 1 is otherwise unordered, but **RP2, RP1, E3, and H3 gate Wave 2** and
        should be assigned *before* the purely additive packages (H4–H12, E5–E8,
        RP3–RP8). **RP2 first of all:** it is the highest fan-in blocker — *two* of
        the four demoted tutorial files (`tutorial_Incidence_BA_ABA_matrices.jl` and
        `tutorial_DF_matrices.jl`) re-home their type content through it, so until
        RP2 lands, T-CLEANUP cannot run and the tutorial quadrant cannot shrink to
        its two-page target — the overhaul's headline outcome. The additive packages
        can finish anytime before NAV and never block cleanup.
      + **Invariant — re-home before delete.** No demoted tutorial `.jl` may be
        deleted (T-CLEANUP) until every package absorbing its content has landed. The
        per-file routing: `tutorial_DFAX.jl`→H3; `network_matrices.jl`→RP1;
        `tutorial_Incidence_BA_ABA_matrices.jl`→RP2 (types) + E3 (concepts);
        `tutorial_DF_matrices.jl`→RP2 (types) + T1 (one example).

  - **Wave 2:** T-CLEANUP + H3 file move (content already re-homed by RP1/RP2/E3).
  - **Wave 3:** NAV (nav + landing page), then a full build to verify.

**Coordination note for parallel agents:** T1 and RP2 both consume
`tutorial_DF_matrices.jl`. Agree that RP2 owns the type-tour material and T1 owns
only the single compute-and-read example. Do not both try to delete the file —
that is T-CLEANUP's job in Wave 2.

**Every agent:** honor §3 gotchas; edit `.jl` not `generated_*.md` in Literate
folders; add your page to `make.jl` nav (or leave a note for NAV); verify with
§2.4 before marking your checkbox.
