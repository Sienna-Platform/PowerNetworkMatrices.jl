# Specification: Flowgate derivation from LODF

**Target package:** PowerNetworkMatrices.jl
**Status:** proposed (verified against codebase 2026-06-09)
**Scope:** one new source file, one exported type, one exported function, accessors.

## 1. Purpose

Given a built `LODF`, derive *flowgates* and rank them by importance. A flowgate is a
contingency (an outaged arc) paired with the set of arcs whose flow that outage most
affects. The capability lets a user pick a small set of contingencies and monitored
elements to study, instead of running the full N-1 set. Everything is computed from the
LODF matrix the package already builds; no new linear algebra is required.

## 2. Definitions

  - **Arc.** The package's branch identifier: a `Tuple{Int, Int}` of `(from_bus, to_bus)`
    numbers. One arc may represent several parallel physical branches equivalenced together
    (`NetworkReductionData.parallel_branch_map`); an arc contingency therefore means the
    simultaneous outage of every parallel branch on that arc. The spec uses "arc"
    throughout; user-facing docs may say "branch" with this caveat.
  - **Contingency.** A single arc taken out of service.
  - **Monitored set.** The arcs most affected by a given contingency, ranked by the
    magnitude of their LODF sensitivity to it.
  - **Flowgate.** A contingency together with its ranked monitored set.
  - **Meshed arc.** An arc whose outage redistributes flow onto at least one other arc. A
    radial arc (a bridge) redistributes nothing — the unit from→to transfer that defines
    its LODF column flows entirely through the arc itself, so its column is zero off the
    diagonal up to numerical noise (the LODF islanding clamp at
    `LODF_ENTRY_TOLERANCE = 1e-6`, `src/definitions.jl:10`, keeps such columns finite
    instead of dividing by ~0). Radial arcs are not valid contingencies and are excluded.

## 3. LODF conventions to use

All of the following are verified against `src/lodf_calculations.jl`:

  - `M = get_lodf_data(lodf)` (defined at `src/lodf_calculations.jl:560`, exported from
    `src/PowerNetworkMatrices.jl:51`) returns `transpose(lodf.data)` — a **lazy
    `LinearAlgebra.Transpose` wrapper, not a copy** — in standard orientation:
    `M[i, j]` is the change in flow on arc `i` when arc `j` is outaged, normalized by the
    pre-outage flow on `j`. Column `j` of `M` is the effect of outaging arc `j` on every
    arc. Internally `lodf.data` is stored transposed (`stores_transpose(::LODF) = true`,
    line 65; `data[i, j]` = sensitivity of arc `j` to outage of arc `i`, struct docstring
    lines 11–12). See §8 for the performance consequences.
  - `get_arc_axis(lodf)` (defined at `src/lodf_calculations.jl:564`) returns
    `lodf.axes[1]::Vector{Tuple{Int, Int}}`, the arc identifiers in matrix-index order.
    Both axes are identical (`(get_arc_axis(A), get_arc_axis(A))`, lines 402, 411, 504).
    **Note:** `get_arc_axis` is *not* exported; the implementation lives inside the module
    so it may call it directly. Do not add it to the export list as part of this feature.
  - The identifier type is concretely `T = Tuple{Int, Int}`. `flowgates(::LODF)` returns
    `Vector{Flowgate{Tuple{Int, Int}}}`. Keep `Flowgate{T}` parametric for future axis
    types, but the LODF method is fully concrete.
  - Diagonal entries are structurally `-1.0` (set in `_apply_lodf_demand!`, line 97, and
    re-added after sparsification via `- LinearAlgebra.I`, line 520) and must be excluded
    from monitored sets and from all metrics.
  - Storage element types: by default (`tol = DEFAULT_AUTO_TOLERANCE`,
    `_dense_tol(::AutoTolerance) = eps(Float64)`, `src/auto_tolerance.jl:244`) **no
    sparsification happens** and `lodf.data` is a dense `Matrix{Float64}`. Only a
    user-supplied `tol::Float64 > eps()` produces a `SparseMatrixCSC{Float64, Int}` via
    `_sparsify_lodf` (lines 398–408, 499–501, 518–521). The implementation must handle
    both via multiple dispatch (§8).
  - **Out of scope:** `VirtualLODF`. `get_lodf_data(::VirtualLODF)` returns the row cache
    `Dict{Int, Vector{Float64}}` (`src/virtual_lodf_calculations.jl:483`), not a matrix.
    Restrict the method signature to `::LODF`; a virtual variant (e.g. via
    `get_partial_lodf_row`) is future work.

Multi-subnetwork systems need no special handling: an arc in a different subnetwork has
zero sensitivity to the contingency, so it is excluded naturally by the metrics below.

## 4. Functional requirements

For each arc `c` taken as a contingency:

 1. **Meshed test.** Compute the largest off-diagonal `|LODF|` in column `c` of `M`. If
    it does not exceed `radial_tol`, skip `c` (it is radial). This test is structural and
    derived entirely from the matrix.
 2. **Monitored set.** Rank the arcs by `|LODF[·, c]|` in descending order, excluding `c`
    itself and any arc with `|LODF| ≤ monitor_tol`. Keep at most `top_n`.
 3. **Importance score.** Compute the two features below over the off-diagonal entries
    `v` of column `c` with `|v| > monitor_tol`, then combine them (see §5).

Return one flowgate per meshed arc, sorted by score, most important first. Break score
ties by ascending matrix index of the contingency so the output order is deterministic.

## 5. Ranking metric

The score balances how *broadly* an outage spreads against how *hard* it hits.

  - **Breadth.** The number of arcs the outage affects. Default: the participation ratio
    `breadth = ‖v‖₁² / ‖v‖₂²`. This is a magnitude-weighted, threshold-free count — it
    equals the true count when all affected arcs carry equal flow, drops toward 1 when one
    arc dominates, and is invariant to scaling. With `effective_count = false`, use the
    literal count of entries with `|v| > monitor_tol` instead.
  - **Magnitude.** How much flow moves. Default: `magnitude = ‖v‖₂`.

Normalize each feature across all meshed contingencies to `[0, 1]` (min-max), then
combine:

```
score = breadth_weight · breadth_norm + (1 − breadth_weight) · magnitude_norm
```

`breadth_weight = 1` ranks purely by spread; `0` purely by magnitude; `0.5` splits them.

The participation ratio is preferred over a literal non-zero count because the LODF
matrix *may* be sparsified by a user-supplied tolerance (it is **not** by default; see
§3), in which case a literal count would depend on that tolerance rather than on the
physics. The participation ratio degrades gracefully: on a sparsified LODF both features
are computed over the surviving entries, so flowgates inherit the truncation of the
input matrix — document this in the `flowgates` docstring.

## 6. API

```julia
struct Flowgate{T}
    contingency::T                  # outaged arc
    monitored::Vector{T}            # affected arcs, most affected first
    sensitivities::Vector{Float64}  # signed LODF values aligned with `monitored`
    score::Float64                  # importance score in [0, 1]
    breadth::Float64                # raw breadth feature
    magnitude::Float64              # raw magnitude feature
end
```

For `flowgates(::LODF)`, `T = Tuple{Int, Int}` (§3). If an inner/outer constructor with
a body is needed, use the `function Flowgate(...) ... end` form per the style guide; the
plain default constructor is acceptable.

Accessors `get_flowgate_contingency`, `get_flowgate_monitored`, `get_flowgate_sensitivities`, `get_flowgate_score`,
`get_flowgate_breadth`, `get_flowgate_magnitude` follow the existing one-line getter pattern (e.g.
`get_axes(M::LODF) = M.axes`, `src/lodf_calculations.jl:59`):

```julia
get_flowgate_contingency(fg::Flowgate) = fg.contingency
```

No name collisions exist in the package for `Flowgate`, `flowgates`, or any of the six
accessors (verified by grep over `src/` and `test/`). `PowerSystems` is imported only
under the `PSY` alias (`import PowerSystems as PSY`, `src/PowerNetworkMatrices.jl:64`),
so its `get_monitored_components` etc. cannot clash.

```julia
flowgates(lodf::LODF;
    top_n::Int = 5,
    radial_tol::Float64 = 1e-5,
    monitor_tol::Float64 = 0.0,
    breadth_weight::Float64 = 0.5,
    effective_count::Bool = true)::Vector{Flowgate{Tuple{Int, Int}}}
```

| Keyword           | Default | Meaning                                                                                                                                                                                                                                             |
|:----------------- |:------- |:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `top_n`           | `5`     | Max arcs in each monitored set.                                                                                                                                                                                                                     |
| `radial_tol`      | `1e-5`  | An arc is meshed if its largest off-diagonal `|LODF|` exceeds this. Must sit above the numerical noise floor of radial columns (the islanding clamp `LODF_ENTRY_TOLERANCE = 1e-6` bounds the denominator, not the noise; `1e-5` is a safe default). |
| `monitor_tol`     | `0.0`   | Arcs with `|LODF|` at or below this are not monitored and do not count toward breadth/magnitude. Default drops exact zeros (test `abs(v) > monitor_tol`; for stored zeros use `iszero(v)` to skip, never `v == 0`).                                 |
| `breadth_weight`  | `0.5`   | Convex weight between breadth and magnitude in the score. Validate `0.0 ≤ breadth_weight ≤ 1.0` with `IS.@assert_op` (the package aliases `import InfrastructureSystems as IS`, `src/PowerNetworkMatrices.jl:63`). Likewise validate `top_n ≥ 1`.   |
| `effective_count` | `true`  | `true` uses the participation ratio for breadth; `false` uses the literal non-zero count. Branch on it with `if/else`, not a ternary.                                                                                                               |

`Flowgate`, `flowgates`, and the six accessors are exported. Add the `export` lines to
the export block at the top of `src/PowerNetworkMatrices.jl` (lines 3–60), following the
existing grouping.

## 7. Edge cases

  - **Fully radial / empty result.** If no arc is meshed, return an empty
    `Vector{Flowgate{Tuple{Int, Int}}}`.
  - **Degenerate normalization.** Min-max normalization is degenerate when there is a
    single meshed contingency *or* when a feature is constant across all meshed
    contingencies (`max == min`); in both cases define that normalized feature as `1.0`
    so the score is well defined.
  - **`top_n ≥ number of arcs`.** Return all available monitored arcs.
  - **Ties in `|LODF|`.** Break ties by first-encountered storage order; the result must
    be deterministic for a given input.
  - **Diagonal.** The self term must never appear in a monitored set. Skip entries with
    `row == col` during accumulation; do not rely on the `-1.0` value (off-diagonal values
    can legitimately reach `±1`).

## 8. Performance requirements

  - Reuse the existing LODF data; do not recompute LODF, PTDF, or any factorization.
  - Total cost `O(n²)` for `n` arcs (dense) or `O(nnz)` (sparse) — at most the order of
    building the LODF — with `O(top_n)` extra work per accepted entry.
  - **Storage-order traversal, not column views.** Because `get_lodf_data` returns a lazy
    `Transpose`, column `c` of `M` is **row** `c` of the stored parent — a stride-`n` view
    on the dense default and an `O(nnz)` scan per row on CSC. Do **not** iterate column
    views of `M`. Instead make a single streaming pass over `P = parent(get_lodf_data(lodf))`
    in storage order, where `P[i, j]` is the sensitivity of monitored arc `j` to outage of
    arc `i`: every visited entry updates contingency `i`'s accumulators
    (running `max|v|`, `Σ|v|`, `Σv²`, count above `monitor_tol`, and a bounded
    top-`top_n` buffer of `(|v|, v, j)`). One pass yields everything §4 needs with optimal
    locality for both layouts. No transposed copy is ever materialized.
  - Provide two accumulation methods selected by multiple dispatch on the parent type —
    `::Matrix{Float64}` (nested column-major loops, `@inbounds` on the hot loop) and
    `::SparseMatrixCSC{Float64, Int}` (iterate `nzrange`/`rowvals`/`nonzeros`; the
    structural `-1.0` diagonal is stored, so the `row == col` skip still applies). Never
    use `isa` checks.
  - Bounded top-`top_n` selection per contingency: insertion into a fixed-size sorted
    buffer is sufficient for the small default (`top_n = 5`); a
    `DataStructures` heap (already a dependency) is acceptable for large `top_n`. Do not
    fully sort columns.
  - Memory: `O(n · top_n)` for the selection buffers plus `O(n)` for the accumulators.
  - The implementation must be type-stable. `@code_warntype flowgates(lodf)` should show a
    concrete return type with no `Any` or wide `Union` in the per-contingency loop. The
    arc axis eltype is concrete (`Tuple{Int, Int}`) through the `LODF{Ax, ...}` type
    parameter, so this is achievable without function barriers; if the
    dense/sparse dispatch is reached through an abstract field, add one function barrier
    at the accumulation call.
  - Comments: terse, explain the non-obvious *why* (e.g. why storage order, why the
    diagonal skip is positional), not the *how*.

## 9. Testing and acceptance

Tests go in a new file `test/test_flowgates.jl`; the runner auto-includes any
`test/test_*.jl` file (`test/PowerNetworkMatricesTests.jl:44-48`), so no registration is
needed. The harness is **ReTest** (`using ReTest`), with aliases `PSB`
(`PowerSystemCaseBuilder`) and `PNM` already set up in the test module. Build systems
exactly as the existing LODF tests do (`test/test_lodf.jl:7`):

```julia
sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
lodf = LODF(sys5)
```

  - A known meshed arc in `c_sys5` (fully meshed, 5-bus PJM) produces the expected
    monitored set and ranking; cross-check `sensitivities` against direct
    `lodf[monitored_arc, outaged_arc]` indexing.
  - A radial arc is excluded. Use `c_sys14` — bus 8 hangs radially off bus 7, and this
    case is already the radial fixture in `test/test_lodf_with_radial_branches.jl:4`
    (which also exercises `"test_RTS_GMLC_sys"` for a larger meshed/radial mix).
  - A sparsified LODF (`LODF(sys; tol = 1e-3)`) exercises the `SparseMatrixCSC` dispatch
    path and returns results consistent with the dense path above `tol`.
  - The diagonal (self arc) never appears in any monitored set.
  - `breadth_weight = 1` and `= 0` produce the pure-breadth and pure-magnitude orderings.
  - `effective_count = false` switches breadth to the literal count.
  - Output is deterministic across repeated calls.
  - Return type is concretely inferred: follow the existing pattern
    `@test (@inferred flowgates(lodf)) isa Vector{Flowgate{Tuple{Int, Int}}}`
    (cf. `test/test_auto_tolerance.jl:32`).

Run with the documented commands (`julia --project=test test/runtests.jl` after
`Pkg.develop(path=".")` in the test project), and run the formatter
(`julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`)
before completion.

Validation (recommended, not a merge blocker): on a meshed reference case such as
`"test_RTS_GMLC_sys"`, confirm that the highest-ranked flowgates correspond to the
contingencies whose actual N-1 results produce the largest monitored-flow changes. Use
this to set a defensible default for `breadth_weight`.

## 10. Integration

  - New file `src/flowgates.jl`. Add `include("flowgates.jl")` to
    `src/PowerNetworkMatrices.jl` **after** `include("lodf_calculations.jl")` (line 144) —
    the file references the `LODF` type, and include order is load-bearing in this module.
    Immediately after line 144 (before `virtual_lodf_calculations.jl`) is the natural slot.
  - Add exports (`Flowgate`, `flowgates`, six accessors) to the export block in
    `src/PowerNetworkMatrices.jl`.
  - Depends only on `get_lodf_data` (public) and `get_arc_axis` (internal,
    module-visible) plus `parent` on the returned `Transpose`.
  - Every exported symbol needs a docstring: the module applies a
    `@template DEFAULT` with `$(SIGNATURES)` (`src/PowerNetworkMatrices.jl:106-109`), and
    `docs/src/reference/public.md` uses `@autodocs` with `Public = true`, so new exports
    are picked up automatically — a missing docstring fails the Documenter
    `missing_docs` check. Add a short docs example showing `flowgates(LODF(sys))`.

## 11. Decisions left to the implementer

  - **Normalization scheme.** Min-max is specified as the default. Divide-by-max
    (preserves ratios) or a rank transform (robust when a few near-bridge lines dominate
    the magnitude distribution) are reasonable alternatives and could be exposed as an
    option later.
  - **Default `breadth_weight`.** `0.5` is a placeholder pending the validation in §9.
  - **Flow weighting (future).** Ranking contingencies by absolute MW shift
    (`LODF × f_c`, with `f_c` the pre-outage flow on the outaged arc) is a useful
    operational variant but needs a base-case flow vector and is out of scope for the
    first version.
  - **VirtualLODF support (future).** Would route through `get_partial_lodf_row` rather
    than a full matrix; out of scope (§3).

## Verification notes

Everything below was checked against the working tree on branch `jd/make_flowgates`
(2026-06-09).

**Confirmed correct (unchanged in substance):**

  - `get_lodf_data(lodf::LODF)` exists at `src/lodf_calculations.jl:560-562` and is
    exported (`src/PowerNetworkMatrices.jl:51`). Orientation claim verified two ways: the
    struct docstring (`src/lodf_calculations.jl:11-12`, "Element (i,j) [of `data`]
    represents the sensitivity of line j flow to line i outage") and
    `Base.getindex(A::LODF, selected_arc, outage_arc) = A.data[outage_idx, selected_idx]`
    (lines 536-539). Since `get_lodf_data` returns `transpose(lodf.data)`,
    `M[i, j]` = effect on arc `i` of outaging arc `j`, exactly as the spec claimed.
  - Diagonal is structurally `-1.0`: set in `_apply_lodf_demand!`
    (`src/lodf_calculations.jl:97`) and re-added post-sparsification as `- I`
    (lines 518-521, including the comment explaining why the diagonal is zeroed before
    `droptol!`).
  - `get_arc_axis(lodf::LODF)` exists at `src/lodf_calculations.jl:564-566`.
  - The radial-column-is-zero claim is consistent with the computation: the LODF column is
    the scaled `PTDF·A` transfer column, which is identically zero off-diagonal for a
    bridge; the islanding clamp `LODF_ENTRY_TOLERANCE = 1e-6` (`src/definitions.jl:10`,
    used at `src/lodf_calculations.jl:83`) keeps radial columns finite.
  - Multi-subnetwork no-op claim: cross-island PTDF transfer sensitivities are zero, and
    `subnetwork_axes` exists on `LODF` (line 54) without affecting `data` layout.
  - Name availability: no existing `Flowgate`, `flowgates`, `get_flowgate_contingency`,
    `get_flowgate_monitored`, `get_flowgate_sensitivities`, `get_flowgate_score`, `get_flowgate_breadth`, or `get_flowgate_magnitude`
    in `src/` or `test/` (grep). `PSY.get_monitored_components` is alias-scoped and cannot
    collide.
  - Test cases: `c_sys5` is built via `PSB.build_system(PSB.PSITestSystems, "c_sys5")` in
    `test/test_lodf.jl:7`; `c_sys14` and `"test_RTS_GMLC_sys"` are the radial fixtures in
    `test/test_lodf_with_radial_branches.jl:4`.

**Corrected:**

 1. **§8 column-view advice (critical fix).** The old spec said "operate on column
    views". `get_lodf_data` returns a lazy `Transpose` of a **transposed-stored** matrix
    (`stores_transpose(::LODF) = true`, `src/lodf_calculations.jl:65`; "NOTE: the LODF
    matrix is saved as transposed!", line 527). A column view of `M` is therefore a
    stride-`n` row view of the dense parent, or a worst-case row scan of a CSC parent.
    Replaced with a single streaming pass over `parent(M)` in storage order with
    per-contingency accumulators, dispatched on `Matrix{Float64}` vs
    `SparseMatrixCSC{Float64, Int}`. The "do not materialize a transposed copy" part was
    already satisfied for free (the wrapper is lazy) and is retained.
 2. **Storage element type and sparsification.** The old §5 implied the LODF "is
    sparsified by a tolerance". By default it is **not**: `LODF(...)` defaults to
    `tol = DEFAULT_AUTO_TOLERANCE` and `_dense_tol(::AutoTolerance) = eps(Float64)`
    (`src/auto_tolerance.jl:243-244`), which fails the `tol_value > eps()` gate
    (`src/lodf_calculations.jl:398, 499`), leaving dense `Matrix{Float64}` storage.
    Sparse CSC storage occurs only for user-supplied `tol > eps()`. §3, §5, and §9 now
    say "may be sparsified" and require testing both dispatch paths.
 3. **`T` made concrete.** The arc axis eltype is `Tuple{Int, Int}` — `(from_bus, to_bus)` — not branch-name strings: `get_arc_axis(::IncidenceMatrix) = M.axes[1]`
    built from `get_arc_axis(fb, tb, bus_axis)::Vector{Tuple{Int, Int}}`
    (`src/Ybus.jl:1145, 1175`), threaded into `LODF` axes at
    `src/lodf_calculations.jl:402/411/504`. §2 adds the parallel-branch caveat: one arc
    can stand for several physical branches (`get_branch_multiplier`,
    `src/PowerNetworkMatrix.jl:350`), so an arc contingency outages all of them.
    "Branch" was renamed to "arc" throughout the normative text.
 4. **`get_arc_axis` is not exported.** Only `get_lodf_data` is in the export list. §3
    and §10 now state the implementation uses it module-internally and must not export it.
 5. **`VirtualLODF` explicitly excluded.** `get_lodf_data(::VirtualLODF)` returns a
    `Dict{Int, Vector{Float64}}` row cache (`src/virtual_lodf_calculations.jl:483`), so a
    generic signature would silently break; the method is restricted to `::LODF`.
 6. **Degenerate normalization extended.** The old spec only covered the
    single-contingency case; a constant feature across multiple meshed contingencies
    (`max == min`) is the same degeneracy and gets the same `1.0` definition (§7).
 7. **Deterministic ordering tightened.** §4 now specifies the score tie-break (ascending
    matrix index) and §7 specifies the `|LODF|` tie-break (first-encountered storage
    order), making "deterministic" concrete.
 8. **Diagonal exclusion made positional.** Skip `row == col`, do not filter on the
    value `-1.0`, since off-diagonal LODF entries can reach `±1` (§7).
 9. **Project conventions folded in** (per repo guidance): multiple dispatch instead of
    `isa` (dense/sparse accumulation methods); `if/else` instead of ternaries
    (`effective_count` branch); `iszero(x)` instead of `x == 0`;
    `IS.@assert_op` for keyword validation (alias `import InfrastructureSystems as IS`,
    `src/PowerNetworkMatrices.jl:63`; usage pattern at `src/lodf_calculations.jl:89-90`);
    one-line getters matching `get_axes(M::LODF) = M.axes`
    (`src/lodf_calculations.jl:59`); `function Foo() ... end` form for any non-default
    constructor; terse why-comments.
10. **Integration specifics added** (§9, §10): include slot after
    `include("lodf_calculations.jl")` (`src/PowerNetworkMatrices.jl:144`); exports in
    the block at lines 3–60; test file auto-discovery via the `test_*.jl` glob
    (`test/PowerNetworkMatricesTests.jl:44-48`); ReTest harness with `PSB`/`PNM`/`IS`
    aliases; `@inferred` test pattern from `test/test_auto_tolerance.jl:32`; Documenter
    `@autodocs`/`missing_docs` implication of the module-wide docstring template
    (`src/PowerNetworkMatrices.jl:106-109`); formatter command requirement.
