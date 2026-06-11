# Spec: automatic sparsification tolerance for PTDF / LODF

## Why

PTDF and LODF are built by solving against the reduced susceptance matrix
`ABA = Aᵀ B A`. `ABA⁻¹` is dense even though `ABA` is sparse, so the factors come
out dense and we sparsify them with `droptol!`. The cutoff used to be a hand-set
`tol::Float64` — a guess: too small keeps round-off, too large discards real
sensitivities.

For **large** systems the matrices are built on demand (`VirtualPTDF` /
`VirtualLODF` / `VirtualMODF`): each requested row/column is computed, sparsified,
and cached. There the cutoff matters most — a dense row of an 80k-bus system is a
memory blow-up, while almost every entry is negligible (a branch is insensitive
to injections far away). The goal of this feature: **a requested column from a
large case comes back sparse, automatically, without the user choosing a number.**

## The mechanism: relative per-row drop

`AutoTolerance` sparsifies each computed row with a cutoff **relative to that
row's own peak**:

    drop entry j of row i  when  |row_i[j]| < α · max|row_i|
    α = clamp(safety · δ, 1e-6, 1e-2)

  - `δ` (`data_precision`) is the relative precision of the branch data. An entry
    smaller than `δ` relative to the row peak is not supported by the input data.
  - `safety` is an aggressiveness multiplier on `δ` (default `1.0`).
  - The clamp keeps `α` in `[1e-6, 1e-2]`: the floor stops full-precision data from
    collapsing `α` to zero (no sparsification), the ceiling keeps `α` well below 1
    so a row's own peak is never dropped.

Because the cutoff is relative to each row's peak, per-column sparsity is
**independent of the matrix scale and of the conditioning of `ABA`**. This is the
key property: it holds at any system size.

### Why not a condition-number bound

The first design used the perturbation bound `tol = safety · κ(ABA) · δ · scale`
as a single global, absolute cutoff. A stable solve amplifies a relative input
perturbation `δ` into a relative inverse error of about `κ(ABA)·δ`, so this is a
sound normwise bound. But it is the wrong tool for *guaranteeing sparse columns*:

  - `κ(ABA)` grows with system size (graph-Laplacian conditioning). For large grids
    `safety·κ·δ ≥ 1`, which trips the "tolerance reached matrix scale" guard and
    falls back to `eps()` — i.e. **no sparsification, dense columns**, the exact
    opposite of the goal. Confirmed empirically (e.g. `AutoTolerance(δ=1e-3)` on a
    κ≈6e5 case collapsed to `eps()`).
  - The normwise `κ` massively overestimates *per-entry* error: a branch's terminal
    PTDF entries (≈0.5) are well-determined regardless of `κ`.

So `κ` no longer enters the cutoff. It is still computed (cheaply, reusing the
factorization) and **logged as a diagnostic** so users can see how ill-conditioned
the network is, but it never scales the tolerance.

## The bus-count gate: only large systems are sparsified

Sparsification only pays off — and is only wanted — at scale. `AutoTolerance`
therefore sparsifies only the **on-demand (virtual)** matrices at or above
`AUTO_TOLERANCE_BUS_LIMIT` (= `2000`) buses:

| path                                          | `< 2000` buses     | `≥ 2000` buses        |
|:--------------------------------------------- |:------------------ |:--------------------- |
| `VirtualPTDF` / `VirtualLODF` / `VirtualMODF` | no-op (exact rows) | relative per-row drop |
| dense `PTDF` / `LODF` (any size)              | no-op (exact)      | no-op (exact)         |

  - **Small systems are never sparsified** under `AutoTolerance` — they are returned
    exactly, so existing results and tolerances are unchanged.
  - **The dense `PTDF`/`LODF` path never sparsifies** under `AutoTolerance`, at any
    size. The dense matrix is the small-system path; keeping it dense preserves its
    stored `Matrix{Float64}` type and the `DC_PTDF_Matrix` alias that downstream
    packages (PowerFlows, PowerSimulations) dispatch on. Large systems use the
    virtual matrices, which is where automatic sparsification happens.

## API

The `tol` keyword on every PTDF/LODF constructor is

    tol::Union{Float64, AutoTolerance} = AutoTolerance()   # data_precision = :auto

  - A **`Float64`** is an explicit **absolute** cutoff, honored verbatim at any size
    and on any path (backward compatible / exact). Use it to sparsify a dense or a
    small-system matrix explicitly, or to pin an exact result with `tol = eps()`.
  - An **`AutoTolerance`** is the default and follows the gate above. There is no
    `nothing` option.

```julia
struct AutoTolerance{D <: Union{Float64, Symbol}}
    data_precision::D     # a number, or :auto
    safety::Float64       # aggressiveness multiplier on δ (default 1.0)
    quantile::Float64     # for :auto discovery (default 0.5)
end

AutoTolerance(; data_precision = :auto, safety = 1.0, quantile = 0.5)
```

### Auto-discovery of `data_precision`

`data_precision = :auto` (the default) recovers `δ` from the branch data. The
precision is set by the significant figures the reactances were specified with,
but it is invisible in the susceptance `b = 1/x` (the reciprocal scrambles the
decimal structure). So discovery inverts the transform first — recovers
`x = 1/b` — then counts significant figures:

```julia
discover_data_precision(susceptances; q = 0.5, maxsig = 12, rtol = 1e-9) -> Float64
```

It returns `0.5·10^(-(s-1))` at the `q`-quantile of the per-branch
significant-figure counts, clamped to `[eps, 1e-2]`. Full-precision data (e.g.
computed equivalent branches from a reduction) reads `maxsig` figures and
collapses to the arithmetic floor — so `:auto` is conservative on clean data and
aggressive on coarse field data, as it should be.

A `Float64` `data_precision` (e.g. `1e-3` for reactances good to 0.1%) sets `δ`
directly and skips discovery.

## Implementation

`src/auto_tolerance.jl`:

  - `AutoTolerance`, `discover_data_precision`, `_relative_alpha` (the clamped
    `safety·δ`).

  - Resolved cutoff types stored on the matrix and applied per row:
    `AbsoluteCutoff(value)` (Float64 path) and `RelativeCutoff(fraction)`
    (AutoTolerance path); `const SparsificationCutoff = Union{…}`. `apply_cutoff`
    dispatches: absolute drops below a fixed value, relative drops below
    `fraction · max|row|`. `cutoff_value` returns the `Float64` for `get_tol`.
  - Resolvers:
    
      + `_resolve_virtual_cutoff(tol, cache, ABA, susceptances)` — `Float64 → AbsoluteCutoff`; `AutoTolerance →` `AbsoluteCutoff(eps())` below the gate, else
        `RelativeCutoff(_relative_alpha(...))` (and logs `κ`).
      + dense `PTDF`/`LODF`: `AutoTolerance →` `eps()` (no-op); `Float64` passes
        through. `_resolve_lodf_tol`, `_resolve_from_ptdf_tol`, and
        `_dense_ptdf_with_tol(::AutoTolerance, …)` implement this.
  - `κ` estimate: `condest!` for KLU; a symmetric Hager 1-norm estimator
    (`_hager_invnorm1`, `_norm1`) reusing the existing factorization for Apple
    Accelerate. Used only by `_log_condition_estimate` (diagnostic).

The virtual matrices store `tol::SparsificationCutoff` and apply it in
`cached_row_lookup` (`row_cache.jl`) when a row is first computed; `get_tol`
returns `cutoff_value`. The dense `PTDF`/`LODF` keep `tol::Base.RefValue{Float64}`
(serialization unchanged).

The structural `-1.0` LODF diagonal is re-asserted (`_restore_lodf_diagonal!`)
after any `droptol!` so the LODF contract holds.

## Behavior summary

```julia
PTDF(sys)                                   # dense, exact (DC_PTDF_Matrix)
PTDF(sys; tol = 1e-3)                        # dense, explicitly sparsified
VirtualPTDF(small_sys)                       # exact rows (below the gate)
VirtualPTDF(large_sys)                       # sparse columns (auto, ≥ 2000 buses)
VirtualPTDF(large_sys; tol = AutoTolerance(; data_precision = 1e-3))   # more aggressive
VirtualPTDF(any_sys; tol = 1e-2)             # explicit absolute cutoff, any size
```

## Acceptance tests (`test/test_auto_tolerance.jl`)

 1. **Construction / discovery.** `AutoTolerance()` defaults; `_sigfigs`,
    `discover_data_precision` recover a known precision and floor on full-precision
    data; `_relative_alpha` clamps to `[1e-6, 1e-2]` and scales with `safety`; all
    `@inferred Float64`.
 2. **Cutoff dispatch.** `apply_cutoff` for absolute vs relative; relative is
    scale-invariant; an `eps()` cutoff is a no-op.
 3. **Bus-count gate.** `_resolve_virtual_cutoff` returns `AbsoluteCutoff(eps())`
    below `AUTO_TOLERANCE_BUS_LIMIT` and `RelativeCutoff` at/above it (synthetic
    grounded path-Laplacian ABA); a `Float64` is an `AbsoluteCutoff` either way.
 4. **Dense no-op.** Dense `PTDF`/`LODF` under `AutoTolerance` equal the `eps()`
    build and stay `DC_PTDF_Matrix`; `Float64` still sparsifies dense.
 5. **Small virtual no-op.** A below-gate `VirtualPTDF` default returns exact rows;
    a `Float64` tol still sparsifies, faithfully (survivors match the dense row).
 6. **Large virtual (manual / `_dev.jl`).** On the real ~83k-bus MISO system the
    default `AutoTolerance()` drops ~45% of each column (more with explicit
    `data_precision`/`safety`); survivors are bit-exact.
 7. **Backward compatibility.** Every existing `Float64` `tol` call is unchanged.

```
```
