# Tolerance & Solver Settings

This page is the reference for two orthogonal build-time settings shared by the
sensitivity-matrix constructors: the **sparsification tolerance** (`tol`) that
controls how aggressively small entries are dropped, and the **linear solver**
(`linear_solver`) that factorizes the `ABA` matrix. For task recipes see the
[set-tolerance how-to guide](../how_to_guides/generated_set_tolerance.md) and the
[choose-a-linear-solver how-to guide](../how_to_guides/generated_choose_linear_solver.md).
Every exported symbol below is documented in full on the
[public API page](public.md).

## Sparsification tolerance

The `tol` keyword on `PTDF` / `LODF` / `VirtualPTDF` / `VirtualLODF` /
`VirtualMODF` has type `Union{Float64, AutoTolerance}` and defaults to
`DEFAULT_AUTO_TOLERANCE` (an [`AutoTolerance`](@ref) with default settings). It is
**not** a plain float default.

  - A **`Float64`** applies a fixed *absolute* cutoff: an entry is dropped when
    `|entry| < tol`. This is the backward-compatible, exact-tolerance path and is
    honored at any system size.
  - An **[`AutoTolerance`](@ref)** applies a *relative per-row* cutoff (see below),
    active only on large systems.

### `AutoTolerance`

```julia
AutoTolerance(; data_precision = :auto, safety = 1.0, quantile = 0.5)
```

Requests automatic, condition-aware sparsification. An entry is dropped when

```text
|entry| < α · max|row|,     α = clamp(safety · δ, 1e-6, 1e-2)
```

where `δ` is the relative precision of the branch data. Because the cutoff is
relative to each row's own peak, columns of large, ill-conditioned systems stay
sparse regardless of the conditioning of `ABA`.

```julia
struct AutoTolerance{D <: Union{Float64, Symbol}}
    data_precision::D
    safety::Float64
    quantile::Float64
end
```

Keyword arguments:

  - `data_precision`: relative precision `δ` of the branch parameters. `:auto`
    (default) discovers it from the branch susceptances via
    [`discover_data_precision`](@ref); a `Float64` sets it explicitly (e.g. `1e-3`
    for reactances good to 0.1%).
  - `safety`: aggressiveness multiplier on `δ`. `> 1` sparsifies more, `< 1` less.
  - `quantile`: used only when `data_precision = :auto`; which quantile of the
    per-branch significant-figure counts to adopt.

The 1-norm condition estimate of `ABA` is computed and **logged as a diagnostic
only** — it never multiplies the cutoff.

The module-wide default is:

```julia
const DEFAULT_AUTO_TOLERANCE = AutoTolerance()
```

!!! note "Bus-count gate"
    
    An `AutoTolerance` sparsifies the on-demand (virtual) matrices only at or above
    `AUTO_TOLERANCE_BUS_LIMIT` (2000 buses). Below that limit it is a no-op and
    rows are returned exactly, so small systems and the test cases stay exact. The
    relative bounds on `α` are `MIN_RELATIVE_TOLERANCE = 1e-6` (a floor that keeps
    full-precision data from disabling sparsification) and
    `MAX_RELATIVE_TOLERANCE = 1e-2` (a ceiling that never lets a row's own peak be
    dropped).

!!! note "Dense path"
    
    On the dense `PTDF` / `LODF` constructors an `AutoTolerance` is a no-op (it
    resolves to an `eps()` cutoff), preserving the `Matrix{Float64}` element type;
    a `Float64` `tol` is still honored there as an absolute cutoff. The relative
    per-row drop is reserved for the on-demand virtual matrices.

### `discover_data_precision`

```julia
discover_data_precision(susceptances; q = 0.5, maxsig = 10, rtol = 1e-9) -> Float64
```

Estimate relative data precision from branch susceptances `b_k`. It recovers the
reactances `x_k = 1/b_k` (the reciprocal exposes the original precision the
susceptance hides), counts the significant figures of each, and returns
`0.5·10^(-(s-1))` at the `q`-quantile of those counts, clamped to `[eps, 1e-2]`.
`maxsig` is coupled to `rtol`; full-precision data (e.g. computed equivalent
branches) hits the `maxsig` cap and yields the tightest precision `5e-10`.

### Cutoff types

`AutoTolerance` and a `Float64` `tol` are both resolved once, at construction, to
a concrete **cutoff** value stored on the matrix and applied per row. These types
are internal (not exported):

  - `AbsoluteCutoff(value::Float64)` — drops entries with `|entry| < value`. The
    `Float64` / backward-compatible path.
  - `RelativeCutoff(fraction::Float64)` — drops entries with
    `|entry| < fraction · max|row|`. The `AutoTolerance` path, so per-column
    sparsity is independent of matrix scale and conditioning.
  - `SparsificationCutoff = Union{AbsoluteCutoff, RelativeCutoff}` — the union used
    throughout the row-cache code.

A `Float64` resolves to an `AbsoluteCutoff` at any size. An `AutoTolerance`
resolves to a `RelativeCutoff` only at or above `AUTO_TOLERANCE_BUS_LIMIT`;
otherwise it resolves to a near-zero `AbsoluteCutoff` (exact rows).

## Linear solver settings

The `linear_solver::String` keyword selects the backend that factorizes `ABA`.
Internally each name resolves to a singleton solver type (subtypes of the internal
abstract type `LinearSolverType`) for compile-time dispatch:

| String                                            | Solver type               | Backend                                               | Availability          |
|:------------------------------------------------- |:------------------------- |:----------------------------------------------------- |:--------------------- |
| `"KLU"`                                           | `KLUSolver`               | `KLUWrapper` submodule (libklu via `SuiteSparse_jll`) | Always present        |
| `"AppleAccelerateLU"` (alias `"AppleAccelerate"`) | `AppleAccelerateLUSolver` | `AccelerateWrapper` submodule (libSparse LU)          | macOS ≥ 15.5 only     |
| `"MKLPardiso"`                                    | `MKLPardisoSolver`        | `MKLPardisoExt` extension                             | Requires `Pardiso.jl` |
| `"Dense"`                                         | `DenseSolver`             | Dense LAPACK factorization                            | Always present        |

```julia
const SUPPORTED_LINEAR_SOLVERS = ("KLU", "MKLPardiso", "AppleAccelerateLU", "Dense")
```

!!! important "Backend classification"
    
    KLU (`src/KLUWrapper/`) and Apple Accelerate (`src/AccelerateWrapper/`) are
    **always-present submodules** compiled into the package; Apple Accelerate is
    runtime-gated to Apple hardware running macOS 15.5 or newer (a
    `@static if Sys.isapple()` guard, with stub fallbacks that throw on non-Apple
    builds). **MKL Pardiso is a weak-dependency extension**
    (`ext/MKLPardisoExt.jl`) that loads only when `Pardiso.jl` is present. Do not
    describe KLU or Apple Accelerate as "extensions."

### Default solver

The default value of the `linear_solver` keyword is chosen per platform:

  - `"AppleAccelerateLU"` on macOS 15.5+ (Apple's built-in libSparse LU).
  - `"KLU"` everywhere else — non-Apple platforms, and macOS older than 15.5.

Requesting `"AppleAccelerateLU"` on an unsupported platform, or `"MKLPardiso"`
without the extension loaded, raises an informative error at construction.

!!! note "Singularity"
    
    Apple Accelerate's LU silently factorizes singular matrices (producing garbage)
    where KLU raises. Prefer `"KLU"` when the `ABA` matrix may be singular (for
    example, full outages that isolate a bus, such as a 3-winding transformer star
    bus).

### Backend preference helpers

The package can persist a preferred sparse backend across sessions via
`Preferences.jl`. These helpers live in `src/linalg_settings.jl` and are not
exported (call them as `PowerNetworkMatrices.<name>`):

  - `set_linalg_backend_preference(linalglib)` — persist a preference. Accepts
    `"MKLPardiso"`, `"AppleAccelerateLU"` / `"AppleAccelerate"`, or `nothing`
    (clear). Also accepts a `Symbol`, or no argument (clears the preference). A
    session restart may be needed for the change to take effect.
  - `get_linalg_backend_preference()` — read the stored preference (or `nothing`).
  - `set_linalg_backend_check(check::Bool)` / `get_linalg_backend_check()` —
    control whether `check_linalg_backend` runs at package load.
  - `check_linalg_backend()` — report the active BLAS/LAPACK library and whether
    the requested sparse backend is loaded, with guidance when it is not.

## See also

  - [Set the sparsification tolerance](../how_to_guides/generated_set_tolerance.md) — choosing `tol` in practice.
  - [Choose a linear solver](../how_to_guides/generated_choose_linear_solver.md) — picking `linear_solver` per platform.
  - [Computational considerations](../explanation/computational_considerations.md) — the reasoning behind these defaults.
  - [Internals](internals.md) — docstrings for the `KLUWrapper` and `AccelerateWrapper` submodules.
  - [Public API](public.md) — full docstrings for [`AutoTolerance`](@ref) and [`discover_data_precision`](@ref).
