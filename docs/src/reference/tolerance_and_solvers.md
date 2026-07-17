# Tolerance & Solver Settings

This page is the canonical reference for two orthogonal build-time settings shared
by the sensitivity-matrix constructors: the **sparsification tolerance** (`tol`)
that controls how aggressively small entries are dropped, and the **linear solver**
(`linear_solver`) that factorizes the `ABA` matrix. For task recipes see
[How to Set the Sparsification Tolerance](@ref) and
[How to Choose a Linear Solver](@ref). Every exported symbol below is documented in
full, with signatures, on the [public API page](public.md).

## Sparsification tolerance

The `tol` keyword on [`PTDF`](@ref) / [`LODF`](@ref) / [`VirtualPTDF`](@ref) /
[`VirtualLODF`](@ref) / [`VirtualMODF`](@ref) has type
`Union{Float64, AutoTolerance}` and defaults to `DEFAULT_AUTO_TOLERANCE` (an
[`AutoTolerance`](@ref) with default settings) — **not** a plain float.

  - A **`Float64`** applies a fixed *absolute* cutoff (`|entry| < tol`), honored at
    any system size. This is the backward-compatible, exact-tolerance path.
  - An **[`AutoTolerance`](@ref)** applies a *relative per-row* cutoff, active only
    on large virtual matrices.

### The AutoTolerance cutoff rule

An [`AutoTolerance`](@ref) drops an entry when

```text
|entry| < α · max|row|,     α = clamp(safety · δ, 1e-6, 1e-2)
```

where `δ` is the relative precision of the branch data (`data_precision`, or
`:auto` to discover it from the branch susceptances via
[`discover_data_precision`](@ref)) and `safety` is an aggressiveness multiplier.
Because the cutoff is relative to each row's own peak, columns of large,
ill-conditioned systems stay sparse regardless of the conditioning of `ABA`. The
1-norm condition estimate of `ABA` is **logged as a diagnostic only** — it never
multiplies the cutoff.

!!! note "Bus-count gate"
    
    An [`AutoTolerance`](@ref) sparsifies the on-demand (virtual) matrices only at
    or above `AUTO_TOLERANCE_BUS_LIMIT` (2000 buses); below that it is a no-op and
    rows are returned exactly, so small systems and the test cases stay exact. The
    relative bounds on `α` are `MIN_RELATIVE_TOLERANCE = 1e-6` and
    `MAX_RELATIVE_TOLERANCE = 1e-2`.

!!! note "Dense path"
    
    On the dense [`PTDF`](@ref) / [`LODF`](@ref) constructors an
    [`AutoTolerance`](@ref) is a no-op (resolves to an `eps()` cutoff), preserving
    the `Matrix{Float64}` element type; a `Float64` `tol` is still honored there as
    an absolute cutoff. The relative per-row drop is reserved for the virtual
    matrices.

### Cutoff types

Both `tol` forms resolve once, at construction, to a concrete internal (not
exported) cutoff stored on the matrix and applied per row:

  - `AbsoluteCutoff` — drops `|entry| < value`; the `Float64` path, at any size.
  - `RelativeCutoff` — drops `|entry| < fraction · max|row|`; the
    [`AutoTolerance`](@ref) path, resolved only at or above
    `AUTO_TOLERANCE_BUS_LIMIT` (otherwise a near-zero `AbsoluteCutoff`, i.e. exact
    rows).

## Linear solver settings

The `linear_solver::String` keyword selects the backend that factorizes `ABA`.
Each name resolves to a singleton solver type for compile-time dispatch:

| String                                            | Backend                                               | Availability          |
|:------------------------------------------------- |:----------------------------------------------------- |:--------------------- |
| `"KLU"`                                           | `KLUWrapper` submodule (libklu via `SuiteSparse_jll`) | Always present        |
| `"AppleAccelerateLU"` (alias `"AppleAccelerate"`) | `AccelerateWrapper` submodule (libSparse LU)          | macOS ≥ 15.5 only     |
| `"MKLPardiso"`                                    | `MKLPardisoExt` extension                             | Requires `Pardiso.jl` |
| `"Dense"`                                         | Dense LAPACK factorization                            | Always present        |

The default is `"AppleAccelerateLU"` on macOS 15.5+ and `"KLU"` everywhere else.
Requesting an unavailable backend raises an informative error at construction.

!!! important "Backend classification"
    
    KLU (`src/KLUWrapper/`) and Apple Accelerate (`src/AccelerateWrapper/`) are
    **always-present submodules** compiled into the package (Apple Accelerate is
    runtime-gated to macOS 15.5+). **MKL Pardiso is a weak-dependency extension**
    (`ext/MKLPardisoExt.jl`) that loads only when `Pardiso.jl` is present. Do not
    describe KLU or Apple Accelerate as "extensions."

!!! note "Singularity"
    
    Apple Accelerate's LU silently factorizes singular matrices (producing garbage)
    where KLU raises. Prefer `"KLU"` when the `ABA` matrix may be singular (for
    example, a full outage that isolates a bus, such as a 3-winding transformer star
    bus).

### Backend preference helpers

The package can persist a preferred sparse backend across sessions via
`Preferences.jl`. These helpers live in `src/linalg_settings.jl` and are not
exported (call them as `PowerNetworkMatrices.<name>`):
`set_linalg_backend_preference`, `get_linalg_backend_preference`,
`set_linalg_backend_check` / `get_linalg_backend_check`, and `check_linalg_backend`
(reports the active BLAS/LAPACK library and whether the requested backend is
loaded).

## See also

  - [How to Set the Sparsification Tolerance](@ref) — choosing `tol` in practice.
  - [How to Choose a Linear Solver](@ref) — picking `linear_solver` per platform.
  - [Computational considerations](../explanation/computational_considerations.md) — the reasoning behind these defaults.
  - [Internals](internals.md) — docstrings for the `KLUWrapper` and `AccelerateWrapper` submodules.
  - [Public API](public.md) — full docstrings for [`AutoTolerance`](@ref) and [`discover_data_precision`](@ref).
