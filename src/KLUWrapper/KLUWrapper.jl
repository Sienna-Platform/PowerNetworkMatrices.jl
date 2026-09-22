"""
    KLUWrapper

A small, allocation-aware wrapper over `libklu` (provided by `SuiteSparse_jll`)
designed for the access patterns of `PowerNetworkMatrices`:

- Cache the symbolic and numeric factorizations of an SPD/asymmetric sparse
  matrix and reuse them across many solves.
- Refactor (numeric only, or full) without re-allocating.
- Solve dense and **sparse** right-hand sides without materializing N×N
  intermediates when the RHS is structurally sparse.

This module is intentionally lighter than `KLU.jl`: it owns no Julia-side
copies of the matrix values, exposes the symbolic/numeric split directly, and
binds only the `SuiteSparse_long` (`klu_l_*`, `klu_zl_*`) entry points used by
the package.
"""
module KLUWrapper

import LinearAlgebra
import SparseArrays
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros, nzrange

"""
    KLU_POOL_DEBUG :: Bool

Compile-time gate for the KLU wrapper's runtime diagnostics — currently
just the precondition snapshot in `solve!` that fires when `klu_l_solve`
returns FALSE. When `false`, the snapshot is folded out by `@static if`
and contributes zero runtime cost. Flip to `true` when reproducing the
libklu cross-cache concurrency failure documented on `_LIBKLU_LOCK`.

This is a `const` rather than a `Preferences.@load_preference` flag so
that toggling it forces a precompile rebuild and the production binary
never carries the diagnostic code.
"""
const KLU_POOL_DEBUG = false

"""
    _LIBKLU_LOCK :: ReentrantLock

Process-wide lock that serializes every libklu ccall. `libklu` corrupts
internal state under concurrent access even when the caller hands out
distinct `Numeric`/`Symbolic`/`Common` triples per thread (i.e. the
access pattern KLU's user guide implies is supported). The corruption
manifests two ways: an intermittent `KLU_INVALID` return with all input
pointers still valid both pre- and post-call, and a `SIGSEGV` inside
`klu_l_solve` (`klu_solve.c:118` in v7.8.3, the row-permutation read in
the `nrhs == 1` chunk). The pre-call snapshot dump in `solve!` made
both modes reproducible on macOS and confirmed the deterministic
Windows-MinGW failure was the same bug. This lock is the only
mechanism we have evidence for that prevents both modes.
"""
const _LIBKLU_LOCK = ReentrantLock()

"""
    @klu_lock expr

Evaluate `expr` while holding `_LIBKLU_LOCK`. Wrap every libklu ccall
so that no two libklu entries can run concurrently in the process.
"""
macro klu_lock(expr)
    return :(@lock _LIBKLU_LOCK $(esc(expr)))
end

export KLULinSolveCache,
    klu_factorize,
    symbolic_factor!,
    symbolic_refactor!,
    numeric_refactor!,
    full_factor!,
    full_refactor!,
    solve!,
    tsolve!,
    solve_sparse!,
    solve_sparse,
    sort_factors!,
    condest!,
    rcond!,
    n_valid,
    is_factored,
    get_reuse_symbolic

include("klu_jll_bindings.jl")
include("klu_cache.jl")
include("solve_dense.jl")
include("solve_sparse_rhs.jl")

end # module
