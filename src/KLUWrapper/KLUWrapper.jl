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
binds only the SuiteSparse_long (`klu_l_*`, `klu_zl_*`) entry points used by
the package.
"""
module KLUWrapper

import LinearAlgebra
import SparseArrays
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros, nzrange

"""
    _LIBKLU_LOCK :: ReentrantLock

Process-wide lock that serializes every libklu ccall. `libklu` corrupts
internal state under concurrent access even when the caller hands out
distinct `Numeric`/`Symbolic`/`Common` triples per thread (i.e. the
access pattern KLU's user guide implies is supported). The corruption
manifests two ways: an intermittent `KLU_INVALID` return with all input
pointers still valid both pre- and post-call, and a `SIGSEGV` inside
`klu_l_solve` (`klu_solve.c:118` in v7.8.3, the row-permutation read in
the `nrhs == 1` chunk). This lock is the only
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
    condest!,
    is_factored

include("klu_jll_bindings.jl")
include("klu_cache.jl")
include("solve_dense.jl")
include("solve_sparse_rhs.jl")

end # module
