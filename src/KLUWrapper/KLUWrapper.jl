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
import ..LinearSolverCache
import SparseArrays
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros, nzrange

"""
    _LIBKLU_LOCK :: ReentrantLock

Process-wide lock that serializes every libklu ccall except the frees, on
Windows only.

It was added for two failures under concurrent use of distinct
`Numeric`/`Symbolic`/`Common` triples: an intermittent `KLU_INVALID` return
with all input pointers valid, and a `SIGSEGV` inside `klu_l_solve`
(`klu_solve.c:118` in v7.8.3, the row-permutation read in the `nrhs == 1`
chunk). Those are attributed, with medium-high confidence, to Julia-side
use-after-free rather than to libklu: `deepcopy` aliasing the raw handles
(now refused by `deepcopy_internal`) and finalizers freeing handles at exit
under unawaited tasks (now skipped via `_PROCESS_EXITING`).

libklu, libamd, libbtf and libcolamd keep no writable globals, and libklu
links no BLAS; the one shared table is SuiteSparse_config's allocator hooks,
pinned to `jl_malloc` before any cache exists (by `__init__` on Julia 1.12+,
by SparseArrays at load on earlier versions). The evidence for lock-free use
is `test/test_klu_threaded.jl` "Distinct caches in parallel". One cache used by
two tasks at once is still unsafe; the per-cache `owner` flag raises on it.
Windows stays locked until a MinGW stress run exists.
"""
const _LIBKLU_LOCK = ReentrantLock()

"""
    @klu_lock expr

Evaluate `expr` while holding `_LIBKLU_LOCK` on Windows; elsewhere a
pass-through. Wraps every libklu ccall except the frees.
"""
macro klu_lock(expr)
    @static if Sys.iswindows()
        return :(@lock _LIBKLU_LOCK $(esc(expr)))
    else
        return esc(expr)
    end
end

# Set at exit, before Julia runs the remaining finalizers, so a finalizer never frees a
# handle that an unawaited task is still factoring or solving on; the OS reclaims it.
const _PROCESS_EXITING = Threads.Atomic{Bool}(false)

function _mark_process_exiting()
    _PROCESS_EXITING[] = true
    return nothing
end

function __init__()
    # Julia 1.12+ switches SuiteSparse to jl_malloc lazily, on first CHOLMOD/UMFPACK use.
    # A libc-born libklu block later freed through jl_free corrupts GC accounting and
    # triggers a collection on nearly every allocation. Switch before any cache exists.
    @static if isdefined(SparseArrays.LibSuiteSparse, :init_suitesparse)
        SparseArrays.LibSuiteSparse.init_suitesparse()
    end
    atexit(_mark_process_exiting)
    return nothing
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
include("lean_lu.jl")
include("klu_cache.jl")
include("solve_dense.jl")
include("solve_sparse_rhs.jl")

end # module
