"""
    AccelerateWrapper

A small, allocation-aware wrapper over Apple's `libSparse.dylib`
(`/System/Library/Frameworks/Accelerate.framework/.../libSparse.dylib`)
designed for the access patterns of `PowerNetworkMatrices`:

- Cache the symbolic and numeric factorizations of a general (unsymmetric)
  sparse matrix (LU with threshold partial pivoting + Inf-norm equilibration
  scaling) and reuse them across many solves.
- Refresh the numeric factor (`numeric_refactor!`) while keeping the symbolic
  analysis, without re-allocating the structural arrays.
- Solve dense and **sparse** right-hand sides in place, with the sparse path
  packing only non-empty RHS columns into a bounded scratch block.
- Compute `A·X` and `A·x` directly via libSparse's `SparseMultiply`.

This module is intentionally lighter than the upstream `AppleAccelerate.jl`
package: it owns no high-level Julia wrappers over libSparse, exposes the
symbolic/numeric split directly, binds only the entry points used by PowerNetworkMatrices,
and is compile-gated to macOS so non-Apple builds never codegen the
`@ccall` sites.
"""
module AccelerateWrapper

import SparseArrays
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros, nzrange
import LinearAlgebra

export AAFactorCache,
    aa_factorize,
    symbolic_factor!,
    numeric_refactor!,
    full_factor!,
    full_refactor!,
    solve!,
    solve_sparse!,
    solve_sparse,
    is_factored,
    aa_spmm!,
    aa_spmv!

@static if Sys.isapple()
    include("libsparse_bindings.jl")
    include("aa_cache.jl")
    include("solve_dense.jl")
    include("solve_sparse_rhs.jl")
    include("spmm.jl")
else
    # Stub layer. Non-Apple builds never bind libSparse symbols, never codegen
    # the `@ccall` sites, and never instantiate `SparseOpaqueFactorization`.
    # The whole submodule reduces to these short bodies on Linux/Windows.
    struct AAFactorCache end

    _unavailable() = error(
        "AccelerateWrapper is macOS-only (Sys.isapple() returned false). " *
        "Use the KLU backend on non-Apple platforms.",
    )

    AAFactorCache(args...; kwargs...) = _unavailable()
    aa_factorize(args...; kwargs...) = _unavailable()
    symbolic_factor!(args...) = _unavailable()
    numeric_refactor!(args...) = _unavailable()
    full_factor!(args...) = _unavailable()
    full_refactor!(args...) = _unavailable()
    solve!(args...) = _unavailable()
    solve_sparse!(args...; kwargs...) = _unavailable()
    solve_sparse(args...; kwargs...) = _unavailable()
    is_factored(::AAFactorCache) = false
    aa_spmm!(args...) = _unavailable()
    aa_spmv!(args...) = _unavailable()
end

end # module
