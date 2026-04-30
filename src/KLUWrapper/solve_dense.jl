"""
    solve!(cache, B) -> B

Solve `A · X = B` in place. `B::StridedVecOrMat{Tv}` must have first-dimension
size equal to `cache.n` and unit stride in the first dimension. Multiple
columns of `B` are handled in a single libklu call.
"""
function solve!(cache::KLULinSolveCache{Tv},
    B::StridedVecOrMat{Tv}) where {Tv <: Union{Float64, ComplexF64}}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    n = _dim(cache)
    size(B, 1) == n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(n)",
    ))
    stride(B, 1) == 1 || throw(ArgumentError(
        "B must have unit stride in the first dimension.",
    ))
    nrhs = Int64(size(B, 2))
    nrhs == 0 && return B
    # Snapshot the inputs KLU validates on entry. `klu_l_solve` returns FALSE
    # with `Common->status == KLU_INVALID` when any of {Numeric, Symbolic, B}
    # is NULL, when `ldim < Numeric->n`, or when `nrhs < 0`. Capturing the
    # observed values *before* the ccall lets a Windows-only failure under
    # the parallel `KLULinSolvePool` path identify which precondition KLU
    # rejected. Negligible overhead (a few field loads) on the hot path.
    pre_numeric = cache.numeric
    pre_symbolic = cache.symbolic
    pre_b_ptr = pointer(B)
    ok = _solve_call(
        Tv, cache.symbolic, cache.numeric, n, nrhs, pointer(B), cache.common,
    )
    if ok == 0
        @error "KLU klu_solve precondition snapshot" tid = Threads.threadid() ldim_n = Int(n) nrhs = Int(nrhs) pre_numeric_null = (pre_numeric == C_NULL) pre_symbolic_null = (pre_symbolic == C_NULL) pre_b_null = (pre_b_ptr == C_NULL) post_numeric_null = (cache.numeric == C_NULL) post_symbolic_null = (cache.symbolic == C_NULL) status = Int(cache.common[].status)
        klu_throw(cache.common[], "klu_solve")
    end
    return B
end

"""
    tsolve!(cache, B; conjugate=false) -> B

In-place solve `Aᵀ · X = B` (or `Aᴴ · X = B` when `conjugate=true` on the
complex path). Same shape requirements as `solve!`. The `conjugate` keyword
is ignored on the real path.
"""
function tsolve!(cache::KLULinSolveCache{Tv},
    B::StridedVecOrMat{Tv}; conjugate::Bool = false,
) where {Tv <: Union{Float64, ComplexF64}}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    n = _dim(cache)
    size(B, 1) == n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(n)",
    ))
    stride(B, 1) == 1 || throw(ArgumentError(
        "B must have unit stride in the first dimension.",
    ))
    nrhs = Int64(size(B, 2))
    nrhs == 0 && return B
    ok = _tsolve_call(
        Tv, cache.symbolic, cache.numeric, n, nrhs, pointer(B), cache.common;
        conjugate = conjugate,
    )
    ok == 0 && klu_throw(cache.common[], "klu_tsolve")
    return B
end

"""
    \\(cache::KLULinSolveCache, B) -> X

Allocating solve, mirroring `LinearAlgebra.Factorization`'s API.
"""
function Base.:\(cache::KLULinSolveCache{Tv},
    B::StridedVecOrMat{Tv}) where {Tv <: Union{Float64, ComplexF64}}
    return solve!(cache, copy(B))
end
