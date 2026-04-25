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
    ok = _solve_call(
        Tv, cache.symbolic, cache.numeric, n, nrhs, pointer(B), cache.common,
    )
    ok == 0 && klu_throw(cache.common[], "klu_solve")
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
