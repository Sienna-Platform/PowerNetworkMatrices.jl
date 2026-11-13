function _check_solve_args(cache::KLULinSolveCache, B::StridedVecOrMat)
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    n = _dim(cache)
    size(B, 1) == Int(n) || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(Int(n))",
    ))
    stride(B, 1) == 1 || throw(ArgumentError(
        "B must have unit stride in the first dimension.",
    ))
    return n
end

"""
    solve!(cache, B) -> B

Solve `A · X = B` in place. `B::StridedVecOrMat{Tv}` must have first-dimension
size equal to `cache.n` and unit stride in the first dimension. Multiple
columns of `B` are handled in a single libklu call.
"""
function solve!(
    cache::KLULinSolveCache{Tv, Ti},
    B::StridedVecOrMat{Tv},
) where {Tv, Ti}
    return _with_owner(cache) do
        n = _check_solve_args(cache, B)
        nrhs = size(B, 2)
        nrhs == 0 && return B
        cache.lean_active && return _lean_solve_columns!(cache, B)
        ok = _solve_call(
            Tv, Ti, cache.symbolic, cache.numeric, n, nrhs, pointer(B), cache.common,
        )
        ok == 0 && klu_throw(cache.common[], "klu_solve")
        return B
    end
end

_lean_solve_columns!(cache::KLULinSolveCache, B::StridedVector) =
    lean_solve!(B, cache.lean_plan, cache.lean_vals)

function _lean_solve_columns!(cache::KLULinSolveCache, B::StridedMatrix)
    for j in axes(B, 2)
        lean_solve!(view(B, :, j), cache.lean_plan, cache.lean_vals)
    end
    return B
end

"""
    tsolve!(cache, B; conjugate=false) -> B

In-place solve `Aᵀ · X = B` (or `Aᴴ · X = B` when `conjugate=true` on the
complex path). Same shape requirements as `solve!`. The `conjugate` keyword
is ignored on the real path.
"""
function tsolve!(
    cache::KLULinSolveCache{Tv, Ti},
    B::StridedVecOrMat{Tv};
    conjugate::Bool = false,
) where {Tv, Ti}
    return _with_owner(cache) do
        n = _check_solve_args(cache, B)
        _require_klu_numeric(cache, "tsolve!")
        nrhs = size(B, 2)
        nrhs == 0 && return B
        ok = _tsolve_call(
            Tv, Ti, cache.symbolic, cache.numeric, n, nrhs, pointer(B), cache.common;
            conjugate = conjugate,
        )
        ok == 0 && klu_throw(cache.common[], "klu_tsolve")
        return B
    end
end

"""
    \\(cache::KLULinSolveCache, B) -> X

Allocating solve, mirroring `LinearAlgebra.Factorization`'s API.
"""
function Base.:\(
    cache::KLULinSolveCache{Tv, Ti},
    B::StridedVecOrMat{Tv},
) where {Tv, Ti}
    return solve!(cache, copy(B))
end
