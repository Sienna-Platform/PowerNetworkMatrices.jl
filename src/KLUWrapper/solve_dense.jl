"""
    solve!(cache, B) -> B

Solve `A · X = B` in place. `B` may be a `Vector` or `Matrix` with eltype
matching the cache. The trailing dimension is the RHS count, which KLU
handles natively in a single call.

Allocation-free; throws if the cache is not factored or `B` is not stride-1
in its first dimension.
"""
function solve!(cache::KLULinSolveCache{Tv},
    B::StridedVecOrMat{Tv}) where {Tv <: Union{Float64, ComplexF64}}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    size(B, 1) == cache.n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(cache.n)",
    ))
    stride(B, 1) == 1 || throw(ArgumentError(
        "B must have unit stride in the first dimension.",
    ))
    nrhs = Int64(size(B, 2))
    nrhs == 0 && return B
    ok = _solve_call(
        Tv, cache.symbolic, cache.numeric, cache.n, nrhs, pointer(B), cache.common,
    )
    ok == 0 && klu_throw(cache.common[], "klu_solve")
    return B
end

"""
    tsolve!(cache, B) -> B

In-place solve `Aᵀ · X = B`. Same shape requirements as `solve!`.
"""
function tsolve!(cache::KLULinSolveCache{Float64},
    B::StridedVecOrMat{Float64})
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    size(B, 1) == cache.n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(cache.n)",
    ))
    stride(B, 1) == 1 || throw(ArgumentError(
        "B must have unit stride in the first dimension.",
    ))
    nrhs = Int64(size(B, 2))
    nrhs == 0 && return B
    ok = klu_l_tsolve(
        cache.symbolic, cache.numeric, cache.n, nrhs, pointer(B), cache.common,
    )
    ok == 0 && klu_throw(cache.common[], "klu_tsolve")
    return B
end

function tsolve!(cache::KLULinSolveCache{ComplexF64},
    B::StridedVecOrMat{ComplexF64}; conjugate::Bool = false)
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    size(B, 1) == cache.n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(cache.n)",
    ))
    stride(B, 1) == 1 || throw(ArgumentError(
        "B must have unit stride in the first dimension.",
    ))
    nrhs = Int64(size(B, 2))
    nrhs == 0 && return B
    ok = klu_zl_tsolve(
        cache.symbolic, cache.numeric, cache.n, nrhs, pointer(B),
        Cint(conjugate), cache.common,
    )
    ok == 0 && klu_throw(cache.common[], "klu_zl_tsolve")
    return B
end

"""
    solve_w_refinement(cache, A, B; tol=1e-6, max_iter=10) -> X

Iterative refinement: repeatedly solve the residual until `‖r‖₁ < ‖B‖₁ · tol`
or `max_iter` is reached. Bails early if the error grows. Returns a fresh
solution `X`; does not mutate `B`. Provided for parity with PowerFlows.jl's
`solve_w_refinement` so downstream packages can consolidate on this cache.
"""
function solve_w_refinement(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int},
    B::StridedVecOrMat{Tv};
    tol::Float64 = 1e-6,
    max_iter::Int = 10,
) where {Tv <: Union{Float64, ComplexF64}}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    bnorm = LinearAlgebra.norm(B, 1)
    X = zeros(Tv, size(B))
    r = B - A * X
    iters = 0
    while iters < max_iter && LinearAlgebra.norm(r, 1) >= bnorm * tol
        last_err = LinearAlgebra.norm(r, 1)
        solve!(cache, r)
        X .+= r
        r .= B .- A * X
        iters += 1
        if LinearAlgebra.norm(r, 1) > last_err
            @warn "solve_w_refinement: error increased; stopping" iters
            return X
        end
    end
    return X
end
