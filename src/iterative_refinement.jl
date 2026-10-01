import LinearAlgebra: norm, mul!

"""
    DEFAULT_REFINEMENT_MAX_ITER :: Int

Default iteration cap for `solve_w_refinement!`.
"""
const DEFAULT_REFINEMENT_MAX_ITER = 25

# --- backend dispatch bridge ---------------------------------------------------
#
# The refinement body uses three primitives — in-place `solve!`, `is_factored`,
# and `_dim` — each of which is defined inside `KLUWrapper` and
# `AccelerateWrapper` as a *separate* function. The bridge below lets the
# shared body reach the right backend by multiple dispatch instead of an `isa`
# branch, and keeps the body itself backend-agnostic.

_refine_solve!(K::KLULinSolveCache, r::StridedVecOrMat) = solve!(K, r)
_refine_solve!(K::AAFactorCache, r::StridedVecOrMat) = AccelerateWrapper.solve!(K, r)

_refine_is_factored(K::KLULinSolveCache) = is_factored(K)
_refine_is_factored(K::AAFactorCache) = AccelerateWrapper.is_factored(K)

_refine_dim(K::KLULinSolveCache) = Int(KLUWrapper._dim(K))
_refine_dim(K::AAFactorCache) = AccelerateWrapper._dim(K)

_refine_solve!(K::LeanLUCache, r::StridedVecOrMat) = solve!(K, r)
_refine_is_factored(K::LeanLUCache) = is_factored(K)
_refine_dim(K::LeanLUCache) = size(K, 1)

"""
    solve_w_refinement!(cache, A, X, B; tol=…, max_iters=…) -> X

Solve `A · X = B` in place using `cache` and apply iterative refinement
until `norm(B − A·X, 1) < norm(B, 1) * tol`, or until refinement stops
improving. `X` must be pre-allocated by the caller with the same shape as
`B`; it is overwritten with the refined solution.

This is the non-allocating variant. For a one-shot allocating variant that
returns a fresh `X`, see `solve_w_refinement`.

Supports `cache::KLULinSolveCache` (KLU backend, any `{Tv, Ti}`),
`cache::AAFactorCache` (Apple Accelerate backend, `Cdouble` only) and
`cache::LeanLUCache` (`Cdouble` only). The
cache must already be factored (`is_factored(cache) == true`). The function
does not mutate `cache`'s factor; it only triggers in-place `solve!` calls
on a residual buffer.

`max_iters` caps the refinement loop (default
`DEFAULT_REFINEMENT_MAX_ITER`). The default `tol` is `sqrt(eps(real(Tv)))`,
which is conservative for power-flow Newton-Raphson Jacobians.

Useful when the cached factor is of an ill-conditioned matrix — e.g.,
a Newton-Raphson Jacobian near a saddle point or a network reduction
interface — where a single back-solve carries enough error to slow NR
convergence. Cost per refinement iteration: one sparse matrix-vector
product plus one `solve!` against the cached factor.
"""
function solve_w_refinement!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseArrays.SparseMatrixCSC{Tv, Ti},
    X::StridedVecOrMat{Tv},
    B::StridedVecOrMat{Tv};
    tol::Real = sqrt(eps(real(Tv))),
    max_iters::Int = DEFAULT_REFINEMENT_MAX_ITER,
) where {Tv, Ti}
    return _solve_w_refinement_body!(cache, A, X, B, tol, max_iters)
end

function solve_w_refinement!(
    cache::Union{AAFactorCache, LeanLUCache},
    A::SparseArrays.SparseMatrixCSC{Cdouble, <:Integer},
    X::StridedVecOrMat{Cdouble},
    B::StridedVecOrMat{Cdouble};
    tol::Real = sqrt(eps(Cdouble)),
    max_iters::Int = DEFAULT_REFINEMENT_MAX_ITER,
)
    return _solve_w_refinement_body!(cache, A, X, B, tol, max_iters)
end

function _solve_w_refinement_body!(
    cache,
    A::SparseArrays.SparseMatrixCSC,
    X::StridedVecOrMat,
    B::StridedVecOrMat,
    tol::Real,
    max_iters::Int,
)
    _refine_is_factored(cache) || error(
        "solve_w_refinement!: cache must be factored. " *
        "Call `full_factor!(cache, A)` first.",
    )
    size(X) == size(B) ||
        throw(DimensionMismatch("X is $(size(X)); B is $(size(B))."))
    size(B, 1) == _refine_dim(cache) || throw(
        DimensionMismatch(
            "size(B, 1) = $(size(B, 1)); cache n = $(_refine_dim(cache)).",
        ),
    )

    Tv = eltype(X)
    fill!(X, zero(Tv))
    # `X = 0` ⇒ initial residual `r = B - A·0 = B`. Copy once; subsequent
    # iterations reuse `r` in place via `mul!(r, A, X); @. r = B - r`,
    # which avoids the temporary `A * X` allocation `B - A * X` would create
    # each refinement step.
    r = copy(B)
    bNorm = norm(B, 1)
    iters = 0
    while iters < max_iters && norm(r, 1) >= bNorm * tol
        prev_err = norm(r, 1)
        _refine_solve!(cache, r)
        X .+= r
        mul!(r, A, X)
        @. r = B - r
        iters += 1
        if norm(r, 1) > prev_err
            @debug "Iterative refinement diverging; returning best-so-far." iters
            return X
        end
    end
    @debug "Iterative refinement converged." iters
    return X
end

"""
    solve_w_refinement(cache, A, B; tol=…, max_iters=…) -> X

Allocating wrapper around `solve_w_refinement!`. Allocates `X` matching
`B`'s shape, then refines.
"""
function solve_w_refinement(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseArrays.SparseMatrixCSC{Tv, Ti},
    B::StridedVecOrMat{Tv};
    tol::Real = sqrt(eps(real(Tv))),
    max_iters::Int = DEFAULT_REFINEMENT_MAX_ITER,
) where {Tv, Ti}
    X = similar(B)
    return solve_w_refinement!(
        cache, A, X, B;
        tol = tol, max_iters = max_iters,
    )
end

function solve_w_refinement(
    cache::Union{AAFactorCache, LeanLUCache},
    A::SparseArrays.SparseMatrixCSC{Cdouble, <:Integer},
    B::StridedVecOrMat{Cdouble};
    tol::Real = sqrt(eps(Cdouble)),
    max_iters::Int = DEFAULT_REFINEMENT_MAX_ITER,
)
    X = similar(B)
    return solve_w_refinement!(
        cache, A, X, B;
        tol = tol, max_iters = max_iters,
    )
end
