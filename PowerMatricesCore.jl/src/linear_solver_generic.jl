# One function name per factor operation for every cache backend. KLU and MKLPardiso already
# extend the `KLUWrapper` functions. These methods add `AAFactorCache` to them.
# `AAFactorCache` has no transpose solve, so `tsolve!` is KLU-only.

symbolic_factor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.symbolic_factor!(c, A)
numeric_refactor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.numeric_refactor!(c, A)
full_factor!(c::AAFactorCache, A::SparseArrays.SparseMatrixCSC) =
    AccelerateWrapper.full_factor!(c, A)
solve!(c::AAFactorCache, b::StridedVecOrMat) = AccelerateWrapper.solve!(c, b)
is_factored(c::AAFactorCache) = AccelerateWrapper.is_factored(c)

"""
    make_linear_solver_cache(backend::LinearSolverType, A; snapshot_values = true)

Return an unfactored cache for `A` on `backend`: `KLULinSolveCache` for `KLUSolver`,
`AAFactorCache` for `AppleAccelerateLUSolver`, `PardisoLinSolveCache` for `MKLPardisoSolver`.
Only the KLU cache uses `snapshot_values`. The other backends ignore it.
"""
function make_linear_solver_cache(
    ::KLUSolver,
    A::SparseArrays.SparseMatrixCSC;
    snapshot_values::Bool = true,
)
    return KLULinSolveCache(A; snapshot_values = snapshot_values)
end

function make_linear_solver_cache(
    ::AppleAccelerateLUSolver,
    A::SparseArrays.SparseMatrixCSC;
    snapshot_values::Bool = true,
)
    return AAFactorCache(A)
end

function make_linear_solver_cache(
    ::MKLPardisoSolver,
    A::SparseArrays.SparseMatrixCSC;
    snapshot_values::Bool = true,
)
    return PardisoLinSolveCache(A)
end

"""Relative tolerance of the probe in [`factorization_ok`](@ref)."""
const SINGULAR_PROBE_RTOL = 1e-6

factorization_ok(::KLULinSolveCache, ::SparseArrays.SparseMatrixCSC) = true

"""
    factorization_ok(cache, A) -> Bool

Return `false` if the factorization in `cache` of `A` is numerically singular. KLU throws
`SingularException` in the factorization, so a `KLULinSolveCache` always returns `true`.
AppleAccelerate and MKLPardiso return finite values for a singular matrix. For them the
function solves `A x = A v` for a known `v` and tests `‖x - v‖ <= SINGULAR_PROBE_RTOL * ‖v‖`.
Call it after `full_factor!` or `numeric_refactor!`. `A` must be the matrix of that call.
These functions do not call it.
"""
function factorization_ok(
    cache::LinearSolverCache,
    A::SparseArrays.SparseMatrixCSC{T},
) where {T}
    v = T.(1:size(A, 1))
    x = A * v
    solve!(cache, x)
    all(isfinite, x) || return false
    err = sqrt(sum(i -> abs2(x[i] - v[i]), eachindex(v)))
    return err <= SINGULAR_PROBE_RTOL * LinearAlgebra.norm(v)
end
