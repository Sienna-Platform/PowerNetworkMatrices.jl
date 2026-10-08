"""
    solve!(cache, B) -> B

Solve `A · X = B` in place, dispatching on the shape of `B`:

  - `B::StridedMatrix{Cdouble}`: multiple right-hand sides handled in a single
    libSparse call; `size(B, 1)` must equal `cache.n`.
  - `B::StridedVector{Cdouble}`: single right-hand side; `length(B)` must
    equal `cache.n`.

Both overloads require `B` to have unit stride in the first dimension and
the cache to be factored (`is_factored(cache) == true`).
"""
function solve!(cache::AAFactorCache{T}, B::StridedMatrix{T}) where {T}
    is_factored(cache) || error("AAFactorCache: not factored yet.")
    n = cache.n
    size(B, 1) == n ||
        throw(DimensionMismatch("size(B, 1) = $(size(B, 1)), cache n = $(n)"))
    stride(B, 1) == 1 ||
        throw(ArgumentError("B must have unit stride in the first dimension."))
    size(B, 2) == 0 && return B
    # Workspace-aware overload — caller-supplied scratch avoids a per-call
    # malloc/free inside libSparse.
    ws = _ensure_solve_workspace!(cache, size(B, 2))
    GC.@preserve cache _sparse_solve_matrix_ws!(cache.numeric, _dense_matrix(B), ws)
    return B
end

function solve!(cache::AAFactorCache{T}, b::StridedVector{T}) where {T}
    is_factored(cache) || error("AAFactorCache: not factored yet.")
    n = cache.n
    length(b) == n || throw(DimensionMismatch("length(b) = $(length(b)), cache n = $(n)"))
    stride(b, 1) == 1 || throw(ArgumentError("b must have unit stride."))
    ws = _ensure_solve_workspace!(cache, 1)
    GC.@preserve cache _sparse_solve_vector_ws!(cache.numeric, _dense_vector(b), ws)
    return b
end

"""
    \\(cache::AAFactorCache, B) -> X

Allocating solve, mirroring `LinearAlgebra.Factorization`'s API.
"""
Base.:\(cache::AAFactorCache{T}, B::StridedVecOrMat{T}) where {T} = solve!(cache, copy(B))
