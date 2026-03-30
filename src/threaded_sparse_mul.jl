"""
Threaded sparse matrix-dense vector/matrix multiplication utilities.

These functions provide multi-threaded implementations of sparse matrix multiplication
for cases where Julia's built-in sparse multiplication does not use threading.
The primary benefit is for sparse × dense operations (SparseMatrixCSC × Vector or
SparseMatrixCSC × Matrix), which are common in PTDF/LODF computations.

# Threading Model
Uses `Threads.@threads` for row-level parallelism. For CSC (Compressed Sparse Column)
format, we iterate over columns and accumulate into result rows. Each output element
is written by exactly one thread to avoid data races.

# When to use
- Sparse × Dense vector: `threaded_mul!(y, A, x)` — parallelizes over output rows
- Sparse × Dense matrix: `threaded_mul!(Y, A, X)` — parallelizes over output columns
- Falls back to standard `mul!` when `Threads.nthreads() == 1`
"""

"""
    threaded_mul!(y::Vector{T}, A::SparseArrays.SparseMatrixCSC{T1, Int}, x::Vector{T2}) where {T, T1, T2}

Compute `y = A * x` using multi-threaded row-parallel sparse matrix-vector multiplication.

For CSC format, iterating over columns is cache-friendly for `A`, but we need to
accumulate into `y` which requires care to avoid races. This implementation uses
a column-parallel approach: each thread processes a subset of columns and accumulates
into a thread-local buffer, then results are reduced.

When only a single thread is available, delegates to `LinearAlgebra.mul!`.

# Arguments
- `y::Vector{T}`: Output vector of length `A.m` (number of rows)
- `A::SparseMatrixCSC`: Sparse matrix
- `x::Vector`: Input vector of length `A.n` (number of columns)
"""
function threaded_mul!(
    y::AbstractVector,
    A::SparseArrays.SparseMatrixCSC,
    x::AbstractVector,
)
    m, n = size(A)
    length(y) == m || throw(DimensionMismatch("y has length $(length(y)), expected $m"))
    length(x) == n || throw(DimensionMismatch("x has length $(length(x)), expected $n"))

    nt = Threads.nthreads()
    if nt == 1
        mul!(y, A, x)
        return y
    end

    # Thread-local accumulators to avoid write conflicts on y
    buffers = [zeros(eltype(y), m) for _ in 1:nt]

    # Each thread processes a chunk of columns
    Threads.@threads for col in 1:n
        tid = Threads.threadid()
        buf = buffers[tid]
        xval = x[col]
        @inbounds for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
            buf[A.rowval[ptr]] += A.nzval[ptr] * xval
        end
    end

    # Reduce thread-local buffers into y
    fill!(y, zero(eltype(y)))
    @inbounds for buf in buffers
        for i in 1:m
            y[i] += buf[i]
        end
    end
    return y
end

"""
    threaded_mul!(Y::Matrix{T}, A::SparseArrays.SparseMatrixCSC, X::Matrix) where T

Compute `Y = A * X` using multi-threaded sparse matrix-dense matrix multiplication.

Parallelizes over the columns of the output matrix `Y`. Each column `Y[:, k] = A * X[:, k]`
is independent and can be computed in parallel. This is the most impactful optimization
for PTDF/LODF computations where the bottleneck is `A * PTDF` (sparse incidence × dense PTDF).

When only a single thread is available, delegates to `LinearAlgebra.mul!`.

# Arguments
- `Y::Matrix`: Output matrix of size `(A.m, size(X, 2))`
- `A::SparseMatrixCSC`: Sparse matrix
- `X::Matrix`: Dense input matrix of size `(A.n, k)`
"""
function threaded_mul!(
    Y::AbstractMatrix,
    A::SparseArrays.SparseMatrixCSC,
    X::AbstractMatrix,
)
    m, n = size(A)
    p = size(X, 2)
    size(Y) == (m, p) ||
        throw(DimensionMismatch("Y has size $(size(Y)), expected ($m, $p)"))
    size(X, 1) == n ||
        throw(DimensionMismatch("X has $(size(X,1)) rows, expected $n"))

    nt = Threads.nthreads()
    if nt == 1
        mul!(Y, A, X)
        return Y
    end

    # Zero Y before accumulation
    fill!(Y, zero(eltype(Y)))

    # Parallelize over output columns — each column is independent
    Threads.@threads for k in 1:p
        @inbounds for col in 1:n
            xval = X[col, k]
            for ptr in A.colptr[col]:(A.colptr[col + 1] - 1)
                Y[A.rowval[ptr], k] += A.nzval[ptr] * xval
            end
        end
    end
    return Y
end

"""
    threaded_sparse_mul(A::SparseArrays.SparseMatrixCSC, x::AbstractVector)

Allocating version: compute and return `A * x` using threaded multiplication.
"""
function threaded_sparse_mul(
    A::SparseArrays.SparseMatrixCSC,
    x::AbstractVector,
)
    T = promote_type(eltype(A), eltype(x))
    y = zeros(T, size(A, 1))
    return threaded_mul!(y, A, x)
end

"""
    threaded_sparse_mul(A::SparseArrays.SparseMatrixCSC, X::AbstractMatrix)

Allocating version: compute and return `A * X` using threaded multiplication.
"""
function threaded_sparse_mul(
    A::SparseArrays.SparseMatrixCSC,
    X::AbstractMatrix,
)
    T = promote_type(eltype(A), eltype(X))
    Y = zeros(T, size(A, 1), size(X, 2))
    return threaded_mul!(Y, A, X)
end
