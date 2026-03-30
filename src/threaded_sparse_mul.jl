"""
Threaded sparse matrix-vector multiplication utilities.

Provides `threaded_mul!` for `y = A * x` and `threaded_tmul!` for `y = A' * x`
using Julia's `Threads.@threads`. These are useful when `Threads.nthreads() > 1`
and the matrix is large enough that threading overhead is amortized.

For small matrices (fewer than `THREADED_MUL_MIN_SIZE` rows/columns), the
functions fall back to a single-threaded implementation to avoid overhead.

## CSC Layout Considerations

`SparseMatrixCSC` stores data column-by-column. This makes the **transpose
multiply** (`A' * x`) naturally parallelisable: each column of `A` contributes
to exactly one element of `y`, so threads write to disjoint output locations.

The **forward multiply** (`A * x`) scatters each column's contribution across
multiple rows of `y`. To avoid data races, we partition columns into per-thread
chunks and use thread-local accumulators that are reduced at the end.
"""

"""
Minimum matrix dimension (rows for `mul!`, columns for `tmul!`) below which
the single-threaded path is always used.
"""
const THREADED_MUL_MIN_SIZE = 1000

"""
    threaded_tmul!(y::Vector, A::SparseMatrixCSC, x::Vector) -> y

Compute `y = A' * x` (transpose multiply) with threading over columns of `A`.

Each thread computes the dot product of one or more columns of `A` with `x`,
writing to disjoint positions in `y`. No synchronisation is needed.

Falls back to single-threaded execution when `Threads.nthreads() == 1` or
when `A.n < THREADED_MUL_MIN_SIZE`.
"""
function threaded_tmul!(
    y::Vector{T},
    A::SparseArrays.SparseMatrixCSC{Tv, Ti},
    x::AbstractVector,
) where {T <: Number, Tv <: Number, Ti <: Integer}
    m, n = size(A)
    @assert length(x) == m
    @assert length(y) == n

    if Threads.nthreads() > 1 && n >= THREADED_MUL_MIN_SIZE
        Threads.@threads for i in 1:n
            tmp = zero(T)
            @inbounds for j in A.colptr[i]:(A.colptr[i + 1] - 1)
                tmp += T(A.nzval[j]) * T(x[A.rowval[j]])
            end
            @inbounds y[i] = tmp
        end
    else
        @inbounds for i in 1:n
            tmp = zero(T)
            for j in A.colptr[i]:(A.colptr[i + 1] - 1)
                tmp += T(A.nzval[j]) * T(x[A.rowval[j]])
            end
            y[i] = tmp
        end
    end
    return y
end

"""
    threaded_mul!(y::Vector, A::SparseMatrixCSC, x::AbstractVector) -> y

Compute `y = A * x` (forward multiply) with threading.

Uses per-thread column chunks with thread-local accumulators to avoid write
races on `y`. The partial results are reduced into `y` after the parallel
section.

Falls back to single-threaded execution when `Threads.nthreads() == 1` or
when `A.m < THREADED_MUL_MIN_SIZE`.
"""
function threaded_mul!(
    y::Vector{T},
    A::SparseArrays.SparseMatrixCSC{Tv, Ti},
    x::AbstractVector,
) where {T <: Number, Tv <: Number, Ti <: Integer}
    m, n = size(A)
    @assert length(x) == n
    @assert length(y) == m

    fill!(y, zero(T))
    nt = Threads.nthreads()

    if nt > 1 && m >= THREADED_MUL_MIN_SIZE
        # Partition columns into nt contiguous chunks
        chunk = cld(n, nt)
        local_ys = [zeros(T, m) for _ in 1:nt]

        Threads.@threads for tid in 1:nt
            col_start = (tid - 1) * chunk + 1
            col_end = min(tid * chunk, n)
            ly = local_ys[tid]
            @inbounds for col in col_start:col_end
                xval = T(x[col])
                for j in A.colptr[col]:(A.colptr[col + 1] - 1)
                    ly[A.rowval[j]] += T(A.nzval[j]) * xval
                end
            end
        end

        # Reduce thread-local results into y
        @inbounds for tid in 1:nt
            ly = local_ys[tid]
            for i in 1:m
                y[i] += ly[i]
            end
        end
    else
        @inbounds for col in 1:n
            xval = T(x[col])
            for j in A.colptr[col]:(A.colptr[col + 1] - 1)
                y[A.rowval[j]] += T(A.nzval[j]) * xval
            end
        end
    end
    return y
end
