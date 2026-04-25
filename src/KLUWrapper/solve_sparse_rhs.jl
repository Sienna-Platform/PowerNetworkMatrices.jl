# Sparse-RHS solve paths.
#
# KLU has no sparse-RHS C entry point. The previous PNM code paths densified
# entire RHS matrices (see `lodf_calculations.jl` originally allocating a
# `zeros(buscount, buscount)` working array, and `ward_reduction.jl` calling
# `Matrix{ComplexF64}(y_eb)`) before handing them to KLU. That burns O(n²)
# bytes on networks with thousands of buses where each RHS column has ≤ 2
# nonzeros. The two routines here scatter sparse columns into a small dense
# scratch and use KLU's native multi-RHS solve, keeping the working set bounded.
#
# Both routines mutate internal scratch buffers and are not thread-safe on
# the same cache instance.

"""
    solve_sparse!(cache, B; out=zeros(...), block=64, skip_empty=false) -> out

Solve `A · X = B` where `B::SparseMatrixCSC` is the right-hand side. Columns
of `B` are scattered into a dense scratch in chunks of `block`, KLU is invoked
once per chunk, and the result is copied into `out`.

Arguments
- `cache::KLULinSolveCache{Tv}` — must be factored.
- `B::SparseMatrixCSC{Tv,Int}` — must have `size(B,1) == cache.n`.

Keyword arguments
- `out::Matrix{Tv}` — destination; allocated if not provided. Must have
  `size(out) == (cache.n, size(B,2))`.
- `block::Int = 64` — number of RHS columns to process at once. Larger blocks
  amortize per-call overhead; smaller blocks shrink the dense scratch.
- `skip_empty::Bool = false` — when true, columns of `B` with no nonzeros are
  skipped; the corresponding columns of `out` are written as zeros without a
  KLU solve. Useful when most RHS columns are structurally empty (e.g. the
  Ward reduction's `y_eb` sub-block).

Not thread-safe.
"""
function solve_sparse!(
    cache::KLULinSolveCache{Tv},
    B::SparseMatrixCSC{Tv, Int};
    out::AbstractMatrix{Tv} = Matrix{Tv}(undef, cache.n, size(B, 2)),
    block::Int = 64,
    skip_empty::Bool = false,
) where {Tv <: Union{Float64, ComplexF64}}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    size(B, 1) == cache.n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(cache.n)",
    ))
    size(out, 1) == cache.n && size(out, 2) == size(B, 2) || throw(DimensionMismatch(
        "out has size $(size(out)); expected $((cache.n, size(B, 2))).",
    ))
    block > 0 || throw(ArgumentError("block must be positive; got $block"))

    nb = size(B, 2)
    nb == 0 && return out

    Bcolptr = getcolptr(B)
    Browval = rowvals(B)
    Bnzval = nonzeros(B)

    width = min(block, nb)
    scratch = Matrix{Tv}(undef, cache.n, width)

    c0 = 1
    while c0 <= nb
        c1 = min(c0 + width - 1, nb)
        w = c1 - c0 + 1

        if skip_empty
            # Find a contiguous run of non-empty columns to keep the scratch tight.
            # If the very first column is empty, emit a zero column and advance.
            if Bcolptr[c0 + 1] == Bcolptr[c0]
                @inbounds for i in 1:cache.n
                    out[i, c0] = zero(Tv)
                end
                c0 += 1
                continue
            end
        end

        @inbounds for k in 0:w-1
            j = c0 + k
            @simd for i in 1:cache.n
                scratch[i, k+1] = zero(Tv)
            end
            for p in Bcolptr[j]:Bcolptr[j+1]-1
                scratch[Browval[p], k+1] = Bnzval[p]
            end
        end

        # KLU solve on the sub-block. Pass the leading dimension via a
        # contiguous view; scratch is column-major so columns 1:w are stride-1.
        nrhs = Int64(w)
        ok = _solve_call(
            Tv, cache.symbolic, cache.numeric, cache.n, nrhs,
            pointer(scratch), cache.common,
        )
        ok == 0 && klu_throw(cache.common[], "klu_solve (sparse RHS)")

        @inbounds for k in 0:w-1
            j = c0 + k
            @simd for i in 1:cache.n
                out[i, j] = scratch[i, k+1]
            end
        end

        c0 = c1 + 1
    end
    return out
end

"""
    solve_sparse(cache, B; ...) -> Matrix

Allocating wrapper around `solve_sparse!`.
"""
function solve_sparse(cache::KLULinSolveCache{Tv},
    B::SparseMatrixCSC{Tv, Int}; kwargs...) where {Tv}
    out = Matrix{Tv}(undef, cache.n, size(B, 2))
    return solve_sparse!(cache, B; out = out, kwargs...)
end
