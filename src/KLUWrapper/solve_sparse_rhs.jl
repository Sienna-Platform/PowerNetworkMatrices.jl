const SPARSE_RHS_DEFAULT_BLOCK = 64

"""
    solve_sparse!(cache, B; out=Matrix{Tv}(undef, cache.n, size(B,2)),
                              block=$(SPARSE_RHS_DEFAULT_BLOCK)) -> out

Solve `A · X = B` for a `SparseMatrixCSC` right-hand side. Empty columns of
`B` are not solved — `out`'s corresponding columns are filled with zeros.
Non-empty columns within each chunk of `block` consecutive RHS columns are
packed into a dense scratch and solved in a single libklu call. `out` may be
any `AbstractMatrix{Tv}` with shape `(cache.n, size(B,2))`, including a view
into a larger matrix.

The `block` chunk size bounds the working set so that processing an
`n × nrhs` sparse RHS requires only `O(n · block)` extra memory regardless of
`nrhs`. The cache reuses its packing buffer across calls; warm calls
allocate nothing in the solver.

Not thread-safe (mutates per-cache scratch). For parallel queries use
`KLULinSolvePool`, where each worker owns an independent cache.
"""
function solve_sparse!(
    cache::KLULinSolveCache{Tv},
    B::SparseMatrixCSC{Tb, Int};
    out::AbstractMatrix{Tv} = Matrix{Tv}(undef, _dim(cache), size(B, 2)),
    block::Int = SPARSE_RHS_DEFAULT_BLOCK,
) where {Tv <: Union{Float64, ComplexF64}, Tb <: Number}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
    block >= 1 || throw(ArgumentError("block must be >= 1; got $(block)"))
    n = _dim(cache)
    size(B, 1) == n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(n)",
    ))
    size(out, 1) == n && size(out, 2) == size(B, 2) || throw(DimensionMismatch(
        "out has size $(size(out)); expected $((n, size(B, 2))).",
    ))

    nb = size(B, 2)
    nb == 0 && return out
    fill!(out, zero(Tv))

    Bcolptr = getcolptr(B)
    Browval = rowvals(B)
    Bnzval = nonzeros(B)

    _ensure_scratch!(cache, block)
    scratch = cache.scratch
    col_map = cache.col_map

    j_start = 1
    @inbounds while j_start <= nb
        j_end = min(j_start + block - 1, nb)

        npack = 0
        for j in j_start:j_end
            Bcolptr[j + 1] > Bcolptr[j] || continue
            npack += 1
            col_map[npack] = j
            fill!(view(scratch, :, npack), zero(Tv))
            for p in Bcolptr[j]:(Bcolptr[j + 1] - 1)
                scratch[Browval[p], npack] = Bnzval[p]
            end
        end

        if npack > 0
            ok = _solve_call(
                Tv, cache.symbolic, cache.numeric, n, Int64(npack),
                pointer(scratch), cache.common,
            )
            ok == 0 && klu_throw(cache.common[], "klu_solve (sparse RHS)")

            for k in 1:npack
                copyto!(view(out, :, col_map[k]), view(scratch, :, k))
            end
        end

        j_start = j_end + 1
    end
    return out
end

"""Allocating wrapper around `solve_sparse!`."""
function solve_sparse(cache::KLULinSolveCache{Tv},
    B::SparseMatrixCSC{<:Number, Int};
    block::Int = SPARSE_RHS_DEFAULT_BLOCK,
) where {Tv}
    return solve_sparse!(cache, B;
        out = Matrix{Tv}(undef, _dim(cache), size(B, 2)),
        block = block,
    )
end
