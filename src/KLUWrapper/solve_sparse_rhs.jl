const SPARSE_RHS_DEFAULT_BLOCK = 64

"""
    solve_sparse!(cache, B, out; block=$(SPARSE_RHS_DEFAULT_BLOCK)) -> out

Solve `A · X = B` for a `SparseMatrixCSC` right-hand side, writing the result
into `out`. Empty columns of `B` are not solved — `out`'s corresponding
columns are filled with zeros. Non-empty columns within each chunk of `block`
consecutive RHS columns are packed into a dense scratch and solved in a
single libklu call. `out` may be any `AbstractMatrix{Tv}` with shape
`(cache.n, size(B,2))`, including a view into a larger matrix. For an
allocating variant, see `solve_sparse`.

The `block` chunk size bounds the working set so that processing an
`n × nrhs` sparse RHS requires only `O(n · block)` extra memory regardless of
`nrhs`. The cache reuses its packing buffer across calls; warm calls
allocate nothing in the solver.

Not thread-safe (mutates per-cache scratch). Callers should serialize
access through a per-cache lock if invoked from multiple threads.
"""
function solve_sparse!(
    cache::KLULinSolveCache{Tv, Ti},
    B::SparseMatrixCSC{Tb, <:Integer},
    out::AbstractMatrix{Tv};
    block::Int = SPARSE_RHS_DEFAULT_BLOCK,
) where {Tv, Ti, Tb <: Number}
    return _with_owner(cache) do
        is_factored(cache) || error("KLULinSolveCache: not factored yet.")
        _require_klu_numeric(cache, "solve_sparse!")
        block >= 1 || throw(ArgumentError("block must be >= 1; got $(block)"))
        n = _dim(cache)
        size(B, 1) == Int(n) || throw(DimensionMismatch(
            "size(B, 1) = $(size(B, 1)), cache n = $(Int(n))",
        ))
        size(out, 1) == Int(n) && size(out, 2) == size(B, 2) ||
            throw(
                DimensionMismatch(
                    "out has size $(size(out)); expected $((Int(n), size(B, 2))).",
                ),
            )

        nb = size(B, 2)
        nb == 0 && return out
        fill!(out, zero(Tv))

        Browval = rowvals(B)
        Bnzval = nonzeros(B)

        _ensure_scratch!(cache, block)
        scratch = cache.scratch
        col_map = cache.col_map

        j_start = 1
        @inbounds while j_start <= nb
            j_end = min(j_start + block - 1, nb)

            # Expand non-empty columns of `B[:, j_start:j_end]` into dense columns
            # of `scratch`, recording the original column index of each packed
            # column in `col_map`. Structurally empty columns are skipped (their
            # solve output is already zero from the `fill!(out, ...)` above).
            npack = 0
            for j in j_start:j_end
                rng = nzrange(B, j)
                isempty(rng) && continue
                npack += 1
                col_map[npack] = j
                fill!(view(scratch, :, npack), zero(Tv))
                for p in rng
                    scratch[Browval[p], npack] = Bnzval[p]
                end
            end

            if npack > 0
                # Retry-on-KLU_INVALID is inlined here (rather than factored into a
                # helper) to keep the inner-loop closure from capturing the per-chunk
                # `npack`, which would force Julia to box it.
                ok = _solve_call(
                    Tv, Ti, cache.symbolic, cache.numeric, n, npack,
                    pointer(scratch), cache.common,
                )
                if ok == 0 && cache.common[].status == KLU_INVALID
                    @warn "klu_solve (sparse RHS) returned KLU_INVALID; freeing numeric handle and re-factoring" cache_id =
                        objectid(cache)
                    _recover_factorization!(cache)
                    ok = _solve_call(
                        Tv, Ti, cache.symbolic, cache.numeric, n, npack,
                        pointer(scratch), cache.common,
                    )
                end
                ok == 0 && klu_throw(cache.common[], "klu_solve (sparse RHS)")

                for k in 1:npack
                    copyto!(view(out, :, col_map[k]), view(scratch, :, k))
                end
            end

            j_start = j_end + 1
        end
        return out
    end
end

"""Allocating wrapper around `solve_sparse!`."""
function solve_sparse(
    cache::KLULinSolveCache{Tv, Ti},
    B::SparseMatrixCSC{<:Number, <:Integer};
    block::Int = SPARSE_RHS_DEFAULT_BLOCK,
) where {Tv, Ti}
    return solve_sparse!(
        cache, B, Matrix{Tv}(undef, Int(_dim(cache)), size(B, 2));
        block = block,
    )
end
