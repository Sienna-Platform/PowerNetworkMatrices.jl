const SPARSE_RHS_DEFAULT_BLOCK = 64

"""
    solve_sparse!(cache, B, out; block=$(SPARSE_RHS_DEFAULT_BLOCK)) -> out

Solve `A · X = B` for a `SparseMatrixCSC` right-hand side, writing the
result into `out`. Empty columns of `B` are not solved — `out`'s
corresponding columns are zeroed. Non-empty columns within each chunk of
`block` consecutive RHS columns are packed into a dense scratch and solved
in a single libSparse call.

The `block` chunk size bounds the working set so that processing an
`n × nrhs` sparse RHS requires only `O(n · block)` extra memory regardless
of `nrhs`. The cache reuses its packing buffer across calls; warm calls
allocate nothing in the solver.

Not thread-safe (mutates per-cache scratch).
"""
function solve_sparse!(
    cache::AAFactorCache{Float64},
    B::SparseMatrixCSC{<:Number, Int},
    out::AbstractMatrix{Cdouble};
    block::Int = SPARSE_RHS_DEFAULT_BLOCK,
)
    is_factored(cache) || error("AAFactorCache: not factored yet.")
    block >= 1 || throw(ArgumentError("block must be >= 1; got $(block)"))
    n = cache.n
    size(B, 1) == n || throw(DimensionMismatch(
        "size(B, 1) = $(size(B, 1)), cache n = $(n)",
    ))
    size(out, 1) == n && size(out, 2) == size(B, 2) || throw(DimensionMismatch(
        "out has size $(size(out)); expected $((n, size(B, 2))).",
    ))

    nb = size(B, 2)
    nb == 0 && return out
    fill!(out, zero(Cdouble))

    Browval = rowvals(B)
    Bnzval = nonzeros(B)

    _ensure_scratch!(cache, block)
    # Size the libSparse solve workspace to the *maximum* chunk size once.
    # Sizing it per chunk would `resize!` on every short tail block, churning
    # allocations in the hot PTDF loop. At 10k nodes a 64-RHS block needs
    # ~15 MiB of scratch; without this, libSparse mallocs+frees it per call.
    ws = _ensure_solve_workspace!(cache, block)
    scratch = cache.scratch
    col_map = cache.col_map

    j_start = 1
    @inbounds while j_start <= nb
        j_end = min(j_start + block - 1, nb)

        npack = 0
        for j in j_start:j_end
            rng = nzrange(B, j)
            isempty(rng) && continue
            npack += 1
            col_map[npack] = j
            fill!(view(scratch, :, npack), zero(Cdouble))
            for p in rng
                scratch[Browval[p], npack] = Bnzval[p]
            end
        end

        if npack > 0
            # Build a DenseMatrix view over scratch[:, 1:npack]. Column stride
            # is `size(scratch, 1) = n` because scratch is column-major and
            # full-height.
            dm = DenseMatrix_t(
                Cint(n),
                Cint(npack),
                Cint(size(scratch, 1)),
                ATT_ORDINARY,
                pointer(scratch),
            )
            GC.@preserve cache _sparse_solve_matrix_ws!(cache.numeric, dm, ws)

            for k in 1:npack
                copyto!(view(out, :, col_map[k]), view(scratch, :, k))
            end
        end

        j_start = j_end + 1
    end
    return out
end

"""Allocating wrapper around `solve_sparse!`."""
function solve_sparse(
    cache::AAFactorCache{Float64},
    B::SparseMatrixCSC{<:Number, Int};
    block::Int = SPARSE_RHS_DEFAULT_BLOCK,
)
    return solve_sparse!(
        cache,
        B,
        Matrix{Cdouble}(undef, cache.n, size(B, 2));
        block = block,
    )
end
