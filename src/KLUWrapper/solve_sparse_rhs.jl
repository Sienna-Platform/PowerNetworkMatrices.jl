"""
    solve_sparse!(cache, B; out=Matrix{Tv}(undef, cache.n, size(B,2))) -> out

Solve `A · X = B` for a `SparseMatrixCSC` right-hand side. Empty columns of
`B` are not solved — `out`'s corresponding columns are filled with zeros.
Non-empty columns are packed into a dense scratch and solved in a single
libklu call. `out` may be any `AbstractMatrix{Tv}` with shape `(cache.n,
size(B,2))`, including a view into a larger matrix.

Not thread-safe (mutates an internal scratch). For parallel queries use
`KLULinSolvePool` (see `pool.jl`).
"""
function solve_sparse!(
    cache::KLULinSolveCache{Tv},
    B::SparseMatrixCSC{Tv, Int};
    out::AbstractMatrix{Tv} = Matrix{Tv}(undef, _dim(cache), size(B, 2)),
) where {Tv <: Union{Float64, ComplexF64}}
    is_factored(cache) || error("KLULinSolveCache: not factored yet.")
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

    npack = 0
    for j in 1:nb
        npack += (Bcolptr[j + 1] > Bcolptr[j])
    end
    npack == 0 && return out

    scratch = zeros(Tv, n, npack)
    col_map = Vector{Int}(undef, npack)
    k = 0
    @inbounds for j in 1:nb
        Bcolptr[j + 1] > Bcolptr[j] || continue
        k += 1
        col_map[k] = j
        for p in Bcolptr[j]:Bcolptr[j + 1] - 1
            scratch[Browval[p], k] = Bnzval[p]
        end
    end

    ok = _solve_call(
        Tv, cache.symbolic, cache.numeric, n, Int64(npack),
        pointer(scratch), cache.common,
    )
    ok == 0 && klu_throw(cache.common[], "klu_solve (sparse RHS)")

    @inbounds for k in 1:npack
        copyto!(view(out, :, col_map[k]), view(scratch, :, k))
    end
    return out
end

"""Allocating wrapper around `solve_sparse!`."""
function solve_sparse(cache::KLULinSolveCache{Tv},
    B::SparseMatrixCSC{Tv, Int}) where {Tv}
    return solve_sparse!(cache, B; out = Matrix{Tv}(undef, _dim(cache), size(B, 2)))
end
