# Static-pivot sparse LU refactorization on a frozen KLU pivot order. Port of
# `lean_lu.hpp` from Fraunhofer IEE's parallel-pandapower-solver (BSD-3-Clause).
#
# KLU picks the pivot order once; refactor and solve then run in plain Julia on flat CSC
# storage, free of libklu's per-call overhead and of `_LIBKLU_LOCK`. The plan is
# read-only, so concurrent tasks share one plan and each owns a `LeanLUWorkspace`.

"""
    LeanLUPlan(cache::KLULinSolveCache{Float64}, A::SparseMatrixCSC{Float64}) -> LeanLUPlan

Freeze the pivot order and L/U pattern of `cache`'s factorization of `A`, for use with
[`lean_refactor!`](@ref) and [`lean_ldiv!`](@ref).

`cache` must be factored, with BTF disabled: set `cache.common[].btf = 0` before
`symbolic_factor!`. Otherwise, if KLU split `A` into coupled blocks, construction throws an
`ArgumentError`.

The plan is immutable. Share one plan across tasks, giving each task its own
[`LeanLUWorkspace`](@ref).
"""
struct LeanLUPlan{Ti <: Union{Int32, Int64}}
    n::Int
    # LU = A[P, Q]: P[k] is the row of A at pivot k, Q[k] the column.
    P::Vector{Ti}
    Q::Vector{Ti}
    # Values of column k live in cp[k]:cp[k+1]-1: U rows ascending, then the diagonal at
    # dpos[k], then L rows (L is unit-diagonal; its diagonal is not stored).
    cp::Vector{Ti}
    dpos::Vector{Ti}
    row::Vector{Ti}
    # A -> LU scatter: column k adds nonzeros(A)[aSrc[e]] into row aRow[e], for e in
    # aK[k]:aK[k+1]-1. Every entry of A is scattered once, so length(aSrc) == nnz(A).
    aK::Vector{Ti}
    aSrc::Vector{Ti}
    aRow::Vector{Ti}
    # Dependencies of column k (U(j,k) != 0, j < k, ascending j -- a valid topological
    # order for the column algorithm), for d in dK[k]:dK[k+1]-1: the U(j,k) slot dU[d]
    # and the value range dLb[d]:dLe[d] of L(:,j).
    dK::Vector{Ti}
    dU::Vector{Ti}
    dLb::Vector{Ti}
    dLe::Vector{Ti}
    # Pivot ratio of A itself on this order; lean_refactor! reports relative to it.
    rcond0::Float64
end

"""
    LeanLUWorkspace(plan::LeanLUPlan)

Factor values and scratch for one task using `plan`. Not safe to share between tasks.
"""
struct LeanLUWorkspace
    LU::Vector{Float64}
    x::Vector{Float64}  # all-zero between calls
    y::Vector{Float64}
end

LeanLUWorkspace(plan::LeanLUPlan) =
    LeanLUWorkspace(zeros(length(plan.row)), zeros(plan.n), zeros(plan.n))

SparseArrays.nnz(plan::LeanLUPlan) = length(plan.row)

_extract_call(num::Ptr{Cvoid}, sym::Ptr{Cvoid}, Lp::Vector{Int32}, args...) = klu_extract(
    reinterpret(NumericPtr32, num), reinterpret(SymbolicPtr32, sym), Lp, args...)
_extract_call(num::Ptr{Cvoid}, sym::Ptr{Cvoid}, Lp::Vector{Int64}, args...) =
    klu_l_extract(reinterpret(NumericPtr, num), reinterpret(SymbolicPtr, sym), Lp, args...)

function LeanLUPlan(
    cache::KLULinSolveCache{Float64, Ti},
    A::SparseMatrixCSC{Float64, Ti},
) where {Ti}
    is_factored(cache) ||
        throw(ArgumentError("LeanLUPlan: cache must be factored (call full_factor!)."))
    n = Int(_dim(cache))
    size(A) == (n, n) || throw(
        DimensionMismatch("LeanLUPlan: cache is $(n)×$(n) but A is $(size(A))."),
    )
    _check_pattern_match(cache, A, "build LeanLUPlan")

    # `lnz`/`unz` are the 3rd and 4th `Ti` fields of the public `klu_numeric` struct in
    # klu.h (n, nblocks, lnz, unz, ...); they size the extract buffers.
    lnz = Int(unsafe_load(Ptr{Ti}(cache.numeric), 3))
    unz = Int(unsafe_load(Ptr{Ti}(cache.numeric), 4))
    Lp = Vector{Ti}(undef, n + 1)
    Li = Vector{Ti}(undef, lnz)
    Up = Vector{Ti}(undef, n + 1)
    Ui = Vector{Ti}(undef, unz)
    # Off-diagonal block entries are a subset of A's entries.
    Fp = Vector{Ti}(undef, n + 1)
    Fi = Vector{Ti}(undef, SparseArrays.nnz(A))
    P = Vector{Ti}(undef, n)
    Q = Vector{Ti}(undef, n)
    ok = _extract_call(cache.numeric, cache.symbolic, Lp, Li, zeros(lnz),
        Up, Ui, zeros(unz), Fp, Fi, zeros(SparseArrays.nnz(A)), P, Q, cache.common)
    ok == 1 || klu_throw(cache.common[], "klu_extract")
    iszero(Fp[end]) || throw(
        ArgumentError(
            "LeanLUPlan: KLU split the matrix into coupled blocks. Set " *
            "`cache.common[].btf = 0` before `symbolic_factor!`.",
        ),
    )
    for v in (Lp, Li, Up, Ui, P, Q)
        v .+= one(Ti)
    end
    Pinv = invperm(P)

    # Flat column layout: U part (ascending), diagonal, L part.
    cp = Vector{Ti}(undef, n + 1)
    dpos = Vector{Ti}(undef, n)
    cp[1] = 1
    for k in 1:n
        cp[k + 1] = cp[k] + (Up[k + 1] - Up[k]) + (Lp[k + 1] - Lp[k]) - 1
    end
    row = Vector{Ti}(undef, cp[n + 1] - 1)
    Lbeg = Vector{Ti}(undef, n)
    Lend = Vector{Ti}(undef, n)
    for k in 1:n
        q = Int(cp[k])
        for p in Up[k]:(Up[k + 1] - 1)
            Ui[p] == k && continue
            row[q] = Ui[p]
            q += 1
        end
        # U rows must be ascending (it is the elimination order); L rows may stay unsorted.
        sort!(view(row, Int(cp[k]):(q - 1)))
        dpos[k] = q
        row[q] = k
        q += 1
        Lbeg[k] = q
        for p in Lp[k]:(Lp[k + 1] - 1)
            Li[p] == k && continue
            row[q] = Li[p]
            q += 1
        end
        Lend[k] = q - 1
    end

    # Scatter map for A and the per-column dependency lists.
    Arows = rowvals(A)
    inpat = falses(n)
    aK, aSrc, aRow = Ti[1], Ti[], Ti[]
    dK, dU, dLb, dLe = Ti[1], Ti[], Ti[], Ti[]
    sizehint!(aSrc, length(Arows))
    sizehint!(aRow, length(Arows))
    for k in 1:n
        cols = Int(cp[k]):(Int(cp[k + 1]) - 1)
        for q in cols
            inpat[row[q]] = true
        end
        for p in nzrange(A, Int(Q[k]))
            r = Pinv[Arows[p]]
            @assert inpat[r]
            push!(aSrc, p)
            push!(aRow, r)
        end
        push!(aK, length(aSrc) + 1)
        for q in Int(cp[k]):(Int(dpos[k]) - 1)
            j = row[q]
            push!(dU, q)
            push!(dLb, Lbeg[j])
            push!(dLe, Lend[j])
        end
        push!(dK, length(dU) + 1)
        for q in cols
            inpat[row[q]] = false
        end
    end

    plan = LeanLUPlan{Ti}(n, P, Q, cp, dpos, row, aK, aSrc, aRow, dK, dU, dLb, dLe, NaN)
    rcond0 = _lean_refactor!(LeanLUWorkspace(plan), plan, nonzeros(A))
    rcond0 > 0 || throw(
        ArgumentError("LeanLUPlan: A has a zero or non-finite pivot on KLU's order."),
    )
    return LeanLUPlan{Ti}(n, P, Q, cp, dpos, row, aK, aSrc, aRow, dK, dU, dLb, dLe, rcond0)
end

function _check_workspace(ws::LeanLUWorkspace, plan::LeanLUPlan)
    (
        length(ws.LU) == length(plan.row) && length(ws.x) == plan.n &&
        length(ws.y) == plan.n
    ) ||
        throw(DimensionMismatch("LeanLUWorkspace was not built for this LeanLUPlan."))
    return nothing
end

"""
    lean_refactor!(ws::LeanLUWorkspace, plan::LeanLUPlan, Ax::AbstractVector{Float64}) -> Float64

Factor the matrix with values `Ax` into `ws`, on `plan`'s pivot order. `Ax` is
`nonzeros` of a matrix with the pattern `plan` was built from.

Returns `min|pivot| / max|pivot|` divided by the same ratio for the matrix `plan` was
built from, so `1.0` for that matrix. Returns `0.0` on a zero or non-finite pivot. When
it falls below a threshold such as `1e-3`, the frozen pivots are unsuitable: factor with
KLU instead.
"""
lean_refactor!(ws::LeanLUWorkspace, plan::LeanLUPlan, Ax::AbstractVector{Float64}) =
    _lean_refactor!(ws, plan, Ax) / plan.rcond0

# Unnormalized min|pivot| / max|pivot|.
function _lean_refactor!(ws::LeanLUWorkspace, plan::LeanLUPlan, Ax::AbstractVector{Float64})
    _check_workspace(ws, plan)
    length(Ax) == length(plan.aSrc) || throw(
        DimensionMismatch(
            "lean_refactor!: Ax has $(length(Ax)) values; expected " *
            "$(length(plan.aSrc)).",
        ),
    )
    LU, x, row = ws.LU, ws.x, plan.row
    aK, aSrc, aRow = plan.aK, plan.aSrc, plan.aRow
    dK, dU, dLb, dLe = plan.dK, plan.dU, plan.dLb, plan.dLe
    cp, dpos = plan.cp, plan.dpos
    pmin, pmax, finite = Inf, 0.0, true
    @inbounds for k in 1:(plan.n)
        for e in aK[k]:(aK[k + 1] - 1)
            x[aRow[e]] += Ax[aSrc[e]]
        end
        for d in dK[k]:(dK[k + 1] - 1)
            u = x[row[dU[d]]]
            iszero(u) && continue
            for s in dLb[d]:dLe[d]
                x[row[s]] -= u * LU[s]
            end
        end
        dp = dpos[k]
        for q in cp[k]:(dp - 1)
            r = row[q]
            LU[q] = x[r]
            x[r] = 0.0
        end
        piv = x[k]
        x[k] = 0.0
        LU[dp] = piv
        a = abs(piv)
        (a > 0.0 && isfinite(a)) || (finite = false)
        pmin = min(pmin, a)
        pmax = max(pmax, a)
        inv = 1.0 / piv
        for q in (dp + 1):(cp[k + 1] - 1)
            r = row[q]
            LU[q] = x[r] * inv
            x[r] = 0.0
        end
    end
    (finite && pmax > 0.0) || return 0.0
    return pmin / pmax
end

"""
    lean_ldiv!(ws::LeanLUWorkspace, plan::LeanLUPlan, b::AbstractVector{Float64}) -> b

Overwrite `b` with `A \\ b`, where `A` is the matrix last factored into `ws` by
[`lean_refactor!`](@ref).
"""
function lean_ldiv!(ws::LeanLUWorkspace, plan::LeanLUPlan, b::AbstractVector{Float64})
    _check_workspace(ws, plan)
    n = plan.n
    length(b) == n ||
        throw(DimensionMismatch("lean_ldiv!: b has length $(length(b)); expected $n."))
    LU, y, row = ws.LU, ws.y, plan.row
    P, Q, cp, dpos = plan.P, plan.Q, plan.cp, plan.dpos
    @inbounds begin
        for k in 1:n
            y[k] = b[P[k]]
        end
        for k in 1:n                       # L y = P b (unit diagonal)
            yk = y[k]
            iszero(yk) && continue
            for q in (dpos[k] + 1):(cp[k + 1] - 1)
                y[row[q]] -= LU[q] * yk
            end
        end
        for k in n:-1:1                    # U z = y
            zk = y[k] / LU[dpos[k]]
            y[k] = zk
            iszero(zk) && continue
            for q in cp[k]:(dpos[k] - 1)
                y[row[q]] -= LU[q] * zk
            end
        end
        for k in 1:n                       # x = Q z
            b[Q[k]] = y[k]
        end
    end
    return b
end

const lean_solve! = lean_ldiv!
