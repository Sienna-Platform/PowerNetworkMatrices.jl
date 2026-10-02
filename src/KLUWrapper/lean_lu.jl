# Static-pivot sparse LU refactorization on a pivot order frozen from one KLU factor.
#
# `klu_refactor` re-runs the left-looking Gilbert-Peierls column algorithm on a fixed
# pivot order but pays for KLU's generality on every call: packed storage interleaving
# indices and values, per-call row scaling and permutation of A, and the BTF block loop.
# The lean kernel runs the same algorithm on flat Int32 CSC storage with a precomputed
# A -> LU scatter. KLU still does everything that needs pivoting: the analysis, the one
# factor that defines the plan, and every fallback when a lean refactor is rejected.
# The plan holds no C handles and is never written after it is built, so one plan can be
# shared by any number of tasks, each owning its own `LeanLUValues`.

const LEAN_REJECT_RATIO = 1e-3

"""
Frozen pivot order and L/U pattern of a single-block KLU factorization of `A`, with
`LU = A[p, q]`. Column `k` occupies slots `cp[k]:cp[k+1]-1`: U rows ascending, the
diagonal at `dpos[k]`, then the L rows (unit diagonal not stored). All indices 1-based.
"""
struct LeanLUPlan
    n::Int32
    # min|pivot| / max|pivot| of the unscaled factorization of the build matrix, in the
    # units `lean_refactor!` returns.
    rcond0::Float64
    p::Vector{Int32}
    q::Vector{Int32}
    cp::Vector{Int32}
    dpos::Vector{Int32}
    row::Vector{Int32}
    # Pivot row of each stored entry of A, in A's storage order.
    a_row::Vector{Int32}
    # L(:, j) slot range of every U(j, k) dependency, in the order the kernel visits them.
    dep_lb::Vector{Int32}
    dep_le::Vector{Int32}
    a_colptr::Vector{Int32}
    a_rowval::Vector{Int32}
end

LeanLUPlan() = LeanLUPlan(
    Int32(0), 0.0, Int32[], Int32[], Int32[1], Int32[], Int32[],
    Int32[], Int32[], Int32[], Int32[1], Int32[],
)

"""
Per-task numeric state for a `LeanLUPlan`: the LU values and two length-`n` scratch
vectors. The work vector is all-zero between calls.
"""
struct LeanLUValues
    lu::Vector{Float64}
    work::Vector{Float64}
    y::Vector{Float64}
end

LeanLUValues(plan::LeanLUPlan) = LeanLUValues(
    zeros(length(plan.row)), zeros(Int(plan.n)), zeros(Int(plan.n)),
)

_lean_nnz(plan::LeanLUPlan) = length(plan.row)

function _int32_or_error(x::Integer, what::AbstractString)
    x <= typemax(Int32) || throw(OverflowError("lean LU: $what = $x exceeds Int32."))
    return Int32(x)
end

"""
    build_lean_plan(A::SparseMatrixCSC{Float64}) -> LeanLUPlan

Factor `A` once with KLU (BTF off, default AMD ordering and partial pivoting), freeze the
resulting pivot order and L/U pattern, and free the KLU handles. Errors if KLU fails, the
factorization is not a single block, or the reference pivots of `A` are zero or
non-finite.
"""
function build_lean_plan(A::SparseMatrixCSC{Float64, <:Integer})
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square; got $(size(A))"))
    n32 = _int32_or_error(n, "n")
    _int32_or_error(SparseArrays.nnz(A), "nnz(A)")
    a_colptr = Vector{Int32}(getcolptr(A))
    a_rowval = Vector{Int32}(rowvals(A))
    ap = a_colptr .- Int32(1)
    ai = a_rowval .- Int32(1)
    ax = Vector{Float64}(nonzeros(A))

    common = Ref(KluCommon())
    klu_defaults!(common)
    common[].btf = Cint(0)
    sym = GC.@preserve ap ai klu_analyze(n32, pointer(ap), pointer(ai), common)
    sym == C_NULL && klu_throw(common[], "klu_analyze (lean plan)")
    num = NumericPtr32(C_NULL)
    try
        num = GC.@preserve ap ai ax klu_factor(
            pointer(ap), pointer(ai), pointer(ax), sym, common,
        )
        num == C_NULL && klu_throw(common[], "klu_factor (lean plan)")
        shead = unsafe_load(Ptr{KluSymbolicHead{Cint}}(sym))
        nhead = unsafe_load(Ptr{KluNumericHead{Cint}}(num))
        (shead.n == n32 && nhead.n == n32) ||
            error("lean LU: klu_symbolic/klu_numeric layout mismatch with klu.h.")
        (shead.nblocks == 1 && iszero(shead.nzoff)) || error(
            "lean LU: expected one block with btf = 0, got nblocks = $(shead.nblocks).",
        )
        lp = Vector{Cint}(undef, n + 1)
        li = Vector{Cint}(undef, nhead.lnz)
        lx = Vector{Float64}(undef, nhead.lnz)
        up = Vector{Cint}(undef, n + 1)
        ui = Vector{Cint}(undef, nhead.unz)
        ux = Vector{Float64}(undef, nhead.unz)
        p = Vector{Cint}(undef, n)
        q = Vector{Cint}(undef, n)
        ok = GC.@preserve lp li lx up ui ux p q klu_extract(
            num, sym,
            pointer(lp), pointer(li), pointer(lx),
            pointer(up), pointer(ui), pointer(ux),
            Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), Ptr{Cdouble}(C_NULL),
            pointer(p), pointer(q), Ptr{Cdouble}(C_NULL), Ptr{Cint}(C_NULL),
            common,
        )
        ok == 1 || klu_throw(common[], "klu_extract (lean plan)")
        return _lean_plan_from_factors(n32, lp, li, up, ui, p, q, a_colptr, a_rowval, ax)
    finally
        if num != C_NULL
            klu_free_numeric!(Ref(num), common)
        end
        klu_free_symbolic!(Ref(sym), common)
    end
end

# lp/li/up/ui/p/q are klu_extract's 0-based outputs.
function _lean_plan_from_factors(
    n::Int32, lp, li, up, ui, p, q, a_colptr, a_rowval, ax,
)
    cp = Vector{Int32}(undef, n + 1)
    cp[1] = 1
    for k in 1:n
        nu = up[k + 1] - up[k] - 1
        nl = lp[k + 1] - lp[k] - 1
        cp[k + 1] = cp[k] + nu + 1 + nl
    end
    row = Vector{Int32}(undef, cp[n + 1] - 1)
    dpos = Vector{Int32}(undef, n)
    for k in 1:n
        s = cp[k]
        for t in (up[k] + 1):up[k + 1]
            r = ui[t] + 1
            if r != k
                row[s] = r
                s += 1
            end
        end
        sort!(view(row, cp[k]:(s - 1)))
        dpos[k] = s
        row[s] = k
        s += 1
        for t in (lp[k] + 1):lp[k + 1]
            r = li[t] + 1
            if r != k
                row[s] = r
                s += 1
            end
        end
        s == cp[k + 1] || error("lean LU: diagonal missing from L or U column $k.")
    end

    pinv = Vector{Int32}(undef, n)
    for k in 1:n
        pinv[p[k] + 1] = k
    end
    a_row = Vector{Int32}(undef, length(a_rowval))
    for e in eachindex(a_rowval)
        a_row[e] = pinv[a_rowval[e]]
    end
    inpat = falses(n)
    for k in 1:n
        for s in cp[k]:(cp[k + 1] - 1)
            inpat[row[s]] = true
        end
        c = q[k] + 1
        for e in a_colptr[c]:(a_colptr[c + 1] - 1)
            inpat[a_row[e]] || error("lean LU: entry of A outside the L/U pattern.")
        end
        for s in cp[k]:(cp[k + 1] - 1)
            inpat[row[s]] = false
        end
    end

    ndep = 0
    for k in 1:n
        ndep += dpos[k] - cp[k]
    end
    dep_lb = Vector{Int32}(undef, ndep)
    dep_le = Vector{Int32}(undef, ndep)
    d = 1
    for k in 1:n, s in cp[k]:(dpos[k] - 1)
        j = row[s]
        dep_lb[d] = dpos[j] + 1
        dep_le[d] = cp[j + 1] - 1
        d += 1
    end

    pp = Vector{Int32}(p) .+ Int32(1)
    qq = Vector{Int32}(q) .+ Int32(1)
    draft = LeanLUPlan(
        n, 0.0, pp, qq, cp, dpos, row, a_row, dep_lb, dep_le, a_colptr, a_rowval,
    )
    rcond0 = lean_refactor!(LeanLUValues(draft), draft, ax)
    iszero(rcond0) && error("lean LU: zero or non-finite pivot in the build matrix.")
    return LeanLUPlan(
        n, rcond0, pp, qq, cp, dpos, row, a_row, dep_lb, dep_le, a_colptr, a_rowval,
    )
end

"""
    lean_refactor!(vals::LeanLUValues, plan::LeanLUPlan, nzval) -> Float64

Numeric LU of the matrix with `plan`'s pattern and values `nzval` (in the matrix's CSC
storage order) on the plan's frozen pivot order, into `vals.lu`. Returns
min|pivot| / max|pivot|, or `0.0` when any pivot is zero or non-finite; in that case
`vals.lu` must not be used. Allocation-free.
"""
function lean_refactor!(
    vals::LeanLUValues,
    plan::LeanLUPlan,
    nzval::AbstractVector{Float64},
)
    length(nzval) == length(plan.a_row) || throw(
        DimensionMismatch(
            "nzval has $(length(nzval)) entries, plan $(length(plan.a_row)).",
        ),
    )
    length(vals.lu) == _lean_nnz(plan) ||
        throw(DimensionMismatch("LeanLUValues was not built for this plan."))
    lu = vals.lu
    x = vals.work
    row = plan.row
    cp = plan.cp
    dpos = plan.dpos
    a_row = plan.a_row
    a_colptr = plan.a_colptr
    q = plan.q
    dep_lb = plan.dep_lb
    dep_le = plan.dep_le
    pmin = Inf
    pmax = 0.0
    bad = false
    d = 1
    @inbounds for k in 1:Int(plan.n)
        c = Int(q[k])
        for e in Int(a_colptr[c]):(Int(a_colptr[c + 1]) - 1)
            x[a_row[e]] += nzval[e]
        end
        dp = Int(dpos[k])
        for s in Int(cp[k]):(dp - 1)
            u = x[row[s]]
            lo = Int(dep_lb[d])
            hi = Int(dep_le[d])
            d += 1
            iszero(u) || _scatter_update!(x, row, lu, u, lo, hi)
        end
        for s in Int(cp[k]):(dp - 1)
            r = row[s]
            lu[s] = x[r]
            x[r] = 0.0
        end
        piv = x[k]
        x[k] = 0.0
        lu[dp] = piv
        a = abs(piv)
        bad |= !(a > 0.0 && a < Inf)
        pmin = ifelse(a < pmin, a, pmin)
        pmax = ifelse(a > pmax, a, pmax)
        inv = 1.0 / piv
        for s in (dp + 1):(Int(cp[k + 1]) - 1)
            r = row[s]
            lu[s] = x[r] * inv
            x[r] = 0.0
        end
    end
    if bad
        return 0.0
    end
    return pmin / pmax
end

# x[row[t]] -= u * lu[t] for t in lo:hi. The rows are distinct, so the unrolled loads
# can be issued ahead of the stores; unrolling measured ~10% on a 20k-row Jacobian.
@inline function _scatter_update!(x, row, lu, u, lo::Int, hi::Int)
    t = lo
    @inbounds while t + 3 <= hi
        r0 = row[t]
        r1 = row[t + 1]
        r2 = row[t + 2]
        r3 = row[t + 3]
        l0 = lu[t]
        l1 = lu[t + 1]
        l2 = lu[t + 2]
        l3 = lu[t + 3]
        x[r0] = muladd(-u, l0, x[r0])
        x[r1] = muladd(-u, l1, x[r1])
        x[r2] = muladd(-u, l2, x[r2])
        x[r3] = muladd(-u, l3, x[r3])
        t += 4
    end
    @inbounds while t <= hi
        r = row[t]
        x[r] = muladd(-u, lu[t], x[r])
        t += 1
    end
    return
end

"""
    lean_solve!(b, plan::LeanLUPlan, vals::LeanLUValues) -> b

Solve `A x = b` in place with the factors of the last accepted `lean_refactor!`.
Allocation-free.
"""
function lean_solve!(b::AbstractVector{Float64}, plan::LeanLUPlan, vals::LeanLUValues)
    n = Int(plan.n)
    length(b) == n || throw(DimensionMismatch("length(b) = $(length(b)), n = $n"))
    lu = vals.lu
    y = vals.y
    row = plan.row
    cp = plan.cp
    dpos = plan.dpos
    p = plan.p
    q = plan.q
    @inbounds begin
        for k in 1:n
            y[k] = b[p[k]]
        end
        for k in 1:n
            yk = y[k]
            iszero(yk) && continue
            for s in (dpos[k] + 1):(cp[k + 1] - 1)
                y[row[s]] -= lu[s] * yk
            end
        end
        for k in n:-1:1
            zk = y[k] / lu[dpos[k]]
            y[k] = zk
            iszero(zk) && continue
            for s in cp[k]:(dpos[k] - 1)
                y[row[s]] -= lu[s] * zk
            end
        end
        for k in 1:n
            b[q[k]] = y[k]
        end
    end
    return b
end

# Placeholders for caches without a lean plan; never written (has_lean_plan is false).
const _NO_LEAN_PLAN = LeanLUPlan()
const _NO_LEAN_VALUES = LeanLUValues(_NO_LEAN_PLAN)
