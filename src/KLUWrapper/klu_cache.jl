import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros

"""
A cached KLU linear solver designed for repeated solves against the same
sparse matrix structure. `numeric_refactor!` and `solve!` allocate nothing
once the cache is built.

Type parameters:
- `Tv ∈ {Float64, ComplexF64}` selects the real/complex KLU path
  (`klu_*_factor`/`klu_z*_factor`).
- `Ti ∈ {Int32, Int64}` selects the index-type entry-point family
  (`klu_*` for `int`/Int32, `klu_l_*` for `SuiteSparse_long`/Int64).
  The cache's `colptr`/`rowval`/`col_map` are stored in this type.

`reuse_symbolic` controls whether `symbolic_refactor!` keeps the analysis;
`check_pattern` adds a structural-equality check on refactor calls and is
only consulted when reusing. `snapshot_values` keeps a copy of the factored
values for `condest!(cache)` and the `solve_sparse!` recovery path; without it
those error and `condest!(cache, A)` takes the values from `A`.
"""
mutable struct KLULinSolveCache{
    Tv <: Union{Float64, ComplexF64},
    Ti <: Union{Int32, Int64},
} <: LinearSolverCache
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    # Copy of the matrix values used in the most recent numeric factorization.
    # Lets `_recover_factorization!` rebuild a corrupted numeric handle without
    # the caller having to re-supply `A`. Empty before the first factor call,
    # and always when `snapshot_values` is false.
    nzval::Vector{Tv}
    # `Base.RefValue{KluLCommon}` for Int64 or `Base.RefValue{KluCommon}` for
    # Int32 — we keep the field untyped here because Julia's untyped
    # `Base.RefValue` lookup is fast enough (the dispatch helpers recover the
    # concrete type at each callsite via `_common_type(Ti)`) and avoiding a
    # third type parameter keeps the surface tidy.
    common::Base.RefValue
    # Opaque Symbolic/Numeric pointers. `Ptr{Cvoid}` is safe: each callsite
    # threads through the typed `SymbolicPtr` / `SymbolicPtr32` (resp.
    # Numeric) at the ccall boundary; on the Julia side the values are
    # treated as black boxes by every consumer.
    symbolic::Ptr{Cvoid}
    numeric::Ptr{Cvoid}
    reuse_symbolic::Bool
    check_pattern::Bool
    snapshot_values::Bool
    # Bounded reusable scratch for `solve_sparse!`. Lazy-grown on first call so
    # the wrapper's working set stays O(n*block) instead of O(n*nrhs); see
    # `solve_sparse_rhs.jl`.
    scratch::Matrix{Tv}
    col_map::Vector{Ti}
    # Task currently inside a cache method (0 = none); see `_acquire!`.
    owner::Threads.Atomic{UInt}
    # Optional lean static-pivot path (Float64 only, see `lean_lu.jl`). While
    # `lean_active`, the current factors live in `lean_vals` and the KLU numeric handle
    # is stale, so only `solve!` may run.
    lean_plan::LeanLUPlan
    lean_vals::LeanLUValues
    has_lean_plan::Bool
    lean_active::Bool
    # While set, `numeric_refactor!` skips the lean path (see `pause_lean!`).
    lean_paused::Bool
    # Set by `defer_symbolic!`: `symbolic` is C_NULL until a KLU factorization needs it.
    analyze_pending::Bool
    # Counters: `lean_counts` and `cold_retries` read them.
    lean_attempts::Int
    lean_rejects::Int
    # Accepted lean factorizations whose solve the caller rejected (`repivot!`).
    lean_solve_failures::Int
    # Deferred analyses that a KLU factorization then needed.
    late_analyses::Int
    # Solves the caller reran from scratch after failing on a reused pivot order.
    cold_retries::Int
end

# Returns true when this call took ownership and must release it, false on
# re-entry by the owning task. A different task inside the cache is a bug.
function _acquire!(c::KLULinSolveCache)
    me = UInt(pointer_from_objref(current_task()))
    prev = Threads.atomic_cas!(c.owner, UInt(0), me)
    iszero(prev) && return true
    prev == me && return false
    error(
        "KLULinSolveCache (n=$(size(c, 1))) used by two tasks at once; caches are not shareable.",
    )
end

@inline function _with_owner(f, c::KLULinSolveCache)
    taken = _acquire!(c)
    try
        return f()
    finally
        taken && (c.owner[] = UInt(0))
    end
end

@inline _dim(cache::KLULinSolveCache{Tv, Ti}) where {Tv, Ti} =
    Ti(length(cache.colptr) - 1)

Base.size(cache::KLULinSolveCache) = (n = Int(_dim(cache)); (n, n))
Base.size(cache::KLULinSolveCache, d::Integer) =
    d <= 2 ? Int(_dim(cache)) : 1
Base.eltype(::Type{KLULinSolveCache{Tv, Ti}}) where {Tv, Ti} = Tv

"""
    is_factored(cache::KLULinSolveCache) -> Bool

Return `true` when `cache` is ready for `solve!`: a current lean factorization, or a
symbolic and a numeric KLU factorization (also ready for `tsolve!` / `solve_sparse!`).
Returns `false` after construction (before `full_factor!`) or after the libklu handles
have been finalized.
"""
function is_factored(cache::KLULinSolveCache)
    return cache.lean_active || (cache.symbolic != C_NULL && cache.numeric != C_NULL)
end

# ---------------------------------------------------------------------------
# Type-paired dispatch helpers — (Tv, Ti) → libklu entry point
# ---------------------------------------------------------------------------

# Map Ti to its concrete `klu_common` struct type.
@inline _common_type(::Type{Int32}) = KluCommon
@inline _common_type(::Type{Int64}) = KluLCommon

# `klu_defaults` initializer.
@inline _defaults!(::Type{Int32}, common::Ref) = klu_defaults!(common)
@inline _defaults!(::Type{Int64}, common::Ref) = klu_l_defaults!(common)

# `klu_analyze` returns the symbolic handle. n is widened to `Ti` so the
# C argument width matches.
@inline _analyze_call(::Type{Int32}, n, ap, ai, common) =
    klu_analyze(Cint(n), ap, ai, common)
@inline _analyze_call(::Type{Int64}, n, ap, ai, common) =
    klu_l_analyze(Int64(n), ap, ai, common)

# `klu_free_symbolic` — takes the opaque pointer ref and the common ref.
# `sym_ref` is a `Ref{Ptr{Cvoid}}` on the Julia side; we reinterpret it
# to the typed `SymbolicPtr` / `SymbolicPtr32` at the ccall site so libklu
# sees the right pointer width.
@inline function _free_symbolic!(::Type{Int32}, sym_ref::Ref{Ptr{Cvoid}}, common::Ref)
    typed = Ref(reinterpret(SymbolicPtr32, sym_ref[]))
    klu_free_symbolic!(typed, common)
    sym_ref[] = reinterpret(Ptr{Cvoid}, typed[])
    return nothing
end
@inline function _free_symbolic!(::Type{Int64}, sym_ref::Ref{Ptr{Cvoid}}, common::Ref)
    typed = Ref(reinterpret(SymbolicPtr, sym_ref[]))
    klu_l_free_symbolic!(typed, common)
    sym_ref[] = reinterpret(Ptr{Cvoid}, typed[])
    return nothing
end

# `klu_factor` — returns the numeric handle as a typed pointer; we
# reinterpret to `Ptr{Cvoid}` for storage. Dispatch on both Tv and Ti.
@inline function _factor_call(::Type{Float64}, ::Type{Int32}, ap, ai, ax, sym, common)
    return reinterpret(
        Ptr{Cvoid},
        klu_factor(ap, ai, ax, reinterpret(SymbolicPtr32, sym), common),
    )
end
@inline function _factor_call(::Type{Float64}, ::Type{Int64}, ap, ai, ax, sym, common)
    return reinterpret(
        Ptr{Cvoid},
        klu_l_factor(ap, ai, ax, reinterpret(SymbolicPtr, sym), common),
    )
end
@inline function _factor_call(
    ::Type{ComplexF64},
    ::Type{Int32},
    ap,
    ai,
    ax,
    sym,
    common,
)
    return reinterpret(
        Ptr{Cvoid},
        klu_z_factor(ap, ai, ax, reinterpret(SymbolicPtr32, sym), common),
    )
end
@inline function _factor_call(
    ::Type{ComplexF64},
    ::Type{Int64},
    ap,
    ai,
    ax,
    sym,
    common,
)
    return reinterpret(
        Ptr{Cvoid},
        klu_zl_factor(ap, ai, ax, reinterpret(SymbolicPtr, sym), common),
    )
end

@inline function _refactor_call(
    ::Type{Float64},
    ::Type{Int32},
    ap,
    ai,
    ax,
    sym,
    num,
    common,
)
    return klu_refactor(
        ap, ai, ax,
        reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num),
        common,
    )
end
@inline function _refactor_call(
    ::Type{Float64},
    ::Type{Int64},
    ap,
    ai,
    ax,
    sym,
    num,
    common,
)
    return klu_l_refactor(
        ap, ai, ax,
        reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num),
        common,
    )
end
@inline function _refactor_call(
    ::Type{ComplexF64},
    ::Type{Int32},
    ap,
    ai,
    ax,
    sym,
    num,
    common,
)
    return klu_z_refactor(
        ap, ai, ax,
        reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num),
        common,
    )
end
@inline function _refactor_call(
    ::Type{ComplexF64},
    ::Type{Int64},
    ap,
    ai,
    ax,
    sym,
    num,
    common,
)
    return klu_zl_refactor(
        ap, ai, ax,
        reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num),
        common,
    )
end

@inline function _solve_call(
    ::Type{Float64},
    ::Type{Int32},
    sym,
    num,
    n,
    nrhs,
    b,
    common,
)
    return klu_solve(
        reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num),
        Cint(n), Cint(nrhs), b, common,
    )
end
@inline function _solve_call(
    ::Type{Float64},
    ::Type{Int64},
    sym,
    num,
    n,
    nrhs,
    b,
    common,
)
    return klu_l_solve(
        reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num),
        Int64(n), Int64(nrhs), b, common,
    )
end
@inline function _solve_call(
    ::Type{ComplexF64},
    ::Type{Int32},
    sym,
    num,
    n,
    nrhs,
    b,
    common,
)
    return klu_z_solve(
        reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num),
        Cint(n), Cint(nrhs), b, common,
    )
end
@inline function _solve_call(
    ::Type{ComplexF64},
    ::Type{Int64},
    sym,
    num,
    n,
    nrhs,
    b,
    common,
)
    return klu_zl_solve(
        reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num),
        Int64(n), Int64(nrhs), b, common,
    )
end

@inline function _tsolve_call(
    ::Type{Float64},
    ::Type{Int32},
    sym,
    num,
    n,
    nrhs,
    b,
    common;
    conjugate::Bool = false,
)
    return klu_tsolve(
        reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num),
        Cint(n), Cint(nrhs), b, common,
    )
end
@inline function _tsolve_call(
    ::Type{Float64},
    ::Type{Int64},
    sym,
    num,
    n,
    nrhs,
    b,
    common;
    conjugate::Bool = false,
)
    return klu_l_tsolve(
        reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num),
        Int64(n), Int64(nrhs), b, common,
    )
end
@inline function _tsolve_call(
    ::Type{ComplexF64},
    ::Type{Int32},
    sym,
    num,
    n,
    nrhs,
    b,
    common;
    conjugate::Bool = false,
)
    return klu_z_tsolve(
        reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num),
        Cint(n), Cint(nrhs), b, Cint(conjugate), common,
    )
end
@inline function _tsolve_call(
    ::Type{ComplexF64},
    ::Type{Int64},
    sym,
    num,
    n,
    nrhs,
    b,
    common;
    conjugate::Bool = false,
)
    return klu_zl_tsolve(
        reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num),
        Int64(n), Int64(nrhs), b, Cint(conjugate), common,
    )
end

@inline function _free_numeric!(
    ::Type{Float64},
    ::Type{Int32},
    num_ref::Ref{Ptr{Cvoid}},
    common::Ref,
)
    typed = Ref(reinterpret(NumericPtr32, num_ref[]))
    klu_free_numeric!(typed, common)
    num_ref[] = reinterpret(Ptr{Cvoid}, typed[])
    return nothing
end
@inline function _free_numeric!(
    ::Type{Float64},
    ::Type{Int64},
    num_ref::Ref{Ptr{Cvoid}},
    common::Ref,
)
    typed = Ref(reinterpret(NumericPtr, num_ref[]))
    klu_l_free_numeric!(typed, common)
    num_ref[] = reinterpret(Ptr{Cvoid}, typed[])
    return nothing
end
@inline function _free_numeric!(
    ::Type{ComplexF64},
    ::Type{Int32},
    num_ref::Ref{Ptr{Cvoid}},
    common::Ref,
)
    typed = Ref(reinterpret(NumericPtr32, num_ref[]))
    klu_z_free_numeric!(typed, common)
    num_ref[] = reinterpret(Ptr{Cvoid}, typed[])
    return nothing
end
@inline function _free_numeric!(
    ::Type{ComplexF64},
    ::Type{Int64},
    num_ref::Ref{Ptr{Cvoid}},
    common::Ref,
)
    typed = Ref(reinterpret(NumericPtr, num_ref[]))
    klu_zl_free_numeric!(typed, common)
    num_ref[] = reinterpret(Ptr{Cvoid}, typed[])
    return nothing
end

# Float64 only.
@inline _condest_call(::Type{Int32}, ap, ax, sym, num, common) =
    klu_condest(
        ap,
        ax,
        reinterpret(SymbolicPtr32, sym),
        reinterpret(NumericPtr32, num),
        common,
    )
@inline _condest_call(::Type{Int64}, ap, ax, sym, num, common) =
    klu_l_condest(
        ap,
        ax,
        reinterpret(SymbolicPtr, sym),
        reinterpret(NumericPtr, num),
        common,
    )

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    KLULinSolveCache(A; reuse_symbolic=true, check_pattern=true, snapshot_values=true)

Build a cache for the sparse matrix `A`. The cache's index type is taken
from `A`: `SparseMatrixCSC{Tv, Int32}` ⇒ `KLULinSolveCache{Tv, Int32}`,
`SparseMatrixCSC{Tv, Int64}` ⇒ `KLULinSolveCache{Tv, Int64}`. Allocates
structural arrays and runs the corresponding `klu_defaults`/`klu_l_defaults`
initializer, but does **not** factorize. Call `full_factor!` (or
`symbolic_factor!` followed by `numeric_refactor!`) before `solve!`.

A finalizer frees libklu handles on GC (but not at process exit, where the
OS reclaims them); call `Base.finalize(cache)` to
release them eagerly. Releasing the handles leaves Julia-side state intact,
so the cache can be re-factorized via `symbolic_factor!`/`numeric_refactor!`
or `full_factor!`.

`snapshot_values = false` skips copying `A`'s values on every
`numeric_refactor!`; `condest!(cache)` and the `solve_sparse!` corruption
recovery then error, and `condest!(cache, A)` must be used instead.
"""
function KLULinSolveCache(
    A::SparseMatrixCSC{Tv, Ti};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
    snapshot_values::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}, Ti <: Union{Int32, Int64}}
    n = size(A, 1)
    n == size(A, 2) ||
        throw(DimensionMismatch("matrix must be square; got $(size(A))"))

    common = Ref(_common_type(Ti)())
    _defaults!(Ti, common)

    colptr = Vector{Ti}(undef, length(getcolptr(A)))
    copyto!(colptr, getcolptr(A))
    colptr .-= one(Ti)
    rowval = Vector{Ti}(undef, length(rowvals(A)))
    copyto!(rowval, rowvals(A))
    rowval .-= one(Ti)

    cache = KLULinSolveCache{Tv, Ti}(
        colptr, rowval, Tv[], common,
        Ptr{Cvoid}(C_NULL), Ptr{Cvoid}(C_NULL),
        reuse_symbolic, check_pattern, snapshot_values,
        Matrix{Tv}(undef, 0, 0),
        Ti[],
        Threads.Atomic{UInt}(0),
        _NO_LEAN_PLAN, _NO_LEAN_VALUES, false, false, false, false, 0, 0, 0, 0, 0,
    )
    finalizer(_finalize_klu_handles!, cache)
    return cache
end

"""
    _ensure_scratch!(cache, block) -> Nothing

Ensure `cache.scratch` is at least `n × block` and `cache.col_map` length
`block`. Grows in place; reuses across `solve_sparse!` calls.
"""
@inline function _ensure_scratch!(
    cache::KLULinSolveCache{Tv, Ti},
    block::Int,
) where {Tv, Ti}
    n = Int(_dim(cache))
    s = cache.scratch
    if size(s, 1) != n || size(s, 2) < block
        cache.scratch = Matrix{Tv}(undef, n, block)
    end
    if length(cache.col_map) < block
        resize!(cache.col_map, block)
    end
    return nothing
end

# Free the numeric handle only; the next `numeric_refactor!` then runs a fresh,
# pivoting `klu_factor` instead of a `klu_refactor` on the old pivots.
function _drop_numeric!(cache::KLULinSolveCache{Tv, Ti}) where {Tv, Ti}
    if cache.numeric != C_NULL
        num_ref = Ref(cache.numeric)
        _free_numeric!(Tv, Ti, num_ref, cache.common)
        cache.numeric = num_ref[]
    end
    return nothing
end

"""
Release the libklu numeric and symbolic handles held by `cache`, leaving the
Julia-side fields (`colptr`, `rowval`, `common`, `scratch`, `col_map`) intact
so the cache remains structurally valid and re-factorable. Idempotent: a
second call hits the `C_NULL` guards. Used both by `symbolic_factor!`
mid-life (drop old handles before re-analyzing) and by the GC finalizer
(via `_finalize_klu_handles!`).
"""
function _free_klu_handles!(
    cache::KLULinSolveCache{Tv, Ti},
) where {Tv, Ti}
    cache.lean_active = false
    cache.analyze_pending = false
    _drop_numeric!(cache)
    if cache.symbolic != C_NULL
        sym_ref = Ref(cache.symbolic)
        _free_symbolic!(Ti, sym_ref, cache.common)
        cache.symbolic = sym_ref[]
    end
    return nothing
end

function _finalize_klu_handles!(cache::KLULinSolveCache)
    _PROCESS_EXITING[] && return nothing
    _free_klu_handles!(cache)
    return nothing
end

# Public eager-release alias for `_free_klu_handles!` (the internal helper
# stays unexported per the KLUWrapper convention).
Base.finalize(cache::KLULinSolveCache) = _free_klu_handles!(cache)

# `deepcopy` is unsafe: it would bit-copy the raw libklu `symbolic`/`numeric`
# pointers, aliasing one factorization across two finalizer-owning caches (a
# double-free/use-after-free). Throw so the caller shares the owning matrix by
# reference instead; the backtrace names the offending site.
function Base.deepcopy_internal(::KLULinSolveCache, ::IdDict)
    error(
        "deepcopy of a KLULinSolveCache is unsafe and was attempted: the cache holds " *
        "raw libklu Symbolic/Numeric pointers that deepcopy would alias by value, " *
        "causing a double-free / use-after-free of the factorization (KLU_INVALID or " *
        "SIGSEGV). Share the owning Virtual matrix by reference instead of deepcopying " *
        "it. The backtrace identifies the offending call site.",
    )
end

"""
Drop a corrupted numeric handle and rebuild it from the values cached in
`cache.nzval`. Used by `solve_sparse!` on the `KLU_INVALID` retry path, where
libklu state has been observed to corrupt without the caller having `A` in
scope. Requires that a numeric factor has been built before (so `cache.nzval`
is populated) and the symbolic factor is still valid.
"""
function _recover_factorization!(
    cache::KLULinSolveCache{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        cache.symbolic == C_NULL && error(
            "KLULinSolveCache: cannot recover without a symbolic factor.",
        )
        _require_klu_numeric(cache, "KLULinSolveCache recovery")
        _check_value_snapshot(cache, "KLULinSolveCache recovery")
        _drop_numeric!(cache)
        num = _factor_call(
            Tv, Ti,
            pointer(cache.colptr), pointer(cache.rowval),
            pointer(cache.nzval), cache.symbolic, cache.common,
        )
        num == C_NULL && klu_throw(cache.common[], "klu_factor (recovery)")
        cache.numeric = num
        return cache
    end
end

function _check_value_snapshot(cache::KLULinSolveCache, op::AbstractString)
    cache.snapshot_values || error(
        "$op: the cache was built with snapshot_values = false and holds no copy of " *
        "the factored values; build it with snapshot_values = true" *
        " (or call condest!(cache, A)).",
    )
    isempty(cache.nzval) && error(
        "$op: requires a previous numeric_refactor! to have populated nzval.",
    )
    return
end

# `zero_based[i] + 1 == one_based[i]` for every i, without writing either array. No early
# exit: the branch-free reduction vectorizes, and a mismatch is the rare (error) case.
function _offset_pattern_equal(
    zero_based::Vector{Ti},
    one_based::AbstractVector{<:Integer},
) where {Ti}
    ok = true
    @inbounds @simd for i in eachindex(zero_based, one_based)
        ok &= zero_based[i] + one(Ti) == one_based[i]
    end
    return ok
end

function _check_pattern_match(
    cache::KLULinSolveCache,
    A::SparseMatrixCSC,
    op::AbstractString,
)
    Acolptr = getcolptr(A)
    Arowval = rowvals(A)
    if length(Acolptr) != length(cache.colptr) ||
       length(Arowval) != length(cache.rowval)
        throw(
            ArgumentError(
                "Cannot $op: matrix has different sparsity structure (length).",
            ),
        )
    end
    if !_offset_pattern_equal(cache.colptr, Acolptr) ||
       !_offset_pattern_equal(cache.rowval, Arowval)
        throw(ArgumentError(
            "Cannot $op: matrix has different sparsity structure.",
        ))
    end
    return nothing
end

"""
    symbolic_factor!(cache, A)

Free any cached symbolic/numeric factor, replace the structural arrays with
`A`'s pattern, and run `klu_analyze` / `klu_l_analyze`.
"""
function symbolic_factor!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        _take_pattern!(cache, A)
        _analyze!(cache)
        return cache
    end
end

"""
    defer_symbolic!(cache, A) -> cache

`symbolic_factor!` without the `klu_analyze`: the cache takes `A`'s pattern and runs the
analysis the first time a KLU factorization needs it (a lean reject, `pivoted_factor!`, or a
`numeric_refactor!` without a lean plan), counted in `lean_counts`. For a cache about to get a
lean plan, whose factorizations and solves never read the symbolic analysis.
"""
function defer_symbolic!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        _take_pattern!(cache, A)
        cache.analyze_pending = true
        return cache
    end
end

function _take_pattern!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    n = _dim(cache)
    if size(A, 1) != Int(n) || size(A, 2) != Int(n)
        throw(
            DimensionMismatch(
                "Cannot factor: cache is $(Int(n))×$(Int(n)) but A is $(size(A)).",
            ),
        )
    end
    _free_klu_handles!(cache)

    Acolptr = getcolptr(A)
    Arowval = rowvals(A)
    resize!(cache.colptr, length(Acolptr))
    copyto!(cache.colptr, Acolptr)
    cache.colptr .-= one(Ti)
    resize!(cache.rowval, length(Arowval))
    copyto!(cache.rowval, Arowval)
    cache.rowval .-= one(Ti)
    cache.has_lean_plan &= _lean_plan_matches(cache, cache.lean_plan)
    return
end

function _analyze!(cache::KLULinSolveCache{Tv, Ti}) where {Tv, Ti}
    sym = _analyze_call(
        Ti, _dim(cache), pointer(cache.colptr), pointer(cache.rowval), cache.common)
    sym == C_NULL && klu_throw(cache.common[], "klu_analyze")
    cache.symbolic = reinterpret(Ptr{Cvoid}, sym)
    cache.analyze_pending = false
    return
end

function _require_symbolic!(cache::KLULinSolveCache, op::AbstractString)
    if cache.analyze_pending
        _analyze!(cache)
        cache.late_analyses += 1
    end
    cache.symbolic == C_NULL &&
        error("KLULinSolveCache: call symbolic_factor! before $op.")
    return
end

"""
    symbolic_refactor!(cache, A)

If `cache.reuse_symbolic`, optionally verify the structure matches and reuse
the existing analysis. Otherwise, rerun `symbolic_factor!`.
"""
function symbolic_refactor!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        if !cache.reuse_symbolic
            return symbolic_factor!(cache, A)
        end
        if cache.check_pattern
            n = _dim(cache)
            if size(A, 1) != Int(n) || size(A, 2) != Int(n)
                throw(
                    DimensionMismatch(
                        "Cannot refactor: cache is $(Int(n))×$(Int(n)) but A is $(size(A)).",
                    ),
                )
            end
            _check_pattern_match(cache, A, "symbolic_refactor")
        end
        return cache
    end
end

"""
    numeric_refactor!(cache, A)

Compute (or refresh) the numeric factorization. The first call after
`symbolic_factor!` invokes `klu_*_factor`; subsequent calls invoke
`klu_*_refactor` and reuse the existing numeric struct.
"""
function numeric_refactor!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        cache.symbolic == C_NULL && !cache.analyze_pending &&
            error(
                "KLULinSolveCache: call symbolic_factor! before numeric_refactor!.",
            )
        Anz = nonzeros(A)
        if cache.has_lean_plan && !cache.lean_paused
            cache.check_pattern && _check_pattern_match(cache, A, "numeric_refactor")
            _lean_numeric!(cache, Anz)
        end
        cache.lean_active || _require_symbolic!(cache, "numeric_refactor!")
        if !cache.lean_active && cache.numeric == C_NULL
            num = _factor_call(
                Tv, Ti,
                pointer(cache.colptr), pointer(cache.rowval),
                pointer(Anz), cache.symbolic, cache.common,
            )
            num == C_NULL && klu_throw(cache.common[], "klu_factor")
            cache.numeric = num
        elseif !cache.lean_active
            cache.check_pattern && _check_pattern_match(cache, A, "numeric_refactor")
            ok = _refactor_call(
                Tv, Ti,
                pointer(cache.colptr), pointer(cache.rowval),
                pointer(Anz), cache.symbolic, cache.numeric, cache.common,
            )
            ok != 1 && klu_throw(cache.common[], "klu_refactor")
        end
        if cache.snapshot_values
            resize!(cache.nzval, length(Anz))
            copyto!(cache.nzval, Anz)
        end
        return cache
    end
end

"""
    full_factor!(cache, A) -> cache

Run a fresh symbolic analysis followed by a numeric factorization on `A`.
Equivalent to `symbolic_factor!(cache, A); numeric_refactor!(cache, A)`. Use
this on a freshly constructed cache, or after `_free_klu_handles!` has cleared
the handles, to bring the cache to a factored state.
"""
function full_factor!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        symbolic_factor!(cache, A)
        numeric_refactor!(cache, A)
        return cache
    end
end

"""
    full_refactor!(cache, A) -> cache

Refresh both the symbolic and numeric factorizations on `A`. Defers to
`symbolic_refactor!` (which reuses the existing analysis when
`cache.reuse_symbolic` is set) followed by `numeric_refactor!`. Use this when
the matrix values have changed; if the structure has also changed and the
cache was built with `reuse_symbolic = false`, the symbolic analysis is rerun
as well.
"""
function full_refactor!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        symbolic_refactor!(cache, A)
        numeric_refactor!(cache, A)
        return cache
    end
end

"""
    klu_factorize(A; reuse_symbolic=true, check_pattern=true, snapshot_values=true) -> KLULinSolveCache

Build a cache for `A` and immediately compute the full factorization.
"""
function klu_factorize(
    A::SparseMatrixCSC{Tv, Ti};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
    snapshot_values::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}, Ti <: Union{Int32, Int64}}
    cache = KLULinSolveCache(
        A;
        reuse_symbolic = reuse_symbolic,
        check_pattern = check_pattern,
        snapshot_values = snapshot_values,
    )
    return full_factor!(cache, A)
end

# ---------------------------------------------------------------------------
# Performance / diagnostic surface
# ---------------------------------------------------------------------------

"""
    condest!(cache) -> Float64
    condest!(cache, A) -> Float64

Compute the 1-norm condition-number estimate of the cached factorization
via libklu's `klu_condest`. The result lands in `cache.common[].condest`
and is also returned. Cost is roughly two extra solves; use sparingly.

Useful when deciding whether iterative refinement is worth running, or for
flagging near-singular Jacobians in Newton-Raphson loops.

The one-argument form reads the values snapshot and errors on a cache built
with `snapshot_values = false`; the two-argument form reads the values of `A`,
which must be the matrix of the last `numeric_refactor!`.

Float64 only.
"""
function condest!(
    cache::KLULinSolveCache{Float64, Ti},
) where {Ti}
    return _with_owner(cache) do
        is_factored(cache) ||
            error("condest!: cache must be factored before condest.")
        _require_klu_numeric(cache, "condest!")
        _check_value_snapshot(cache, "condest!")
        return _condest!(cache, cache.nzval)
    end
end

function condest!(
    cache::KLULinSolveCache{Float64, Ti},
    A::SparseMatrixCSC{Float64, Ti},
) where {Ti}
    return _with_owner(cache) do
        is_factored(cache) ||
            error("condest!: cache must be factored before condest.")
        _require_klu_numeric(cache, "condest!")
        _check_pattern_match(cache, A, "condest!")
        return _condest!(cache, nonzeros(A))
    end
end

function _condest!(cache::KLULinSolveCache{Float64, Ti}, ax::Vector{Float64}) where {Ti}
    ok = GC.@preserve ax _condest_call(
        Ti,
        pointer(cache.colptr),
        pointer(ax),
        cache.symbolic,
        cache.numeric,
        cache.common,
    )
    ok != 1 && klu_throw(cache.common[], "klu_condest")
    return Float64(cache.common[].condest)
end

# ---------------------------------------------------------------------------
# Lean static-pivot path
# ---------------------------------------------------------------------------

function _lean_plan_matches(cache::KLULinSolveCache, plan::LeanLUPlan)
    return length(cache.colptr) == length(plan.a_colptr) &&
           length(cache.rowval) == length(plan.a_rowval) &&
           _offset_pattern_equal(cache.colptr, plan.a_colptr) &&
           _offset_pattern_equal(cache.rowval, plan.a_rowval)
end

"""
    set_lean_plan!(cache::KLULinSolveCache{Float64}, plan::LeanLUPlan) -> cache

Route `numeric_refactor!` and `solve!` through the lean static-pivot LU on `plan`, which
must have been built (`build_lean_plan`) on a matrix with the cache's pattern. The plan is
shared by reference; the cache gets its own LU values. A refactor whose pivot ratio falls
below `LEAN_REJECT_RATIO * plan.rcond0`, or hits a zero or non-finite pivot, is redone
with a fresh `klu_factor` in this cache and counted in `lean_counts`.

A lean factorization supports only `solve!`. `tsolve!`, `solve_sparse!` and `condest!` need a
KLU numeric factorization and raise an error on a lean one. To use them, call
`pause_lean!(cache, true)` and then `numeric_refactor!`, or call `pivoted_factor!`.
"""
function set_lean_plan!(cache::KLULinSolveCache{Float64}, plan::LeanLUPlan)
    return _with_owner(cache) do
        _lean_plan_matches(cache, plan) || throw(
            ArgumentError("set_lean_plan!: the plan's pattern differs from the cache's."),
        )
        # An accepted lean refactor leaves the KLU numeric holding older factors.
        cache.lean_active && _drop_numeric!(cache)
        cache.lean_plan = plan
        cache.lean_vals = LeanLUValues(plan)
        cache.has_lean_plan = true
        cache.lean_active = false
        cache.lean_paused = false
        return cache
    end
end

"""
    share_lean_plan!(dst, src) -> dst

`set_lean_plan!(dst, plan)` with the lean plan of `src`.
"""
function share_lean_plan!(dst::KLULinSolveCache{Float64}, src::KLULinSolveCache{Float64})
    src.has_lean_plan || throw(ArgumentError("share_lean_plan!: src has no lean plan."))
    return set_lean_plan!(dst, src.lean_plan)
end

has_lean_plan(cache::KLULinSolveCache) = cache.has_lean_plan

"""Whether the current factorization is a lean LU (the last `numeric_refactor!` took the
lean path and was not rejected)."""
lean_active(cache::KLULinSolveCache) = cache.lean_active

"""
    pause_lean!(cache, paused::Bool) -> cache

While `paused`, `numeric_refactor!` skips the lean path: `klu_refactor` on the current KLU
pivot order, or `klu_factor` when there is none. For a caller whose matrices are about to
repeat a lean reject. A paused cache keeps its plan; `pause_lean!(cache, false)` resumes.
"""
function pause_lean!(cache::KLULinSolveCache, paused::Bool)
    return _with_owner(cache) do
        if paused && cache.lean_active
            # The KLU numeric is older than the lean factors; refactoring it would be stale.
            cache.lean_active = false
            _drop_numeric!(cache)
        end
        cache.lean_paused = paused
        return cache
    end
end

"""
    lean_counts(cache) -> (; attempts, rejects, solve_failures, late_analyses)

Return four counters for `cache`. `attempts` counts the lean refactors that the cache tried.
`rejects` counts the lean refactors that failed the pivot-ratio test, so the cache used
`klu_factor` instead. `solve_failures` counts the lean factorizations that passed the test but
failed their solve, so `repivot!` replaced them. `late_analyses` counts the KLU factorizations
that had to run the symbolic analysis that `defer_symbolic!` postponed.
"""
function lean_counts(cache::KLULinSolveCache)
    return (;
        attempts = cache.lean_attempts,
        rejects = cache.lean_rejects,
        solve_failures = cache.lean_solve_failures,
        late_analyses = cache.late_analyses,
    )
end

"""
    swap_lean_columns!(cache, plan::LeanLUPlan, pairs) -> cache

Pivot on `plan`, the lean plan set on `cache`, with the two columns of each `(a, b)` in `pairs`
exchanged: the lean step that factored column `a` of the plan's matrix factors column `b`, and
vice versa, each on its planned pivot row. Solutions come back in the original column order,
and the pivot-ratio test still judges every refactor. A pair whose columns differ in pattern is
left in place (a step's L/U pattern covers only its own column). If the planned pivots then
fail the pivot-ratio test, the refactor uses `klu_factor`. An empty `pairs` restores `plan`'s
own order.

PowerFlows uses it when a bus changes between PV and REF: the swap exchanges the bus's two
state columns, so the planned pivots stay non-zero.
"""
function swap_lean_columns!(cache::KLULinSolveCache{Float64}, plan::LeanLUPlan, pairs)
    return _with_owner(cache) do
        (cache.has_lean_plan && cache.lean_plan.p === plan.p) || throw(
            ArgumentError("swap_lean_columns!: `plan` is not the cache's lean plan."),
        )
        isempty(pairs) && cache.lean_plan.q === plan.q && return cache
        # The current lean factors were computed in the old column order.
        if cache.lean_active
            cache.lean_active = false
            _drop_numeric!(cache)
        end
        cache.lean_plan = plan
        isempty(pairs) && return cache
        q = copy(plan.q)
        for (a, b) in pairs
            _same_column_pattern(plan, a, b) || continue
            ia = findfirst(==(a), q)
            ib = findfirst(==(b), q)
            q[ia] = b
            q[ib] = a
        end
        cache.lean_plan = LeanLUPlan(
            plan.n, plan.rcond0, plan.p, q, plan.cp, plan.dpos, plan.row, plan.a_row,
            plan.dep_lb, plan.dep_le, plan.a_colptr, plan.a_rowval,
        )
        return cache
    end
end

function _same_column_pattern(plan::LeanLUPlan, a::Integer, b::Integer)
    cp = plan.a_colptr
    ra = cp[a]:(cp[a + 1] - 1)
    rb = cp[b]:(cp[b + 1] - 1)
    length(ra) == length(rb) || return false
    return view(plan.a_rowval, ra) == view(plan.a_rowval, rb)
end

"""
    cold_restart!(cache) -> cache

Prepare `cache` to rerun a solve that failed on a reused pivot order (a lean plan, or a KLU
numeric kept from an earlier solve) as a fresh KLU solve would run it: free the numeric
factorization and pause the lean path, so the next `numeric_refactor!` pivots afresh with
`klu_factor` on the kept symbolic analysis. Counted in `cold_retries`.
"""
function cold_restart!(cache::KLULinSolveCache)
    return _with_owner(cache) do
        drop_numeric!(cache)
        cache.lean_paused = true
        cache.cold_retries += 1
        return cache
    end
end

"""Solves rerun on `cache` by [`cold_restart!`](@ref)."""
cold_retries(cache::KLULinSolveCache) = cache.cold_retries

function _lean_numeric!(
    cache::KLULinSolveCache{Float64, Ti},
    Anz::AbstractVector{Float64},
) where {Ti}
    ratio = lean_refactor!(cache.lean_vals, cache.lean_plan, Anz)
    cache.lean_attempts += 1
    cache.lean_active = ratio >= LEAN_REJECT_RATIO * cache.lean_plan.rcond0
    if !cache.lean_active
        cache.lean_rejects += 1
        # The fallback re-pivots: a refactor would reuse an order chosen on older values.
        _drop_numeric!(cache)
    end
    return
end

"""
    drop_numeric!(cache) -> cache

Free the current numeric factorization, KLU or lean, keeping the symbolic analysis and any
lean plan. The next `numeric_refactor!` then factors afresh: lean first when a plan is set,
otherwise `klu_factor`.
"""
function drop_numeric!(cache::KLULinSolveCache)
    return _with_owner(cache) do
        cache.lean_active = false
        _drop_numeric!(cache)
        return cache
    end
end

"""
    pivoted_factor!(cache, A) -> cache

`klu_factor` of `A` on the cache's symbolic analysis: fresh partial pivoting, bypassing the
lean plan for this call only (later `numeric_refactor!` calls use it again). Leaves a KLU
numeric factorization current, so `tsolve!` and `condest!` work.
"""
function pivoted_factor!(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC{Tv, Ti},
) where {Tv, Ti}
    return _with_owner(cache) do
        _require_symbolic!(cache, "pivoted_factor!")
        cache.check_pattern && _check_pattern_match(cache, A, "pivoted_factor")
        drop_numeric!(cache)
        Anz = nonzeros(A)
        num = _factor_call(
            Tv, Ti,
            pointer(cache.colptr), pointer(cache.rowval),
            pointer(Anz), cache.symbolic, cache.common,
        )
        num == C_NULL && klu_throw(cache.common[], "klu_factor")
        cache.numeric = num
        if cache.snapshot_values
            resize!(cache.nzval, length(Anz))
            copyto!(cache.nzval, Anz)
        end
        return cache
    end
end

"""
    repivot!(cache, A) -> cache

`pivoted_factor!` after a solve on the current factorization of `A` failed: the pattern is
unchanged, so the symbolic analysis is kept. The lean path stays paused for the rest of that
solve (`pause_lean!`); a replaced lean factorization counts as a `lean_counts` solve failure.
"""
function repivot!(cache::KLULinSolveCache, A::SparseMatrixCSC)
    return _with_owner(cache) do
        cache.lean_active && (cache.lean_solve_failures += 1)
        pivoted_factor!(cache, A)
        cache.lean_paused = true
        return cache
    end
end

function _require_klu_numeric(cache::KLULinSolveCache, op::AbstractString)
    cache.lean_active && error(
        "$op: the current factorization is a lean LU, which supports solve! only.",
    )
    return
end

"""
    _factor_stats(cache) -> (lnz, unz, nzoff, nblocks, flops)

Fill and work of the cache's current KLU numeric factorization: entries of L and U (each
including its diagonal, summed over the BTF blocks), entries of the off-diagonal BTF
blocks, the number of blocks, and `klu_flops`'s flop count.
"""
function _factor_stats(cache::KLULinSolveCache{Float64, Ti}) where {Ti}
    return _with_owner(cache) do
        _require_klu_numeric(cache, "_factor_stats")
        (cache.symbolic != C_NULL && cache.numeric != C_NULL) ||
            error("_factor_stats: cache has no KLU numeric factorization.")
        ok = _flops_call(Ti, cache.symbolic, cache.numeric, cache.common)
        ok != 1 && klu_throw(cache.common[], "klu_flops")
        shead = unsafe_load(Ptr{KluSymbolicHead{Ti}}(cache.symbolic))
        nhead = unsafe_load(Ptr{KluNumericHead{Ti}}(cache.numeric))
        return (
            lnz = Int(nhead.lnz),
            unz = Int(nhead.unz),
            nzoff = Int(shead.nzoff),
            nblocks = Int(shead.nblocks),
            flops = Float64(cache.common[].flops),
        )
    end
end

_flops_call(::Type{Int32}, sym, num, common) =
    klu_flops(reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num), common)
_flops_call(::Type{Int64}, sym, num, common) =
    klu_l_flops(reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num), common)
