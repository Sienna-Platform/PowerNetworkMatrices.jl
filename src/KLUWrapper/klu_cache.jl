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
only consulted when reusing.
"""
mutable struct KLULinSolveCache{
    Tv <: Union{Float64, ComplexF64},
    Ti <: Union{Int32, Int64},
}
    colptr::Vector{Ti}
    rowval::Vector{Ti}
    # Copy of the matrix values used in the most recent numeric factorization.
    # Lets `_recover_factorization!` rebuild a corrupted numeric handle without
    # the caller having to re-supply `A`. Empty before the first factor call.
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
    # Bounded reusable scratch for `solve_sparse!`. Lazy-grown on first call so
    # the wrapper's working set stays O(n*block) instead of O(n*nrhs); see
    # `solve_sparse_rhs.jl`.
    scratch::Matrix{Tv}
    col_map::Vector{Ti}
end

@inline _dim(cache::KLULinSolveCache{Tv, Ti}) where {Tv, Ti} =
    Ti(length(cache.colptr) - 1)

Base.size(cache::KLULinSolveCache) = (n = Int(_dim(cache)); (n, n))
Base.size(cache::KLULinSolveCache, d::Integer) =
    d <= 2 ? Int(_dim(cache)) : 1
Base.eltype(::Type{KLULinSolveCache{Tv, Ti}}) where {Tv, Ti} = Tv
get_reuse_symbolic(cache::KLULinSolveCache) = cache.reuse_symbolic

"""
    is_factored(cache::KLULinSolveCache) -> Bool

Return `true` when `cache` holds both a symbolic and a numeric factorization
ready for `solve!` / `tsolve!` / `solve_sparse!`. Returns `false` after
construction (before `full_factor!`) or after the libklu handles have been
finalized.
"""
is_factored(cache::KLULinSolveCache) =
    cache.symbolic != C_NULL && cache.numeric != C_NULL

# A cache holds at most one valid factorization; `is_factored` is the
# truthy form. Kept around for tests that want a uniform numeric reading.
n_valid(cache::KLULinSolveCache) = is_factored(cache) ? 1 : 0

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

# Performance-knob dispatchers — Float64 only. libklu exposes these in the
# ComplexF64 build too; we'll add bindings if a consumer asks.
@inline _sort_call(::Type{Int32}, sym, num, common) =
    klu_sort(reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num), common)
@inline _sort_call(::Type{Int64}, sym, num, common) =
    klu_l_sort(reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num), common)

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

@inline _rcond_call(::Type{Int32}, sym, num, common) =
    klu_rcond(reinterpret(SymbolicPtr32, sym), reinterpret(NumericPtr32, num), common)
@inline _rcond_call(::Type{Int64}, sym, num, common) =
    klu_l_rcond(reinterpret(SymbolicPtr, sym), reinterpret(NumericPtr, num), common)

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    KLULinSolveCache(A; reuse_symbolic=true, check_pattern=true)

Build a cache for the sparse matrix `A`. The cache's index type is taken
from `A`: `SparseMatrixCSC{Tv, Int32}` ⇒ `KLULinSolveCache{Tv, Int32}`,
`SparseMatrixCSC{Tv, Int64}` ⇒ `KLULinSolveCache{Tv, Int64}`. Allocates
structural arrays and runs the corresponding `klu_defaults`/`klu_l_defaults`
initializer, but does **not** factorize. Call `full_factor!` (or
`symbolic_factor!` followed by `numeric_refactor!`) before `solve!`.

A finalizer frees libklu handles on GC; call `Base.finalize(cache)` to
release them eagerly. Releasing the handles leaves Julia-side state intact,
so the cache can be re-factorized via `symbolic_factor!`/`numeric_refactor!`
or `full_factor!`.
"""
function KLULinSolveCache(
    A::SparseMatrixCSC{Tv, Ti};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
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
        reuse_symbolic, check_pattern,
        Matrix{Tv}(undef, 0, 0),
        Ti[],
    )
    finalizer(_free_klu_handles!, cache)
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

"""
Release the libklu numeric and symbolic handles held by `cache`, leaving the
Julia-side fields (`colptr`, `rowval`, `common`, `scratch`, `col_map`) intact
so the cache remains structurally valid and re-factorable. Idempotent: a
second call hits the `C_NULL` guards. Used both by `symbolic_factor!`
mid-life (drop old handles before re-analyzing) and by the GC finalizer.
"""
function _free_klu_handles!(
    cache::KLULinSolveCache{Tv, Ti},
) where {Tv, Ti}
    if cache.numeric != C_NULL
        num_ref = Ref(cache.numeric)
        _free_numeric!(Tv, Ti, num_ref, cache.common)
        cache.numeric = num_ref[]
    end
    if cache.symbolic != C_NULL
        sym_ref = Ref(cache.symbolic)
        _free_symbolic!(Ti, sym_ref, cache.common)
        cache.symbolic = sym_ref[]
    end
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
    cache.symbolic == C_NULL && error(
        "KLULinSolveCache: cannot recover without a symbolic factor.",
    )
    isempty(cache.nzval) && error(
        "KLULinSolveCache: cannot recover; no cached numerical values yet.",
    )
    if cache.numeric != C_NULL
        num_ref = Ref(cache.numeric)
        _free_numeric!(Tv, Ti, num_ref, cache.common)
        cache.numeric = num_ref[]
    end
    num = _factor_call(
        Tv, Ti,
        pointer(cache.colptr), pointer(cache.rowval),
        pointer(cache.nzval), cache.symbolic, cache.common,
    )
    num == C_NULL && klu_throw(cache.common[], "klu_factor (recovery)")
    cache.numeric = num
    return cache
end

@inline function _check_pattern_match(
    cache::KLULinSolveCache{Tv, Ti},
    A::SparseMatrixCSC,
    op::AbstractString,
) where {Tv, Ti}
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
    # Increment-compare-decrement: avoids allocating a 1-indexed copy. The
    # `try/finally` restores `colptr`/`rowval` even on InterruptException so
    # the cache is never left with off-by-one structural arrays.
    cache.colptr .+= one(Ti)
    cache.rowval .+= one(Ti)
    bad = try
        (cache.colptr != Acolptr) || (cache.rowval != Arowval)
    finally
        cache.colptr .-= one(Ti)
        cache.rowval .-= one(Ti)
    end
    if bad
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

    sym = _analyze_call(Ti, n, pointer(cache.colptr), pointer(cache.rowval), cache.common)
    sym == C_NULL && klu_throw(cache.common[], "klu_analyze")
    cache.symbolic = reinterpret(Ptr{Cvoid}, sym)
    return cache
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
    cache.symbolic == C_NULL && error(
        "KLULinSolveCache: call symbolic_factor! before numeric_refactor!.",
    )
    Anz = nonzeros(A)
    if cache.numeric == C_NULL
        num = _factor_call(
            Tv, Ti,
            pointer(cache.colptr), pointer(cache.rowval),
            pointer(Anz), cache.symbolic, cache.common,
        )
        num == C_NULL && klu_throw(cache.common[], "klu_factor")
        cache.numeric = num
    else
        cache.check_pattern && _check_pattern_match(cache, A, "numeric_refactor")
        ok = _refactor_call(
            Tv, Ti,
            pointer(cache.colptr), pointer(cache.rowval),
            pointer(Anz), cache.symbolic, cache.numeric, cache.common,
        )
        ok != 1 && klu_throw(cache.common[], "klu_refactor")
    end
    # Snapshot the values used so `_recover_factorization!` can rebuild the
    # numeric handle without the caller having to re-supply A.
    resize!(cache.nzval, length(Anz))
    copyto!(cache.nzval, Anz)
    return cache
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
    symbolic_factor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
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
    symbolic_refactor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

"""
    klu_factorize(A; reuse_symbolic=true, check_pattern=true) -> KLULinSolveCache

Build a cache for `A` and immediately compute the full factorization.
"""
function klu_factorize(
    A::SparseMatrixCSC{Tv, Ti};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}, Ti <: Union{Int32, Int64}}
    cache = KLULinSolveCache(
        A;
        reuse_symbolic = reuse_symbolic,
        check_pattern = check_pattern,
    )
    return full_factor!(cache, A)
end

# ---------------------------------------------------------------------------
# Performance / diagnostic surface
# ---------------------------------------------------------------------------

"""
    sort_factors!(cache) -> cache

Sort the columns of the cached L and U factors in place via libklu's
`klu_sort` / `klu_l_sort`. KLU's numeric phase stores factor columns in
arbitrary order; sorting once after the first factor improves cache locality
on every subsequent `solve!` / `tsolve!`. The cost is `O(nnz_factor)` and
is amortized over many repeated solves — a win whenever the cache is used
for ≥ a few solves on the same factorization.

Idempotent and `refactor`-stable: sorting after the initial factor persists
through `numeric_refactor!` because refactor preserves the column layout.
Only call this if the cache will be reused for multiple solves; for a
one-shot solve it is pure overhead.

Float64 only.
"""
function sort_factors!(cache::KLULinSolveCache{Float64, Ti}) where {Ti}
    is_factored(cache) ||
        error("sort_factors!: cache must be factored before sorting.")
    ok = _sort_call(Ti, cache.symbolic, cache.numeric, cache.common)
    ok != 1 && klu_throw(cache.common[], "klu_sort")
    return cache
end

"""
    condest!(cache) -> Float64

Compute the 1-norm condition-number estimate of the cached factorization
via libklu's `klu_condest`. The result lands in `cache.common[].condest`
and is also returned. Cost is roughly two extra solves; use sparingly.

Useful when deciding whether iterative refinement is worth running, or for
flagging near-singular Jacobians in Newton-Raphson loops.

Float64 only.
"""
function condest!(
    cache::KLULinSolveCache{Float64, Ti},
) where {Ti}
    is_factored(cache) ||
        error("condest!: cache must be factored before condest.")
    isempty(cache.nzval) && error(
        "condest!: requires a previous numeric_refactor! to have populated nzval.",
    )
    ok = _condest_call(
        Ti,
        pointer(cache.colptr),
        pointer(cache.nzval),
        cache.symbolic,
        cache.numeric,
        cache.common,
    )
    ok != 1 && klu_throw(cache.common[], "klu_condest")
    return Float64(cache.common[].condest)
end

"""
    rcond!(cache) -> Float64

Compute the cheap reciprocal-condition estimate
`min(|diag(U)|)/max(|diag(U)|)` via libklu's `klu_rcond`. The result lands
in `cache.common[].rcond` and is also returned. Faster than `condest!` but
less reliable as a conditioning indicator.

Float64 only.
"""
function rcond!(cache::KLULinSolveCache{Float64, Ti}) where {Ti}
    is_factored(cache) ||
        error("rcond!: cache must be factored before rcond.")
    ok = _rcond_call(Ti, cache.symbolic, cache.numeric, cache.common)
    ok != 1 && klu_throw(cache.common[], "klu_rcond")
    return Float64(cache.common[].rcond)
end
