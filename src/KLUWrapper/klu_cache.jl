import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros

# --- index conversions (kept private) -------------------------------------

@inline function _decrement!(v::Vector{Int64})
    @inbounds for i in eachindex(v)
        v[i] -= 1
    end
    return v
end

@inline function _increment!(v::Vector{Int64})
    @inbounds for i in eachindex(v)
        v[i] += 1
    end
    return v
end

# --- cache struct ---------------------------------------------------------

"""
A cached KLU linear solver designed for repeated solves against the same
sparse matrix structure.

Holds a 0-indexed copy of the sparsity pattern (`colptr`/`rowval`) and
opaque pointers to KLU's symbolic and numeric factorizations. `numeric_refactor!`
and `solve!` allocate nothing once the cache is built.

Type parameter `Tv ∈ {Float64, ComplexF64}` selects the real/complex KLU path.
The integer parameter is fixed to `Int64` since SuiteSparse's `_l_` entry points
take 64-bit indices and PowerNetworkMatrices only constructs `SparseMatrixCSC{T, Int}`.

# Fields
- `n`: matrix dimension.
- `colptr`, `rowval`: 0-indexed structural arrays.
- `common`: KLU's parameter/status struct (kept in a `Ref` for ccall).
- `symbolic`, `numeric`: opaque libklu pointers; `C_NULL` until factored.
- `reuse_symbolic`: if true, `symbolic_refactor!` keeps the analysis.
- `check_pattern`: if true, refactor calls verify the structural arrays
  match the matrix being passed in.
"""
mutable struct KLULinSolveCache{Tv <: Union{Float64, ComplexF64}}
    n::Int64
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    common::Base.RefValue{KluLCommon}
    symbolic::SymbolicPtr
    numeric::NumericPtr
    reuse_symbolic::Bool
    check_pattern::Bool
end

Base.size(cache::KLULinSolveCache) = (cache.n, cache.n)
Base.size(cache::KLULinSolveCache, d::Integer) = d <= 2 ? cache.n : 1
Base.eltype(::Type{KLULinSolveCache{Tv}}) where {Tv} = Tv
get_reuse_symbolic(cache::KLULinSolveCache) = cache.reuse_symbolic
is_factored(cache::KLULinSolveCache) =
    cache.symbolic != C_NULL && cache.numeric != C_NULL

# --- ccall dispatch helpers ----------------------------------------------

@inline _factor_call(::Type{Float64}, ap, ai, ax, sym, common) =
    klu_l_factor(ap, ai, ax, sym, common)
@inline _factor_call(::Type{ComplexF64}, ap, ai, ax, sym, common) =
    klu_zl_factor(ap, ai, ax, sym, common)

@inline _refactor_call(::Type{Float64}, ap, ai, ax, sym, num, common) =
    klu_l_refactor(ap, ai, ax, sym, num, common)
@inline _refactor_call(::Type{ComplexF64}, ap, ai, ax, sym, num, common) =
    klu_zl_refactor(ap, ai, ax, sym, num, common)

@inline _solve_call(::Type{Float64}, sym, num, n, nrhs, b, common) =
    klu_l_solve(sym, num, n, nrhs, b, common)
@inline _solve_call(::Type{ComplexF64}, sym, num, n, nrhs, b, common) =
    klu_zl_solve(sym, num, n, nrhs, b, common)

@inline _free_numeric!(::Type{Float64}, num_ref, common) =
    klu_l_free_numeric!(num_ref, common)
@inline _free_numeric!(::Type{ComplexF64}, num_ref, common) =
    klu_zl_free_numeric!(num_ref, common)

# --- constructor ----------------------------------------------------------

"""
    KLULinSolveCache(A; reuse_symbolic=true, check_pattern=true)

Build a cache for the sparse matrix `A`. Allocates structural arrays and runs
`klu_l_defaults`, but does **not** factorize. Call `full_factor!(cache, A)` (or
`symbolic_factor!` followed by `numeric_refactor!`) before `solve!`.

A finalizer is attached so symbolic/numeric handles are freed if the cache
is garbage collected. Prefer calling `finalize!(cache)` explicitly when the
cache is no longer needed in long-running processes.
"""
function KLULinSolveCache(
    A::SparseMatrixCSC{Tv, Int};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}}
    Int === Int64 || error(
        "KLULinSolveCache requires 64-bit Int (Julia >= 1.10 on 64-bit). " *
        "Got Int = $(Int).",
    )
    n = Int64(size(A, 1))
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square; got $(size(A))"))

    common = Ref(KluLCommon())
    klu_l_defaults!(common)

    colptr = Vector{Int64}(undef, length(getcolptr(A)))
    copyto!(colptr, getcolptr(A))
    _decrement!(colptr)
    rowval = Vector{Int64}(undef, length(rowvals(A)))
    copyto!(rowval, rowvals(A))
    _decrement!(rowval)

    cache = KLULinSolveCache{Tv}(
        n, colptr, rowval, common,
        convert(SymbolicPtr, C_NULL),
        convert(NumericPtr, C_NULL),
        reuse_symbolic, check_pattern,
    )
    finalizer(_finalize_cache!, cache)
    return cache
end

function _finalize_cache!(cache::KLULinSolveCache{Tv}) where {Tv}
    if cache.numeric != C_NULL
        num_ref = Ref(cache.numeric)
        _free_numeric!(Tv, num_ref, cache.common)
        cache.numeric = num_ref[]
    end
    if cache.symbolic != C_NULL
        sym_ref = Ref(cache.symbolic)
        klu_l_free_symbolic!(sym_ref, cache.common)
        cache.symbolic = sym_ref[]
    end
    return nothing
end

"""Explicitly free libklu resources held by `cache`. Idempotent."""
finalize!(cache::KLULinSolveCache) = _finalize_cache!(cache)

# --- structural sync ------------------------------------------------------

@inline function _check_pattern_match(cache::KLULinSolveCache,
    A::SparseMatrixCSC, op::AbstractString)
    Acolptr = getcolptr(A)
    Arowval = rowvals(A)
    if length(Acolptr) != length(cache.colptr) ||
       length(Arowval) != length(cache.rowval)
        throw(ArgumentError(
            "Cannot $op: matrix has different sparsity structure (length).",
        ))
    end
    # KLU stores 0-indexed; A is 1-indexed. Increment in place, compare,
    # decrement back. Avoids a temporary copy.
    _increment!(cache.colptr)
    _increment!(cache.rowval)
    bad = (cache.colptr != Acolptr) || (cache.rowval != Arowval)
    _decrement!(cache.colptr)
    _decrement!(cache.rowval)
    if bad
        throw(ArgumentError(
            "Cannot $op: matrix has different sparsity structure.",
        ))
    end
    return nothing
end

# --- factor / refactor ----------------------------------------------------

"""
    symbolic_factor!(cache, A)

Free any cached symbolic/numeric factor, replace the structural arrays with
`A`'s pattern, and run `klu_l_analyze`. Use this when the matrix dimensions
or sparsity have changed.
"""
function symbolic_factor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    if size(A, 1) != cache.n || size(A, 2) != cache.n
        throw(DimensionMismatch(
            "Cannot factor: cache is $(cache.n)×$(cache.n) but A is $(size(A)).",
        ))
    end
    _finalize_cache!(cache)

    Acolptr = getcolptr(A)
    Arowval = rowvals(A)
    resize!(cache.colptr, length(Acolptr))
    copyto!(cache.colptr, Acolptr)
    _decrement!(cache.colptr)
    resize!(cache.rowval, length(Arowval))
    copyto!(cache.rowval, Arowval)
    _decrement!(cache.rowval)

    sym = klu_l_analyze(
        cache.n, pointer(cache.colptr), pointer(cache.rowval), cache.common,
    )
    sym == C_NULL && klu_throw(cache.common[], "klu_l_analyze")
    cache.symbolic = sym
    return cache
end

"""
    symbolic_refactor!(cache, A)

If `cache.reuse_symbolic`, optionally verify the structure matches and reuse
the existing analysis. Otherwise, rerun `symbolic_factor!`.
"""
function symbolic_refactor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    if !cache.reuse_symbolic
        return symbolic_factor!(cache, A)
    end
    if cache.check_pattern
        if size(A, 1) != cache.n || size(A, 2) != cache.n
            throw(DimensionMismatch(
                "Cannot refactor: cache is $(cache.n)×$(cache.n) but A is $(size(A)).",
            ))
        end
        _check_pattern_match(cache, A, "symbolic_refactor")
    end
    return cache
end

"""
    numeric_refactor!(cache, A)

Compute (or refresh) the numeric factorization. The first call after
`symbolic_factor!` invokes `klu_*_factor`; subsequent calls invoke
`klu_*_refactor` and reuse the existing numeric struct (much cheaper, no
allocation in the wrapper).

When `cache.check_pattern` is true the call verifies that `A` still has the
cached sparsity pattern and throws otherwise.
"""
function numeric_refactor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    cache.symbolic == C_NULL && error(
        "KLULinSolveCache: call symbolic_factor! before numeric_refactor!.",
    )
    if cache.numeric == C_NULL
        num = _factor_call(
            Tv, pointer(cache.colptr), pointer(cache.rowval),
            pointer(nonzeros(A)), cache.symbolic, cache.common,
        )
        if num == C_NULL
            klu_throw(cache.common[], "klu_factor")
        end
        cache.numeric = num
    else
        if cache.check_pattern
            _check_pattern_match(cache, A, "numeric_refactor")
        end
        ok = _refactor_call(
            Tv, pointer(cache.colptr), pointer(cache.rowval),
            pointer(nonzeros(A)), cache.symbolic, cache.numeric, cache.common,
        )
        ok != 1 && klu_throw(cache.common[], "klu_refactor")
    end
    return cache
end

"""Run `symbolic_factor!` then `numeric_refactor!`."""
function full_factor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    symbolic_factor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

"""Run `symbolic_refactor!` then `numeric_refactor!`."""
function full_refactor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    symbolic_refactor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

"""
    klu_factorize(A; reuse_symbolic=true, check_pattern=true) -> KLULinSolveCache

Convenience: build a cache for `A` and immediately compute the full factorization.
Mirrors the role of `KLU.klu(A)` in the previous code path.
"""
function klu_factorize(A::SparseMatrixCSC{Tv, Int};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}}
    cache = KLULinSolveCache(A;
        reuse_symbolic = reuse_symbolic, check_pattern = check_pattern)
    return full_factor!(cache, A)
end
