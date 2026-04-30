import SparseArrays: SparseMatrixCSC, getcolptr, rowvals, nonzeros

"""
A cached KLU linear solver designed for repeated solves against the same
sparse matrix structure. `numeric_refactor!` and `solve!` allocate nothing
once the cache is built.

The type parameter `Tv ∈ {Float64, ComplexF64}` selects the real/complex KLU
path. Indices are always `Int64` (SuiteSparse_long).

`reuse_symbolic` controls whether `symbolic_refactor!` keeps the analysis;
`check_pattern` adds a structural-equality check on refactor calls and is
only consulted when reusing.
"""
mutable struct KLULinSolveCache{Tv <: Union{Float64, ComplexF64}}
    colptr::Vector{Int64}
    rowval::Vector{Int64}
    common::Base.RefValue{KluLCommon}
    symbolic::SymbolicPtr
    numeric::NumericPtr
    reuse_symbolic::Bool
    check_pattern::Bool
    # Bounded reusable scratch for `solve_sparse!`. Lazy-grown on first call so
    # the wrapper's working set stays O(n*block) instead of O(n*nrhs); see
    # `solve_sparse_rhs.jl`. Each cache owns its own buffer, so a
    # `KLULinSolvePool` worker is thread-safe by construction.
    scratch::Matrix{Tv}
    col_map::Vector{Int64}
end

@inline _dim(cache::KLULinSolveCache) = Int64(length(cache.colptr) - 1)

Base.size(cache::KLULinSolveCache) = (n = _dim(cache); (n, n))
Base.size(cache::KLULinSolveCache, d::Integer) = d <= 2 ? _dim(cache) : 1
Base.eltype(::Type{KLULinSolveCache{Tv}}) where {Tv} = Tv
get_reuse_symbolic(cache::KLULinSolveCache) = cache.reuse_symbolic
is_factored(cache::KLULinSolveCache) =
    cache.symbolic != C_NULL && cache.numeric != C_NULL

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

@inline _tsolve_call(::Type{Float64}, sym, num, n, nrhs, b, common; conjugate = false) =
    klu_l_tsolve(sym, num, n, nrhs, b, common)
@inline _tsolve_call(::Type{ComplexF64}, sym, num, n, nrhs, b, common; conjugate = false) =
    klu_zl_tsolve(sym, num, n, nrhs, b, Cint(conjugate), common)

@inline _free_numeric!(::Type{Float64}, num_ref, common) =
    klu_l_free_numeric!(num_ref, common)
@inline _free_numeric!(::Type{ComplexF64}, num_ref, common) =
    klu_zl_free_numeric!(num_ref, common)

"""
    KLULinSolveCache(A; reuse_symbolic=true, check_pattern=true)

Build a cache for the sparse matrix `A`. Allocates structural arrays and
runs `klu_l_defaults`, but does **not** factorize. Call `full_factor!`
(or `symbolic_factor!` followed by `numeric_refactor!`) before `solve!`.

A finalizer frees libklu handles on GC; call `Base.finalize(cache)` to
release them eagerly. Releasing the handles leaves Julia-side state intact,
so the cache can be re-factorized via `symbolic_factor!`/`numeric_refactor!`
or `full_factor!`.
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
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square; got $(size(A))"))

    common = Ref(KluLCommon())
    klu_l_defaults!(common)

    colptr = Vector{Int64}(undef, length(getcolptr(A)))
    copyto!(colptr, getcolptr(A))
    colptr .-= 1
    rowval = Vector{Int64}(undef, length(rowvals(A)))
    copyto!(rowval, rowvals(A))
    rowval .-= 1

    cache = KLULinSolveCache{Tv}(
        colptr, rowval, common,
        convert(SymbolicPtr, C_NULL),
        convert(NumericPtr, C_NULL),
        reuse_symbolic, check_pattern,
        Matrix{Tv}(undef, 0, 0),
        Int64[],
    )
    finalizer(_free_klu_handles!, cache)
    return cache
end

"""
    _ensure_scratch!(cache, block) -> Nothing

Ensure `cache.scratch` is at least `n × block` and `cache.col_map` length
`block`. Grows in place; reuses across `solve_sparse!` calls.
"""
@inline function _ensure_scratch!(cache::KLULinSolveCache{Tv}, block::Int) where {Tv}
    n = _dim(cache)
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
function _free_klu_handles!(cache::KLULinSolveCache{Tv}) where {Tv}
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

# Public eager-release entry point. Overload of `Base.finalize` so callers can
# trigger handle release without waiting for GC; delegates to the internal
# helper to keep the GC-end-of-life semantics distinct from the in-method
# "drop old handles before re-analyze" use inside `symbolic_factor!`.
Base.finalize(cache::KLULinSolveCache) = _free_klu_handles!(cache)

@inline function _check_pattern_match(cache::KLULinSolveCache,
    A::SparseMatrixCSC, op::AbstractString)
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
    # Increment-compare-decrement: avoids allocating a 1-indexed copy.
    cache.colptr .+= 1
    cache.rowval .+= 1
    bad = (cache.colptr != Acolptr) || (cache.rowval != Arowval)
    cache.colptr .-= 1
    cache.rowval .-= 1
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
`A`'s pattern, and run `klu_l_analyze`.
"""
function symbolic_factor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    n = _dim(cache)
    if size(A, 1) != n || size(A, 2) != n
        throw(DimensionMismatch(
            "Cannot factor: cache is $(n)×$(n) but A is $(size(A)).",
        ))
    end
    _free_klu_handles!(cache)

    Acolptr = getcolptr(A)
    Arowval = rowvals(A)
    resize!(cache.colptr, length(Acolptr))
    copyto!(cache.colptr, Acolptr)
    cache.colptr .-= 1
    resize!(cache.rowval, length(Arowval))
    copyto!(cache.rowval, Arowval)
    cache.rowval .-= 1

    sym = @klu_gc_preserve cache begin
        klu_l_analyze(
            Int64(n), pointer(cache.colptr), pointer(cache.rowval), cache.common,
        )
    end
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
        n = _dim(cache)
        if size(A, 1) != n || size(A, 2) != n
            throw(
                DimensionMismatch(
                    "Cannot refactor: cache is $(n)×$(n) but A is $(size(A)).",
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
function numeric_refactor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    cache.symbolic == C_NULL && error(
        "KLULinSolveCache: call symbolic_factor! before numeric_refactor!.",
    )
    if cache.numeric == C_NULL
        num = @klu_gc_preserve cache A begin
            _factor_call(
                Tv, pointer(cache.colptr), pointer(cache.rowval),
                pointer(nonzeros(A)), cache.symbolic, cache.common,
            )
        end
        num == C_NULL && klu_throw(cache.common[], "klu_factor")
        cache.numeric = num
    else
        cache.check_pattern && _check_pattern_match(cache, A, "numeric_refactor")
        ok = @klu_gc_preserve cache A begin
            _refactor_call(
                Tv, pointer(cache.colptr), pointer(cache.rowval),
                pointer(nonzeros(A)), cache.symbolic, cache.numeric, cache.common,
            )
        end
        ok != 1 && klu_throw(cache.common[], "klu_refactor")
    end
    return cache
end

function full_factor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    symbolic_factor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

function full_refactor!(cache::KLULinSolveCache{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    symbolic_refactor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

"""
    klu_factorize(A; reuse_symbolic=true, check_pattern=true) -> KLULinSolveCache

Build a cache for `A` and immediately compute the full factorization.
"""
function klu_factorize(A::SparseMatrixCSC{Tv, Int};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}}
    cache = KLULinSolveCache(A;
        reuse_symbolic = reuse_symbolic, check_pattern = check_pattern)
    return full_factor!(cache, A)
end
