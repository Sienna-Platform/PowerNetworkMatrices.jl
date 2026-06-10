"""
A cached libSparse linear solver for repeated solves against the same sparse
matrix structure. `numeric_refactor!` and `solve!` allocate nothing once the
cache is built.

Cache the symbolic and numeric factorizations of a general (unsymmetric) sparse
matrix (LU with threshold partial pivoting + Inf-norm equilibration scaling) and
reuse them across many solves. The full CSC pattern of `A` is stored and the
libSparse structure view is marked `ATT_ORDINARY`. No symmetry requirement
(matches KLU's pivoting model).

Float64 only. Requires macOS 15.5+ (enforced by the backend selection in
`linalg_settings.jl`).

`reuse_symbolic` controls whether `symbolic_refactor!` keeps the analysis;
`check_pattern` adds a structural-equality check on refactor calls and is
only consulted when reusing.
"""
mutable struct AAFactorCache
    # Apple-side 0-based, narrower-integer copies of the full input CSC pattern.
    # Reused as-is across `numeric_refactor!` calls.
    columnStarts::Vector{Clong}
    rowIndices::Vector{Cint}
    nzval::Vector{Cdouble}
    n::Int
    # Count of stored entries: full `nnz(A)`.
    nnz::Int
    symbolic::SparseOpaqueSymbolicFactorization
    numeric::SparseOpaqueFactorization_t
    reuse_symbolic::Bool
    check_pattern::Bool
    # Bounded reusable scratch for `solve_sparse!`. Lazy-grown on first call.
    scratch::Matrix{Cdouble}
    col_map::Vector{Int}
    # Reusable libSparse solve workspace. Sized to
    # `static + nrhs * per_rhs` bytes; supplied to the workspace-aware
    # `SparseSolve` overloads so libSparse does not malloc/free per call.
    # Float64 storage chosen for 16-byte alignment; length is in Float64s.
    solve_workspace::Vector{Float64}
    # Scaling method passed to libSparse at numeric-factor time.
    # `SparseScalingEquilibriationInf` reduces fill on symmetric SPD-ish inputs
    # like ABA (~4× faster multi-RHS solve, residual still O(1e-13)). Set via
    # the `scaling` kwarg of `AAFactorCache` / `aa_factorize`.
    scaling::SparseScaling_t
end

@inline _dim(cache::AAFactorCache) = cache.n

Base.size(cache::AAFactorCache) = (cache.n, cache.n)
function Base.size(cache::AAFactorCache, d::Integer)
    if d <= 2
        return cache.n
    else
        return 1
    end
end
Base.eltype(::Type{AAFactorCache}) = Cdouble

"""
    is_factored(cache::AAFactorCache) -> Bool

`true` when `cache` holds a valid numeric factorization ready for `solve!` /
`solve_sparse!`. `false` after construction (before `full_factor!`) or after
the libSparse handles have been finalized.
"""
function is_factored(cache::AAFactorCache)
    return cache.numeric.status == SparseStatusOk && cache.symbolic.status == SparseStatusOk
end

"""
    AAFactorCache(A; reuse_symbolic=true, check_pattern=true,
                  scaling=SparseScalingEquilibriationInf)

Build a cache for the square sparse matrix `A`. Allocates the Apple-side
structural arrays (`columnStarts`, `rowIndices`, `nzval`) but does **not**
factorize. Call `full_factor!` (or `symbolic_factor!` followed by
`numeric_refactor!`) before `solve!`.

The arrays are sized to the full pattern of `A` (general/LU mode). No symmetry
requirement — matches KLU's pivoting model.

A finalizer frees libSparse handles on GC; call `Base.finalize(cache)` to
release them eagerly.
"""
function AAFactorCache(
    A::SparseMatrixCSC{Float64, Int};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
    scaling::SparseScaling_t = SparseScalingEquilibriationInf,
)
    n = size(A, 1)
    n == size(A, 2) || throw(DimensionMismatch("matrix must be square; got $(size(A))"))
    stored_nnz = SparseArrays.nnz(A)
    cache = AAFactorCache(
        Vector{Clong}(undef, n + 1),
        Vector{Cint}(undef, stored_nnz),
        Vector{Cdouble}(undef, 0),
        n,
        stored_nnz,
        _null_symbolic(),
        _null_factorization(),
        reuse_symbolic,
        check_pattern,
        Matrix{Cdouble}(undef, 0, 0),
        Int[],
        Float64[],
        scaling,
    )
    _populate_pattern!(cache, A)
    finalizer(_free_handles!, cache)
    return cache
end

# --- general (LU) mode helpers ----------------------------------------------

# Copy A's full CSC pattern into `cache.columnStarts` / `cache.rowIndices`
# as 0-based narrowed indices. Caller sized these to (n+1) and nnz.
function _populate_pattern!(
    cache::AAFactorCache,
    A::SparseMatrixCSC{Float64, Int},
)
    cp = getcolptr(A)
    rv = rowvals(A)
    @inbounds for k in eachindex(cp)
        cache.columnStarts[k] = Clong(cp[k] - 1)
    end
    @inbounds for k in eachindex(rv)
        cache.rowIndices[k] = Cint(rv[k] - 1)
    end
    return cache
end

# Snapshot the full nonzeros into `cache.nzval`, growing if needed.
function _populate_values!(
    cache::AAFactorCache,
    A::SparseMatrixCSC{Float64, Int},
)
    if length(cache.nzval) != cache.nnz
        resize!(cache.nzval, cache.nnz)
    end
    copyto!(cache.nzval, nonzeros(A))
    return cache
end

# --- pattern guard ----------------------------------------------------------

# Pattern-match guard for `numeric_refactor!`: assert the incoming CSC's
# stored pattern is identical to what was analyzed (full pattern, LU mode).
# Apple's arrays are 0-based but we always store them 0-based — no flipping.
function _check_pattern_match(
    cache::AAFactorCache,
    A::SparseMatrixCSC{Float64, Int},
    op::AbstractString,
)
    n = cache.n
    if size(A, 1) != n || size(A, 2) != n
        throw(DimensionMismatch("Cannot $op: cache is $(n)×$(n) but A is $(size(A))."))
    end
    cp = getcolptr(A)
    rv = rowvals(A)
    length(rv) == cache.nnz || return _pattern_mismatch(op)
    @inbounds for k in eachindex(cp)
        cache.columnStarts[k] == Clong(cp[k] - 1) || return _pattern_mismatch(op)
    end
    @inbounds for k in eachindex(rv)
        cache.rowIndices[k] == Cint(rv[k] - 1) || return _pattern_mismatch(op)
    end
    return nothing
end

_pattern_mismatch(op::AbstractString) =
    throw(ArgumentError("Cannot $op: matrix has different sparsity structure."))

"""
Release the libSparse numeric and symbolic handles held by `cache`, leaving
Julia-side state intact. Idempotent.
"""
function _free_handles!(cache::AAFactorCache)
    if cache.numeric.status == SparseStatusOk
        _sparse_cleanup_factor!(cache.numeric)
        cache.numeric = _null_factorization()
    end
    if cache.symbolic.status == SparseStatusOk
        _sparse_cleanup_symbolic!(cache.symbolic)
        cache.symbolic = _null_symbolic()
    end
    return nothing
end

Base.finalize(cache::AAFactorCache) = _free_handles!(cache)

# `deepcopy` is unsafe: it would bit-copy the raw libSparse factor handles,
# aliasing one factorization across two finalizer-owning caches (a
# double-free/use-after-free). Throw so the caller shares the owning matrix by
# reference instead; the backtrace names the offending site.
function Base.deepcopy_internal(::AAFactorCache, ::IdDict)
    error(
        "deepcopy of an AAFactorCache is unsafe and was attempted: the cache holds raw " *
        "libSparse factor handles that deepcopy would alias by value, causing a " *
        "double-free / use-after-free of the factorization (SIGSEGV/SIGILL in SparseSolve). " *
        "Share the owning Virtual matrix by reference instead of deepcopying it. The " *
        "backtrace identifies the offending call site.",
    )
end

# Build the libSparse `SparseMatrixStructure` view that points into the
# cache's owned arrays. Always marked `ATT_ORDINARY` — the LU path treats
# `cache.nzval` as the full matrix with no symmetry assumed.
function _structure_view(cache::AAFactorCache)
    return SparseMatrixStructure(
        Cint(cache.n),
        Cint(cache.n),
        pointer(cache.columnStarts),
        pointer(cache.rowIndices),
        ATT_ORDINARY,
        UInt8(1),
    )
end

function _matrix_view(cache::AAFactorCache)
    return SparseMatrix_t(_structure_view(cache), pointer(cache.nzval))
end

"""
    symbolic_factor!(cache, A)

Free any cached symbolic/numeric factor, replace the structural arrays with
`A`'s full pattern, and analyze. Subsequent `numeric_refactor!` calls reuse
the analysis.
"""
function symbolic_factor!(cache::AAFactorCache, A::SparseMatrixCSC{Float64, Int})
    n = cache.n
    if size(A, 1) != n || size(A, 2) != n
        throw(DimensionMismatch("Cannot factor: cache is $(n)×$(n) but A is $(size(A))."))
    end
    _free_handles!(cache)
    new_nnz = SparseArrays.nnz(A)
    if new_nnz != cache.nnz
        resize!(cache.rowIndices, new_nnz)
        cache.nnz = new_nnz
    end
    if length(cache.columnStarts) != n + 1
        resize!(cache.columnStarts, n + 1)
    end
    _populate_pattern!(cache, A)
    sym = _sparse_symbolic_factor(
        SparseFactorizationLU,
        _structure_view(cache),
        SparseSymbolicFactorOptions(),
    )
    if sym.status != SparseStatusOk
        # libSparse may have allocated C-side state before deciding to fail —
        # release it before throwing so we don't leak per failed factor.
        _sparse_cleanup_symbolic!(sym)
        _libsparse_throw(sym.status, "symbolic factor")
    end
    cache.symbolic = sym
    return cache
end

"""
    numeric_refactor!(cache, A)

Refresh the numeric factor on top of the existing symbolic analysis. Errors
if `symbolic_factor!` has not been called yet.
"""
function numeric_refactor!(cache::AAFactorCache, A::SparseMatrixCSC{Float64, Int})
    cache.symbolic.status == SparseStatusOk ||
        error("AAFactorCache: call symbolic_factor! before numeric_refactor!.")
    cache.check_pattern && _check_pattern_match(cache, A, "numeric_refactor")
    _populate_values!(cache, A)
    if cache.numeric.status == SparseStatusOk
        _sparse_cleanup_factor!(cache.numeric)
        cache.numeric = _null_factorization()
    end
    num = _sparse_numeric_factor(
        cache.symbolic,
        _matrix_view(cache),
        SparseNumericFactorOptions(cache.scaling),
    )
    if num.status != SparseStatusOk
        # Same rationale as in symbolic_factor!: release before throwing.
        _sparse_cleanup_factor!(num)
        _libsparse_throw(num.status, "numeric factor")
    end
    cache.numeric = num
    return cache
end

"""
    symbolic_refactor!(cache, A)

If `cache.reuse_symbolic`, optionally verify the structure matches and reuse
the existing analysis. Otherwise, rerun `symbolic_factor!`.
"""
function symbolic_refactor!(cache::AAFactorCache, A::SparseMatrixCSC{Float64, Int})
    if !cache.reuse_symbolic
        return symbolic_factor!(cache, A)
    end
    cache.check_pattern && _check_pattern_match(cache, A, "symbolic_refactor")
    return cache
end

"""
    full_factor!(cache, A) -> cache

Run a fresh symbolic analysis followed by a numeric factorization on `A`.
"""
function full_factor!(cache::AAFactorCache, A::SparseMatrixCSC{Float64, Int})
    symbolic_factor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

"""
    full_refactor!(cache, A) -> cache

Refresh both factorizations on `A`. Defers to `symbolic_refactor!` (which
reuses the existing analysis when `cache.reuse_symbolic` is set) followed by
`numeric_refactor!`.
"""
function full_refactor!(cache::AAFactorCache, A::SparseMatrixCSC{Float64, Int})
    symbolic_refactor!(cache, A)
    numeric_refactor!(cache, A)
    return cache
end

"""
    aa_factorize(A; reuse_symbolic=true, check_pattern=true,
                  scaling=SparseScalingEquilibriationInf) -> AAFactorCache

Build a cache for `A` and immediately compute the full LU factorization. See
`AAFactorCache` for the kwarg semantics.
"""
function aa_factorize(
    A::SparseMatrixCSC{Float64, Int};
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
    scaling::SparseScaling_t = SparseScalingEquilibriationInf,
)
    cache = AAFactorCache(
        A;
        reuse_symbolic = reuse_symbolic,
        check_pattern = check_pattern,
        scaling = scaling,
    )
    return full_factor!(cache, A)
end

"""
    _ensure_scratch!(cache, block) -> Nothing

Ensure `cache.scratch` is at least `n × block` and `cache.col_map` length
`block`. Used by `solve_sparse!`.
"""
@inline function _ensure_scratch!(cache::AAFactorCache, block::Int)
    n = cache.n
    s = cache.scratch
    if size(s, 1) != n || size(s, 2) < block
        cache.scratch = Matrix{Cdouble}(undef, n, block)
    end
    if length(cache.col_map) < block
        resize!(cache.col_map, block)
    end
    return nothing
end

"""
    _ensure_solve_workspace!(cache, nrhs) -> Ptr{Cvoid}

Ensure `cache.solve_workspace` is large enough to back a libSparse
`SparseSolve` call with `nrhs` right-hand sides (`static + nrhs * per_rhs`
bytes per the factor's documented requirements). Grows only when too small
— steady state is no-op. Returns a `Ptr{Cvoid}` to pass to the
workspace-aware ccall (16-byte aligned by virtue of `Vector{Float64}`'s
allocator).
"""
@inline function _ensure_solve_workspace!(cache::AAFactorCache, nrhs::Integer)
    nbytes = _solve_workspace_bytes(cache.numeric, nrhs)
    # Round up to whole Float64s; min 1 element so `pointer` is well-defined
    # when libSparse asks for zero bytes (it may still dereference).
    need = max(cld(nbytes, 8), 1)
    if length(cache.solve_workspace) < need
        resize!(cache.solve_workspace, need)
    end
    return convert(Ptr{Cvoid}, pointer(cache.solve_workspace))
end
