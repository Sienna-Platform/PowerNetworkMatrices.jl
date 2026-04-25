"""
A pool of independent `KLULinSolveCache` workers, each holding its own
factorization of the same matrix.

KLU mutates per-numeric scratch and the `klu_l_common` `status` field during
`klu_solve`, so a single cache cannot be safely shared across threads.
The pool sidesteps that by allocating one full cache per worker. Acquisition
is gated on a `Channel{Int}`; pair acquire/release via [`with_worker`](@ref)
so the worker is always returned even on exception.

The matrix is factored `nworkers` times — pay this cost up front to gain
O(nworkers) parallel solves later.
"""
mutable struct KLULinSolvePool{Tv <: Union{Float64, ComplexF64}}
    workers::Vector{KLULinSolveCache{Tv}}
    available::Channel{Int}
end

Base.size(pool::KLULinSolvePool) = size(first(pool.workers))
Base.size(pool::KLULinSolvePool, d::Integer) = size(first(pool.workers), d)
Base.eltype(::Type{KLULinSolvePool{Tv}}) where {Tv} = Tv
Base.length(pool::KLULinSolvePool) = length(pool.workers)
nworkers(pool::KLULinSolvePool) = length(pool.workers)

"""
    KLULinSolvePool(A; nworkers=Threads.nthreads(), reuse_symbolic=true,
                       check_pattern=true) -> KLULinSolvePool

Factor `A` once per worker. Default `nworkers` matches `Threads.nthreads()`;
pass `1` for a serial pool that still exposes the `with_worker` API.
"""
function KLULinSolvePool(
    A::SparseMatrixCSC{Tv, Int};
    nworkers::Int = Threads.nthreads(),
    reuse_symbolic::Bool = true,
    check_pattern::Bool = true,
) where {Tv <: Union{Float64, ComplexF64}}
    nworkers >= 1 || throw(ArgumentError("nworkers must be >= 1; got $(nworkers)"))
    workers = Vector{KLULinSolveCache{Tv}}(undef, nworkers)
    for w in 1:nworkers
        workers[w] = klu_factorize(A;
            reuse_symbolic = reuse_symbolic, check_pattern = check_pattern)
    end
    available = Channel{Int}(nworkers)
    for w in 1:nworkers
        put!(available, w)
    end
    return KLULinSolvePool{Tv}(workers, available)
end

"""
    acquire!(pool) -> (cache, idx)

Block until a worker is available; return the worker cache and its index.
Pair every `acquire!` with `release!(pool, idx)` — prefer `with_worker`.
"""
function acquire!(pool::KLULinSolvePool)
    idx = take!(pool.available)
    return pool.workers[idx], idx
end

"""Return a worker, identified by its `idx`, to the pool."""
release!(pool::KLULinSolvePool, idx::Int) = put!(pool.available, idx)

"""
    with_worker(f, pool) -> result

Acquire a worker from `pool`, invoke `f(cache, idx)`, and release the worker
when `f` returns or throws.
"""
function with_worker(f, pool::KLULinSolvePool)
    cache, idx = acquire!(pool)
    try
        return f(cache, idx)
    finally
        release!(pool, idx)
    end
end

"""
    numeric_refactor!(pool, A)

Refresh the numeric factorization on every worker. Blocks until all workers
are idle.
"""
function numeric_refactor!(pool::KLULinSolvePool{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    n = nworkers(pool)
    drained = Vector{Int}(undef, n)
    for k in 1:n
        drained[k] = take!(pool.available)
    end
    try
        for idx in drained
            numeric_refactor!(pool.workers[idx], A)
        end
    finally
        for idx in drained
            put!(pool.available, idx)
        end
    end
    return pool
end

"""Free every worker's libklu handles. Idempotent."""
function Base.finalize(pool::KLULinSolvePool)
    for cache in pool.workers
        Base.finalize(cache)
    end
    return nothing
end
