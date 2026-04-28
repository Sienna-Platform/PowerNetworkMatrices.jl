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
    # Per-worker factorization status. `valid[i] == false` flags a failed
    # factorization: worker `i`'s last refactor threw and its libklu numeric
    # handle is in an undefined state; the worker is held out of `available`
    # until `reset!` (or auto-reset) restores it. Mutations are guarded by
    # `state_lock`; reads from `acquire!` use the lock so that a concurrent
    # refactor cannot mix observed status with channel state.
    valid::Vector{Bool}
    state_lock::ReentrantLock
end

# Auto-reset is triggered when more than this fraction of workers fail to
# refactor (and the failure is not unanimous — the unanimous case is treated
# as a bad matrix and surfaces immediately without an auto-reset attempt).
const POOL_RESET_THRESHOLD = 0.5

Base.size(pool::KLULinSolvePool) = size(first(pool.workers))
Base.size(pool::KLULinSolvePool, d::Integer) = size(first(pool.workers), d)
Base.eltype(::Type{KLULinSolvePool{Tv}}) where {Tv} = Tv
Base.length(pool::KLULinSolvePool) = length(pool.workers)
nworkers(pool::KLULinSolvePool) = length(pool.workers)

"""Number of workers currently holding a valid factorization."""
function n_valid(pool::KLULinSolvePool)
    @lock pool.state_lock count(pool.valid)
end

"""
    KLULinSolvePool(A; nworkers=max(1, Threads.nthreads() - 1),
                       reuse_symbolic=true, check_pattern=true) -> KLULinSolvePool

Factor `A` once per worker. The default leaves one logical thread free for
the calling task and clamps to `1` when Julia is single-threaded; pass an
explicit `nworkers=Threads.nthreads()` to use every thread.
"""
function KLULinSolvePool(
    A::SparseMatrixCSC{Tv, Int};
    nworkers::Int = max(1, Threads.nthreads() - 1),
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
    pool = KLULinSolvePool{Tv}(workers, available, fill(true, nworkers),
        ReentrantLock())
    finalizer(Base.finalize, pool)
    return pool
end

"""
    acquire!(pool) -> (cache, idx)

Block until a worker is available; return the worker cache and its index.
Pair every `acquire!` with `release!(pool, idx)` — prefer `with_worker`.
Throws if every worker is in a failed-factorization state (`reset!` required).
"""
function acquire!(pool::KLULinSolvePool)
    @lock pool.state_lock begin
        any(pool.valid) || error(
            "KLULinSolvePool: all workers have a failed factorization. " *
            "Call `reset!(pool, A)` with a known-good matrix to recover.",
        )
    end
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

Refresh the numeric factorization on every currently-valid worker. Blocks
until all valid workers are idle (i.e., have been returned to the pool by
their holders), then dispatches by failure rate. Admin operations
(`numeric_refactor!` and `reset!`) must be serialized vs each other by the
caller. Failure rates:

- `0` failures: all workers refreshed, pool unchanged.
- `1 .. ⌊n/2⌋` failures (degraded mode): failed workers are held out of the
  channel; the pool keeps running with the surviving workers. A warning is
  emitted. The first underlying error is rethrown so callers can react if
  they want to.
- `> ⌊n/2⌋` and `< n` failures: triggers an auto-reset — `full_factor!` is
  invoked on every worker (including survivors) to restore a uniform state.
  If the reset succeeds, the pool is fully restored. If it does not, every
  worker that still fails keeps a failed factorization and the first reset
  error is thrown.
- `n` failures (every worker): the matrix is treated as fundamentally bad;
  no auto-reset is attempted (`full_factor!` would just fail again on the
  same matrix). All workers are flagged as failed factorizations, no workers
  are returned to the channel, and the first error is thrown. Recover by
  calling `reset!(pool, A_known_good)`.
"""
function numeric_refactor!(pool::KLULinSolvePool{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    n = nworkers(pool)
    @lock pool.state_lock begin
        any(pool.valid) || error(
            "KLULinSolvePool: all workers have a failed factorization; " *
            "call `reset!` first.",
        )
    end

    drained = _drain_available!(pool)
    failed_idxs = Int[]
    first_err = Ref{Any}(nothing)

    for idx in drained
        try
            numeric_refactor!(pool.workers[idx], A)
        catch e
            push!(failed_idxs, idx)
            first_err[] === nothing && (first_err[] = e)
        end
    end

    n_failed = length(failed_idxs)

    if n_failed == 0
        _return_all!(pool, drained, true)
        return pool
    end

    if n_failed == n
        # Matrix is bad; do not retry. Flag all as failed factorizations
        # and surface the error.
        _set_validity!(pool, drained, false)
        throw(first_err[])
    end

    if n_failed > POOL_RESET_THRESHOLD * n
        @warn "KLULinSolvePool: $(n_failed)/$(n) workers failed refactor; " *
              "triggering auto-reset"
        return _auto_reset!(pool, A, drained)
    end

    # Degraded: minority failed. Keep survivors in rotation; held-out failed
    # workers wait for `reset!` to recover.
    survivors = filter(idx -> idx ∉ failed_idxs, drained)
    _set_validity!(pool, failed_idxs, false)
    _set_validity!(pool, survivors, true)
    for idx in survivors
        put!(pool.available, idx)
    end
    @warn "KLULinSolvePool degraded: $(n_failed)/$(n) workers failed " *
          "refactor; pool operating with $(n - n_failed) workers"
    throw(first_err[])
end

"""
    reset!(pool, A) -> pool

Drop the prior factorization on every worker and rebuild from scratch via
`full_factor!(worker, A)` (free → analyze → factor). Blocks until every
currently-valid worker has been returned to the pool before touching any
factorization state, then refreshes every worker (including ones flagged
as failed). Use to recover a pool that has workers with a failed
factorization — including the case where `numeric_refactor!` threw because
every worker failed on a singular matrix. The caller is responsible for
passing a matrix `A` that is expected to factor cleanly; workers that
still fail after the reset keep a failed factorization. Admin operations
(`numeric_refactor!` and `reset!`) must be serialized vs each other by the
caller.
"""
function reset!(pool::KLULinSolvePool{Tv},
    A::SparseMatrixCSC{Tv, Int}) where {Tv}
    n = nworkers(pool)
    _drain_available!(pool)  # discards return value: we touch every worker

    failed_idxs = Int[]
    first_err = Ref{Any}(nothing)

    for idx in 1:n
        try
            full_factor!(pool.workers[idx], A)
        catch e
            push!(failed_idxs, idx)
            first_err[] === nothing && (first_err[] = e)
        end
    end

    survivors = filter(idx -> idx ∉ failed_idxs, 1:n)
    _set_validity!(pool, failed_idxs, false)
    _set_validity!(pool, survivors, true)
    for idx in survivors
        put!(pool.available, idx)
    end

    if length(failed_idxs) == n
        error(
            "KLULinSolvePool reset failed: every worker still has a failed " *
            "factorization. Underlying error: $(first_err[])",
        )
    elseif !isempty(failed_idxs)
        @warn "KLULinSolvePool reset partially recovered: " *
              "$(length(failed_idxs))/$(n) workers still have a failed " *
              "factorization"
    end
    return pool
end

# --- internal helpers ---

# Drain every currently-valid worker from `available`, blocking on each
# `take!` until the worker has been returned by its holder. Failed workers
# (`pool.valid[i] == false`) are intentionally held out of the channel and
# are already in the pool's exclusive ownership, so they are not part of
# the drain count. The `pool.valid` snapshot is taken under `state_lock`
# for consistency; admin operations (`numeric_refactor!`, `reset!`) must
# still be serialized vs each other by the caller, but solves running
# through `with_worker` no longer race against the refactor — the drain
# waits for them.
function _drain_available!(pool::KLULinSolvePool)
    n_to_drain = @lock pool.state_lock count(pool.valid)
    drained = Vector{Int}(undef, n_to_drain)
    for i in 1:n_to_drain
        drained[i] = take!(pool.available)
    end
    return drained
end

function _set_validity!(pool::KLULinSolvePool, idxs, value::Bool)
    @lock pool.state_lock begin
        for idx in idxs
            pool.valid[idx] = value
        end
    end
    return nothing
end

function _return_all!(pool::KLULinSolvePool, idxs, valid::Bool)
    _set_validity!(pool, idxs, valid)
    if valid
        for idx in idxs
            put!(pool.available, idx)
        end
    end
    return nothing
end

# Re-run `full_factor!` on every drained worker. Returns the pool on full
# success; throws on partial or total failure (after returning whatever
# survivors are still valid to the channel).
function _auto_reset!(pool::KLULinSolvePool{Tv},
    A::SparseMatrixCSC{Tv, Int},
    drained::Vector{Int}) where {Tv}
    reset_failed = Int[]
    reset_err = Ref{Any}(nothing)
    for idx in drained
        try
            full_factor!(pool.workers[idx], A)
        catch e
            push!(reset_failed, idx)
            reset_err[] === nothing && (reset_err[] = e)
        end
    end

    survivors = filter(idx -> idx ∉ reset_failed, drained)
    _set_validity!(pool, reset_failed, false)
    _set_validity!(pool, survivors, true)
    for idx in survivors
        put!(pool.available, idx)
    end

    if !isempty(reset_failed)
        # Auto-reset itself failed for some workers. Surface the reset error
        # rather than the original refactor error; reset state is what the
        # caller now needs to react to.
        throw(reset_err[])
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
