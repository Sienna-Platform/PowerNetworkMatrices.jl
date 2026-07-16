# Concurrency and the KLU lock

The virtual matrices are designed to be **thread-safe to read**, but reading them
concurrently does **not** make them faster. This page explains why the solver
work serializes, what that means for multithreaded code, and why the per-arc
solve loop cannot be sped up by threading.

## Concurrent `getindex` is safe but serialized

Indexing a [`VirtualPTDF`](@ref), [`VirtualLODF`](@ref), or [`VirtualMODF`](@ref)
from several threads produces correct results with no data races. Two locking
layers guarantee it:

  - A **per-cache `ReentrantLock`** on each virtual matrix guards its row cache and
    its solver scratch buffers. Cache lookups and inserts, and the factorization
    solve itself, happen under this lock.
  - A **process-wide `_LIBKLU_LOCK`** (`src/KLUWrapper/KLUWrapper.jl`) wraps *every*
    libklu call in the entire process, across all matrices and all threads.

The consequence is that the expensive part of a row computation — the KLU solve —
runs one at a time. Concurrent readers get correct answers, but they queue: the
throughput of `N` threads all missing the cache is essentially the throughput of
one. **Do not expect a parallel speedup from threading factor solves.** The value
of the locking is safety and correctness under concurrency, not scaling.

The cache-miss path is written to hold locks as briefly as correctness allows.
The shared `cached_row_lookup` pattern (`src/row_cache.jl`) takes the cache lock
to test for a hit, runs the row computation, then takes the lock again to insert,
double-checking for a row a concurrent producer may have inserted in the
meantime. The compute itself is still serialized — through the per-cache solver
lock and `_LIBKLU_LOCK` — because that is where the libklu work lives.

## Why libklu must be serialized

`_LIBKLU_LOCK` is not conservative caution; it reflects a measured property of
the library. A design with a **pool of independent KLU caches** — distinct
`Numeric`, `Symbolic`, and `Common` objects per thread, so that each thread could
in principle solve without touching another's state — was implemented and then
removed. Empirically, per-thread objects did **not** prevent libklu state
corruption: distinct handles still interfered, producing intermittent wrong
results. The global lock is what makes concurrent use correct, so it stays, and
the per-thread pool was dropped because it added complexity with no throughput
benefit once the global lock was required anyway. What remains is one factor and
one cache per virtual matrix.

The Apple Accelerate backend (`src/AccelerateWrapper/`) has no documented
cross-handle corruption issue analogous to libklu's, so its solves are guarded by
the per-cache `solver_lock` alone, without a process-wide lock. It is still
serialized per matrix, for the same buffer-safety reason.

## Why the per-arc solve loop is inherently serial

Building a PTDF or answering a set of contingency queries means one linear solve
per arc (or per contingency) against the factorized `ABA`. It is tempting to
parallelize that loop. It does not work, for two independent reasons:

 1. **KLU cannot do concurrent solves**, even given per-thread workspaces. The
    library serializes internally (this is the same property that forced
    `_LIBKLU_LOCK`), so handing each thread its own scratch buffers does not let
    the solves overlap — they still queue in the library.
 2. **The query pattern is incremental, not batched.** Sienna consumers ask for
    rows one arc or one contingency at a time, as a study progresses. There is no
    point at which a large batch of right-hand sides is available to solve
    together, so a multi-RHS reformulation — the usual way to extract parallelism
    from a factorized system — does not fit the access pattern.

Because both the library and the workload resist it, threading the solve loop is
not a lever this package offers. The realized performance work went into making
each individual build cheaper (for example, the Ybus adjacency assembly), not
into running solves in parallel.

## Practical guidance

  - **Reading virtual matrices from multiple threads is safe.** Correctness is
    guaranteed; use it when a thread happens to need a row.
  - **Do not thread hoping for solver speedup.** The solves serialize on
    `_LIBKLU_LOCK` (KLU) or the per-cache lock (Accelerate); more threads means more
    queueing, not more throughput.
  - **Independent, non-solver work can still overlap.** The locks cover libklu
    calls and each cache's buffers, not your surrounding logic. Parallelism is
    worth pursuing above the solve, in how you organize a study, rather than inside
    it.

For the solver backends themselves, see the
[tolerance and solver reference](../reference/tolerance_and_solvers.md) and the
[choose-a-linear-solver how-to](../how_to_guides/generated_choose_linear_solver.md).
