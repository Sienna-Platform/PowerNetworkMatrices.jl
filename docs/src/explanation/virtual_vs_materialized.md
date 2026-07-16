# Virtual vs. materialized matrices

The sensitivity matrices [`PTDF`](@ref) and [`LODF`](@ref) come in two shapes: a
**materialized** (dense) form that computes and stores the whole matrix up front,
and a **virtual** form ([`VirtualPTDF`](@ref), [`VirtualLODF`](@ref),
[`VirtualMODF`](@ref)) that computes rows on demand and caches them. This page
explains the trade-off the two forms make and how to decide between them.

## The underlying tension

A PTDF is `Nₐ × N_b` (arcs by buses) and a LODF is `Nₐ × Nₐ`; both are
mathematically dense — `ABA⁻¹` fills in even though `ABA` is sparse (see
[Computational Considerations](computational_considerations.md)). For a large
grid the materialized matrix is tens of thousands of rows, each with one entry
per bus. Holding all of it costs memory quadratic in system size, and building
all of it costs one solve per row whether or not you ever read that row.

Most studies do not need the whole matrix. They query a handful of monitored
branches, one contingency at a time, incrementally. That mismatch — a dense
object built in full to answer sparse, incremental questions — is what the
virtual matrices exist to resolve. They **trade compute for memory**: nothing is
precomputed, and a row is materialized only when it is first indexed.

## How the virtual form works

A virtual matrix keeps the factorized `ABA` (and the arc/bus axes) but no matrix
entries. On the first `getindex` of a row, it runs the per-arc solve, optionally
sparsifies the result, and stores it in a bounded [row cache](#The-row-cache).
Later reads of the same row hit the cache and return immediately; reads of a
row that has since been evicted pay the solve again.

Because rows are produced independently and on demand, sparsification is applied
**per row**. At or above `AUTO_TOLERANCE_BUS_LIMIT` (2000) buses the default
[`AutoTolerance`](@ref) drops each row's negligible entries relative to that
row's own peak, so a cached row of a large system is stored *sparsely* — this is
what keeps a single column of a large case from costing one dense vector per
bus. Below that bus count `AutoTolerance` is a no-op and rows are returned
exactly. (The dense constructors never sparsify under `AutoTolerance`; pass a
`Float64` `tol` to sparsify them explicitly.)

### The row cache

The cache is a `RowCache` (`src/row_cache.jl`): a dictionary of stored rows plus
an LRU (least-recently-used) eviction policy. It is bounded by two limits derived
from the `max_cache_size` construction kwarg (default `MAX_CACHE_SIZE_MiB`,
100 MiB): a byte budget and a maximum row count. When inserting a new row would
exceed the budget, the least-recently-used non-persistent row is purged. A
working set larger than the cache therefore thrashes — rows are recomputed
because they were evicted before being reused — so the cache size is the knob
that buys back speed with memory.

**Persistent rows.** The virtual constructors accept a `persistent_arcs` kwarg
(a vector of arc tuples). Those arcs' rows are pinned in the cache and never
evicted, so a small set of hot monitored branches that you query repeatedly stays
resident regardless of how many other rows churn through the LRU set. Pinned rows
count against the cache budget; the constructor errors if the pinned set alone
would exceed `max_cache_size`.

Caches are cleared with [`clear_caches!`](@ref) / [`clear_all_caches!`](@ref).
For the mechanics of sizing, pinning, and clearing, see the
[cache how-to guide](../how_to_guides/generated_virtual_cache.md).

## When each form wins

**Prefer the materialized `PTDF` / `LODF` when:**

  - The system is small or moderate, so the dense matrix fits in memory comfortably.
  - You will read most of the entries, or read entries in an unpredictable random
    pattern — amortizing one full build over many reads beats repeated on-demand
    solves.
  - You want to **serialize** the result. Only the dense [`PTDF`](@ref) can be
    written to and read from HDF5 (see the
    [serialization reference](../reference/serialization.md)). Virtual matrices are
    **never serialized** — they hold a factorization and a volatile cache, not a
    stored matrix, so there is nothing meaningful to persist. A deserialized PTDF
    is a fully materialized matrix.

**Prefer the virtual form when:**

  - The system is large — especially past `AUTO_TOLERANCE_BUS_LIMIT` — and the
    dense matrix would not fit, or would waste memory you need elsewhere.
  - You only touch a subset of rows (a set of monitored branches, a few
    contingencies) and query them incrementally. This is the common
    security-analysis pattern, and it is exactly what the row cache is tuned for.
  - Memory is the binding constraint and you can afford to recompute cold rows.

For contingency analysis there is no materialized alternative: only
[`VirtualMODF`](@ref) exists — there is no dense `MODF` type. Post-contingency
distribution factors are always computed on demand.

## Summary of the trade-off

|                        | Materialized `PTDF`/`LODF` | Virtual `PTDF`/`LODF`/`MODF`   |
|:---------------------- |:-------------------------- |:------------------------------ |
| Build cost             | All rows up front          | Per row, on first access       |
| Memory                 | Full dense matrix          | Bounded cache (LRU)            |
| Repeated random access | Fast (already stored)      | Fast if cached, else recompute |
| Sparsification         | Exact unless `Float64` tol | Per row above the bus limit    |
| Serializable           | Yes (PTDF, HDF5)           | No                             |

See the [matrix type reference](../reference/matrix_types.md) for the
constructors and keyword arguments of each type.
