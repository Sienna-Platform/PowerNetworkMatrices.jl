# Flowgate Methodology

This page explains how the `VirtualMODF` matrix in PowerNetworkMatrices can be
used to evaluate *flowgates* — post-contingency distribution factors of
monitored transmission elements. It describes the mathematics behind the
Woodbury-based computation and shows how to query distribution factors using
the current API.

## Background

A flowgate is a monitored transmission element paired with a contingency. The
central quantity of interest is the *distribution factor*: the fraction of a
source-to-sink transfer that appears as flow on the monitored element after
the contingency occurs.

In the DC power-flow model, this quantity can be expressed in closed form in
terms of the base-case PTDF and the LODF. `VirtualMODF` generalizes that
relationship by computing the full post-contingency PTDF row directly, which
extends naturally to multi-element contingencies.

## How `VirtualMODF` computes post-contingency PTDF rows

Given a base-case PTDF and a contingency described by a `NetworkModification`,
`VirtualMODF` returns the post-contingency PTDF row for any monitored arc
using the Woodbury matrix identity:

```math
\mathrm{post\_ptdf}[m, :] \; = \; \mathrm{base\_ptdf}[m, :]
  \; + \; \text{Woodbury correction}(m, \text{modification}).
```

The Woodbury factors depend only on the contingency (which arcs are outaged
and by how much) and not on the monitored element. They are computed once per
contingency and cached in the `VirtualMODF`. Each subsequent monitored-arc
query requires one additional KLU solve against the factorized `ABA` matrix.

The distribution factor of a transfer path from source bus ``s`` to sink bus
``k`` through a monitored arc ``m`` under contingency ``c`` is the difference
of two entries of the post-contingency row:

```math
\mathrm{DF} \; = \; \mathrm{post\_ptdf}[m, s] \; - \; \mathrm{post\_ptdf}[m, k].
```

For a single-element (N-1) contingency this is equivalent to the explicit
LODF expansion

```math
\mathrm{DF} \; = \; \mathrm{PTDF}[m, s] - \mathrm{PTDF}[m, k]
  + \mathrm{LODF}[m, c] \bigl(\mathrm{PTDF}[c, s] - \mathrm{PTDF}[c, k]\bigr),
```

but the Woodbury form generalizes to N-2 and higher-order contingencies
without additional derivation.

## Describing a contingency

`VirtualMODF` queries are keyed by a `NetworkModification` (or by a
`ContingencySpec` or a `PSY.Outage` that resolves to one). A
`NetworkModification` can be built in several ways:

```julia
using PowerSystems
using PowerNetworkMatrices

# Outage of a single arc identified by a (from_bus, to_bus) tuple
mod_arc = NetworkModification(vmodf, (1, 2))

# Outage of a specific branch component
branch = get_component(Line, sys, "Line-1-2")
mod_branch = NetworkModification(vmodf, branch)

# Contingency resolved from a PowerSystems Outage supplemental attribute
mod_outage = NetworkModification(vmodf, sys, outage)
```

When a `VirtualMODF` is constructed from a `PSY.System`, all `PSY.Outage`
supplemental attributes in the system are automatically resolved and
registered. Registered contingencies can be inspected with
`get_registered_contingencies(vmodf)` and queried directly by `PSY.Outage`.

## Querying post-contingency rows

The post-contingency PTDF row for a monitored arc is obtained by indexing the
`VirtualMODF`:

```julia
# Build a VirtualMODF (auto-registers outage attributes in the system)
vmodf = VirtualMODF(sys)

# Post-contingency row for monitored arc (1, 4) under a NetworkModification
row = vmodf[(1, 4), mod_arc]

# Same query keyed by a registered ContingencySpec
ctg = first(values(get_registered_contingencies(vmodf)))
row = vmodf[(1, 4), ctg]

# Same query keyed by a PSY.Outage attribute
row = vmodf[(1, 4), outage]
```

Monitored arcs can be passed as either an arc tuple `(from_bus, to_bus)` or
an integer index.

Each returned row is a `Vector{Float64}` of length equal to the number of
buses in the (possibly reduced) network. The distribution factor for a
source/sink path is obtained by subtracting two of its entries:

```julia
bus_lookup = get_bus_lookup(vmodf)
df = row[bus_lookup[source_bus]] - row[bus_lookup[sink_bus]]
```

## Caching and sparsification

`VirtualMODF` maintains two caches, both keyed by `NetworkModification`:

  - A Woodbury-factor cache (one entry per contingency, populated on first
    query for that contingency).
  - A per-contingency LRU row cache that stores the post-contingency rows
    produced for each monitored arc. The maximum cache size is controlled by
    the `max_cache_size` keyword (MiB per contingency).

The `tol` keyword of the `VirtualMODF` constructor enables row-level
sparsification: entries whose magnitude is below `tol` are dropped from the
cached row. This reduces memory use and downstream arithmetic cost when many
rows are retained, at the expense of discarding small distribution-factor
contributions. The default `tol = eps()` keeps all entries.

`clear_caches!(vmodf)` drops the Woodbury and row caches but retains the
contingency registrations, so subsequent queries will simply recompute. Use
`clear_all_caches!(vmodf)` to also drop the registrations (after which the
`VirtualMODF` can no longer be queried).

## Relationship to other matrix types

| Matrix type           | Role in post-contingency analysis                                               |
|:--------------------- |:------------------------------------------------------------------------------- |
| `PTDF`                | Base-case sensitivities                                                         |
| `VirtualLODF`         | Single-element line outage distribution factors (N-1)                           |
| `VirtualMODF`         | Post-contingency PTDF rows via Woodbury; supports N-1 and multi-element outages |
| `NetworkModification` | Contingency specification; keys the Woodbury and row caches in `VirtualMODF`    |
| `ContingencySpec`     | Pairs a `PSY.Outage` UUID with its resolved `NetworkModification`               |

## Limitations

  - The implementation assumes DC power flow (lossless, linearized). Voltage
    and stability limits that define some flowgate transfer capabilities must
    be handled externally.
  - MOD-030 flowgate screening (OTDF thresholding, AFC and ATC arithmetic,
    interconnection-wide congestion management procedures) is not provided by
    this package. `VirtualMODF` computes the distribution factors that such a
    layer would consume; the MOD-030 policy vocabulary is not part of the
    current API.
