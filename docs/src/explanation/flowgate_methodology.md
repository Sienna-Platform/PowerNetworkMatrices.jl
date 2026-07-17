# Flowgate Methodology

This page explains how the [`VirtualMODF`](@ref) matrix in PowerNetworkMatrices can
be used to evaluate *flowgates* — post-contingency distribution factors of
monitored transmission elements. It describes the mathematics behind the
Woodbury-based computation and shows how to query distribution factors using
the current API.

For a hands-on walkthrough that maps every industry DFAX flavor (GSF, LSF,
LODF, OTDF, transfer DFAX, flowgate DFAX, and N-k DFAX) onto the matrices
this page describes, see the
[Reproduce industry DFAX values](../how_to_guides/generated_reproduce_dfax_values.md)
how-to guide.

## Background

A flowgate is a monitored transmission element paired with a contingency. The
central quantity of interest is the *distribution factor*: the fraction of a
source-to-sink transfer that appears as flow on the monitored element after
the contingency occurs.

In the DC power-flow model, this quantity can be expressed in closed form in
terms of the base-case [`PTDF`](@ref) and the [`LODF`](@ref). [`VirtualMODF`](@ref) generalizes that
relationship by computing the full post-contingency PTDF row directly, which
extends naturally to multi-element contingencies.

For the constructor signatures and the contingency/modification types named
below, see the [contingency & modification reference](../reference/contingencies.md);
for a task-oriented walkthrough of building and querying a `VirtualMODF`, see the
[contingencies how-to](../how_to_guides/generated_contingencies.md).

## Why a low-rank (Woodbury) update

A contingency changes the network by removing (or scaling) a handful of
branches. In DC terms, that perturbs the reduced susceptance matrix ``ABA`` in a
way that is **low rank**: outaging ``k`` branches is a rank-``k`` modification,
because each branch contributes a single susceptance term ``b_c\,a_c a_c^\top``
to ``ABA`` (with ``a_c`` the branch's incidence column). The post-contingency
susceptance matrix is therefore

```math
\widetilde{ABA} \; = \; ABA \; - \; U\,\Sigma\,U^\top,
```

where ``U`` collects the incidence columns of the outaged branches and
``\Sigma`` their susceptance changes — a matrix of rank ``k \ll N``.

The naïve way to get post-contingency sensitivities would be to rebuild and
re-factorize ``\widetilde{ABA}`` for *every* contingency, an ``O(N^3)``
factorization each time. The Woodbury (Sherman–Morrison–Woodbury) matrix
identity[^woodbury] avoids this entirely. It expresses the inverse of a low-rank update in
terms of the *already-computed* factorization of the base ``ABA`` plus the
solution of a small ``k \times k`` system:

```math
\widetilde{ABA}^{-1} \; = \; ABA^{-1}
  \; + \; ABA^{-1} U \bigl(\Sigma^{-1} - U^\top ABA^{-1} U\bigr)^{-1} U^\top ABA^{-1}.
```

The base ``ABA`` is factorized once at construction. Each contingency then costs
only a few solves against that stored factorization to form the Woodbury factors,
and the expensive ``O(N^3)`` work is never repeated. This is the core reason
post-contingency analysis of many contingencies is tractable at scale: the
dominant cost is paid once and reused, and the per-contingency cost scales with
the (small) number of outaged elements, not the network size.

The same structure is why there is **no dense `MODF` type** in the package. A
materialized post-contingency factor matrix would be one dense ``N_a \times N_b``
matrix *per contingency* — the product of two already-large dimensions with a
third — which is prohibitive for any realistic contingency list. `VirtualMODF`
instead keeps only the base factorization and the small Woodbury factors, and
materializes individual post-contingency rows on demand.

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
LODF expansion[^lodf]

```math
\mathrm{DF} \; = \; \mathrm{PTDF}[m, s] - \mathrm{PTDF}[m, k]
  + \mathrm{LODF}[m, c] \bigl(\mathrm{PTDF}[c, s] - \mathrm{PTDF}[c, k]\bigr),
```

but the Woodbury form generalizes to N-2 and higher-order contingencies
without additional derivation.

## Describing a contingency

[`VirtualMODF`](@ref) queries are keyed by a [`NetworkModification`](@ref) (or by a
[`ContingencySpec`](@ref) or a [`Outage`](@extref PowerSystems.Outage) that resolves
to one). A [`NetworkModification`](@ref) can be built in several ways:

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

When a [`VirtualMODF`](@ref) is constructed from a
[`System`](@extref PowerSystems.System), all
[`Outage`](@extref PowerSystems.Outage) supplemental attributes in the system are
automatically resolved and registered. Registered contingencies can be inspected
with [`get_registered_contingencies`](@ref) and queried directly by
[`Outage`](@extref PowerSystems.Outage).

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

[`VirtualMODF`](@ref) maintains two caches, both keyed by
[`NetworkModification`](@ref):

  - A Woodbury-factor cache (one entry per contingency, populated on first
    query for that contingency).
  - A per-contingency LRU row cache that stores the post-contingency rows
    produced for each monitored arc. The maximum cache size is controlled by
    the `max_cache_size` keyword (MiB per contingency).

The `tol` keyword of the `VirtualMODF` constructor
(`tol::Union{Float64, AutoTolerance}`, default `DEFAULT_AUTO_TOLERANCE`) controls
row-level sparsification: entries whose magnitude falls below the resolved cutoff
are dropped from the cached row. This reduces memory use and downstream
arithmetic cost when many rows are retained, at the expense of discarding small
distribution-factor contributions. Crucially, the cutoff is applied to the
*final* post-contingency row — after the exact Woodbury solve — so sparsification
never enters the correction itself, and the bound holds even when the Woodbury
update is severely ill-conditioned (a near-islanding contingency). Pass an
explicit `Float64` `tol` (e.g. `eps()`) when you need every entry retained; see
[Computational considerations](computational_considerations.md) for the
per-row [`AutoTolerance`](@ref) rule and its accuracy trade-offs.

[`clear_caches!`](@ref) drops the Woodbury and row caches but retains the
contingency registrations, so subsequent queries will simply recompute. Use
[`clear_all_caches!`](@ref) to also drop the registrations (after which the
[`VirtualMODF`](@ref) can no longer be queried).

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
  - MOD-030 flowgate screening[^mod030] (OTDF thresholding, AFC and ATC
    arithmetic, interconnection-wide congestion management procedures) is not
    provided by this package. [`VirtualMODF`](@ref) computes the distribution
    factors that such a layer would consume; the MOD-030 policy vocabulary is not
    part of the current API.

## References

[^woodbury]: The Sherman–Morrison–Woodbury identity; see G. H. Golub and
    C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins, 2013, §2.1.4,
    or [https://en.wikipedia.org/wiki/Woodbury_matrix_identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity).
[^mod030]: NERC Reliability Standard MOD-030, *Flowgate Methodology* (Available
    Flowgate Capability). [https://www.nerc.com](https://www.nerc.com)
[^lodf]: The line outage distribution factor and its expression in terms of the
    base-case PTDF are standard results; see A. J. Wood, B. F. Wollenberg, and
    G. B. Sheblé, *Power Generation, Operation, and Control*, 3rd ed., Wiley, 2013,
    and J. Guo, Y. Fu, Z. Li, and M. Shahidehpour, "Direct Calculation of Line
    Outage Distribution Factors," *IEEE Transactions on Power Systems*, vol. 24,
    no. 3, pp. 1633–1634, 2009.
