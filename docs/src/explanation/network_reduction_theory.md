# Network Reduction Theory

This page explains the theory and mathematics behind network reduction. Power
systems can have thousands of buses and branches; reduction shrinks the network —
making large studies tractable and matrices faster to build and invert — while
preserving the power-flow relationships at the retained buses. Which
characteristics survive depends on the strategy ([`RadialReduction`](@ref),
[`DegreeTwoReduction`](@ref), [`WardReduction`](@ref)).

## The graph and susceptance structure reduction operates on

Network reduction is fundamentally a *graph* operation, and to understand why it
is well-defined it helps to see the three matrices that encode the network's
topology and electrical strength. These are the same building blocks the DC
sensitivity matrices are assembled from, and what every reduction manipulates
under the hood. (For constructors and accessors see the
[matrix type reference](../reference/matrix_types.md); the discussion here is about
*what they mean*.)

### The incidence matrix: pure topology

The [`IncidenceMatrix`](@ref) ``A`` is the oriented node–arc incidence matrix: one
row per arc, one column per bus, with a ``+1`` at the arc's *from* bus, a ``-1`` at
its *to* bus, and zeros elsewhere. It carries **topology only** — which bus
connects to which, and with what orientation — and nothing electrical. The
reference-bus column is dropped so the downstream susceptance matrix is
non-singular. Reading ``A`` column by column recovers each bus's degree, which is
exactly the quantity radial (degree 1) and degree-two reductions key off of.

### The BA matrix: topology weighted by electrical strength

The [`BA_Matrix`](@ref) is the product ``B A``, where ``B`` is the diagonal matrix
of branch susceptances (``b = 1/x`` under the DC approximation). Where ``A`` says
*which* buses a branch connects, ``BA`` scales each connection by *how electrically
strong* it is. Mapped onto bus angles it returns branch flows — the linear operator
behind ``P_{ij} = (\theta_i - \theta_j)/x_{ij}``.

### The ABA matrix: the grounded graph Laplacian

The [`ABA_Matrix`](@ref) is ``A^\top B A``, the reduced nodal susceptance matrix —
a **weighted graph Laplacian** with the reference bus grounded out. Solving
``ABA\,\theta = P`` *is* the DC power flow, and inverting it produces the dense
[`PTDF`](@ref)/[`LODF`](@ref) sensitivities. Because it is a Laplacian, eliminating
a bus is a Kron elimination on this matrix — which is precisely why degree-two
reduction (below) is exact rather than approximate.

### Why this makes reduction well-posed

Every reduction is defined on this susceptance graph and then propagates uniformly
to the downstream matrices (see also
[Computational considerations](computational_considerations.md), on why reductions
are applied to the [`Ybus`](@ref) first). A radial bus is a degree-1 node; a
degree-two bus a degree-2 node; Ward reduction Kron-eliminates the external
subgraph. Because all are operations on the incidence/susceptance structure, the
same reduction map applies to [`PTDF`](@ref), [`LODF`](@ref), and their virtual
variants without re-deriving anything per matrix.

### A subtlety: the susceptance graph is not the admittance graph

Connectivity and reduction do not always see the same network.
[`find_subnetworks`](@ref) walks the **admittance** graph of the [`Ybus`](@ref),
whereas `ABA` is built from the **susceptance** graph. These differ for branches
with ``r > 0`` and ``x = 0``: such a branch has finite admittance but zero
susceptance (``b = 1/x`` absent), so it appears in the [`Ybus`](@ref) graph but
*not* in [`BA_Matrix`](@ref). A network that is one connected island electrically
can therefore fragment into several components in the susceptance graph, leaving
blocks with no reference bus and a **singular `ABA`**. This is why zero-impedance
handling must resolve both endpoints of such a branch to a common node before
building the susceptance matrix.

## Radial branch reduction

A radial branch connects to a bus with only one connection to the rest of the
network — a "dead-end."

```
Main Network --- Bus A --- Bus B (radial)
                         \
                          Bus C (radial)
```

For a radial bus ``k`` connected to bus ``j``, the DC relation
``P_k \approx (\theta_j - \theta_k)/X_{jk}`` gives

```math
\theta_k = \theta_j - P_k X_{jk}.
```

The radial-bus angle is completely determined by its parent, so the bus can be
eliminated by transferring its load to the parent. Power flows and angles at the
core (non-radial) network are preserved exactly; only the eliminated locations lose
their explicit representation. Radial reduction is therefore for studies that
target the transmission backbone rather than distribution feeders.

## Degree-two (Kron) reduction

A degree-two bus connects exactly two others, acting as a pass-through:

```
Bus A --- Bus B (degree 2) --- Bus C
```

Kron reduction eliminates buses from the admittance matrix. Partitioning into
retained (``r``) and eliminated (``e``) buses with no injection at the eliminated
buses (``I_e = 0``):

```math
Y_{reduced} = Y_{rr} - Y_{re} Y_{ee}^{-1} Y_{er}.
```

For a single degree-two bus ``k`` between ``i`` and ``j`` (no shunt at ``k``) this
collapses to the series-impedance combination

```math
y_{ij}^{new} = y_{ij}^{old} + \frac{y_{ik}\,y_{kj}}{y_{ik} + y_{kj}}.
```

Equivalently, a pass-through bus with ``P_k = 0`` has an angle that is the
reactance-weighted average of its neighbors,
``\theta_k = (X_{kj}\theta_i + X_{ik}\theta_j)/(X_{ik} + X_{kj})``, so replacing it
with a direct equivalent branch preserves the flow relationship. Retained-bus flows
and overall impedances are preserved; the eliminated bus and its explicit branches
are replaced by one equivalent branch. Avoid it where the degree-two bus carries
measurements, controls, or a significant shunt.

## Combining reductions

Reductions can be applied in sequence, and order matters — each pass can expose new
candidates for the next. Apply [`RadialReduction`](@ref) before
[`DegreeTwoReduction`](@ref): removing dead-ends often exposes new degree-two buses.

Eliminating a bus with no injection is clean; a bus carrying load or generation has
those devices mapped to a retained bus, and shunt admittances affect the equivalent
admittance matrix. [`RadialReduction`](@ref) and [`DegreeTwoReduction`](@ref) accept
buses to protect from elimination (e.g. injector hosts). Reduction cannot preserve
detailed voltage profiles, local dynamics, or exact AC behavior at eliminated
locations — keep the full network for state estimation, protection coordination, or
local voltage studies.

## Further reading

  - [`RadialReduction`](@ref)
  - [`DegreeTwoReduction`](@ref)
