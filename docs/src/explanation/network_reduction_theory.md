# Network Reduction Theory

This document explains the theory and mathematics behind network reduction techniques in power systems.

## Why Network Reduction?

Power system networks can be very large, with thousands of buses and branches. Network reduction techniques simplify these networks while preserving essential characteristics for analysis.

### Benefits of Reduction:

 1. **Computational Efficiency**: Smaller matrices are faster to compute and invert
 2. **Focus**: Reduces complexity to focus on areas of interest
 3. **Scalability**: Makes large-scale studies tractable
 4. **Clarity**: Simplifies visualization and understanding

### Preservation Goals:

Network reduction aims to preserve:

  - Power flow relationships at retained buses
  - Impedance relationships between key points
  - Stability characteristics (when applicable)
  - Essential topology features

The type of reduction (e.g. `RadialReduction`, `DegreeTwoReduction`, `WardReduction`) determines the extent to which these characteristics are retained

## The graph and susceptance structure reduction operates on

Network reduction is fundamentally a *graph* operation, and to understand why
it is well-defined it helps to see the three matrices that encode the network's
topology and electrical strength. These are the same building blocks the DC
sensitivity matrices are assembled from, and they are what every reduction
manipulates under the hood. (For constructor signatures and field accessors,
see the [matrix type reference](../reference/matrix_types.md); the discussion
here is about *what they mean*.)

### The incidence matrix: pure topology

The `IncidenceMatrix` ``A`` is the oriented node–arc incidence matrix of the
network graph. It has one row per arc and one column per bus, with a ``+1`` at
the arc's *from* bus, a ``-1`` at its *to* bus, and zeros elsewhere. It carries
**topology only** — which bus connects to which, and with what orientation — and
nothing about the electrical parameters. The reference bus column is dropped so
that the downstream susceptance matrix is non-singular. Reading ``A`` row by row
recovers the graph; reading it column by column recovers each bus's degree,
which is exactly the quantity radial (degree 1) and degree-two reductions key
off of.

### The BA matrix: topology weighted by electrical strength

The `BA_Matrix` is the product ``B A``, where ``B`` is the diagonal matrix of
branch susceptances (under the DC approximation, ``b = 1/x`` for each branch).
Where ``A`` says *which* buses a branch connects, ``BA`` scales each connection
by *how electrically strong* it is: a low-reactance branch couples its endpoints
tightly, a high-reactance branch loosely. Mapped onto bus angles, ``BA`` returns
branch flows — it is the linear operator behind ``P_{ij} = (\theta_i - \theta_j)/x_{ij}``. (For computational reasons the raw data is stored transposed,
but the axes still read arcs × buses.)

### The ABA matrix: the grounded graph Laplacian

The `ABA_Matrix` is ``A^\top B A``, the reduced nodal susceptance matrix. This
is a **weighted graph Laplacian** with the reference bus grounded out: its
off-diagonal entry ``(i,j)`` is the negative susceptance of the branch between
``i`` and ``j``, and its diagonal is the sum of susceptances incident to each
bus. Solving ``ABA\,\theta = P`` *is* the DC power flow, and inverting it is
what produces the dense `PTDF`/`LODF` sensitivities. Because it is a Laplacian,
its structure mirrors the graph directly: eliminating a bus from the network is
a Kron elimination on this matrix, which is precisely why degree-two reduction
(below) is exact rather than approximate.

### Why this makes reduction well-posed

Every reduction is defined on this susceptance graph and then propagates
uniformly to the downstream matrices (see also
[Computational considerations](computational_considerations.md), which explains
why reductions are applied to the `Ybus` first). A radial bus is a degree-1 node
in the graph; a degree-two bus is a degree-2 node; Ward reduction partitions the
graph into study and external subgraphs and Kron-eliminates the external one.
Because all of these are operations on the incidence/susceptance structure, the
same reduction map applies to `PTDF`, `LODF`, and their virtual variants without
re-deriving anything per matrix.

### A subtlety: the susceptance graph is not the admittance graph

Connectivity and reduction do not always see the same network. `find_subnetworks`
walks the **admittance** graph of the `Ybus`, whereas `ABA` is built from the
**susceptance** graph. These differ for branches with ``r > 0`` and ``x = 0``:
such a branch has a finite admittance but zero susceptance (``b = 1/x`` is
undefined / treated as absent), so it appears in the `Ybus` graph but *not* in
`BA`. A network that is a single connected island electrically can therefore
fragment into several disconnected components in the susceptance graph, leaving
blocks with no reference bus and a **singular `ABA`**. This is why zero-impedance
handling must resolve both endpoints of such branches to a common node before
building the susceptance matrix — a reduction that ignored it would silently
produce a singular DC power flow.

## Radial Branch Reduction

### What is a Radial Branch?

A radial branch is one that connects to a bus that has only one connection to the rest of the network. Think of it as a "dead-end" in the network.

```
Main Network --- Bus A --- Bus B (radial)
                         \
                          Bus C (radial)
```

In this example, buses B and C are radial - they each connect to only one other bus.

### Mathematical Basis

Radial buses can be reduced because power flow to/from them is uniquely determined by the connection bus. The reduction involves:

 1. **Identifying radial buses**: Buses with degree = 1
 2. **Transferring loads**: Moving load/generation to the connection point
 3. **Removing the branch**: Eliminating the radial connection

### Why Radial Reduction Works

For a radial bus $k$ connected to bus $j$:

$$P_k = \frac{V_j V_k}{X_{jk}} \sin(\theta_j - \theta_k)$$

Under DC approximation with $V_j \approx V_k \approx 1$:

$$P_k \approx \frac{\theta_j - \theta_k}{X_{jk}}$$

Since $P_k$ is determined by the load at bus $k$, the angle $\theta_k$ is uniquely determined:

$$\theta_k = \theta_j - P_k X_{jk}$$

The radial bus angle is completely determined by its parent bus, so it can be eliminated by transferring the load.

### Implications of Radial Reduction

**Preserved:**

  - Power flow at non-radial buses
  - Voltage angles at non-radial buses
  - System topology at the core network

**Changed:**

  - Number of buses and branches (reduced)
  - Detailed behavior at radial locations
  - Local voltage profiles

**When to Use:**

  - Radial connections are not of interest for analysis
  - Focusing on transmission backbone
  - Computational efficiency is important

**When to Avoid:**

  - Studying distribution feeders (often radial)
  - Detailed voltage analysis needed
  - Radial buses have critical measurements

## Degree Two Reduction (Kron Reduction)

### What is a Degree Two Bus?

A degree two bus connects exactly two other buses, acting as a "pass-through" point.

```
Bus A --- Bus B (degree 2) --- Bus C
```

Bus B has degree 2 - it connects A and C but has no other connections.

### Mathematical Basis: Kron Reduction

Kron reduction is a systematic method to eliminate buses from the admittance matrix. Degree two reduction is a simple case of Kron reduction for eliminating degree two nodes.

Given admittance matrix:

$$\begin{bmatrix} I_r \\ I_e \end{bmatrix} = \begin{bmatrix} Y_{rr} & Y_{re} \\ Y_{er} & Y_{ee} \end{bmatrix} \begin{bmatrix} V_r \\ V_e \end{bmatrix}$$

where subscript $r$ denotes retained buses and $e$ denotes eliminated buses.

If $I_e = 0$ (no injection at eliminated buses):

$$Y_{ee} V_e = -Y_{er} V_r$$
$$V_e = -Y_{ee}^{-1} Y_{er} V_r$$

Substituting back:

$$I_r = (Y_{rr} - Y_{re} Y_{ee}^{-1} Y_{er}) V_r = Y_{reduced} V_r$$

The reduced admittance matrix is:

$$Y_{reduced} = Y_{rr} - Y_{re} Y_{ee}^{-1} Y_{er}$$

### For a Single Degree Two Bus

Consider bus $k$ connecting buses $i$ and $j$:

```
Bus i ----(y_ik)---- Bus k ----(y_kj)---- Bus j
```

The equivalent admittance directly between $i$ and $j$ after eliminating $k$ is:

$$y_{ij}^{new} = y_{ij}^{old} + \frac{y_{ik} \cdot y_{kj}}{y_{ik} + y_{kj} + y_{kk}}$$

If there's no shunt at bus $k$ ($y_{kk} = 0$):

$$y_{ij}^{new} = y_{ij}^{old} + \frac{y_{ik} \cdot y_{kj}}{y_{ik} + y_{kj}}$$

This is analogous to combining series impedances in circuit theory.

### Physical Interpretation

Eliminating a degree two bus:

  - Combines the series impedances of the two connecting branches
  - Creates an equivalent direct connection
  - Preserves the overall impedance between endpoints

### Why Degree Two Reduction Works

The key insight is that a degree two bus with no injection ($P_k = 0$) serves only to pass power through. Its voltage angle is determined by:

$$P_k = \frac{\theta_i - \theta_k}{X_{ik}} + \frac{\theta_j - \theta_k}{X_{kj}} = 0$$

Solving for $\theta_k$:

$$\theta_k = \frac{X_{kj} \theta_i + X_{ik} \theta_j}{X_{ik} + X_{kj}}$$

The angle is a weighted average of its neighbors, so eliminating it and creating a direct equivalent connection preserves the flow relationship.

### Implications of Degree Two Reduction

**Preserved:**

  - Power flows at retained buses
  - Voltage angles at retained buses
  - Overall impedance relationships
  - Equivalence for power flow studies

**Changed:**

  - Number of buses (reduced)
  - Detailed behavior at eliminated bus
  - Branch topology (new equivalent branches created)

**When to Use:**

  - Pass-through buses are not of analytical interest
  - Focusing on injection/load buses
  - Simplifying large transmission systems
  - Reducing computational burden

**When to Avoid:**

  - The degree two bus has measurements or controls
  - Detailed branch flows needed at that location
  - Bus has significant shunt elements
  - Studying protection or relay settings

## Combining Reductions

Multiple reductions can be applied sequentially. In these cases, the order of the reduction matters as after each reduction, new candidate buses may appear. In general, it is recommended that radial reduction is applied before degree two reduction as applying radial reduction first may expose new degree two buses.

## Practical Considerations

### Load and Generation at Reduced Buses

Eliminating buses without any connected injection components can be done cleanly. For eliminated buses that have load or generation attached, those devices must be mapped to retained buses. For buses with shunt admittances, this mapping process can affect the equivalent admittance matrix. `RadialReduction` and `DegreeTwoReduction` have options for specifying buses that should not be eliminated (e.g. due to the presence of connected injectors).

### Validation

Always verify reductions:

  - Compare power flows before and after
  - Check that key quantities are preserved
  - Validate on a small test system first

## Limitations

### What Reduction Cannot Do:

  - Preserve detailed voltage profiles everywhere
  - Capture local dynamics at eliminated buses
  - Represent phenomena requiring those buses (e.g., local stability)
  - Maintain exact AC power flow at eliminated locations

### When Full Network is Required:

  - Detailed state estimation
  - Protection coordination
  - Local voltage studies
  - Distributed generation integration at eliminated buses

## Further Reading

For practical application, see:

  - [RadialReduction](@ref)
  - [DegreeTwoReduction](@ref)
