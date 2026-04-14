## Core Matrix Types

### Incidence Matrix ([`IncidenceMatrix`](@ref))

The incidence matrix $A$ represents the bus-arc connectivity of the network, showing which arcs connect to which buses. It's a fundamental building block for other matrices.

**Properties:**

  - Rectangular matrix (arcs × buses)
  - Values indicate connection direction (+1 for "from" bus, -1 for "to" bus, or 0)

**Purpose:**
The incidence matrix captures pure network topology without electrical parameters, making it useful for graph-theoretic analysis.

### Adjacency Matrix ([`AdjacencyMatrix`](@ref))

The incidence matrix $A$ represents the directed connectivity between buses, showing which buses are connected to eachother. It's a fundamental building block for other matrices.

**Properties:**

  - Square matrix (arcs × buses)
  - Values indicate connection direction (+1 for "from" bus, -1 for "to" bus, or 0)
  - Sparsity pattern matches the bus admittance matrix.

**Purpose:**
The adjaency matrix captures pure network topology without electrical parameters, making it useful for graph-theoretic analysis.

### Bus Admittance Matrix ([`Ybus`](@ref))

The bus admittance matrix combines network topology with electrical parameters (impedance/admittance).

**Properties:**

  - Square matrix (buses × buses)
  - Diagonal elements are self-admittances
  - Off-diagonal elements are mutual admittances
  - Symmetric for passive networks

**Purpose:**
Forms the basis for power flow calculations and relates bus voltages to injected currents.

**Note:**
The [`ArcAdmittanceMatrix`](@ref) is a rectangular matrix (arcs x buses) consisting of the off diagonal entries of the [`Ybus`](@ref). These matrices are useful when computing line flows from bus voltages and can be optionally created when building the [`Ybus`](@ref).

### BA Matrix ([`BA_Matrix`](@ref))

The BA matrix represents the arc-bus incidence matrix weighted by arc susceptances.

**Purpose:**
Serves as an intermediate calculation in deriving ABA, PTDF and LODF matrices.

### ABA Matrix ([`ABA_Matrix`](@ref))

The ABA matrix is derived from $A \cdot B \cdot A^T$ where $A$ is the incidence matrix and $B$ contains branch admittances.

**Purpose:**
Serves as an intermediate calculation in deriving PTDF and LODF matrices.

## Sensitivity Matrices

### Power Transfer Distribution Factors ([`PTDF`](@ref))

PTDF matrices answer the question: "If I inject 1 MW at bus $i$ and withdraw 1 MW at bus $j$, how much does the flow on branch $k$ change?"

**Key Characteristics:**

  - Linearized approximation of power flow
  - Valid for small perturbations around operating point
  - Fast to compute and evaluate
  - Widely used in market operations and security analysis

**Note:**
See [`VirtualPTDF`](@ref) for cases where it is not possible to compute or store the full PTDF matrix.

### Line Outage Distribution Factors ([`LODF`](@ref))

LODF matrices answer: "If branch $m$ fails, how much does the flow redistribute to branch $k$?"

**Key Characteristics:**

  - Predicts post-contingency flows
  - Essential for N-1 security analysis
  - Computed from PTDF matrix
  - Helps identify critical lines

**Mathematical Relationship:**
LODF is derived from PTDF using matrix operations that model the effect of removing a branch from the network.

See [`VirtualLODF`](@ref) for cases where it is not possible to compute or store the full LODF matrix.
