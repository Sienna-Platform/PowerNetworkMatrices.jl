## Core Matrix Types

### Incidence Matrix ([`IncidenceMatrix`](@ref))

The incidence matrix $A$ represents the bus-arc connectivity of the network, showing which arcs connect to which buses. It's a fundamental building block for other matrices.

**Properties:**

  - Rectangular matrix (arcs × buses)
  - Values indicate connection direction (+1 for "from" bus, -1 for "to" bus, or 0)
  - Rows are indexed by arc tuples `(from_bus_number, to_bus_number)`
  - Columns are indexed by bus numbers

**Purpose:**
The incidence matrix captures pure network topology without electrical parameters, making it useful for graph-theoretic analysis.

### Adjacency Matrix ([`AdjacencyMatrix`](@ref))

The adjacency matrix represents the directed connectivity between buses, showing which buses are connected to each other. It's a fundamental building block for other matrices.

**Properties:**

  - Square matrix (buses × buses)
  - Values indicate connection direction (+1 for "from" bus, -1 for "to" bus, or 0)
  - Sparsity pattern matches the bus admittance matrix.

**Purpose:**
The adjacency matrix captures pure network topology without electrical parameters, making it useful for graph-theoretic analysis.

### Bus Admittance Matrix ([`Ybus`](@ref))

The bus admittance matrix combines network topology with electrical parameters (impedance/admittance).

**Properties:**

  - Square matrix (buses × buses)
  - Diagonal elements are self-admittances
  - Off-diagonal elements are mutual admittances
  - Symmetric for passive networks
  - Both dimensions are indexed by bus numbers

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
  - Rows are indexed by arc tuples `(from_bus, to_bus)`, columns by bus numbers

**Note:**
See [`VirtualPTDF`](@ref) for cases where it is not possible to compute or store the full PTDF matrix.

### Line Outage Distribution Factors ([`LODF`](@ref))

LODF matrices answer: "If branch $m$ fails, how much does the flow redistribute to branch $k$?"

**Key Characteristics:**

  - Predicts post-contingency flows
  - Essential for N-1 security analysis
  - Computed from PTDF matrix
  - Helps identify critical lines
  - Both dimensions are indexed by arc tuples `(from_bus, to_bus)`

**Mathematical Relationship:**
LODF is derived from PTDF using matrix operations that model the effect of removing a branch from the network.

See [`VirtualLODF`](@ref) for cases where it is not possible to compute or store the full LODF matrix.

## Arc-Based Indexing

All matrices that involve branches use **arc tuples** as identifiers instead of branch name strings. An arc tuple is a `Tuple{Int, Int}` of the form `(from_bus_number, to_bus_number)`, representing the directed connection between two buses.

This indexing approach:

  - Provides compact, unambiguous identification of network elements
  - Naturally handles network reductions where branches may be merged or eliminated
  - Aligns with the mathematical formulation where branches are defined by their endpoint buses

Each matrix stores `axes` and `lookup` fields:

  - **`axes`**: A tuple of vectors listing the arc tuples and/or bus numbers for each dimension
  - **`lookup`**: A tuple of dictionaries that map arc tuples or bus numbers to integer matrix indices

**Indexing summary by matrix type:**

| Matrix            | Dimension 1 (rows)    | Dimension 2 (columns) |
|:----------------- |:--------------------- |:--------------------- |
| `IncidenceMatrix` | Arc tuples            | Bus numbers           |
| `BA_Matrix`       | Arc tuples            | Bus numbers           |
| `PTDF`            | Arc tuples            | Bus numbers           |
| `LODF`            | Arc tuples            | Arc tuples            |
| `Ybus`            | Bus numbers           | Bus numbers           |
| `VirtualPTDF`     | Arc tuples            | Bus numbers           |
| `VirtualLODF`     | Arc tuples            | Arc tuples            |

!!! note
    For backward compatibility, branch name strings can also be used to index PTDF and LODF matrices. This internally maps the branch name to its corresponding arc tuple via the network reduction data. Using arc tuples directly is recommended for new code.

!!! note
    When network reductions are applied (e.g. `RadialReduction`, `DegreeTwoReduction`), some branches are eliminated from the network. Attempting to index a matrix with an arc tuple that was reduced will result in an error. Use `get_axes` to inspect the available arc tuples after reduction.
