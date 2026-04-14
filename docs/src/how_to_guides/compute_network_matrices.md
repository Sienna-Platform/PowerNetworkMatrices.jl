# How to Compute Network Matrices

This guide shows you how to compute various network matrices for your power system.

## Prerequisites

  - `PowerNetworkMatrices.jl` installed
  - A power system model loaded (see [Quick Start Guide](@ref))

## Computing PTDF Matrix

To compute the Power Transfer Distribution Factor matrix using [`PTDF`](@ref):

```julia
using PowerNetworkMatrices
const PNM = PowerNetworkMatrices

# Assuming you have a system loaded
ptdf_matrix = PNM.PTDF(sys)

# Access the matrix data (in standard arcs × buses orientation)
matrix_data = PNM.get_ptdf_data(ptdf_matrix)
```

### Indexing PTDF Elements

The PTDF matrix is indexed by **arc tuples** `(from_bus, to_bus)` and **bus numbers**:

```julia
# Access PTDF element for arc (1, 2) and bus 3
ptdf_matrix[(1, 2), 3]

# Inspect available axes and lookup dictionaries
PNM.get_axes(ptdf_matrix)
PNM.get_lookup(ptdf_matrix)
```

## Computing LODF Matrix

To compute the Line Outage Distribution Factor matrix using [`LODF`](@ref):

```julia
lodf_matrix = PNM.LODF(sys)
matrix_data = PNM.get_lodf_data(lodf_matrix)
```

### Indexing LODF Elements

The LODF matrix is indexed by **arc tuples** for both dimensions:

```julia
# Access LODF element: flow change on arc (1, 4) due to outage of arc (2, 3)
lodf_matrix[(1, 4), (2, 3)]

# Inspect available axes and lookup dictionaries
PNM.get_axes(lodf_matrix)
PNM.get_lookup(lodf_matrix)
```

## Computing Virtual PTDF Matrix

For systems where you need virtual representation using [`VirtualPTDF`](@ref):

```julia
vptdf_matrix = PNM.VirtualPTDF(sys)

# Access element by arc tuple and bus number
vptdf_matrix[(1, 2), 3]
```

## Computing Virtual LODF Matrix

Similarly for virtual LODF using [`VirtualLODF`](@ref):

```julia
vlodf_matrix = PNM.VirtualLODF(sys)

# Access element by arc tuples
vlodf_matrix[(1, 2), (3, 4)]
```

## Computing Incidence and BA Matrices

For the fundamental network topology matrices:

Compute the incidence matrix using [`IncidenceMatrix`](@ref):

```julia
incidence_matrix = PNM.IncidenceMatrix(sys)

# Axes are (arc_tuples, bus_numbers)
PNM.get_axes(incidence_matrix)
```

Compute the BA matrix (Bus-Admittance) using [`BA_Matrix`](@ref):

```julia
ba_matrix = PNM.BA_Matrix(sys)
```

Compute the ABA matrix using [`ABA_Matrix`](@ref):

```julia
aba_matrix = PNM.ABA_Matrix(sys)
```

## Working with Pre-computed Matrices

If you have already computed the incidence and BA matrices, you can use them to compute [`PTDF`](@ref):

```julia
# Compute base matrices first
ba_matrix = PNM.BA_Matrix(sys)
a_matrix = PNM.IncidenceMatrix(sys)

# Use them to compute PTDF
ptdf_matrix = PNM.PTDF(a_matrix, ba_matrix)
```

## Understanding Axes and Lookup Dictionaries

All network matrices store `axes` and `lookup` fields that describe how rows and columns map to physical network elements:

  - **`axes`**: A tuple of vectors containing the identifiers for each dimension
  - **`lookup`**: A tuple of dictionaries mapping identifiers to integer indices

For matrices involving branches (IncidenceMatrix, PTDF, LODF), branches are represented as **arc tuples** `(from_bus_number, to_bus_number)` rather than branch name strings. This provides a compact, unambiguous identifier for each directed branch in the network.

| Matrix            | Dimension 1 (rows)    | Dimension 2 (columns) |
|:----------------- |:--------------------- |:--------------------- |
| `IncidenceMatrix` | Arc tuples            | Bus numbers           |
| `PTDF`            | Arc tuples            | Bus numbers           |
| `LODF`            | Arc tuples            | Arc tuples            |
| `Ybus`            | Bus numbers           | Bus numbers           |
| `VirtualPTDF`     | Arc tuples            | Bus numbers           |
| `VirtualLODF`     | Arc tuples            | Arc tuples            |

!!! note
    For backward compatibility, branch name strings can also be used to index PTDF and LODF matrices. This uses the `get_branch_multiplier` function internally to map names to arc tuples. Using arc tuples directly is recommended.

## Next Steps

  - Learn about choosing linear solvers for optimal performance
  - Understand the theory behind network matrices in the Explanation section
