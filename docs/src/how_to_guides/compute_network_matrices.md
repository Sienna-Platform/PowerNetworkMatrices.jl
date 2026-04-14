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

# Access the matrix data
matrix_data = PNM.get_ptdf_data(ptdf_matrix)
```

## Computing LODF Matrix

To compute the Line Outage Distribution Factor matrix using [`LODF`](@ref):

```julia
lodf_matrix = PNM.LODF(sys)
matrix_data = PNM.get_lodf_data(lodf_matrix)
```

## Computing Virtual PTDF Matrix

For systems where you need virtual representation using [`VirtualPTDF`](@ref):

```julia
vptdf_matrix = PNM.VirtualPTDF(sys)
matrix_data = PNM.get_ptdf_data(vptdf_matrix)
```

## Computing Virtual LODF Matrix

Similarly for virtual LODF using [`VirtualLODF`](@ref):

```julia
vlodf_matrix = PNM.VirtualLODF(sys)
matrix_data = PNM.get_lodf_data(vlodf_matrix)
```

## Computing Incidence and BA Matrices

For the fundamental network topology matrices:

Compute the incidence matrix using [`IncidenceMatrix`](@ref):

```julia
incidence_matrix = PNM.IncidenceMatrix(sys)
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

## Next Steps

  - Learn about choosing linear solvers for optimal performance
  - Understand the theory behind network matrices in the Explanation section
