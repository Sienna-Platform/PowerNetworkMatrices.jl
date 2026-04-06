# How to Reduce Repeated Operations When Building Multiple Matrices

When building several network matrices for the same system, intermediate
matrices such as the [`IncidenceMatrix`](@ref), [`BA_Matrix`](@ref), and
[`ABA_Matrix`](@ref) are recomputed each time. You can avoid this overhead by
computing them once and passing them to downstream constructors.

## The Problem

Calling [`PTDF`](@ref) and [`LODF`](@ref) independently each rebuilds the
incidence and admittance matrices from scratch:

```julia
# Each call builds IncidenceMatrix, BA_Matrix, and ABA_Matrix internally
ptdf = PTDF(sys)
lodf = LODF(sys)
```

## The Solution

Compute the shared intermediate matrices once, then pass them directly:

```julia
using PowerNetworkMatrices
using PowerSystems
using PowerSystemCaseBuilder

sys = build_system(PSITestSystems, "c_sys5")

# Step 1: build intermediate matrices once
a = IncidenceMatrix(sys)
ba = BA_Matrix(sys)

# Step 2: build PTDF from the pre-computed matrices
ptdf = PTDF(a, ba)

# Step 3: build LODF from the incidence matrix and PTDF
lodf = LODF(a, ptdf)
```

This pattern is especially beneficial for large systems where the intermediate
matrix construction is expensive.

## Sharing the ABA factorization

If you also need the [`ABA_Matrix`](@ref) (for example, to build
[`VirtualPTDF`](@ref) or [`VirtualLODF`](@ref)), factorize it once and reuse:

```julia
aba = ABA_Matrix(sys; factorize = true)

# Use IncidenceMatrix, ABA_Matrix, and BA_Matrix together
lodf_alt = LODF(a, aba, ba)
```

## When this matters

The savings grow with system size. For a system with thousands of buses,
building the [`Ybus`](@ref) and downstream matrices can take several seconds.
Sharing intermediate results avoids repeating that work.

## Related Topics

  - [Network Matrices](@ref) — tutorial covering all matrix types
  - [How to Choose a Linear Solver](@ref) — additional performance guidance
