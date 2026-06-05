## Computational Considerations

### Matrix Construction

All matrices in `PowerNetworkMatrices.jl` are derived from the `Ybus` matrix (i.e. building any matrix starts with building the `Ybus`). Additionally, all network reductions are applied to the `Ybus` matrix prior to computing the downstream matrices. This design choice is key for enabling high performance and code maintainability: Looping through the system objects is required only when building the `Ybus` (slow) and subsequent operations are built on fast matrix operations on (often sparse) matrices. In addition, network reductions are only defined for the `Ybus` but can be applied uniformly across all matrices.

### Sparsity

Power networks are sparse; most buses connect to only a few others. This sparsity is exploited for computational efficiency via sparse linear solvers:

  - Incidence and admittance matrices are very sparse.
  - Common sensitivity matrices (e.g. PTDF and LODF) are dense.

### Automatic Sparsification Tolerance

The PTDF and LODF are obtained by solving against the reduced susceptance matrix `ABA = Aᵀ B A`, which is sparse. In theory, however, **the inverse of a sparse matrix is dense**: `ABA` is a grounded graph Laplacian, and `ABA⁻¹` has essentially no zeros even though `ABA` does. So the sensitivity matrices come out dense, and for a large network a single PTDF column has one entry per bus — tens of thousands of numbers, almost all of them negligible (a branch is effectively insensitive to an injection electrically far away).

To recover sparsity we apply a **tolerance** and drop entries below it (`droptol!`). The size of that cutoff is the question this package answers automatically. Every PTDF/LODF constructor takes a `tol` keyword:

```julia
tol::Union{Float64, AutoTolerance} = AutoTolerance()
```

  - A **`Float64`** is an explicit, *absolute* cutoff: any entry with `|x| ≤ tol` is dropped. Use it to pin an exact result (`tol = eps()`) or to sparsify by a fixed number. This is the backward-compatible path.

  - An **[`AutoTolerance`](@ref)** (the default) chooses the cutoff from the data instead of a hand-tuned guess. It drops an entry of a row only when it is below the precision the input data can justify, *relative to that row's own peak*:
    
    ```
    drop entry j of row i   when   |row_i[j]| < α · max|row_i|,
    α = clamp(safety · δ, 1e-6, 1e-2)
    ```
    
    where `δ` is the relative precision of the branch reactances (auto-discovered from their significant figures, or set explicitly via `data_precision`). Because the cutoff is *relative to each row's peak*, the achieved column sparsity is independent of the matrix scale and of how ill-conditioned `ABA` is. The 1-norm condition number of `ABA` is still estimated and logged as a diagnostic, but it never enters the cutoff.

Sparsification only matters at scale, so `AutoTolerance` acts only where it pays off:

  - **On-demand (virtual) matrices at or above `AUTO_TOLERANCE_BUS_LIMIT` buses** are sparsified per requested row/column — this is what lets a column of a large system come back sparse instead of dense.
  - **Small systems and the dense `PTDF`/`LODF` constructors are returned exactly** (`AutoTolerance` is a no-op there), preserving their dense type and numerical values. Pass a `Float64` `tol` to sparsify those explicitly.

For very large studies, prefer the [`VirtualPTDF`](@ref)/[`VirtualLODF`](@ref) variants: they compute rows on demand and, with the default `AutoTolerance`, store each one sparsely.

### Matrix Sizes

A system with $N_b$ buses and $N_a$ arcs has matrix dimensions:

  - Incidence: $N_a × N_b$ (sparse)
  - Admittance: $N_b × N_b$ (sparse)
  - PTDF: $N_a × N_b$ (dense)
  - LODF: $N_a × N_a$ (dense)

### Computational Complexity

| Operation         | Complexity           | Notes                          |
|:----------------- |:-------------------- |:------------------------------ |
| Incidence Matrix  | O($N_a$)             | Simple topology scan           |
| Admittance Matrix | O($N_a$)             | Includes electrical parameters |
| PTDF              | O($N_b^3$)           | Requires matrix inversion      |
| LODF              | O($N_a \cdot N_b^2$) | Derived from PTDF              |
