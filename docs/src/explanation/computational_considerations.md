## Computational Considerations

### Matrix Construction

All matrices in `PowerNetworkMatrices.jl` are derived from the `Ybus` matrix (i.e. building any matrix starts with building the `Ybus`). Additionally, all network reductions are applied to the `Ybus` matrix prior to computing the downstream matrices. This design choice is key for enabling high performance and code maintainability: Looping through the system objects is required only when building the `Ybus` (slow) and subsequent operations are built on fast matrix operations on (often sparse) matrices. In addition, network reductions are only defined for the `Ybus` but can be applied uniformly across all matrices.

### Sparsity

Power networks are sparse; most buses connect to only a few others. This sparsity is exploited for computational efficiency via sparse linear solvers:

  - Incidence and admittance matrices are very sparse.
  - Common sensitivity matrices (e.g. PTDF and LODF) are dense.

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
