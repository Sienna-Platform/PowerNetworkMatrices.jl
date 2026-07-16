# Computational Considerations

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

#### Accuracy and limitations

Sparsification trades exactness for memory, and the relative per-row rule has consequences worth understanding before you rely on a sparsified matrix for a sensitive calculation. Each dropped entry is below `α · max|row|` (`α ≤ 1e-2`, and typically `α ≈ δ ≈ 5e-4`), so the dominant sensitivities are never touched — but the following hold:

  - **The error is one-signed, not zero-mean.** A dropped entry is set to exactly zero, never rounded, so each row's total mass strictly decreases. When you sum many small entries of a row (for example, aggregating a flow contribution across many buses), the truncation errors accumulate in the same direction instead of cancelling. The bias is bounded by `(number of dropped entries) · α · max|row|`.

  - **The cutoff is per row, so global invariants are not preserved.** Each row is sparsified against *its own* peak, with no coupling between rows or columns, so quantities depending on cross-row or cross-column structure — Kirchhoff's current law, a column sum, or the *difference* of two entries — are not conserved. The sharp case is two buses `j, k` both far from branch `i`: `PTDF[i,j]` and `PTDF[i,k]` may be similar in magnitude yet fall on opposite sides of the cutoff. Each *absolute* error stays below the threshold, but the *relative* error on the (tiny) difference `PTDF[i,j] − PTDF[i,k]` can approach 100%.
  - **Auto-discovered precision assumes a power-of-10 base.** With `data_precision = :auto`, `δ` counts the significant figures of the branch reactances. Decimal significant-figure counts are invariant under multiplication by a power of ten (the conventional 100 MVA base) but **not** under an arbitrary impedance base: data converted by a non-power-of-10 base (e.g. `Z_base = kV²/MVA = 190.44 Ω`) reads more figures than it carries, so `:auto` *over-estimates* precision. The direction is safe — a smaller `α`, hence *less* aggressive dropping — but on such data prefer an explicit `data_precision`.
  - **Contingency (Woodbury) corrections do not amplify the error.** In [`VirtualMODF`](@ref) the cutoff is applied to the *final* post-contingency row, after the exact Woodbury solve; the correction is computed from exact factorization solves, never sparsified rows. The error stays bounded by the cutoff however near-critical (near-islanding) the contingency is, even when the Woodbury update is severely ill-conditioned. Verified directly in the test suite.

When you need an exact result — to preserve KCL, to difference two small sensitivities, or to validate against a reference — pass a `Float64` `tol`: use `tol = eps()` for an unsparsified matrix, or a deliberate fixed cutoff for a reproducible absolute tolerance. `AutoTolerance` is the memory-versus-accuracy lever; the explicit `Float64` paths remain available for when exactness matters more than size.

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
