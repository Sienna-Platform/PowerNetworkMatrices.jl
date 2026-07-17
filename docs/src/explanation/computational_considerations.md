# Computational Considerations

## Matrix construction

All matrices in `PowerNetworkMatrices.jl` are derived from the [`Ybus`](@ref) (i.e.
building any matrix starts with building the [`Ybus`](@ref)), and all network
reductions are applied to the [`Ybus`](@ref) before computing the downstream
matrices. This is key for performance and maintainability: looping through system
objects is required only when building the [`Ybus`](@ref) (slow); subsequent
operations are fast (often sparse) matrix operations, and a reduction defined once
on the [`Ybus`](@ref) applies uniformly to every matrix.

## Sparsity

Power networks are sparse — most buses connect to only a few others — and this is
exploited via sparse linear solvers. Incidence and admittance matrices are very
sparse; the common sensitivity matrices ([`PTDF`](@ref), [`LODF`](@ref)) are dense.

## Why sensitivity matrices come out dense

The [`PTDF`](@ref) and [`LODF`](@ref) are obtained by solving against the reduced
susceptance matrix `ABA = Aᵀ B A`, which is sparse. But **the inverse of a sparse
matrix is dense**: `ABA` is a grounded graph Laplacian, and `ABA⁻¹` has essentially
no zeros. So the sensitivity matrices are dense, and for a large network a single
[`PTDF`](@ref) column has one entry per bus — mostly negligible values, since a
branch is effectively insensitive to an injection electrically far away.

To recover sparsity the package applies a **tolerance** and drops entries below it.
The default [`AutoTolerance`](@ref) chooses that cutoff from the data — a *relative
per-row* drop that keeps large virtual matrices sparse while leaving small systems
and the dense constructors exact. The exact rule, the bus-count gate, and the
`Float64` alternative are documented in the [`AutoTolerance`](@ref) docstring; for very
large studies prefer the [`VirtualPTDF`](@ref)/[`VirtualLODF`](@ref) variants, which
compute rows on demand and store each one sparsely.

### Accuracy and limitations

Sparsification trades exactness for memory, and the relative per-row rule has
consequences worth understanding before you rely on a sparsified matrix for a
sensitive calculation. Each dropped entry is below `α · max|row|` (`α ≤ 1e-2`,
typically `α ≈ 5e-4`), so the dominant sensitivities are never touched — but:

  - **The error is one-signed, not zero-mean.** A dropped entry is set to exactly
    zero, so each row's total mass strictly decreases. Summing many small entries of
    a row (e.g. aggregating a flow contribution across many buses) accumulates
    truncation error in one direction instead of cancelling. The bias is bounded by
    `(number of dropped entries) · α · max|row|`.
  - **The cutoff is per row, so global invariants are not preserved.** Each row is
    sparsified against its own peak, so quantities depending on cross-row/column
    structure — Kirchhoff's current law, a column sum, or the *difference* of two
    entries — are not conserved. Two buses `j, k` both far from branch `i` may have
    similar `PTDF[i,j]`, `PTDF[i,k]` that fall on opposite sides of the cutoff: each
    absolute error stays below threshold, but the relative error on the tiny
    difference can approach 100%.
  - **Auto-discovered precision assumes a power-of-10 base.** With
    `data_precision = :auto`, `δ` counts significant figures of the branch
    reactances — invariant under the 100 MVA base but not under an arbitrary
    impedance base, so such data reads more figures than it carries and `:auto`
    *over-estimates* precision. The direction is safe (less aggressive dropping), but
    prefer an explicit `data_precision` there.
  - **Contingency (Woodbury) corrections do not amplify the error.** In
    [`VirtualMODF`](@ref) the cutoff is applied to the *final* post-contingency row,
    after the exact Woodbury solve, so the error stays bounded by the cutoff however
    near-critical the contingency is. Verified in the test suite.

When you need an exact result — to preserve KCL, to difference two small
sensitivities, or to validate against a reference — pass a `Float64` `tol`
(`tol = eps()` for an unsparsified matrix, or a deliberate fixed cutoff).

## Matrix sizes and complexity

A system with $N_b$ buses and $N_a$ arcs:

| Operation         | Dimensions           | Complexity           | Notes                          |
|:----------------- |:-------------------- |:-------------------- |:------------------------------ |
| Incidence Matrix  | $N_a × N_b$ (sparse) | O($N_a$)             | Simple topology scan           |
| Admittance Matrix | $N_b × N_b$ (sparse) | O($N_a$)             | Includes electrical parameters |
| PTDF              | $N_a × N_b$ (dense)  | O($N_b^3$)           | Requires matrix inversion      |
| LODF              | $N_a × N_a$ (dense)  | O($N_a \cdot N_b^2$) | Derived from PTDF              |
