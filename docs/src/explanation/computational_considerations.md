# Computational Considerations

## Sparsity

Power networks are sparse — most buses connect to only a few others — and this is
exploited via sparse linear solvers, so the connectivity matrices ([`IncidenceMatrix`](@ref) and [`Ybus`](@ref))
are very sparse. The sensitivity matrices ([`PTDF`](@ref), [`LODF`](@ref)), however, are obviously dense: e.g., one entry per bus in every
[`PTDF`](@ref) column, most of them negligible because a branch is nearly insensitive
to an injection electrically far away.

## Sparsification and tolerance

Those negligible entries can be dropped to recover sparsity. The **tolerance** is the
cutoff below which an entry is set to zero. The default [`AutoTolerance`](@ref) picks
it from the data as a *relative per-row* drop — an entry is dropped when
`|x| < α · max|row|` — which keeps large matrices sparse while leaving small systems
and the dense constructors exact. The exact rule (including the bus-count gate that
makes it a no-op on small systems and the `Float64` `tol` alternative) is in the
[`AutoTolerance`](@ref) docstring. For very large studies prefer the
[`VirtualPTDF`](@ref)/[`VirtualLODF`](@ref) variants, which compute rows on demand
and store each one sparsely.

### Accuracy and limitations

Sparsification trades exactness for memory. Because each dropped entry is below
`α · max|row|` (`α ≤ 1e-2`, typically `α ≈ 5e-4`), the dominant sensitivities are
never touched — but two properties are worth understanding before relying on a
sparsified matrix for a sensitive calculation:

  - **The error is one-signed, not zero-mean.** A dropped entry becomes exactly
    zero, so a row's total mass strictly decreases. Summing many small entries of a
    row — e.g. aggregating a transfer's flow contribution across a subsystem's buses —
    accumulates that truncation in one direction instead of cancelling. The bias is
    bounded by `(number of dropped entries) · α · max|row|`.
  - **The cutoff is per row, so cross-row/column invariants are not preserved.** Each
    row is sparsified against its own peak, so quantities that couple different rows or
    columns — Kirchhoff's current law, a column sum, the *difference* of two entries —
    are not conserved. Two buses `j, k` both far from branch `i` can have similar
    `PTDF[i,j]`, `PTDF[i,k]` land on opposite sides of the cutoff: each absolute error
    stays under threshold, yet the relative error on their tiny difference can approach
    100%.

Contingency corrections do not compound this: in [`VirtualMODF`](@ref) the cutoff is
applied to the *final* post-contingency row, after the exact Woodbury solve, so the
error stays bounded by the cutoff however near-critical the contingency.

When you need an exact result — to preserve KCL, to difference two small
sensitivities, or to validate against a reference — pass a `tol::Float64`
(`tol = eps()` for an unsparsified matrix, or a deliberate fixed cutoff).

## Matrix sizes and complexity

A system with $N_b$ buses and $N_a$ arcs:

| Operation         | Dimensions           | Complexity           | Notes                          |
|:----------------- |:-------------------- |:-------------------- |:------------------------------ |
| Incidence Matrix  | $N_a × N_b$ (sparse) | $O(N_a)$             | Simple topology scan           |
| Admittance Matrix | $N_b × N_b$ (sparse) | $O(N_a)$             | Includes electrical parameters |
| PTDF              | $N_a × N_b$ (dense)  | $O(N_b^3)$           | Requires matrix inversion      |
| LODF              | $N_a × N_a$ (dense)  | $O(N_a \cdot N_b^2)$ | Derived from PTDF              |
