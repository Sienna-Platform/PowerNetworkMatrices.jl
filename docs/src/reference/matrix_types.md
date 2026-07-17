# Matrix type reference

This page describes what each matrix type PowerNetworkMatrices exports
represents and the gotchas specific to it. It is a companion to the
[matrix overview and indexing hub](network_matrices_overview.md), which covers how
`A[row, col]` resolves. For the authoritative constructor signatures, keyword
arguments, and every docstring, see the [full public API](public.md).

Unless noted otherwise, every type below is a subtype of the common supertype
`PowerNetworkMatrix` and shares the same indexing and accessor interface.

## The `PowerNetworkMatrix` supertype

```julia
abstract type PowerNetworkMatrix{T} <: AbstractArray{T, 2} end
```

Because it is an `AbstractArray{T,2}`, every concrete matrix supports `size`,
`axes`, and `getindex`. Indexing is overloaded so rows and columns are addressed
by *power-system identifiers* (bus numbers, arc tuples, branch names,
[`PSY`](@extref PowerSystems.System) components) rather than integer positions —
see the [overview hub](network_matrices_overview.md) for the accepted key types.
Concrete types carry a `data` field (dense `Matrix` or
[`SparseMatrixCSC`](@extref Julia SparseArrays.SparseMatrixCSC)), an `axes` tuple
of identifier vectors, a `lookup` tuple of `Dict`s, and a
[`NetworkReductionData`](@ref) describing any reduction applied at construction.

Several types store their `data` **transposed** for efficiency
(`stores_transpose` is `true` for [`PTDF`](@ref), [`LODF`](@ref), and
[`BA_Matrix`](@ref)). Indexing hides this; use the type-specific data accessors
([`get_ptdf_data`](@ref), [`get_lodf_data`](@ref)) to obtain the standard
(non-transposed) orientation.

## Distribution-factor matrices

[`PTDF`](@ref) and [`LODF`](@ref) are dense distribution-factor matrices sharing
construction options (`linear_solver`, `tol`) and arc-tuple indexing; they differ
only in what each entry means.

  - **[`PTDF`](@ref)** — the Power Transfer Distribution Factor matrix.
    `PTDF[arc, bus]` is the sensitivity of the flow on `arc` to a unit injection
    at `bus`, under the DC approximation. Rows are arcs, columns are buses.
  - **[`LODF`](@ref)** — the Line Outage Distribution Factor matrix.
    `LODF[monitored, outaged]` is the fraction of `outaged`'s pre-outage flow that
    redistributes onto `monitored`. **Both** dimensions are arcs; diagonal entries
    are structurally `-1.0`, and are preserved when sparsifying.

Network reductions reach both through the `Ybus` `kwargs...`
(`network_reductions = [...]`) — there is no `reduce_*` keyword. `LODF(A, PTDFm)`
warns and densifies if the supplied [`PTDF`](@ref) was itself sparsified, since
that degrades LODF accuracy. Only [`PTDF`](@ref) supports HDF5 serialization (see
the [serialization reference](serialization.md)).

## Virtual (on-demand) matrices

Virtual matrices trade compute for memory: instead of materializing a dense
matrix they store the factorized system data and compute any single row on
demand, caching each row in an LRU `RowCache`. They expose the same
identifier-based indexing as their dense counterparts, are best for large systems
where only a subset of rows is needed, and are **not** serializable.

  - **[`VirtualPTDF`](@ref)** — lazy [`PTDF`](@ref); entries mean the same thing.
  - **[`VirtualLODF`](@ref)** — lazy [`LODF`](@ref); both dimensions are arcs.
  - **[`VirtualMODF`](@ref)** — the on-demand Modification (post-contingency)
    Distribution Factor matrix: the [`PTDF`](@ref) row of a monitored arc *after* a
    modification/contingency, via Woodbury updates. **There is no dense `MODF`
    type.** Query it with the [`ContingencySpec`](@ref) / [`NetworkModification`](@ref)
    types (see the [contingency and modification reference](contingencies.md)).

!!! note "Distributed slack: `Dict` vs `Vector`"
    
    [`PTDF`](@ref)/[`VirtualPTDF`](@ref) take `dist_slack` as a
    `Dict{Int, Float64}` (bus → weight); [`VirtualLODF`](@ref) and
    [`VirtualMODF`](@ref) take it as a `Vector{Float64}`. The empty default uses a
    single reference bus. See [How to Use a Distributed Slack](@ref).

## Admittance and network-structure matrices

  - **[`Ybus`](@ref)** — the complex nodal admittance matrix (`YBUS_ELTYPE` is
    `ComplexF64`). `Ybus[i, j]` is the mutual admittance between buses `i` and `j`
    (off-diagonal) or the self-admittance of `i` (diagonal). It is the foundation
    matrix — every DC and virtual matrix is built from it — and stays complex, so
    it is factorized with KLU rather than the real-only backends. Asymmetry is
    legitimate for phase-shifting transformers and must not be "corrected". Pass
    `make_arc_admittance_matrices = true` to also build the two
    [`ArcAdmittanceMatrix`](@ref) objects.
  - **[`ArcAdmittanceMatrix`](@ref)** — per-arc admittance in one direction
    (`:FromTo` or `:ToFrom`), for power-flow use. Not built standalone by typical
    users; it is produced inside [`Ybus`](@ref). Query the direction with
    `get_direction`.
  - **[`IncidenceMatrix`](@ref)** — the bus-branch incidence matrix `A`.
    `A[arc, bus]` is `+1` at the from-bus, `-1` at the to-bus, `0` otherwise
    (exactly two nonzeros per arc row). Structural building block for the DC
    matrices.
  - **[`BA_Matrix`](@ref)** — the susceptance-weighted incidence matrix `B · A`.
    Axes match [`IncidenceMatrix`](@ref); stored transposed. The reference-bus
    column is dropped (one fewer column than the bus count).
  - **[`ABA_Matrix`](@ref)** — the reduced bus-susceptance matrix `Aᵀ · B · A` with
    reference buses removed for invertibility — the DC-power-flow system matrix.
    Its `K` field optionally holds a KLU factorization; build it factorized
    (`factorize = true`), or [`factorize`](@ref)/[`is_factorized`](@ref) after the
    fact. See [How to Factorize and Reuse an ABA Matrix](@ref).
  - **[`AdjacencyMatrix`](@ref)** — a symmetric bus-by-bus connectivity matrix
    (`Int8`): nonzero where two buses share a branch, zero on the diagonal. Used by
    [`validate_connectivity`](@ref) and [`find_subnetworks`](@ref).

## Concrete type aliases

`PowerflowMatrixTypes.jl` defines concrete aliases for the fully-parameterized
matrix types that appear in hot paths and downstream dispatch. Use them in method
signatures when you need to fix a concrete storage layout. All are exported, as is
`YBUS_ELTYPE`.

| Alias                        | Underlying type                                   | Meaning                                             |
|:---------------------------- |:------------------------------------------------- |:--------------------------------------------------- |
| `DC_PTDF_Matrix`             | `PTDF{…, Matrix{Float64}}`                        | Dense PTDF with bus/arc axes.                       |
| `DC_vPTDF_Matrix`            | `VirtualPTDF{…, K} where {K}`                     | Virtual PTDF (factorization type `K` left free).    |
| `DC_BA_Matrix`               | `BA_Matrix{…}`                                    | BA matrix with bus/arc axes.                        |
| `DC_ABA_Matrix_Factorized`   | `ABA_Matrix{…, KLULinSolveCache{Float64, Int64}}` | ABA matrix carrying a KLU factorization.            |
| `DC_ABA_Matrix_Unfactorized` | `ABA_Matrix{…, Nothing}`                          | ABA matrix with no factorization (`K === nothing`). |
| `AC_Ybus_Matrix`             | `Ybus{…}`                                         | Ybus with integer bus axes.                         |

## See also

  - [Matrix overview and indexing hub](network_matrices_overview.md) — how
    `A[row, col]` resolves and the per-type axis summary.
  - [Full public API](public.md) — authoritative docstrings and signatures.
  - [Network reduction reference](network_reduction.md) — the
    [`NetworkReduction`](@ref) spec types passed via `network_reductions`.
  - [Contingency and modification reference](contingencies.md) — types used to
    query [`VirtualMODF`](@ref).
  - [Tolerance and solver settings](tolerance_and_solvers.md) — [`AutoTolerance`](@ref)
    and `linear_solver` backends.
