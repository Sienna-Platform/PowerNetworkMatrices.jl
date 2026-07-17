# Accessor functions

This page documents the functions used to read structural and numeric data from
the network matrices — matrix data, axes, lookup dictionaries, reference buses,
reduction data, and system provenance. For the matrix constructors see the
[Matrix type reference](matrix_types.md); for how indexing resolves see
[Matrix overview & indexing](network_matrices_overview.md).

Every matrix is a subtype of `PowerNetworkMatrix{T} <: AbstractArray{T,2}` and
carries two structural fields — `axes` (identifier vectors, one per dimension) and
`lookup` (identifier → integer-position dictionaries) — plus a backing `data`
array.

## Exported vs. internal accessors

The exported accessors are documented in full on the [Full public API](public.md);
the internal helpers below are reached through the module prefix (commonly aliased
`PNM.`) and are **not** on that page, so this is their reference.

  - **Exported:** [`get_ptdf_data`](@ref), [`get_lodf_data`](@ref),
    [`get_partial_lodf_row`](@ref), [`get_network_reduction_data`](@ref),
    [`get_system_uuid`](@ref).
  - **Internal (call via `PNM.`):** `get_data`, `get_axes`, `get_lookup`,
    `get_bus_axis`, `get_arc_axis`, `get_bus_lookup`, `get_arc_lookup`,
    `get_ref_bus`, `get_ref_bus_position`.

## Data extraction

  - **`PNM.get_data(mat)`** — the raw backing array (`mat.data`) exactly as stored:
    a complex [`SparseMatrixCSC`](@extref Julia SparseArrays.SparseMatrixCSC) for
    [`Ybus`](@ref), the internally **transposed** dense matrix for
    [`PTDF`](@ref)/[`LODF`](@ref).
  - **[`get_ptdf_data`](@ref)** / **[`get_lodf_data`](@ref)** — the matrix in
    standard (non-transposed) orientation, via a lazy `transpose` (not a copy). For
    a [`VirtualLODF`](@ref), [`get_lodf_data`](@ref) instead returns the LRU cache
    contents as a `Dict{Int, Vector{Float64}}` of already-computed rows.
  - **[`get_partial_lodf_row`](@ref)** — the LODF row for a **partial** susceptance
    change `delta_b` on one arc (full outage: `delta_b = -arc_susceptance`). The
    entry point for partial outages that a plain [`VirtualLODF`](@ref) row (which
    assumes a full outage) does not cover.

## Axes and lookups

`PNM.get_axes(mat)` returns `mat.axes` and `PNM.get_lookup(mat)` returns
`mat.lookup`, each a 2-tuple ordered `(dimension 1, dimension 2)`. The axis vector
lists identifiers (bus numbers as `Int`, arcs as `Tuple{Int,Int}`) in position
order; the matching lookup maps each identifier back to its integer position in
`data`. These are the authoritative way to enumerate valid indices — especially
after a reduction, where some arcs/buses are no longer present. Defined for every
matrix type.

The dimension-specific accessors `PNM.get_bus_axis` / `get_arc_axis` /
`get_bus_lookup` / `get_arc_lookup` select the correct dimension without the caller
knowing which index (1 or 2) is the bus or arc dimension for a given matrix type.
They are defined only for the dimensions a matrix actually has:

| Matrix                | `get_bus_axis`    | `get_arc_axis`    |
|:--------------------- |:-----------------:|:-----------------:|
| `IncidenceMatrix`     | `axes[2]`         | `axes[1]`         |
| `AdjacencyMatrix`     | `axes[1]`         | — (both dims bus) |
| `Ybus`                | `axes[1]`         | — (both dims bus) |
| `ArcAdmittanceMatrix` | `axes[2]`         | `axes[1]`         |
| `BA_Matrix`           | `axes[1]`         | `axes[2]`         |
| `ABA_Matrix`          | `axes[1]`         | — (both dims bus) |
| `PTDF`                | `axes[1]`         | `axes[2]`         |
| `LODF`                | — (both dims arc) | `axes[1]`         |
| `VirtualPTDF`         | ✓                 | ✓                 |
| `VirtualLODF`         | — (both dims arc) | `axes[1]`         |
| `VirtualMODF`         | `axes[2]`         | `axes[1]`         |

## Reference buses

`PNM.get_ref_bus(mat)` returns the sorted reference (slack) bus numbers — one per
electrical island — and `PNM.get_ref_bus_position(mat)` their integer positions in
the bus dimension. Together they identify the slack bus(es) held fixed when the
matrix was built, which matters for interpreting [`PTDF`](@ref) columns and for
reduction/contingency math. Defined for the distribution-factor, incidence,
adjacency, BA/ABA, and arc-admittance matrices.

## Reduction data

**[`get_network_reduction_data`](@ref)** returns the [`NetworkReductionData`](@ref)
for the matrix — which buses/arcs were merged or eliminated and how they map back
(empty when no reduction was applied). This is the object queried by the reduction
accessors (`get_bus_reduction_map`, `get_removed_buses`, `get_reductions`, …); see
the [Network reduction reference](network_reduction.md).

!!! note "Serialization drops reduction data"
    
    A [`PTDF`](@ref) rehydrated from HDF5 via [`from_hdf5`](@ref) carries an
    **empty** [`NetworkReductionData`](@ref). See [Serialization](serialization.md).

## System provenance

**[`get_system_uuid`](@ref)** returns the UUID of the
[`System`](@extref PowerSystems.System) the matrix was built from, or `nothing` for
types that do not track origin. [`VirtualPTDF`](@ref) and [`VirtualMODF`](@ref)
store it; it backs the consistency check that a matrix and a system passed together
share a source.

## See also

  - [Matrix overview & indexing](network_matrices_overview.md) — the `A[row, column]` resolution rules.
  - [Matrix type reference](matrix_types.md) — constructors and keyword arguments.
  - [Network reduction reference](network_reduction.md) — the full `NetworkReductionData` accessor set.
  - [Full public API](public.md) — autodocs for the exported accessors.
