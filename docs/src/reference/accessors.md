# Accessor functions

This page documents the functions used to read structural and numeric data from
the network matrices — matrix data, axes, lookup dictionaries, reference buses,
reduction data, and system provenance. For the matrix constructors themselves
see the [Matrix type reference](matrix_types.md); for how indexing resolves see
[Matrix overview & indexing](network_matrices_overview.md).

Every matrix is a subtype of `PowerNetworkMatrix{T} <: AbstractArray{T,2}` and
carries two structural fields — `axes` (identifier vectors, one per dimension)
and `lookup` (identifier → integer-position dictionaries, one per dimension) —
plus a backing `data` array.

## Exported vs. internal accessors

A subset of accessors is exported and appears in the [Full public API](public.md)
autodocs; the rest are internal helpers reached through the module prefix
(`PowerNetworkMatrices.get_axes(...)`, commonly aliased `PNM.`).

  - **Exported:** `get_ptdf_data`, `get_lodf_data`, `get_partial_lodf_row`,
    `get_network_reduction_data`, `get_system_uuid`.
  - **Internal (call via `PNM.`):** `get_data`, `get_axes`, `get_lookup`,
    `get_bus_axis`, `get_arc_axis`, `get_bus_lookup`, `get_arc_lookup`,
    `get_ref_bus`, `get_ref_bus_position`.

The internal accessors are stable enough to use in downstream code but are not
part of the exported surface, so their docstrings are not carried on
[public.md](public.md).

## Data extraction

### `get_data`

```julia
PNM.get_data(mat::PowerNetworkMatrix)
```

Returns the raw backing array (`mat.data`) exactly as stored. For `Ybus` this is
the complex `SparseMatrixCSC`; for `PTDF`/`LODF` this is the internally
**transposed** dense matrix (see `get_ptdf_data` / `get_lodf_data` for the
standard orientation). Use this when you need the concrete array for matrix
algebra.

### `get_ptdf_data`

```julia
get_ptdf_data(ptdf::PTDF) -> AbstractArray{Float64, 2}
```

Returns the PTDF matrix in the standard (non-transposed) orientation — rows are
arcs, columns are buses. Because `PTDF` stores its data transposed internally,
this returns `transpose(ptdf.data)` (a lazy `Transpose` wrapper, not a copy).
Source: `src/ptdf_calculations.jl`.

### `get_lodf_data`

```julia
get_lodf_data(lodf::LODF) -> AbstractArray{Float64, 2}
get_lodf_data(mat::VirtualLODF) -> Dict{Int, Vector{Float64}}
```

For a dense `LODF`, returns the matrix in standard orientation
(`transpose(lodf.data)`), rows and columns both arcs. Source:
`src/lodf_calculations.jl`.

For a [`VirtualLODF`](@ref), there is no full matrix; this returns the internal
LRU cache contents as a `Dict{Int, Vector{Float64}}` mapping already-computed row
indices to their lazily materialized row vectors. Uncomputed rows are absent.
Source: `src/virtual_lodf_calculations.jl`.

### `get_partial_lodf_row`

```julia
get_partial_lodf_row(vlodf::VirtualLODF, arc_idx::Int, delta_b::Float64) -> Vector{Float64}
get_partial_lodf_row(vlodf::VirtualLODF, arc::Tuple{Int, Int}, delta_b::Float64) ->
    Vector{Float64}
```

Computes the LODF row corresponding to a **partial** susceptance change
`delta_b` on a single arc, returning a `Vector{Float64}` of length `n_arcs`. The
arc may be given by integer index or by arc tuple (the tuple form maps through
`vlodf.lookup[1]`).

Conventions for `delta_b`:

  - Full outage: `delta_b = -arc_susceptance`.
  - Single-circuit outage on a double-circuit arc with total susceptance
    `b_total`: `delta_b = -b_circuit`.

This is the entry point for modeling partial (non-full) branch outages that a
plain `VirtualLODF` row (which assumes a full outage) does not cover. Source:
`src/virtual_lodf_calculations.jl`.

## Axes and lookups

### `get_axes` and `get_lookup`

```julia
PNM.get_axes(mat::PowerNetworkMatrix)    # -> Tuple of identifier vectors, one per dimension
PNM.get_lookup(mat::PowerNetworkMatrix)  # -> Tuple of Dicts, identifier -> integer position
```

`get_axes` returns `mat.axes` and `get_lookup` returns `mat.lookup`. Each is a
2-tuple ordered `(dimension 1, dimension 2)`. The axis vector for a dimension
lists the identifiers (bus numbers as `Int`, arcs as `Tuple{Int,Int}`) in
position order; the matching lookup dictionary maps each identifier back to its
integer position in `data`. These are the authoritative way to enumerate the
valid indices — especially after a network reduction, where some arcs/buses are
no longer present.

Defined for every matrix type.

### Dimension-specific axis and lookup accessors

```julia
PNM.get_bus_axis(mat)     # identifier vector for the bus-indexed dimension
PNM.get_arc_axis(mat)     # identifier vector for the arc-indexed dimension
PNM.get_bus_lookup(mat)   # Dict for the bus-indexed dimension
PNM.get_arc_lookup(mat)   # Dict for the arc-indexed dimension
```

These convenience accessors select the correct dimension of `axes` / `lookup`
without the caller having to know which dimension index (1 or 2) is the bus or
arc dimension for a given matrix type. They are defined only for the dimensions a
matrix actually has:

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

The `get_bus_lookup` / `get_arc_lookup` pair follows the same per-type dimension
mapping against `lookup`. Sources: the per-matrix files
(`src/ptdf_calculations.jl`, `src/lodf_calculations.jl`, `src/Ybus.jl`,
`src/BA_ABA_matrices.jl`, `src/IncidenceMatrix.jl`, `src/AdjacencyMatrix.jl`,
`src/ArcAdmittanceMatrix.jl`, and the virtual-matrix files).

## Reference buses

```julia
PNM.get_ref_bus(mat)           # -> Vector{Int}, sorted reference-bus numbers
PNM.get_ref_bus_position(mat)  # -> Vector{Int}, their integer positions
```

`get_ref_bus` returns the sorted list of reference (slack) bus numbers — one per
electrical subnetwork/island — collected from the matrix's `subnetwork_axes`.
`get_ref_bus_position` returns the corresponding integer positions within the
bus dimension (i.e. the columns/rows the reference buses occupy in `data`).
Together they identify the slack bus(es) that were held fixed when the matrix was
built, which matters for interpreting PTDF columns and for reduction/contingency
math.

Defined for `PTDF`, `LODF`, `VirtualPTDF`, `VirtualLODF`, `VirtualMODF`,
`IncidenceMatrix`, `AdjacencyMatrix`, `BA_Matrix`, `ABA_Matrix`,
`ArcAdmittanceMatrix` (and `get_ref_bus` on `VirtualMODF`).

## Reduction data

### `get_network_reduction_data`

```julia
get_network_reduction_data(mat::PowerNetworkMatrix) -> NetworkReductionData
```

Returns the [`NetworkReductionData`](@ref) associated with the matrix — the
record of which buses/arcs were merged or eliminated and how they map back to the
original network. Defined for every matrix type (it returns
`mat.network_reduction_data`). When no reduction was applied, an empty
`NetworkReductionData` is returned. This is the object queried by the reduction
accessors (`get_bus_reduction_map`, `get_removed_buses`, `get_reductions`, …);
see the [Network reduction reference](network_reduction.md).

!!! note "Serialization drops reduction data"
    
    A `PTDF` rehydrated from HDF5 via `from_hdf5` carries an **empty**
    `NetworkReductionData` — reduction information is not persisted. See
    [Serialization](serialization.md).

## System provenance

### `get_system_uuid`

```julia
get_system_uuid(mat::PowerNetworkMatrix) -> Union{Base.UUID, Nothing}
```

Returns the UUID of the `PSY.System` the matrix was constructed from, or
`nothing` for matrix types that do not track system origin. The default method
returns `nothing`; matrix types that store their origin — notably
[`VirtualPTDF`](@ref) and [`VirtualMODF`](@ref) — override it to return the
stored UUID. This UUID backs the internal consistency check that a matrix and a
system passed together actually originate from the same source. Source:
`src/PowerNetworkMatrix.jl`, `src/virtual_ptdf_calculations.jl`,
`src/virtual_modf_calculations.jl`.

## See also

  - [Matrix overview & indexing](network_matrices_overview.md) — the reference
    hub and the `A[row, column]` resolution rules.
  - [Matrix type reference](matrix_types.md) — constructors and keyword
    arguments.
  - [Network reduction reference](network_reduction.md) — the full
    `NetworkReductionData` accessor set.
  - [Full public API](public.md) — autodocs for the exported accessors.
