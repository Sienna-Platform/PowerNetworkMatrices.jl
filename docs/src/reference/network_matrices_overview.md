# Matrix overview & indexing

This page is the reference hub for the network matrices provided by
`PowerNetworkMatrices.jl` (PNM). It summarizes every matrix type, shows the shared
construction and indexing pattern, documents how element indexing (`A[row, column]`)
resolves, and documents the accessor functions used to read data, axes, lookups,
reference buses, reduction data, and provenance back out.

All matrix types are concrete subtypes of the abstract supertype
`PowerNetworkMatrix{T} <: AbstractArray{T, 2}` (`src/PowerNetworkMatrix.jl`). Because
they are `AbstractArray{T,2}` subtypes, they support the standard array interface
(`size`, `axes`, `getindex`, iteration), but indexing is overloaded so that rows and
columns are addressed by domain identifiers (bus numbers and arc tuples) rather than
by raw integer positions. `LinearIndices` and `CartesianIndices` are intentionally
disabled.

## Matrix taxonomy

Every matrix stores two structural fields:

  - **`axes`**: a 2-tuple of vectors listing the identifiers (arc tuples and/or
    bus numbers) for each dimension.
  - **`lookup`**: a 2-tuple of dictionaries mapping those identifiers to integer
    positions into the stored `data`.

The storage form is one of three kinds:

  - **Dense** — the full matrix is materialized as a `Matrix{Float64}` (or, for
    `Ybus`, a `SparseMatrixCSC`).
  - **Sparse** — stored as a `SparseMatrixCSC`.
  - **Virtual / lazy** — no full matrix is stored; rows are computed on demand
    and cached in an LRU row cache. Use these past large-system limits instead of
    the dense forms. Virtual matrices are not serializable.

| Matrix                        | Rows        | Columns     | Storage          | Represents                                                  |
|:----------------------------- |:----------- |:----------- |:---------------- |:----------------------------------------------------------- |
| [`IncidenceMatrix`](@ref)     | arc tuples  | bus numbers | sparse           | signed bus–arc topology (`+1` from-bus, `-1` to-bus)        |
| [`AdjacencyMatrix`](@ref)     | bus numbers | bus numbers | sparse           | signed bus–bus connectivity (Ybus sparsity pattern)         |
| [`Ybus`](@ref)                | bus numbers | bus numbers | sparse (complex) | nodal admittance (topology + electrical parameters)         |
| [`ArcAdmittanceMatrix`](@ref) | arc tuples  | bus numbers | sparse (complex) | off-diagonal Ybus entries; built as part of `Ybus`          |
| [`BA_Matrix`](@ref)           | bus numbers | arc tuples  | sparse           | incidence weighted by branch susceptance                    |
| [`ABA_Matrix`](@ref)          | bus numbers | bus numbers | sparse           | ``A B A^\top`` DC susceptance matrix; optionally factorized |
| [`PTDF`](@ref)                | arc tuples  | bus numbers | dense            | power transfer distribution factors                         |
| [`LODF`](@ref)                | arc tuples  | arc tuples  | dense            | line outage distribution factors                            |
| [`VirtualPTDF`](@ref)         | arc tuples  | bus numbers | virtual          | lazy per-row PTDF                                           |
| [`VirtualLODF`](@ref)         | arc tuples  | arc tuples  | virtual          | lazy per-row LODF                                           |
| [`VirtualMODF`](@ref)         | arc tuples  | bus numbers | virtual          | post-modification / post-contingency PTDF rows              |

Notes on the taxonomy:

  - `Ybus` is complex-valued (`ComplexF64`); all other numeric matrices are
    real (`Float64`). `IncidenceMatrix` / `AdjacencyMatrix` store signed `Int8`
    topology.
  - `PTDF` and `LODF` store their data **transposed** internally; `getindex`
    and [`get_ptdf_data`](@ref) / [`get_lodf_data`](@ref) hide
    this so callers always see the standard `(row, column)` orientation.
  - There is **no dense `MODF` type** — only [`VirtualMODF`](@ref). Contrast
    with PTDF/LODF, which have both dense and virtual forms.
  - `ArcAdmittanceMatrix` is produced as a byproduct of building `Ybus` (via a
    construction keyword) rather than being independently constructed by typical
    users.

Full constructor signatures, keyword arguments, and concrete type aliases
(`DC_PTDF_Matrix`, `DC_ABA_Matrix_Factorized`, `AC_Ybus_Matrix`, …) are on the
[Matrix type reference](matrix_types.md).

## Constructing matrices

Every matrix type is a constructor that takes the
[`System`](@extref PowerSystems.System) and returns the matrix object. The call is
identical across types — only the name changes:

```julia
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

ptdf = PNM.PTDF(sys)
lodf = PNM.LODF(sys)
ybus = PNM.Ybus(sys)
aba = PNM.ABA_Matrix(sys)
```

The shared build-time keywords — `network_reductions`, `tol`, `linear_solver`,
`dist_slack` — work on every constructor that accepts them; each has its own how-to.
The lazy [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref) / [`VirtualMODF`](@ref) forms
build and index exactly like their materialized counterparts — swap the type name;
they compute rows on demand and cache them instead of storing the whole matrix.

Some constructors also accept **already-built matrices** instead of a
[`System`](@extref PowerSystems.System), so shared intermediates (`Ybus`, incidence,
BA) are computed once and reused. See
[How to Build Multiple Matrices Without Repeating Work](@ref).

## Arc-tuple indexing

Matrices that involve branches identify each branch by an **arc tuple** — a
`Tuple{Int, Int}` of the form `(from_bus_number, to_bus_number)` giving the
directed connection between two buses. Arc tuples, rather than branch-name
strings, are the canonical branch identifier because they:

  - identify a network element compactly and unambiguously;
  - survive network reductions, where named branches may be merged or eliminated
    but the surviving equivalent arc keeps a well-defined endpoint pair;
  - match the mathematical formulation, in which a branch is defined by its two
    endpoint buses.

## How `A[row, column]` resolves

Indexing is fully overloaded on `PowerNetworkMatrix` (`src/PowerNetworkMatrix.jl`).
`A[row, column]` calls `to_index(A, row, column)`, which maps each supplied
identifier to an integer position through the per-dimension `lookup` dictionary
(via the internal `lookup_index` helper), then reads the underlying `data`.

The accepted element types for `row` and `column`, and how each resolves:

| Index value                            | Resolves via                                                                                                                          | Supported                      |
|:-------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| `Int` (bus number)                     | direct `lookup[i]`                                                                                                                    | ✅                              |
| arc tuple `(from, to)::Tuple{Int,Int}` | direct `lookup[i]`                                                                                                                    | ✅                              |
| `PSY.ACBus`                            | `lookup_index` specialization → `Base.to_index(bus) = get_number(bus)`                                                                | ✅                              |
| `PSY.Arc`                              | `lookup_index` specialization → `Base.to_index(arc) = get_arc_tuple(arc)`                                                             | ✅                              |
| branch-name `String`                   | dedicated `getindex` on `PTDF` / `LODF` / `VirtualPTDF` (maps name → arc via reduction data, applies parallel/aggregation multiplier) | ✅ (PTDF/LODF/VirtualPTDF only) |
| `Colon` (`:`)                          | returns the whole row/column                                                                                                          | ✅                              |
| `PowerNetworkMatrixKey`                | `A[k]` splats `k.I` back into `A[k.I...]`                                                                                             | ✅                              |
| raw `Int` position pair                | dense positional fast path (`A.data[…]`)                                                                                              | ✅                              |
| `PSY.ACBranch`                         | —                                                                                                                                     | ❌ raises `KeyError`            |

!!! warning "Branch objects are not directly indexable"
    
    A [`ACBranch`](@extref PowerSystems.ACBranch) component **cannot** be passed as
    an index — doing so raises a `KeyError`. Although `Base.to_index(::PSY.ACBranch)`
    is defined (returning the branch's arc tuple), the matrix `getindex` path routes
    only [`ACBus`](@extref PowerSystems.ACBus) and [`Arc`](@extref PowerSystems.Arc)
    through `Base.to_index`; branch components are not converted. Index a branch by
    its **arc tuple** (the branch's [`Arc`](@extref PowerSystems.Arc)), or, for
    [`PTDF`](@ref)/[`LODF`](@ref)/[`VirtualPTDF`](@ref), by its **name string**.

Only `PTDF`, `LODF`, and `VirtualPTDF` accept branch-name `String` indices;
`LODF` requires a `String` for both dimensions when using names. Name indexing
maps the name to an arc tuple through the network reduction data and multiplies
by the appropriate parallel/aggregation factor, so it is retained for backward
compatibility but is slower and less direct than arc-tuple indexing.

### Examples

```julia
using PowerNetworkMatrices
import PowerSystems as PSY
import PowerSystemCaseBuilder as PSB
import PowerNetworkMatrices as PNM

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
ptdf = PTDF(sys)

# By bus number and arc tuple (canonical):
ptdf[(2, 3), 1]

# By PSY component objects:
bus1 = first(b for b in PSY.get_components(PSY.ACBus, sys) if PSY.get_number(b) == 1)
branch = first(PSY.get_components(PSY.ACBranch, sys))
ptdf[PSY.get_arc(branch), bus1]        # PSY.Arc row, PSY.ACBus column

# By branch name (PTDF/LODF/VirtualPTDF only):
ptdf[PSY.get_name(branch), 1]

# NOT allowed — raises KeyError:
# ptdf[branch, 1]                      # a PSY.ACBranch object

# Whole row / column with a Colon:
ptdf[:, 1]                             # column for bus 1
```

The `Ybus` accepts bus numbers or `PSY.ACBus` objects on both dimensions:

```julia
ybus = Ybus(sys)
ybus[3, 3]
ybus[PSY.get_number(bus1), PSY.get_number(bus1)]
```

!!! note "Reduced arcs are not indexable"
    
    When network reductions (e.g. `RadialReduction`, `DegreeTwoReduction`) are
    applied, eliminated branches are absent from the matrix. Indexing with an arc
    tuple that was reduced away raises an error. Inspect the surviving
    identifiers with `PNM.get_axes(A)` (see
    [Accessors: axes, lookups, and data](@ref)).

## Accessors: axes, lookups, and data

The functions below read structural and numeric data from any matrix — the backing
array, axes, lookup dictionaries, reference buses, reduction data, and system
provenance. Exported accessors are documented in full on the
[Full public API](public.md); the internal helpers are reached through the module
prefix (commonly aliased `PNM.`) and are documented here.

  - **Exported:** [`get_ptdf_data`](@ref), [`get_lodf_data`](@ref),
    [`get_partial_lodf_row`](@ref), [`get_network_reduction_data`](@ref),
    [`get_system_uuid`](@ref).
  - **Internal (call via `PNM.`):** `get_data`, `get_axes`, `get_lookup`,
    `get_bus_axis`, `get_arc_axis`, `get_bus_lookup`, `get_arc_lookup`,
    `get_ref_bus`, `get_ref_bus_position`.

### Data extraction

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

```julia
PNM.get_axes(ptdf)     # (bus-number vector, arc-tuple vector)
PNM.get_lookup(ptdf)   # (bus lookup Dict, arc lookup Dict)
PNM.get_data(ybus)     # raw SparseMatrixCSC
```

### Axes and lookups

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

### Reference buses

`PNM.get_ref_bus(mat)` returns the sorted reference (slack) bus numbers — one per
electrical island — and `PNM.get_ref_bus_position(mat)` their integer positions in
the bus dimension. Together they identify the slack bus(es) held fixed when the
matrix was built, which matters for interpreting [`PTDF`](@ref) columns and for
reduction/contingency math. Defined for the distribution-factor, incidence,
adjacency, BA/ABA, and arc-admittance matrices.

### Reduction data

**[`get_network_reduction_data`](@ref)** returns the [`NetworkReductionData`](@ref)
for the matrix — which buses/arcs were merged or eliminated and how they map back
(empty when no reduction was applied). This is the object queried by the reduction
accessors (`get_bus_reduction_map`, `get_removed_buses`, `get_reductions`, …); its
fields and accessors are documented on the [`NetworkReductionData`](@ref) docstring.

!!! note "Serialization drops reduction data"
    
    A [`PTDF`](@ref) rehydrated from HDF5 via [`from_hdf5`](@ref) carries an
    **empty** [`NetworkReductionData`](@ref). See the [`to_hdf5`](@ref) docstring.

### System provenance

**[`get_system_uuid`](@ref)** returns the UUID of the
[`System`](@extref PowerSystems.System) the matrix was built from, or `nothing` for
types that do not track origin. [`VirtualPTDF`](@ref) and [`VirtualMODF`](@ref)
store it; it backs the consistency check that a matrix and a system passed together
share a source.

## Reference map

This overview is the entry point. Detailed reference lives on the sibling pages:

  - [Matrix type reference](matrix_types.md) — constructor signatures, keyword
    arguments, and concrete type aliases for every matrix type.
  - [How to Diagnose a Disconnected Network](@ref) — testing whether the network is
    connected and enumerating electrical islands.
  - [How to Define and Apply Contingencies](@ref) — `ArcModification`,
    `ShuntModification`, `NetworkModification`, `ContingencySpec`, and the Woodbury
    tooling.
  - [`AutoTolerance`](@ref) and [How to Choose a Linear Solver](@ref) — the
    sparsification `tol` and the linear-solver backends.
  - [`to_hdf5`](@ref) / [`from_hdf5`](@ref) — HDF5 persistence (PTDF only).
  - [Full public API](public.md) — the curated autodocs for every exported
    symbol, including the reduction specs, `NetworkReductionData`, and the
    aggregated-branch rating functions.
  - [Internals](internals.md) — the KLU and Accelerate solver submodules.

```
```
