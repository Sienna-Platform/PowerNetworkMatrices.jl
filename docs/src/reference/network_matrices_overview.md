# Matrix overview & indexing

This page is the reference hub for the network matrices provided by
`PowerNetworkMatrices.jl` (PNM). It summarizes every matrix type, documents how
element indexing (`A[row, column]`) resolves, and links to the detailed
reference pages for construction, accessors, reduction, contingencies,
tolerances, solvers, and serialization.

All matrix types are concrete subtypes of the abstract supertype

```julia
abstract type PowerNetworkMatrix{T} <: AbstractArray{T, 2} end
```

(`src/PowerNetworkMatrix.jl`). Because they are `AbstractArray{T,2}` subtypes,
they support the standard array interface (`size`, `axes`, `getindex`,
iteration), but indexing is overloaded so that rows and columns are addressed by
domain identifiers (bus numbers and arc tuples) rather than by raw integer
positions. `LinearIndices` and `CartesianIndices` are intentionally disabled.

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

| Matrix                        | Rows        | Columns     | Storage          | Represents                                                              |
|:----------------------------- |:----------- |:----------- |:---------------- |:----------------------------------------------------------------------- |
| [`IncidenceMatrix`](@ref)     | arc tuples  | bus numbers | sparse           | signed bus–arc topology (`+1` from-bus, `-1` to-bus)                    |
| [`AdjacencyMatrix`](@ref)     | bus numbers | bus numbers | sparse           | signed bus–bus connectivity (Ybus sparsity pattern)                     |
| [`Ybus`](@ref)                | bus numbers | bus numbers | sparse (complex) | nodal admittance (topology + electrical parameters)                     |
| [`ArcAdmittanceMatrix`](@ref) | arc tuples  | bus numbers | sparse (complex) | off-diagonal Ybus entries; built as part of `Ybus`                      |
| [`BA_Matrix`](@ref)           | bus numbers | arc tuples  | sparse           | incidence weighted by branch susceptance                                |
| [`ABA_Matrix`](@ref)          | bus numbers | bus numbers | sparse           | ``A \cdot B \cdot A^\top`` DC susceptance matrix; optionally factorized |
| [`PTDF`](@ref)                | arc tuples  | bus numbers | dense            | power transfer distribution factors                                     |
| [`LODF`](@ref)                | arc tuples  | arc tuples  | dense            | line outage distribution factors                                        |
| [`VirtualPTDF`](@ref)         | arc tuples  | bus numbers | virtual          | lazy per-row PTDF                                                       |
| [`VirtualLODF`](@ref)         | arc tuples  | arc tuples  | virtual          | lazy per-row LODF                                                       |
| [`VirtualMODF`](@ref)         | arc tuples  | bus numbers | virtual          | post-modification / post-contingency PTDF rows                          |

Notes on the taxonomy:

  - `Ybus` is complex-valued (`ComplexF64`); all other numeric matrices are
    real (`Float64`). `IncidenceMatrix` / `AdjacencyMatrix` store signed `Int8`
    topology.
  - `PTDF` and `LODF` store their data **transposed** internally; `getindex`
    and [`get_ptdf_data`](accessors.md) / [`get_lodf_data`](accessors.md) hide
    this so callers always see the standard `(row, column)` orientation.
  - There is **no dense `MODF` type** — only [`VirtualMODF`](@ref). Contrast
    with PTDF/LODF, which have both dense and virtual forms.
  - `ArcAdmittanceMatrix` is produced as a byproduct of building `Ybus` (via a
    construction keyword) rather than being independently constructed by typical
    users.

Full constructor signatures, keyword arguments, and concrete type aliases
(`DC_PTDF_Matrix`, `DC_ABA_Matrix_Factorized`, `AC_Ybus_Matrix`, …) are on the
[Matrix type reference](matrix_types.md).

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
    
    A `PSY.ACBranch` component **cannot** be passed as an index — doing so raises
    a `KeyError`. Although `Base.to_index(::PSY.ACBranch)` is defined (returning
    the branch's arc tuple), the matrix `getindex` path routes only `PSY.ACBus`
    and `PSY.Arc` through `Base.to_index`; branch components are not converted.
    Index a branch by its **arc tuple** (`get_arc_tuple(branch, ...)` / the
    branch's `PSY.Arc`), or, for `PTDF`/`LODF`/`VirtualPTDF`, by its **name
    string**.

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
    identifiers with `PNM.get_axes(A)` (see the [Accessor
    reference](accessors.md)).

## Reading axes, lookups, and raw data

The identifier lists, lookup dictionaries, reference buses, and raw backing
array are exposed through accessor functions — `get_axes`, `get_lookup`,
`get_bus_axis`, `get_arc_axis`, `get_ref_bus`, `get_data`, `get_ptdf_data`,
`get_lodf_data`, and others. These are documented in full on the
[Accessor function reference](accessors.md).

```julia
PNM.get_axes(ptdf)     # (bus-number vector, arc-tuple vector)
PNM.get_lookup(ptdf)   # (bus lookup Dict, arc lookup Dict)
PNM.get_data(ybus)     # raw SparseMatrixCSC
```

## Reference map

This overview is the entry point. Detailed reference lives on the sibling pages:

  - [Matrix type reference](matrix_types.md) — constructor signatures, keyword
    arguments, and concrete type aliases for every matrix type.
  - [Accessor functions](accessors.md) — axes, lookups, reference buses, data
    extraction, reduction-data and system-UUID getters.
  - [Network reduction reference](network_reduction.md) — the
    `NetworkReduction` spec types and `NetworkReductionData`.
  - [Aggregated-branch types](aggregated_branches.md) — series/parallel/
    three-winding equivalents and their rating strategies.
  - [Contingency & modification types](contingencies.md) — `ArcModification`,
    `ShuntModification`, `NetworkModification`, `ContingencySpec`, Woodbury
    tooling.
  - [Tolerance & solver settings](tolerance_and_solvers.md) — `AutoTolerance`,
    sparsification cutoffs, and the linear-solver backends.
  - [Serialization](serialization.md) — HDF5 persistence (PTDF only).
  - [Full public API](public.md) — the curated autodocs for every exported
    symbol.
  - [Internals](internals.md) — the KLU and Accelerate solver submodules.
