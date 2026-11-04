# Matrix overview & indexing

This page is the reference hub for the matrix types: what each one stores and costs
to build, how element indexing (`A[row, column]`) resolves, and which accessors read
a matrix's data, axes, lookups, reference buses, reduction data, and provenance.

All matrix types are concrete subtypes of the abstract supertype
`PowerNetworkMatrix{T} <: AbstractArray{T, 2}` (`src/PowerNetworkMatrix.jl`). Because
they are `AbstractArray{T,2}` subtypes, they support the standard array interface
(`size`, `axes`, `getindex`, iteration), but indexing is overloaded so that rows and
columns are addressed by domain identifiers (bus numbers and arc tuples) rather than
by raw integer positions.

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

Build cost is for a network of ``N_b`` buses and ``N_a`` arcs.

| Matrix                        | Rows        | Columns     | Storage          | Build cost                     | Represents                                                  |
|:----------------------------- |:----------- |:----------- |:---------------- |:------------------------------ |:----------------------------------------------------------- |
| [`IncidenceMatrix`](@ref)     | arc tuples  | bus numbers | sparse           | ``O(N_a)``                     | signed bus–arc topology (`+1` from-bus, `-1` to-bus)        |
| [`AdjacencyMatrix`](@ref)     | bus numbers | bus numbers | sparse           | ``O(N_a)``                     | signed bus–bus connectivity (Ybus sparsity pattern)         |
| [`Ybus`](@ref)                | bus numbers | bus numbers | sparse (complex) | ``O(N_a)``                     | nodal admittance (topology + electrical parameters)         |
| [`ArcAdmittanceMatrix`](@ref) | arc tuples  | bus numbers | sparse (complex) | ``O(N_a)``, with `Ybus`        | off-diagonal Ybus entries; built as part of `Ybus`          |
| [`BA_Matrix`](@ref)           | bus numbers | arc tuples  | sparse           | ``O(N_a)``                     | ``B A``: incidence weighted by branch susceptance           |
| [`ABA_Matrix`](@ref)          | bus numbers | bus numbers | sparse           | ``O(N_a)``, plus factorization | ``A^\top B A`` DC susceptance matrix; optionally factorized |
| [`PTDF`](@ref)                | arc tuples  | bus numbers | dense            | ``O(N_b^3)``                   | power transfer distribution factors                         |
| [`LODF`](@ref)                | arc tuples  | arc tuples  | dense            | ``O(N_a \cdot N_b^2)``         | line outage distribution factors                            |
| [`VirtualPTDF`](@ref)         | arc tuples  | bus numbers | virtual          | one solve per row, on demand   | lazy per-row PTDF                                           |
| [`VirtualLODF`](@ref)         | arc tuples  | arc tuples  | virtual          | one solve per row, on demand   | lazy per-row LODF                                           |
| [`VirtualMODF`](@ref)         | arc tuples  | bus numbers | virtual          | one solve per row, on demand   | post-modification / post-contingency PTDF rows              |

Notes on the taxonomy:

  - `Ybus` is complex-valued (`YBUS_ELTYPE`, i.e. `ComplexF32`); all other numeric
    matrices are real (`Float64`). `IncidenceMatrix` / `AdjacencyMatrix` store signed
    `Int8` topology.
  - `PTDF` and `LODF` store their data **transposed** internally; `getindex`
    and [`get_ptdf_data`](@ref) / [`get_lodf_data`](@ref) hide
    this so callers always see the standard `(row, column)` orientation.
  - There is **no dense `MODF` type** — only [`VirtualMODF`](@ref). Contrast
    with PTDF/LODF, which have both dense and virtual forms.
  - `ArcAdmittanceMatrix` is produced as a byproduct of building `Ybus` (via a
    construction keyword) rather than being independently constructed by typical
    users.
  - The dense build costs are why [`PTDF`](@ref)/[`LODF`](@ref) are the matrices to
    avoid at scale; see [Computational Considerations](@ref) for the sparsity and
    sparsification trade-offs behind that.

Every type is constructed by calling it on a [`System`](@extref PowerSystems.System)
— `PTDF(sys)`, `Ybus(sys)`, and so on — and the lazy forms build and index exactly
like their materialized counterparts. Full constructor signatures and keyword
arguments are in the [public API reference](public.md); worked examples are in the
[Introduction](@ref) tutorial, and
[How to Build Multiple Matrices Without Repeating Work](@ref) covers passing
already-built matrices (`Ybus`, incidence, BA) to the constructors that accept them
so shared intermediates are computed once.

## Arc-tuple indexing

Matrices that involve branches identify each branch by an **arc tuple** — a
`Tuple{Int, Int}` of the form `(from_bus_number, to_bus_number)` giving the
directed connection between two buses. Arc tuples, rather than branch-name strings,
are the canonical branch identifier: they are compact and unambiguous, they match
the mathematical formulation in which a branch is defined by its two endpoint buses,
and they survive network reductions, where named branches may be merged or
eliminated but the surviving equivalent arc keeps a well-defined endpoint pair.

## How `A[row, column]` resolves

Indexing is fully overloaded on `PowerNetworkMatrix` (`src/PowerNetworkMatrix.jl`).
`A[row, column]` calls `to_index(A, row, column)`, which maps each supplied
identifier to an integer position through the per-dimension `lookup` dictionary
(via the internal `lookup_index` helper), then reads the underlying `data`.

The accepted element types for `row` and `column`, and how each resolves:

| Index value                                              | Resolves via                                                                                                                          | Supported                      |
|:-------------------------------------------------------- |:------------------------------------------------------------------------------------------------------------------------------------- |:------------------------------ |
| `Int` (bus number)                                       | direct `lookup[i]`                                                                                                                    | ✅                              |
| arc tuple `(from, to)::Tuple{Int,Int}`                   | direct `lookup[i]`                                                                                                                    | ✅                              |
| [`PowerSystems.ACBus`](@extref PowerSystems.ACBus)       | `lookup_index` specialization → `Base.to_index(bus) = get_number(bus)`                                                                | ✅                              |
| [`PowerSystems.Arc`](@extref PowerSystems.Arc)           | `lookup_index` specialization → `Base.to_index(arc) = get_arc_tuple(arc)`                                                             | ✅                              |
| branch-name `String`                                     | dedicated `getindex` on `PTDF` / `LODF` / `VirtualPTDF` (maps name → arc via reduction data, applies parallel/aggregation multiplier) | ✅ (PTDF/LODF/VirtualPTDF only) |
| `Colon` (`:`)                                            | returns the whole row/column                                                                                                          | ✅                              |
| `PowerNetworkMatrixKey`                                  | `A[k]` splats `k.I` back into `A[k.I...]`                                                                                             | ✅                              |
| raw `Int` position pair                                  | dense positional fast path (`A.data[…]`)                                                                                              | ✅                              |
| [`PowerSystems.ACBranch`](@extref PowerSystems.ACBranch) | —                                                                                                                                     | ❌ raises `KeyError`            |

Only [`PTDF`](@ref), [`LODF`](@ref), and [`VirtualPTDF`](@ref) accept branch-name
`String` indices, and [`LODF`](@ref) requires one on both dimensions. Name indexing
is retained for backward compatibility but is slower and less direct than arc-tuple
indexing.

!!! warning "Branch objects are not directly indexable"
    
    Passing an [`ACBranch`](@extref PowerSystems.ACBranch) component as an index
    raises a `KeyError`. Although `Base.to_index(::PowerSystems.ACBranch)` is defined
    (returning the branch's arc tuple), `getindex` routes only
    [`ACBus`](@extref PowerSystems.ACBus) and [`Arc`](@extref PowerSystems.Arc)
    through `Base.to_index`. Index a branch by its **arc tuple**, or by its **name
    string** on the three types above.

!!! note "Reduced arcs are not indexable"
    
    When network reductions (e.g. `RadialReduction`, `DegreeTwoReduction`) are
    applied, eliminated branches are absent from the matrix, and indexing with an arc
    tuple that was reduced away raises an error. Enumerate the surviving identifiers
    with [`get_axes`](@ref).

### Examples

````@example overview
using PowerNetworkMatrices
import PowerSystems
import PowerSystemCaseBuilder

sys = PowerSystemCaseBuilder.build_system(
    PowerSystemCaseBuilder.PSITestSystems,
    "c_sys5",
)

ptdf = PTDF(sys)
ybus = Ybus(sys)
nothing # hide
````

````@example overview
# By bus number and arc tuple (canonical):
ptdf[(2, 3), 1]
````

````@example overview
# By PowerSystems component objects: an Arc row and an ACBus column
bus1 = first(
    b for b in PowerSystems.get_components(PowerSystems.ACBus, sys) if
    PowerSystems.get_number(b) == 1
)
branch = first(PowerSystems.get_components(PowerSystems.ACBranch, sys))
ptdf[PowerSystems.get_arc(branch), bus1]
````

````@example overview
# By branch name (PTDF/LODF/VirtualPTDF only):
ptdf[PowerSystems.get_name(branch), 1]

# NOT allowed — raises KeyError:
# ptdf[branch, 1]                      # a PowerSystems.ACBranch object
````

````@example overview
# Whole row / column with a Colon:
ptdf[:, 1]                             # column for bus 1
````

## Axes and lookups

[`get_axes`](@ref) returns `mat.axes` and [`get_lookup`](@ref) returns `mat.lookup`,
each a 2-tuple ordered `(dimension 1, dimension 2)`. The axis vector lists
identifiers (bus numbers as `Int`, arcs as `Tuple{Int,Int}`) in position order; the
matching lookup maps each identifier back to its integer position in `data`. These
are the authoritative way to enumerate valid indices — especially after a reduction,
where some arcs/buses are no longer present — and are defined for every matrix type.

````@example overview
get_axes(ptdf)                        # (bus-number vector, arc-tuple vector)
````

The dimension-specific accessors [`get_bus_axis`](@ref) / [`get_arc_axis`](@ref) /
[`get_bus_lookup`](@ref) / [`get_arc_lookup`](@ref) select the correct dimension
without the caller knowing which index (1 or 2) is the bus or arc dimension for a
given matrix type. They are defined only for the dimensions a matrix actually has.

## See also

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
