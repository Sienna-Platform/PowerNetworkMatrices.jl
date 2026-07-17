# Connectivity & islands

This page is the reference for testing whether a system's network is fully
connected and for enumerating its electrical islands (subnetworks). Both entry
points below are documented in full, with signatures, on the
[public API page](public.md).

A singular `ABA` matrix or a failed DC power flow is frequently just a
disconnected network. When you hit one, checking connectivity first localizes the
problem before you look at the numerics.

## Entry points

  - **[`validate_connectivity`](@ref)** — returns `true` when the whole system
    forms a single connected component, via a depth-first traversal of the
    [`Ybus`](@ref) graph. Accepts a [`System`](@extref PowerSystems.System) or an
    already-built matrix.
  - **[`find_subnetworks`](@ref)** — returns a `Dict` mapping each subnetwork's
    reference bus to the set of bus numbers in that island. A connected system
    yields a single entry. Also accepts a `System` or a built matrix.

Both accept a built [`AdjacencyMatrix`](@ref) or [`Ybus`](@ref) directly, so a
prior matrix is reused rather than rebuilt:

```julia
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

PNM.validate_connectivity(sys)   # true — one connected component
PNM.find_subnetworks(sys)        # Dict(ref_bus => Set(bus numbers))

adj = PNM.AdjacencyMatrix(sys)
PNM.validate_connectivity(adj)   # same answer, no rebuild
```

## Traversal algorithm

The lower-level method `find_subnetworks(M, bus_numbers; subnetwork_algorithm)`
operates on a sparse connectivity matrix and selects the graph-traversal strategy:

  - **[`iterative_union_find`](@ref)** (the default) — a union-find disjoint-set
    structure.
  - **[`depth_first_search`](@ref)** — recursive graph traversal.

```julia
bus_numbers = PNM.get_bus_axis(adj)
PNM.find_subnetworks(adj.data, bus_numbers; subnetwork_algorithm = PNM.iterative_union_find)
PNM.find_subnetworks(adj.data, bus_numbers; subnetwork_algorithm = PNM.depth_first_search)
```

Both return the same island decomposition; the choice is a performance detail, not
a correctness one. Each also warns for any islanded (degree-zero) bus it
encounters.

## Build-time consistency

The `subnetwork_algorithm` keyword is threaded through the matrix constructors
(e.g. [`ABA_Matrix`](@ref)), so islands are detected the same way at build time as
they are by an explicit [`find_subnetworks`](@ref) call. Per-island axes let the
matrices index transparently across electrical islands.

## See also

  - [Matrix type reference](matrix_types.md) — the [`AdjacencyMatrix`](@ref) and
    [`Ybus`](@ref) graphs these checks traverse.
  - [Accessor functions](accessors.md) — the axis and lookup getters used above.
  - [Public API](public.md) — full docstrings for [`validate_connectivity`](@ref),
    [`find_subnetworks`](@ref), [`iterative_union_find`](@ref), and
    [`depth_first_search`](@ref).
