# Connectivity & islands

Tools for testing whether a system's network is fully connected and for enumerating
its electrical islands (subnetworks). A singular `ABA` matrix or a failed DC power
flow is often just a disconnected network — see
[How to Diagnose a Disconnected Network](@ref) for the diagnostic workflow and for
choosing between the union-find and depth-first traversals.

## Functions

  - [`validate_connectivity`](@ref) — `true` when the whole system forms a single
    connected component. Accepts a [`System`](@extref PowerSystems.System) or an
    already-built [`AdjacencyMatrix`](@ref) / [`Ybus`](@ref).
  - [`find_subnetworks`](@ref) — a `Dict` mapping each subnetwork's reference bus to
    the set of bus numbers in that island (a connected system yields one entry).
    Takes the same inputs, or the lower-level
    `find_subnetworks(M, bus_numbers; subnetwork_algorithm)` over a sparse
    connectivity matrix.
  - [`iterative_union_find`](@ref) / [`depth_first_search`](@ref) — the two
    `subnetwork_algorithm` strategies; both yield the same island decomposition.

The `subnetwork_algorithm` keyword threads through the matrix constructors (e.g.
[`ABA_Matrix`](@ref)), so islands are detected the same way at build time as by an
explicit [`find_subnetworks`](@ref) call. Per-island axes let the matrices index
transparently across electrical islands.

Full signatures and docstrings are on the [public API page](public.md).

## See also

  - [How to Diagnose a Disconnected Network](@ref) — the diagnostic recipe and
    traversal-algorithm guidance.
  - [Matrix overview & indexing](network_matrices_overview.md) — the
    [`AdjacencyMatrix`](@ref) and [`Ybus`](@ref) graphs these checks traverse.
  - [Network reduction theory](../explanation/network_reduction_theory.md) — the
    susceptance-vs-admittance graph subtlety that can fragment `ABA`.
