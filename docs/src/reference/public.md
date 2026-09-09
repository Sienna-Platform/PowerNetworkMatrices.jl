# Public API Reference

```@meta
CurrentModule = PowerNetworkMatrices
```

## Matrix types

Dense, virtual, and structural network matrices. See
[Matrix Overview](network_matrices_overview.md) for the taxonomy and indexing
rules.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    PTDF, LODF, VirtualPTDF, VirtualLODF, VirtualMODF,
    Ybus, ArcAdmittanceMatrix, BA_Matrix, ABA_Matrix,
    IncidenceMatrix, AdjacencyMatrix,
)
```

## Network reductions

Reduction specifications, the reduction-data container, and its accessors. See the
[`NetworkReduction`](@ref) docstring for the `network_reductions` keyword and rules.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    NetworkReduction, RadialReduction, DegreeTwoReduction, WardReduction,
    NetworkReductionData,
    get_bus_reduction_map, get_network_reduction_data,
    get_reductions, get_ward_reduction,
)
```

## Aggregated-branch ratings

Rating-aggregation strategies for equivalent branches produced by network
reduction.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    get_sum_of_max_rating,
    get_single_element_contingency_rating,
    get_impedance_averaged_rating,
)
```

## Contingencies & modifications

Modification and contingency specification types, Ybus-delta application, and the
Woodbury-based post-contingency PTDF update. See the
[contingencies how-to](../how_to_guides/generated_contingencies.md) for the type model
and worked examples.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    ArcModification, ShuntModification, NetworkModification, ContingencySpec,
    apply_ybus_modification, compute_ybus_delta,
    apply_woodbury_correction, compute_woodbury_factors,
    get_post_modification_ptdf_row, get_registered_contingencies,
)
```

## Solvers & tolerance

Sparsification tolerance and factorization controls. See
[How to Choose a Linear Solver](../how_to_guides/generated_choose_linear_solver.md) for
the solver backends.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    AutoTolerance, discover_data_precision,
    factorize, is_factorized,
)
```

## Serialization

HDF5 persistence for `PTDF` (only).

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    to_hdf5, from_hdf5, get_system_uuid,
)
```

## Axes, lookups, and reference buses

Read a matrix's structure: the identifier vectors for each dimension, the
dictionaries mapping those identifiers to integer positions, and the reference
(slack) buses held fixed when the matrix was built. See
[Accessors: axes, lookups, and data](@ref) for which dimension each selects per
matrix type.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    get_axes, get_lookup,
    get_bus_axis, get_arc_axis, get_bus_lookup, get_arc_lookup,
    get_ref_bus, get_ref_bus_position,
)
```

## Data accessors

Extract the underlying numeric data from computed matrices. The raw backing array is
reached as `PowerNetworkMatrices.get_data`, which is deliberately not exported because
`PowerSystems.get_data` claims the same name.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    get_ptdf_data, get_lodf_data, get_partial_lodf_row,
)
```

## Cache management

Control and reset the virtual-matrix row caches.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    clear_caches!, clear_all_caches!,
)
```

## Connectivity

Island detection and connectivity validation.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
Private = false
Order = [:type, :constant, :function, :macro]
Filter = t -> t in (
    find_subnetworks, validate_connectivity,
    depth_first_search, iterative_union_find,
)
```

## Internal (non-exported) symbols

The symbols below are **not** exported and are **not** part of the supported API.
They are documented only so the manual covers every docstring shipped in the main
module; they may change without notice. Internal solver submodules
(`KLUWrapper`, `AccelerateWrapper`) are documented on the
[Internals](internals.md) page.

```@autodocs
Modules = [PowerNetworkMatrices]
Public = false
Private = true
Order = [:type, :constant, :function, :macro]
```
