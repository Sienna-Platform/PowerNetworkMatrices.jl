# Network reduction reference

This page documents the network-reduction specification types, the
[`NetworkReductionData`](@ref) record produced when a reduction is applied, and
its accessor functions. It is a dry reference; for the theory of *why* and *when*
to reduce a network see
[Network reduction theory](../explanation/network_reduction_theory.md), and for
task recipes see the how-to guides
[Apply network reductions at construction](../how_to_guides/generated_apply_network_reductions.md)
and [Inspect a reduced network](../how_to_guides/generated_inspect_reduced_network.md).

Every symbol whose name is exported also appears in the
[full public API reference](public.md); internal helper types and accessors are
noted as such below.

## Supplying reductions

Reductions are passed to a matrix constructor **only** through the
`network_reductions` keyword, which takes a `Vector{NetworkReduction}`. There is
**no** `reduce_radial_branches` or `reduce_degree_two_branches` keyword on any
PowerNetworkMatrices constructor. (Those flags belong to PowerSimulations'
`NetworkModel` wrapper, which translates them into a `network_reductions`
vector.)

```julia
using PowerNetworkMatrices

reductions = [RadialReduction(), DegreeTwoReduction()]
ptdf = PTDF(sys; network_reductions = reductions)
```

Reductions are applied in vector order. Validation rules
(see the `ReductionContainer` section below) enforce that:

  - a given reduction type is applied at most once,
  - `WardReduction` must be the last reduction in the vector,
  - `ZeroImpedanceBranchReduction` may not be listed (it is auto-applied during
    `Ybus` construction),
  - applying `DegreeTwoReduction` before `RadialReduction` emits a warning
    (radial-first is usually more effective).

## Specification types

### `NetworkReduction` (abstract)

```julia
abstract type NetworkReduction end
```

Abstract supertype for all reduction algorithms. Defined in
`src/NetworkReduction.jl`. Two `NetworkReduction` values compare equal
(`==`) when they are of the same concrete type and all their fields are equal.
The concrete subtypes are `RadialReduction`, `DegreeTwoReduction`, and
`WardReduction`.

### Radial reduction

```julia
RadialReduction()
```

Eliminates leaf (degree-1) buses and their branches, walking through chains of
degree-2 buses toward the core network. Defined in `src/radial_reduction.jl` as
a field-less `@kwdef struct RadialReduction <: NetworkReduction`. It takes no
constructor arguments. Buses that must be preserved from elimination are
supplied separately through `Ybus(sys; irreducible_buses = ...)`, not through
the spec.

### Degree-two reduction

```julia
DegreeTwoReduction(; reduce_reactive_power_injectors = true)
```

Folds chains of degree-2 buses into equivalent series branches
([`BranchesSeries`](aggregated_branches.md)). Defined in
`src/degree_two_reduction.jl`.

Fields:

  - `reduce_reactive_power_injectors::Bool = true` — when `true`, buses whose
    only injectors support reactive power (e.g. a `SynchronousCondenser` or a
    purely susceptive `FixedAdmittance`) are treated as reduction candidates.
    When `false`, such reactive-only injector hosts are retained. Buses hosting
    an active-power injector are always kept. Capability is read from the
    PowerSystems `supports_active_power` / `supports_reactive_power` traits.

Accessor: `get_reduce_reactive_power_injectors(nr::DegreeTwoReduction)`.

In addition to any user-supplied `irreducible_buses`, `DegreeTwoReduction`
protects system-derived buses: static-injection hosts, HVDC terminals, and
area-interchange / `TransmissionInterface` endpoints.

!!! note
    
    `reduce_reactive_power_injectors = true` is correct for DC analysis but is
    electrically wrong for AC power flow. AC consumers must set it to `false`.

### Ward reduction

```julia
WardReduction(study_buses::Vector{Int})
```

Eliminates external buses via Ward equivalencing while preserving the electrical
behavior seen from the study area. External buses are mapped to boundary buses by
impedance proximity, and equivalent admittances / arc impedances are computed and
injected as `PSY.FixedAdmittance` and `PSY.GenericArcImpedance` objects. Defined
in `src/ward_reduction.jl`.

Fields:

  - `study_buses::Vector{Int}` — the bus numbers to retain in the reduced network.

Accessor: `get_study_buses(nr::WardReduction)`.

`WardReduction` must be the last reduction applied. If no boundary buses are
found between the study and external areas, all external buses are mapped to the
first reference study bus and an error is logged.

```julia
reduction = WardReduction([101, 102, 103])
ybus = Ybus(system; network_reductions = NetworkReduction[reduction])
```

## `NetworkReductionData`

```julia
NetworkReductionData
```

Mutable record holding all mappings and metadata describing how buses and
branches were combined or eliminated. Defined in `src/NetworkReductionData.jl`.
Every constructed matrix carries one; retrieve it with
`get_network_reduction_data(matrix)`. A matrix built without reductions carries
an empty `NetworkReductionData` (`isempty` returns `true`).

!!! note
    
    Serialization (`to_hdf5` / `from_hdf5`) does **not** persist reduction data.
    A `PTDF` rehydrated from HDF5 is reconstructed with an empty
    `NetworkReductionData`.

### Fields

Selected fields (see the struct docstring for the complete list):

| Field                          | Type                                                   | Meaning                                               |
|:------------------------------ |:------------------------------------------------------ |:----------------------------------------------------- |
| `irreducible_buses`            | `Set{Int}`                                             | Buses that cannot be reduced                          |
| `bus_reduction_map`            | `Dict{Int, Set{Int}}`                                  | Retained bus → set of eliminated buses folded into it |
| `reverse_bus_search_map`       | `Dict{Int, Int}`                                       | Eliminated bus → its surviving parent bus             |
| `direct_branch_map`            | `Dict{Tuple{Int,Int}, PSY.ACTransmission}`             | Arc → single retained branch                          |
| `parallel_branch_map`          | `Dict{Tuple{Int,Int}, AbstractBranchesParallel}`       | Arc → parallel-branch group                           |
| `series_branch_map`            | `Dict{Tuple{Int,Int}, BranchesSeries}`                 | Arc → series chain                                    |
| `transformer3W_map`            | `Dict{Tuple{Int,Int}, ThreeWindingTransformerWinding}` | Arc → three-winding winding                           |
| `removed_buses`                | `Set{Int}`                                             | Buses eliminated from the network                     |
| `removed_arcs`                 | `Set{Tuple{Int,Int}}`                                  | Arcs eliminated from the network                      |
| `merged_bus_pairs`             | `Dict{Int, Int}`                                       | Removed bus → surviving bus for zero-impedance merges |
| `removed_arc_to_surviving_bus` | `Dict{Tuple{Int,Int}, Int}`                            | Removed arc → the surviving bus it connected to       |
| `added_admittance_map`         | `Dict{Int, PSY.FixedAdmittance}`                       | Shunt admittances added during reduction (Ward)       |
| `added_arc_impedance_map`      | `Dict{Tuple{Int,Int}, PSY.GenericArcImpedance}`        | Equivalent arcs added during reduction (Ward)         |
| `reductions`                   | `ReductionContainer`                                   | Which reduction algorithms were applied               |

Each of the `*_branch_map` fields also has a `reverse_*` counterpart mapping a
branch component back to its arc tuple.

### Accessors

Field getters (each returns the like-named field):

```julia
get_irreducible_buses(nrd)
get_bus_reduction_map(nrd)
get_reverse_bus_search_map(nrd)
get_direct_branch_map(nrd)
get_reverse_direct_branch_map(nrd)
get_parallel_branch_map(nrd)
get_reverse_parallel_branch_map(nrd)
get_series_branch_map(nrd)
get_reverse_series_branch_map(nrd)
get_transformer3W_map(nrd)
get_reverse_transformer3W_map(nrd)
get_removed_buses(nrd)
get_removed_arcs(nrd)
get_removed_arc_to_surviving_bus(nrd)
get_added_admittance_map(nrd)
get_added_arc_impedance_map(nrd)
get_all_branch_maps_by_type(nrd)
get_reductions(nrd)
```

Reduction predicates (forwarded to the embedded `ReductionContainer`):

```julia
has_radial_reduction(nrd)      # -> Bool
has_degree_two_reduction(nrd)  # -> Bool
has_ward_reduction(nrd)        # -> Bool
has_filtered_branches(nrd)     # -> Bool, true if branch-type filters were applied
```

Bus remapping — resolve a bus number (or `PSY.ACBus`) to its surviving parent;
returns the input unchanged when the bus was not reduced:

```julia
get_mapped_bus_number(nrd, bus_number::Int)   # -> Int
get_mapped_bus_number(nrd, bus::PSY.ACBus)    # -> Int
```

Arc / axis queries:

```julia
get_arc_axis(nrd)                         # -> Vector{Tuple{Int,Int}} of surviving arcs
is_arc_in_series_map(nrd, arc)            # -> Bool
get_mapped_series_branch(nrd, arc)        # -> BranchesSeries (errors if arc absent)
```

Retained-branch queries:

```julia
get_retained_branches_names(nrd)  # -> Vector{String}; direct (one-to-one) branches only
get_ac_transmission_types(nrd)    # -> Set{DataType} of all retained branch types
```

Type-organized branch maps (populated lazily by
`populate_branch_maps_by_type!(nrd, filters = Dict())`):

```julia
get_component_to_reduction_name_map(nrd)
get_component_to_reduction_name_map(nrd, ::Type{T})   # T <: PSY.ACTransmission
get_name_to_arc_maps(nrd)
get_name_to_arc_map(nrd, ::Type{T})                   # T <: PSY.ACTransmission
```

!!! note
    
    A universal test for "did this bus survive the reduction?" is
    `bus in keys(get_bus_reduction_map(nrd))`. Radial and degree-two survivors
    also appear as keys; Ward survivors are the `study_buses`.

`get_retained_branches_names` returns names only for branches with a one-to-one
arc correspondence after reduction; it excludes parallel groups and branches
folded into a series chain. The `Base.show` method prints a summary of remapped
bus, branch-map, and removed-element counts.

## `ReductionContainer` (internal)

```julia
ReductionContainer
```

Internal record (not exported; defined in `src/ReductionContainer.jl`) that
tracks the user-supplied irreducible bus set plus which reduction specs have been
applied. Held in `NetworkReductionData.reductions`. Fields:

  - `user_irreducible_buses::Set{Int}`
  - `zero_impedance_reduction::Union{Nothing, ZeroImpedanceBranchReduction}`
  - `radial_reduction::Union{Nothing, RadialReduction}`
  - `degree_two_reduction::Union{Nothing, DegreeTwoReduction}`
  - `ward_reduction::Union{Nothing, WardReduction}`

Predicates and getters:

```julia
get_user_irreducible_buses(rc)
get_zero_impedance_reduction(rc)
has_zero_impedance_reduction(rc)  # -> Bool
has_radial_reduction(rc)          # -> Bool
has_degree_two_reduction(rc)      # -> Bool
has_ward_reduction(rc)            # -> Bool
```

`validate_reduction_type(::NetworkReduction, prior::ReductionContainer)` enforces
the ordering and uniqueness rules described under
[Supplying reductions](#supplying-reductions). `isempty(rc)` reports whether any
reduction algorithm has been applied (the configuration fields
`user_irreducible_buses` and `zero_impedance_reduction` do not count toward
emptiness).

## See also

  - [Matrix overview & indexing](network_matrices_overview.md) — the reference hub.
  - [Aggregated-branch types](aggregated_branches.md) — `BranchesSeries`,
    `BranchesParallel`, and the equivalent-parameter accessors referenced by the
    branch maps above.
  - [Full public API reference](public.md).
