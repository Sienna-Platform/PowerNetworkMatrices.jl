# Network reduction reference

This page documents the network-reduction specification types, the
[`NetworkReductionData`](@ref) record produced when a reduction is applied, and its
accessor functions. For the theory of *why* and *when* to reduce see
[Network reduction theory](../explanation/network_reduction_theory.md); for a task
recipe see
[Apply network reductions at construction](../how_to_guides/generated_apply_network_reductions.md).
Exported symbols are documented in full on the [public API page](public.md);
internal helpers (not on that page) are marked below and reached via `PNM.`.

## Supplying reductions

Reductions are passed to a matrix constructor **only** through the
`network_reductions` keyword, a `Vector{NetworkReduction}`. There is **no**
`reduce_radial_branches` / `reduce_degree_two_branches` keyword on any
PowerNetworkMatrices constructor (those flags belong to PowerSimulations'
`NetworkModel`, which translates them into a `network_reductions` vector).

```julia
reductions = [RadialReduction(), DegreeTwoReduction()]
ptdf = PTDF(sys; network_reductions = reductions)
```

Reductions are applied in vector order, subject to validation:

  - a given reduction type is applied at most once,
  - [`WardReduction`](@ref) must be last,
  - `ZeroImpedanceBranchReduction` may not be listed (it is auto-applied during
    [`Ybus`](@ref) construction),
  - [`DegreeTwoReduction`](@ref) before [`RadialReduction`](@ref) emits a warning
    (radial-first is usually more effective).

## Specification types

The abstract supertype is [`NetworkReduction`](@ref); two values compare equal when
same-type with equal fields. The concrete subtypes:

  - **[`RadialReduction`](@ref)** — eliminates leaf (degree-1) buses and their
    branches, walking chains of degree-2 buses toward the core. Field-less. Buses to
    preserve come from `Ybus(sys; irreducible_buses = ...)`, not the spec.
  - **[`DegreeTwoReduction`](@ref)** — folds chains of degree-2 buses into
    equivalent series branches ([`BranchesSeries`](aggregated_branches.md)). Its one
    field, `reduce_reactive_power_injectors::Bool = true`, treats buses whose only
    injectors support reactive power as reduction candidates; buses hosting an
    active-power injector are always kept.
  - **[`WardReduction`](@ref)** — eliminates external buses via Ward equivalencing,
    preserving the behavior seen from `study_buses::Vector{Int}`. Equivalent
    admittances / arc impedances are injected as
    [`FixedAdmittance`](@extref PowerSystems.FixedAdmittance) and
    `PSY.GenericArcImpedance` objects. Must be applied last; if no boundary buses
    exist, all external buses map to the first reference study bus and an error is
    logged.

!!! note "DegreeTwoReduction and AC"
    
    `reduce_reactive_power_injectors = true` is correct for DC but electrically
    wrong for AC power flow. AC consumers must set it to `false`.

## `NetworkReductionData`

A mutable record holding all mappings and metadata describing how buses and
branches were combined or eliminated. Every constructed matrix carries one;
retrieve it with [`get_network_reduction_data`](@ref). A matrix built without
reductions carries an empty [`NetworkReductionData`](@ref) (`isempty` is `true`).
Serialization does **not** persist it (see [Serialization](serialization.md)).

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

Each `*_branch_map` also has a `reverse_*` counterpart mapping a branch component
back to its arc tuple.

### Accessors

Field getters, each returning the like-named field (exported:
[`get_bus_reduction_map`](@ref), [`get_reductions`](@ref); the rest internal, via
`PNM.`): `get_irreducible_buses`, `get_reverse_bus_search_map`,
`get_direct_branch_map`, `get_parallel_branch_map`, `get_series_branch_map`,
`get_transformer3W_map` (plus each `reverse_*`), `get_removed_buses`,
`get_removed_arcs`, `get_removed_arc_to_surviving_bus`, `get_added_admittance_map`,
`get_added_arc_impedance_map`, `get_all_branch_maps_by_type`.

Reduction predicates (forwarded to the embedded `ReductionContainer`):
`has_radial_reduction`, `has_degree_two_reduction`, `has_ward_reduction`,
`has_filtered_branches`.

Other queries: `get_mapped_bus_number(nrd, bus)` resolves a bus number (or
[`ACBus`](@extref PowerSystems.ACBus)) to its surviving parent (unchanged if not
reduced); `get_arc_axis`, `is_arc_in_series_map`, `get_mapped_series_branch`;
`get_retained_branches_names` (direct one-to-one branches only),
`get_ac_transmission_types`; and the lazily-populated type-organized maps
`get_component_to_reduction_name_map` / `get_name_to_arc_map(s)`.

!!! note "Did this bus survive?"
    
    The universal test is `bus in keys(get_bus_reduction_map(nrd))`. Radial and
    degree-two survivors appear as keys; Ward survivors are the `study_buses`.

## `ReductionContainer` (internal)

An internal record (not exported) tracking the user-supplied irreducible bus set
and which reduction specs were applied; held in `NetworkReductionData.reductions`.
Its getters/predicates (`get_user_irreducible_buses`,
`get_zero_impedance_reduction`, `has_zero_impedance_reduction`, and the
`has_*_reduction` set) mirror those above.
`validate_reduction_type(::NetworkReduction, prior::ReductionContainer)` enforces
the ordering/uniqueness rules; `isempty(rc)` reports whether any reduction algorithm
has been applied (configuration fields do not count toward emptiness).

## See also

  - [Matrix overview & indexing](network_matrices_overview.md) — the reference hub.
  - [Aggregated-branch types](aggregated_branches.md) — `BranchesSeries`,
    `BranchesParallel`, and the equivalent-parameter accessors.
  - [Full public API reference](public.md).
