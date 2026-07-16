# Contingency & Modification Types

This page is the reference for the value types and functions that describe
topology changes to a network and compute post-modification sensitivities. They
form the machinery behind [`VirtualMODF`](@ref) and the Woodbury post-contingency
path. For a task-oriented walkthrough see the
[contingencies how-to guide](../how_to_guides/generated_contingencies.md); for the
underlying theory see the
[flowgate / post-contingency methodology](../explanation/flowgate_methodology.md).
Every symbol below is documented in full on the [public API page](public.md).

## Overview

A network change is described in three layers, from the smallest unit to the
solver-ready form:

| Type                          | Represents                                                                              | Scope                      |
|:----------------------------- |:--------------------------------------------------------------------------------------- |:-------------------------- |
| [`ArcModification`](@ref)     | A susceptance change on one aggregated arc, plus optional Ybus Pi-model deltas          | One arc                    |
| [`ShuntModification`](@ref)   | A diagonal admittance change on one bus                                                 | One bus                    |
| [`NetworkModification`](@ref) | A canonical, `System`-independent bundle of arc and shunt changes plus islanding status | Whole modification         |
| [`ContingencySpec`](@ref)     | A `NetworkModification` tagged with the source `PSY.Outage` UUID                        | One registered contingency |
| [`WoodburyFactors`](@ref)     | Cached Woodbury intermediates for one modification, reusable across monitored arcs      | One modification           |

`NetworkModification` is the canonical representation: once built it holds no
reference to the source `PSY.System`. It is used as a cache key inside
`VirtualMODF` (via custom `hash`/`==`), so its collections are stored as immutable
tuples and the `label` field is deliberately excluded from equality — two
physically identical modifications compare equal regardless of their names.

## Modification building blocks

### `ArcModification`

A susceptance change on a single aggregated arc, identified by its integer
`arc_index` in the network matrix, with optional Ybus Pi-model deltas.

```julia
struct ArcModification
    arc_index::Int
    delta_b::Float64        # change in susceptance (negative for an outage/reduction)
    delta_y11::ComplexF32   # Pi-model self-admittance change, from bus
    delta_y12::ComplexF32   # Pi-model mutual admittance change, from -> to
    delta_y21::ComplexF32   # Pi-model mutual admittance change, to -> from
    delta_y22::ComplexF32   # Pi-model self-admittance change, to bus
end
```

A full outage sets `delta_b = -b_arc`; dropping one circuit of a double-circuit
arc sets `delta_b = -b_circuit`. A backward-compatible constructor takes only the
susceptance change and zeroes all four Pi-model entries:

```julia
ArcModification(arc_index::Int, delta_b::Float64)
```

The `ComplexF32` element type of the Pi-model fields is the module constant
`YBUS_ELTYPE`.

### `ShuntModification`

A diagonal admittance change on a single bus, used to track shunt-component
outages (`PSY.FixedAdmittance`, `PSY.SwitchedAdmittance`, `PSY.StandardLoad`) that
affect the Ybus but not DC sensitivity factors.

```julia
struct ShuntModification
    bus_index::Int
    delta_y::ComplexF32     # change in shunt admittance (negative for an outage)
end
```

### `NetworkModification`

The canonical, self-contained description of a topology change. Arc and shunt
modification vectors are merged (entries on the same arc/bus are summed) and
frozen into tuples at construction.

```julia
struct NetworkModification
    label::String
    arc_modifications::Tuple{Vararg{ArcModification}}
    shunt_modifications::Tuple{Vararg{ShuntModification}}
    is_islanding::Bool
end
```

Direct constructors from prebuilt modification vectors:

```julia
NetworkModification(label::String, mods::Vector{ArcModification})
NetworkModification(label::String,
    mods::Vector{ArcModification},
    shunt_mods::Vector{ShuntModification},
    is_islanding::Bool)
```

Convenience constructors that resolve a change against a built matrix's network
reduction maps (classifying each branch as direct, parallel, series, or 3-winding
and computing the correct `delta_b` and Pi-model deltas):

```julia
NetworkModification(mat::PowerNetworkMatrix, arc::Tuple{Int, Int})
NetworkModification(mat::PowerNetworkMatrix, branch::PSY.ACTransmission)
NetworkModification(mat::PowerNetworkMatrix, branch::PSY.ThreeWindingTransformer)
NetworkModification(mat::PowerNetworkMatrix, sys::PSY.System, outage::PSY.Outage)
NetworkModification(ctg::ContingencySpec)
```

  - The `Tuple{Int, Int}` form builds a **full arc outage**, looking up the arc
    susceptance from `mat` and setting `Δb = -b_arc`.
  - The `PSY.ACTransmission` and `PSY.ThreeWindingTransformer` forms classify the
    component through the matrix's reduction maps. A `ThreeWindingTransformer` is
    decomposed into its three winding arcs; for a single-winding (partial) trip,
    build from a `ThreeWindingTransformerWinding` instead.
  - The `PSY.Outage` form resolves the outage's associated components through the
    system, groups series chains, and folds shunt-component outages into
    `shunt_modifications`. It validates that the system UUID matches `mat`.
  - The `ContingencySpec` form simply unwraps the stored modification.

!!! note
    
    Partial (non-full-outage) susceptance changes are supported only on direct and
    parallel arcs. Series-reduced arcs and 3-winding transformer windings accept
    only a full outage of the equivalent; anything else raises an error.

## `ContingencySpec`

A resolved, self-contained contingency: a [`NetworkModification`](@ref) plus the
UUID of the source `PSY.Outage` supplemental attribute it was built from. The UUID
is the cache key that links a registered contingency back to its outage.

```julia
struct ContingencySpec
    uuid::Base.UUID
    modification::NetworkModification
end
```

## `WoodburyFactors`

Cached Woodbury intermediates shared across monitored arcs for one contingency,
following the low-rank update
`B_m⁻¹ = B_r⁻¹ − B_r⁻¹ U (A⁻¹ + Uᵀ B_r⁻¹ U)⁻¹ Uᵀ B_r⁻¹`.

```julia
struct WoodburyFactors
    Z::Matrix{Float64}              # B⁻¹U, one column per modified arc (n_bus × M)
    W_inv::Matrix{Float64}         # pre-inverted W = (A⁻¹ + UᵀB⁻¹U)⁻¹ (M × M)
    arc_indices::Vector{Int}       # arc indices of modified arcs
    delta_b::Vector{Float64}       # susceptance change per modified arc
    is_islanding::Bool             # whether this contingency islands the network
    bus_island_labels::Vector{Int} # post-contingency component label per bus (empty unless islanding)
end
```

For `M ≤ 2` modified arcs `W_inv` is formed analytically; for `M > 2` it is
computed via LU factorization. When the contingency islands the network,
`bus_island_labels` records the post-contingency connected component of each bus
so that entries for buses disconnected from the monitored arc are forced to
exactly zero.

`WoodburyFactors` is not exported. Users obtain one from
[`compute_woodbury_factors`](@ref) and pass it to
[`apply_woodbury_correction`](@ref).

## Ybus modification functions

These bridge the DC-sensitivity representation (`NetworkModification`) to the AC
admittance representation (`Ybus`).

### `compute_ybus_delta`

```julia
compute_ybus_delta(ybus::Ybus, mod::NetworkModification) ->
    SparseMatrixCSC{YBUS_ELTYPE, Int}
```

Assemble the sparse ΔYbus matrix implied by `mod`. Arc modifications contribute
their four Pi-model entries at the `(from, to)` bus positions; shunt modifications
contribute a single diagonal entry each. Returns an `n_bus × n_bus` sparse matrix.

### `apply_ybus_modification`

```julia
apply_ybus_modification(ybus::Ybus, mod::NetworkModification) -> SparseMatrixCSC
```

Convenience wrapper that returns `ybus.data + compute_ybus_delta(ybus, mod)` — the
modified admittance matrix.

## Post-modification PTDF-row functions

These compute post-modification/post-contingency PTDF rows from a
[`VirtualPTDF`](@ref) using the Woodbury identity.

### `compute_woodbury_factors`

```julia
compute_woodbury_factors(vptdf::VirtualPTDF, mod::NetworkModification) -> WoodburyFactors
```

Precompute the [`WoodburyFactors`](@ref) for a modification. Reuse the result
across many monitored arcs — this is the recommended path for optimization loops
where factors are formed once per modification and many rows are queried.

### `apply_woodbury_correction`

```julia
apply_woodbury_correction(vptdf::VirtualPTDF, monitored_arc, wf::WoodburyFactors) ->
    Vector{Float64}
```

Compute the post-modification PTDF row for one monitored arc from precomputed
factors. `monitored_arc` may be an integer arc index or a `Tuple{Int, Int}` bus
pair.

### `get_post_modification_ptdf_row`

```julia
get_post_modification_ptdf_row(
    vptdf::VirtualPTDF,
    monitored_arc,
    mod::NetworkModification,
) -> Vector{Float64}
get_post_modification_ptdf_row(
    vptdf::VirtualPTDF,
    monitored_arc,
    sys::PSY.System,
    outage::PSY.Outage,
) -> Vector{Float64}
```

One-shot convenience: compute the post-modification row for a single monitored arc
in one call. Internally calls `compute_woodbury_factors` then
`apply_woodbury_correction`, with **no caching** — each call recomputes the
factors. When querying multiple monitored arcs for the same modification, use the
two-step API instead. `monitored_arc` accepts an integer index or a
`Tuple{Int, Int}` bus pair. The `(sys, outage)` form builds the
`NetworkModification` from a `PSY.Outage` automatically.

!!! note
    
    Concurrent callers of these functions serialize on the per-cache solver lock
    (and, on the KLU backend, the process-wide libklu lock). The correction is
    thread-safe but not parallelized. See the
    [concurrency explanation](../explanation/computational_considerations.md).

## Registered contingencies

Contingency registration is **automatic**. There is no public
`register_contingency` function. When a [`VirtualMODF`](@ref) is constructed with
`automatically_register_outages = true` (the default), every `PSY.Outage`
supplemental attribute attached to the source system is resolved to a
[`ContingencySpec`](@ref) and cached, keyed by the outage UUID.

### `get_registered_contingencies`

```julia
get_registered_contingencies(vmodf::VirtualMODF) -> Dict{Base.UUID, ContingencySpec}
```

Return the cached registrations for inspection. The keys are the source outage
UUIDs; the values are the resolved [`ContingencySpec`](@ref)s.

To query a post-contingency row, index the `VirtualMODF` with a monitored arc and
a modification/spec/outage:

```julia
vmodf[monitored_arc, mod]        # mod::NetworkModification
vmodf[monitored_arc, spec]       # spec::ContingencySpec
vmodf[monitored_arc, outage]     # outage::PSY.Outage (resolved by UUID)
```

where `monitored_arc` is an integer arc index or a `Tuple{Int, Int}` bus pair.

## See also

  - [Contingencies how-to guide](../how_to_guides/generated_contingencies.md) — building and applying contingencies end to end.
  - [Flowgate / post-contingency methodology](../explanation/flowgate_methodology.md) — the Woodbury derivation and when to use it.
  - [Matrix type reference](matrix_types.md) — [`VirtualMODF`](@ref) and [`VirtualPTDF`](@ref).
  - [Public API](public.md) — full docstrings for every exported symbol.
