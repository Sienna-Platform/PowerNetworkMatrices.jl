# Contingency & Modification Types

This page is the reference for the value types and functions that describe
topology changes and compute post-modification sensitivities — the machinery
behind [`VirtualMODF`](@ref) and the Woodbury post-contingency path. For a
task-oriented walkthrough see the
[contingencies how-to guide](../how_to_guides/generated_contingencies.md); for the
theory see the
[flowgate / post-contingency methodology](../explanation/flowgate_methodology.md).
Every symbol below is documented in full, with signatures, on the
[public API page](public.md).

## Overview

A network change is described in three layers, from the smallest unit to the
solver-ready form:

| Type                          | Represents                                                                                                             | Scope                      |
|:----------------------------- |:---------------------------------------------------------------------------------------------------------------------- |:-------------------------- |
| [`ArcModification`](@ref)     | A susceptance change on one aggregated arc, plus optional Ybus Pi-model deltas                                         | One arc                    |
| [`ShuntModification`](@ref)   | A diagonal admittance change on one bus                                                                                | One bus                    |
| [`NetworkModification`](@ref) | A canonical, [`System`](@extref PowerSystems.System)-independent bundle of arc and shunt changes plus islanding status | Whole modification         |
| [`ContingencySpec`](@ref)     | A [`NetworkModification`](@ref) tagged with the source [`Outage`](@extref PowerSystems.Outage) UUID                    | One registered contingency |
| `WoodburyFactors`             | Cached Woodbury intermediates for one modification, reusable across monitored arcs                                     | One modification           |

## Modification building blocks

  - **[`ArcModification`](@ref)** — a susceptance change (`delta_b`, negative for an
    outage) on one aggregated arc identified by integer `arc_index`, plus four
    optional Ybus Pi-model deltas (`ComplexF32`, the module constant `YBUS_ELTYPE`).
    A full outage sets `delta_b = -b_arc`; dropping one circuit of a double-circuit
    arc sets `delta_b = -b_circuit`.
  - **[`ShuntModification`](@ref)** — a diagonal admittance change (`delta_y`) on one
    bus, tracking shunt-component outages
    ([`FixedAdmittance`](@extref PowerSystems.FixedAdmittance),
    [`SwitchedAdmittance`](@extref PowerSystems.SwitchedAdmittance),
    [`StandardLoad`](@extref PowerSystems.StandardLoad)) that affect the
    [`Ybus`](@ref) but not the DC factors.

[`NetworkModification`](@ref) is the canonical representation: once built it holds
no reference to the source [`System`](@extref PowerSystems.System) and is used as
a cache key inside [`VirtualMODF`](@ref) (via custom `hash`/`==`). Its collections
are frozen into immutable tuples, and the `label` field is excluded from equality —
two physically identical modifications compare equal regardless of their names.

Convenience constructors resolve a change against a built matrix's reduction maps,
classifying each branch as direct, parallel, series, or 3-winding and computing the
correct `delta_b` and Pi-model deltas. The
[`ACTransmission`](@extref PowerSystems.ACTransmission) and
[`ThreeWindingTransformer`](@extref PowerSystems.ThreeWindingTransformer) forms
decompose accordingly; the [`Outage`](@extref PowerSystems.Outage) form resolves
associated components through the system (validating the system UUID matches),
groups series chains, and folds shunt-component outages into `shunt_modifications`.

!!! note
    
    Partial (non-full-outage) susceptance changes are supported only on direct and
    parallel arcs. Series-reduced arcs and 3-winding transformer windings accept
    only a full outage of the equivalent; anything else raises an error.

## Woodbury factors

`WoodburyFactors` caches the low-rank intermediates shared across monitored arcs
for one contingency
(`B_m⁻¹ = B_r⁻¹ − B_r⁻¹ U (A⁻¹ + Uᵀ B_r⁻¹ U)⁻¹ Uᵀ B_r⁻¹`). For `M ≤ 2` modified
arcs the inner `W_inv` is formed analytically; for `M > 2` via LU. When the
contingency islands the network it records the post-contingency component of each
bus, so entries for buses disconnected from the monitored arc are forced to exactly
zero. `WoodburyFactors` is not exported — obtain one from
[`compute_woodbury_factors`](@ref) and pass it to
[`apply_woodbury_correction`](@ref).

## Ybus modification functions

Bridge the DC-sensitivity representation ([`NetworkModification`](@ref)) to the AC
admittance representation ([`Ybus`](@ref)):

  - **`compute_ybus_delta`** — assemble the sparse ΔYbus
    ([`SparseMatrixCSC`](@extref Julia SparseArrays.SparseMatrixCSC), `n_bus × n_bus`)
    implied by a modification: four Pi-model entries per arc modification, one
    diagonal entry per shunt modification.
  - **[`apply_ybus_modification`](@ref)** — the modified admittance matrix,
    `ybus.data + compute_ybus_delta(ybus, mod)`.

## Post-modification PTDF rows

Compute post-modification/post-contingency [`PTDF`](@ref) rows from a
[`VirtualPTDF`](@ref) using the Woodbury identity:

  - **[`compute_woodbury_factors`](@ref)** — precompute the factors for a
    modification; reuse across many monitored arcs (the recommended path for
    optimization loops).
  - **[`apply_woodbury_correction`](@ref)** — the post-modification row for one
    monitored arc from precomputed factors.
  - **[`get_post_modification_ptdf_row`](@ref)** — one-shot convenience combining
    the two, with **no caching** (each call recomputes). Also accepts a
    `(sys, outage)` pair, building the modification from an
    [`Outage`](@extref PowerSystems.Outage) automatically.

A `monitored_arc` argument is always either an integer arc index or a
`Tuple{Int, Int}` bus pair.

!!! note
    
    Concurrent callers serialize on the per-cache solver lock (and, on KLU, the
    process-wide libklu lock). The correction is thread-safe but not parallelized.
    See the [concurrency explanation](../explanation/concurrency.md).

## Registered contingencies

Registration is **automatic** — there is no public `register_contingency`. When a
[`VirtualMODF`](@ref) is constructed with `automatically_register_outages = true`
(the default), every [`Outage`](@extref PowerSystems.Outage) supplemental attribute
on the source system is resolved to a [`ContingencySpec`](@ref) and cached, keyed by
the outage UUID. Inspect the set with [`get_registered_contingencies`](@ref).

Query a post-contingency row by indexing the [`VirtualMODF`](@ref) with a monitored
arc and a modification, spec, or outage:

```julia
vmodf[monitored_arc, mod]        # mod::NetworkModification
vmodf[monitored_arc, spec]       # spec::ContingencySpec
vmodf[monitored_arc, outage]     # outage::PSY.Outage (resolved by UUID)
```

## See also

  - [Contingencies how-to guide](../how_to_guides/generated_contingencies.md) — building and applying contingencies end to end.
  - [Flowgate / post-contingency methodology](../explanation/flowgate_methodology.md) — the Woodbury derivation and when to use it.
  - [Matrix type reference](matrix_types.md) — [`VirtualMODF`](@ref) and [`VirtualPTDF`](@ref).
  - [Public API](public.md) — full docstrings for every exported symbol.
