# # How to Define and Apply Contingencies

# This guide shows how to compute post-contingency PTDF rows with
# [`VirtualMODF`](@ref) — the lazy Multiple Outage Distribution Factor matrix.
# You will attach outages to a system, let them auto-register, query monitored
# arcs under a contingency, and build manual modifications when you need full
# control.

# !!! note
#     There is **no dense `MODF` type**. Post-contingency factors are only
#     available through `VirtualMODF`, which computes rows on demand via the
#     Woodbury identity. See [Flowgate Methodology](@ref) for the theory.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` and `PowerSystems.jl` installed
#   - A power system model

import PowerNetworkMatrices as PNM
import PowerSystems as PSY
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Attach Outages to the System

# Contingencies are defined as `PSY.Outage` supplemental attributes on the
# components they trip. When a contingency only needs to *exist*, use a
# `PSY.FixedForcedOutage` with `outage_status = 1.0` (outaged) and attach it to
# each branch:

for branch in PSY.get_components(PSY.ACTransmission, sys)
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    PSY.add_supplemental_attribute!(sys, branch, outage)
end

# ## Build the VirtualMODF

# Registration is **automatic**: the constructor scans the system for `PSY.Outage`
# attributes and resolves each into a [`ContingencySpec`](@ref). There is no
# public `register_contingency` — construct the matrix from a system that already
# carries its outages.

vmodf = PNM.VirtualMODF(sys)

# Inspect what was registered with [`get_registered_contingencies`](@ref). It
# returns a `Dict{UUID, ContingencySpec}` keyed by the source outage's UUID:

contingencies = PNM.get_registered_contingencies(vmodf)

# ## Query a Monitored Arc Under a Contingency

# Index the matrix as `vmodf[monitored_arc, spec]`. The monitored arc is an arc
# tuple `(from, to)` (or its integer index); the returned value is the full
# post-contingency PTDF row for that arc — one sensitivity per bus.

# Pick a monitored arc and a registered contingency:

monitored_arc = first(PNM.get_arc_axis(vmodf));
ctg = first(values(contingencies));

# The spec argument accepts three equivalent forms. By [`ContingencySpec`](@ref):

# ```julia
# vmodf[monitored_arc, ctg]
# ```

# By the underlying [`NetworkModification`](@ref):

# ```julia
# vmodf[monitored_arc, ctg.modification]
# ```

# By the original `PSY.Outage` (its UUID must be registered):

# ```julia
# branch = first(PSY.get_components(PSY.ACTransmission, sys))
# outage = first(PSY.get_supplemental_attributes(branch))
# vmodf[monitored_arc, outage]
# ```

# All three resolve to the same `NetworkModification` and share the cached
# Woodbury factors, so repeated queries for the same contingency across different
# monitored arcs reuse work.

# ## Build a Modification Manually

# When you want a contingency that is not backed by a `PSY.Outage`, build a
# [`NetworkModification`](@ref) directly. The simplest path is the convenience
# constructor that outages an entire arc by bus pair — it looks up the arc's
# susceptance and populates the deltas for you:

# ```julia
# mod = PNM.NetworkModification(vmodf, (1, 4))
# vmodf[monitored_arc, mod]
# ```

# For full control, assemble the low-level building blocks. An
# [`ArcModification`](@ref) is a susceptance change on one arc (`delta_b` negative
# for an outage); a [`ShuntModification`](@ref) is an admittance change on one bus.
# Both are indexed by their **integer** position in the matrix:

# ```julia
# arc_index = PNM.get_arc_lookup(vmodf)[(1, 4)]
# arc_mod = PNM.ArcModification(arc_index, -5.0)   # Δb removes the arc's susceptance
#
# bus_index = PNM.get_bus_lookup(vmodf)[3]
# shunt_mod = PNM.ShuntModification(bus_index, ComplexF32(-0.1im))
#
# # Combine arc and shunt changes into one modification (label, arcs, shunts, islanding)
# custom = PNM.NetworkModification("arc_and_shunt", [arc_mod], [shunt_mod], false)
# vmodf[monitored_arc, custom]
# ```

# Prefer the convenience constructors (`NetworkModification(matrix, arc)` or
# `NetworkModification(matrix, branch)`) — they compute physically consistent
# `delta_b` and Pi-model deltas from the network data, whereas hand-built
# `ArcModification` values are your responsibility to get right.

# ## One-Shot Post-Modification Rows from a VirtualPTDF

# If you already hold a [`VirtualPTDF`](@ref) and want a single post-modification
# row without registering contingencies, use
# [`get_post_modification_ptdf_row`](@ref). It applies a `NetworkModification`
# through the same Woodbury correction:

# ```julia
# vptdf = PNM.VirtualPTDF(sys)
# mod = PNM.NetworkModification(vptdf, (1, 4))
# row = PNM.get_post_modification_ptdf_row(vptdf, monitored_arc, mod)
#
# # Equivalent indexing form
# vptdf[monitored_arc, mod]
# ```

# This function does **no caching** — each call recomputes. When querying many
# monitored arcs for the *same* modification, precompute once with
# `compute_woodbury_factors` and reuse via `apply_woodbury_correction`.

# ## Contingencies and Network Reduction

# If you build the `VirtualMODF` with `network_reductions`, any branch that a
# contingency outages or monitors must survive every reduction step. Outage and
# monitored-component buses are auto-protected from reduction. Declare monitored
# branches on the outage so their buses are kept:

# ```julia
# monitored_line = PSY.get_component(PSY.ACTransmission, sys, "2")
# PSY.set_monitored_components!(outage, [monitored_line])
# ```

# Querying a monitored arc that was reduced away raises a clear error rather than
# silently returning the base row.

# ## See Also
#
#   - [Contingency & Modification Types](@ref) — reference for the value types
#   - [Flowgate Methodology](@ref) — the Woodbury post-contingency theory
