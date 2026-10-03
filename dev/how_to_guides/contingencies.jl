# # How to Define and Apply Contingencies

# This guide shows how to compute post-contingency PTDF rows with
# [`VirtualMODF`](@ref) — the lazy Multiple Outage Distribution Factor matrix.
# You will attach outages to a system, let them auto-register, query monitored
# arcs under a contingency, and build manual modifications when you need full
# control.

# !!! note
#     There is **no dense `MODF` type**. Post-contingency factors are only
#     available through [`VirtualMODF`](@ref), which computes rows on demand via the
#     Woodbury identity. See [Flowgate Methodology](@ref) for the theory.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` and `PowerSystems.jl` installed
#   - A power system model

using PowerNetworkMatrices
import PowerSystems
import PowerSystemCaseBuilder

sys = PowerSystemCaseBuilder.build_system(
    PowerSystemCaseBuilder.PSITestSystems,
    "c_sys5",
);

# ## Attach Outages to the System

# Contingencies are defined as [`PowerSystems.Outage`](@extref PowerSystems.Outage)
# supplemental attributes on the components they trip. When a contingency only needs to
# *exist*, use a
# [`PowerSystems.FixedForcedOutage`](@extref PowerSystems.FixedForcedOutage) with
# `outage_status = 1.0` (outaged) and attach it to each branch:

for branch in PowerSystems.get_components(PowerSystems.ACTransmission, sys)
    outage = PowerSystems.FixedForcedOutage(; outage_status = 1.0)
    PowerSystems.add_supplemental_attribute!(sys, branch, outage)
end

# ## Build the VirtualMODF

# Registration is **automatic**: the [`VirtualMODF`](@ref) constructor scans the system
# for [`PowerSystems.Outage`](@extref PowerSystems.Outage) attributes and resolves each
# into a [`ContingencySpec`](@ref).

vmodf = VirtualMODF(sys)

# Inspect what was registered with [`get_registered_contingencies`](@ref). It
# returns a `Dict{UUID, ContingencySpec}` keyed by the source outage's UUID:

contingencies = get_registered_contingencies(vmodf)

# ## Query a Monitored Arc Under a Contingency

# Index the matrix as `vmodf[monitored_arc, spec]`. The monitored arc is an arc
# tuple `(from, to)` (or its integer index); the returned value is the full
# post-contingency PTDF row for that arc — one sensitivity per bus.

# Pick a monitored arc and outage a *different* arc — monitoring an element that
# the contingency itself outages is undefined and raises. [`get_arc_axis`](@ref) lists
# the arcs available to pick from, and the convenience [`NetworkModification`](@ref)
# constructor builds the spec straight from one of them:

arcs = get_arc_axis(vmodf)
monitored_arc = arcs[1]
ctg = NetworkModification(vmodf, arcs[2])

# The returned row carries one post-contingency sensitivity per bus:

row = vmodf[monitored_arc, ctg]

# The second index accepts three equivalent forms — the
# [`NetworkModification`](@ref) used above, a [`ContingencySpec`](@ref) from the
# registered set, or the original [`PowerSystems.Outage`](@extref PowerSystems.Outage)
# (by its registered UUID). All resolve to the same [`NetworkModification`](@ref) and
# share the cached Woodbury factors, so repeated queries for one contingency across
# different monitored arcs reuse work.

# A [`ContingencySpec`](@ref) pairs a [`NetworkModification`](@ref) with the UUID of the
# outage it came from, so the registered spec matching `ctg` is the one whose
# `modification` field compares equal to it:

spec = first(s for s in values(contingencies) if s.modification == ctg)
vmodf[monitored_arc, spec] == vmodf[monitored_arc, spec.modification] == row

# Indexing by the [`PowerSystems.Outage`](@extref PowerSystems.Outage) attribute itself
# gives the same row again. Fetch the outage from the branch that `ctg` trips:

outaged_branch = first(
    branch for branch in PowerSystems.get_components(PowerSystems.ACTransmission, sys)
    if PowerSystems.get_number(PowerSystems.get_from(PowerSystems.get_arc(branch))) ==
    arcs[2][1] &&
    PowerSystems.get_number(PowerSystems.get_to(PowerSystems.get_arc(branch))) ==
    arcs[2][2]
)
outage =
    first(PowerSystems.get_supplemental_attributes(PowerSystems.Outage, outaged_branch))
vmodf[monitored_arc, outage] == row

# ## The modification type model

# Under the convenience constructor sit a few value types, layered from the smallest
# unit up to the solver-ready form. It helps to know them before dropping to the
# manual path:
#
# | Type                          | Represents                                                             | Scope                      |
# |:----------------------------- |:---------------------------------------------------------------------- |:-------------------------- |
# | [`ArcModification`](@ref)     | A susceptance change on one aggregated arc, plus optional Ybus Pi-model deltas | One arc              |
# | [`ShuntModification`](@ref)   | A diagonal admittance change on one bus                                | One bus                    |
# | [`NetworkModification`](@ref) | A canonical, [`System`](@extref PowerSystems.System)-independent bundle of arc and shunt changes plus islanding status | Whole modification |
# | [`ContingencySpec`](@ref)     | A [`NetworkModification`](@ref) tagged with the source [`PowerSystems.Outage`](@extref PowerSystems.Outage) UUID | One registered contingency |
#
# [`NetworkModification`](@ref) is the canonical representation: once built it holds no
# reference to the source [`System`](@extref PowerSystems.System) and serves as the
# cache key inside [`VirtualMODF`](@ref) (its `label` is excluded from equality, so two
# physically identical modifications compare equal regardless of name).
#
# !!! note
#     Partial (non-full-outage) susceptance changes are supported only on **direct and
#     parallel** arcs. Series-reduced arcs and 3-winding transformer windings accept
#     only a full outage of the equivalent; anything else raises an error.

# ## Build a Modification Manually

# The convenience constructor used above (`NetworkModification(matrix, arc)`, or
# `NetworkModification(matrix, branch)`) is the simplest path — it looks up the
# arc's susceptance and populates the deltas for you. When you want full control,
# assemble the low-level building blocks instead. An [`ArcModification`](@ref) is a
# susceptance change on one arc (`delta_b` negative for an outage); a
# [`ShuntModification`](@ref) is an admittance change on one bus. Both are indexed
# by their **integer** position in the matrix, which [`get_arc_lookup`](@ref) and
# [`get_bus_lookup`](@ref) provide:

arc_index = get_arc_lookup(vmodf)[(1, 4)]
arc_mod = ArcModification(arc_index, -5.0)   # a partial Δb on a direct arc

# A [`ShuntModification`](@ref) is built the same way, from the bus position that
# [`get_bus_lookup`](@ref) returns:

bus_index = get_bus_lookup(vmodf)[3]
shunt_mod = ShuntModification(bus_index, ComplexF32(-0.1im))

# Combining the two gives one [`NetworkModification`](@ref) — the constructor takes a
# label, the arc changes, the shunt changes, and whether the modification islands the
# network:

custom = NetworkModification("arc_and_shunt", [arc_mod], [shunt_mod], false)
vmodf[monitored_arc, custom]

# Prefer the convenience constructors over hand-built [`ArcModification`](@ref)
# values: they compute physically consistent `delta_b` and Pi-model deltas from the
# network data, which is otherwise your responsibility to get right. Compare the `-5.0`
# above with the `delta_b` that [`NetworkModification`](@ref) derives for a full outage
# of the same arc:

only(NetworkModification(vmodf, (1, 4)).arc_modifications).delta_b

# ## One-Shot Post-Modification Rows from a VirtualPTDF

# If you already hold a [`VirtualPTDF`](@ref) and want a single post-modification
# row without registering contingencies, use
# [`get_post_modification_ptdf_row`](@ref). It applies a [`NetworkModification`](@ref)
# through the same Woodbury correction, on an arc taken from
# [`get_arc_axis`](@ref):

vptdf = VirtualPTDF(sys)
varcs = get_arc_axis(vptdf)
mod = NetworkModification(vptdf, varcs[2])
row_oneshot = get_post_modification_ptdf_row(vptdf, varcs[1], mod)

# Indexing is the equivalent form — it returns the same row:

isapprox(vptdf[varcs[1], mod], row_oneshot)

# This function does **no caching** — each call recomputes. When querying many
# monitored arcs for the *same* modification, precompute once with
# [`compute_woodbury_factors`](@ref) and reuse via [`apply_woodbury_correction`](@ref).

# ## Contingencies and Network Reduction

# If you build the [`VirtualMODF`](@ref) with `network_reductions`, any branch that a
# contingency outages or monitors must survive every reduction step. Outage and
# monitored-component buses are auto-protected from reduction. Declare monitored
# branches on the outage so their buses are kept:

monitored_line = PowerSystems.get_component(PowerSystems.ACTransmission, sys, "2")
PowerSystems.set_monitored_components!(outage, [monitored_line])

# The reduced [`VirtualMODF`](@ref) then keeps every arc the registered contingencies
# need. Reductions are passed as a vector of [`NetworkReduction`](@ref) specs, and
# [`get_arc_axis`](@ref) reports what survived. `c_sys5` has no radial buses to remove,
# so here [`RadialReduction`](@ref) leaves the arc axis untouched; on a network that
# does reduce, the protected arcs are the ones guaranteed to survive:

vmodf_reduced = VirtualMODF(sys; network_reductions = NetworkReduction[RadialReduction()])
get_arc_axis(vmodf_reduced)

# Querying a monitored arc that was reduced away raises a clear error rather than
# silently returning the base row.

# ## See Also
#
#   - [Public API Reference](@ref) — full docstrings for [`NetworkModification`](@ref),
#     [`ContingencySpec`](@ref), [`compute_woodbury_factors`](@ref), and the rest
#   - [Flowgate Methodology](@ref) — the Woodbury post-contingency theory
