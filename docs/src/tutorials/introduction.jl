# # Introduction

# This tutorial uses a [`PTDF`](@ref) to examine current power flow, a [`LODF`](@ref) to examine how a tripped line's power redirects, and network reductions to make the analysis faster to repeat. Together this will answer:
#
# > If a key transmission line trips, which other lines are most at risk of overloading?

using PowerNetworkMatrices
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# !!! note
#
#     `PowerSystemCaseBuilder.jl` only supplies the ready-made example systems used
#     throughout this documentation. To build a
#     [`System`](@extref PowerSystems.System) from your own data, see the
#     [PowerSystems.jl documentation](https://sienna-platform.github.io/PowerSystems.jl/stable).

# ## Step 1 — Load the network

# Network matrices are built from a [`PowerSystems.System`](@extref PowerSystems.System).

sys = PSB.build_system(PSB.PSSEParsingTestSystems, "psse_14_network_reduction_test_system"); #hide

# We will focus on the line between buses `103` and `104`.

# ## Step 2 — Determine important lines with the PTDF

# A [`PTDF`](@ref) (Power Transfer Distribution Factor) matrix is indexed by an **arc tuple** `(from_bus, to_bus)` and a **bus number**, answering what proportion of power injected at a given bus flows through a given branch (when withdrawn at the reference bus). One column of the [`PTDF`](@ref) corresponds to one bus's influence on every branch.

ptdf = PTDF(sys)
ptdf[(103, 104), 103]

# Compare a bus that barely touches this line:

ptdf[(103, 104), 102]

# Since power is withdrawn at reference buses, power injected at a reference bus does not flow.

ref_buses = PNM.get_ref_bus(ptdf)
@assert ptdf[(103, 104), only(ref_buses)] == 0.0
@assert ptdf[(102, 103), only(ref_buses)] == 0.0

# ## Step 3 — Determine post-contingency power flow with the LODF

# The [`LODF`](@ref) (Line Outage Distribution Factor) is indexed by two arc tuples, the **monitored** arc and the **outaged** arc, answering how flow is redistributed in an outage.

lodf = LODF(sys)

# We can filter for monitored arcs that absorb flow past a given threshold.

outaged = (103, 104)
monitored = [(arc, lodf[arc, outaged]) for arc in lodf.axes[1] if arc != outaged]
filter!(pair -> abs(pair[2]) > 0.05, monitored)
sort!(monitored; by = pair -> -abs(pair[2]))
monitored

# Branch `(102, 103)` inherits the entire flow of the outaged line, a factor of `-1.0`, because it is the series partner on the far side of bus `103`. The two parallel paths `101–115–102` and `101–117–118–104` each absorb roughly `65%`, and branch `(102, 104)` takes the remaining `35%`. Every other branch is unaffected.

# A negative factor means the redistributed flow runs *against* the monitored branch's `(from, to)` orientation. Individual factors can also be read directly.

lodf[(102, 103), (103, 104)]

# ## Step 4 — Reduce the network to repeat the analysis cheaply

# Reliability studies are rerun across every credible outage and every operating point, so the matrices should be as small as possible. Networks typically contain buses that do not affect a study of this kind: dead-end (radial) buses and pass-through (degree-two) buses. PNM can reduce them away, shrinking the matrices while leaving the surviving sensitivities unchanged.

# Reductions are supplied to any constructor through the `network_reductions` keyword. Here we combine [`RadialReduction`](@ref), which drops dangling buses, with [`DegreeTwoReduction`](@ref), which fuses pass-through chains.

reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
ptdf_reduced = PTDF(sys; network_reductions = reductions)
size(ptdf_reduced), size(ptdf)

# The reduced matrix has fewer branch rows and fewer bus columns, but the sensitivity from Step 2 is unchanged.

ptdf_reduced[(103, 104), 103], ptdf[(103, 104), 103]

# The same holds for the [`LODF`](@ref) screen.

lodf_reduced = LODF(sys; network_reductions = reductions)
lodf_reduced[(102, 103), (103, 104)], lodf[(102, 103), (103, 104)]

# A reduction therefore gives a smaller and faster pair of matrices that answer identically for every element that survives it.

# !!! warning
#     When screening a *reduced* network against ratings, a degree-two merge fuses several
#     branches into one equivalent arc whose limit is an aggregate. Use
#     [`get_single_element_contingency_rating`](@ref) and the related aggregated-rating
#     accessors rather than a single branch's `PSY.get_rating`.

# Note that every lookup above used an **arc tuple** rather than a branch name. Arc tuples are an unambiguous identifier and they survive reductions, whereas named branches may be merged away, so they are the recommended identifier to use in code that runs both before and after a reduction.

# ## Where to go next
#
#   - [Analysis at Scale](@ref) — the second tutorial: screening *many* contingencies
#     on a large network with the memory-light virtual matrices and cache control.
#   - [Matrix overview & indexing](@ref) — the reference for every matrix type, its
#     axes, indexing, and accessors.
#   - [The DC Power Flow Approximation](@ref) — the theory behind these sensitivities.
#   - [How to Define and Apply Contingencies](@ref) — post-contingency factors beyond
#     the single-outage [`LODF`](@ref).
