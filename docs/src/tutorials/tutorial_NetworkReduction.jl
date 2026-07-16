# # Network Reduction

# Large networks carry buses that add no information to a study: dead-end
# (radial) buses hanging off the edge of the grid, and pass-through buses that
# merely relay power between two neighbors. This tutorial walks you through
# *reducing* such a network — collapsing those buses away — and then shows the
# payoff: the network matrices get smaller while the sensitivities you care about
# stay exactly the same.

# We use two reduction strategies together:
#
#   - [`RadialReduction`](@ref) removes radial (dangling) buses — leaf nodes with
#     a single connection — and their branches. They cannot affect flows in the
#     core network, so dropping them is lossless.
#   - [`DegreeTwoReduction`](@ref) removes buses with exactly two connections,
#     fusing the two incident branches into one equivalent branch. It is applied
#     recursively, collapsing whole chains of pass-through buses.
#
# The electrical theory — why these reductions preserve the retained network's
# behavior, and how equivalent branches are derived — lives in
# [Network Reduction Theory](@ref). Here we focus on doing it.

using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# ## Step 1 — Start with the full network

# We load a PSS/E test system built specifically to exercise network reduction,
# and compute its `PTDF` matrix as a baseline. The matrix dimensions tell us how
# big the problem is: `PTDF` is indexed by buses along one axis and branches
# (arcs) along the other.

sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")

ptdf_full = PNM.PTDF(sys)
size(ptdf_full)

# Pick one sensitivity to track through the reduction — the response of the flow
# on branch `(103, 104)` to an injection at bus `103`. Both this branch and this
# bus are part of the core network, so a good reduction must leave the value
# untouched.

ptdf_full[(103, 104), 103]

# ## Step 2 — Apply the reduction

# Reductions are supplied to any matrix constructor through the
# `network_reductions` keyword, as a vector of reduction specifications. They are
# applied in order; running `RadialReduction` first is the usual choice, because
# removing dangling buses can expose new degree-two buses for the second pass to
# collapse.

reductions = [RadialReduction(), DegreeTwoReduction()]
ptdf_reduced = PNM.PTDF(sys; network_reductions = reductions)
size(ptdf_reduced)

# The matrix is smaller — fewer bus columns and fewer branch rows — because the
# reduced buses and their branches are gone, and collapsed chains are represented
# by single equivalent branches.

# ## Step 3 — See what was removed

# Every reduced matrix carries a [`NetworkReductionData`](@ref) record describing
# exactly what changed. Retrieve it with [`get_network_reduction_data`](@ref):

reduction_data = get_network_reduction_data(ptdf_reduced)

# The buses that were eliminated:

PNM.get_removed_buses(reduction_data)

# The branches (arcs) that were eliminated:

PNM.get_removed_arcs(reduction_data)

# And [`get_bus_reduction_map`](@ref) shows where each removed bus went — the
# parent bus that absorbed it:

get_bus_reduction_map(reduction_data)

# ## Step 4 — Confirm the payoff

# The whole point is that the retained network behaves identically. Read the same
# sensitivity from the reduced matrix — indexing works exactly as before, by arc
# tuple and bus number:

ptdf_reduced[(103, 104), 103]

# It matches the full-network value from Step 1 to the last digit. You have a
# smaller, faster matrix that gives the same answer for every element that
# survived the reduction.

# ## Where to go next
#
#   - [Network Reduction Theory](@ref) — why the reductions are lossless and how
#     equivalent-branch parameters are derived.
#   - `WardReduction` — a third strategy that reduces the network around a chosen
#     set of study buses, for when you care about one area of a large grid.
