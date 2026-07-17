# # Getting Started

# This tutorial is a guided tour of `PowerNetworkMatrices.jl` (PNM). By the end you
# will have built a few different network matrices from a power system, read
# meaningful values out of them, swapped in a memory-light *virtual* matrix, and
# shrunk a network with a reduction — the whole arc of what the package does, in one
# sitting. Follow along and experiment; each stop links onward to the how-to guides
# and reference for depth.

# We keep to a single happy path. The point here is to *see* the pieces fit
# together, not to cover every option — those live in the how-to guides and the
# [Matrix overview & indexing](@ref) reference.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# !!! note
#
#     `PowerSystemCaseBuilder.jl` is a helper library for reproducing the examples
#     in this documentation. See the
#     [PowerSystemCaseBuilder documentation](https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable)
#     for how to load your own data.

# ## Step 1 — Load a power system

# Network matrices are built from a [`PowerSystems.System`](@extref PowerSystems.System).
# Here we load a small five-bus test system so the example is reproducible; in your
# own work you would build the [`System`](@extref PowerSystems.System) from your data
# files instead.

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# This system has five buses (numbered `1`–`5`) and six branches (named `"1"`
# through `"6"`).

# ## Step 2 — Build a PTDF and read a sensitivity

# The [`PTDF`](@ref) (Power Transfer Distribution Factor) matrix is a good first
# stop. Build it by calling the constructor on the system — this computes the whole
# matrix once and stores it, along with the axes that let you index it by physical
# network elements.

ptdf = PNM.PTDF(sys)

# Each entry answers one question: *if one unit of power is injected at a given bus
# (and withdrawn at the reference bus), how much of it flows along a given branch?*
# Rows are branches, columns are buses. Index by a **branch name** and a **bus
# number** directly — the matrix maps those to its internal positions for you:

ptdf["1", 2]

# The result is about `-0.48`. Read it like this: injecting 1 MW at bus 2 (and
# withdrawing it at the reference bus) changes the flow on branch `"1"` by roughly
# `-0.48` MW. The magnitude — close to half — tells you branch `"1"` is a major path
# for power leaving bus 2; the sign tells you the flow moves *against* the branch's
# `(from, to)` orientation.

# Now explore. The same branch responds differently to injections at different
# buses — each column of the matrix is a different bus's influence on this branch:

ptdf["1", 3]

# And you can index by an **arc tuple** `(from_bus, to_bus)` instead of a branch
# name. Arc tuples are the canonical, unambiguous identifier — they survive network
# reductions, where named branches may be merged away:

ptdf[(2, 3), 1]

# ## Step 3 — A different matrix answers a different question

# Every matrix in PNM is built the same way — `Type(sys)` — and indexed the same
# way. Only the *question* changes. The [`LODF`](@ref) (Line Outage Distribution
# Factor) matrix asks: *if one branch is outaged, how does its flow redistribute
# onto the others?* Here both dimensions are branches (a monitored arc and an
# outaged arc):

lodf = PNM.LODF(sys)
lodf[(1, 4), (2, 3)]

# That value is the fraction of branch `(2, 3)`'s pre-outage flow that lands on
# branch `(1, 4)` when `(2, 3)` trips. The [Matrix overview & indexing](@ref)
# reference lists every matrix type and the identifiers each dimension takes.

# ## Step 4 — Virtual matrices are drop-in

# For large systems, materializing a full dense [`PTDF`](@ref) can be expensive. The
# lazy [`VirtualPTDF`](@ref) computes rows on demand and caches them, but builds and
# indexes **exactly** like the materialized form — swap the type name and nothing
# else changes:

vptdf = PNM.VirtualPTDF(sys)
vptdf[(2, 3), 1]

# Same answer as the materialized `ptdf[(2, 3), 1]` above, computed only for the row
# you asked for. [Virtual vs. materialized matrices](@ref) explains when to prefer
# each.

# ## Step 5 — Shrink a network with a reduction

# Real grids carry buses that add no information to a study: dead-end (radial) buses
# and pass-through (degree-two) buses. PNM can *reduce* them away, making the
# matrices smaller while leaving the sensitivities you care about unchanged. Let's
# see the payoff on a system built to exercise reduction.

sys14 =
    PSB.build_system(PSB.PSSEParsingTestSystems, "psse_14_network_reduction_test_system")

ptdf_full = PNM.PTDF(sys14)
size(ptdf_full)

# Pick one sensitivity to track through the reduction — the response of branch
# `(103, 104)` to an injection at bus `103`, both in the core network:

ptdf_full[(103, 104), 103]

# Reductions are supplied to any constructor through the `network_reductions`
# keyword. We combine two strategies — [`RadialReduction`](@ref) (drop dangling
# buses) then [`DegreeTwoReduction`](@ref) (fuse pass-through chains):

reductions = [PNM.RadialReduction(), PNM.DegreeTwoReduction()]
ptdf_reduced = PNM.PTDF(sys14; network_reductions = reductions)
size(ptdf_reduced)

# The matrix is smaller — fewer bus columns and fewer branch rows. Yet the tracked
# sensitivity is unchanged, to the last digit:

ptdf_reduced[(103, 104), 103]

# A smaller, faster matrix that gives the same answer for every element that
# survived the reduction. That is the whole point of reduction.

# ## Where to go next
#
# You have now touched the core of the package. To go deeper:
#
#   - [Matrix overview & indexing](@ref) — the reference for every matrix type, its
#     axes, indexing, and accessors.
#   - [How to Build Multiple Matrices Without Repeating Work](@ref) — reuse shared
#     intermediates when you need several matrices for the same system.
#   - [How to Apply Network Reductions at Construction](@ref) — combining and
#     ordering reductions, and reading back what changed.
#   - [The DC Power Flow Approximation](@ref) — the theory behind these
#     sensitivities.
