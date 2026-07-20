#src EXECUTE = TRUE
# # Getting Started

# This tutorial answers one operational question end to end with
# `PowerNetworkMatrices.jl` (PNM):
#
# > If a key transmission line trips, which other lines are most at risk of overloading?
#
# We take it in three moves — a [`PTDF`](@ref) to see how power routes today, an
# [`LODF`](@ref) to see where a tripped line's flow goes, and a network *reduction* to
# make the study cheap enough to repeat.

import Logging
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

## Silence the informational build logs so the tutorial output stays readable.
Logging.disable_logging(Logging.Info)

# !!! note
#
#     `PowerSystemCaseBuilder.jl` only supplies the ready-made example systems used
#     throughout this documentation. To build a
#     [`System`](@extref PowerSystems.System) from your own data, see the
#     [PowerSystems.jl documentation](https://sienna-platform.github.io/PowerSystems.jl/stable).

# ## Step 1 — Load the network

# Network matrices are built from a [`PowerSystems.System`](@extref PowerSystems.System).
# We use a small IEEE-14-style test network; in your own work you would build the
# [`System`](@extref PowerSystems.System) from your data files instead.

sys = Logging.with_logger(Logging.NullLogger()) do
    PSB.build_system(PSB.PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
end;

# The line we are worried about is the one running between buses `103` and `104` — a
# central link in this grid. Our whole analysis is about what happens to the rest of the
# network if that line goes out.

# ## Step 2 — See how power routes today with a PTDF

# Before asking what happens when a line trips, we need to know how power flows through
# the network today. That is the [`PTDF`](@ref) (Power Transfer Distribution Factor)
# matrix. Build it by calling the constructor on the system — this computes the whole
# matrix once and stores it with the axes that let you index it by physical elements.

ptdf = PNM.PTDF(sys)

# Each entry answers one question: *if one unit of power is injected at a given bus
# (and withdrawn at the reference bus), how much of it flows along a given branch?*
# Index by an **arc tuple** `(from_bus, to_bus)` and a **bus number** directly — the
# matrix maps those to its internal positions for you. Here is line `(103, 104)`'s
# response to an injection at bus `103`:

ptdf[(103, 104), 103]

# The result is about `0.71`. Read it like this: injecting 1 MW at bus `103` (and
# withdrawing it at the reference bus) sends roughly `0.71` MW down line `(103, 104)`.
# That large fraction tells us `(103, 104)` is a dominant path for power leaving bus
# `103` — exactly why its outage is worth studying.

# Compare a bus that barely touches this line:

ptdf[(103, 104), 102]

# About `0.01` — an injection at bus `102` splits away through other paths and hardly
# loads it. The **reference bus** is special: since every injection is withdrawn there,
# injecting *at* it moves nothing, so its whole PTDF column is exactly zero. You do not
# have to know which bus that is ahead of time — ask the matrix:

ref_buses = PNM.get_ref_bus(ptdf)

# One reference bus per electrical island; here there is a single island with bus `101`.
# Confirm its column is zero by checking a couple of unrelated branches:

ptdf[(103, 104), only(ref_buses)]

#-

ptdf[(102, 103), only(ref_buses)]

# Both exactly `0.0`. A whole column of the PTDF is one bus's influence on every branch;
# a whole row is how one branch responds to every bus. This is the background flow
# picture. Now the outage.

# ## Step 3 — Where does a tripped line's flow go? Ask the LODF

# The [`LODF`](@ref) (Line Outage Distribution Factor) matrix answers our question
# directly: *if one branch is outaged, what fraction of its pre-outage flow lands on
# each other branch?* Both dimensions are branches — a **monitored** arc and an
# **outaged** arc.

lodf = PNM.LODF(sys)

# Trip our line and scan every other branch for the share it inherits. We keep only the
# branches that pick up a meaningful fraction and rank them worst-first — this is exactly
# the screen an operator runs:

outaged = (103, 104)
responders = [(arc, lodf[arc, outaged]) for arc in lodf.axes[1] if arc != outaged]
filter!(pair -> abs(pair[2]) > 0.05, responders)
sort!(responders; by = pair -> -abs(pair[2]))
responders

# There is our answer. Branch `(102, 103)` inherits **the entire** flow of the tripped
# line (a factor of `-1.0`): it is the series partner on the far side of bus `103`, so
# it is by far the most at risk. Behind it, two parallel paths each absorb about `65%` —
# the `101–115–102` path and the `101–117–118–104` path — and branch `(102, 104)` takes
# the remaining `35%`. Every other branch is untouched. An operator would watch
# `(102, 103)` first, then those two paths.

# The sign carries meaning too: a negative factor means the redistributed flow runs
# *against* the monitored branch's `(from, to)` orientation. You can always pull a
# single factor out directly:

lodf[(102, 103), (103, 104)]

# ## Step 4 — Make the study cheap enough to repeat

# We answered the question on the full network. But reliability studies get run over
# and over — every credible outage, every operating point — so the matrices want to be
# as small as possible. Real grids carry buses that add nothing to a study like this:
# dead-end (radial) buses and pass-through (degree-two) buses. PNM can *reduce* them
# away, shrinking the matrices while leaving the sensitivities we care about unchanged.

# Reductions are supplied to any constructor through the `network_reductions` keyword.
# We combine two strategies — [`RadialReduction`](@ref) (drop dangling buses) then
# [`DegreeTwoReduction`](@ref) (fuse pass-through chains):

reductions = PNM.NetworkReduction[PNM.RadialReduction(), PNM.DegreeTwoReduction()]
ptdf_reduced = PNM.PTDF(sys; network_reductions = reductions)
size(ptdf_reduced), size(ptdf)

# Fewer branch rows and fewer bus columns. Yet the sensitivity that anchored our whole
# analysis is unchanged, to the last digit:

ptdf_reduced[(103, 104), 103]

# The same holds for the outage screen. Rebuild the [`LODF`](@ref) with the same
# reductions and the worst-responder factor from Step 3 is identical:

lodf_reduced = PNM.LODF(sys; network_reductions = reductions)
lodf_reduced[(102, 103), (103, 104)]

# Still `-1.0`. A smaller, faster pair of matrices that gives the same answer for every
# element that survived the reduction. Note that we indexed with the **arc tuple**
# `(103, 104)` throughout:
# arc tuples are the canonical, unambiguous identifier and they survive reductions,
# where named branches may be merged away. That is why they are the identifier to reach
# for in code that runs before and after a reduction.

# ## Where to go next
#
# You have taken one question from raw network to answer and back, using PNM's core
# matrices. To go further:
#
#   - [Analysis at Scale](@ref) — the second tutorial: screening *many* contingencies
#     on a large network with the memory-light virtual matrices and cache control.
#   - [Matrix overview & indexing](@ref) — the reference for every matrix type, its
#     axes, indexing, and accessors.
#   - [The DC Power Flow Approximation](@ref) — the theory behind these sensitivities.
#   - [How to Define and Apply Contingencies](@ref) — post-contingency factors beyond
#     the single-outage LODF.
