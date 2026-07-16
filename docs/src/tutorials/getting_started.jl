# # Getting Started

# This tutorial takes you from a power system to your first network-sensitivity
# number. By the end you will have built a Power Transfer Distribution Factor
# (`PTDF`) matrix and read a single, meaningful value out of it — the change in
# one branch's flow caused by an injection at one bus.

# We keep to a single path here. `PowerNetworkMatrices.jl` builds many other
# matrices (`LODF`, `Ybus`, `BA_Matrix`, `ABA_Matrix`, and lazy `Virtual*`
# variants); those are covered in [How to Compute Network Matrices](@ref) and the
# [Matrix overview & indexing](@ref) reference, not in your first five minutes.

# ## Step 1 — Load a power system

# Network matrices are built from a `PowerSystems.System`. Here we use
# `PowerSystemCaseBuilder.jl` to load a small five-bus test system so the example
# is reproducible; in your own work you would build the `System` from your data
# files instead.

# !!! note
#
#     `PowerSystemCaseBuilder.jl` is a helper library for reproducing the examples
#     in this documentation. See the
#     [PowerSystemCaseBuilder documentation](https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable)
#     for how to load your own data.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# This system has five buses (numbered `1`–`5`) and six branches (named `"1"`
# through `"6"`).

# ## Step 2 — Build the PTDF matrix

# Call [`PTDF`](@ref) on the system. This computes the whole matrix once and
# stores it, along with the axes that let you index it by physical network
# elements.

ptdf = PNM.PTDF(sys)

# Each entry answers one question: *if one unit of power is injected at a given
# bus (and withdrawn at the reference bus), how much of it flows along a given
# branch?* Rows are branches, columns are buses.

# ## Step 3 — Read one sensitivity

# You index the matrix by a **branch name** and a **bus number** directly — the
# matrix maps those to its internal positions for you. Let's ask how the flow on
# branch `"1"` (which connects buses 1 and 2) responds to an injection at bus 2:

ptdf["1", 2]

# The result is about `-0.48`. Read it like this: injecting 1 MW at bus 2 (and
# withdrawing it at the reference bus) changes the flow on branch `"1"` by
# roughly `-0.48` MW. The magnitude — close to half — tells you branch `"1"` is a
# major path for power leaving bus 2; the sign tells you the flow moves *against*
# the branch's `(from, to)` orientation.

# That single number is a network sensitivity, and computing it for every
# branch/bus pair is exactly what the `PTDF` matrix gives you.

# ## Where to go next
#
#   - [How to Compute Network Matrices](@ref) — build `LODF`, `Ybus`, and the
#     other matrices the same way.
#   - [Network Reduction](@ref) — shrink a large system before building matrices.
#   - [The DC Power Flow Approximation](@ref) — the theory behind these
#     sensitivities.
