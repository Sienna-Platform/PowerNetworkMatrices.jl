# # How to Inspect a Reduced Network

# When you build a matrix with `network_reductions`, the mappings that describe
# which buses and arcs were eliminated are attached to the matrix as a
# [`NetworkReductionData`](@ref) object. This guide shows how to read that data:
# which buses survived, where eliminated buses were mapped, and which reduction
# algorithms were applied.

# For a first, learning-oriented walkthrough of building a reduced matrix and
# seeing the payoff, start with the [Network Reduction](@ref) tutorial. This
# how-to is the reference recipe for querying the reduction data afterward.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` installed
#   - Familiarity with applying reductions at construction (see the reduction
#     how-to and [Network reduction reference](@ref))

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# Load a system that has radial and degree-two structure to reduce:

sys = PSB.build_system(PSB.PSSEParsingTestSystems,
    "psse_14_network_reduction_test_system");

# ## Build a Reduced Matrix

# Apply reductions through the `network_reductions` keyword. Reductions are
# supplied only as a `Vector{NetworkReduction}` — there is no per-flag keyword.

reductions = PNM.NetworkReduction[PNM.RadialReduction(), PNM.DegreeTwoReduction()];
ptdf = PNM.PTDF(sys; network_reductions = reductions);

# ## Retrieve the Reduction Data

# Every reduced matrix exposes its `NetworkReductionData` through
# [`get_network_reduction_data`](@ref):

nrd = PNM.get_network_reduction_data(ptdf)

# The `show` method above prints a summary. The accessors below let you query it
# programmatically.

# ## Which Reductions Were Applied

# The predicate accessors report which algorithms ran:

PNM.has_radial_reduction(nrd)

#

PNM.has_degree_two_reduction(nrd)

#

PNM.has_ward_reduction(nrd)

# [`get_reductions`](@ref) returns the `ReductionContainer` holding the applied
# reduction specs in order:

PNM.get_reductions(nrd)

# ## Removed Buses and Arcs

# `get_removed_buses` returns the set of bus numbers eliminated from the network,
# and `get_removed_arcs` the set of eliminated arc tuples `(from, to)`:

removed_buses = PNM.get_removed_buses(nrd)

#

removed_arcs = PNM.get_removed_arcs(nrd)

# ## The Bus Reduction Map

# `get_bus_reduction_map` maps each **retained** bus to the set of buses that were
# folded into it. Its keys are the buses that survived the reduction:

bus_map = PNM.get_bus_reduction_map(nrd)

# The retained bus numbers are simply the keys — the universal "did this bus
# survive" check:

retained_buses = keys(bus_map)

# ## Mapping an Eliminated Bus to Its Parent

# To find the surviving bus that an eliminated bus was mapped to, use
# `get_mapped_bus_number`. It accepts a bus number or a `PSY.ACBus`. For a
# retained bus it returns the bus unchanged; for an eliminated bus it returns the
# parent it was folded into:

if !isempty(removed_buses)
    eliminated = first(removed_buses)
    parent = PNM.get_mapped_bus_number(nrd, eliminated)
    (eliminated, parent)
end

# This is the accessor to use when you have a bus from the original system and need
# to locate its representative in the reduced matrix.

# ## See Also
#
#   - [Network reduction reference](@ref) — full accessor list and spec types
#   - [Network Reduction Theory](@ref) — why these reductions preserve behavior
