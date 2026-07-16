# # How to Check Connectivity and Find Islands

# This guide shows you how to test whether a system's network is fully connected
# and how to enumerate its electrical islands (subnetworks). A singular ABA
# matrix or a failed power flow is often just a disconnected network — these
# checks locate the problem.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` installed
#   - A power system model loaded (see [Getting Started](@ref))

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Check whether the network is connected

# [`validate_connectivity`](@ref) returns `true` when the whole system forms a
# single connected component. It uses a depth-first traversal of the Ybus graph:

PNM.validate_connectivity(sys)

# ## Enumerate the subnetworks

# [`find_subnetworks`](@ref) returns a `Dict` mapping each subnetwork's reference
# bus to the set of bus numbers in that island. A connected system yields a
# single entry:

PNM.find_subnetworks(sys)

# ## Choose the traversal algorithm

# The lower-level `find_subnetworks(M, bus_numbers; subnetwork_algorithm)`
# operates on a sparse connectivity matrix and lets you pick the algorithm.
# Build an [`AdjacencyMatrix`](@ref) and pass its data and bus axis:

adj = PNM.AdjacencyMatrix(sys)
bus_numbers = PNM.get_bus_axis(adj);

# [`iterative_union_find`](@ref) (the default) uses a union-find disjoint-set
# structure:

PNM.find_subnetworks(adj.data, bus_numbers; subnetwork_algorithm = PNM.iterative_union_find)

# [`depth_first_search`](@ref) traverses the graph recursively instead:

PNM.find_subnetworks(adj.data, bus_numbers; subnetwork_algorithm = PNM.depth_first_search)

# Both return the same island decomposition; each also warns for any islanded
# (degree-zero) bus it encounters.

# ## Notes
#
#   - `validate_connectivity` and `find_subnetworks` also accept a built matrix
#     directly (`AdjacencyMatrix`, `Ybus`), avoiding a rebuild if you already
#     have one.
#   - The `subnetwork_algorithm` kwarg is threaded through the matrix
#     constructors (e.g. `ABA_Matrix`), so islands are detected consistently at
#     build time.
#   - See the [accessor reference](../reference/accessors.md) for the axis and
#     lookup getters used above.
