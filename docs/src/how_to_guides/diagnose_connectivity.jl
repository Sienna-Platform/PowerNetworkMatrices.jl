# # How to Diagnose a Disconnected Network

# A singular `ABA` matrix or a failed DC power flow is frequently just a
# disconnected network. When you hit one, checking connectivity first localizes the
# problem before you dig into the numerics. This guide walks through it.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Step 1 — Is the whole network connected?

# [`validate_connectivity`](@ref) returns `true` when the system forms a single
# connected component:

PNM.validate_connectivity(sys)

# ## Step 2 — Enumerate the islands

# [`find_subnetworks`](@ref) returns a `Dict` mapping each island's reference bus to
# the set of bus numbers in it. A **connected** system yields a single entry:

PNM.find_subnetworks(sys)

# Both functions also accept an already-built [`AdjacencyMatrix`](@ref) or
# [`Ybus`](@ref), so a prior matrix is reused instead of rebuilt:

adj = PNM.AdjacencyMatrix(sys)
PNM.validate_connectivity(adj)

# ## Step 3 — Read a fragmented result

# When the network is split, [`find_subnetworks`](@ref) returns **more than one**
# entry — one island per reference bus. To see what that looks like, run the
# lower-level method on a small connectivity matrix with two disconnected bus pairs,
# `{1, 2}` and `{3, 4}`:

import SparseArrays

M = SparseArrays.sparse(
    [1, 2, 1, 2, 3, 4, 3, 4],
    [1, 1, 2, 2, 3, 3, 4, 4],
    ones(Int, 8),
    4,
    4,
)

PNM.find_subnetworks(M, [1, 2, 3, 4])

# Two entries: buses `1`–`2` form one island and `3`–`4` another. On a real system,
# each island needs its own reference bus; a block with no reference bus is exactly
# what leaves `ABA` singular. The bus sets tell you which buses to reconnect, or
# which island to study on its own.

# ## Choose a traversal algorithm

# The lower-level `find_subnetworks(M, bus_numbers; subnetwork_algorithm)` selects
# how the graph is walked:
#
#   - [`iterative_union_find`](@ref) (the **default**) — an iterative union-find
#     disjoint-set, safe on networks of any size.
#   - [`depth_first_search`](@ref) — a recursive traversal.
#
# Both return the **same** island decomposition, so the choice is about performance,
# not correctness. Prefer the default union-find; it avoids the deep recursion that
# `depth_first_search` can hit on very large networks.

PNM.find_subnetworks(M, [1, 2, 3, 4]; subnetwork_algorithm = PNM.depth_first_search)

# The `subnetwork_algorithm` keyword also threads through the matrix constructors
# (e.g. [`ABA_Matrix`](@ref)), so islands are detected the same way at build time as
# by an explicit [`find_subnetworks`](@ref) call.

# ## See also
#
#   - [Connectivity & islands](@ref) — the reference for these functions.
#   - [Network Reduction Theory](@ref) — how the susceptance graph can fragment into
#     more islands than the admittance graph, and why that matters for `ABA`.
