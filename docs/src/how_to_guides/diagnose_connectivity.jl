# # How to Diagnose a Disconnected Network

# A singular `ABA` matrix or a failed DC power flow is frequently just a
# disconnected network: an island with no reference bus leaves `ABA` singular.
# Checking connectivity first localizes the problem before you dig into the numerics.
# This guide walks through the check, then deliberately breaks a network so you can see
# exactly what a fragmented result looks like — and how to get back.

import PowerNetworkMatrices as PNM
import PowerSystems as PSY
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Step 1 — Confirm a healthy network is connected

# [`validate_connectivity`](@ref) returns `true` when the system forms a single
# connected component:

PNM.validate_connectivity(sys)

# [`find_subnetworks`](@ref) shows the decomposition behind that answer: a `Dict`
# mapping each island's reference bus to the set of bus numbers in it. A connected
# system yields a **single** entry:

PNM.find_subnetworks(sys)

# Both functions also accept an already-built [`AdjacencyMatrix`](@ref) or
# [`Ybus`](@ref), so a matrix you already have on hand is reused instead of rebuilt:

adj = PNM.AdjacencyMatrix(sys)
PNM.validate_connectivity(adj)

# ## Step 2 — Disconnect a bus and watch it split

# To see a fragmented result on a real system, let's isolate one bus. A bus goes silent
# when every branch touching it is out of service, so we find bus `5`'s incident
# branches and mark them unavailable — `Ybus` (and therefore the connectivity check)
# only includes available branches:

isolated_bus = 5
incident = [
    br for br in PSY.get_components(PSY.ACBranch, sys) if
    PSY.get_number(PSY.get_from(PSY.get_arc(br))) == isolated_bus ||
    PSY.get_number(PSY.get_to(PSY.get_arc(br))) == isolated_bus
]

for br in incident
    PSY.set_available!(br, false)
end

# The network is now split. `validate_connectivity` reports it:

PNM.validate_connectivity(sys)

# ...and `find_subnetworks` returns **two** entries — the main island, and bus `5`
# stranded on its own:

PNM.find_subnetworks(sys)

# There is the diagnosis. That second island — the isolated `{5}` — has no reference
# bus of its own, which is exactly the block that would have made `ABA` singular. The
# bus set tells you precisely which buses to reconnect (or which island to study in
# isolation). Here it points straight back at the bus we broke.

# ## Step 3 — Reconnect and recover

# Restoring the branches we took out puts the network back together — bus `5` rejoins
# the main island and connectivity is whole again:

for br in incident
    PSY.set_available!(br, true)
end

PNM.validate_connectivity(sys)

# ...and the decomposition is back to a single island, identical to where we started:

PNM.find_subnetworks(sys)

# ## Choosing a traversal algorithm

# The lower-level `find_subnetworks(M, bus_numbers; subnetwork_algorithm)` — which runs
# over a raw sparse connectivity matrix — lets you pick how the graph is walked:
#
#   - [`iterative_union_find`](@ref) (the **default**) — an iterative union-find
#     disjoint-set, safe on networks of any size.
#   - [`depth_first_search`](@ref) — a recursive traversal.
#
# Both return the **same** island decomposition, so the choice is about performance,
# not correctness. Prefer the default union-find; it avoids the deep recursion that
# `depth_first_search` can hit on very large networks. The `subnetwork_algorithm`
# keyword also threads through the matrix constructors, so islands are detected the same
# way at build time as by an explicit [`find_subnetworks`](@ref) call:

PNM.ABA_Matrix(sys; subnetwork_algorithm = PNM.depth_first_search);

# ## See also
#
#   - [Matrix overview & indexing](@ref) — the [`AdjacencyMatrix`](@ref) and
#     [`Ybus`](@ref) graphs these checks traverse, and per-island axes.
#   - [Network Reduction Theory](@ref) — how the susceptance graph can fragment into
#     more islands than the admittance graph, and why that matters for `ABA`.
