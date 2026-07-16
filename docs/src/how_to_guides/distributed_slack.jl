# # How to Use a Distributed Slack

# This guide shows you how to spread the slack (reference) injection across
# several buses instead of a single reference bus, using the `dist_slack`
# keyword. For why this changes PTDF values and how to choose weights, see
# [Slack distribution and reference-bus conventions](../explanation/slack_conventions.md).

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## The `dist_slack` keyword

# By default `dist_slack` is empty, and the matrix uses the conventional single
# reference bus. Supplying weights distributes the slack across the named buses,
# in proportion to the weights (they are normalized internally).

# The type of `dist_slack` **differs by matrix**:
#
#   - [`PTDF`](@ref) / [`VirtualPTDF`](@ref): `Dict{Int, Float64}` mapping a bus
#     number to its weight.
#   - [`VirtualLODF`](@ref) / [`VirtualMODF`](@ref): `Vector{Float64}`, one
#     weight per bus (ordered like the bus axis).

# ## Weighted PTDF

# Pass a `Dict` of bus number → weight. Here the slack is split across buses 1,
# 3, and 4:

weights = Dict(1 => 0.5, 3 => 0.3, 4 => 0.2)
ptdf = PNM.PTDF(sys; dist_slack = weights);

# Weights need not sum to one — they are normalized:

ptdf = PNM.PTDF(sys; dist_slack = Dict(1 => 5.0, 3 => 3.0, 4 => 2.0));

# The same `Dict` form works for the on-demand [`VirtualPTDF`](@ref):

vptdf = PNM.VirtualPTDF(sys; dist_slack = weights);

# ## Distributed slack for virtual LODF / MODF

# For [`VirtualLODF`](@ref) and [`VirtualMODF`](@ref), pass a `Vector{Float64}`
# with one entry per bus rather than a `Dict`:

nbus = size(PNM.Ybus(sys), 1)             # Ybus is buses x buses
slack_vector = fill(1.0 / nbus, nbus)     # uniform distribution across all buses
vlodf = PNM.VirtualLODF(sys; dist_slack = slack_vector);
