# # How to Apply Network Reductions at Construction

# This guide shows you how to reduce a network while building a matrix, by
# passing reduction specs to the constructor. For the theory of what each
# reduction does, see
# [Network Reduction Theory](../explanation/network_reduction_theory.md); for the
# reduction spec types and the [`NetworkReductionData`](@ref) record they produce,
# see the [full public API](../reference/public.md).

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## The `network_reductions` keyword

# Reductions are applied **only** through the `network_reductions` keyword, a
# `Vector{NetworkReduction}`. Every matrix constructor that builds from a
# [`System`](@extref PowerSystems.System) accepts it: [`PTDF`](@ref), [`Ybus`](@ref), [`BA_Matrix`](@ref),
# [`ABA_Matrix`](@ref), [`LODF`](@ref), [`VirtualPTDF`](@ref),
# [`VirtualLODF`](@ref), and [`VirtualMODF`](@ref).

# The specs in the vector are applied in order. Pass an empty vector (the
# default) for no reduction.

# !!! tip
#     Write the vector with the `NetworkReduction[...]` element-type prefix. A
#     bare `[RadialReduction()]` infers the narrower `Vector{RadialReduction}`,
#     which the keyword (typed `Vector{NetworkReduction}`) will not accept. The
#     prefix is unnecessary only when the vector already holds two or more
#     different spec types, whose common element type is [`NetworkReduction`](@ref).

ptdf = PNM.PTDF(sys; network_reductions = PNM.NetworkReduction[PNM.RadialReduction()]);

# ## The three reduction specs

# ### Radial reduction

# [`RadialReduction`](@ref) eliminates leaf (degree-1) buses and their branches.
# It takes no fields:

reductions = PNM.NetworkReduction[PNM.RadialReduction()];

# ### Degree-two reduction

# [`DegreeTwoReduction`](@ref) folds degree-2 buses into equivalent series
# branches. Its one field controls whether reactive-only injector hosts are
# reduced:

PNM.DegreeTwoReduction()                                        # reduce_reactive_power_injectors = true (default)
PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false);

# The default `true` gives more reduction and is correct for DC use. Set it
# `false` for AC consumers, which must keep reactive-only buses.

# ### Ward reduction

# [`WardReduction`](@ref) eliminates buses outside a study area, keeping the
# electrical behavior seen from the retained buses. It requires the study-bus set
# (bus numbers to keep):

ward = PNM.WardReduction([1, 2, 3]);

# ## Combining reductions

# Pass several specs to apply them in sequence:

ptdf = PNM.PTDF(sys;
    network_reductions = PNM.NetworkReduction[
        PNM.RadialReduction(),
        PNM.DegreeTwoReduction(),
    ]);

# The same vector works on any constructor:

ybus = PNM.Ybus(sys; network_reductions = PNM.NetworkReduction[PNM.RadialReduction()])
aba = PNM.ABA_Matrix(sys; network_reductions = PNM.NetworkReduction[PNM.RadialReduction()]);

# ### Ordering and validation rules

# The specs are applied in vector order, subject to a few rules (a violation throws
# or warns at construction):
#
#   - each reduction **type** may appear at most once;
#   - [`WardReduction`](@ref) must be **last** when present;
#   - `ZeroImpedanceBranchReduction` may not be listed — it is auto-applied during
#     [`Ybus`](@ref) construction;
#   - a [`DegreeTwoReduction`](@ref) placed before a [`RadialReduction`](@ref) emits
#     a warning, because running radial first usually exposes more degree-two buses
#     for the second pass to collapse.

# ## Interaction with `tol`

# `network_reductions` composes with the sparsification `tol` keyword — the
# reduction changes the network, then `tol` sparsifies the resulting rows:

ptdf = PNM.PTDF(
    sys;
    network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()],
    tol = 1e-5,
);

# Note that reductions synthesize *equivalent* branches (series/parallel folds).
# These carry full numerical precision, which shifts what
# [`AutoTolerance`](@ref) infers as the data precision. See
# [How to Set the Sparsification Tolerance](@ref) for details.

# ## Reading the result

# The applied reductions are recorded on the matrix. Retrieve them with
# [`get_network_reduction_data`](@ref):

nrd = PNM.get_network_reduction_data(ptdf);

# The [`NetworkReductionData`](@ref) record exposes what changed through like-named
# `get_*` accessors — for example the eliminated buses:

PNM.get_removed_buses(nrd)

# ...and the eliminated arcs:

PNM.get_removed_arcs(nrd)

# To test whether a specific bus survived the reduction, check the bus-reduction
# map's keys (radial and degree-two survivors are keys; Ward survivors are the
# `study_buses`):

3 in keys(PNM.get_bus_reduction_map(nrd))

# See the [`NetworkReductionData`](@ref) docstring for the full field and accessor
# list.
