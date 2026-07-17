# # How to Compute Phase Shifter Factors

# A **Phase Shifter Factor** (PSF) is the sensitivity of a monitored branch flow to
# the phase angle of a phase-shifting transformer. It belongs to the DFAX family
# (see [Reproduce industry DFAX values](@ref)), but it is **not** a first-class
# primitive in `PowerNetworkMatrices`: a phase-shifting transformer is not a
# topology change, so you cannot build it through [`NetworkModification`](@ref) —
# in fact contingencies on a [`PhaseShiftingTransformer`](@extref
# PowerSystems.PhaseShiftingTransformer) are explicitly unsupported. Instead the
# phase shift enters the DC model as a pair of nodal injections, and the factor
# falls straight out of the [`PTDF`](@ref).

# ## The recipe

# In the DC power-flow model a phase shift ``\alpha`` on the branch ``c`` from bus
# ``f`` to bus ``t``, with series susceptance ``b_c``, is equivalent to injecting
# ``+b_c\,\alpha`` at ``f`` and ``-b_c\,\alpha`` at ``t``. Because the [`PTDF`](@ref)
# already maps injections to branch flows, the change in flow on a monitored arc
# ``m`` per radian of shift is

# ```math
# \mathrm{PSF}[m, c] \;=\; b_c \,\bigl(\mathrm{PTDF}[m, f] - \mathrm{PTDF}[m, t]\bigr).
# ```

# The two [`PTDF`](@ref) columns are the injection pair; ``b_c`` scales the angle
# into an injection.

# ## Prerequisites
#
#   - `PowerNetworkMatrices` and `PowerSystems` installed
#   - A [`System`](@extref PowerSystems.System) containing a
#     [`PhaseShiftingTransformer`](@extref PowerSystems.PhaseShiftingTransformer)

import PowerSystems as PSY
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# We start from `c_sys5` and add a phase-shifting transformer so the example is
# self-contained. Pick two existing buses, connect them with a new
# [`Arc`](@extref PowerSystems.Arc), and place a
# [`PhaseShiftingTransformer`](@extref PowerSystems.PhaseShiftingTransformer) on it:

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

buses = collect(PSY.get_components(PSY.ACBus, sys));
arc = PSY.Arc(; from = buses[1], to = buses[2]);
PSY.add_component!(sys, arc);

pst = PSY.PhaseShiftingTransformer(;
    name = "PST_demo",
    available = true,
    active_power_flow = 0.0,
    reactive_power_flow = 0.0,
    arc = arc,
    r = 0.0,
    x = 0.2,
    primary_shunt = 0.0,
    tap = 1.0,
    α = 0.0,
    rating = 1.0,
    base_power = 100.0,
);
PSY.add_component!(sys, pst);

# ## Read the phase shifter's data

# The factor needs three numbers from the phase shifter: its from-bus, its to-bus,
# and its series susceptance. The bus numbers come from the shifter's
# [`Arc`](@extref PowerSystems.Arc); the susceptance is `get_series_susceptance`,
# which for a [`PhaseShiftingTransformer`](@extref
# PowerSystems.PhaseShiftingTransformer) returns the tap-adjusted ``1/(a\,x)`` and,
# correctly, ignores the phase angle:

f = PSY.get_number(PSY.get_from(PSY.get_arc(pst)));
t = PSY.get_number(PSY.get_to(PSY.get_arc(pst)));
b_c = PSY.get_series_susceptance(pst)

# ## Compute the factor

# Build the [`PTDF`](@ref) — it now includes the shifter's arc — and apply the
# formula for one monitored arc:

ptdf = PNM.PTDF(sys);

m = (4, 5);     # monitored arc
psf = b_c * (ptdf[m, f] - ptdf[m, t])

# `psf` is the megawatt change on arc `m` per radian of phase shift (per-unit on
# the system base). A positive value means advancing the shifter's angle pushes
# flow onto `m`.

# ## All monitored arcs at once

# The same injection pair drives every monitored arc, so the whole PSF column is a
# single matrix–vector product against the [`PTDF`](@ref). Build the injection
# vector — `+b_c` at the from-bus, `-b_c` at the to-bus — and read one factor per
# monitored arc:

bus_lookup = PNM.get_bus_lookup(ptdf);
injection = zeros(size(PNM.get_ptdf_data(ptdf), 2));
injection[bus_lookup[f]] = b_c;
injection[bus_lookup[t]] = -b_c;

psf_all = PNM.get_ptdf_data(ptdf) * injection;

# The entry for arc `m` matches the scalar computed above:

isapprox(psf_all[PNM.get_arc_lookup(ptdf)[m]], psf; rtol = 1e-10)

# ## See Also
#
#   - [Reproduce industry DFAX values](@ref) — the rest of the DFAX family
#   - [Flowgate Methodology](@ref) — how injections map to flows through the PTDF
