# Industry DFAX values

In this tutorial we map the industry's "DFAX" vocabulary (Distribution
Factors, as used in PSS/E's DFAX activity and downstream by the NERC IDC for
TLR and Congestion Management Procedures) onto the matrices provided by
`PowerNetworkMatrices`. Every flavor of DFAX — GSF, LSF, LODF, OTDF, transfer
DFAX, flowgate DFAX, and multi-element (N-k) DFAX — is a special case of one
unified formula. The tutorial walks through each case with executable
examples on the RTS-GMLC system.

We assume you have already worked through the [PTDF matrix](@ref) and
[LODF matrix](@ref) tutorials. For the Woodbury-identity derivation behind
post-contingency PTDF rows, see the [Flowgate Methodology](@ref) explanation.

## What is DFAX?

"DFAX" is shorthand for *distribution factor*. The term originated with
PSS/E's DFAX activity, which writes a `.dfx` file consumed by the NERC
Interchange Distribution Calculator (IDC). The IDC uses these distribution
factors during Transmission Loading Relief (TLR) procedures and Congestion
Management Procedures (CMP) — for example, applying a 5% threshold to decide
whether a given source-to-sink transfer is a "significant" contributor to a
monitored flowgate.

In practice DFAX is an umbrella term that covers several specific
quantities. The table below maps each industry term to the
`PowerNetworkMatrices` primitive that computes it:

| Industry term             | What it answers                                  | PNM primitive                                                          |
|:------------------------- |:------------------------------------------------ |:---------------------------------------------------------------------- |
| GSF / ISF                 | Δflow on `m` per 1 MW injection at bus `b`       | `PTDF[m, b]`                                                           |
| LSF                       | Same as GSF for loads (opposite sign)            | `-PTDF[m, b]`                                                          |
| LODF                      | Flow redistribution after one branch trips       | `LODF[m, c]`                                                           |
| OTDF                      | GSF with a contingency already in place          | `PTDF[m,b] + LODF[m,c]·PTDF[c,b]`, or one entry of a `VirtualMODF` row |
| Transfer DFAX (pre-cont.) | Fraction of a source→sink transfer reaching `m`  | `PTDF[m,:]·(s_v − k_v)`                                                |
| Flowgate DFAX             | Transfer DFAX on a (monitored, contingency) pair | `VirtualMODF[m, ctg]·(s_v − k_v)`                                      |
| Multi-element (N-k) DFAX  | Same with multiple simultaneous outages          | `VirtualMODF` (multi-arc `NetworkModification`)                        |

The Phase Shifter Factor (PSF) is part of the broader DFAX vocabulary but is
not a first-class primitive in `PowerNetworkMatrices`. Users who need it can
build it manually through `NetworkModification` and `Ybus`; the present
tutorial covers only flow-based distribution factors.

## The unified DFAX formula

In the DC power-flow model every flavor of DFAX is a special case of the
same quantity. For a monitored arc ``m``, a source participation vector
``s_v``, a sink participation vector ``k_v``, and a (possibly empty) set of
network modifications ``C``,

```math
\mathrm{DFAX}(m,\ s \to k,\ C) \;=\; \mathrm{PTDF}_C[m,\,:] \cdot (s_v - k_v),
```

where ``\mathrm{PTDF}_C`` is the post-modification PTDF (equal to the base
``\mathrm{PTDF}`` when ``C = \emptyset``). The formula has two degrees of
freedom — *who is shifting* (the source/sink vectors) and *what state the
network is in* (the contingency ``C``). Each section below fixes one or
both.

The reference (slack) bus is implicit in `PTDF`: a row of `PTDF` already
encodes "inject at bus ``b``, absorb at the slack". So setting ``k_v = 0``
in the formula means "let the slack absorb the sink", and the GSF section
below reduces to a single `PTDF` entry.

## Setup

All subsequent sections build on this setup block. It loads the RTS-GMLC
system and constructs the three matrices the rest of the tutorial uses.

```@repl tutorial_DFAX
using PowerSystems
using PowerNetworkMatrices
using PowerSystemCaseBuilder
using DataFrames

const PSY = PowerSystems
const PNM = PowerNetworkMatrices
const PSB = PowerSystemCaseBuilder

sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys");

ptdf = PTDF(sys);
lodf = LODF(sys);
vmodf = VirtualMODF(sys);
```

`VirtualMODF` is the most general object — it can compute post-modification
PTDF rows under any contingency. We also build `PTDF` and `LODF` up front
because the pre-contingency and single-element-outage sections below use
them directly (faster than going through Woodbury when those special cases
apply).

## GSF and LSF (no contingency, point source)

The simplest special case sets ``k_v = 0`` (the slack absorbs the sink) and
``s_v = e_b`` (a unit vector at one bus). The unified formula collapses to
a single `PTDF` entry — this is the **Generation Shift Factor**:

```@repl tutorial_DFAX
m = (107, 203);     # monitored arc AB1 (Area 1 → Area 2)
b = 101;            # injection bus in Area 1
gsf = ptdf[m, b]
```

The **Load Shift Factor** is the same quantity with the opposite sign
(loads withdraw power instead of inject):

```@repl tutorial_DFAX
lsf = -gsf
```

### Subsystem-aggregated GSF

In practice analysts care about a *subsystem* of generators (for example,
all generators in an area) rather than a single bus. Build a participation
vector by weighting each generator's bus by its `Pmax` share within the
subsystem, then dot the vector with the `PTDF` row:

```@repl tutorial_DFAX
area1_gens = filter(
    g -> PSY.get_name(PSY.get_area(PSY.get_bus(g))) == "1",
    collect(PSY.get_available_components(PSY.Generator, sys)),
);

total_pmax = sum(PSY.get_max_active_power, area1_gens);

src_weights = Dict{Int, Float64}();
for g in area1_gens
    bn = PSY.get_number(PSY.get_bus(g))
    src_weights[bn] = get(src_weights, bn, 0.0) +
                      PSY.get_max_active_power(g) / total_pmax
end

gsf_area1 = sum(w * ptdf[m, bn] for (bn, w) in src_weights)
```

`gsf_area1` is the fraction of an aggregate 1 MW dispatch increase across
all Area 1 generators (split by `Pmax`) that lands on AB1. The slack still
absorbs the corresponding withdrawal — this is a *one-sided* shift.

If the swing should be distributed across many buses instead of falling on
the single reference bus, pass a `dist_slack` dictionary to the `PTDF`
constructor (see the [PTDF matrix](@ref) tutorial). That is a different
concept from subsystem aggregation: `dist_slack` redefines the reference,
whereas the participation vector above defines the *source* of the
transfer.

## Transfer DFAX (no contingency, multi-bus source and sink)

When both source and sink are subsystems, the unified formula is the
difference of two weighted `PTDF` row dot-products:

```math
\mathrm{TDF}(m,\ s \to k) \;=\; \mathrm{PTDF}[m,\,:] \cdot s_v
                              \;-\; \mathrm{PTDF}[m,\,:] \cdot k_v.
```

Build the sink vector from Area 2 loads, max-active-power weighted. We
filter to `PSY.PowerLoad` because the abstract `ElectricLoad` type also
covers shunt admittance components (`FixedAdmittance`) that don't carry a
real-power weight:

```@repl tutorial_DFAX
area2_loads = filter(
    l -> PSY.get_name(PSY.get_area(PSY.get_bus(l))) == "2",
    collect(PSY.get_available_components(PSY.PowerLoad, sys)),
);

total_load = sum(PSY.get_max_active_power, area2_loads);

snk_weights = Dict{Int, Float64}();
for l in area2_loads
    bn = PSY.get_number(PSY.get_bus(l))
    snk_weights[bn] = get(snk_weights, bn, 0.0) +
                      PSY.get_max_active_power(l) / total_load
end
```

The pre-contingency transfer DFAX for Area 1 → Area 2 on AB1 is then:

```@repl tutorial_DFAX
tdf_pre =
    sum(w * ptdf[m, bn] for (bn, w) in src_weights) -
    sum(w * ptdf[m, bn] for (bn, w) in snk_weights)
```

`tdf_pre` answers: *if Area 1 ramps up by 1 MW (split by generator `Pmax`)
and Area 2's load grows by 1 MW (split by load size), what fraction of
that transfer shows up on AB1?* In market and TLR settings, this is the
pre-contingency component of the flowgate impact.

## OTDF (single contingency, point source)

The **Outage Transfer Distribution Factor** is the GSF you would observe
if a specific outage were already in effect. For a single-element
contingency on arc ``c``, OTDF has a closed-form expression in terms of
`PTDF` and `LODF`:

```math
\mathrm{OTDF}(m, b, c) \;=\; \mathrm{PTDF}[m, b] + \mathrm{LODF}[m, c] \cdot \mathrm{PTDF}[c, b].
```

This is the unified formula with ``C = \{c\}`` and the slack absorbing the
sink. `VirtualMODF` computes the same quantity through the Woodbury
identity, which generalizes naturally to multi-element contingencies (see
the N-k section below). For a single outage the two routes agree:

```@repl tutorial_DFAX
c = (113, 215);                       # contingency: AB2 outage
otdf_closed = ptdf[m, b] + lodf[m, c] * ptdf[c, b]

ctg = NetworkModification(vmodf, c);
row_c = vmodf[m, ctg];
bus_lookup = PNM.get_bus_lookup(vmodf);
otdf_vmodf = row_c[bus_lookup[b]]

isapprox(otdf_closed, otdf_vmodf; rtol = 1e-10)
```

The `isapprox` check is the tutorial's internal validation: `VirtualMODF`
and the closed-form LODF expansion are the same calculation expressed two
different ways. Whenever both apply, they agree to floating-point
tolerance.

## Flowgate DFAX (single contingency, source–sink transfer)

A *flowgate* in NERC parlance is the pair `(monitored facility, contingency)`. The flowgate DFAX is the unified formula with both
nontrivial source/sink vectors and a nonempty ``C``: the source–sink
subtraction from the transfer-DFAX section applied to the post-contingency
row from the OTDF section. We reuse `row_c`, `src_weights`, and
`snk_weights` already in scope:

```@repl tutorial_DFAX
flowgate_dfax =
    sum(w * row_c[bus_lookup[bn]] for (bn, w) in src_weights) -
    sum(w * row_c[bus_lookup[bn]] for (bn, w) in snk_weights)
```

The NERC 5% rule treats a transfer as a "significant" contributor to a
flowgate when the absolute DFAX exceeds 0.05. The check is one line:

```@repl tutorial_DFAX
significant = abs(flowgate_dfax) >= 0.05
```

When `significant == true`, the transfer is subject to curtailment or
mitigation under the relevant TLR / CMP procedure.

## N-k DFAX (multi-element contingency)

When the contingency `C` contains more than one element, the closed-form
LODF expansion of the OTDF section no longer applies — there is no scalar
`LODF[m, c]` when `c` is itself a set. The unified formula still applies,
and `VirtualMODF` is built to handle it directly. Build the multi-element
modification by merging the `arc_modifications` of each single-arc
`NetworkModification` into one combined object:

```@repl tutorial_DFAX
mod_ab2 = NetworkModification(vmodf, (113, 215));    # AB2 outage
mod_ab3 = NetworkModification(vmodf, (123, 217));    # AB3 outage

ctg_n2 = NetworkModification(
    "AB2_and_AB3_out",
    vcat(collect(mod_ab2.arc_modifications),
        collect(mod_ab3.arc_modifications)),
);

row_n2 = vmodf[m, ctg_n2];

flowgate_dfax_n2 =
    sum(w * row_n2[bus_lookup[bn]] for (bn, w) in src_weights) -
    sum(w * row_n2[bus_lookup[bn]] for (bn, w) in snk_weights)
```

Removing two of the three parallel Area 1 → Area 2 paths forces a much
larger fraction of any inter-area transfer onto AB1, so `flowgate_dfax_n2`
is substantially larger than the N-1 value computed in the previous
section. The same indexing call (`vmodf[m, ctg]`) handles N-1, N-2, and
higher orders — that is the operational advantage of going through
`VirtualMODF`.

## Capstone: assembling a DFAX report

In production use, an analyst typically wants a *table* of distribution
factors covering several transfers, several monitored facilities, and
several contingencies — the kind of report that a PSS/E `.dfx` file plus
IDC post-processing produces. Building that report is one nested loop
around the unified formula.

For this example we use one transfer (Area 1 → Area 2 from the transfer
DFAX section), three monitored arcs (the three parallel Area 1 → Area 2
paths), and three contingencies (each of the other two paths individually,
plus the N-2 double-outage from the previous section). Each contingency
carries the set of arc tuples it outages so that we can skip the
ill-defined case of monitoring an outaged element:

```@repl tutorial_DFAX
monitored = [(107, 203), (113, 215), (123, 217)];   # AB1, AB2, AB3

ctg_ab2 = NetworkModification(vmodf, (113, 215));
ctg_ab3 = NetworkModification(vmodf, (123, 217));
ctg_ab2_ab3 = NetworkModification(
    "AB2_and_AB3_out",
    vcat(collect(ctg_ab2.arc_modifications),
        collect(ctg_ab3.arc_modifications)),
);

contingencies = [
    ("AB2 out", Set([(113, 215)]), ctg_ab2),
    ("AB3 out", Set([(123, 217)]), ctg_ab3),
    ("AB2 & AB3 out", Set([(113, 215), (123, 217)]), ctg_ab2_ab3),
];

rows = NamedTuple[]
for mon in monitored
    for (label, outaged, ctg_k) in contingencies
        mon in outaged && continue
        row = vmodf[mon, ctg_k]
        df =
            sum(w * row[bus_lookup[bn]] for (bn, w) in src_weights) -
            sum(w * row[bus_lookup[bn]] for (bn, w) in snk_weights)
        push!(
            rows,
            (
                monitored = mon,
                contingency = label,
                dfax = df,
                significant = abs(df) >= 0.05,
            ),
        )
    end
end

report = sort(DataFrame(rows), :dfax; by = abs, rev = true)
```

The sort by `abs(dfax)` puts the largest flowgate impacts at the top — the
ones a TLR coordinator would investigate first. Filtering to
`report[report.significant, :]` would keep only NERC-significant rows.

This table is the kind of output that drives downstream congestion and
seams-coordination workflows; assembling it requires nothing beyond the
matrices in this tutorial.

## When to use which primitive

| Need                                                | Reach for                                     |
|:--------------------------------------------------- |:--------------------------------------------- |
| Many transfers, no contingencies                    | `PTDF`                                        |
| One contingency, all monitored branches             | `LODF` + `PTDF`                               |
| Specific flowgates `(monitored, contingency)` pairs | `VirtualMODF`                                 |
| Multi-element / N-k contingencies                   | `VirtualMODF`                                 |
| Memory-constrained or sparse usage                  | `VirtualPTDF` / `VirtualLODF` / `VirtualMODF` |

`VirtualMODF` is strictly more general than `LODF`-based OTDF arithmetic,
but the closed-form route in the OTDF section is faster when you only have
a single outage and many bus injections to evaluate. The decision is about
which *direction* of the matrix you traverse most often, not about which
one is "correct".
