# Tests for the π-model branch admittance helpers (`branch_admittance`,
# `reduced_arc_admittance`, `winding_admittance`, `three_winding_arcs`, `branch_flow_limits`).
# Ported from PowerOperationsModels.jl's native DCP/ACP model tests, where this logic
# previously lived as private `_*` helpers before being promoted to PNM's public API.

@testset "branch_admittance primitives" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    line = first(PSY.get_components(PSY.Line, sys))
    a = PNM.branch_admittance(line)
    r, x = PSY.get_r(line, PSY.SU), PSY.get_x(line, PSY.SU)
    y = inv(complex(r, x))
    @test a.g ≈ real(y)
    @test a.b ≈ imag(y)
    @test a.tap == 1.0
    @test a.shift == 0.0
end

@testset "branch_flow_limits MonitoredLine" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    ml = first(PSY.get_components(PSY.MonitoredLine, sys))
    fl = PNM.branch_flow_limits(ml)
    psy_fl = PSY.get_flow_limits(ml, PSY.DU)
    @test fl.from_to == psy_fl.from_to
    @test fl.to_from == psy_fl.to_from
end

@testset "reduced arc admittance uses PNM series equivalent, not original branch" begin
    # `case10_radial_series_reductions` is purpose-built to produce series arcs under the
    # radial + degree-two reduction, exercising the same NetworkReductionData the build path
    # stores on the network model.
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    ybus = PNM.Ybus(
        sys;
        network_reductions = PNM.NetworkReduction[
            PNM.RadialReduction(),
            PNM.DegreeTwoReduction(),
        ],
    )
    nr = deepcopy(PNM.get_network_reduction_data(ybus))
    @test !isempty(nr)

    series_map = PNM.get_series_branch_map(nr)
    @test !isempty(series_map)  # degree-2 reduction produces series arcs

    (from_no, to_no), chain = first(series_map)
    resolved = PNM.reduced_arc_admittance(nr, from_no, to_no)
    @test resolved !== nothing
    expected = PNM.branch_admittance(chain, nr)
    @test isapprox(resolved.b, expected.b; atol = 1e-9)

    # Non-triviality: the series equivalent is the MERGED admittance of the chain, so it must
    # differ from any single constituent branch's own admittance. This is the whole point of
    # leveraging the reduction-aware equivalent rather than a single branch's value. Compare
    # against a plain `Line` member — PNM wrapper members (nested parallel/series segments,
    # 3W windings) resolve through their dedicated `branch_admittance`/`winding_admittance`
    # methods, not the single-arg physical-branch form.
    members = collect(chain)
    @test length(members) >= 2
    line_members = filter(m -> m isa PSY.Line, members)
    if !isempty(line_members)
        member_b = PNM.branch_admittance(line_members[1]).b
        @test !isapprox(resolved.b, member_b; rtol = 1e-3)
    end

    # Reversed-orientation arc exercises the `_reverse_admittance` path: series b is symmetric,
    # from/to shunts swap, and any phase shift negates.
    if !haskey(series_map, (to_no, from_no))
        reversed = PNM.reduced_arc_admittance(nr, to_no, from_no)
        @test reversed !== nothing
        @test isapprox(reversed.b, resolved.b; atol = 1e-9)
        @test isapprox(reversed.b_fr, resolved.b_to; atol = 1e-9)
        @test isapprox(reversed.shift, -resolved.shift; atol = 1e-12)
    end

    # A direct (un-reduced) arc resolves to `nothing` — the caller falls back to the branch's
    # own admittance.
    @test PNM.reduced_arc_admittance(nr, -1, -2) === nothing
end

# Build a `ThreeWindingTransformer` (new struct shape) into `sys`, wiring three terminal
# buses to a hidden star bus. Pairwise impedances are stored device-base on `bp` (= system
# base here, so SU == DU keeps the hand-computed literals clean); each winding carries its
# own arc, base power, base voltage, and rating. Returns the attached transformer.
function _add_three_winding_transformer!(
    sys,
    busP,
    busS,
    busT,
    star_bus;
    name = "T3W",
    r12 = 0.01, x12 = 0.1,
    r23 = 0.01, x23 = 0.1,
    r13 = 0.01, x13 = 0.1,
    bp = 100.0,
    magnetizing_shunt = 0.0 + 0.0im,
    ratings = (1.0, 1.0, 0.5),
)
    arcs = (
        PSY.Arc(; from = busP, to = star_bus),
        PSY.Arc(; from = busS, to = star_bus),
        PSY.Arc(; from = busT, to = star_bus),
    )
    foreach(a -> PSY.add_component!(sys, a), arcs)
    windings = ntuple(
        i -> PSY.TransformerWinding(;
            arc = arcs[i],
            available = true,
            base_power = bp,
            base_voltage = PSY.get_base_voltage(PSY.get_from(arcs[i])),
            rating = ratings[i],
        ),
        3,
    )
    t3w = PSY.ThreeWindingTransformer(;
        name = name,
        primary_winding = windings[1],
        secondary_winding = windings[2],
        tertiary_winding = windings[3],
        star_bus = star_bus,
        r_12 = r12, x_12 = x12,
        r_23 = r23, x_23 = x23,
        r_13 = r13, x_13 = x13,
        base_power_12 = bp, base_power_23 = bp, base_power_13 = bp,
        magnetizing_shunt = magnetizing_shunt,
    )
    PSY.add_component!(sys, t3w)
    return t3w
end

function _add_star_buses!(sys, busD; numbers = (101, 102, 103))
    return map(numbers) do n
        b = PSY.ACBus(;
            number = n,
            name = "Bus3WT_$n",
            available = true,
            bustype = PSY.ACBusTypes.PQ,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.95, max = 1.05),
            base_voltage = 230.0,
            area = PSY.get_area(busD),
            load_zone = PSY.get_load_zone(busD),
        )
        PSY.add_component!(sys, b)
        b
    end
end

@testset "ThreeWindingTransformer winding_admittance and three_winding_arcs decomposition" begin
    # Unit test the per-winding admittance helper against a real PNM
    # `ThreeWindingTransformerWinding`: for a winding whose derived star-leg impedance is
    # R + jX the helper must return the series admittance 1/(R + jX), the winding's PNM shunt
    # on the from/to sides, no phase shift, and (here) a unit tap. R/X are read back through
    # PNM so the assertion is robust to per-unit base conversions.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD)
    transformer3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus; name = "Transformer3W_busD",
    )

    w = PNM.ThreeWindingTransformerWinding(transformer3w, 1)
    adm = PNM.winding_admittance(w)

    r = PNM.get_equivalent_r(w)
    x = PNM.get_equivalent_x(w)
    y = inv(complex(r, x))
    @test isapprox(adm.g, real(y); atol = 1e-12)
    @test isapprox(adm.b, imag(y); atol = 1e-12)

    b_sh = PNM.get_equivalent_b(w)
    @test adm.g_fr == 0.0
    @test adm.b_fr == b_sh.from
    @test adm.g_to == 0.0
    @test adm.b_to == b_sh.to
    @test adm.tap == 1.0

    # `three_winding_arcs` decomposes the device into its three windings, exposing the
    # star-point arc, rating, and winding object the native builders consume.
    arcs = PNM.three_winding_arcs(transformer3w)
    windings = PSY.get_windings(transformer3w)
    @test length(arcs) == 3
    @test [a.suffix for a in arcs] == ["winding_1", "winding_2", "winding_3"]
    @test arcs[1].arc == PSY.get_arc(windings[1])
    @test arcs[2].arc == PSY.get_arc(windings[2])
    @test arcs[3].arc == PSY.get_arc(windings[3])
    # Winding admittance computed from the decomposition matches the standalone helper.
    @test PNM.winding_admittance(arcs[1].winding).b ≈ adm.b
end

@testset "ThreeWindingTransformerWinding star derivation (hand-computed)" begin
    # Asymmetric pairwise impedances, all on the system base (bp == 100), so SU == DU and the
    # star-leg identity applies directly to the literals below. The identity is
    #   z1 = (z12 + z13 - z23)/2, z2 = (z12 + z23 - z13)/2, z3 = (z13 + z23 - z12)/2.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD; numbers = (201, 202, 203))
    # Values chosen so no star leg derives to ~0 (flooring is exercised separately below).
    r12, x12 = 0.03, 0.20
    r23, x23 = 0.04, 0.30
    r13, x13 = 0.05, 0.40
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus;
        name = "T3W_star_math",
        r12 = r12, x12 = x12, r23 = r23, x23 = x23, r13 = r13, x13 = x13,
    )
    z12 = complex(r12, x12)
    z23 = complex(r23, x23)
    z13 = complex(r13, x13)
    expected = (
        (z12 + z13 - z23) / 2,
        (z12 + z23 - z13) / 2,
        (z13 + z23 - z12) / 2,
    )
    for i in 1:3
        w = PNM.ThreeWindingTransformerWinding(t3w, i)
        @test isapprox(PNM.get_equivalent_r(w), real(expected[i]); atol = 1e-12)
        @test isapprox(PNM.get_equivalent_x(w), imag(expected[i]); atol = 1e-12)
    end
    # Hand-computed spot check for winding 1: r1 = (0.03 + 0.05 - 0.04)/2 = 0.02,
    # x1 = (0.20 + 0.40 - 0.30)/2 = 0.15.
    w1 = PNM.ThreeWindingTransformerWinding(t3w, 1)
    @test isapprox(PNM.get_equivalent_r(w1), 0.02; atol = 1e-12)
    @test isapprox(PNM.get_equivalent_x(w1), 0.15; atol = 1e-12)
end

@testset "PST-3W winding series susceptance (pinned behavior)" begin
    # `pti_case14_with_pst3w_sys` is the only fixture with a genuine phase-shifting
    # ThreeWindingTransformer (nonzero winding α parsed from PSS/E ANG1/ANG2/ANG3 fields).
    # No fixture currently exercises a PST-3W through PNM's series-reduction paths
    # (BranchesSeries/BranchesParallel), so this testset exists purely to PIN the wrapper's
    # `get_series_susceptance` model against silent drift — see the note below and
    # `docs/superpowers/plans/2026-07-08-transformer-refactor-pnm.md` ("Residual decision
    # items") for the modeling decision this leaves open.
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "pti_case14_with_pst3w_sys";
        force_build = true,
    )
    t = first(
        Iterators.filter(
            PSY.is_phase_shifting,
            PSY.get_components(PSY.ThreeWindingTransformer, sys),
        ),
    )
    windings = PSY.get_windings(t)
    winding_number = findfirst(w -> !iszero(PSY.get_α(w)), windings)
    @test winding_number !== nothing
    tw = PNM.ThreeWindingTransformerWinding(t, winding_number)
    @test !iszero(PSY.get_α(PSY.get_windings(t)[winding_number]))

    # (a) Pinned current model: r-aware `imag(1/Z)` of the derived star leg, computed from
    # the wrapper's own stored fields.
    pinned = imag(1 / (tw.r + tw.x * im))
    @test PSY.get_series_susceptance(tw, PSY.SU) ≈ pinned

    # (b) Independent hand-derivation from the fixture's raw pairwise data, so (a) is not
    # purely self-referential. `case14_with_pst3w.raw`'s two 3W transformers both carry
    # r_12 = r_23 = r_13 = 0.0, x_12 = x_23 = x_13 = 0.0002 pu on their own (100 MVA) base,
    # which equals the system base here, so SU reads back the raw values unchanged. The
    # standard delta->star identity gives every star leg r = (0+0-0)/2 = 0.0,
    # x = (0.0002+0.0002-0.0002)/2 = 0.0001 (independent of which winding), well above
    # `STAR_LEG_ZERO_REACTANCE_ATOL` so no flooring applies. Susceptance is then
    # imag(1/(0 + j*0.0001)) = imag(-j*10000) = -10000.0 exactly.
    r12, x12 = PSY.get_r_12(t, PSY.SU), PSY.get_x_12(t, PSY.SU)
    r23, x23 = PSY.get_r_23(t, PSY.SU), PSY.get_x_23(t, PSY.SU)
    r13, x13 = PSY.get_r_13(t, PSY.SU), PSY.get_x_13(t, PSY.SU)
    z12, z23, z13 = complex(r12, x12), complex(r23, x23), complex(r13, x13)
    z_by_winding = (
        (z12 + z13 - z23) / 2,
        (z12 + z23 - z13) / 2,
        (z13 + z23 - z12) / 2,
    )
    z_star = z_by_winding[winding_number]
    hand_derived_susceptance = imag(1 / z_star)
    @test hand_derived_susceptance ≈ -10000.0
    @test PSY.get_series_susceptance(tw, PSY.SU) ≈ hand_derived_susceptance

    # !!! note "This DIFFERS from pre-refactor PhaseShiftingTransformer3W semantics"
    #     The old PSY `PhaseShiftingTransformer3W` computed series susceptance per leg as
    #     `(1/x)/turns_ratio` — r-free, and divided by the winding's turns ratio (tap) —
    #     wherever a PST-3W entered PNM's series-reduction sums. The new wrapper instead
    #     returns `imag(1/(r+jx))` uniformly for ALL `ThreeWindingTransformer` legs
    #     (phase-shifting or not), matching the plain `Transformer3W` model instead: r-aware,
    #     tap/turns-ratio-free. For this fixture's star legs (r=0) the two forms happen to
    #     coincide numerically at unit tap (`(1/x)/1.0 == imag(1/(j*x))`), which is exactly
    #     why the divergence has been unobservable so far — it is a LATENT gap, not a proven
    #     equivalence: any PST-3W with r != 0 or a non-unit turns ratio on the star leg would
    #     produce different reduction susceptances under the two models.
    #     This divergence is only reachable through series-reduction code paths
    #     (`BranchesSeries`/`BranchesParallel`, `network_modification.jl`,
    #     `virtual_factor_helpers.jl`) — no current fixture routes a PST-3W through those
    #     paths, so it does not affect any passing test today. Changing the model (e.g. to
    #     apply turns-ratio division, or to special-case phase-shifting legs) is a modeling
    #     decision, not a bug fix; it is tracked as a residual decision item in
    #     `docs/superpowers/plans/2026-07-08-transformer-refactor-pnm.md`. Do not "fix" this
    #     unilaterally.
end

@testset "winding_admittance tap contract: plain 3W winding tap flows through (deliberate change)" begin
    # !!! note "Deliberate contract change"
    #     Old world: `_winding_tap`/`branch_admittance` reported a hardcoded `1.0` for every
    #     3W winding EXCEPT a phase-shifting one (which read the real winding tap). The
    #     refactored `winding_admittance` now reads `get_equivalent_tap(w)` (== the winding's
    #     own `PSY.get_tap`) unconditionally, for PST and plain 3W windings alike. This test
    #     pins that a plain (non-phase-shifting) winding with a non-unit tap flows its real
    #     tap through `winding_admittance`, rather than being hardcoded to `1.0`.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD; numbers = (501, 502, 503))
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus;
        name = "T3W_nonunit_tap",
    )
    winding1 = PSY.get_windings(t3w)[1]
    PSY.set_tap!(winding1, 1.05)
    w1 = PNM.ThreeWindingTransformerWinding(t3w, 1)
    adm = PNM.winding_admittance(w1)
    @test adm.tap == 1.05
    @test adm.tap != 1.0
end

@testset "ThreeWindingTransformerWinding zero-reactance flooring" begin
    # Choose pairwise reactances so the primary star leg derives to exactly zero:
    # x1 = (x12 + x13 - x23)/2 = (0.05 + 0.05 - 0.10)/2 = 0.0. The wrapper must floor it to
    # `STAR_LEG_REACTANCE_FLOOR` (recovered from the deleted PowerFlowFileParser rule); the
    # frozen baseline for `psse_4_zero_impedance_3wt_test_system` implies exactly this: its
    # zero-reactance star leg carries Y_t = 1/(r + j*1e-4).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD; numbers = (301, 302, 303))
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus;
        name = "T3W_zero_x",
        r12 = 0.01, x12 = 0.05, r23 = 0.01, x23 = 0.10, r13 = 0.01, x13 = 0.05,
    )
    w1 = PNM.ThreeWindingTransformerWinding(t3w, 1)
    @test PNM.get_equivalent_x(w1) == PNM.STAR_LEG_REACTANCE_FLOOR
    @test PNM.STAR_LEG_REACTANCE_FLOOR == 1e-4
    # Non-floored legs are unaffected: x2 = (0.05 + 0.10 - 0.05)/2 = 0.05.
    w2 = PNM.ThreeWindingTransformerWinding(t3w, 2)
    @test isapprox(PNM.get_equivalent_x(w2), 0.05; atol = 1e-12)
end

@testset "ThreeWindingTransformerWinding lookup identity is mutation-insensitive" begin
    # Lookup identity must be `{parent, winding_number}`, not full field-egal equality: `r`/`x`
    # are a build-time snapshot of the parent's pairwise data, so a wrapper constructed
    # before a pairwise-impedance mutation and one constructed after must still be treated as
    # "the same winding" by `==`/`hash`/Dict lookups.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD; numbers = (401, 402, 403))
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus;
        name = "T3W_identity",
    )

    w1_before = PNM.ThreeWindingTransformerWinding(t3w, 1)
    PSY.set_x_12!(t3w, 0.9 * PSY.SU)
    w1_after = PNM.ThreeWindingTransformerWinding(t3w, 1)

    # Same parent + winding number: equal and hash-equal even though the mutation changed the
    # derived `r`/`x` snapshot between the two constructions.
    @test w1_before.x != w1_after.x
    @test w1_before == w1_after
    @test hash(w1_before) == hash(w1_after)

    # Different winding number on the same parent: unequal.
    w2 = PNM.ThreeWindingTransformerWinding(t3w, 2)
    @test w1_after != w2
    @test hash(w1_after) != hash(w2)

    # Dict lookup round-trip: a fresh wrapper (rebuilt after further mutation) must resolve
    # the same map entry as the wrapper originally used as the key.
    d = Dict(w1_before => "winding_1")
    PSY.set_r_12!(t3w, 0.02 * PSY.SU)
    w1_rebuilt = PNM.ThreeWindingTransformerWinding(t3w, 1)
    @test d[w1_rebuilt] == "winding_1"
end
