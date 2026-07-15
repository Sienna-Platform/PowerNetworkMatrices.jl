# Tests for the π-model branch admittance helpers (`branch_admittance`,
# `reduced_arc_admittance`, `winding_admittance`, `three_winding_arcs`, `branch_flow_limits`).

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

# Build a `ThreeWindingTransformer` into `sys`, wiring three terminal
# buses to a hidden star bus. The circuit-resident star-leg series impedances are derived
# from the pairwise data here (as PFFP does at parse) and stored per circuit on `bp`
# (= system base here, so SU == DU keeps the hand-computed literals clean); the pairwise data
# stays on the parent. The magnetizing shunt and its location live on the parent transformer.
# Each circuit carries its own arc, base power, base voltages, and rating. Returns the
# attached transformer.
function _add_three_winding_transformer!(
    sys,
    busP,
    busS,
    busT,
    star_bus;
    name = "T3W",
    r12 = 0.01, x12 = 0.1,
    r23 = 0.01, x23 = 0.1,
    r31 = 0.01, x31 = 0.1,
    bp = 100.0,
    magnetizing_shunt = 0.0 + 0.0im,
    shunt_location = PSY.ThreeWindingTransformerShuntLocation.PRIMARY,
    ratings = (1.0, 1.0, 0.5),
)
    arcs = (
        PSY.Arc(; from = busP, to = star_bus),
        PSY.Arc(; from = busS, to = star_bus),
        PSY.Arc(; from = busT, to = star_bus),
    )
    foreach(a -> PSY.add_component!(sys, a), arcs)
    z12, z23, z31 = complex(r12, x12), complex(r23, x23), complex(r31, x31)
    legs = (
        (z12 + z31 - z23) / 2,
        (z12 + z23 - z31) / 2,
        (z31 + z23 - z12) / 2,
    )
    circuits = ntuple(
        i -> PSY.TransformerCircuit(;
            arc = arcs[i],
            available = true,
            base_power = bp,
            base_voltage_primary = PSY.get_base_voltage(PSY.get_from(arcs[i])),
            r = real(legs[i]),
            x = imag(legs[i]),
            rating = ratings[i],
        ),
        3,
    )
    t3w = PSY.ThreeWindingTransformer(;
        name = name,
        primary_circuit = circuits[1],
        secondary_circuit = circuits[2],
        tertiary_circuit = circuits[3],
        star_bus = star_bus,
        r_12 = r12, x_12 = x12,
        r_23 = r23, x_23 = x23,
        r_31 = r31, x_31 = x31,
        base_power_12 = bp, base_power_23 = bp, base_power_31 = bp,
        magnetizing_shunt = magnetizing_shunt,
        shunt_location = shunt_location,
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
    # Unit test the per-circuit admittance helper against a real PNM
    # `ThreeWindingTransformerCircuit`: for a circuit whose derived star-leg impedance is
    # R + jX the helper must return the series admittance 1/(R + jX), the parent's PNM shunt
    # on the from/to sides, no phase shift, and (here) a unit tap. R/X are read back through
    # PNM so the assertion is robust to per-unit base conversions.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD)
    transformer3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus; name = "Transformer3W_busD",
    )

    w = PNM.ThreeWindingTransformerCircuit(transformer3w, 1)
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

    # `three_winding_arcs` decomposes the device into its three circuits, exposing the
    # star-point arc, rating, and circuit object the native builders consume.
    arcs = PNM.three_winding_arcs(transformer3w)
    circuits = PSY.get_circuits(transformer3w)
    @test length(arcs) == 3
    @test [a.suffix for a in arcs] == ["winding_1", "winding_2", "winding_3"]
    @test arcs[1].arc == PSY.get_arc(circuits[1])
    @test arcs[2].arc == PSY.get_arc(circuits[2])
    @test arcs[3].arc == PSY.get_arc(circuits[3])
    # Circuit admittance computed from the decomposition matches the standalone helper.
    @test PNM.winding_admittance(arcs[1].circuit).b ≈ adm.b
end

@testset "PST-3W winding series susceptance (pinned behavior)" begin
    # `pti_case14_with_pst3w_sys` is the only fixture with a genuine phase-shifting
    # ThreeWindingTransformer (nonzero winding α parsed from PSS/E ANG1/ANG2/ANG3 fields).
    # This testset pins the wrapper's `get_series_susceptance` model — `(1/x)/tap` on the
    # derived star leg, uniform with the `TwoWindingTransformer` and `ACTransmission`
    # conventions in BranchAdmittance.jl — against silent drift.
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
    windings = PSY.get_circuits(t)
    winding_number = findfirst(w -> !iszero(PSY.get_α(w)), windings)
    @test winding_number !== nothing
    tw = PNM.ThreeWindingTransformerCircuit(t, winding_number)
    @test !iszero(PSY.get_α(PSY.get_circuits(t)[winding_number]))

    # (a) Pinned model: reactance-only `1/x` of the winding's star leg, divided by the
    # winding tap, read back through the wrapper.
    tap = PSY.get_tap(windings[winding_number])
    pinned = (1 / PNM.get_equivalent_x(tw)) / tap
    @test PNM.get_series_susceptance(tw, PSY.SU) ≈ pinned

    # (b) Independent hand-derivation from the fixture's raw pairwise data, so (a) is not
    # purely self-referential. `case14_with_pst3w.raw`'s two 3W transformers both carry
    # r_12 = r_23 = r_31 = 0.0, x_12 = x_23 = x_31 = 0.0002 pu on their own (100 MVA) base,
    # which equals the system base here, so SU reads back the raw values unchanged. The
    # standard delta->star identity (applied by PFFP at parse) gives every star leg
    # r = (0+0-0)/2 = 0.0, x = (0.0002+0.0002-0.0002)/2 = 0.0001 (independent of which
    # winding). The phase-shifting winding carries tap = 1.0, so the susceptance is
    # (1/0.0001)/1.0 = +10000.0 exactly. Note the sign: `1/x` is positive for x > 0, whereas
    # the r-aware complex form `imag(1/(j*x)) = -1/x` is negative — the two forms are NOT
    # interchangeable.
    r12, x12 = PSY.get_r_12(t, PSY.SU), PSY.get_x_12(t, PSY.SU)
    r23, x23 = PSY.get_r_23(t, PSY.SU), PSY.get_x_23(t, PSY.SU)
    r31, x31 = PSY.get_r_31(t, PSY.SU), PSY.get_x_31(t, PSY.SU)
    z12, z23, z31 = complex(r12, x12), complex(r23, x23), complex(r31, x31)
    z_by_winding = (
        (z12 + z31 - z23) / 2,
        (z12 + z23 - z31) / 2,
        (z31 + z23 - z12) / 2,
    )
    z_star = z_by_winding[winding_number]
    hand_derived_susceptance = (1 / imag(z_star)) / tap
    @test hand_derived_susceptance ≈ 10000.0
    @test PNM.get_series_susceptance(tw, PSY.SU) ≈ hand_derived_susceptance

    # (c) Tap division: winding 3 of this transformer carries a non-unit tap (1.05) on the
    # same star-leg reactance (0.0001), so its susceptance must be 10000/1.05.
    tap3 = PSY.get_tap(windings[3])
    @test tap3 == 1.05
    tw3 = PNM.ThreeWindingTransformerCircuit(t, 3)
    @test PNM.get_series_susceptance(tw3, PSY.SU) ≈ 10000.0 / 1.05
end

@testset "winding_admittance applies the winding tap for all 3W windings" begin
    # !!! note "Tap contract"
    #     `winding_admittance` reads `get_equivalent_tap(w)` (== the winding's own
    #     `PSY.get_tap`) for all 3W windings, phase-shifting or not. This test pins that a
    #     plain (non-phase-shifting) winding with a non-unit tap flows its real tap through
    #     `winding_admittance`.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD; numbers = (501, 502, 503))
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus;
        name = "T3W_nonunit_tap",
    )
    winding1 = PSY.get_circuits(t3w)[1]
    PSY.set_tap!(winding1, 1.05)
    w1 = PNM.ThreeWindingTransformerCircuit(t3w, 1)
    adm = PNM.winding_admittance(w1)
    @test adm.tap == 1.05
    @test adm.tap != 1.0
end

@testset "ThreeWindingTransformerCircuit lookup identity" begin
    # Lookup identity is `{parent, winding_number}`: two wrappers for the same winding of the
    # same parent are `==`/`hash`-equal, so a fresh wrapper resolves a Dict/Set entry keyed by
    # an earlier-built one.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD; numbers = (401, 402, 403))
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus;
        name = "T3W_identity",
    )

    w1_a = PNM.ThreeWindingTransformerCircuit(t3w, 1)
    w1_b = PNM.ThreeWindingTransformerCircuit(t3w, 1)

    # Same parent + winding number: equal and hash-equal.
    @test w1_a == w1_b
    @test hash(w1_a) == hash(w1_b)

    # Different winding number on the same parent: unequal.
    w2 = PNM.ThreeWindingTransformerCircuit(t3w, 2)
    @test w1_a != w2
    @test hash(w1_a) != hash(w2)

    # Dict lookup round-trip: a fresh wrapper must resolve the same map entry as the wrapper
    # originally used as the key.
    d = Dict(w1_a => "winding_1")
    w1_rebuilt = PNM.ThreeWindingTransformerCircuit(t3w, 1)
    @test d[w1_rebuilt] == "winding_1"
end

@testset "TwoWindingTransformer series susceptance divides by the winding tap" begin
    sys = PSY.System(100.0)
    busA = PSY.ACBus(;
        number = 1,
        name = "busA",
        available = true,
        bustype = PSY.ACBusTypes.REF,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 138.0,
    )
    busB = PSY.ACBus(;
        number = 2,
        name = "busB",
        available = true,
        bustype = PSY.ACBusTypes.PV,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 138.0,
    )
    PSY.add_component!(sys, busA)
    PSY.add_component!(sys, busB)
    arc = PSY.Arc(; from = busA, to = busB)
    PSY.add_component!(sys, arc)
    t = PSY.TwoWindingTransformer(;
        name = "T2W",
        circuit = PSY.TransformerCircuit(;
            arc = arc,
            tap = 1.0,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 1.0,
            base_power = 100.0,
            base_voltage_primary = 138.0,
            r = 0.01,
            x = 0.1,
        ),
    )
    PSY.add_component!(sys, t)

    # circuit base_power (100.0) == system base, so DU == SU here.
    @test PNM.get_series_susceptance(t, PSY.SU) ≈ 1 / 0.1
    PSY.set_tap!(PSY.get_circuit(t), 1.05)
    @test PNM.get_series_susceptance(t, PSY.SU) ≈ (1 / 0.1) / 1.05
    @test PNM.get_series_susceptance(t, PSY.SU) ≈ 9.523809523809524
end

@testset "Magnetizing shunt placement (2W enum + 3W enum, parent-resident)" begin
    # Nonzero conductance (not just susceptance), to exercise the g-side of the shunt split.
    # r = 0.01, x = 0.1 for both transformer types below (2W directly; 3W via symmetric
    # pairwise legs r12 = r23 = r31 = 0.02, x12 = x23 = x31 = 0.2, which derive every star
    # leg to (0.02 + 0.02 - 0.02)/2 = 0.01, (0.2 + 0.2 - 0.2)/2 = 0.1), so `Y_t` is identical
    # in both cases. The magnetizing shunt and its location are PARENT-transformer
    # fields; the 2W and 3W enums are distinct types.
    y_shunt = 0.005 + 0.012im
    r, x = 0.01, 0.1
    Y_t = inv(complex(r, x))

    # 2W: PRIMARY -> from side, SECONDARY -> to side, SPLIT -> full value both sides.
    twoW_locations = (
        (location = PSY.TwoWindingTransformerShuntLocation.PRIMARY, fr = true, to = false),
        (
            location = PSY.TwoWindingTransformerShuntLocation.SECONDARY,
            fr = false,
            to = true,
        ),
        (location = PSY.TwoWindingTransformerShuntLocation.SPLIT, fr = true, to = true),
    )

    function _t2w_with_shunt(shunt_location)
        sys = PSY.System(100.0)
        busA = PSY.ACBus(;
            number = 1,
            name = "busA",
            available = true,
            bustype = PSY.ACBusTypes.REF,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 138.0,
        )
        busB = PSY.ACBus(;
            number = 2,
            name = "busB",
            available = true,
            bustype = PSY.ACBusTypes.PV,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 138.0,
        )
        PSY.add_component!(sys, busA)
        PSY.add_component!(sys, busB)
        arc = PSY.Arc(; from = busA, to = busB)
        PSY.add_component!(sys, arc)
        t = PSY.TwoWindingTransformer(;
            name = "T2W_shunt",
            circuit = PSY.TransformerCircuit(;
                arc = arc,
                available = true,
                rating = 1.0,
                base_power = 100.0,
                base_voltage_primary = 138.0,
                r = r,
                x = x,
            ),
            magnetizing_shunt = y_shunt,
            shunt_location = shunt_location,
        )
        PSY.add_component!(sys, t)
        return t
    end

    for (location, fr, to) in twoW_locations
        t = _t2w_with_shunt(location)
        (Y11, Y12, Y21, Y22) = PNM.ybus_branch_entries(t)
        @test isapprox(Y11, Y_t + (fr ? y_shunt : 0.0 + 0.0im); atol = 1e-12)
        @test isapprox(Y22, Y_t + (to ? y_shunt : 0.0 + 0.0im); atol = 1e-12)
        @test isapprox(Y12, -Y_t; atol = 1e-12)
        @test isapprox(Y21, -Y_t; atol = 1e-12)

        adm = PNM.branch_admittance(t)
        @test adm.g_fr == (fr ? real(y_shunt) : 0.0)
        @test adm.b_fr == (fr ? imag(y_shunt) : 0.0)
        @test adm.g_to == (to ? real(y_shunt) : 0.0)
        @test adm.b_to == (to ? imag(y_shunt) : 0.0)
    end

    # SPLIT applies the FULL value on both sides -- not halved.
    t_split = _t2w_with_shunt(PSY.TwoWindingTransformerShuntLocation.SPLIT)
    (Y11, _, _, Y22) = PNM.ybus_branch_entries(t_split)
    @test isapprox(Y11 - Y_t, y_shunt; atol = 1e-12)
    @test isapprox(Y22 - Y_t, y_shunt; atol = 1e-12)
    @test !isapprox(Y11 - Y_t, y_shunt / 2; atol = 1e-9)

    # 3W: the parent shunt lands on circuit 1 only. PRIMARY places it on the terminal (from)
    # side, STAR on the star-node (to) side; circuits 2 and 3 never carry it.
    threeW_locations = (
        (
            location = PSY.ThreeWindingTransformerShuntLocation.PRIMARY,
            fr = true,
            to = false,
        ),
        (location = PSY.ThreeWindingTransformerShuntLocation.STAR, fr = false, to = true),
    )

    function _t3w_with_shunt(shunt_location, suffix)
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
        busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
        sec_bus, ter_bus, star_bus =
            _add_star_buses!(
                sys,
                busD;
                numbers = (601 + suffix, 602 + suffix, 603 + suffix),
            )
        return _add_three_winding_transformer!(
            sys, busD, sec_bus, ter_bus, star_bus;
            name = "T3W_shunt_$suffix",
            r12 = 0.02, x12 = 0.2, r23 = 0.02, x23 = 0.2, r31 = 0.02, x31 = 0.2,
            magnetizing_shunt = y_shunt,
            shunt_location = shunt_location,
        )
    end

    for (i, (location, fr, to)) in enumerate(threeW_locations)
        t3w = _t3w_with_shunt(location, i)
        w1 = PNM.ThreeWindingTransformerCircuit(t3w, 1)
        (Y11, Y12, Y21, Y22) = PNM.ybus_branch_entries(w1)
        # STAR lands the shunt on the star-bus diagonal (circuit-1 Y22); PRIMARY on the
        # terminal-bus diagonal (Y11). Hand-computed: the whole value, on one side only.
        @test isapprox(Y11, Y_t + (fr ? y_shunt : 0.0 + 0.0im); atol = 1e-12)
        @test isapprox(Y22, Y_t + (to ? y_shunt : 0.0 + 0.0im); atol = 1e-12)

        adm = PNM.winding_admittance(w1)
        @test adm.g_fr == (fr ? real(y_shunt) : 0.0)
        @test adm.b_fr == (fr ? imag(y_shunt) : 0.0)
        @test adm.g_to == (to ? real(y_shunt) : 0.0)
        @test adm.b_to == (to ? imag(y_shunt) : 0.0)

        # Circuits 2 and 3 carry no shunt regardless of the parent location: their Ybus
        # diagonals stay at the bare series admittance (unit tap here).
        for cn in (2, 3)
            wc = PNM.ThreeWindingTransformerCircuit(t3w, cn)
            (c11, _, _, c22) = PNM.ybus_branch_entries(wc)
            @test isapprox(c11, Y_t; atol = 1e-12)
            @test isapprox(c22, Y_t; atol = 1e-12)
            b_sh = PNM.get_equivalent_b(wc)
            @test b_sh.from == 0.0
            @test b_sh.to == 0.0
            cadm = PNM.winding_admittance(wc)
            @test cadm.g_fr == 0.0 && cadm.b_fr == 0.0
            @test cadm.g_to == 0.0 && cadm.b_to == 0.0
        end
    end
end
