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

@testset "Transformer3W winding_admittance and three_winding_arcs decomposition" begin
    # Unit test the per-winding admittance helper against a real PNM
    # `ThreeWindingTransformerWinding`: for a winding with series impedance R + jX the helper
    # must return the series admittance 1/(R + jX), the winding's PNM shunt on the from/to
    # sides, no phase shift, and (here) a unit tap. R/X are read back through PNM so the
    # assertion is robust to per-unit base conversions.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    star_bus = PSY.ACBus(;
        number = 103,
        name = "Star_Bus_T3W",
        available = true,
        bustype = PSY.ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.95, max = 1.05),
        base_voltage = 230.0,
        area = PSY.get_area(busD),
        load_zone = PSY.get_load_zone(busD),
    )
    PSY.add_component!(sys, star_bus)
    sec_bus = PSY.ACBus(;
        number = 101,
        name = "Bus3WT_1",
        available = true,
        bustype = PSY.ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.95, max = 1.05),
        base_voltage = 230.0,
        area = PSY.get_area(busD),
        load_zone = PSY.get_load_zone(busD),
    )
    PSY.add_component!(sys, sec_bus)
    ter_bus = PSY.ACBus(;
        number = 102,
        name = "Bus3WT_2",
        available = true,
        bustype = PSY.ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.95, max = 1.05),
        base_voltage = 230.0,
        area = PSY.get_area(busD),
        load_zone = PSY.get_load_zone(busD),
    )
    PSY.add_component!(sys, ter_bus)
    transformer3w = PSY.Transformer3W(;
        name = "Transformer3W_busD",
        available = true,
        primary_star_arc = PSY.Arc(; from = busD, to = star_bus),
        secondary_star_arc = PSY.Arc(; from = sec_bus, to = star_bus),
        tertiary_star_arc = PSY.Arc(; from = ter_bus, to = star_bus),
        star_bus = star_bus,
        active_power_flow_primary = 0.0,
        reactive_power_flow_primary = 0.0,
        active_power_flow_secondary = 0.0,
        reactive_power_flow_secondary = 0.0,
        active_power_flow_tertiary = 0.0,
        reactive_power_flow_tertiary = 0.0,
        r_primary = 0.01,
        x_primary = 0.1,
        r_secondary = 0.01,
        x_secondary = 0.1,
        r_tertiary = 0.01,
        x_tertiary = 0.1,
        r_12 = 0.01,
        x_12 = 0.1,
        r_23 = 0.01,
        x_23 = 0.1,
        r_13 = 0.01,
        x_13 = 0.1,
        base_power_12 = 100.0,
        base_power_23 = 100.0,
        base_power_13 = 100.0,
        rating = nothing,
        rating_primary = 1.0,
        rating_secondary = 1.0,
        rating_tertiary = 0.5,
    )
    PSY.add_component!(sys, transformer3w)

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
    @test length(arcs) == 3
    @test [a.suffix for a in arcs] == ["winding_1", "winding_2", "winding_3"]
    @test arcs[1].arc == PSY.get_primary_star_arc(transformer3w)
    @test arcs[2].arc == PSY.get_secondary_star_arc(transformer3w)
    @test arcs[3].arc == PSY.get_tertiary_star_arc(transformer3w)
    # Winding admittance computed from the decomposition matches the standalone helper.
    @test PNM.winding_admittance(arcs[1].winding).b ≈ adm.b
end
