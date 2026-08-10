@testset "Large 2d Reduction Test" begin
    sys = build_system(
        MatpowerTestSystems,
        "matpower_ACTIVSg10k_sys",
    )
    ybus = Ybus(sys)
    reduction = PNM.get_reduction(ybus, sys, DegreeTwoReduction())
    @test !isempty(reduction.series_branch_map)
end

# Five-bus meshed system whose degree-two reduction folds a 3W-transformer winding into a
# series chain that lies inside a loop:
#
#   B1 --L12-- B2
#   |           \
#   L1_10      winding2
#   |             \
#   B10 -winding1- STAR(99) -winding3- B3 (radial)
#
# B10 hosts no injection and has degree 2 (one line + one winding), so DegreeTwoReduction
# folds [L1_10, winding1] into composite arc (1, 99). That arc closes the loop
# 1 - 2 - STAR - 1, so reduced DC sensitivities are impedance-weighted through the chain
# equivalent — unlike a radial chain, where PTDF entries are topological (0/±1) and
# insensitive to the chain's susceptance value.
function _build_meshed_3wt_loop_system()
    sys = PSY.System(100.0)
    mkbus(n, name, bt) = PSY.ACBus(;
        number = n,
        name = name,
        available = true,
        bustype = bt,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 138.0,
    )
    b1 = mkbus(1, "B1", PSY.ACBusTypes.REF)
    b2 = mkbus(2, "B2", PSY.ACBusTypes.PV)
    b10 = mkbus(10, "B10", PSY.ACBusTypes.PQ)
    b3 = mkbus(3, "B3", PSY.ACBusTypes.PQ)
    star = mkbus(99, "STAR", PSY.ACBusTypes.PQ)
    foreach(b -> PSY.add_component!(sys, b), (b1, b2, b10, b3, star))
    for (bus, name) in ((b1, "g1"), (b2, "g2"))
        gen = PSY.ThermalStandard(;
            name = name,
            available = true,
            status = true,
            bus = bus,
            active_power = 1.0,
            reactive_power = 0.0,
            rating = 5.0,
            active_power_limits = (min = 0.0, max = 5.0),
            reactive_power_limits = (min = -5.0, max = 5.0),
            ramp_limits = nothing,
            time_limits = nothing,
            operation_cost = PSY.ThermalGenerationCost(nothing),
            base_power = 100.0,
            prime_mover_type = PSY.PrimeMovers.ST,
            fuel = PSY.ThermalFuels.COAL,
        )
        PSY.add_component!(sys, gen)
    end
    load = PSY.PowerLoad(;
        name = "load_star",
        available = true,
        bus = star,
        active_power = 1.5,
        reactive_power = 0.0,
        base_power = 100.0,
        max_active_power = 1.5,
        max_reactive_power = 0.0,
    )
    PSY.add_component!(sys, load)
    function mkline(name, f, t, x)
        arc = PSY.Arc(; from = f, to = t)
        PSY.add_component!(sys, arc)
        line = PSY.Line(;
            name = name,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = arc,
            r = 0.01,
            x = x,
            b = (from = 0.0, to = 0.0),
            rating = 10.0,
            angle_limits = (min = -1.5, max = 1.5),
        )
        PSY.add_component!(sys, line)
        return line
    end
    mkline("L12", b1, b2, 0.10)
    mkline("L1_10", b1, b10, 0.05)
    arcs = (
        PSY.Arc(; from = b10, to = star),
        PSY.Arc(; from = b2, to = star),
        PSY.Arc(; from = b3, to = star),
    )
    foreach(a -> PSY.add_component!(sys, a), arcs)
    # Winding-resident star-leg impedances derived from the pairwise data (as PFFP does at
    # parse): z1 = (z12 + z31 - z23)/2, z2 = (z12 + z23 - z31)/2, z3 = (z31 + z23 - z12)/2.
    z12, z23, z31 = complex(0.01, 0.10), complex(0.01, 0.12), complex(0.01, 0.08)
    legs = (
        (z12 + z31 - z23) / 2,
        (z12 + z23 - z31) / 2,
        (z31 + z23 - z12) / 2,
    )
    circuits = ntuple(
        i -> PSY.TransformerCircuit(;
            arc = arcs[i],
            available = true,
            base_power = 100.0,
            base_voltage_primary = 138.0,
            r = real(legs[i]),
            x = imag(legs[i]),
            rating = 5.0,
        ),
        3,
    )
    t3w = PSY.ThreeWindingTransformer(;
        name = "T3W",
        primary_circuit = circuits[1],
        secondary_circuit = circuits[2],
        tertiary_circuit = circuits[3],
        star_bus = star,
        r_12 = 0.01, x_12 = 0.10,
        r_23 = 0.01, x_23 = 0.12,
        r_31 = 0.01, x_31 = 0.08,
        base_power_12 = 100.0, base_power_23 = 100.0, base_power_31 = 100.0,
    )
    PSY.add_component!(sys, t3w)
    return sys
end

@testset "2d reduction of a 3W winding in a loop preserves PTDF" begin
    sys = _build_meshed_3wt_loop_system()
    reductions = PNM.NetworkReduction[PNM.RadialReduction(), PNM.DegreeTwoReduction()]

    # Guard the fixture: the reduction must fold a ThreeWindingTransformerCircuit into a
    # series chain, otherwise this testset silently stops covering the winding path.
    ybus = PNM.Ybus(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(ybus)
    series_map = PNM.get_series_branch_map(nr)
    @test any(
        any(m isa PNM.ThreeWindingTransformerCircuit for m in chain)
        for chain in values(series_map)
    )

    # Reduced PTDF must reproduce the full PTDF on every arc surviving the reduction
    # (matching either orientation, with the sign flipped for a reversed arc).
    ptdf_full = PNM.PTDF(sys)
    ptdf_red = PNM.PTDF(sys; network_reductions = reductions)
    full_buses, full_arcs = ptdf_full.axes
    red_buses, red_arcs = ptdf_red.axes
    red_arc_lookup = Dict(a => i for (i, a) in enumerate(red_arcs))
    red_bus_lookup = Dict(b => i for (i, b) in enumerate(red_buses))
    common_buses = intersect(Set(full_buses), Set(red_buses))
    n_compared = 0
    for (ia, arc) in enumerate(full_arcs)
        sgn = 1.0
        ja = get(red_arc_lookup, arc, 0)
        if ja == 0
            ja = get(red_arc_lookup, (arc[2], arc[1]), 0)
            sgn = -1.0
        end
        ja == 0 && continue
        for bus in common_buses
            v_full = ptdf_full.data[findfirst(==(bus), full_buses), ia]
            v_red = sgn * ptdf_red.data[red_bus_lookup[bus], ja]
            @test isapprox(v_full, v_red; atol = 1e-6)
            n_compared += 1
        end
    end
    @test n_compared > 0
end

@testset "BranchesSeries carries an arc identity" begin
    bs = PNM.BranchesSeries((7, 11))
    @test PNM.get_arc_tuple(bs) == (7, 11)
    # Remapped through a reduction the same way a parallel group's arc_key is.
    nrd = PNM.NetworkReductionData(; reverse_bus_search_map = Dict(11 => 4))
    @test PNM.get_arc_tuple(bs, nrd) == (7, 4)
end

@testset "parallel group of series chains yields the summed two-port" begin
    # Two three-segment chains between Bus 1 and Bus 4 of c_sys14, built by hand so the
    # aggregate can be exercised without depending on the reduction that will create it.
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    nrd = PNM.NetworkReductionData()
    lines = collect(get_components(Line, sys))
    chain_a = PNM.BranchesSeries((1, 4))
    PNM.add_branch!(chain_a, lines[1], :FromTo)
    PNM.add_branch!(chain_a, lines[2], :FromTo)
    chain_b = PNM.BranchesSeries((1, 4))
    PNM.add_branch!(chain_b, lines[3], :FromTo)
    PNM.add_branch!(chain_b, lines[4], :FromTo)

    group = PNM.BranchesParallel(PNM.BranchesSeries[chain_a, chain_b])
    @test PNM.get_arc_tuple(group, nrd) == (1, 4)

    ya = PNM.ybus_branch_entries(chain_a, nrd)
    yb = PNM.ybus_branch_entries(chain_b, nrd)
    yg = PNM.ybus_branch_entries(group, nrd)
    # Parallel members in the same arc frame add entry-wise. A single chain's two-port comes
    # back as ComplexF32 (Ybus storage precision), while the group accumulates in ComplexF64;
    # promote before summing so the comparison isn't dominated by a second, avoidable F32
    # rounding of the addition itself.
    for i in 1:4
        @test isapprox(yg[i], ComplexF64(ya[i]) + ComplexF64(yb[i]); rtol = 1e-10)
    end
end
