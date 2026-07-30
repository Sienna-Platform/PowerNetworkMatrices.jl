# Tests for the DC phase-shifter injection API: `get_series_phase_shift`,
# `arc_dc_phase_shift`, `arc_dc_shift_injection`, `compute_parallel_circulating_flow`,
# and `arc_dc_resistance`.

@testset "dc phase shift: per-branch accessor" begin
    (line, _, pst1, _) = _mk_detached_pst_fixture()
    @test iszero(PNM.get_series_phase_shift(line))
    @test PNM.get_series_phase_shift(pst1) ≈ 0.15
    @test PNM.get_series_phase_shift(PSY.get_circuit(pst1)) ≈ 0.15
end

@testset "dc phase shift: 3W winding accessor" begin
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "pti_case14_with_pst3w_sys";
        force_build = true,
    )
    shifters = collect(
        Iterators.filter(
            PSY.is_phase_shifting,
            PSY.get_components(PSY.ThreeWindingTransformer, sys),
        ),
    )
    @test !isempty(shifters)

    t = first(shifters)
    windings = PSY.get_circuits(t)
    winding_number = findfirst(w -> !iszero(PSY.get_α(w)), windings)
    @test !isnothing(winding_number)
    tw = PNM.ThreeWindingTransformerCircuit(t, winding_number)
    @test PNM.get_series_phase_shift(tw) ≈ PSY.get_α(windings[winding_number])

    # Not every phase-shifting transformer has an idle winding, so search across all of
    # them for a genuine zero-α winding rather than assuming one on `t`.
    zero_tw = nothing
    for candidate in shifters
        zero_ix = findfirst(w -> iszero(PSY.get_α(w)), PSY.get_circuits(candidate))
        if !isnothing(zero_ix)
            zero_tw = PNM.ThreeWindingTransformerCircuit(candidate, zero_ix)
            break
        end
    end
    @test !isnothing(zero_tw)
    @test iszero(PNM.get_series_phase_shift(zero_tw))
end

@testset "dc phase shift: parallel group and per-arc accessors" begin
    sys = _mk_line_pst_parallel_system()   # L1 (x=0.1) ∥ PST (x=0.2, α=0.15) on (1,2); L2 on (2,3)
    ybus = Ybus(sys)
    nr = ybus.network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]

    @test PNM.get_series_phase_shift(bp, nr) ≈ 0.05          # (5·0.15)/15
    @test PNM.arc_dc_phase_shift(nr, (1, 2)) ≈ 0.05
    @test PNM.arc_dc_shift_injection(nr, (1, 2)) ≈ 0.75      # 15·0.05
    @test PNM.arc_dc_phase_shift(nr, (2, 1)) ≈ -0.05         # reversed orientation negates
    @test PNM.arc_dc_shift_injection(nr, (2, 1)) ≈ -0.75
    @test iszero(PNM.arc_dc_phase_shift(nr, (2, 3)))          # plain direct line
    @test iszero(PNM.arc_dc_shift_injection(nr, (2, 3)))

    # Lossy shifted group: b = 1/(t·x) is r-independent; the accessor must not throw
    # (unlike arc_equivalent_branch's single-π extraction).
    sys_lossy = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr_lossy = Ybus(sys_lossy).network_reduction_data
    @test PNM.arc_dc_phase_shift(nr_lossy, (1, 2)) ≈ 0.05

    err = try
        PNM.arc_dc_phase_shift(nr, (7, 9))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
end

@testset "dc phase shift: anti-parallel member orientation" begin
    # Copy of the "anti-parallel asymmetric group: phase-shifting transformer member"
    # system construction (test/test_ybus_reductions.jl:725): ZIB merges bus 3 into bus 1,
    # putting the PST (tap=1.05, α=0.15, x=0.2) anti-parallel to L1 (x=0.1) on arc (1, 2).
    sys = System(100.0)
    buses = ACBus[]
    for i in 1:3
        bustype = ACBusTypes.PV
        if i == 1
            bustype = ACBusTypes.REF
        end
        b = ACBus(;
            number = i,
            name = "b$i",
            available = true,
            bustype = bustype,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        )
        add_component!(sys, b)
        push!(buses, b)
    end
    function _mk_anti_parallel_line(name, f, t, x)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = arc,
                r = 0.0,
                x = x,
                b = (from = 0.0, to = 0.0),
                rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
    end
    _mk_anti_parallel_line("L1", 1, 2, 0.1)        # symmetric
    _mk_anti_parallel_line("ZIB", 1, 3, 1e-5)      # zero-impedance: merges bus 3 into bus 1
    arc = Arc(; from = buses[2], to = buses[3])
    add_component!(sys, arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = arc,
                tap = 1.05,
                α = 0.15,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                rating = 1.0,
                base_power = 100.0,
                base_voltage_primary = 230.0,
                r = 0.0,
                x = 0.2,
            ),
            magnetizing_shunt = Complex(0.0, 0.3),
        ),
    )

    ybus = Ybus(sys)
    nr = ybus.network_reduction_data
    key, bp = first(nr.parallel_branch_map)

    line = PSY.get_component(Line, sys, "L1")
    pst = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")
    b_line = PNM.get_series_susceptance(line, PSY.SU)
    b_pst = PNM.get_series_susceptance(pst, PSY.SU)
    # PST's own arc is (2, 3); through `nr` bus 3 remaps to bus 1, giving (2, 1), which
    # disagrees with the group's (1, 2) frame -- its α enters negated.
    expected = (b_pst * -0.15) / (b_line + b_pst)
    @test PNM.get_series_phase_shift(bp, nr) ≈ expected
end

@testset "dc phase shift: series chain sums segment angles" begin
    # Copy of the "Two transformers in series with different phase angle" construction
    # (test/test_equivalent_getters.jl:401-432): t1 α=0, t3 α=0.2, joined by a degree-two bus.
    t1 = PSY.TwoWindingTransformer(;
        name = "tfw_1",
        circuit = PSY.TransformerCircuit(;
            arc = PSY.Arc(nothing),
            tap = 1.0,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 80.0,
            base_power = 100.0,
            base_voltage_primary = 1.0,
            base_voltage_secondary = 1.0,
            r = 0.122,
            x = 0.1,
        ),
        magnetizing_shunt = 0.01 + im * 0.02,
    )
    t3 = PSY.TwoWindingTransformer(;
        name = "tfw_3",
        circuit = PSY.TransformerCircuit(;
            arc = PSY.Arc(nothing),
            tap = 1.0,
            α = 0.2,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 80.0,
            base_power = 100.0,
            base_voltage_primary = 1.0,
            base_voltage_secondary = 1.0,
            r = 0.3,
            x = 0.13,
        ),
        magnetizing_shunt = 0.02 + im * 0.021,
    )
    vector_branches = [t1, t3]
    sys = System(100.0)
    n_buses = length(vector_branches) + 1
    for bus_ix in 1:n_buses
        bus = ACBus(;
            number = bus_ix,
            name = "bus$(bus_ix)",
            available = true,
            bustype = ACBusTypes.PQ,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.0, max = 1.0),
            base_voltage = 1.0,
            area = nothing,
            load_zone = nothing,
        )
        add_component!(sys, bus)
    end
    for (ix, br) in enumerate(vector_branches)
        br_copy = deepcopy(br)
        set_arc!(
            br_copy,
            Arc(;
                from = get_component(ACBus, sys, "bus$(ix)"),
                to = get_component(ACBus, sys, "bus$(ix+1)"),
            ),
        )
        add_component!(sys, br_copy)
    end
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = ybus.network_reduction_data
    arc = (1, n_buses)
    bs = PNM.get_series_branch_map(nr)[arc]

    @test PNM.get_series_phase_shift(bs, nr) ≈ 0.2      # 0 + 0.2, orientation :FromTo
    @test PNM.arc_dc_phase_shift(nr, arc) ≈ 0.2
    b_eq = PNM.get_series_susceptance(bs, PSY.SU)
    @test PNM.arc_dc_shift_injection(nr, arc) ≈ b_eq * 0.2
    # Cross-check against the existing numeric extraction (small-angle agreement).
    @test PNM.arc_dc_phase_shift(nr, arc) ≈
          PNM.get_equivalent_shift(PNM.arc_equivalent_branch(nr, arc)) atol = 1e-3
end

@testset "dc phase shift: parallel circulating flow" begin
    sys = _mk_line_pst_parallel_system()
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    line = PSY.get_component(Line, sys, "L1")
    pst = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")

    @test PNM.compute_parallel_circulating_flow(bp, nr, line) ≈ 0.5    # 10·(0.05 − 0)
    @test PNM.compute_parallel_circulating_flow(bp, nr, pst) ≈ -0.5    # 5·(0.05 − 0.15)
    # Sums to zero over the group up to floating-point noise, not bit-exactly (α_eq is a
    # weighted average, not a value either member carries exactly).
    @test PNM.compute_parallel_circulating_flow(bp, nr, line) +
          PNM.compute_parallel_circulating_flow(bp, nr, pst) ≈ 0.0 atol = 1e-12

    (_, line2, _, _) = _mk_detached_pst_fixture()
    err = try
        PNM.compute_parallel_circulating_flow(bp, nr, line2)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
end

@testset "dc resistance: total on lossy shifted groups" begin
    # Non-shifting arcs are bit-identical to the single-π path.
    sys = _mk_line_pst_parallel_system()
    nr = Ybus(sys).network_reduction_data
    @test PNM.arc_dc_resistance(nr, (2, 3)) ==
          PNM.get_equivalent_r(PNM.arc_equivalent_branch(nr, (2, 3)))
    # Lossless shifted group: member reactances only, r = 0.
    @test iszero(PNM.arc_dc_resistance(nr, (1, 2)))

    # Lossy shifted group: arc_equivalent_branch throws; arc_dc_resistance must not.
    sys_lossy = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr_lossy = Ybus(sys_lossy).network_reduction_data
    z_line = complex(0.0, 0.1)
    z_pst = complex(0.05, 0.2)
    expected = real(inv(inv(z_line) + inv(z_pst)))
    @test PNM.arc_dc_resistance(nr_lossy, (1, 2)) ≈ expected
end
