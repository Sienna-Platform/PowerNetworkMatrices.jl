# Phase-shifting transformers as outage components: the DC shift-injection delta each arc
# modification carries, and the lifting of the classification guard.

@testset "pst outage: ArcModification carries and merges delta_shift_injection" begin
    z = zero(PNM.YBUS_ELTYPE)
    a = ArcModification(3, -1.0, -0.2, z, z, z, z)
    b = ArcModification(3, -2.0, -0.3, z, z, z, z)
    @test a.delta_shift_injection == -0.2
    @test ArcModification(3, -1.0).delta_shift_injection == 0.0

    merged = PNM._merge_arc_modifications([a, b])
    @test length(merged) == 1
    @test merged[1].delta_b ≈ -3.0
    @test merged[1].delta_shift_injection ≈ -0.5

    # Physically distinct shift deltas must be distinct cache keys.
    m1 = NetworkModification("x", [a])
    m2 = NetworkModification("y", [ArcModification(3, -1.0, 0.0, z, z, z, z)])
    m3 = NetworkModification("z", [ArcModification(3, -1.0, -0.2, z, z, z, z)])
    @test m1 != m2
    @test hash(m1) != hash(m2)
    @test m1 == m3
    @test hash(m1) == hash(m3)
end

# Three buses in a triangle with a PST on (1, 2): tripping the PST leaves the network
# connected, so the post-outage PTDF exists.
function _mk_pst_triangle_system(; alpha = 0.15)
    sys, buses = _mk_bus_system(3)
    function _mk_sys_line(name, f, t)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name, available = true, active_power_flow = 0.0,
                reactive_power_flow = 0.0, arc = arc, r = 0.0, x = 0.1,
                b = (from = 0.0, to = 0.0), rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
        return arc
    end
    _mk_sys_line("L2", 2, 3)
    _mk_sys_line("L3", 1, 3)
    pst_arc = Arc(; from = buses[1], to = buses[2])
    add_component!(sys, pst_arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = pst_arc, tap = 1.0, α = alpha, available = true,
                active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                base_power = 100.0, base_voltage_primary = 230.0,
                r = 0.0, x = 0.2,
            ),
            magnetizing_shunt = Complex(0.0, 0.0),
        ),
    )
    return sys
end

# Bus 1 -- t1 -- bus 2 -- segment -- bus 3, with a degree-two reduction merging the chain
# into arc (1, 3). `parallel_segment` puts a line in parallel with the shifted t3. Both
# transformers shift, so a partial trip leaves a chain that still shifts by a different
# amount -- the surviving-path result stays distinct from the arc-opens early return.
function _mk_pst_series_system(; parallel_segment::Bool = false)
    sys, buses = _mk_bus_system(3)
    function _mk_transformer(name, f, t, α, x)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            PSY.TwoWindingTransformer(;
                name = name,
                circuit = PSY.TransformerCircuit(;
                    arc = arc, tap = 1.0, α = α, available = true,
                    active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                    base_power = 100.0, base_voltage_primary = 230.0,
                    r = 0.0, x = x,
                ),
                magnetizing_shunt = Complex(0.0, 0.0),
            ),
        )
        return arc
    end
    _mk_transformer("t1", 1, 2, 0.05, 0.1)
    t3_arc = _mk_transformer("t3", 2, 3, 0.2, 0.13)
    if parallel_segment
        add_component!(
            sys,
            Line(;
                name = "L23", available = true, active_power_flow = 0.0,
                reactive_power_flow = 0.0, arc = t3_arc, r = 0.0, x = 0.1,
                b = (from = 0.0, to = 0.0), rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
    end
    return sys
end

@testset "pst outage: direct PST arc carries minus its shift injection" begin
    sys = _mk_pst_triangle_system()
    vptdf = VirtualPTDF(sys)
    nr = PNM.get_network_reduction_data(vptdf)
    pst = get_component(PSY.TwoWindingTransformer, sys, "PST")

    mod = NetworkModification(vptdf, pst)
    m = only(mod.arc_modifications)
    @test PNM.get_arc_axis(vptdf)[m.arc_index] == (1, 2)
    @test m.delta_shift_injection ≈ -0.75              # -(1/0.2)·0.15
    @test m.delta_shift_injection ≈ -PNM.arc_dc_shift_injection(nr, (1, 2))

    # Unshifted branches carry zero.
    l2 = get_component(Line, sys, "L2")
    @test iszero(
        only(NetworkModification(vptdf, l2).arc_modifications).delta_shift_injection,
    )

    # The bus-pair constructor agrees with the component constructor.
    @test only(NetworkModification(vptdf, (1, 2)).arc_modifications).delta_shift_injection ≈
          -0.75
end

@testset "pst outage: MODF row and Ybus delta match the rebuilt network" begin
    sys = _mk_pst_triangle_system()
    vptdf = VirtualPTDF(sys)
    ybus = Ybus(sys)
    pst = get_component(PSY.TwoWindingTransformer, sys, "PST")
    mod = NetworkModification(vptdf, pst)

    # Reference: rebuild with the PST out of service.
    set_available!(pst, false)
    ptdf_ref = VirtualPTDF(sys)
    ybus_ref = Ybus(sys)
    set_available!(pst, true)

    @test isapprox(apply_ybus_modification(ybus, mod), ybus_ref.data, atol = 1e-6)

    arc_lookup = PNM.get_arc_lookup(vptdf)
    bus_lookup = PNM.get_bus_lookup(vptdf)
    ref_lookup = PNM.get_bus_lookup(ptdf_ref)
    for arc in ((2, 3), (1, 3))
        row = get_post_modification_ptdf_row(vptdf, arc_lookup[arc], mod)
        ref = ptdf_ref[arc, :]
        for bus in PNM.get_bus_axis(vptdf)
            @test isapprox(row[bus_lookup[bus]], ref[ref_lookup[bus]]; atol = 1e-8)
        end
    end
end

@testset "pst outage: parallel member carries its own share of the group injection" begin
    sys = _mk_line_pst_parallel_system()          # L1 (x=0.1) ∥ PST (x=0.2, α=0.15) on (1,2)
    vptdf = VirtualPTDF(sys)
    pst = get_component(PSY.TwoWindingTransformer, sys, "PST")
    l1 = get_component(Line, sys, "L1")

    m_pst = only(NetworkModification(vptdf, pst).arc_modifications)
    @test m_pst.delta_shift_injection ≈ -0.75
    @test m_pst.delta_b ≈ -5.0

    m_l1 = only(NetworkModification(vptdf, l1).arc_modifications)
    @test iszero(m_l1.delta_shift_injection)
    @test m_l1.delta_b ≈ -10.0
end

@testset "pst outage: series chain opens and drops its whole injection" begin
    sys = _mk_pst_series_system()
    vptdf = VirtualPTDF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = PNM.get_network_reduction_data(vptdf)
    @test haskey(PNM.get_series_branch_map(nr), (1, 3))
    t3 = get_component(PSY.TwoWindingTransformer, sys, "t3")

    m = only(NetworkModification(vptdf, t3).arc_modifications)
    injection = PNM.arc_dc_shift_injection(nr, (1, 3))
    @test !iszero(injection)
    @test m.delta_shift_injection ≈ -injection
    @test m.delta_b ≈
          -PNM.get_series_susceptance(PNM.get_series_branch_map(nr)[(1, 3)], PSY.SU)
end

@testset "pst outage: partial trip inside a series chain keeps the arc" begin
    sys = _mk_pst_series_system(; parallel_segment = true)
    vptdf = VirtualPTDF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = PNM.get_network_reduction_data(vptdf)
    t3 = get_component(PSY.TwoWindingTransformer, sys, "t3")
    chain = PNM.get_series_branch_map(nr)[(1, 3)]

    delta_b = PNM._compute_series_outage_delta_b(chain, t3, nr)
    delta_shift = PNM._compute_series_outage_delta_shift_injection(chain, [t3], nr)

    # Independent reference built from the fixture impedances: segment 2 is the
    # susceptance-weighted parallel equivalent of t3 and L23, segment 1 is t1 alone.
    t1 = get_component(PSY.TwoWindingTransformer, sys, "t1")
    l23 = get_component(Line, sys, "L23")
    b_t1 = PNM.get_series_susceptance(t1, PSY.SU)
    b_t3 = PNM.get_series_susceptance(t3, PSY.SU)
    b_l23 = PNM.get_series_susceptance(l23, PSY.SU)
    α_t1 = PSY.get_α(PSY.get_circuit(t1))
    α_t3 = PSY.get_α(PSY.get_circuit(t3))

    b_seg_old = b_t3 + b_l23
    b_eq_old = 1 / (1 / b_t1 + 1 / b_seg_old)
    injection_old = b_eq_old * (α_t1 + b_t3 * α_t3 / b_seg_old)
    # t3 out leaves L23 alone on segment 2, so only t1 still shifts the chain.
    b_eq_new = 1 / (1 / b_t1 + 1 / b_l23)
    injection_new = b_eq_new * α_t1

    @test injection_old ≈ PNM.arc_dc_shift_injection(nr, (1, 3))
    @test delta_b ≈ b_eq_new - b_eq_old
    # The arc survives on t1 and L23, so this is not a full outage.
    @test !isapprox(delta_b, -b_eq_old; atol = 1e-9)
    @test delta_shift ≈ injection_new - injection_old
    # The surviving chain still shifts, so this is not the arc-opens early return either.
    @test !isapprox(delta_shift, -injection_old; atol = 1e-9)
    @test !iszero(delta_shift)

    # A partial trip on a series-reduced arc has no Pi-model delta, for any branch type.
    @test_throws ErrorException NetworkModification(vptdf, t3)
    @test_throws ErrorException NetworkModification(vptdf, get_component(Line, sys, "L23"))
end

@testset "pst outage: shifted three-winding winding classifies through the same path" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "pti_case14_with_pst3w_sys")
    shifter = first(
        Iterators.filter(
            PSY.is_phase_shifting,
            PSY.get_components(PSY.ThreeWindingTransformer, sys),
        ),
    )
    vptdf = VirtualPTDF(sys)
    nr = PNM.get_network_reduction_data(vptdf)
    arc_ax = PNM.get_arc_axis(vptdf)

    mod = NetworkModification(vptdf, shifter)
    @test !isempty(mod.arc_modifications)
    n_shifted = 0
    for m in mod.arc_modifications
        arc = arc_ax[m.arc_index]
        haskey(PNM.get_direct_branch_map(nr), arc) || continue
        @test m.delta_shift_injection ≈ -PNM.arc_dc_shift_injection(nr, arc)
        iszero(m.delta_shift_injection) || (n_shifted += 1)
    end
    @test n_shifted >= 1
end

@testset "pst outage: registers through the Outage attribute path" begin
    sys = _mk_pst_triangle_system()
    pst = get_component(PSY.TwoWindingTransformer, sys, "PST")
    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, pst, outage)

    vptdf = VirtualPTDF(sys)
    mod = NetworkModification(vptdf, sys, outage)
    @test only(mod.arc_modifications).delta_shift_injection ≈ -0.75

    vmodf = VirtualMODF(sys)
    @test haskey(PNM.get_registered_contingencies(vmodf), IS.get_id(outage))
end
