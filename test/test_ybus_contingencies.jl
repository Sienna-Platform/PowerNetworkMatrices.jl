@testset "compute_ybus_delta: N-1 branch outages match rebuilt Ybus" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)
    vptdf = VirtualPTDF(sys)

    for branch in get_components(
        x -> !(typeof(x) <: DiscreteControlledACBranch || _is_phase_shifting_2w(x)),
        ACTransmission,
        sys,
    )
        mod_new = NetworkModification(vptdf, branch)
        result_new = apply_ybus_modification(ybus, mod_new)

        # Reference: disable branch and rebuild
        set_available!(branch, false)
        ybus_ref = Ybus(sys)
        set_available!(branch, true)

        @test isapprox(result_new, ybus_ref.data, atol = 1e-4)
    end
end

@testset "compute_ybus_delta: multiple branch outage (N-2)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)
    vptdf = VirtualPTDF(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")

    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, line1, outage)
    add_supplemental_attribute!(sys, line2, outage)

    mod = NetworkModification(vptdf, sys, outage)
    result = apply_ybus_modification(ybus, mod)

    # Reference: disable both and rebuild
    set_available!(line1, false)
    set_available!(line2, false)
    ybus_ref = Ybus(sys)

    @test isapprox(result, ybus_ref.data, atol = 1e-4)
end

@testset "compute_ybus_delta: N-3 contingency (3 branches)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)
    vptdf = VirtualPTDF(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    line3 = get_component(Line, sys, "3")

    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, line1, outage)
    add_supplemental_attribute!(sys, line2, outage)
    add_supplemental_attribute!(sys, line3, outage)

    mod = NetworkModification(vptdf, sys, outage)
    result = apply_ybus_modification(ybus, mod)

    # Reference: disable all three and rebuild
    set_available!(line1, false)
    set_available!(line2, false)
    set_available!(line3, false)
    ybus_ref = Ybus(sys)

    @test isapprox(result, ybus_ref.data, atol = 1e-4)
    @test length(mod.arc_modifications) >= 1
end

@testset "compute_ybus_delta: N-4 contingency (4 branches)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)
    vptdf = VirtualPTDF(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    line3 = get_component(Line, sys, "3")
    line4 = get_component(Line, sys, "4")

    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, line1, outage)
    add_supplemental_attribute!(sys, line2, outage)
    add_supplemental_attribute!(sys, line3, outage)
    add_supplemental_attribute!(sys, line4, outage)

    mod = NetworkModification(vptdf, sys, outage)
    result = apply_ybus_modification(ybus, mod)

    # Reference: disable all four and rebuild
    set_available!(line1, false)
    set_available!(line2, false)
    set_available!(line3, false)
    set_available!(line4, false)
    ybus_ref = Ybus(sys)

    @test isapprox(result, ybus_ref.data, atol = 1e-4)
    @test length(mod.arc_modifications) >= 1
end

@testset "compute_ybus_delta: parallel branch outage on RTS_GMLC" begin
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    ybus = Ybus(sys)
    vptdf = VirtualPTDF(sys)
    nr = PNM.get_network_reduction_data(ybus)

    # Find a branch in the parallel map
    parallel_branch = nothing
    for (br, _) in nr.reverse_parallel_branch_map
        parallel_branch = br
        break
    end

    if !isnothing(parallel_branch)
        mod = NetworkModification(vptdf, parallel_branch)
        result = apply_ybus_modification(ybus, mod)

        # Reference: disable branch and rebuild
        set_available!(parallel_branch, false)
        ybus_ref = Ybus(sys)

        @test isapprox(result, ybus_ref.data, atol = 1e-4)
    end
end

@testset "compute_ybus_delta: series chain outage with DegreeTwoReduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[DegreeTwoReduction()]
    ybus = Ybus(sys; network_reductions = reductions)
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(ybus)

    # Find an ACBranch (not ThreeWindingTransformerCircuit) in the series map.
    # Dict iteration order is not stable across Julia versions, so sort the
    # candidates by name for determinism. Only standalone segments (not nested
    # inside BranchesParallel) yield a full-chain outage when removed; partial
    # series outages are unsupported by `_compute_arc_ybus_delta`.
    series_branch = nothing
    candidates = [
        entry for entry in nr.reverse_series_branch_map if
        !(entry[1] isa PNM.ThreeWindingTransformerCircuit)
    ]
    sort!(candidates; by = entry -> PSY.get_name(entry[1]))
    for (br, composite_arc) in candidates
        series_chain = nr.series_branch_map[composite_arc]
        is_standalone_segment =
            any(seg -> seg === br, Iterators.flatten(values(series_chain.branches)))
        if is_standalone_segment
            series_branch = br
            break
        end
    end

    if !isnothing(series_branch)
        mod = NetworkModification(vptdf, series_branch)
        result = apply_ybus_modification(ybus, mod)

        @test size(result) == size(ybus.data)
        @test length(mod.arc_modifications) == 1
    end
end

@testset "compute_ybus_delta: contingency with no associated components errors" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)

    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    @test_throws ErrorException NetworkModification(vptdf, sys, outage)
end

@testset "compute_ybus_delta: shunt outage produces correct diagonal delta" begin
    sys = PSB.build_system(PSB.PSYTestSystems, "psse_3bus_gen_cls_sys")
    bus_103 = PSY.get_component(PSY.ACBus, sys, "BUS 3")
    fix_shunt = FixedAdmittance("FixAdm_Bus3", true, bus_103, 0.0 + 0.2im)
    add_component!(sys, fix_shunt)

    ybus = Ybus(sys)
    bus_lookup = PNM.get_bus_lookup(ybus)
    nr = PNM.get_network_reduction_data(ybus)
    bus_ix = PNM.get_bus_index(fix_shunt, bus_lookup, nr)

    mod = NetworkModification(
        "shunt_outage",
        ArcModification[],
        [PNM.ShuntModification(bus_ix, PNM.YBUS_ELTYPE(-(0.0 + 0.2im)))],
        false,
    )
    result = apply_ybus_modification(ybus, mod)

    # Reference: disable shunt and rebuild
    set_available!(fix_shunt, false)
    ybus_ref = Ybus(sys)

    @test isapprox(result, ybus_ref.data, atol = 1e-4)
end

@testset "issue 305: contingencies on a Line ∥ PST group" begin
    sys = _mk_line_pst_parallel_system()
    line = get_component(Line, sys, "L1")
    pst = get_component(TwoWindingTransformer, sys, "PST")
    vptdf = VirtualPTDF(sys)

    # Tripping the non-shifting member classifies as a parallel-arc modification
    # carrying only that member's susceptance.
    mod = NetworkModification(vptdf, line)
    @test length(mod.arc_modifications) == 1
    @test isapprox(
        mod.arc_modifications[1].delta_b,
        -PNM.get_series_susceptance(line, PSY.SU),
    )

    # Tripping the phase-shifting member is rejected loudly.
    err = try
        NetworkModification(vptdf, pst)
        nothing
    catch ex
        ex
    end
    @test err isa ErrorException
    @test occursin("phase-shifting", err.msg)
end

@testset "parallel member outage resolves by identity, not susceptance value" begin
    # White-box: PST filed FIRST so the old value-scan would hit it before the line.
    # Line x=0.1 and PST (tap=1.0, x=0.1) both have b = 10.0 — deliberate collision.
    # Attached (not detached, as in `_mk_detached_pst_fixture`) because
    # `ybus_branch_entries`/`get_series_susceptance` read impedances in `PSY.SU`, which
    # needs a system base power; `NetworkReductionData` is still built manually (PST
    # filed first) so map order, not `add_component!` order, drives the collision.
    sys, buses = _mk_bus_system(2)
    arc = Arc(; from = buses[1], to = buses[2])
    add_component!(sys, arc)
    line = Line(;
        name = "L1",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.0,
        x = 0.1,
        b = (from = 0.0, to = 0.0),
        rating = 1.0,
        angle_limits = (min = -1.5, max = 1.5),
    )
    add_component!(sys, line)
    pst = PSY.TwoWindingTransformer(;
        name = "PSTx01",
        circuit = PSY.TransformerCircuit(;
            arc = arc,
            tap = 1.0,
            α = 0.15,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 1.0,
            base_power = 100.0,
            base_voltage_primary = 230.0,
            r = 0.0,
            x = 0.1,
        ),
        magnetizing_shunt = Complex(0.0, 0.0),
    )
    add_component!(sys, pst)

    nr = PNM.NetworkReductionData()
    PNM.add_to_branch_maps!(nr, PSY.get_arc(pst), pst)
    PNM.add_to_branch_maps!(nr, PSY.get_arc(line), line)
    b_line = PNM.get_series_susceptance(line, PSY.SU)

    # Identity-resolved delta for tripping the line: the negated LINE pi-model
    # (symmetric), never the PST's asymmetric one.
    dy = PNM._compute_arc_ybus_delta(nr, (1, 2), -b_line, line)
    expected = PNM.ybus_branch_entries(line)
    @test dy[1] ≈ -expected[1]
    @test dy[2] ≈ -expected[2]
    @test dy[3] ≈ -expected[3]
    @test dy[4] ≈ -expected[4]
    @test dy[2] ≈ dy[3]

    # The arc-tuple-only entry point cannot know which member tripped: partial
    # parallel outages must be rejected loudly, not value-guessed.
    err = try
        PNM._compute_arc_ybus_delta(nr, (1, 2), -b_line)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("identity", err.msg) || occursin("component", err.msg)

    # A component that is not a member of the group is a loud error.
    (_, line2, _, _) = _mk_detached_pst_fixture()
    err2 = try
        PNM._compute_arc_ybus_delta(nr, (1, 2), -b_line, line2)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
    @test occursin("not a member", err2.msg)
end

@testset "black-box: tripping the line member of an equal-b Line ∥ PST group" begin
    sys = _mk_line_pst_parallel_system(; pst_x = 0.1)
    line = get_component(Line, sys, "L1")
    vptdf = VirtualPTDF(sys)

    mod = NetworkModification(vptdf, line)
    @test length(mod.arc_modifications) == 1
    m = mod.arc_modifications[1]
    @test m.delta_b == -PNM.get_series_susceptance(line, PSY.SU)
    @test m.delta_y12 ≈ m.delta_y21
    @test m.delta_y11 ≈ -PNM.ybus_branch_entries(line)[1]
end

@testset "ArcModification stores correct Ybus delta entries" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)
    nr = PNM.get_network_reduction_data(vptdf)

    for branch in get_components(
        x -> !(typeof(x) <: DiscreteControlledACBranch || _is_phase_shifting_2w(x)),
        ACTransmission,
        sys,
    )
        mod = NetworkModification(vptdf, branch)
        for arc_mod in mod.arc_modifications
            # ΔY fields should be nonzero for a real outage
            @test !iszero(arc_mod.delta_y12)

            # Verify off-diagonal ΔY matches negated ybus_branch_entries for full outage
            arc_tuple = PNM.get_arc_axis(nr)[arc_mod.arc_index]
            if haskey(nr.direct_branch_map, arc_tuple)
                br = nr.direct_branch_map[arc_tuple]
                _, Y12, _, _ = PNM.ybus_branch_entries(br)
                @test isapprox(arc_mod.delta_y12, PNM.YBUS_ELTYPE(-Y12); atol = 1e-6)
            end
        end
    end
end

@testset "anti-parallel asymmetric member outage delta is swapped into the key frame" begin
    # 3-bus: L1 on (1,2), ZIB on (1,3) merges bus 3 into 1, transformer PST on (2,3)
    # becomes anti-parallel to L1. tap = 1.05 makes PST's diagonals asymmetric
    # (Y11 != Y22); α = 0 keeps it non-phase-shifting so the outage is allowed.
    # Copied from the "anti-parallel asymmetric group: phase-shifting transformer
    # member" testset (test/test_ybus_reductions.jl:725-823), with α = 0.0.
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
    function _mk_pin_line(name, f, t, x)
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
    _mk_pin_line("L1", 1, 2, 0.1)        # symmetric
    _mk_pin_line("ZIB", 1, 3, 1e-5)      # zero-impedance: merges bus 3 into bus 1
    # Asymmetric member oriented (2, 3); becomes anti-parallel to L1 after the merge.
    arc = Arc(; from = buses[2], to = buses[3])
    add_component!(sys, arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = arc,
                tap = 1.05,
                α = 0.0,
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
    (_, bp) = first(PNM.get_parallel_branch_map(nr))
    t = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")
    l1 = get_component(Line, sys, "L1")

    @test length(PNM.get_parallel_branch_map(nr)) == 1
    @test PNM.get_arc_tuple(t, nr) != PNM.get_arc_tuple(bp, nr)

    vptdf = VirtualPTDF(sys)
    mod = NetworkModification(vptdf, t)
    @test length(mod.arc_modifications) == 1

    # Oracle: removal delta == (remaining member alone) - (full group), both already
    # oriented in the key frame (L1 seeds the key, so no swap on the survivor).
    group_entries = PNM.ybus_branch_entries(bp, nr)
    remaining_entries = PNM.ybus_branch_entries(l1)
    m = only(mod.arc_modifications)
    @test m.delta_y11 ≈ remaining_entries[1] - group_entries[1] atol = 1e-5
    @test m.delta_y12 ≈ remaining_entries[2] - group_entries[2] atol = 1e-5
    @test m.delta_y21 ≈ remaining_entries[3] - group_entries[3] atol = 1e-5
    @test m.delta_y22 ≈ remaining_entries[4] - group_entries[4] atol = 1e-5
end
