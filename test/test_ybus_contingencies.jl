@testset "compute_ybus_delta: N-1 branch outages match rebuilt Ybus" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)
    vptdf = VirtualPTDF(sys)

    for branch in get_components(
        x -> !(typeof(x) <: Union{PhaseShiftingTransformer, DiscreteControlledACBranch}),
        ACTransmission, sys,
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
    @test length(mod.modifications) >= 1
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
    @test length(mod.modifications) >= 1
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

    if parallel_branch !== nothing
        mod = NetworkModification(vptdf, parallel_branch)
        result = apply_ybus_modification(ybus, mod)

        # Reference: disable branch and rebuild
        set_available!(parallel_branch, false)
        ybus_ref = Ybus(sys)

        @test isapprox(result, ybus_ref.data, atol = 1e-4)
    end
end

@testset "compute_ybus_delta: series chain outage with DegreeTwoReduction" begin
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    reductions = NetworkReduction[DegreeTwoReduction()]
    ybus = Ybus(sys; network_reductions = reductions)
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(ybus)

    # Find an ACBranch (not ThreeWindingTransformerWinding) in the series map
    series_branch = nothing
    for (br, _) in nr.reverse_series_branch_map
        if !(br isa PNM.ThreeWindingTransformerWinding)
            series_branch = br
            break
        end
    end

    if series_branch !== nothing
        mod = NetworkModification(vptdf, series_branch)
        result = apply_ybus_modification(ybus, mod)

        @test size(result) == size(ybus.data)
        @test length(mod.modifications) == 1
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
