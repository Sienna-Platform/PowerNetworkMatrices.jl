@testset "YbusModification: all N-1 branch outages on c_sys5" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    for branch in get_components(
        x -> !(typeof(x) <: Union{
            PhaseShiftingTransformer,
            DiscreteControlledACBranch,
        }),
        ACTransmission,
        sys,
    )
        mod = YbusModification(ybus, PSY.Component[branch])
        modified_data = apply_ybus_modification(ybus, mod)

        # Reference: disable branch and rebuild
        set_available!(branch, false)
        ybus_ref = Ybus(sys)
        set_available!(branch, true)

        @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
    end
end

@testset "YbusModification: sparsity check" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line = get_component(Line, sys, "1")
    mod = YbusModification(ybus, PSY.Component[line])

    # A single branch outage should produce at most 4 nonzero entries
    @test SparseArrays.nnz(mod.data) <= 4
    @test length(mod.component_names) == 1
    @test mod.component_names[1] == "1"
    @test mod.component_types[1] == Line
end

@testset "YbusModification: multiple branch outage" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    mod = YbusModification(ybus, PSY.Component[line1, line2])
    modified_data = apply_ybus_modification(ybus, mod)

    # Reference: disable both lines and rebuild
    set_available!(line1, false)
    set_available!(line2, false)
    ybus_ref = Ybus(sys)

    @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
end

@testset "YbusModification: N-3 contingency (3 branches)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    line3 = get_component(Line, sys, "3")
    mod = YbusModification(ybus, PSY.Component[line1, line2, line3])
    modified_data = apply_ybus_modification(ybus, mod)

    # Reference: disable all three and rebuild
    set_available!(line1, false)
    set_available!(line2, false)
    set_available!(line3, false)
    ybus_ref = Ybus(sys)

    @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
    @test length(mod.component_names) == 3
end

@testset "YbusModification: N-4 contingency (4 branches)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    line3 = get_component(Line, sys, "3")
    line4 = get_component(Line, sys, "4")
    mod = YbusModification(ybus, PSY.Component[line1, line2, line3, line4])
    modified_data = apply_ybus_modification(ybus, mod)

    # Reference: disable all four and rebuild
    set_available!(line1, false)
    set_available!(line2, false)
    set_available!(line3, false)
    set_available!(line4, false)
    ybus_ref = Ybus(sys)

    @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
    @test length(mod.component_names) == 4

    # Also verify sparsity: 4 branches × at most 4 entries each = at most 16 nonzeros
    # (fewer if branches share buses, since sparse constructor sums duplicates)
    @test SparseArrays.nnz(mod.data) <= 16
end

@testset "YbusModification: N-3 contingency via supplemental attribute" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    # Attach a single outage to 3 branches
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

    # Create modification from contingency
    mod_ctg = YbusModification(ybus, sys, outage)

    # Create modification from components directly
    mod_direct = YbusModification(ybus, PSY.Component[line1, line2, line3])

    # Should be identical
    @test isapprox(mod_ctg.data, mod_direct.data, atol = 1e-10)
    @test length(mod_ctg.component_names) == 3
end

@testset "YbusModification: combining modifications with +" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    mod1 = YbusModification(ybus, PSY.Component[line1])
    mod2 = YbusModification(ybus, PSY.Component[line2])
    mod_combined = mod1 + mod2

    # Should match a single modification with both lines
    mod_both = YbusModification(ybus, PSY.Component[line1, line2])
    @test isapprox(
        mod_combined.data, mod_both.data, atol = 1e-10,
    )
    @test length(mod_combined.component_names) == 2
end

@testset "YbusModification: apply_ybus_modification returns sparse matrix" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line = get_component(Line, sys, "1")
    mod = YbusModification(ybus, PSY.Component[line])
    result = apply_ybus_modification(ybus, mod)

    @test result isa SparseArrays.SparseMatrixCSC{PNM.YBUS_ELTYPE, Int}
    @test size(result) == size(ybus.data)
end

@testset "YbusModification: show method" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line = get_component(Line, sys, "1")
    mod = YbusModification(ybus, PSY.Component[line])

    io = IOBuffer()
    show(io, MIME"text/plain"(), mod)
    output = String(take!(io))
    @test occursin("YbusModification", output)
    @test occursin("1 component(s)", output)
end

@testset "YbusModification: contingency API with supplemental attribute" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    # Attach an outage to a branch
    line = get_component(Line, sys, "1")
    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, line, outage)

    # Create modification from contingency
    mod_ctg = YbusModification(ybus, sys, outage)

    # Create modification from component directly
    mod_direct = YbusModification(ybus, PSY.Component[line])

    # Should be identical
    @test isapprox(mod_ctg.data, mod_direct.data, atol = 1e-10)
end

@testset "YbusModification: PhaseShiftingTransformer rejection" begin
    sys = build_system(PSSEParsingTestSystems, "pti_case14_with_pst3w_sys")
    ybus = Ybus(sys)
    pst = first(get_components(PhaseShiftingTransformer, sys))
    @test_throws ErrorException YbusModification(
        ybus, PSY.Component[pst],
    )
end

@testset "YbusModification: impedance change with delta" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    line = get_component(Line, sys, "1")
    original_x = get_x(line)

    # Apply a reactance increase via delta impedance
    delta_z = original_x * 1.0im
    mod = YbusModification(ybus, line, delta_z)

    # Reference: modify the branch in the system and rebuild Ybus
    set_x!(line, original_x * 2.0)
    ybus_ref = Ybus(sys)
    set_x!(line, original_x)

    # The modification applied to original Ybus should match the rebuilt Ybus
    modified_data = apply_ybus_modification(ybus, mod)
    @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
    @test length(mod.component_names) == 1
    @test mod.is_islanding == false

    # Delta larger than branch impedance should error
    z_mag = abs(get_r(line) + get_x(line) * im)
    @test_throws ErrorException YbusModification(ybus, line, z_mag * 2.0im)
end

@testset "YbusModification: impedance change on series chain with DegreeTwoReduction" begin
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    reductions = NetworkReduction[DegreeTwoReduction()]
    ybus = Ybus(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(ybus)

    series_branch = nothing
    for (br, arc) in nr.reverse_series_branch_map
        if br isa PSY.ACTransmission && !(br isa PNM.ThreeWindingTransformerWinding)
            series_branch = br
            break
        end
    end

    if series_branch !== nothing
        original_x = get_x(series_branch)
        delta_z = original_x * 0.5im

        mod = YbusModification(ybus, series_branch, delta_z)

        # Reference: modify the branch and rebuild
        set_x!(series_branch, original_x * 1.5)
        ybus_ref = Ybus(sys; network_reductions = reductions)
        set_x!(series_branch, original_x)

        # Series chain reduction compounds Float32 rounding across platforms
        modified_data = apply_ybus_modification(ybus, mod)
        @test isapprox(modified_data, ybus_ref.data, atol = 1e-3)
    end
end

@testset "YbusModification: parallel branch outage on RTS_GMLC" begin
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    ybus = Ybus(sys)
    nr = PNM.get_network_reduction_data(ybus)

    # Find a branch that is in the parallel map
    parallel_branch = nothing
    for (br, arc) in nr.reverse_parallel_branch_map
        parallel_branch = br
        break
    end

    if parallel_branch !== nothing
        mod = YbusModification(ybus, PSY.Component[parallel_branch])
        modified_data = apply_ybus_modification(ybus, mod)

        # Reference: disable branch and rebuild
        set_available!(parallel_branch, false)
        ybus_ref = Ybus(sys)

        @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
    end
end

@testset "YbusModification: series chain outage with DegreeTwoReduction" begin
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = PNM.get_network_reduction_data(ybus)

    # Find an ACBranch (not ThreeWindingTransformerWinding) in the series map
    series_branch = nothing
    for (br, arc) in nr.reverse_series_branch_map
        if !(br isa PNM.ThreeWindingTransformerWinding)
            series_branch = br
            break
        end
    end

    if series_branch !== nothing
        mod = YbusModification(ybus, PSY.Component[series_branch])

        # Verify modification has the expected structure (4 entries for one arc)
        @test SparseArrays.nnz(mod.data) <= 4
        @test length(mod.component_names) == 1

        # Apply and verify the matrix is valid (dimensions, type)
        modified_data = apply_ybus_modification(ybus, mod)
        @test size(modified_data) == size(ybus.data)
    end
end

@testset "YbusModification: FixedAdmittance shunt outage" begin
    sys = PSB.build_system(PSB.PSYTestSystems, "psse_3bus_gen_cls_sys")
    bus_103 = PSY.get_component(PSY.ACBus, sys, "BUS 3")
    fix_shunt = FixedAdmittance("FixAdm_Bus3", true, bus_103, 0.0 + 0.2im)
    add_component!(sys, fix_shunt)

    ybus = Ybus(sys)
    mod = YbusModification(ybus, PSY.Component[fix_shunt])
    modified_data = apply_ybus_modification(ybus, mod)

    # Reference: disable shunt and rebuild
    set_available!(fix_shunt, false)
    ybus_ref = Ybus(sys)

    @test isapprox(modified_data, ybus_ref.data, atol = 1e-4)
    @test length(mod.component_names) == 1
    # Shunt outage should only affect one diagonal entry
    @test SparseArrays.nnz(mod.data) == 1
end

@testset "YbusModification: unsupported component warning" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    # Use a ThermalStandard (generator) which does not affect Ybus
    gen = first(get_components(ThermalStandard, sys))
    mod = @test_logs(
        (:warn, r"does not affect the Ybus"),
        YbusModification(ybus, PSY.Component[gen]),
    )
    # Should produce an empty (all-zero) delta
    @test SparseArrays.nnz(mod.data) == 0
end

@testset "YbusModification: series chain full outage with DegreeTwoReduction" begin
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = PNM.get_network_reduction_data(ybus)

    # Find all branches in one series chain and trip them all (full chain outage)
    if !isempty(nr.series_branch_map)
        arc_tuple, series_chain = first(nr.series_branch_map)
        all_segments = PSY.ACTransmission[]
        for segment in series_chain
            if segment isa PNM.BranchesParallel
                for b in segment
                    push!(all_segments, b)
                end
            elseif segment isa PSY.ACTransmission
                push!(all_segments, segment)
            end
        end

        if !isempty(all_segments)
            mod = YbusModification(ybus, PSY.Component[all_segments...])
            modified_data = apply_ybus_modification(ybus, mod)
            @test size(modified_data) == size(ybus.data)
            @test length(mod.component_names) <= length(all_segments)
            @test length(mod.component_names) >= 1
        end
    end
end

@testset "YbusModification: contingency with no associated components" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)

    # Create an outage not attached to any component
    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    # Don't attach to any component — should error
    @test_throws ErrorException YbusModification(ybus, sys, outage)
end
