@testset "_resolve_branch_arc: classifies direct, parallel, series, and unknown branches" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[DegreeTwoReduction()]
    ybus = Ybus(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(ybus)

    if !isempty(nr.reverse_direct_branch_map)
        branch, expected_arc = first(nr.reverse_direct_branch_map)
        tag, arc = PNM._resolve_branch_arc(nr, branch)
        @test tag === :direct
        @test arc == expected_arc
    end

    if !isempty(nr.reverse_parallel_branch_map)
        branch, expected_arc = first(nr.reverse_parallel_branch_map)
        tag, arc = PNM._resolve_branch_arc(nr, branch)
        @test tag === :parallel
        @test arc == expected_arc
    end

    if !isempty(nr.reverse_series_branch_map)
        branch, expected_arc = first(nr.reverse_series_branch_map)
        tag, arc = PNM._resolve_branch_arc(nr, branch)
        @test tag === :series
        @test arc == expected_arc
    end

    # Three-winding transformer windings are one-to-one arcs held in the direct maps, so
    # they classify as :direct like any other single branch on its arc.
    winding_entry = nothing
    for (branch, expected_arc) in nr.reverse_direct_branch_map
        if branch isa PNM.ThreeWindingTransformerCircuit
            winding_entry = (branch, expected_arc)
            break
        end
    end
    if winding_entry !== nothing
        winding, expected_arc = winding_entry
        tag, arc = PNM._resolve_branch_arc(nr, winding)
        @test tag === :direct
        @test arc == expected_arc
    end
end

@testset "_resolve_branch_arc: returns :not_found for eliminated branches" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ybus = Ybus(sys)
    nr = PNM.get_network_reduction_data(ybus)

    line = first(PSY.get_components(PSY.Line, sys))
    PSY.set_available!(line, false)
    ybus2 = Ybus(sys)
    nr2 = PNM.get_network_reduction_data(ybus2)

    tag, arc = PNM._resolve_branch_arc(nr2, line)
    @test tag === :not_found
    @test isnothing(arc)
end
