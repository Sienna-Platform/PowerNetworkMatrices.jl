@testset "_resolve_branch_arc: classifies direct, parallel, series, and unknown branches" begin
    sys = PSB.build_system(
        PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
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

    if !isempty(nr.reverse_transformer3W_map)
        winding, expected_arc = first(nr.reverse_transformer3W_map)
        tag, arc = PNM._resolve_branch_arc(nr, winding)
        @test tag === :transformer3w
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
    @test arc === nothing
end
