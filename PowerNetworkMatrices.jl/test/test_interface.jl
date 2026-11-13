@testset "Test component_to_reduction_name_map" begin
    # This tests that each branch is included in the component_to_reduction_name_map.
    # component_to_reduction_name_map is used in PSI for building N-1 problem so that
    # outages associated with branches can be mapped to the appropriate reduction entry.
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    catalog = PNM.get_branch_catalog(ybus)
    component_name_map = PNM.get_component_to_reduction_name_map(catalog)
    for g in get_components(ACTransmission, sys)
        (typeof(g) <: ThreeWindingTransformer) && continue      # Not yet supported
        (typeof(g) <: DiscreteControlledACBranch) && continue   # Automatically reduced 
        @test haskey(component_name_map[typeof(g)], get_name(g))
    end
end
