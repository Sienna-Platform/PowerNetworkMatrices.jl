@testset "BranchCatalog round-trips the reduction maps" begin
    # Rebuild the reduction maps from the catalog's per-type buckets and check the round trip.
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    catalog = PNM.get_branch_catalog(ybus)
    nrd = PNM.get_network_reduction_data(catalog)
    all_branch_maps_by_type = PNM.get_all_branch_maps_by_type(catalog)
    nrd_rebuild = NetworkReductionData()
    for (map_key, v1) in all_branch_maps_by_type
        for (type, v2) in v1
            for (entry, v3) in v2
                map = getproperty(nrd_rebuild, Symbol(map_key))
                map[entry] = v3
            end
        end
    end
    for entry in [
        :direct_branch_map,
        :reverse_direct_branch_map,
        :parallel_branch_map,
        :reverse_parallel_branch_map,
        :series_branch_map,
        :reverse_series_branch_map,
    ]
        original_map = getproperty(PNM.get_network_reduction_data(catalog), entry)
        rebuilt_map = getproperty(nrd_rebuild, entry)
        @test original_map == rebuilt_map
    end
end
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
