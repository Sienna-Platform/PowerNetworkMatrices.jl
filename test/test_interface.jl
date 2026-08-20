@testset "Test build_branch_maps_by_type" begin
    # This tests the function build_branch_maps_by_type by rebuilding the original branch maps and testing the round trip.
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = ybus.network_reduction_data
    all_branch_maps_by_type, _, _ = PNM.build_branch_maps_by_type(nrd)
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
        original_map = getproperty(nrd, entry)
        rebuilt_map = getproperty(nrd_rebuild, entry)
        @test original_map == rebuilt_map
    end
end
@testset "Test build_branch_maps_by_type does not mutate the reduction" begin
    # The maps depend on the caller's filters, not on the reduction alone, so the reduction
    # a matrix holds must come back untouched no matter how many callers derive maps from it.
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = ybus.network_reduction_data
    # Compared structurally, not against a deepcopy: the maps hold PSY components, which
    # have no `==` method, so a copied component never compares equal to its original.
    # `objectid` catches a reassigned field; `length` catches one mutated in place.
    fields = fieldnames(NetworkReductionData)
    sized = filter(f -> f !== :reductions, collect(fields))
    ids_before = Dict(f => objectid(getfield(nrd, f)) for f in fields)
    lengths_before = Dict(f => length(getfield(nrd, f)) for f in sized)

    PNM.build_branch_maps_by_type(nrd)
    PNM.build_branch_maps_by_type(nrd, Dict(Line => l -> false))

    for f in fields
        @test objectid(getfield(nrd, f)) == ids_before[f]
    end
    for f in sized
        @test length(getfield(nrd, f)) == lengths_before[f]
    end
end
@testset "Test build_branch_maps_by_type is filter-independent across calls" begin
    # The mutating predecessor accumulated name entries across calls while replacing the
    # by-type maps, so a second call with different filters left the two inconsistent.
    # Deriving fresh maps per call cannot drift that way.
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = ybus.network_reduction_data
    unfiltered_maps, unfiltered_names, _ = PNM.build_branch_maps_by_type(nrd)
    # Keep a line the reduction left as a one-to-one direct entry. Picking any line from the
    # system risks picking one the degree-two reduction absorbed into a series chain, which
    # carries no direct entry to retain.
    direct_lines = PNM.get_typed_direct_branch_map(unfiltered_maps, Line)
    @test !isempty(direct_lines)
    keep = get_name(first(values(direct_lines)))

    filtered_maps, filtered_names, _ =
        PNM.build_branch_maps_by_type(nrd, Dict(Line => l -> get_name(l) == keep))

    @test haskey(filtered_names, Line)
    @test length(filtered_names[Line]) < length(unfiltered_names[Line])
    @test haskey(filtered_names[Line], keep)
    # Every retained name resolves in the by-type maps built by the same call. The mutating
    # predecessor could not promise this once a second call used different filters.
    for (name, (arc, reduction_map)) in filtered_names[Line]
        @test haskey(filtered_maps[reduction_map][Line], arc)
    end
end
@testset "Test build_branch_maps_by_type omits fully filtered types" begin
    # A filter that excludes every branch of a type drops the type key entirely rather than
    # leaving an empty bucket, so callers must treat a missing key as "no entries".
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = ybus.network_reduction_data
    maps, names, _ = PNM.build_branch_maps_by_type(nrd, Dict(Line => l -> false))
    @test !haskey(names, Line)
    @test !haskey(maps.direct_branch_map, Line)
end
@testset "Test component_to_reduction_name_map" begin
    # This tests that each branch is included in the component_to_reduction_name_map.
    # component_to_reduction_name_map is used in POM for building N-1 problem so that
    # outages associated with branches can be mapped to the appropriate reduction entry.
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = ybus.network_reduction_data
    _, _, component_name_map = PNM.build_branch_maps_by_type(nrd)
    for g in get_components(ACTransmission, sys)
        (typeof(g) <: ThreeWindingTransformer) && continue      # Not yet supported
        (typeof(g) <: DiscreteControlledACBranch) && continue   # Automatically reduced
        @test haskey(component_name_map[typeof(g)], get_name(g))
    end
end
