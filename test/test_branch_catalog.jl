@testset "BranchCatalog characterization: current index is complete" begin
    for (label, sys, reductions) in branch_catalog_test_cases()
        ybus = Ybus(sys; network_reductions = deepcopy(reductions))
        nrd = PNM.get_network_reduction_data(ybus)
        PNM.populate_branch_maps_by_type!(nrd)
        fp = nrd_fingerprint(nrd)

        # A reducing fixture must produce a non-empty index. This is the assertion that
        # would have caught the vacuous-pass footgun in the downstream tests.
        @test !isempty(fp.by_type)
        @test !isempty(fp.names)

        # Every arc the index exposes must have both endpoints on the reduced bus axis.
        retained = keys(PNM.get_bus_reduction_map(nrd))
        for (_, _, arc, _) in fp.names
            @test arc[1] in retained
            @test arc[2] in retained
        end

        # Every available branch is either reachable by name through the redirect map, or
        # was absorbed by the reduction. A radial reduction absorbs a branch outright --
        # both its endpoints leave the bus axis and it carries no reduction entry at all --
        # so "present in the index" is not the invariant; "present unless absorbed" is.
        removed_arcs = PNM.get_removed_arcs(nrd)
        for branch in PSY.get_available_components(PSY.ACTransmission, sys)
            branch isa PSY.ThreeWindingTransformer && continue
            branch isa PSY.DiscreteControlledACBranch && continue
            T = string(typeof(branch))
            name = PSY.get_name(branch)
            arc = PNM.get_arc_tuple(branch)
            absorbed = arc in removed_arcs || reverse(arc) in removed_arcs
            @test absorbed || any(r -> r[1] == T && r[2] == name, fp.redirects)
        end
    end
end

@testset "BranchCatalog characterization: filters restrict the index" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")

    base_nrd = PNM.get_network_reduction_data(
        Ybus(sys; network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()]),
    )
    PNM.populate_branch_maps_by_type!(base_nrd)
    base = nrd_fingerprint(base_nrd)

    # Reject every Line. Lines are the only type with enough members in this fixture to
    # make the restriction observable in all three index structures.
    filtered_nrd = PNM.get_network_reduction_data(
        Ybus(sys; network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()]),
    )
    PNM.populate_branch_maps_by_type!(
        filtered_nrd,
        Dict{DataType, Function}(PSY.Line => (_ -> false)),
    )
    filtered = nrd_fingerprint(filtered_nrd)

    @test issubset(Set(filtered.names), Set(base.names))
    @test issubset(Set(filtered.by_type), Set(base.by_type))
    @test length(filtered.names) < length(base.names)
end
