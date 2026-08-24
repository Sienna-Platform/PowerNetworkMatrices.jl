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

@testset "BranchCatalog base constructor matches populate_branch_maps_by_type!" begin
    for (label, sys, reductions) in branch_catalog_test_cases()
        ybus = Ybus(sys; network_reductions = deepcopy(reductions))
        nrd = PNM.get_network_reduction_data(ybus)
        PNM.populate_branch_maps_by_type!(nrd)

        catalog = PNM.BranchCatalog(nrd)
        @test catalog_fingerprint(catalog) == nrd_fingerprint(nrd)
        # The catalog indexes the NRD it was handed, not a copy.
        @test PNM.get_network_reduction_data(catalog) === nrd
        # An absent branch type yields an empty map, never a KeyError. PF and POM both
        # need this: a type can be missing when every branch of it was absorbed by a
        # radial reduction. Probe whichever concrete types this fixture lacks rather than
        # naming one, so the assertion holds across fixtures.
        indexed = keys(PNM.get_name_to_arc_maps(catalog))
        for T in (
            PSY.Line, PSY.MonitoredLine, PSY.TwoWindingTransformer,
            PSY.ThreeWindingTransformer, PSY.GenericArcImpedance,
            PSY.DiscreteControlledACBranch,
        )
            T in indexed && continue
            @test isempty(PNM.get_name_to_arc_map(catalog, T))
        end
    end
end

@testset "BranchCatalog filtered constructor matches populate with filters" begin
    # Each entry: a label, and filters keyed by PSY concrete type exactly as
    # IOM._get_filters produces them (keys are DeviceModel component types).
    filter_configs = [
        ("reject_all_lines", Dict{DataType, Function}(PSY.Line => (_ -> false))),
        ("keep_all_lines", Dict{DataType, Function}(PSY.Line => (_ -> true))),
        # The important one: a filter keeping SOME members is what distinguishes the ANY
        # rule on BranchesParallel from the ALL rule on MixedBranchesParallel and
        # BranchesSeries. Reject-all and keep-all cannot tell them apart.
        ("reject_half_lines", Dict{DataType, Function}(
            PSY.Line => (b -> isodd(length(PSY.get_name(b)))),
        )),
    ]
    for (label, sys, reductions) in branch_catalog_test_cases()
        for (fl, filters) in filter_configs
            base_nrd = PNM.get_network_reduction_data(
                Ybus(sys; network_reductions = deepcopy(reductions)),
            )
            PNM.populate_branch_maps_by_type!(base_nrd, filters)

            catalog_nrd = PNM.get_network_reduction_data(
                Ybus(sys; network_reductions = deepcopy(reductions)),
            )
            predicate = (T, c) -> haskey(filters, T) ? filters[T](c) : true
            catalog = PNM.BranchCatalog(catalog_nrd, predicate)

            @test catalog_fingerprint(catalog) == nrd_fingerprint(base_nrd)
        end
    end
end

@testset "BranchCatalog filtered is a subset sharing one reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(sys; network_reductions = PNM.NetworkReduction[PNM.DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    base = PNM.BranchCatalog(nrd)
    filtered = PNM.BranchCatalog(nrd, (T, c) -> T !== PSY.Line)

    @test PNM.get_network_reduction_data(filtered) === PNM.get_network_reduction_data(base)
    bf, ff = catalog_fingerprint(base), catalog_fingerprint(filtered)
    @test issubset(Set(ff.names), Set(bf.names))
    @test issubset(Set(ff.by_type), Set(bf.by_type))
    @test length(ff.names) < length(bf.names)
end

@testset "BranchCatalog indexes a 3W winding promoted into a parallel group" begin
    # A non-winding zero-impedance branch between two terminal buses of one 3W transformer
    # merges them, so two winding arcs collide on (merged, star) and become a parallel
    # group. Both are indexed under the PARENT transformer type, never the wrapper.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD)
    _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus; name = "T3W_degenerate",
    )
    zi_arc = PSY.Arc(; from = busD, to = sec_bus)
    PSY.add_component!(sys, zi_arc)
    PSY.add_component!(sys, PSY.Line(;
        name = "zi_line", available = true, active_power_flow = 0.0,
        reactive_power_flow = 0.0, arc = zi_arc, r = 0.0, x = 1.0e-5,
        b = (from = 0.0, to = 0.0), rating = 10.0,
        angle_limits = (min = -1.57, max = 1.57),
    ))

    ybus = Ybus(sys)
    nrd = PNM.get_network_reduction_data(ybus)
    merged_arc = (PSY.get_number(busD), PSY.get_number(star_bus))
    @assert haskey(nrd.parallel_branch_map, merged_arc) "fixture no longer promotes the windings"

    # Constructing the catalog must not throw (defect A).
    catalog = PNM.BranchCatalog(nrd)

    # The group is reachable under the parent transformer type (defect B).
    parent = PSY.ThreeWindingTransformer
    arc_map = PNM.get_name_to_arc_map(catalog, parent)
    @test any(v -> v == (merged_arc, :parallel_branch_map), values(arc_map))

    # And through the wrapper-keyed accessor, which redirects to the parent.
    wrapper_map = PNM.get_name_to_arc_map(catalog, PNM.ThreeWindingTransformerCircuit)
    @test any(v -> v == (merged_arc, :parallel_branch_map), values(wrapper_map))

    # No bucket anywhere is keyed by the wrapper type.
    maps = PNM.get_all_branch_maps_by_type(catalog)
    for (_, per_type) in maps
        @test !haskey(per_type, PNM.ThreeWindingTransformerCircuit)
    end
    @test !haskey(PNM.get_name_to_arc_maps(catalog), PNM.ThreeWindingTransformerCircuit)

    # Each winding still redirects to the group entry that carries its flow.
    redirects = PNM.get_component_to_reduction_name_map(catalog, parent)
    @test haskey(redirects, "T3W_degenerate_winding_1")
    @test haskey(redirects, "T3W_degenerate_winding_2")
end
