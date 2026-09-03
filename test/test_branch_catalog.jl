@testset "BranchCatalog indexes every branch the reduction retains" begin
    for (label, sys, reductions) in branch_catalog_test_cases()
        ybus = Ybus(sys; network_reductions = deepcopy(reductions))
        catalog = PNM.get_branch_catalog(ybus)
        nrd = PNM.get_network_reduction_data(catalog)
        fp = catalog_fingerprint(catalog)

        # A reducing fixture must produce a non-empty index.
        @test !isempty(fp.by_type)
        @test !isempty(fp.names)

        # Every arc the index exposes must have both endpoints on the reduced bus axis.
        retained = keys(PNM.get_bus_reduction_map(nrd))
        for (_, _, arc) in fp.names
            @test arc[1] in retained
            @test arc[2] in retained
        end

        # The invariant is "indexed unless absorbed": a radial reduction absorbs a branch
        # outright, taking both endpoints off the bus axis and leaving no reduction entry.
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
    PSY.add_component!(
        sys,
        PSY.Line(;
            name = "zi_line", available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = zi_arc, r = 0.0, x = 1.0e-5,
            b = (from = 0.0, to = 0.0), rating = 10.0,
            angle_limits = (min = -1.57, max = 1.57),
        ),
    )

    ybus = Ybus(sys)
    nrd = PNM.get_network_reduction_data(ybus)
    merged_arc = (PSY.get_number(busD), PSY.get_number(star_bus))
    @assert haskey(nrd.parallel_branch_map, merged_arc) "fixture no longer promotes the windings"

    # Constructing the catalog must not throw.
    catalog = PNM.BranchCatalog(nrd)

    # The group is reachable under the parent transformer type.
    parent = PSY.ThreeWindingTransformer
    arc_map = PNM.get_name_to_arc_map(catalog, parent)
    @test merged_arc in values(arc_map)

    # And through the wrapper-keyed accessor, which redirects to the parent.
    wrapper_map = PNM.get_name_to_arc_map(catalog, PNM.ThreeWindingTransformerCircuit)
    @test merged_arc in values(wrapper_map)

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

@testset "BranchCatalog is built eagerly on every matrix" begin
    for (label, sys, reductions) in branch_catalog_test_cases()
        ybus = Ybus(sys; network_reductions = deepcopy(reductions),
            make_arc_admittance_matrices = true)
        for m in (ybus, PNM.VirtualFactorCore(ybus), IncidenceMatrix(ybus),
            AdjacencyMatrix(ybus), BA_Matrix(ybus), ABA_Matrix(ybus))
            catalog = PNM.get_branch_catalog(m)
            # The index exists because the matrix exists.
            @test !isempty(catalog_fingerprint(catalog).names)
            # The delegating getter resolves to the catalog's own reduction.
            @test PNM.get_network_reduction_data(m) ===
                  PNM.get_network_reduction_data(catalog)
        end
        # A contained arc-admittance matrix shares its parent's reduction object, so the
        # two can never describe different networks.
        @test PNM.get_network_reduction_data(ybus.arc_admittance_from_to) ===
              PNM.get_network_reduction_data(ybus)
    end
end

@testset "BranchCatalog resolves branch names for matrix indexing" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    ptdf = PTDF(Ybus(sys; network_reductions = PNM.NetworkReduction[]))

    # A branch owning its arc one-to-one scales by 1.0.
    nr = PNM.get_network_reduction_data(ptdf)
    arc, branch = first(PNM.get_direct_branch_map(nr))
    mult, resolved = PNM.get_branch_multiplier(ptdf, PNM.get_name(branch))
    @test mult == 1.0
    @test resolved == arc

    @test_throws ErrorException PNM.get_branch_multiplier(ptdf, "no_such_branch")
end

@testset "BranchCatalog reports colliding branch names instead of guessing" begin
    # PSY names are unique only per type, so a Line and a transformer may share one.
    # A bare-name lookup cannot resolve that, and must say so rather than pick one.
    # c_sys14 is the fixture that carries both a Line and a TwoWindingTransformer.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = first(PSY.get_available_components(PSY.Line, sys))
    transformer = first(PSY.get_available_components(PSY.TwoWindingTransformer, sys))
    shared = PSY.get_name(line)
    PSY.set_name!(sys, transformer, shared)

    ptdf = PTDF(Ybus(sys; network_reductions = PNM.NetworkReduction[]))
    err = try
        PNM.get_branch_multiplier(ptdf, shared)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Line", err.msg)
    @test occursin("TwoWindingTransformer", err.msg)
end

@testset "BranchCatalog files one reporting row per series segment" begin
    # `name_to_arc`'s keys are the rows results are reported under, and the rule is not
    # "one per arc": a lossless chain carries the same flow in every segment, so each
    # segment is a row a caller can look up under a real component name. Parallel members
    # are the exception -- their individual flows are never computed -- so a parallel group
    # reports once, whether it stands alone on an arc or sits inside a chain.
    #
    # Collapsing a chain to a single row passes every other test in this suite while
    # silently deleting result columns (measured: 691 rows -> 332 on ACTIVSg10k), which is
    # why the shape is pinned here explicitly.
    sys = PSB.build_system(PSITestSystems, "case11_network_reductions")
    ybus = Ybus(
        sys;
        network_reductions = PNM.NetworkReduction[
            PNM.RadialReduction(),
            PNM.DegreeTwoReduction(; reduce_reactive_power_injectors = false),
        ],
    )
    catalog = PNM.get_branch_catalog(ybus)
    line_rows = PNM.get_name_to_arc_map(catalog, PSY.Line)

    # The (1,2) chain is three plain segments -> three rows, all on the one arc.
    for name in ("1-6-i_1", "6-7-i_1", "7-2-i_1")
        @test haskey(line_rows, name)
        @test line_rows[name] == (1, 2)
    end

    # The (2,3) chain is one plain segment plus a parallel pair. The pair is ONE row, under
    # the group's own name -- neither member name is a row.
    @test haskey(line_rows, "10-3-i_1")
    @test line_rows["10-3-i_1"] == (2, 3)
    @test haskey(line_rows, "2_10_double_circuit")
    @test line_rows["2_10_double_circuit"] == (2, 3)
    @test !haskey(line_rows, "2-10-i_1")
    @test !haskey(line_rows, "2-10-i_2")

    # A top-level parallel group behaves the same way.
    @test haskey(line_rows, "1_4_double_circuit")
    @test !haskey(line_rows, "1-4-i_1")
    @test !haskey(line_rows, "1-4-i_2")

    # Redirects point every component at the row its flow is reported under: itself for a
    # plain segment, the group for a parallel member.
    redirects = PNM.get_component_to_reduction_name_map(catalog, PSY.Line)
    @test redirects["1-6-i_1"] == "1-6-i_1"
    @test redirects["2-10-i_1"] == "2_10_double_circuit"
    @test redirects["2-10-i_2"] == "2_10_double_circuit"
    @test redirects["1-4-i_1"] == "1_4_double_circuit"
    # Every redirect target must actually be a row, or the lookup dead-ends.
    for (_, entry_name) in redirects
        @test haskey(line_rows, entry_name)
    end
end

@testset "BranchCatalog reverse indexing defers to the forward filter verdict" begin
    # `_entry_matches` is not uniform across aggregates: `BranchesParallel` matches on
    # `any`, `MixedBranchesParallel` on `all`. Forward indexing evaluates the predicate on
    # the GROUP, reverse indexing on each MEMBER, so under `all` a member can pass while its
    # group did not. The forward pass owns `arcs`; reverse must take its verdict rather than
    # reach a different one, or it names an entry that is not a row.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = first(PSY.get_components(PSY.Line, sys))
    xfmr = first(PSY.get_components(PSY.TwoWindingTransformer, sys))
    arc = PSY.get_arc(line)

    # Force the two onto one arc, which promotes them to a mixed parallel group.
    nrd = PNM.NetworkReductionData()
    PNM.add_to_branch_maps!(nrd, arc, line)
    PNM.add_to_branch_maps!(nrd, arc, xfmr)
    arc_tuple = PNM.get_arc_tuple(arc, nrd)
    group = PNM.get_parallel_branch_map(nrd)[arc_tuple]
    @test group isa PNM.MixedBranchesParallel

    # Unfiltered: the group is a row and both members redirect to it. Without this the
    # filtered assertions below would pass on an empty catalog.
    base = PNM.BranchCatalog(nrd)
    group_name = PNM.get_name(PNM.get_reduction_entry(base, arc_tuple))
    @test PNM.get_name_to_arc_map(base, PSY.Line)[group_name] == arc_tuple
    @test PNM.get_component_to_reduction_name_map(base, PSY.Line)[PSY.get_name(line)] ==
          group_name
    @test PNM.get_component_to_reduction_name_map(
        base, PSY.TwoWindingTransformer,
    )[PSY.get_name(xfmr)] == group_name

    # Filtering out `Line` makes `all` reject the group while the transformer member still
    # passes. This threw `KeyError: $(arc_tuple)` from the reverse pass.
    filtered = PNM.BranchCatalog(nrd, (T, c) -> T !== PSY.Line)
    @test filtered isa PNM.BranchCatalog

    # The group is not a row anywhere...
    for T in (PSY.Line, PSY.TwoWindingTransformer)
        @test !haskey(PNM.get_name_to_arc_map(filtered, T), group_name)
    end
    # ...so no member may redirect to it, the transformer included even though it matched.
    @test !haskey(
        PNM.get_component_to_reduction_name_map(filtered, PSY.TwoWindingTransformer),
        PSY.get_name(xfmr),
    )
    @test !haskey(
        PNM.get_component_to_reduction_name_map(filtered, PSY.Line),
        PSY.get_name(line),
    )

    # The general invariant the above is one instance of: a redirect that names no row is a
    # dead end, which is what the pre-guard code wrote before it started throwing.
    for (T, redirects) in PNM.get_component_to_reduction_name_map(filtered)
        rows = PNM.get_name_to_arc_map(filtered, T)
        for (_, entry_name) in redirects
            @test haskey(rows, entry_name)
        end
    end
end
