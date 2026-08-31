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
        for (_, _, arc, _) in fp.names
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
    @test any(v -> v == (merged_arc, PNM.ParallelArc()), values(arc_map))

    # And through the wrapper-keyed accessor, which redirects to the parent.
    wrapper_map = PNM.get_name_to_arc_map(catalog, PNM.ThreeWindingTransformerCircuit)
    @test any(v -> v == (merged_arc, PNM.ParallelArc()), values(wrapper_map))

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

@testset "arc_provenance answers from the entry's type" begin
    # The blanket arm is `::PSY.ACTransmission`, which aggregates subtype. Each aggregate
    # needs its own arm or it is silently reported as an untouched physical branch -- the
    # hazard `AbstractReductionAggregate` exists to expose.
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    line = first(PSY.get_components(PSY.Line, sys))
    @test PNM.arc_provenance(line) == PNM.DirectArc()

    chains = PNM.get_series_branch_map(nrd)
    groups = PNM.get_parallel_branch_map(nrd)
    @test !isempty(groups)
    for (_, group) in groups
        @test PNM.arc_provenance(group) == PNM.ParallelArc()
        @test PNM.arc_provenance(group) != PNM.DirectArc()
    end
    for (_, chain) in chains
        @test PNM.arc_provenance(chain) == PNM.SeriesArc()
        @test PNM.arc_provenance(chain) != PNM.DirectArc()
    end

    # A Ward equivalent is a PSY component, so only its own arm keeps it from claiming the
    # component backing that `DirectArc` asserts.
    ward_arc = PSY.GenericArcImpedance(;
        name = "ward_equivalent",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        max_flow = 1e6,
        arc = PSY.Arc(nothing),
        r = 0.01,
        x = 0.1,
    )
    @test PNM.arc_provenance(ward_arc) == PNM.SyntheticArc()
end

@testset "Provenance reaches the name-to-arc index and the multiplier" begin
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    catalog = PNM.get_branch_catalog(ybus)

    # Every entry carries a provenance singleton, never a bare symbol.
    seen = Set{PNM.ArcProvenance}()
    for (_, by_name) in PNM.get_name_to_arc_maps(catalog)
        for (_, provenance) in values(by_name)
            @test provenance isa PNM.ArcProvenance
            push!(seen, provenance)
        end
    end
    # This fixture folds sibling chains into one parallel group, so both kinds appear.
    @test PNM.ParallelArc() in seen

    # `get_branch_multiplier` now routes by provenance rather than by an equality test that
    # sent everything non-direct down the parallel path.
    @test PNM._branch_multiplier(PNM.DirectArc(), catalog, "anything", (1, 3)) == 1.0
    @test PNM._branch_multiplier(PNM.SyntheticArc(), catalog, "anything", (1, 3)) == 1.0

    # A series segment reports why it has no multiplier instead of failing as a missing key
    # in the parallel map.
    err = try
        PNM._branch_multiplier(PNM.SeriesArc(), catalog, "L_1_10", (1, 3))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("series chain", err.msg)
    @test occursin("component identity", err.msg)
end
