function types_in_series_reduction(nrd::PNM.NetworkReductionData)
    types = Set{DataType}()
    for segments in values(PNM.get_series_branch_map(nrd))
        for comp in segments
            push!(types, typeof(comp))
        end
    end
    return types
end

function find_parallel_arc(sys::System)
    arcs_seen = Set{Tuple{Int, Int}}()
    for br in PSY.get_components(
        x -> !(typeof(x) <: PSY.ThreeWindingTransformer),
        PSY.ACBranch,
        sys,
    )
        arc = PNM.get_arc_tuple(br)
        if arc in arcs_seen
            return arc
        else
            push!(arcs_seen, arc)
        end
    end
    error("No parallel arcs found")
    return (-1, -1)
end

function test_all_subtypes(sys::System, network_reductions)
    for T in subtypes(PNM.PowerNetworkMatrix)
        # arc admittance matrix constructor has different args.
        (T == PNM.Ybus || T == PNM.ArcAdmittanceMatrix) && continue
        M = T(sys; network_reductions = deepcopy(network_reductions))
        # test that it runs without error
        @test M isa T
    end
    # return Ybus so we can inspect the network reduction data.
    Y = Ybus(sys; network_reductions = deepcopy(network_reductions))
    @test Y isa PNM.Ybus
    return Y
end

@testset "3WT + radial" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[RadialReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the 3WT arc was actually reduced
    trf = first(get_components(PSY.Transformer3W, sys))
    trf_arcs = Tuple{Int, Int}[
        PNM.get_arc_tuple(PSY.get_primary_star_arc(trf)),
        PNM.get_arc_tuple(PSY.get_secondary_star_arc(trf)),
        PNM.get_arc_tuple(PSY.get_tertiary_star_arc(trf)),
    ]
    nrd = PNM.get_network_reduction_data(Y)
    @test any(arc in PNM.get_removed_arcs(nrd) for arc in trf_arcs) ||
          any(reverse(arc) in PNM.get_removed_arcs(nrd) for arc in trf_arcs)
end

@testset "3WT + degree-2" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[DegreeTwoReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the 3WT arc was actually reduced
    nrd = PNM.get_network_reduction_data(Y)
    @test PNM.ThreeWindingTransformerWinding{Transformer3W} in
          types_in_series_reduction(nrd)
end

@testset "Parallel lines + radial" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[RadialReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the parallel lines were reduced
    nrd = PNM.get_network_reduction_data(Y)
    parallel_arc = find_parallel_arc(sys)
    @test parallel_arc in keys(nrd.parallel_branch_map)
end

@testset "Parallel lines + degree-2" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[DegreeTwoReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the parallel lines were reduced
    nrd = PNM.get_network_reduction_data(Y)
    @test PNM.BranchesParallel{Line} in types_in_series_reduction(nrd)
end

@testset "MixedBranchesParallel construction and methods" begin
    bus1 = PSY.ACBus(;
        number = 101,
        name = "bus_101",
        available = true,
        bustype = PSY.ACBusTypes.PV,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
    bus2 = PSY.ACBus(;
        number = 102,
        name = "bus_102",
        available = true,
        bustype = PSY.ACBusTypes.PV,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
    line = PSY.Line(;
        name = "mixed_line",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = bus1, to = bus2),
        r = 0.05,
        x = 0.10,
        b = (from = 0.01, to = 0.01),
        g = (from = 0.0, to = 0.0),
        rating = 100.0,
        angle_limits = (min = -π / 2, max = π / 2),
    )
    tap = PSY.TapTransformer(;
        name = "mixed_tap",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = bus1, to = bus2),
        r = 0.122,
        x = 0.10,
        primary_shunt = 0.01 + im * 0.02,
        tap = 1.0,
        rating = 80.0,
        base_power = 100.0,
        winding_group_number = WindingGroupNumber.GROUP_11,
    )

    # Attach the branches to a system so the unit-aware getters used by
    # ybus_branch_entries can resolve the system base.
    sys = System(100.0)
    add_component!(sys, bus1)
    add_component!(sys, bus2)
    add_component!(sys, line)
    add_component!(sys, tap)

    # Homogeneous group dispatches to BranchesParallel{Line}.
    bp_homog = PNM.BranchesParallel([line])
    @test bp_homog isa PNM.BranchesParallel{PSY.Line}
    @test bp_homog isa PNM.AbstractBranchesParallel

    # The {T} parameter must be concrete; abstract T should fail at construction.
    @test_throws ErrorException PNM.BranchesParallel{PSY.ACTransmission}(
        PSY.ACTransmission[line, tap],
        nothing,
    )

    # MixedBranchesParallel holds heterogeneous branches.
    mbp = PNM.MixedBranchesParallel([line, tap])
    @test mbp isa PNM.MixedBranchesParallel
    @test mbp isa PNM.AbstractBranchesParallel
    @test length(mbp) == 2
    @test eltype(mbp.branches) === PSY.ACTransmission

    # ybus_branch_entries on the mixed group should equal the sum of the parts.
    Y11_l, Y12_l, Y21_l, Y22_l = PNM.ybus_branch_entries(line)
    Y11_t, Y12_t, Y21_t, Y22_t = PNM.ybus_branch_entries(tap)
    Y11_m, Y12_m, Y21_m, Y22_m = PNM.ybus_branch_entries(mbp)
    @test Y11_m ≈ Y11_l + Y11_t
    @test Y12_m ≈ Y12_l + Y12_t
    @test Y21_m ≈ Y21_l + Y21_t
    @test Y22_m ≈ Y22_l + Y22_t

    # Explicit equivalent-rating strategies for parallel groups; emergency rating sums them.
    @test PNM.get_sum_of_max_rating(mbp) ≈ 100.0 + 80.0
    @test PNM.get_single_element_contingency_rating(mbp) ≈ 80.0
    @test PNM.get_equivalent_emergency_rating(mbp) ≈ 100.0 + 80.0

    # add_to_map: empty filters short-circuit (no warning).
    @test PNM.add_to_map(mbp, Dict{DataType, Function}()) == true

    # add_to_map: non-empty filters trigger the mixed-type warning.
    filters = Dict{DataType, Function}(PSY.Line => x -> true)
    @test_logs (:warn, r"mixed branch types") match_mode = :any begin
        @test PNM.add_to_map(mbp, filters) == true
    end
end

@testset "Test Reductions with filters" begin
    sys_rts_da = build_system(PSISystems, "modified_RTS_GMLC_DA_sys")

    ptdf = VirtualPTDF(
        sys_rts_da;
        network_reductions = NetworkReduction[
            RadialReduction(),
            DegreeTwoReduction(),
        ],
    )
    PowerNetworkMatrices.populate_branch_maps_by_type!(PNM.get_network_reduction_data(ptdf),
        Dict(Line => x -> occursin("B", get_name(x)),
            TapTransformer => x -> occursin("B", get_name(x))))
    @test PNM.has_filtered_branches(PNM.get_network_reduction_data(ptdf))
    for k in keys(PNM.get_network_reduction_data(ptdf).name_to_arc_map[Line])
        @test occursin("B", k)
    end
    PNM.empty!(PNM.get_network_reduction_data(ptdf))
    @test isempty(PNM.get_network_reduction_data(ptdf))
end
