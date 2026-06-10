@testset "Test connected networks" begin
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_case5_sys")
    @test validate_connectivity(sys)
    @test(
        @test_logs (
            :info,
            "Validating connectivity with depth first search (network traversal)",
        ) match_mode = :any validate_connectivity(sys)
    )
end

@testset "Test disconnected networks" begin
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_case5_sys")
    remove_components!(sys, Line)
    @test(
        @test_logs (
            :warn,
            "Bus 1 is islanded",
        ) match_mode = :any validate_connectivity(sys) == false
    )
end

@testset "Test connected components" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    M = Ybus(sys5)
    subnetworks = find_subnetworks(M)
    @test length(subnetworks) == 1

    sys10 = PSB.build_system(PSISystems, "2Area 5 Bus System")
    M = Ybus(sys10)
    subnetworks_m = find_subnetworks(M)
    @test length(subnetworks_m) == 2
    @test all([6, 1] .∈ keys(subnetworks_m))

    subnetworks_sys = find_subnetworks(sys10)
    @test all([4, 9] .∈ keys(subnetworks_sys))
end

@testset "Test find subnetworks" begin
    n = 11
    buses = 100 .+ collect(1:n)
    edge_inds = [(1, 2), (2, 3), (3, 1), # cycle
        (4, 5), (6, 7), (8, 4), (8, 6), # two short chains that merge.
        # 9 is isolated.
        (10, 11)]
    A = SparseArrays.sparse(I(n))
    for (i, j) in edge_inds
        A[i, j] = 1
        A[j, i] = 1
    end
    test_subnetworks = PNM.find_subnetworks(A, buses)
    expected =
        [Set(100 .+ (1:3)), Set(100 .+ (4:8)), Set(100 .+ (9:9)), Set(100 .+ (10:11))]
    @test length(values(test_subnetworks)) == length(expected)
    for (k, v) in test_subnetworks
        @test k in v
    end
    for k in expected
        @test k in values(test_subnetworks)
    end
end

@testset "Test matrices for connectivity corner cases" begin
    sys = PSB.build_system(PSISystems, "HVDC_TWO_RTO_RTS_5min_sys")
    ybus = Ybus(sys)
    A = IncidenceMatrix(ybus)
    ptdf = PTDF(sys)
    lodf = LODF(sys)
    vptdf = VirtualPTDF(sys)
    vlodf = VirtualLODF(sys)

    @test length(ybus.subnetwork_axes) == 2
    @test keys(ybus.subnetwork_axes) == keys(ptdf.subnetwork_axes) ==
          keys(lodf.subnetwork_axes) == keys(vptdf.subnetwork_axes) ==
          keys(vlodf.subnetwork_axes)
    for k in keys(ptdf.subnetwork_axes)
        @test iszero([ptdf[x, k] for x in PNM.get_arc_axis(ptdf)])
    end
    ref_bus_numbers = [
        get_number(x) for
        x in PSY.get_components(x -> get_bustype(x) == ACBusTypes.REF, ACBus, sys)
    ]
    for ref_bus in ref_bus_numbers
        @test ref_bus ∈ keys(ybus.subnetwork_axes)
    end
end

@testset "Small island corner cases" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    ybus_1 = Ybus(sys)
    ptdf_1 = PTDF(sys)
    lodf_1 = LODF(sys)
    vptdf_1 = VirtualPTDF(sys)
    vlodf_1 = VirtualLODF(sys)

    sys = build_hvdc_with_single_bus_island()
    ybus_2 = Ybus(sys)
    ptdf_2 = PTDF(sys)
    lodf_2 = LODF(sys)
    vptdf_2 = VirtualPTDF(sys)
    vlodf_2 = VirtualLODF(sys)

    sys = build_hvdc_with_small_island()
    ybus_3 = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set(collect(1:14)),
    )
    ptdf_3 = PTDF(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set(collect(1:14)),
    )
    lodf_3 = LODF(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set(collect(1:14)),
    )
    vptdf_3 = VirtualPTDF(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set(collect(1:14)),
    )
    vlodf_3 = VirtualLODF(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set(collect(1:14)),
    )

    for i in ptdf_1.axes[1], j in ptdf_1.axes[2]
        @test ptdf_1[j, i] == ptdf_2[j, i] == ptdf_3[j, i]
    end
    for i in lodf_1.axes[1], j in lodf_1.axes[2]
        @test lodf_1[i, j] == lodf_2[i, j] == lodf_3[i, j]
    end
    for i in vptdf_1.axes[1], j in vptdf_1.axes[2]
        @test vptdf_1[i, j] == vptdf_2[i, j] == vptdf_3[i, j]
    end
    for i in vlodf_1.axes[1], j in vlodf_1.axes[2]
        @test vlodf_1[i, j] == vlodf_2[i, j] == vlodf_3[i, j]
    end
end

@testset "Anti-parallel branches stay connected for DFS connectivity" begin
    # Two branches between the same bus pair with opposite from/to arcs each contribute
    # +1/-1 to the signed bus adjacency; summed naively they cancel to zero, which hides
    # the connection from the value-based DFS connectivity check (`find_connected_components`,
    # the scalable alternative to Goderya). The Ybus build must retain the last branch's
    # orientation (matching the original overwrite semantics) and warn. Real datasets
    # contain anti-parallel lines, so this must hold.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    grid_bus = first(
        b for b in get_components(ACBus, sys) if
        get_bustype(b) != PSY.ACBusTypes.REF && get_bustype(b) != PSY.ACBusTypes.ISOLATED
    )
    stub = deepcopy(first(get_components(ACBus, sys)))
    stub.internal = IS.InfrastructureSystemsInternal()
    set_number!(stub, 950)
    set_name!(stub, "ANTIPARALLEL_STUB")
    set_bustype!(stub, PSY.ACBusTypes.PQ)
    add_component!(sys, stub)
    function _mk_line(name, from, to)
        arc = Arc(from, to)
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(
                name,
                true,
                0.0,
                0.0,
                arc,
                0.01,
                0.10,
                (from = 0.0, to = 0.0),
                100.0,
                (-1.5, 1.5),
            ),
        )
    end
    # The stub is reachable from the grid only through this anti-parallel pair.
    _mk_line("AP_fwd", grid_bus, stub)   # grid -> stub
    _mk_line("AP_rev", stub, grid_bus)   # stub -> grid (anti-parallel)

    ybus =
        @test_logs (:warn, r"Anti-parallel branches between buses") match_mode = :any Ybus(
            sys,
        )

    i = ybus.lookup[1][get_number(grid_bus)]
    j = ybus.lookup[1][950]
    # The signed entries are retained (not cancelled to zero) and stay antisymmetric.
    @test ybus.adjacency_data[i, j] != 0
    @test ybus.adjacency_data[j, i] != 0
    @test ybus.adjacency_data[i, j] == -ybus.adjacency_data[j, i]

    # Value-based DFS connectivity sees the stub connected: a single component.
    cc = PNM.find_connected_components(ybus.adjacency_data, ybus.lookup[1])
    @test length(cc) == 1
    # Structure-based union find agrees.
    @test length(PNM.find_subnetworks(ybus.adjacency_data, ybus.axes[1])) == 1
end

@testset "Subnetwork algorithms" begin
    sys = build_hvdc_with_small_island()
    ybus = @test_logs (:info, r"Finding subnetworks via iterative union find") match_mode =
        :any Ybus(sys)
    ybus =
        @test_logs (:info, r"Finding subnetworks via depth first search") match_mode = :any Ybus(
            sys;
            subnetwork_algorithm = depth_first_search,
        )

    sub_1 = PNM.find_subnetworks(
        ybus.data,
        ybus.axes[1];
        subnetwork_algorithm = depth_first_search,
    )
    sub_2 = PNM.find_subnetworks(
        ybus.data,
        ybus.axes[1];
        subnetwork_algorithm = iterative_union_find,
    )
    @test sub_1 == sub_2
end
