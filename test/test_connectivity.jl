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

    # The three constructions agree to floating-point noise. Compare with an
    # absolute tolerance: many entries are physically zero, so a relative
    # tolerance (the `≈`/isapprox default) is meaningless there, and the dense
    # and on-demand paths drift by a few ULPs run-to-run on the IS4/psy6 stack.
    atol = 1e-10
    for i in ptdf_1.axes[1], j in ptdf_1.axes[2]
        @test isapprox(ptdf_1[j, i], ptdf_2[j, i]; atol = atol) &&
              isapprox(ptdf_2[j, i], ptdf_3[j, i]; atol = atol)
    end
    for i in lodf_1.axes[1], j in lodf_1.axes[2]
        @test isapprox(lodf_1[i, j], lodf_2[i, j]; atol = atol) &&
              isapprox(lodf_2[i, j], lodf_3[i, j]; atol = atol)
    end
    for i in vptdf_1.axes[1], j in vptdf_1.axes[2]
        @test isapprox(vptdf_1[i, j], vptdf_2[i, j]; atol = atol) &&
              isapprox(vptdf_2[i, j], vptdf_3[i, j]; atol = atol)
    end
    for i in vlodf_1.axes[1], j in vlodf_1.axes[2]
        @test isapprox(vlodf_1[i, j], vlodf_2[i, j]; atol = atol) &&
              isapprox(vlodf_2[i, j], vlodf_3[i, j]; atol = atol)
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

@testset "assign_reference_buses! supports multiple swings per island (multi-swing)" begin
    # A network island may hold more than one swing (reference) bus; each is a valid
    # fixed-complex-voltage anchor. assign_reference_buses! must accept that (not error) and
    # key the island by its smallest-angle swing (tie-break: smallest bus number).
    # Island A = {1,2,3} with two swings (1 and 2); island B = {4,5} with one swing (4).

    # Unequal angles: swing 2 has the smaller angle, so it keys island A (NOT bus number 1).
    subnetworks = Dict{Int, Set{Int}}(3 => Set([1, 2, 3]), 5 => Set([4, 5]))
    ref_buses = Set([1, 2, 4])
    ref_angles = Dict(1 => 0.20, 2 => 0.05, 4 => 0.0)
    groups = PNM.assign_reference_buses!(subnetworks, ref_buses, ref_angles)
    @test length(groups) == 2
    for (k, buses) in groups
        @test k in intersect(ref_buses, buses)   # keyed by one of its OWN swings
    end
    @test haskey(groups, 2) && groups[2] == Set([1, 2, 3])   # min-angle swing
    @test Set([4, 5]) in values(groups)

    # Equal angles (e.g. flat start): tie-break on smallest bus number → swing 1.
    sub2 = Dict{Int, Set{Int}}(3 => Set([1, 2, 3]), 5 => Set([4, 5]))
    groups2 = PNM.assign_reference_buses!(
        sub2, Set([1, 2, 4]), Dict(1 => 0.0, 2 => 0.0, 4 => 0.0),
    )
    @test haskey(groups2, 1) && groups2[1] == Set([1, 2, 3])
end

function _mk_star_bus(number, name, bustype, angle)
    return ACBus(;
        number = number,
        name = name,
        available = true,
        bustype = bustype,
        angle = angle,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
end

function _mk_star_line!(sys, name, from, to, x)
    arc = Arc(; from = from, to = to)
    add_component!(sys, arc)
    return _add_test_line!(sys, name, arc, 0.0, x)
end

# x = 1e-4 gives susceptance 1e4, at or above the default ZIBR merge threshold.
_mk_star_jumper!(sys, name, from, to) = _mk_star_line!(sys, name, from, to, 1e-4)
_mk_star_normal_line!(sys, name, from, to) = _mk_star_line!(sys, name, from, to, 0.1)

# A star -- hub bus 10 (REF) with leaves 11-15 joined to it by zero-impedance jumpers, plus
# two ordinary-line buses (20, 21) -- where leaf 12 has the smallest swing angle and is
# therefore the island representative `assign_reference_buses!` picks, but the ZIBR merge
# direction (arc from->to, hub is `from`) absorbs it into hub 10. Shared by every testset
# below that needs a system where a ZIBR merge removes the island's representative bus.
function _mk_zibr_merged_representative_system()
    sys = System(100.0)
    hub = _mk_star_bus(10, "hub", ACBusTypes.REF, 0.30)
    leaf11 = _mk_star_bus(11, "leaf11", ACBusTypes.PV, 0.0)
    leaf12 = _mk_star_bus(12, "leaf12", ACBusTypes.REF, 0.01)   # smallest angle -> representative
    leaf13 = _mk_star_bus(13, "leaf13", ACBusTypes.PQ, 0.0)
    leaf14 = _mk_star_bus(14, "leaf14", ACBusTypes.PQ, 0.0)
    leaf15 = _mk_star_bus(15, "leaf15", ACBusTypes.PQ, 0.0)
    # Two buses on ordinary (non-zero-impedance) lines so the reduced network still has
    # non-reference buses left for ABA_Matrix / PTDF / etc. to build a non-trivial system from.
    pq20 = _mk_star_bus(20, "pq20", ACBusTypes.PQ, 0.0)
    pq21 = _mk_star_bus(21, "pq21", ACBusTypes.PQ, 0.0)
    for b in (hub, leaf11, leaf12, leaf13, leaf14, leaf15, pq20, pq21)
        add_component!(sys, b)
    end
    _mk_star_jumper!(sys, "J11", hub, leaf11)
    _mk_star_jumper!(sys, "J12", hub, leaf12)
    _mk_star_jumper!(sys, "J13", hub, leaf13)
    _mk_star_jumper!(sys, "J14", hub, leaf14)
    _mk_star_jumper!(sys, "J15", hub, leaf15)
    _mk_star_normal_line!(sys, "N20", hub, pq20)
    _mk_star_normal_line!(sys, "N21", pq20, pq21)
    return sys
end

@testset "A reduction re-keys a removed subnetwork representative to a surviving bus" begin
    # Regression: _make_subnetwork_axes replaced a removed representative with `pop!(axis_1)`
    # -- an arbitrary island member that the same reduction could also be removing, and which
    # `pop!` then deleted from the island's own bus list. That left subnetwork_axes keyed by a
    # bus absent from the matrix, which only worked because get_ref_bus_position chased the
    # reduction's reverse map at lookup time. The key must be a live bus by construction.
    sys = _mk_zibr_merged_representative_system()
    ybus = Ybus(sys)
    bus_lookup = PNM.get_bus_lookup(ybus)

    for (ref_bus, subnetwork_axis) in ybus.subnetwork_axes
        @test haskey(bus_lookup, ref_bus)
        @test ref_bus in subnetwork_axis[1]
    end

    # Leaf 12 was the representative and ZIBR merged it into hub 10, so 10 inherits the role.
    @test PNM.get_ref_bus(ybus) == [10]
end

@testset "get_ref_bus_position survives a ZIBR merge of the island representative" begin
    # Regression: assign_reference_buses! keys an island by its smallest-angle swing,
    # independent of which bus a ZeroImpedanceBranchReduction later merges away. When the
    # representative itself gets merged, get_ref_bus_position must resolve the removed
    # representative to its surviving bus via reverse_bus_search_map instead of throwing a
    # bare KeyError.
    sys = _mk_zibr_merged_representative_system()

    ybus = Ybus(sys)
    @test Set(keys(PNM.get_bus_lookup(ybus))) == Set([10, 20, 21])

    ref_bus_positions = PNM.get_ref_bus_position(ybus)
    @test all(1 .<= ref_bus_positions .<= length(PNM.get_bus_axis(ybus)))
    @test PNM.get_bus_axis(ybus)[only(ref_bus_positions)] == 10

    aba = ABA_Matrix(ybus; factorize = true)
    @test aba isa ABA_Matrix
end

@testset "get_ref_bus_position survives a ZIBR merge on the sibling matrix constructors" begin
    # Same fixture and defect class as the Ybus testset above, exercised through the other
    # PowerNetworkMatrix subtypes that share the generic get_ref_bus_position(M::PowerNetworkMatrix)
    # method (PowerNetworkMatrix.jl) or delegate to the VirtualFactorCore equivalent
    # (virtual_factor_core.jl): each must resolve the removed representative through
    # reverse_bus_search_map instead of throwing.
    #
    # LODF, VirtualLODF, and ArcAdmittanceMatrix are excluded here: they already throw
    # (MethodError / FieldError, not KeyError) on ANY get_ref_bus_position call, independent of
    # ZIBR, because they are arc-indexed / lack a subnetwork_axes field -- a separate,
    # pre-existing defect out of scope for this fix (see comments at their definitions).
    sys = _mk_zibr_merged_representative_system()
    ybus = Ybus(sys)

    ba = PNM.BA_Matrix(ybus)
    @test PNM.get_bus_axis(ba)[only(PNM.get_ref_bus_position(ba))] == 10

    inc = IncidenceMatrix(ybus)
    @test PNM.get_bus_axis(inc)[only(PNM.get_ref_bus_position(inc))] == 10

    adj = PNM.AdjacencyMatrix(ybus)
    @test PNM.get_bus_axis(adj)[only(PNM.get_ref_bus_position(adj))] == 10

    ptdf = PTDF(sys)
    @test PNM.get_bus_axis(ptdf)[only(PNM.get_ref_bus_position(ptdf))] == 10

    vptdf = VirtualPTDF(sys)
    @test PNM.get_bus_axis(vptdf)[only(PNM.get_ref_bus_position(vptdf))] == 10
end
