@testset "Radial Branches" begin
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    n = first(get_components(ACBus, sys))
    n2 = deepcopy(n)
    n2.internal = PowerSystems.IS.InfrastructureSystemsInternal()
    set_name!(n2, "TestBus")
    set_number!(n2, 61)
    set_base_voltage!(n2, 18.0)
    add_component!(sys, n2)
    arc = Arc(get_component(ACBus, sys, "Bus 8"), n2)
    add_component!(sys, arc)
    add_component!(
        sys,
        Line(
            "tl",
            true,
            0.0,
            0.0,
            arc,
            0.0,
            0.01,    #cannot have zero impedance line
            (from = 0.0, to = 0.0),
            100.0,
            (0.0, 0.0),
        ),
    )
    Y = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    rb = get_network_reduction_data(Y)
    @test rb.bus_reduction_map[7] == Set([61, 8])
    @test rb.reverse_bus_search_map[61] == rb.reverse_bus_search_map[8] == 7
    @test length(rb.direct_branch_map) == 19
    @test length(rb.reverse_direct_branch_map) == 19
    @test length(rb.parallel_branch_map) == 0
    @test length(rb.reverse_parallel_branch_map) == 0
    @test length(rb.series_branch_map) == 0
    @test length(rb.reverse_series_branch_map) == 0
    @test length(rb.transformer3W_map) == 0
    @test length(rb.reverse_transformer3W_map) == 0
    @test length(rb.removed_buses) == 0
    @test rb.removed_arcs == Set([(7, 8), (8, 61)])
    @test get_reductions(rb) ==
          PNM.ReductionContainer(;
        radial_reduction = RadialReduction(),
        zero_impedance_reduction = PNM.ZeroImpedanceBranchReduction(),
    )
end

@testset "Radial Branches Large" begin
    sys = build_system(
        MatpowerTestSystems,
        "matpower_ACTIVSg10k_sys",
    )
    Y = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    rb = get_network_reduction_data(Y)
    for (k, v) in get_bus_reduction_map(rb)
        @test k ∉ v
    end
end

@testset "Check reference bus in Radial Branches" begin
    for name in ["matpower_ACTIVSg2000_sys", "matpower_ACTIVSg10k_sys"]
        sys = build_system(
            MatpowerTestSystems,
            name,
        )
        a_mat = IncidenceMatrix(sys)
        Y = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
        rb = get_network_reduction_data(Y)
        leaf_buses = Int[]
        for i in keys(rb.bus_reduction_map)
            append!(leaf_buses, collect(rb.bus_reduction_map[i]))
        end
        leaf_positions = [a_mat.lookup[2][x] for x in leaf_buses]
        @test all(PNM.get_ref_bus_position(a_mat) .∉ leaf_positions)
    end
end

@testset "Small island exception for radial reduction" begin
    sys = build_hvdc_with_small_island()
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
    )
    rr = get_network_reduction_data(ybus)
    @test haskey(rr.reverse_bus_search_map, 16)
    @test haskey(rr.reverse_bus_search_map, 17)
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set([16, 17]),
    )
    rr = get_network_reduction_data(ybus)
    @test !haskey(rr.reverse_bus_search_map, 16)
    @test !haskey(rr.reverse_bus_search_map, 17)
end

@testset "calculate_radial_arcs tolerates a self-loop arc (all-zero incidence row)" begin
    # Regression: the MMWG case contains a self-loop branch (from == to). Its
    # incidence row cancels to all zeros, so `_build_row_to_cols` never assigns
    # `row_to_cols[row]`; the adjacency builder then read garbage and indexed
    # `adj[0]` -> BoundsError. Model a 1-2-3-4 chain (bus 1 = reference) plus a
    # self-loop on bus 2.
    I = Int[]
    J = Int[]
    V = Int8[]
    for (r, (f, t)) in enumerate([(1, 2), (2, 3), (3, 4)])
        push!(I, r, r)
        push!(J, f, t)
        push!(V, Int8(1), Int8(-1))
    end
    # self-loop arc on bus 2 (row 4): +1 and -1 in the same column cancel to 0
    push!(I, 4, 4)
    push!(J, 2, 2)
    push!(V, Int8(1), Int8(-1))
    A = SparseArrays.dropzeros!(SparseArrays.sparse(I, J, V, 4, 4))
    arc_map = Dict((1, 2) => 1, (2, 3) => 2, (3, 4) => 3, (2, 2) => 4)
    bus_map = Dict(1 => 1, 2 => 2, 3 => 3, 4 => 4)
    ref_bus_positions = Set([1])

    bus_reduction_map, reverse_map, radial_arcs, final_arc_map =
        PNM.calculate_radial_arcs(A, arc_map, bus_map, ref_bus_positions)

    # The self-loop carries no connectivity and must not be treated as radial.
    @test (2, 2) ∉ radial_arcs
    # The 1-2-3-4 chain collapses toward the reference bus.
    @test (3, 4) in radial_arcs
    @test (2, 3) in radial_arcs
end
