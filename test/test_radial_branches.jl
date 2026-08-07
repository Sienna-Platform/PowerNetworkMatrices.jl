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
    # This system has no three-winding transformers, so no winding wrappers are in the
    # direct maps.
    @test !any(v -> v isa PNM.ThreeWindingTransformerCircuit, values(rb.direct_branch_map))
    @test length(rb.removed_buses) == 0
    @test rb.removed_arcs == Set([(7, 8), (8, 61)])
    @test get_reductions(rb) == PNM.ReductionContainer(;
        radial_reduction = RadialReduction(),
        zero_impedance_reduction = PNM.ZeroImpedanceBranchReduction(),
    )
end

@testset "Radial Branches Large" begin
    sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg10k_sys")
    Y = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    rb = get_network_reduction_data(Y)
    for (k, v) in get_bus_reduction_map(rb)
        @test k ∉ v
    end
end

@testset "Check reference bus in Radial Branches" begin
    for name in ["matpower_ACTIVSg2000_sys", "matpower_ACTIVSg10k_sys"]
        sys = build_system(MatpowerTestSystems, name)
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
    ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
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

@testset "Radial reduction cascades through a bus whose live degree drops to one" begin
    # Meshed core 1-2-3 (bus 1 = reference) with hub bus 4 off bus 2 carrying leaves 5
    # and 6. Peeling both leaves makes bus 4 radial in turn.
    arcs = [(1, 2), (2, 3), (3, 1), (2, 4), (4, 5), (4, 6)]
    I = Int[]
    J = Int[]
    V = Int8[]
    for (r, (f, t)) in enumerate(arcs)
        push!(I, r, r)
        push!(J, f, t)
        push!(V, Int8(1), Int8(-1))
    end
    A = SparseArrays.dropzeros!(SparseArrays.sparse(I, J, V, length(arcs), 6))
    arc_map = Dict(a => r for (r, a) in enumerate(arcs))
    bus_map = Dict(i => i for i in 1:6)

    bus_reduction_map, reverse_map, radial_arcs, final_arc_map =
        PNM.calculate_radial_arcs(A, arc_map, bus_map, Set([1]))

    @test radial_arcs == Set([(4, 5), (4, 6), (2, 4)])
    @test bus_reduction_map[2] == Set([4, 5, 6])
    @test reverse_map[4] == reverse_map[5] == reverse_map[6] == 2
    # Arcs (4, 5) and (4, 6) lose both endpoints, so they carry no diagonal correction.
    @test final_arc_map == Dict((2, 4) => 2)
end

@testset "Cascaded radial reduction preserves PTDF flows" begin
    # Hub bus off well-connected Bus 4 carrying two leaves: peeling both makes the hub
    # radial, so all three buses collapse into Bus 4 and the leaf load aggregates up two
    # levels.
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    anchor = get_component(ACBus, sys, "Bus 4")
    for (number, name) in [(60, "Hub"), (61, "Leaf1"), (62, "Leaf2")]
        bus = deepcopy(anchor)
        bus.internal = PowerSystems.IS.InfrastructureSystemsInternal()
        set_name!(bus, name)
        set_number!(bus, number)
        add_component!(sys, bus)
    end
    for (from, to, name) in [
        ("Bus 4", "Hub", "hub_tie"),
        ("Hub", "Leaf1", "stub1"),
        ("Hub", "Leaf2", "stub2"),
    ]
        arc = Arc(get_component(ACBus, sys, from), get_component(ACBus, sys, to))
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(name, true, 0.0, 0.0, arc, 0.0, 0.05, (from = 0.0, to = 0.0), 100.0,
                (0.0, 0.0)),
        )
    end
    add_component!(
        sys,
        PowerLoad(;
            name = "Leaf1_load",
            available = true,
            bus = get_component(ACBus, sys, "Leaf1"),
            active_power = 0.25,
            reactive_power = 0.0,
            base_power = 100.0,
            max_active_power = 0.25,
            max_reactive_power = 0.0,
        ),
    )

    reductions = NetworkReduction[RadialReduction()]
    rb = get_network_reduction_data(Ybus(sys; network_reductions = reductions))
    # Guard against the fixture silently ceasing to cascade.
    @test Set([60, 61, 62]) ⊆ rb.bus_reduction_map[4]
    @test rb.reverse_bus_search_map[60] == 4
    @test Set([(4, 60), (60, 61), (60, 62)]) ⊆ rb.removed_arcs

    ptdf = PTDF(sys)
    ptdf_rad = PTDF(sys; network_reductions = reductions)

    injection = zeros(Float64, size(ptdf.data, 1))
    for source in get_components(
        d -> !isa(d, Union{PSY.ElectricLoad, PSY.SynchronousCondenser}),
        PSY.StaticInjection,
        sys,
    )
        PSY.get_available(source) || continue
        injection[ptdf.lookup[1][PSY.get_number(PSY.get_bus(source))]] +=
            PSY.get_active_power(source, PSY.SU)
    end
    for load in get_components(x -> !isa(x, PSY.FixedAdmittance), PSY.ElectricLoad, sys)
        PSY.get_available(load) || continue
        injection[ptdf.lookup[1][PSY.get_number(PSY.get_bus(load))]] -=
            PSY.get_active_power(load, PSY.SU)
    end

    reduced_injection = deepcopy(injection)
    for (parent, children) in rb.bus_reduction_map, child in children
        reduced_injection[ptdf.lookup[1][parent]] += injection[ptdf.lookup[1][child]]
    end
    removed_buses = reduce(union!, values(rb.bus_reduction_map); init = Set{Int}())
    bus_idx = setdiff(
        1:size(ptdf.data, 1),
        [ptdf.lookup[1][b] for b in removed_buses],
    )
    br_idx = setdiff(1:size(ptdf.data, 2), [ptdf.lookup[2][a] for a in rb.removed_arcs])

    @test isapprox(
        (transpose(ptdf.data) * injection)[br_idx],
        transpose(ptdf_rad.data) * reduced_injection[bus_idx],
    )
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
