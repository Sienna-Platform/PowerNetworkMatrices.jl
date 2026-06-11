# Build a standalone 4-bus chain (1-2-3-4). Buses 1 and 4 host active-power
# injectors; bus 2 hosts only a reactive SynchronousCondenser; bus 3 is bare.
# This isolates the reactive-only gating decision in the degree-2 reduction.
function _build_reactive_only_degree2_system()
    sys = PSY.System(100.0)
    buses = ACBus[]
    for i in 1:4
        if i == 1
            bustype = ACBusTypes.REF
        else
            bustype = ACBusTypes.PV
        end
        bus = ACBus(;
            number = i,
            name = "Bus $i",
            available = true,
            bustype = bustype,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.05),
            base_voltage = 138.0,
        )
        add_component!(sys, bus)
        push!(buses, bus)
    end
    for (i, (b_from, b_to)) in enumerate([(1, 2), (2, 3), (3, 4)])
        line = Line(;
            name = "Line $i",
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = Arc(; from = buses[b_from], to = buses[b_to]),
            r = 0.01,
            x = 0.1,
            b = (from = 0.0, to = 0.0),
            rating = 2.0,
            angle_limits = (min = -1.5, max = 1.5),
        )
        add_component!(sys, line)
    end
    for (bus, name) in [(buses[1], "Gen 1"), (buses[4], "Gen 4")]
        gen = ThermalStandard(;
            name = name,
            available = true,
            status = true,
            bus = bus,
            active_power = 0.0,
            reactive_power = 0.0,
            rating = 1.0,
            active_power_limits = (min = 0.0, max = 1.0),
            reactive_power_limits = nothing,
            ramp_limits = (up = 1.0, down = 1.0),
            operation_cost = ThermalGenerationCost(nothing),
            base_power = 100.0,
            time_limits = (up = 1.0, down = 1.0),
            must_run = false,
            prime_mover_type = PrimeMovers.CC,
            fuel = ThermalFuels.NATURAL_GAS,
        )
        add_component!(sys, gen)
    end
    condenser = SynchronousCondenser(;
        name = "Cond 2",
        available = true,
        bus = buses[2],
        reactive_power = 0.0,
        rating = 1.0,
        reactive_power_limits = (min = -1.0, max = 1.0),
        base_power = 100.0,
    )
    add_component!(sys, condenser)
    return sys
end

@testset "Degree-2 reactive-injector gating: irreducible-bus set" begin
    sys = _build_reactive_only_degree2_system()

    kept_reduce = PNM._system_derived_irreducible_buses(sys, true)
    @test 1 in kept_reduce
    @test 4 in kept_reduce
    @test !(2 in kept_reduce)  # condenser-only bus dropped when reducing reactive
    @test !(3 in kept_reduce)  # bare bus never kept

    kept_keep = PNM._system_derived_irreducible_buses(sys, false)
    @test 1 in kept_keep
    @test 4 in kept_keep
    @test 2 in kept_keep      # condenser bus retained when keeping reactive
    @test !(3 in kept_keep)
end

@testset "Degree-2 reactive-injector gating: get_reduction wiring" begin
    sys = _build_reactive_only_degree2_system()
    adj = AdjacencyMatrix(sys)

    reduced = PNM.get_reduction(adj, sys, DegreeTwoReduction())  # default reduce = true
    @test !(2 in reduced.irreducible_buses)         # condenser bus is a reduction candidate
    @test 2 in reduced.removed_buses                # and is actually folded out of the chain

    kept = PNM.get_reduction(adj, sys, DegreeTwoReduction(; reduce_reactive_power_injectors = false))
    @test 2 in kept.irreducible_buses               # condenser bus retained
    @test !(2 in kept.removed_buses)                # and not folded
end
