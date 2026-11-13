const IRREDUCIBLE_SELECTOR_SYSTEM = "psse_14_network_reduction_test_system"

_irreducible_selector_system() =
    PSB.build_system(PSSEParsingTestSystems, IRREDUCIBLE_SELECTOR_SYSTEM)

function _reduction_summary(ybus::Ybus)
    nrd = get_network_reduction_data(ybus)
    return (
        irreducible = nrd.irreducible_buses,
        removed_buses = nrd.removed_buses,
        removed_arcs = nrd.removed_arcs,
        bus_reduction_map = nrd.bus_reduction_map,
    )
end

@testset "get_irreducible_bus_numbers: bus rules per component type" begin
    sys = _irreducible_selector_system()
    get_buses(selector) = get_irreducible_bus_numbers(sys, selector)
    vsc_name = PSY.get_name(only(PSY.get_components(TwoTerminalVSCLine, sys)))

    @test get_buses(make_selector(ACBus, "BUS 116_116")) == Set([116])
    @test get_buses(make_selector(Line, "BUS 103_103-BUS 116_116-i_1")) == Set([103, 116])
    @test get_buses(make_selector(TwoWindingTransformer, "BUS 104_104-BUS 109_109-i_1")) ==
          Set([104, 109])
    @test get_buses(
        make_selector(ThreeWindingTransformer, "BUS 113_113-BUS 110_110-BUS 114_114-i_1"),
    ) == Set([113, 110, 114, 1002])
    @test get_buses(make_selector(TwoTerminalVSCLine, vsc_name)) ==
          Set([112, 106])
    @test get_buses(make_selector(ThermalStandard, "generator-108-1")) == Set([108])
    @test isempty(get_buses(make_selector()))
end

@testset "get_irreducible_bus_numbers: filters and unions" begin
    sys = _irreducible_selector_system()
    from_101(line) = PSY.get_number(PSY.get_from(PSY.get_arc(line))) == 101
    @test get_irreducible_bus_numbers(sys, make_selector(from_101, Line)) ==
          Set([101, 115, 117])

    union_selector = make_selector(
        make_selector(ACBus, "BUS 116_116"),
        make_selector(Line, "BUS 116_117-BUS 116_118-i_1"),
    )
    @test get_irreducible_bus_numbers(sys, union_selector) == Set([116, 117, 118])
end

@testset "get_irreducible_bus_numbers: unavailable components give no buses" begin
    sys = _irreducible_selector_system()
    line = PSY.get_component(Line, sys, "BUS 103_103-BUS 116_116-i_1")
    PSY.set_available!(line, false)
    @test isempty(get_irreducible_bus_numbers(sys, make_selector(Line, PSY.get_name(line))))
    generator = PSY.get_component(ThermalStandard, sys, "generator-108-1")
    PSY.set_available!(generator, false)
    @test isempty(
        get_irreducible_bus_numbers(sys, make_selector(ThermalStandard, "generator-108-1")),
    )
end

@testset "get_irreducible_bus_numbers: unsupported component type errors" begin
    sys = _irreducible_selector_system()
    interface = PSY.TransmissionInterface(nothing)
    PSY.set_name!(interface, "interface")
    PSY.set_available!(interface, true)
    PSY.add_component!(sys, interface)
    @test_throws ArgumentError get_irreducible_bus_numbers(
        sys,
        make_selector(TransmissionInterface, "interface"),
    )
    @test_throws ArgumentError Ybus(
        sys;
        irreducible_buses = make_selector(TransmissionInterface, "interface"),
    )
end

@testset "Ybus: a component selector gives the same reduction as a bus selector" begin
    sys = _irreducible_selector_system()

    # Radial reduction removes bus 116 unless the branch to it is selected.
    radial = NetworkReduction[RadialReduction()]
    from_selector = Ybus(
        sys;
        network_reductions = radial,
        irreducible_buses = make_selector(Line, "BUS 103_103-BUS 116_116-i_1"),
    )
    from_numbers = Ybus(
        sys;
        network_reductions = radial,
        irreducible_buses = bus_number_selector([103, 116]),
    )
    @test !haskey(get_network_reduction_data(from_selector).reverse_bus_search_map, 116)
    @test _reduction_summary(from_selector) == _reduction_summary(from_numbers)

    # Degree-two reduction removes bus 117 unless the bus is selected.
    degree_two = NetworkReduction[DegreeTwoReduction()]
    from_selector = Ybus(
        sys;
        network_reductions = degree_two,
        irreducible_buses = make_selector(ACBus, "BUS 116_117"),
    )
    from_numbers =
        Ybus(
            sys;
            network_reductions = degree_two,
            irreducible_buses = bus_number_selector(Set([117])),
        )
    @test 117 ∉ get_network_reduction_data(from_selector).removed_buses
    @test _reduction_summary(from_selector) == _reduction_summary(from_numbers)
end

@testset "Constructors that forward to Ybus accept a selector" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    selector = make_selector(ACBus, PSY.get_name(PSY.get_bus(sys, 8)))
    reductions = NetworkReduction[RadialReduction()]
    ptdf_selector = PTDF(sys; network_reductions = reductions, irreducible_buses = selector)
    ptdf_numbers = PTDF(
        sys;
        network_reductions = reductions,
        irreducible_buses = bus_number_selector([8]),
    )
    @test ptdf_selector.axes == ptdf_numbers.axes
    @test isapprox(ptdf_selector.data, ptdf_numbers.data)

    modf_selector =
        VirtualMODF(sys; network_reductions = reductions, irreducible_buses = selector)
    @test 8 ∈ get_network_reduction_data(modf_selector).irreducible_buses
end

@testset "irreducible_buses rejects bus numbers" begin
    sys = PSB.build_system(PSITestSystems, "c_sys14")
    @test_throws TypeError Ybus(sys; irreducible_buses = [8])
    @test_throws TypeError VirtualMODF(sys; irreducible_buses = Set([8]))
end
