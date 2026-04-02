function _basic_test_ward_reduction(sys, study_buses)
    ybus = Ybus(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    wr = get_network_reduction_data(ybus)
    @test isa(wr, NetworkReductionData)
    @test get_reductions(wr) ==
          PNM.ReductionContainer(; ward_reduction = WardReduction(study_buses))
    external_buses =
        setdiff([get_number(x) for x in get_components(ACBus, sys)], study_buses)
    @test !isempty(wr.added_admittance_map)
    @test !isempty(wr.added_arc_impedance_map)
    for external_bus in external_buses
        @test external_bus ∉ keys(wr.bus_reduction_map)
        @test external_bus ∈ keys(wr.reverse_bus_search_map)
    end
    # Validate boundary_bus_to_removed_arcs: for each boundary bus, its arc set should
    # match the set of arcs in the original system that connect it to external buses
    A_full = IncidenceMatrix(sys)
    study_buses_set = Set(study_buses)
    expected_bb_to_arcs = Dict{Int, Set{Tuple{Int, Int}}}()
    for arc in PNM.get_arc_axis(A_full)
        if arc[1] ∈ study_buses_set && arc[2] ∉ study_buses_set
            set = get!(expected_bb_to_arcs, arc[1], Set{Tuple{Int, Int}}())
            push!(set, arc)
        elseif arc[2] ∈ study_buses_set && arc[1] ∉ study_buses_set
            set = get!(expected_bb_to_arcs, arc[2], Set{Tuple{Int, Int}}())
            push!(set, arc)
        end
    end
    @test wr.boundary_bus_to_removed_arcs == expected_bb_to_arcs
end

function _check_for_repeated_arcs(matrix)
    nrd = get_network_reduction_data(matrix)
    arc_axis = PNM.get_arc_axis(matrix)
    for arc_tuple in keys(nrd.added_arc_impedance_map)
        @test arc_tuple ∈ arc_axis
        @test (arc_tuple[2], arc_tuple[1]) ∉ arc_axis
    end
    return
end

function _test_matrices_ward_reduction(sys, study_buses)
    ybus = Ybus(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    wr = get_network_reduction_data(ybus)
    added_branch_arcs = [x for x in keys(wr.added_arc_impedance_map)]
    direct_arcs = [x for x in keys(wr.direct_branch_map)]
    parallel_arcs = [x for x in keys(wr.parallel_branch_map)]
    expected_arc_axis = union(Set(added_branch_arcs), Set(direct_arcs), Set(parallel_arcs))
    A = IncidenceMatrix(
        sys;
        network_reductions = NetworkReduction[WardReduction(study_buses)],
    )
    _check_for_repeated_arcs(A)
    @test Set(A.axes[1]) == expected_arc_axis
    @test Set(A.axes[2]) == Set(study_buses)

    Adj = AdjacencyMatrix(
        sys;
        network_reductions = NetworkReduction[WardReduction(study_buses)],
    )
    @test Set(Adj.axes[1]) == Set(study_buses)
    @test Set(Adj.axes[2]) == Set(study_buses)

    BA = BA_Matrix(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    _check_for_repeated_arcs(BA)
    @test Set(BA.axes[2]) == expected_arc_axis
    @test Set(BA.axes[1]) == Set(study_buses)

    Y = Ybus(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    @test Set(Y.axes[1]) == Set(study_buses)
    @test Set(Y.axes[2]) == Set(study_buses)

    PTDF_ = PTDF(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    _check_for_repeated_arcs(PTDF_)
    @test Set(PNM.get_bus_axis(PTDF_)) == Set(study_buses)
    @test Set(PNM.get_arc_axis(PTDF_)) == expected_arc_axis

    LODF_ = LODF(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    _check_for_repeated_arcs(LODF_)
    @test Set(LODF_.axes[1]) == expected_arc_axis
    @test Set(LODF_.axes[2]) == expected_arc_axis

    vPTDF_ =
        VirtualPTDF(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    _check_for_repeated_arcs(vPTDF_)
    @test Set(PNM.get_bus_axis(vPTDF_)) == Set(study_buses)
    @test Set(PNM.get_arc_axis(vPTDF_)) == expected_arc_axis

    vLODF_ =
        VirtualLODF(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    _check_for_repeated_arcs(vLODF_)
    @test Set(vLODF_.axes[1]) == expected_arc_axis
    @test Set(vLODF_.axes[2]) == expected_arc_axis
end

@testset "Basic ward reduction" begin
    sys = PSB.build_system(PSB.PSIDSystems, "3 Bus Inverter Base")
    study_buses = [101, 102]
    _basic_test_ward_reduction(sys, study_buses)
    _test_matrices_ward_reduction(sys, study_buses)

    sys = PSB.build_system(PSB.PSIDTestSystems, "psid_test_ieee_9bus")
    study_buses = [1, 2, 5, 4, 7]
    _basic_test_ward_reduction(sys, study_buses)
    _test_matrices_ward_reduction(sys, study_buses)

    sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    bus_numbers = [get_number(x) for x in get_components(ACBus, sys)]
    study_buses = filter!(x -> digits(x)[end] == 1, bus_numbers)  #study buses are from area 1
    _basic_test_ward_reduction(sys, study_buses)
    _test_matrices_ward_reduction(sys, study_buses)

    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    study_buses = [101, 114, 110, 111]
    _basic_test_ward_reduction(sys, study_buses)
    _test_matrices_ward_reduction(sys, study_buses)
end

@testset "Test similarity of PTDF with Ward" begin
    sys = PSB.build_system(PSB.PSIDSystems, "3 Bus Inverter Base")
    ptdf_3 = PTDF(sys)
    ptdf_2 = @test_logs (
        :warn,
        r"Equivalent arc impedance computed during Ward reduction is in parallel with existing system arc.",
    ) match_mode = :any PTDF(
        sys;
        network_reductions = NetworkReduction[WardReduction([101, 102])],
    )
    existing_line_susceptance = PSY.get_series_susceptance(
        ptdf_2.network_reduction_data.direct_branch_map[(101, 102)],
    )
    ward_line_susceptance = PSY.get_series_susceptance(
        ptdf_2.network_reduction_data.added_arc_impedance_map[(101, 102)],
    )
    ward_multiplier =
        existing_line_susceptance / (existing_line_susceptance + ward_line_susceptance)
    @test abs(ptdf_3[(101, 102), 102] - ward_multiplier * ptdf_2[(101, 102), 102]) < 0.007
end

@testset "Ward corner cases" begin
    sys = build_hvdc_with_small_island()
    wr = get_network_reduction_data(
        Ybus(sys; network_reductions = NetworkReduction[WardReduction([1, 2, 3])]),
    )

    @test isa(wr, NetworkReductionData)
    @test length(wr.bus_reduction_map) == 3

    wr = get_network_reduction_data(
        Ybus(sys; network_reductions = NetworkReduction[WardReduction([1])]),
    )
    @test isa(wr, NetworkReductionData)
    @test length(wr.bus_reduction_map) == 1
    @test length(wr.added_arc_impedance_map) == 0
    @test length(wr.added_admittance_map) == 1

    wr =
        @test_logs (:error, r"no boundary buses found") match_mode = :any get_network_reduction_data(
            Ybus(sys; network_reductions = NetworkReduction[WardReduction([15, 16, 17])]),
        )
    @test isa(wr, NetworkReductionData)
    @test length(wr.added_arc_impedance_map) == 0
    @test length(wr.added_admittance_map) == 0

    @test_throws IS.DataFormatError get_network_reduction_data(
        Ybus(
            sys;
            network_reductions = NetworkReduction[WardReduction([1, 2, 3, 4, 5, 17])],
        ),
    )
    @test_throws IS.DataFormatError get_network_reduction_data(
        Ybus(
            sys;
            network_reductions = NetworkReduction[WardReduction([1, 2, 3, 4, 5, 100])],
        ),
    )
    @test_throws IS.DataFormatError get_network_reduction_data(
        Ybus(sys; network_reductions = NetworkReduction[WardReduction([2, 3, 4])]),
    )
end

@testset "WardReduction with isolated buses" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys_with_isolated = deepcopy(sys)
    bus6 = ACBus(;
        number = 6,
        name = "Bus 6",
        available = true,
        bustype = ACBusTypes.REF,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.05),
        base_voltage = 69.0,
    )
    add_component!(sys_with_isolated, bus6)
    bus5 = get_component(ACBus, sys_with_isolated, "nodeD")
    hvdc1 = TwoTerminalHVDCLine(;
        name = "Line18",
        available = true,
        active_power_flow = 0.0,
        arc = Arc(; from = bus5, to = bus6),
        active_power_limits_from = (min = -100.0, max = 100.0),
        active_power_limits_to = (min = -100.0, max = 100.0),
        reactive_power_limits_from = (min = -100.0, max = 100.0),
        reactive_power_limits_to = (min = -100.0, max = 100.0),
    )
    add_component!(sys_with_isolated, hvdc1)
    ybus = Ybus(sys; network_reductions = NetworkReduction[WardReduction([1, 2, 3, 4])])
    ybus_with_isolated = Ybus(
        sys_with_isolated;
        network_reductions = NetworkReduction[WardReduction([1, 2, 3, 4])],
    )
    @test ybus.data == ybus_with_isolated.data
    # Building PTDF tests handling of subnetwork_axes when an entire subnetwork is eliminated during reduction:
    @test isa(PTDF(ybus_with_isolated), PTDF)
end
