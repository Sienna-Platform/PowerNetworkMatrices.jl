@testset "Equivalent getters for BranchesParallel and BranchesSeries (non-physical parameters)" begin
    # Create test system with parallel branches
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")

    # Create two test lines with known parameters for parallel configuration
    bus1 = first(PSY.get_components(PSY.ACBus, sys))
    bus2 = collect(PSY.get_components(PSY.ACBus, sys))[2]

    # Create test branches with specific values
    line1 = PSY.Line(;
        name = "test_line_1",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = bus1, to = bus2),
        r = 0.1,  # resistance
        x = 0.2,  # reactance
        b = (from = 0.01, to = 0.01),  # susceptance
        g = (from = 0.01, to = 0.01),  # conductance
        rating = 100.0,  # rating
        angle_limits = (min = -π / 2, max = π / 2),
    )

    line2 = PSY.Line(;
        name = "test_line_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = bus1, to = bus2),
        r = 0.2,  # resistance
        x = 0.4,  # reactance
        b = (from = 0.02, to = 0.02),  # susceptance
        g = (from = 0.02, to = 0.02),  # conductance
        rating = 150.0,  # rating
        angle_limits = (min = -π / 2, max = π / 2),
    )
    # Create BranchesParallel
    bp = PNM.BranchesParallel([line1, line2])
    # Test get_rating: Rating = (Rating1 + Rating2) / n = (100.0 + 150.0) / 2 = 125.0
    rating_eq = PNM.get_equivalent_rating(bp)
    @test rating_eq ≈ 125.0 atol = 1e-6

    emergency_rating_eq = PNM.get_equivalent_emergency_rating(bp)
    @test emergency_rating_eq ≈ 250.0 atol = 1e-6

    bs = PNM.BranchesSeries()
    PNM.add_branch!(bs, line1, :FromTo)
    PNM.add_branch!(bs, line2, :FromTo)
    # Test get_rating: Rating = minimum rating for series branches (weakest link)
    rating_eq = PNM.get_equivalent_rating(bs)
    @test rating_eq ≈ 100.0 atol = 1e-6

    emergency_rating_eq = PNM.get_equivalent_emergency_rating(bs)
    @test emergency_rating_eq ≈ 100.0 atol = 1e-6

    #Add test parrallel circuit + line1
    bs = PNM.BranchesSeries()
    PNM.add_branch!(bs, bp, :FromTo)
    PNM.add_branch!(bs, line2, :FromTo)
    # Test get_rating: Rating = minimum rating for series branches (weakest link)
    rating_eq = PNM.get_equivalent_rating(bs)
    @test rating_eq ≈ 125.0 atol = 1e-6

    emergency_rating_eq = PNM.get_equivalent_emergency_rating(bs)
    @test emergency_rating_eq ≈ 150.0 atol = 1e-6

    # Test get_available: all branches must be available
    @test PSY.get_available(bp) == true
    @test PSY.get_available(bs) == true
end

@testset "Equivalent getters for ThreeWindingTransformerWinding" begin
    # Create a test system with three-winding transformers
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")

    # Get a three-winding transformer from the system
    trf = first(collect(PSY.get_components(PSY.ThreeWindingTransformer, sys)))

    rating3 = PNM.get_equivalent_rating(PNM.ThreeWindingTransformerWinding(trf, 3))
    # Should return winding-specific rating if non-zero, else transformer rating
    expected_rating3 =
        trf.rating_tertiary == 0.0 ? PSY.get_rating(trf) : trf.rating_tertiary
    @test rating3 == expected_rating3

    set_available_secondary!(trf, false)
    @test PNM.get_equivalent_available(PNM.ThreeWindingTransformerWinding(trf, 3)) == true
    @test PNM.get_equivalent_available(PNM.ThreeWindingTransformerWinding(trf, 2)) == false
end

function test_ybus_equivalence_branches_parallel(vector_branches)
    sys = System(100.0)
    bus1 = ACBus(;
        number = 1,
        name = "bus1",
        available = true,
        bustype = ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.0, max = 1.0),
        base_voltage = 1.0,
        area = nothing,
        load_zone = nothing)
    bus2 = ACBus(;
        number = 2,
        name = "bus2",
        available = true,
        bustype = ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.0, max = 1.0),
        base_voltage = 1.0,
        area = nothing,
        load_zone = nothing)

    add_component!(sys, bus1)
    add_component!(sys, bus2)
    for br in vector_branches
        br_copy = deepcopy(br)
        set_arc!(br_copy, Arc(; from = bus1, to = bus2))
        add_component!(sys, br_copy)
    end
    ybus = Ybus(sys)
    branches_parallel = ybus.network_reduction_data.parallel_branch_map[(1, 2)]
    sys_equivalent = deepcopy(sys)
    for l in get_components(ACTransmission, sys_equivalent)
        remove_component!(sys_equivalent, l)
    end
    bus1 = get_component(ACBus, sys_equivalent, "bus1")
    bus2 = get_component(ACBus, sys_equivalent, "bus2")
    equivalent_pbranch = PNM.get_equivalent_physical_branch_parameters(branches_parallel)
    if PNM.get_equivalent_shift(equivalent_pbranch) == 0.0
        equivalent_branch = PSY.Line(;
            name = "equivalent_line",
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = PSY.Arc(; from = bus1, to = bus2),
            r = PNM.get_equivalent_r(equivalent_pbranch),  # resistance
            x = PNM.get_equivalent_x(equivalent_pbranch),   # reactance
            b = (
                from = PNM.get_equivalent_b_from(equivalent_pbranch),
                to = PNM.get_equivalent_b_to(equivalent_pbranch),
            ),  # susceptance
            g = (
                from = PNM.get_equivalent_g_from(equivalent_pbranch),
                to = PNM.get_equivalent_g_to(equivalent_pbranch),
            ),  # conductance
            rating = 80.0,  # rating
            angle_limits = (min = -π / 2, max = π / 2),
        )
        add_component!(sys_equivalent, equivalent_branch)
    else
        equivalent_transformer = PSY.PhaseShiftingTransformer(;
            name = "equivalent_transformer",
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = PSY.Arc(; from = bus1, to = bus2),
            r = PNM.get_equivalent_r(equivalent_pbranch),  # resistance
            x = PNM.get_equivalent_x(equivalent_pbranch),   # reactance
            primary_shunt = Complex(
                PNM.get_equivalent_g_from(equivalent_pbranch),
                PNM.get_equivalent_b_from(equivalent_pbranch),
            ),
            tap = PNM.get_equivalent_tap(equivalent_pbranch),
            α = PNM.get_equivalent_shift(equivalent_pbranch),
            rating = 80.0,  # rating
            base_power = 100.0,
        )
        equivalent_admittance = PSY.FixedAdmittance(;
            name = "equivalent_admittance",
            available = true,
            bus = bus2,
            Y = Complex(
                PNM.get_equivalent_g_to(equivalent_pbranch),
                PNM.get_equivalent_b_to(equivalent_pbranch),
            ),
        )
        add_component!(sys_equivalent, equivalent_transformer)
        add_component!(sys_equivalent, equivalent_admittance)
    end
    ybus_equivalent = Ybus(sys_equivalent)
    #display(Matrix(ybus.data)) - for debug
    #display(Matrix(ybus_equivalent.data)) - for debug
    @test all(isapprox.(ybus.data, ybus_equivalent.data; atol = 1e-5))
end

function test_ybus_equivalence_branches_series(vector_branches)
    sys = System(100.0)
    n_buses = length(vector_branches) + 1
    for bus_ix in 1:n_buses
        bus = ACBus(;
            number = bus_ix,
            name = "bus$(bus_ix)",
            available = true,
            bustype = ACBusTypes.PQ,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.0, max = 1.0),
            base_voltage = 1.0,
            area = nothing,
            load_zone = nothing)
        add_component!(sys, bus)
    end
    for (ix, br) in enumerate(vector_branches)
        br_copy = deepcopy(br)
        set_arc!(
            br_copy,
            Arc(;
                from = get_component(ACBus, sys, "bus$(ix)"),
                to = get_component(ACBus, sys, "bus$(ix+1)"),
            ),
        )
        add_component!(sys, br_copy)
    end
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    branches_series = ybus.network_reduction_data.series_branch_map[(1, n_buses)]
    sys_equivalent = deepcopy(sys)
    for l in get_components(ACTransmission, sys_equivalent)
        remove_component!(sys_equivalent, l)
    end
    for bus in get_components(ACBus, sys_equivalent)
        bus.number ∈ [1, 2] && continue
        remove_component!(sys_equivalent, bus)
    end
    bus1 = get_component(ACBus, sys_equivalent, "bus1")
    bus2 = get_component(ACBus, sys_equivalent, "bus2")
    equivalent_pbranch = PNM.get_equivalent_physical_branch_parameters(branches_series)
    if PNM.get_equivalent_shift(equivalent_pbranch) == 0.0
        equivalent_branch = PSY.Line(;
            name = "equivalent_line",
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = PSY.Arc(; from = bus1, to = bus2),
            r = PNM.get_equivalent_r(equivalent_pbranch),  # resistance
            x = PNM.get_equivalent_x(equivalent_pbranch),   # reactance
            b = (
                from = PNM.get_equivalent_b_from(equivalent_pbranch),
                to = PNM.get_equivalent_b_to(equivalent_pbranch),
            ),  # susceptance
            g = (
                from = PNM.get_equivalent_g_from(equivalent_pbranch),
                to = PNM.get_equivalent_g_to(equivalent_pbranch),
            ),  # conductance
            rating = 80.0,  # rating
            angle_limits = (min = -π / 2, max = π / 2),
        )
        add_component!(sys_equivalent, equivalent_branch)
    else
        equivalent_transformer = PSY.PhaseShiftingTransformer(;
            name = "equivalent_transformer",
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = PSY.Arc(; from = bus1, to = bus2),
            r = PNM.get_equivalent_r(equivalent_pbranch),  # resistance
            x = PNM.get_equivalent_x(equivalent_pbranch),   # reactance
            primary_shunt = Complex(
                PNM.get_equivalent_g_from(equivalent_pbranch),
                PNM.get_equivalent_b_from(equivalent_pbranch),
            ),
            tap = PNM.get_equivalent_tap(equivalent_pbranch),
            α = PNM.get_equivalent_shift(equivalent_pbranch),
            rating = 80.0,  # rating
            base_power = 100.0,
            #angle_limits = (min = -π / 2, max = π / 2),
        )
        equivalent_admittance = PSY.FixedAdmittance(;
            name = "equivalent_admittance",
            available = true,
            bus = bus2,
            Y = Complex(
                PNM.get_equivalent_g_to(equivalent_pbranch),
                PNM.get_equivalent_b_to(equivalent_pbranch),
            ),
        )
        add_component!(sys_equivalent, equivalent_transformer)
        add_component!(sys_equivalent, equivalent_admittance)
    end
    ybus_equivalent = Ybus(sys_equivalent)
    #display(Matrix(ybus.data)) - for debug
    #display(Matrix(ybus_equivalent.data)) - for debug
    @test all(isapprox.(ybus.data, ybus_equivalent.data; atol = 1e-5))
end
@testset "Ybus correctness for equivalent parameters of BranchesSeries and BranchesParallel" begin
    l1 = PSY.Line(;
        name = "line_1",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(nothing),
        r = 0.05,  # resistance
        x = 0.1,   # reactance
        b = (from = 0.01, to = 0.01),  # susceptance
        g = (from = 0.01, to = 0.01),  # conductance
        rating = 100.0,  # rating
        angle_limits = (min = -π / 2, max = π / 2),
    )
    l2 = PSY.Line(;
        name = "line_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(nothing),
        r = 0.15,  # resistance
        x = 0.3,   # reactance
        b = (from = 0.03, to = 0.02),  # susceptance
        g = (from = 0.03, to = 0.02),  # conductance
        rating = 80.0,  # rating
        angle_limits = (min = -π / 2, max = π / 2),
    )
    l3 = PSY.Line(;
        name = "line_3",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(nothing),
        r = 0.122,  # resistance
        x = 0.1,   # reactance
        b = (from = 0.01, to = 0.02),  # susceptance
        g = (from = 0.035, to = 0.015),  # conductance
        rating = 80.0,  # rating
        angle_limits = (min = -π / 2, max = π / 2),
    )
    t1 = PSY.TapTransformer(;
        name = "tfw_1",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(nothing),
        r = 0.122,  # resistance
        x = 0.1,   # reactance
        primary_shunt = 0.01 + im * 0.02,
        tap = 1.0,
        rating = 80.0,  # rating
        base_power = 100.0,
        winding_group_number = WindingGroupNumber.GROUP_11,
    )
    t2 = PSY.TapTransformer(;
        name = "tfw_2",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(nothing),
        r = 0.3,  # resistance
        x = 0.13,   # reactance
        primary_shunt = 0.02 + im * 0.021,
        tap = 1.0,
        rating = 80.0,  # rating
        base_power = 100.0,
        winding_group_number = WindingGroupNumber.GROUP_11,
    )
    t3 = PSY.PhaseShiftingTransformer(;
        name = "tfw_3",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(nothing),
        r = 0.3,  # resistance
        x = 0.13,   # reactance
        primary_shunt = 0.02 + im * 0.021,
        tap = 1.0,
        α = 0.2,
        rating = 80.0,  # rating
        base_power = 100.0,
    )
    # Two lines in parallel:
    test_ybus_equivalence_branches_parallel([l1, l2])
    # Two lines in series:
    test_ybus_equivalence_branches_series([l1, l2])
    # Three lines in parallel:
    test_ybus_equivalence_branches_parallel([l1, l2, l3])
    # Three lines in series:
    test_ybus_equivalence_branches_series([l1, l2, l3])
    # Two transformers in parallel with the same phase angle (winding group):
    test_ybus_equivalence_branches_parallel([t1, t2])
    # Two transformers in series with the same phase angle (winding group):
    test_ybus_equivalence_branches_series([t1, t2])
    # Two transformers in series with different phase angle
    test_ybus_equivalence_branches_series([t1, t3])
end

@testset "Compute equivalent physical parameters for WECC 240 bus" begin
    sys = PSB.build_system(PSYTestSystems, "psse_240_parsing_sys"; runchecks = false)
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    for branches_parallel in values(ybus.network_reduction_data.parallel_branch_map)
        @test isa(
            PNM.get_equivalent_physical_branch_parameters(branches_parallel),
            PNM.EquivalentBranch,
        )
    end
    for branches_series in values(ybus.network_reduction_data.series_branch_map)
        @test isa(
            PNM.get_equivalent_physical_branch_parameters(branches_series),
            PNM.EquivalentBranch,
        )
    end
end
