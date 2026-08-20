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
    # Attach the branches so the system-base getters (e.g. the susceptance
    # weighting in get_impedance_averaged_rating) can resolve the system base.
    # These lines carry round, illustrative values, so skip data validation.
    PSY.add_component!(sys, line1; skip_validation = true)
    PSY.add_component!(sys, line2; skip_validation = true)

    # Create BranchesParallel
    bp = PNM.BranchesParallel([line1, line2])

    # Sum of individual thermal limits.
    @test PNM.get_sum_of_max_rating(bp) ≈ 250.0 atol = 1e-6

    # N-1: capacity remaining after the largest-rated circuit trips.
    # sum(100, 150) - max(100, 150) = 100.0
    @test PNM.get_single_element_contingency_rating(bp) ≈ 100.0 atol = 1e-6

    # Susceptance-weighted average.
    # b1 = x1/(r1²+x1²) = 0.2/0.05 = 4.0, b2 = 0.4/0.20 = 2.0, b_total = 6.0
    # f1 = 2/3, f2 = 1/3 → (2/3)*100 + (1/3)*150 = 350/3 ≈ 116.667
    @test PNM.get_impedance_averaged_rating(bp) ≈ 350.0 / 3.0 atol = 1e-6

    emergency_rating_eq = PNM.get_equivalent_emergency_rating(bp)
    @test emergency_rating_eq ≈ 250.0 atol = 1e-6

    # `get_equivalent_rating` is the unexported in-PNM fallback and sums its members: every
    # circuit on the arc carries flow at once. It is not what consumers get by default — a
    # series chain applies the N-1 value to an embedded parallel block
    # (`_series_member_rating`), and POM selects the aggregate per `DeviceModel`, also
    # defaulting to N-1.
    @test PNM.get_equivalent_rating(bp) ≈ 250.0 atol = 1e-6
    @test PNM.get_single_element_contingency_rating(bp) ≈ 100.0 atol = 1e-6

    # Regression: `branch_flow_limits` used to reach a `get_equivalent_rating` with no
    # parallel-group method and raise a MethodError.
    fl_bp = PNM.branch_flow_limits(bp)
    @test fl_bp.from_to ≈ 250.0 atol = 1e-6
    @test fl_bp.to_from ≈ 250.0 atol = 1e-6

    bs = PNM.BranchesSeries((PSY.get_number(bus1), PSY.get_number(bus2)))
    PNM.add_branch!(bs, line1, :FromTo)
    PNM.add_branch!(bs, line2, :FromTo)
    # Series weakest-link rule: min(100, 150) = 100.0
    rating_eq = PNM.get_equivalent_rating(bs)
    @test rating_eq ≈ 100.0 atol = 1e-6

    emergency_rating_eq = PNM.get_equivalent_emergency_rating(bs)
    @test emergency_rating_eq ≈ 100.0 atol = 1e-6

    # Series chain containing a parallel block: the block contributes its N-1
    # single-element-contingency rating (100.0), so min(100, 150) = 100.0.
    bs = PNM.BranchesSeries((PSY.get_number(bus1), PSY.get_number(bus2)))
    PNM.add_branch!(bs, bp, :FromTo)
    PNM.add_branch!(bs, line2, :FromTo)
    rating_eq = PNM.get_equivalent_rating(bs)
    @test rating_eq ≈ 100.0 atol = 1e-6

    emergency_rating_eq = PNM.get_equivalent_emergency_rating(bs)
    @test emergency_rating_eq ≈ 150.0 atol = 1e-6

    # Test get_available: all branches must be available
    @test PSY.get_available(bp) == true
    @test PSY.get_available(bs) == true
end

@testset "Equivalent getters for ThreeWindingTransformerCircuit" begin
    # Create a test system with three-winding transformers
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")

    # Get a three-winding transformer from the system
    trf = first(collect(PSY.get_components(PSY.ThreeWindingTransformer, sys)))

    rating3 = PNM.get_equivalent_rating(PNM.ThreeWindingTransformerCircuit(trf, 3))
    # The circuit's own rating (device base); there is no parent-level rating to fall back to.
    expected_rating3 = PSY.get_rating(PSY.get_tertiary_circuit(trf), PSY.DU)
    @test rating3 == expected_rating3

    PSY.set_available!(PSY.get_secondary_circuit(trf), false)
    @test PNM.get_equivalent_available(PNM.ThreeWindingTransformerCircuit(trf, 3)) == true
    @test PNM.get_equivalent_available(PNM.ThreeWindingTransformerCircuit(trf, 2)) == false
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
        load_zone = nothing,
    )
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
        load_zone = nothing,
    )

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
    equivalent_pbranch =
        PNM.get_equivalent_physical_branch_parameters(
            branches_parallel,
            ybus.network_reduction_data,
        )
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
        equivalent_transformer = PSY.TwoWindingTransformer(;
            name = "equivalent_transformer",
            circuit = PSY.TransformerCircuit(;
                arc = PSY.Arc(; from = bus1, to = bus2),
                tap = PNM.get_equivalent_tap(equivalent_pbranch),
                α = PNM.get_equivalent_shift(equivalent_pbranch),
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                rating = 80.0,
                base_power = 100.0,
                base_voltage_primary = 1.0,
                r = PNM.get_equivalent_r(equivalent_pbranch),  # resistance
                x = PNM.get_equivalent_x(equivalent_pbranch),   # reactance
            ),
            magnetizing_shunt = Complex(
                PNM.get_equivalent_g_from(equivalent_pbranch),
                PNM.get_equivalent_b_from(equivalent_pbranch),
            ),
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
            load_zone = nothing,
        )
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
    equivalent_pbranch =
        PNM.get_equivalent_physical_branch_parameters(
            branches_series,
            ybus.network_reduction_data,
        )
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
        equivalent_transformer = PSY.TwoWindingTransformer(;
            name = "equivalent_transformer",
            circuit = PSY.TransformerCircuit(;
                arc = PSY.Arc(; from = bus1, to = bus2),
                tap = PNM.get_equivalent_tap(equivalent_pbranch),
                α = PNM.get_equivalent_shift(equivalent_pbranch),
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                rating = 80.0,
                base_power = 100.0,
                base_voltage_primary = 1.0,
                r = PNM.get_equivalent_r(equivalent_pbranch),  # resistance
                x = PNM.get_equivalent_x(equivalent_pbranch),   # reactance
            ),
            magnetizing_shunt = Complex(
                PNM.get_equivalent_g_from(equivalent_pbranch),
                PNM.get_equivalent_b_from(equivalent_pbranch),
            ),
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
    t1 = PSY.TwoWindingTransformer(;
        name = "tfw_1",
        circuit = PSY.TransformerCircuit(;
            arc = PSY.Arc(nothing),
            tap = 1.0,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 80.0,
            base_power = 100.0,
            base_voltage_primary = 1.0,
            base_voltage_secondary = 1.0,
            r = 0.122,  # resistance
            x = 0.1,   # reactance
        ),
        magnetizing_shunt = 0.01 + im * 0.02,
    )
    t2 = PSY.TwoWindingTransformer(;
        name = "tfw_2",
        circuit = PSY.TransformerCircuit(;
            arc = PSY.Arc(nothing),
            tap = 1.0,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 80.0,
            base_power = 100.0,
            base_voltage_primary = 1.0,
            base_voltage_secondary = 1.0,
            r = 0.3,  # resistance
            x = 0.13,   # reactance
        ),
        magnetizing_shunt = 0.02 + im * 0.021,
    )
    t3 = PSY.TwoWindingTransformer(;
        name = "tfw_3",
        circuit = PSY.TransformerCircuit(;
            arc = PSY.Arc(nothing),
            tap = 1.0,
            α = 0.2,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            rating = 80.0,
            base_power = 100.0,
            base_voltage_primary = 1.0,
            base_voltage_secondary = 1.0,
            r = 0.3,  # resistance
            x = 0.13,   # reactance
        ),
        magnetizing_shunt = 0.02 + im * 0.021,
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

@testset "grouped chains rate as the sum of each chain's weakest link" begin
    sys = build_two_parallel_degree_two_chains()
    # Distinct per-link ratings so minimum, maximum, and the cross-chain sum cannot
    # be confused with each other: chain A carries 8.0/3.0/5.0 (min 3.0, max 8.0),
    # chain B carries 9.0/4.0/6.0 (min 4.0, max 9.0), and the group must report
    # 3.0 + 4.0 = 7.0, a value that matches none of the individual link or
    # per-chain numbers above.
    #
    # Each chain's minimum sits in the middle segment, not the first or last, so the
    # assertion holds regardless of which direction chain discovery traverses the
    # chain: under either direction the middle segment is still neither the first
    # nor the last one visited, so an aggregate that (wrongly) returned the first-
    # or last-visited segment's rating, instead of the true minimum, would be caught.
    chain_a_ratings = Dict("L_1_10" => 8.0, "L_10_11" => 3.0, "L_11_3" => 5.0)
    chain_b_ratings = Dict("L_1_20" => 9.0, "L_20_21" => 4.0, "L_21_3" => 6.0)
    for (name, rating) in merge(chain_a_ratings, chain_b_ratings)
        PSY.set_rating!(PSY.get_component(PSY.Line, sys, name), rating * PSY.DU)
    end

    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)
    arc = only(
        k for (k, v) in PNM.get_parallel_branch_map(nrd)
        if all(m isa PNM.BranchesSeries for m in v)
    )
    group = PNM.get_parallel_branch_map(nrd)[arc]

    per_chain = [PNM.get_equivalent_rating(chain) for chain in group]
    @test sort(per_chain) == [3.0, 4.0]
    for chain in group
        # min of the links in that chain: a wrong aggregate (e.g. max or sum)
        # would return 8.0/9.0 or 16.0/19.0 instead, none of which equal 3.0/4.0.
        @test PNM.get_equivalent_rating(chain) ==
              minimum(PNM.get_equivalent_rating(seg) for seg in chain)
    end
    # The group sums its two chains' weakest-link ratings: 3.0 + 4.0 = 7.0, a value
    # distinct from every link rating and from both chains' individual minima/maxima.
    @test PNM.get_equivalent_rating(group) == 7.0
    @test PNM.get_equivalent_rating(group) == sum(per_chain)
end

# The mirror case — a parallel block nested inside a chain contributes its N-1 rating via
# `_series_member_rating(::AbstractBranchesParallel)`, not covered by the testset above.

@testset "Compute equivalent physical parameters for WECC 240 bus" begin
    sys = PSB.build_system(PSYTestSystems, "psse_240_parsing_sys"; runchecks = false)
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = ybus.network_reduction_data
    for branches_parallel in values(nr.parallel_branch_map)
        @test isa(
            PNM.get_equivalent_physical_branch_parameters(branches_parallel, nr),
            PNM.EquivalentBranch,
        )
    end
    for branches_series in values(nr.series_branch_map)
        @test isa(
            PNM.get_equivalent_physical_branch_parameters(branches_series, nr),
            PNM.EquivalentBranch,
        )
    end
end
