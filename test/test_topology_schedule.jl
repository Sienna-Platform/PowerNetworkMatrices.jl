_c_sys14() = PSB.build_system(PSB.PSITestSystems, "c_sys14")

function _max_row_error(rows, builder, out_names, vptdf; skip_arc = nothing)
    sys_ref = builder()
    for n in out_names
        PSY.set_available!(PSY.get_component(PSY.Line, sys_ref, n), false)
    end
    ref = PTDF(sys_ref)
    arc_lookup = PNM.get_arc_lookup(ref)
    bus_lookup = PNM.get_bus_lookup(ref)
    err = 0.0
    for arc in PNM.get_arc_axis(vptdf)
        (!haskey(arc_lookup, arc) || arc == skip_arc) && continue
        row = rows(arc)
        for (j, bus) in enumerate(PNM.get_bus_axis(vptdf))
            err = max(err, abs(row[j] - ref[arc_lookup[arc], bus_lookup[bus]]))
        end
    end
    return err
end

@testset "TopologySchedule rows, indices and arcs out" begin
    sys = _c_sys14()
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    a, b = lines[1], lines[2]
    removed = [
        Set{PSY.Component}(),
        Set{PSY.Component}([a]),
        Set{PSY.Component}([a]),
        Set{PSY.Component}([a, b]),
        Set{PSY.Component}(),
    ]
    ts = TopologySchedule(vmodf, sys, removed)
    @test get_topology_index(ts, 1) == 0
    @test get_topology_index(ts, 5) == 0
    @test isnothing(get_network_modification(ts, 1))
    @test get_topology_index(ts, 2) == get_topology_index(ts, 3)
    @test get_topology_index(ts, 2) != get_topology_index(ts, 4)
    @test length(unique(get_topology_index(ts, t) for t in 1:5)) == 3

    arc_a = _arc_numbers(a)
    @test !is_arc_out(ts, arc_a, 1)
    @test is_arc_out(ts, arc_a, 2)

    bus_ax = PNM.get_bus_axis(vptdf)
    for (t, out_names) in
        ((1, String[]), (2, [PSY.get_name(a)]), (4, PSY.get_name.([a, b])))
        ptdf_ref, _ = _ref_ptdf_without(_c_sys14, out_names)
        ref_arc_lookup = PNM.get_arc_lookup(ptdf_ref)
        ref_bus_lookup = PNM.get_bus_lookup(ptdf_ref)
        for arc in PNM.get_arc_axis(vptdf)
            haskey(ref_arc_lookup, arc) || continue
            expected =
                [ptdf_ref[ref_arc_lookup[arc], ref_bus_lookup[bus]] for bus in bus_ax]
            @test isapprox(ts[arc, t], expected; atol = 1e-6)
        end
    end
    first_arc = PNM.get_arc_axis(vptdf)[1]
    @test ts[first_arc, 1] ≈ get_ptdf_row(vptdf, first_arc)
end

@testset "get_ptdf_row on a VirtualMODF" begin
    sys = _c_sys14()
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    @test all(
        isapprox(get_ptdf_row(vmodf, arc), Vector(get_ptdf_row(vptdf, arc)); atol = 1e-8)
        for arc in PNM.get_arc_axis(vptdf)
    )
    @test_throws ErrorException get_ptdf_row(vmodf, (9999, 10000))
end

@testset "Base rows are cached in the VirtualMODF across schedules" begin
    sys = _c_sys14()
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    arc = PNM.get_arc_axis(vptdf)[1]
    idx = PNM.get_arc_lookup(vptdf)[arc]
    base_steps = [Set{PSY.Component}()]
    row = TopologySchedule(vmodf, sys, base_steps)[arc, 1]
    @test row ≈ Vector(get_ptdf_row(vptdf, arc)) atol = 1e-8
    rc = PNM.get_row_caches(vmodf)[NetworkModification("", PNM.ArcModification[])]
    @test haskey(rc, idx)
    # A sentinel in the cache proves later reads, from a new schedule too, never re-solve.
    sentinel = fill(42.0, length(row))
    rc[idx] = sentinel
    @test TopologySchedule(vmodf, sys, base_steps)[arc, 1] == sentinel
    @test get_ptdf_row(vmodf, arc) == sentinel
    @test get_ptdf_row(vmodf, arc) !== get_ptdf_row(vmodf, arc)
end

@testset "TopologySchedule islands" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    bridge = _find_bridge_line(sys)
    @test !isnothing(bridge)
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}(), Set{PSY.Component}([bridge])])
    islands0, refs0 = get_islands(ts, 1)
    islands1, refs1 = get_islands(ts, 2)
    @test length(islands1) == length(islands0) + 1
    @test length(refs1) == length(refs0)

    PSY.set_available!(bridge, false)
    expected = Set(Set(v) for v in values(PNM.find_subnetworks(PNM.Ybus(sys))))
    PSY.set_available!(bridge, true)
    retained = Set(PNM.get_bus_axis(vptdf))
    @test Set(Set(filter(in(retained), isl)) for isl in islands1) ⊇
          Set(filter(!isempty, [Set(filter(in(retained), collect(s))) for s in expected]))
end

@testset "Parallel group outages with unequal r/x match a rebuilt PTDF" begin
    for r_scale in (1.0, 5.0)
        builder = () -> _c_sys14_with_parallel_pair(; r_scale = r_scale)[1]
        sys, base, dup = _c_sys14_with_parallel_pair(; r_scale = r_scale)
        vptdf = VirtualPTDF(sys)
        vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
        arc = _arc_numbers(base)
        nr = PNM.get_network_reduction_data(vptdf)
        @test haskey(PNM.get_parallel_branch_map(nr), arc)
        arc_idx = PNM.get_arc_lookup(vptdf)[arc]

        partial = NetworkModification(vptdf, sys, [dup])
        full = NetworkModification(vptdf, sys, [base, dup])
        @test only(partial.arc_modifications).opened == 1
        @test only(full.arc_modifications).opened == 2
        b_arc = PNM._get_arc_susceptances(vptdf)[arc_idx]
        @test only(full.arc_modifications).delta_b == -b_arc
        # a single-branch constructor goes through the same joint rule
        @test NetworkModification(vptdf, dup).arc_modifications ==
              partial.arc_modifications

        for (mod, names, skip) in
            ((partial, ["dup"], nothing), (full, ["dup", PSY.get_name(base)], arc))
            err = _max_row_error(
                a -> vptdf[a, mod], builder, names, vptdf; skip_arc = skip,
            )
            @test err < 1e-9
        end

        ts = TopologySchedule(
            vmodf,
            sys,
            [
                Set{PSY.Component}(),
                Set{PSY.Component}([dup]),
                Set{PSY.Component}([base, dup]),
            ],
        )
        @test Set(get_in_service_members(ts, arc, 1)) == Set([base, dup])
        @test get_in_service_members(ts, arc, 2) == [base]
        @test isempty(get_in_service_members(ts, arc, 3))
        @test !is_arc_out(ts, arc, 2)
        @test is_arc_out(ts, arc, 3)
        other_arc = first(a for a in PNM.get_arc_axis(vptdf) if a != arc)
        @test isempty(get_in_service_members(ts, other_arc, 1))
        @test _max_row_error(a -> ts[a, 2], builder, ["dup"], vptdf) < 1e-9
        @test _max_row_error(
            a -> ts[a, 3], builder, ["dup", PSY.get_name(base)], vptdf; skip_arc = arc,
        ) < 1e-9

        # a planned set and a contingency still resolve jointly
        ctg = PNM._post_contingency_modification(ts, get_topology_index(ts, 2), [base])
        @test ctg == full
    end
end

@testset "Lossy parallel bridge: islands agree with the kernel" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    base = _find_bridge_line(sys)
    @test !isnothing(base)
    _, _, dup = _add_parallel_line!(sys, base, 5.0)
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    arc = _arc_numbers(base)
    @test haskey(PNM.get_parallel_branch_map(PNM.get_network_reduction_data(vptdf)), arc)
    ts = TopologySchedule(
        vmodf,
        sys,
        [Set{PSY.Component}(), Set{PSY.Component}([dup]), Set{PSY.Component}([base, dup])],
    )
    islands1, _ = get_islands(ts, 1)
    islands2, _ = get_islands(ts, 2)
    islands3, refs3 = get_islands(ts, 3)
    @test length(islands2) == length(islands1)
    @test length(islands3) == length(islands1) + 1
    @test !is_arc_out(ts, arc, 2)
    @test is_arc_out(ts, arc, 3)

    bus_ax = PNM.get_bus_axis(vptdf)
    cut_off = [
        findfirst(==(bus), bus_ax) for (k, isl) in enumerate(islands3) if k ∉ refs3 for
        bus in isl
    ]
    @test !isempty(cut_off)
    # kernel: the cut-off buses feed no other arc, and the removed arc's own row is zero
    @test all(iszero, ts[arc, 3])
    @test all(
        all(abs.(ts[other, 3][cut_off]) .< 1e-12) for
        other in PNM.get_arc_axis(vptdf) if other != arc
    )
    @test any(!iszero, ts[arc, 2])
end

@testset "Ward equivalent on the arc of a tripped branch keeps conducting" begin
    sys = _c_sys14()
    # Buses 4 and 5 are boundary buses joined by a line, so the Ward equivalent of the
    # external network lands on the line's own arc.
    reductions = NetworkReduction[WardReduction([1, 2, 3, 4, 5])]
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    nr = PNM.get_network_reduction_data(vptdf)
    @test length(PNM.get_bus_axis(vptdf)) >= 4
    arc = (4, 5)
    line = PNM.get_direct_branch_map(nr)[arc]
    @test haskey(PNM.get_added_arc_impedance_map(nr), arc)
    arc_idx = PNM.get_arc_lookup(vptdf)[arc]
    b_arc = PNM._get_arc_susceptances(vptdf)[arc_idx]
    b_ward = PNM._added_arc_susceptance(nr, arc)
    @test b_ward > 0

    mod = NetworkModification(vptdf, sys, [line])
    m = only(mod.arc_modifications)
    @test m.delta_b ≈ b_ward - b_arc
    @test b_arc + m.delta_b ≈ b_ward

    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}([line])])
    @test !is_arc_out(ts, arc, 1)
    @test length(first(get_islands(ts, 1))) == 1
    @test get_in_service_entry(ts, arc, 1) === PNM.get_added_arc_impedance_map(nr)[arc]
    @test PNM.get_equivalent_rating(get_in_service_entry(ts, arc, 1)) == 1e6
    @test get_arc_shift_injection(ts, arc, 1) == 0.0

    # The line sits inside the study area, so removing it leaves the Ward equivalent as is.
    PSY.set_available!(line, false)
    ref = VirtualPTDF(sys; network_reductions = reductions)
    PSY.set_available!(line, true)
    @test Set(PNM.get_arc_axis(ref)) == Set(PNM.get_arc_axis(vptdf))
    @test PNM.get_bus_axis(ref) == PNM.get_bus_axis(vptdf)
    @test all(
        isapprox(ts[a, 1], Vector(get_ptdf_row(ref, a)); atol = 1e-9) for
        a in PNM.get_arc_axis(vptdf)
    )
    @test !isapprox(ts[arc, 1], Vector(get_ptdf_row(vptdf, arc)); atol = 1e-6)
end

@testset "TopologySchedule islands on a system with two subnetworks" begin
    sys = build_hvdc_with_single_bus_island()
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    line = first(sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name))
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}(), Set{PSY.Component}([line])])
    expected = Set(Set(v) for v in values(PNM.find_subnetworks(PNM.Ybus(sys))))
    @test length(expected) == 2
    for t in 1:2
        islands, refs = get_islands(ts, t)
        @test Set(Set(island) for island in islands) == expected
        @test refs == Set(1:2)
    end
end

# A shifter with `base`'s reactance on `base`'s arc, so it joins `base`'s parallel group.
_make_parallel_shifter!(sys, base; α) =
    _add_shifter!(sys, PSY.get_arc(base), PSY.get_x(base, u"SU"); α = α)

# c_sys14 with a shifter from a new bus 99 to `base`'s from-bus, and a zero-impedance line
# from `base`'s to-bus to bus 99. Once the zero-impedance reduction merges bus 99 into the
# to-bus, the shifter is an anti-parallel member of `base`'s group. With `with_dup`, the
# group also holds the "dup" line.
function _c_sys14_with_merged_antiparallel_shifter(; with_dup = true)
    sys = _c_sys14()
    base = first(sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name))
    dup = nothing
    if with_dup
        _, _, dup = _add_parallel_line!(sys, base, 1.0)
    end
    from = PSY.get_from(PSY.get_arc(base))
    to = PSY.get_to(PSY.get_arc(base))
    stub = PSY.ACBus(;
        number = 99,
        name = "stub",
        available = true,
        bustype = PSY.ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = PSY.get_base_voltage(to),
        area = PSY.get_area(to),
        load_zone = PSY.get_load_zone(to),
        input_basis = u"CU",
    )
    PSY.add_component!(sys, stub)
    zero_arc = PSY.Arc(; from = to, to = stub)
    PSY.add_component!(sys, zero_arc)
    PSY.add_component!(
        sys,
        PSY.Line(;
            input_basis = u"CU",
            name = "zero",
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = zero_arc,
            r = 0.0,
            x = 0.0,
            b = (from = 0.0, to = 0.0),
            rating = 10.0,
            angle_limits = (min = -1.5, max = 1.5),
        ),
    )
    shifter_arc = PSY.Arc(; from = stub, to = from)
    PSY.add_component!(sys, shifter_arc)
    shifter = _add_shifter!(sys, shifter_arc, PSY.get_x(base, u"SU"); α = 0.06)
    return sys, base, dup, shifter
end

@testset "In-service entry of a parallel group" begin
    sys, base, dup = _c_sys14_with_parallel_pair(; r_scale = 1.7)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    arc = _arc_numbers(base)
    ts = TopologySchedule(
        vmodf,
        sys,
        [Set{PSY.Component}(), Set{PSY.Component}([dup]), Set{PSY.Component}([base, dup])],
    )
    nr = get_network_reduction_data(vmodf)
    @test get_in_service_entry(ts, arc, 1) === PNM.get_parallel_branch_map(nr)[arc]
    @test get_in_service_entry(ts, arc, 2) === base
    @test isnothing(get_in_service_entry(ts, arc, 3))
    other = _arc_numbers(
        sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)[2],
    )
    @test get_in_service_entry(ts, other, 2) ===
          PNM.get_reduction_entry(PNM.get_branch_catalog(vmodf), other)
end

@testset "In-service entry of a partly out group of three is a group of the rest" begin
    sys, base, dup = _c_sys14_with_parallel_pair(; r_scale = 1.0)
    shifter = _make_parallel_shifter!(sys, base; α = 0.08)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    arc = _arc_numbers(base)
    nr = get_network_reduction_data(vmodf)
    group = PNM.get_parallel_branch_map(nr)[arc]
    @test length(group) == 3
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}([dup])])
    entry = get_in_service_entry(ts, arc, 1)
    @test entry isa PNM.MixedBranchesParallel
    @test Set(collect(entry)) == Set{PSY.ACTransmission}([base, shifter])
    @test PNM.get_arc_tuple(entry, nr) == PNM.get_arc_tuple(group, nr)
end

@testset "A contingency on a step rates the group by the members in service in the union" begin
    sys, base, dup = _c_sys14_with_parallel_pair(; r_scale = 1.0)
    shifter = _make_parallel_shifter!(sys, base; α = 0.08)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    arc = _arc_numbers(base)
    group = PNM.get_parallel_branch_map(get_network_reduction_data(vmodf))[arc]
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}([dup])])
    at_step = get_in_service_entry(ts, arc, 1)
    tripped(cs...) = Base.IdSet{PSY.ACTransmission}(cs)

    @test PNM.get_post_outage_emergency_rating(at_step, tripped(shifter)) ==
          PNM.get_post_outage_emergency_rating(group, tripped(dup, shifter)) ==
          PNM.get_equivalent_emergency_rating(base)
    @test PNM.get_post_outage_emergency_rating(at_step, tripped()) ==
          PNM.get_post_outage_emergency_rating(group, tripped(dup))
    @test !PNM.is_open_after_outage(at_step, tripped(shifter))
    @test PNM.is_open_after_outage(at_step, tripped(base, shifter))
    @test PNM.is_open_after_outage(group, tripped(dup, base, shifter))
    @test !PNM.is_open_after_outage(group, tripped(base, shifter))
end

@testset "Shift injection per step matches a rebuild" begin
    sys = _c_sys14()
    trafo = first(
        sort!(
            collect(PSY.get_components(PSY.TwoWindingTransformer, sys));
            by = PSY.get_name,
        ),
    )
    PSY.set_α!(PSY.get_circuit(trafo), 0.1)
    arc = _arc_numbers(trafo)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}(), Set{PSY.Component}([trafo])])
    nr = get_network_reduction_data(vmodf)
    @test get_arc_shift_injection(ts, arc, 1) == PNM.arc_dc_shift_injection(nr, arc)
    @test !iszero(get_arc_shift_injection(ts, arc, 1))
    @test get_arc_shift_injection(ts, arc, 2) == 0.0
    other = first(a for a in PNM.get_arc_axis(vmodf) if a != arc)
    @test get_arc_shift_injection(ts, other, 2) == PNM.arc_dc_shift_injection(nr, other)
end

@testset "Shift injection of a partially out parallel group matches a rebuild" begin
    sys, base, dup = _c_sys14_with_parallel_pair(; r_scale = 1.0)
    _make_parallel_shifter!(sys, base; α = 0.08)
    arc = _arc_numbers(base)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    ts = TopologySchedule(
        vmodf,
        sys,
        [Set{PSY.Component}([dup]), Set{PSY.Component}([base, dup])],
    )
    ref_sys = _c_sys14()
    ref_base = PSY.get_component(PSY.Line, ref_sys, PSY.get_name(base))
    _make_parallel_shifter!(ref_sys, ref_base; α = 0.08)
    ref_nr = get_network_reduction_data(Ybus(ref_sys))
    @test isapprox(
        get_arc_shift_injection(ts, arc, 1),
        PNM.arc_dc_shift_injection(ref_nr, arc);
        atol = 1e-10,
    )
    @test !iszero(get_arc_shift_injection(ts, arc, 1))

    # The shifter alone: a lone member answers with its own b·α.
    PSY.set_available!(ref_base, false)
    lone_nr = get_network_reduction_data(Ybus(ref_sys))
    @test isapprox(
        get_arc_shift_injection(ts, arc, 2),
        PNM.arc_dc_shift_injection(lone_nr, arc);
        atol = 1e-10,
    )
end

@testset "An anti-parallel member keeps the group's frame when partly out" begin
    reductions = NetworkReduction[ZeroImpedanceBranchReduction()]
    sys, base, dup, shifter = _c_sys14_with_merged_antiparallel_shifter()
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(vptdf, sys; automatically_register_outages = false)
    nr = get_network_reduction_data(vmodf)
    arc = PNM.get_arc_tuple(base, nr)
    group = PNM.get_parallel_branch_map(nr)[arc]
    @test Set(collect(group)) == Set{PSY.ACTransmission}([base, dup, shifter])
    @test PNM.get_arc_tuple(shifter, nr) == reverse(arc)
    ts = TopologySchedule(
        vmodf,
        sys,
        [Set{PSY.Component}([dup]), Set{PSY.Component}([base, dup])],
    )

    ref_sys, _, _, _ = _c_sys14_with_merged_antiparallel_shifter(; with_dup = false)
    ref = VirtualPTDF(ref_sys; network_reductions = reductions)
    ref_nr = get_network_reduction_data(ref)
    @test PNM.get_bus_axis(ref) == PNM.get_bus_axis(vptdf)
    @test isapprox(
        get_arc_shift_injection(ts, arc, 1),
        PNM.arc_dc_shift_injection(ref_nr, arc);
        atol = 1e-10,
    )
    @test !iszero(get_arc_shift_injection(ts, arc, 1))
    @test all(
        isapprox(ts[a, 1], Vector(get_ptdf_row(ref, a)); atol = 1e-8) for
        a in PNM.get_arc_axis(ref)
    )

    # The shifter alone is registered on the reversed arc, so its injection flips sign in
    # the group's frame.
    @test get_in_service_entry(ts, arc, 2) === shifter
    PSY.set_available!(PSY.get_component(PSY.Line, ref_sys, PSY.get_name(base)), false)
    lone_nr = get_network_reduction_data(Ybus(ref_sys; network_reductions = reductions))
    @test get_arc_shift_injection(ts, arc, 2) ≈
          -PNM.arc_dc_shift_injection(lone_nr, reverse(arc)) atol = 1e-10
end

@testset "Unattached components with equal ids are distinct topologies" begin
    sys = _c_sys14()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    base = first(sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name))
    detached = [
        PSY.Line(;
            input_basis = u"CU",
            name = name,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = PSY.get_arc(base),
            r = 0.0,
            x = 0.1,
            b = (from = 0.0, to = 0.0),
            rating = 1.0,
            angle_limits = (min = -1.5, max = 1.5),
        ) for name in ("detached_1", "detached_2")
    ]
    @test IS.get_id(detached[1]) == IS.get_id(detached[2])
    not_found = (:info, r"not found in any reduction map")
    no_mods = (:info, r"No valid arc or shunt modifications")
    ts = @test_logs not_found no_mods not_found no_mods TopologySchedule(
        vmodf,
        sys,
        [Set{PSY.Component}([detached[1]]), Set{PSY.Component}([detached[2]])],
    )
    @test get_topology_index(ts, 1) != get_topology_index(ts, 2)
end

function _row_batch_schedule(sys, vmodf)
    line(name) = PSY.get_component(PSY.Line, sys, name)
    line1, dup, line7 = line("Line1"), line("dup"), line("Line7")
    shifter = PSY.get_component(PSY.TwoWindingTransformer, sys, "shifter")
    trans4 = PSY.get_component(PSY.TwoWindingTransformer, sys, "Trans4")
    removed = [
        Set{PSY.Component}(),
        Set{PSY.Component}([line1, dup]),
        Set{PSY.Component}([dup]),
        Set{PSY.Component}([shifter]),
        Set{PSY.Component}([trans4]),
        Set{PSY.Component}([line7, shifter, dup]),
        Set{PSY.Component}([dup]),
    ]
    return TopologySchedule(vmodf, sys, removed)
end

@testset "Rows of one arc on every topology equal per-topology rows bit for bit" begin
    sys = _c_sys14_for_row_batches()
    for tol in (nothing, 1e-3)
        new_vmodf() = VirtualMODF(
            isnothing(tol) ? VirtualPTDF(sys) : VirtualPTDF(sys; tol = tol),
            sys;
            automatically_register_outages = false,
        )
        reference = _row_batch_schedule(sys, new_vmodf())
        batched = _row_batch_schedule(sys, new_vmodf())
        @test length(batched.modifications) == 5
        @test any(
            compute_woodbury_factors(batched.vmodf, mod).is_islanding for
            mod in batched.modifications
        )
        first_step = Dict(get_topology_index(reference, t) => t for t in 7:-1:1)
        topologies = collect(0:5)
        n_zero_rows = 0
        for arc in PNM.get_arc_axis(batched.vmodf)
            rows = get_topology_ptdf_rows(batched, arc)
            @test length(rows) == length(topologies)
            for (k, row) in zip(topologies, rows)
                expected = reference[arc, first_step[k]]
                @test _bitwise_equal(row, expected)
                n_zero_rows += iszero(row) && is_arc_out(batched, arc, first_step[k])
            end
            # Cache hits, a subset and repeated topologies give the same rows.
            again = get_topology_ptdf_rows(batched, arc, [3, 0, 3])
            @test all(_bitwise_equal.(again, rows[[4, 1, 4]]))
            @test _bitwise_equal(batched[arc, 2], rows[get_topology_index(batched, 2) + 1])
        end
        @test n_zero_rows > 0
        isnothing(tol) || @test all(
            r -> r isa PNM.SparseArrays.SparseVector,
            get_topology_ptdf_rows(batched, first(PNM.get_arc_axis(batched.vmodf))),
        )
    end
end

@testset "Rows on several modifications fill only the missing rows" begin
    sys = _c_sys14_for_row_batches()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    ts = _row_batch_schedule(sys, vmodf)
    arc = (1, 2)
    partly_cached = vmodf[arc, ts.modifications[2]]
    mods = [PNM._BASE_NETWORK; ts.modifications]
    rows = get_post_modification_ptdf_rows(vmodf, arc, mods)
    @test _bitwise_equal(rows[3], partly_cached)
    fresh = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    for (mod, row) in zip(mods, rows)
        @test haskey(PNM.get_row_caches(vmodf)[mod], PNM.get_arc_lookup(vmodf)[arc])
        @test _bitwise_equal(row, fresh[arc, mod])
    end
    # The caller owns the returned rows.
    rows[1][1] += 1.0
    @test _bitwise_equal(
        get_post_modification_ptdf_rows(vmodf, arc, mods[1:1])[1],
        fresh[arc, mods[1]],
    )
    # An integer monitored index gives the same rows as its bus pair.
    by_index = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    index_rows = get_post_modification_ptdf_rows(
        by_index,
        PNM.get_arc_lookup(by_index)[arc],
        mods,
    )
    @test all(_bitwise_equal.(index_rows, [fresh[arc, mod] for mod in mods]))
end

_row_batch_component(sys, name) = something(
    PSY.get_component(PSY.Line, sys, name),
    PSY.get_component(PSY.TwoWindingTransformer, sys, name),
)

const _ORACLE_CASES = [
    String[],
    ["Line7"],
    ["dup"],
    ["shifter"],
    ["Line1", "dup"],
    ["Line7", "dup", "shifter"],
]

@testset "Rows on several topologies match PTDFs of rebuilt systems" begin
    sys = _c_sys14_for_row_batches()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    removed = [
        Set{PSY.Component}(_row_batch_component(sys, n) for n in names) for
        names in _ORACLE_CASES
    ]
    ts = TopologySchedule(vmodf, sys, removed)
    arcs = PNM.get_arc_axis(vmodf)
    bus_axis = PNM.get_bus_axis(vmodf)
    rows = Dict(arc => get_topology_ptdf_rows(ts, arc) for arc in arcs)
    for (t, names) in enumerate(_ORACLE_CASES)
        k = get_topology_index(ts, t) + 1
        err = _max_error_against_rebuild(a -> rows[a][k], arcs, bus_axis, names)
        @test err < 1e-10
    end
end

@testset "Rows on several modifications with distributed slack match rebuilt systems" begin
    sys = _c_sys14_for_row_batches()
    uniform(s) = Dict(
        PSY.get_number(b) => 1.0 / length(PSY.get_components(PSY.ACBus, s)) for
        b in PSY.get_components(PSY.ACBus, s)
    )
    vptdf = VirtualPTDF(sys; dist_slack = uniform(sys))
    mods = [
        NetworkModification(vptdf, sys, [_row_batch_component(sys, n) for n in names])
        for
        names in _ORACLE_CASES
    ]
    arcs = PNM.get_arc_axis(vptdf)
    rows = Dict(arc => get_post_modification_ptdf_rows(vptdf, arc, mods) for arc in arcs)
    for arc in arcs, (k, mod) in enumerate(mods)
        @test _bitwise_equal(rows[arc][k], vptdf[arc, mod])
    end
    for (k, names) in enumerate(_ORACLE_CASES)
        sys_ref = _c_sys14_for_row_batches()
        err = _max_error_against_rebuild(
            a -> rows[a][k],
            arcs,
            PNM.get_bus_axis(vptdf),
            names;
            dist_slack = uniform(sys_ref),
        )
        @test err < 1e-10
    end
end

@testset "A removed 3W transformer takes its winding out of a parallel group" begin
    sys, trf, beside, arc = _case10_with_transformer_beside_winding()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys; automatically_register_outages = false)
    nr = get_network_reduction_data(vmodf)
    group = PNM.get_parallel_branch_map(nr)[arc]
    @test length(group) == 2
    @test !iszero(PNM.arc_dc_shift_injection(nr, arc))
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}(), Set{PSY.Component}([trf])])
    @test get_in_service_entry(ts, arc, 1) === group
    @test get_in_service_members(ts, arc, 2) == PSY.ACTransmission[beside]
    @test get_in_service_entry(ts, arc, 2) === beside

    ref_sys, _, _, _ = _case10_with_transformer_beside_winding(; with_three_winding = false)
    ref_nr = get_network_reduction_data(Ybus(ref_sys))
    @test get_arc_shift_injection(ts, arc, 2) ≈ PNM.arc_dc_shift_injection(ref_nr, arc) atol =
        1e-12
    @test PNM.get_series_susceptance(get_in_service_entry(ts, arc, 2), u"SU") ≈
          PNM.get_series_susceptance(PNM.get_direct_branch_map(ref_nr)[arc], u"SU")
end

@testset "retain_modifications! evicts every other modification's caches" begin
    sys = _c_sys14_for_row_batches()
    line7 = _row_batch_component(sys, "Line7")
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    PSY.add_supplemental_attribute!(sys, line7, outage)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    contingency = only(values(get_registered_contingencies(vmodf))).modification
    mods = [
        NetworkModification(vmodf, sys, [_row_batch_component(sys, n) for n in names])
        for names in (["dup"], ["shifter"], ["Line1", "dup"])
    ]
    arc = (1, 2)
    for mod in [PNM._BASE_NETWORK, contingency, mods...]
        vmodf[arc, mod]
    end
    @test length(PNM.get_row_caches(vmodf)) == 5
    @test length(PNM.get_woodbury_cache(vmodf)) == 4

    # A modification rebuilt from the same components is the one retained.
    kept = NetworkModification(vmodf, sys, [_row_batch_component(sys, "dup")])
    retain_modifications!(vmodf, [kept])
    @test Set(keys(PNM.get_row_caches(vmodf))) == Set([PNM._BASE_NETWORK, mods[1]])
    @test Set(keys(PNM.get_woodbury_cache(vmodf))) == Set([mods[1]])
    @test haskey(get_registered_contingencies(vmodf), IS.get_id(outage))

    # Evicted modifications, the contingency among them, recompute the same rows.
    fresh = VirtualMODF(VirtualPTDF(sys), sys)
    @test vmodf[arc, mods[3]] ≈ fresh[arc, mods[3]]
    @test vmodf[arc, outage] ≈ fresh[arc, outage]
    retain_modifications!(vmodf, [contingency])
    @test Set(keys(PNM.get_row_caches(vmodf))) == Set([PNM._BASE_NETWORK, contingency])
    @test Set(keys(PNM.get_woodbury_cache(vmodf))) == Set([contingency])
    retain_modifications!(vmodf, NetworkModification[])
    @test Set(keys(PNM.get_row_caches(vmodf))) == Set([PNM._BASE_NETWORK])
    @test isempty(PNM.get_woodbury_cache(vmodf))
end

# The topologies and post-contingency networks a window of `removed` resolves rows on.
function _window_in_use(vmodf, sys, removed, contingency_id)
    ts = TopologySchedule(vmodf, sys, removed)
    in_use = Set{Union{NetworkModification, PostContingencyNetwork}}()
    for t in eachindex(removed)
        mod = get_network_modification(ts, t)
        isnothing(mod) || push!(in_use, mod)
        net = get_post_contingency_network(ts, t, contingency_id)
        isnothing(net) || push!(in_use, net)
    end
    return ts, in_use
end

function _query_window!(vmodf, ts, contingency_id, arcs, n_steps)
    for t in 1:n_steps, arc in arcs
        ts[arc, t]
        net = get_post_contingency_network(ts, t, contingency_id)
        isnothing(net) || get_post_contingency_ptdf_rows(vmodf, arc, [net])
    end
    return
end

@testset "A window with an outage at every step evicts the contingency-only caches" begin
    sys = _c_sys14_for_row_batches()
    line(name) = PSY.get_component(PSY.Line, sys, name)
    line7 = line("Line7")
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    PSY.add_supplemental_attribute!(sys, line7, outage)
    contingency_id = IS.get_id(outage)
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    contingency = only(values(get_registered_contingencies(vmodf))).modification
    arcs = [(1, 2), (2, 3), (4, 5)]
    windows = [
        [Set{PSY.Component}(), Set{PSY.Component}([line("Line1")])],
        [Set{PSY.Component}([line("Line1")]), Set{PSY.Component}([line("dup")])],
        [Set{PSY.Component}(), Set{PSY.Component}([line("dup")])],
    ]
    has_contingency = Bool[]
    for removed in windows
        ts, in_use = _window_in_use(vmodf, sys, removed, contingency_id)
        retain_modifications!(vmodf, in_use)
        _query_window!(vmodf, ts, contingency_id, arcs, length(removed))
        topologies = Set(
            get_network_modification(ts, t) for t in eachindex(removed) if
            !isnothing(get_network_modification(ts, t))
        )
        networks = Set(n for n in in_use if n isa PostContingencyNetwork)
        base_tier2 = Set(n.tier2 for n in networks if PNM._on_base_topology(n))
        others = Set(n for n in networks if !PNM._on_base_topology(n))
        @test Set(keys(PNM.get_row_caches(vmodf))) ==
              union(topologies, base_tier2, [PNM._BASE_NETWORK])
        @test Set(keys(PNM.get_woodbury_cache(vmodf))) == union(topologies, base_tier2)
        @test Set(keys(PNM.get_tier2_row_caches(vmodf))) == others
        push!(has_contingency, haskey(PNM.get_row_caches(vmodf), contingency))

        # Rows resolved after the eviction equal those of a matrix that never evicted.
        fresh = VirtualMODF(VirtualPTDF(sys), sys)
        fresh_ts = TopologySchedule(fresh, sys, removed)
        for t in eachindex(removed), arc in arcs
            @test ts[arc, t] ≈ fresh_ts[arc, t]
            net = get_post_contingency_network(ts, t, contingency_id)
            fresh_net = get_post_contingency_network(fresh_ts, t, contingency_id)
            @test only(get_post_contingency_ptdf_rows(vmodf, arc, [net])) ≈
                  only(get_post_contingency_ptdf_rows(fresh, arc, [fresh_net]))
        end
    end
    @test has_contingency == [true, false, true]
end
