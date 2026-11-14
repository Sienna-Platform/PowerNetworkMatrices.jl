import Logging

function _sys_with_open_switch(; x = 0.05, r = 0.01)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses = sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number)
    arc = PSY.Arc(buses[1], buses[3])
    PSY.add_component!(sys, arc)
    switch = PSY.DiscreteControlledACBranch(nothing)
    switch.name = "tie_switch"
    switch.arc = arc
    switch.x = x
    switch.r = r
    switch.available = true
    switch.branch_status = PSY.DiscreteControlledBranchStatus.OPEN
    PSY.add_component!(sys, switch)
    return sys, switch, (PSY.get_number(buses[1]), PSY.get_number(buses[3]))
end

const _STUB_BUS = 99

# A new leaf bus joined to the network by a line and, in parallel, an open switch. A radial
# reduction removes the leaf with both branches unless the leaf is kept.
function _sys_with_open_stub_switch()
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    hub = first(sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number))
    leaf = PSY.ACBus(;
        number = _STUB_BUS,
        name = "leaf_bus",
        available = true,
        bustype = PSY.ACBusTypes.PQ,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = PSY.get_base_voltage(hub),
        input_basis = u"CU",
    )
    PSY.add_component!(sys, leaf)
    arc = PSY.Arc(; from = hub, to = leaf)
    PSY.add_component!(sys, arc)
    line = PSY.Line(; input_basis = u"CU",
        name = "stub_line",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.01,
        x = 0.1,
        b = (from = 0.0, to = 0.0),
        rating = 10.0,
        angle_limits = (min = -1.5, max = 1.5),
    )
    PSY.add_component!(sys, line)
    switch = PSY.DiscreteControlledACBranch(nothing)
    switch.name = "stub_switch"
    switch.arc = arc
    switch.x = 0.05
    switch.r = 0.01
    switch.available = true
    switch.branch_status = PSY.DiscreteControlledBranchStatus.OPEN
    PSY.add_component!(sys, switch)
    return sys, switch
end

function _has_arc(ybus, arc)
    nrd = PNM.get_network_reduction_data(ybus)
    return haskey(PNM.get_direct_branch_map(nrd), arc) ||
           haskey(PNM.get_parallel_branch_map(nrd), arc)
end

_schedule_pinned(ybus) =
    PNM.get_schedule_pinned_buses(PNM.get_reductions(PNM.get_network_reduction_data(ybus)))

_arc_bus_numbers(branch) = (
    PSY.get_number(PSY.get_from(PSY.get_arc(branch))),
    PSY.get_number(PSY.get_to(PSY.get_arc(branch))),
)

@testset "closed_switches includes an OPEN switch in the base network" begin
    sys, switch, arc = _sys_with_open_switch()
    @test !_has_arc(PNM.Ybus(sys), arc)
    ybus = PNM.Ybus(sys; closed_switches = [switch])
    @test _has_arc(ybus, arc)

    sys_closed, switch_closed, _ = _sys_with_open_switch()
    switch_closed.branch_status = PSY.DiscreteControlledBranchStatus.CLOSED
    @test PNM.Ybus(sys_closed).data ≈ ybus.data
end

@testset "closed_switches never adds an unavailable switch" begin
    sys, switch, arc = _sys_with_open_switch()
    PSY.set_available!(switch, false)
    @test !_has_arc(PNM.Ybus(sys; closed_switches = [switch]), arc)
end

@testset "Zero-impedance switch kept for its schedule survives and logs at debug" begin
    # r = 0 with |y| above the threshold qualifies for zero-impedance merging without the
    # r = x = 0 substitution warning that assembly would emit.
    sys, switch, arc = _sys_with_open_switch(; x = 1e-6, r = 0.0)
    pins = Set(collect(arc))
    for kwargs in (
        (closed_switches = [switch],),
        (closed_switches = [switch], scheduled_branches = [switch]),
    )
        ybus = @test_logs min_level = Logging.Warn PNM.Ybus(sys; kwargs...)
        @test _has_arc(ybus, arc)
        @test _schedule_pinned(ybus) == pins

        @test_logs (:debug, r"between schedule-pinned buses") match_mode = :any min_level =
            Logging.Debug PNM.Ybus(sys; kwargs...)
    end

    switch.branch_status = PSY.DiscreteControlledBranchStatus.CLOSED
    @test_logs (:warn, r"Zero-impedance branch between two irreducible buses") match_mode =
        :any PNM.Ybus(sys; irreducible_buses = bus_number_selector(pins))
end

@testset "closed_switches rejects a switch that is not in the system" begin
    sys, _, _ = _sys_with_open_switch()
    _, foreign_switch, _ = _sys_with_open_switch()
    @test_throws IS.ConflictingInputsError PNM.Ybus(sys; closed_switches = [foreign_switch])
end

@testset "scheduled_branches rejects a branch that is not in the system" begin
    sys, _, _ = _sys_with_open_switch()
    foreign_sys, _, _ = _sys_with_open_switch()
    foreign_line = first(PSY.get_components(PSY.Line, foreign_sys))
    @test_throws IS.ConflictingInputsError PNM.Ybus(
        sys;
        scheduled_branches = [foreign_line],
    )
end

@testset "ReductionContainer records closed switches and protected buses apart from the user set" begin
    sys, switch, arc = _sys_with_open_switch()
    pins = Set(collect(arc))
    other = first(
        PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys) if
        !(PSY.get_number(b) in pins)
    )
    ybus = PNM.Ybus(
        sys;
        closed_switches = [switch],
        irreducible_buses = bus_number_selector([other]),
    )
    rc = PNM.get_reductions(PNM.get_network_reduction_data(ybus))
    @test PNM.get_closed_switches(rc) == Set(["tie_switch"])
    @test PNM.get_user_irreducible_buses(rc) == Set([other])
    @test PNM.get_schedule_pinned_buses(rc) == pins
    @test PNM.get_reduction_exempt_buses(rc) == union(pins, Set([other]))

    @test rc != PNM.ReductionContainer(;
        user_irreducible_buses = Set([other]),
        schedule_pinned_buses = pins,
    )
    empty!(rc)
    @test isempty(PNM.get_closed_switches(rc))
    @test isempty(PNM.get_schedule_pinned_buses(rc))
    @test rc == PNM.ReductionContainer()

    PSY.set_available!(switch, false)
    rc_unavailable = PNM.get_reductions(
        PNM.get_network_reduction_data(
            PNM.Ybus(sys; closed_switches = [switch], scheduled_branches = [switch]),
        ),
    )
    @test isempty(PNM.get_closed_switches(rc_unavailable))
    @test isempty(PNM.get_schedule_pinned_buses(rc_unavailable))
end

@testset "Reductions keep the buses of scheduled branches" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    for reduction in (RadialReduction(), DegreeTwoReduction())
        reduced = PNM.get_bus_axis(
            PNM.Ybus(sys; network_reductions = NetworkReduction[reduction]),
        )
        eliminated = setdiff(
            Set(PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)),
            Set(reduced),
        )
        @test !isempty(eliminated)
        branch = first(
            sort!(
                [
                    br for br in PSY.get_components(PSY.Line, sys) if
                    !isdisjoint(_arc_bus_numbers(br), eliminated)
                ];
                by = PSY.get_name,
            ),
        )
        ends = Set(_arc_bus_numbers(branch))
        ybus = PNM.Ybus(
            sys;
            network_reductions = NetworkReduction[reduction],
            scheduled_branches = [branch],
        )
        @test issubset(ends, PNM.get_bus_axis(ybus))
        rc = PNM.get_reductions(PNM.get_network_reduction_data(ybus))
        @test isempty(PNM.get_user_irreducible_buses(rc))
        @test PNM.get_schedule_pinned_buses(rc) == ends
    end
end

@testset "A switch closed by its schedule has a reduction record" begin
    sys, switch = _sys_with_open_stub_switch()
    reductions = NetworkReduction[RadialReduction()]
    open_vptdf = VirtualPTDF(sys; network_reductions = reductions)
    @test _STUB_BUS ∉ PNM.get_bus_axis(open_vptdf)
    @test branch_reduction(BranchReductionTable(open_vptdf, sys), IS.get_id(switch)) ==
          BRANCH_REDUCTION_NOT_IN_NETWORK

    vptdf = VirtualPTDF(sys; closed_switches = [switch], network_reductions = reductions)
    @test _STUB_BUS in PNM.get_bus_axis(vptdf)
    table = BranchReductionTable(vptdf, sys)
    @test branch_reduction(table, IS.get_id(switch)) == BRANCH_REDUCTION_HAS_ARC
end

@testset "A closed switch keeps its buses without being listed as scheduled" begin
    reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
    closed_sys, closed_switch = _sys_with_open_stub_switch()
    closed_switch.branch_status = PSY.DiscreteControlledBranchStatus.CLOSED
    @test _STUB_BUS ∉
          PNM.get_bus_axis(PNM.Ybus(closed_sys; network_reductions = reductions))

    sys, switch = _sys_with_open_stub_switch()
    ybus = PNM.Ybus(sys; closed_switches = [switch], network_reductions = reductions)
    buses = PNM.get_schedule_pinned_buses(
        PNM.get_reductions(PNM.get_network_reduction_data(ybus)),
    )
    from_bus, to_bus = _arc_bus_numbers(switch)
    @test from_bus in buses
    @test to_bus in buses
    @test issubset((from_bus, to_bus), PNM.get_bus_axis(ybus))
    @test _has_arc(ybus, (from_bus, to_bus))
end

@testset "A closed switch that leaves the branch maps without a record is an error" begin
    sys, switch, arc = _sys_with_open_switch()
    vptdf = VirtualPTDF(sys; closed_switches = [switch])
    @test branch_reduction(BranchReductionTable(vptdf, sys), IS.get_id(switch)) ==
          BRANCH_REDUCTION_HAS_ARC
    delete!(PNM.get_direct_branch_map(PNM.get_network_reduction_data(vptdf)), arc)
    @test_throws r"no branch map entry and no reduction record" BranchReductionTable(
        vptdf,
        sys,
    )
end
