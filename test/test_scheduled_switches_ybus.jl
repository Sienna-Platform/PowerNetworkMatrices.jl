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

function _has_arc(ybus, arc)
    nrd = PNM.get_network_reduction_data(ybus)
    return haskey(PNM.get_direct_branch_map(nrd), arc) ||
           haskey(PNM.get_parallel_branch_map(nrd), arc)
end

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

@testset "Schedule-pinned zero-impedance switch survives and logs at debug" begin
    # r = 0 with |y| above the threshold qualifies for zero-impedance merging without the
    # r = x = 0 substitution warning that assembly would emit.
    sys, switch, arc = _sys_with_open_switch(; x = 1e-6, r = 0.0)
    pins = Set(collect(arc))
    ybus = @test_logs min_level = Logging.Warn PNM.Ybus(
        sys;
        closed_switches = [switch],
        schedule_pinned_buses = pins,
    )
    @test _has_arc(ybus, arc)
    @test PNM.get_schedule_pinned_buses(
        PNM.get_reductions(PNM.get_network_reduction_data(ybus)),
    ) == pins

    @test_logs (:debug, r"between schedule-pinned buses") match_mode = :any min_level =
        Logging.Debug PNM.Ybus(
        sys;
        closed_switches = [switch],
        schedule_pinned_buses = pins,
    )

    @test_logs (:warn, r"Zero-impedance branch between two irreducible buses") match_mode =
        :any PNM.Ybus(
        sys;
        closed_switches = [switch],
        irreducible_buses = bus_number_selector(pins),
    )
end

@testset "closed_switches rejects a switch that is not in the system" begin
    sys, _, _ = _sys_with_open_switch()
    _, foreign_switch, _ = _sys_with_open_switch()
    @test_throws IS.ConflictingInputsError PNM.Ybus(sys; closed_switches = [foreign_switch])
end

@testset "schedule_pinned_buses rejects a bus number that is not in the system" begin
    sys, _, arc = _sys_with_open_switch()
    @test_throws IS.ConflictingInputsError PNM.Ybus(
        sys;
        schedule_pinned_buses = Set([arc[1], 99_999]),
    )
end

@testset "ReductionContainer records closed switches and pins apart from the user set" begin
    sys, switch, arc = _sys_with_open_switch()
    pins = Set(collect(arc))
    other = first(
        PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys) if
        !(PSY.get_number(b) in pins)
    )
    ybus = PNM.Ybus(
        sys;
        closed_switches = [switch],
        schedule_pinned_buses = pins,
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
    @test rc == PNM.ReductionContainer()

    PSY.set_available!(switch, false)
    rc_unavailable = PNM.get_reductions(
        PNM.get_network_reduction_data(PNM.Ybus(sys; closed_switches = [switch])),
    )
    @test isempty(PNM.get_closed_switches(rc_unavailable))
end

@testset "Reductions keep schedule-pinned buses" begin
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
        pinned = minimum(eliminated)
        ybus = PNM.Ybus(
            sys;
            network_reductions = NetworkReduction[reduction],
            schedule_pinned_buses = Set([pinned]),
        )
        @test pinned in PNM.get_bus_axis(ybus)
        rc = PNM.get_reductions(PNM.get_network_reduction_data(ybus))
        @test isempty(PNM.get_user_irreducible_buses(rc))
        @test PNM.get_schedule_pinned_buses(rc) == Set([pinned])
    end
end
