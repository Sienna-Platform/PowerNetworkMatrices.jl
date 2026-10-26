function _sys_with_schedules()
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses = sort!(collect(PSY.get_components(PSY.ACBus, sys)); by = PSY.get_number)
    arc = PSY.Arc(buses[1], buses[3])
    PSY.add_component!(sys, arc)
    switch = PSY.DiscreteControlledACBranch(nothing)
    switch.name = "tie_switch"
    switch.arc = arc
    switch.x = 0.05
    switch.available = true
    switch.branch_status = PSY.DiscreteControlledBranchStatus.OPEN
    PSY.add_component!(sys, switch)
    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    PSY.add_supplemental_attribute!(
        sys,
        switch,
        PSY.SwitchingSchedule(; status_schedule = "plan"),
    )
    planned = PSY.PlannedOutage(; outage_schedule = "maint")
    PSY.add_supplemental_attribute!(sys, lines[1], planned)
    PSY.add_supplemental_attribute!(sys, lines[2], planned)
    return sys, switch, lines
end

_arc_numbers(b) = (
    PSY.get_number(PSY.get_from(PSY.get_arc(b))),
    PSY.get_number(PSY.get_to(PSY.get_arc(b))),
)

@testset "Scheduled components, switches and pins" begin
    sys, switch, lines = _sys_with_schedules()
    r = topology_schedule_components(sys)
    @test Set(PSY.get_name.(r.components)) ==
          Set([PSY.get_name(switch), PSY.get_name(lines[1]), PSY.get_name(lines[2])])
    @test length(r.components) == 3
    @test r.closed_switches == [switch]
    @test r.pinned_buses == union(
        Set(_arc_numbers(switch)),
        Set(_arc_numbers(lines[1])),
        Set(_arc_numbers(lines[2])),
    )
end

@testset "A switch with both schedule types is still closed" begin
    sys, switch, lines = _sys_with_schedules()
    PSY.add_supplemental_attribute!(
        sys,
        switch,
        PSY.PlannedOutage(; outage_schedule = "switch_maint"),
    )
    r = topology_schedule_components(sys)
    @test r.closed_switches == [switch]
    @test count(c -> c === switch, r.components) == 1
end

@testset "Unavailable scheduled components are excluded" begin
    sys, switch, lines = _sys_with_schedules()
    PSY.set_available!(switch, false)
    PSY.set_available!(lines[1], false)
    r = topology_schedule_components(sys)
    @test PSY.get_name.(r.components) == [PSY.get_name(lines[2])]
    @test isempty(r.closed_switches)
    @test r.pinned_buses == Set(_arc_numbers(lines[2]))
end

@testset "A planned outage on a generator pins nothing and does not error" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    gen = first(PSY.get_components(PSY.ThermalStandard, sys))
    PSY.add_supplemental_attribute!(sys, gen, PSY.PlannedOutage(; outage_schedule = "g"))
    r = topology_schedule_components(sys)
    @test r.components == [gen]
    @test isempty(r.pinned_buses)
end

@testset "No schedules means nothing changes" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    r = topology_schedule_components(sys)
    @test isempty(r.components)
    @test isempty(r.closed_switches)
    @test isempty(r.pinned_buses)
    ybus = PNM.Ybus(
        sys;
        closed_switches = r.closed_switches,
        schedule_pinned_buses = r.pinned_buses,
    )
    @test ybus.data == PNM.Ybus(sys).data
end

@testset "A scheduled ThreeWindingTransformer pins its terminals and its star bus" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    transformer = only(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    PSY.add_supplemental_attribute!(
        sys,
        transformer,
        PSY.PlannedOutage(; outage_schedule = "t3w"),
    )
    r = topology_schedule_components(sys)
    @test r.components == [transformer]
    @test isempty(r.closed_switches)
    star = PSY.get_number(PSY.get_star_bus(transformer))
    terminals = Set(
        PSY.get_number(PSY.get_from(PSY.get_arc(c))) for c in PSY.get_circuits(transformer)
    )
    @test length(terminals) == 3
    @test r.pinned_buses == union(terminals, Set([star]))
end

@testset "A scheduled two-terminal HVDC line pins nothing" begin
    sys = build_hvdc_with_single_bus_island()
    hvdc = only(PSY.get_components(PSY.TwoTerminalGenericHVDCLine, sys))
    PSY.add_supplemental_attribute!(sys, hvdc, PSY.PlannedOutage(; outage_schedule = "dc"))
    r = topology_schedule_components(sys)
    @test r.components == [hvdc]
    @test isempty(r.closed_switches)
    @test isempty(r.pinned_buses)
end
