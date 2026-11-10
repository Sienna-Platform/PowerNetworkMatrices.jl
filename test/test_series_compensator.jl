# Disabled until the SeriesCompensator `bypassed` schema reaches PowerSystems psy6.
#=
# A series capacitor inserted into one line of the 5-bus case: line `from -> mid`, then the
# series compensator `mid -> to`. DC flows must match the line with the capacitor folded into it.

"""The 5-bus case with `COMPENSATED_LINE`'s reactance changed by `dx`."""
function folded_sys5(dx)
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    line = get_component(Line, sys, COMPENSATED_LINE)
    set_x!(line, (get_x(line, u"SU") + dx) * u"SU")
    return sys, line
end

"""Whether `a` and `b` give the same PTDF entry for every arc of `reference` but the
compensated one, and every bus of `reference`. `a`'s merged buses are read through its
reduction."""
function same_other_flows(a, b, reference, compensated)
    merged = PNM.get_reverse_bus_search_map(PNM.get_network_reduction_data(a))
    at(bus) = get(merged, bus, bus)
    buses = get_number.(get_components(ACBus, reference))
    arcs = filter(
        !=(compensated),
        [PNM.get_arc_tuple(l) for l in get_components(Line, reference)],
    )
    return all(
        isapprox(a[(at(arc[1]), at(arc[2])), at(bus)], b[arc, bus]; atol = 1e-9) for
        arc in arcs, bus in buses
    )
end

@testset "A series compensator acts as its reactance in series with the line" begin
    reference = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    x_c = -0.4 * get_x(get_component(Line, reference, COMPENSATED_LINE), u"SU")
    compensated, line = compensated_sys5(x_c)
    folded, folded_line = folded_sys5(x_c)

    compensator = get_component(SeriesCompensator, compensated, "compensator")
    branch = PNM.equivalent_branch(compensator)
    @test branch.equivalent_r == 0.0
    @test branch.equivalent_x == x_c

    P_split, P_folded = PTDF(compensated), PTDF(folded)
    @test same_other_flows(P_split, P_folded, folded, PNM.get_arc_tuple(folded_line))
    # The line now ends at the mid bus; it carries what the folded line carries.
    split_arc, folded_arc = PNM.get_arc_tuple(line), PNM.get_arc_tuple(folded_line)
    for bus in get_number.(get_components(ACBus, folded))
        @test P_split[split_arc, bus] ≈ P_folded[folded_arc, bus] atol = 1e-9
    end
    @test LODF(compensated) isa LODF
end

@testset "A bypassed series compensator is a zero-impedance tie" begin
    reference = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ref_line = get_component(Line, reference, COMPENSATED_LINE)
    line_x = get_x(ref_line, u"SU")
    compensated, _ = compensated_sys5(-0.4 * line_x; bypassed = true)

    compensator = get_component(SeriesCompensator, compensated, "compensator")
    @test PNM.equivalent_branch(compensator).equivalent_x == PNM.ZERO_IMPEDANCE_X_EPSILON
    ybus = Ybus(compensated)
    # The mid bus merges into a compensator end, so the network is the reference one.
    @test size(ybus.data) == size(Ybus(reference).data)
    @test same_other_flows(
        PTDF(compensated), PTDF(reference), reference, PNM.get_arc_tuple(ref_line),
    )
end
=#
