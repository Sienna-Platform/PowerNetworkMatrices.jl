@testset "First tests for arc admittance matrices" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[DegreeTwoReduction()],
        make_arc_admittance_matrices = true,
    )
    A = IncidenceMatrix(ybus)
    yft = ybus.arc_admittance_from_to
    ytf = ybus.arc_admittance_to_from
    @test isa(ybus, Ybus)
    @test isa(yft, ArcAdmittanceMatrix)
    @test isa(ytf, ArcAdmittanceMatrix)
    @test size(A) == size(yft) == size(ytf)

    sys = build_system(PSISystems, "RTS_GMLC_DA_sys")
    y = Ybus(sys; make_arc_admittance_matrices = true)
    Yft = y.arc_admittance_to_from
    @test length(PNM.get_arc_lookup(Yft)) == size(Yft.data)[1]
    @test length(PNM.get_bus_lookup(Yft)) == size(Yft.data)[2]
end

@testset "grouped composite arc appears in the arc-admittance matrices" begin
    sys = build_reversed_asymmetric_degree_two_chains()
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[DegreeTwoReduction()],
        make_arc_admittance_matrices = true,
    )
    nrd = get_network_reduction_data(ybus)
    yft = ybus.arc_admittance_from_to
    ytf = ybus.arc_admittance_to_from

    # A grouped composite arc gets its own arc-admittance row, which is how PTDF/LODF/MODF
    # reach it; the arcs its chains consumed are gone from the axis.
    arc_ax = PNM.get_arc_axis(yft)
    @test (1, 3) in arc_ax
    for consumed in ((1, 10), (10, 11), (11, 3), (3, 20), (20, 21), (21, 1))
        @test consumed ∉ arc_ax
    end
    @test length(arc_ax) == size(yft.data, 1) == size(ytf.data, 1)

    # The row carries the group's own two-port, in the group's arc frame.
    group = PNM.get_parallel_branch_map(nrd)[(1, 3)]
    @test length(group) == 2
    Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(group, nrd)
    row = PNM.get_arc_lookup(yft)[(1, 3)]
    bus = PNM.get_bus_lookup(yft)
    @test isapprox(yft.data[row, bus[1]], Y11; rtol = 1e-4)
    @test isapprox(yft.data[row, bus[3]], Y12; rtol = 1e-4)
    @test isapprox(ytf.data[row, bus[3]], Y22; rtol = 1e-4)
    @test isapprox(ytf.data[row, bus[1]], Y21; rtol = 1e-4)
    # A transposed append would put Y22 where Y11 belongs; the fixture is asymmetric enough
    # for that to be distinguishable.
    @test abs(Y11 - Y22) > 0.1
end
