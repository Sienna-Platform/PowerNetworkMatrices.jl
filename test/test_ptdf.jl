@testset "Test PTDF matrices, w/ and w/o tolerance" for solver in (
    "KLU",
    "Dense",
    "MKLPardiso",
    "AppleAccelerate",
)
    if !PowerNetworkMatrices._has_mkl_pardiso_ext() && solver == "MKLPardiso"
        @info "Skipped MKLPardiso tests (extension not loaded)"
        continue
    end
    if !PowerNetworkMatrices._has_apple_accelerate_ext() && solver == "AppleAccelerate"
        @info "Skipped AppleAccelerate tests (extension not loaded)"
        continue
    end
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses_5 = nodes5()
    branches_5 = branches5(buses_5)
    A = IncidenceMatrix(sys5)
    BA = BA_Matrix(sys5)
    P5 = PTDF(A, BA; linear_solver = solver)
    P5_sparse = PTDF(A, BA; linear_solver = solver, tol = 1e-3)

    # test method with branches and buses
    @test isapprox(maximum(P5.data), maximum(S5_slackB4), atol = 1e-3)
    @test isapprox(P5[PSY.get_arc(branches_5[1]), buses_5[1]], 0.1939166051164976)

    # additional test with other set of branches and buses
    buses_14 = nodes14()
    branches_14 = branches14(buses_14)
    sys_14 = System(100.0)
    for b in buses_14
        add_component!(sys_14, b)
    end
    for br in branches_14
        add_component!(sys_14, br)
    end
    P14 = PTDF(sys_14)
    @test isapprox(maximum(P14.data), maximum(S14_slackB1), atol = 1e-3)

    # check getindex (arc tuple and bus number)
    P5NS = PTDF(sys5)
    for (br, arc) in zip(Br5NS_ids, Arc5NS_ids), b in Bu5NS_ids
        @test isapprox(
            getindex(P5NS, arc, b),
            S5_slackB4[br, b],
            atol = 1e-3,
        )
    end

    # check getindex (iterated according to rows and columns)
    P5NS_mod = zeros(size(S5_slackB4))
    for (br, arc) in zip(Br5NS_ids, Arc5NS_ids), b in Bu5NS_ids
        # get the correct indices
        row = P5NS.lookup[2][arc]
        col = P5NS.lookup[1][b]
        P5NS_mod[row, col] = S5_slackB4[br, b]
        @test isapprox(
            getindex(P5NS, row, col),
            S5_slackB4[br, b],
            atol = 1e-3,
        )
    end

    # adjust the matrix to have same indexing
    @test isapprox(get_ptdf_data(P5NS), P5NS_mod, atol = 1e-3)

    PRTS = PTDF(RTS; linear_solver = solver)
    PRTS_sparse = PTDF(RTS; linear_solver = solver, tol = 1e-3)
    bnums = sort([PSY.get_number(b) for b in PSY.get_components(Bus, RTS)])
    for (ibr, br) in enumerate(RTS_branchnames), (ib, b) in enumerate(bnums)
        @test isapprox(getindex(PRTS, br, b), SRTS_GMLC[ibr, ib], atol = 1e-3)
    end

    # manually sparsify the matrix
    PRTS.data[abs.(PRTS.data) .< 1e-3] .= 0
    @test sum(abs.(get_ptdf_data(PRTS) - get_ptdf_data(PRTS_sparse)) .> 1e-3) == 0
    #end

    # check axes values
    P5 = PTDF(sys5)
    arc_tuples = [PNM.get_arc_tuple(arc) for arc in get_components(Arc, sys5)]
    @test setdiff(PNM.get_arc_axis(P5), arc_tuples) ==
          Tuple{Int, Int}[]
    @test setdiff(
        PNM.get_bus_axis(P5),
        PSY.get_number.(PSY.get_available_components(PSY.ACBus, sys5)),
    ) == String[]

    # auxiliary function
    PRTS_sparse = PTDF(RTS; tol = 1e-3)
    @test PNM.get_tol(PRTS_sparse).x == Base.RefValue(1e-3).x
end

@testset "Test PTDF matrices for 10 bus system with 2 reference buses" begin
    sys = PSB.build_system(PSISystems, "2Area 5 Bus System")   # get the system composed by 2 5-bus ones connected by a DC line
    ptdf_complete_klu = PTDF(sys; linear_solver = "KLU")
    ptdf_complete_dense = PTDF(sys; linear_solver = "Dense")
    IncidenceMatrix(sys)
    @test sum(ptdf_complete_klu.data - ptdf_complete_dense.data) < 1e-9
    @test isapprox(ptdf_complete_klu.data, ptdf_complete_dense.data, atol = 1e-6)

    # check submatrices: system has identical areas connected by a single hvdc, areas must have the same numbers for corresponding buses and arcs
    corresponding_buses = [(1, 6), (2, 7), (3, 8), (4, 9), (5, 10)]
    corresponding_arcs = [
        ((1, 2), (6, 7)),
        ((4, 5), (9, 10)),
        ((1, 5), (6, 10)),
        ((1, 4), (6, 9)),
        ((3, 4), (8, 9)),
        ((2, 3), (7, 8)),
    ]
    for bus_pair in corresponding_buses, arc_pair in corresponding_arcs
        @test isapprox(
            ptdf_complete_klu[arc_pair[1], bus_pair[1]],
            ptdf_complete_klu[arc_pair[2], bus_pair[2]],
            atol = 1e-6,
        )
    end
end

@testset "Test serialization of PTDF matrices to HDF5" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    P5 = PTDF(sys5; linear_solver = "KLU")
    P5_sparse = PTDF(sys5; linear_solver = "KLU", tol = 1e-3)
    for ptdf in (P5, P5_sparse)
        for compress in (true, false)
            path = mktempdir()
            filename = joinpath(path, "ptdf.h5")
            @test !isfile(filename)
            to_hdf5(ptdf, filename; compress = compress)
            @test isfile(filename)
            ptdf2 = PTDF(filename)
            @test ptdf == ptdf2
        end
    end
end

@testset "Test System with isolated buses" begin
    sys_1 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    PSY.add_component!(
        sys_1,
        PSY.ACBus(;
            number = 6,
            name = "isolated_node_1",
            available = true,
            bustype = PSY.ACBusTypes.ISOLATED,
            angle = 0.0,
            magnitude = 1.1,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        ),
    )
    PSY.add_component!(
        sys_1,
        PSY.ACBus(;
            number = 7,
            name = "isolated_node_2",
            available = true,
            bustype = PSY.ACBusTypes.ISOLATED,
            angle = 0.0,
            magnitude = 1.1,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        ),
    )
    ptdf_1 = PTDF(sys_1)
    # Test that the isolated buses are not included in the PTDF matrix
    @test length(axes(ptdf_1)[1]) == 5

    sys_2 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    branch_2 = PSY.get_component(PSY.Line, sys_2, "2")

    PSY.add_component!(
        sys_2,
        PSY.ACBus(;
            number = 6,
            name = "isolated_node_1",
            available = true,
            bustype = PSY.ACBusTypes.ISOLATED,
            angle = 0.0,
            magnitude = 1.1,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        ),
    )

    add_component!(
        sys_2,
        PSY.Line(;
            name = "7",
            available = branch_2.available,
            active_power_flow = branch_2.active_power_flow,
            reactive_power_flow = branch_2.reactive_power_flow,
            arc = PSY.Arc(;
                from = PSY.get_component(PSY.ACBus, sys_2, "nodeA"),
                to = PSY.get_component(PSY.ACBus, sys_2, "isolated_node_1"),
            ),
            r = branch_2.r,
            x = branch_2.x,
            b = branch_2.b,
            rating = get_rating(branch_2, Float64),
            angle_limits = get_angle_limits(branch_2),
        ),
    )

    # Test Throw error when isolated buses are connected to an available branch
    @test_throws IS.ConflictingInputsError ptdf_2 = PTDF(sys_2)
end

@testset "Test PTDF matrices with distributed slack" begin
    """
    CAUTION: this test just test that all the matrices are the same, but there
    are no reference values.
    """

    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    buscount = length(PSY.get_available_components(PSY.ACBus, sys5))

    dist_slack = 1 / buscount * ones(buscount)
    slack_array = Dict(i => dist_slack[i] / sum(dist_slack) for i in 1:buscount)

    P5_1 = PTDF(sys5; dist_slack = slack_array, linear_solver = "KLU")
    P5_2 = PTDF(sys5; dist_slack = slack_array, linear_solver = "Dense")
    @test isapprox(P5_1.data, P5_2.data, atol = 1e-5)
    if PowerNetworkMatrices._has_mkl_pardiso_ext()
        P5_3 = PTDF(sys5; dist_slack = slack_array, linear_solver = "MKLPardiso")
        @test isapprox(P5_2.data, P5_3.data, atol = 1e-5)
        @test isapprox(P5_1.data, P5_3.data, atol = 1e-5)
    end
end

@testset "Test PTDF matrix with distributed bus and with 2 reference buses" begin

    # check if an error is correctly thrown

    # 2 reference bus system
    sys = PSB.build_system(PSISystems, "2Area 5 Bus System")
    buscount = length(PSY.get_available_components(PSY.ACBus, sys))
    dist_slack = 1 / buscount * ones(buscount)
    slack_array = Dict(i => dist_slack[i] / sum(dist_slack) for i in 1:buscount)

    @test_throws ErrorException ptdf_1 =
        PTDF(sys; dist_slack = slack_array)

    # incorrect dist_slack array length
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buscount = length(PSY.get_available_components(PSY.ACBus, sys5)) + 1
    dist_slack = 1 / buscount * ones(buscount)
    slack_array = Dict(i => dist_slack[i] / sum(dist_slack) for i in 1:buscount)

    @test_throws IS.InvalidValue ptdf_2 = PTDF(sys5; dist_slack = slack_array)
end
