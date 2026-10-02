@testset "Test LODF matrices" begin
    """
    NOTE: LODF is transposed
    """

    # get 5 bus system
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses_5 = nodes5()
    branches_5 = branches5(buses_5)

    # get LODF with system
    ix_to_tuple_map =
        Dict(1 => (1, 2), 2 => (1, 4), 3 => (1, 5), 4 => (2, 3), 5 => (3, 4), 6 => (4, 5))
    L5NS = LODF(sys5)
    @test getindex(L5NS, (3, 4), (4, 5)) - -0.3071 <= 1e-4
    @test isapprox(maximum(L5NS.data), maximum(Lodf_5), atol = 1e-3)
    @test isapprox(
        L5NS[PSY.get_name(branches_5[1]), PSY.get_name(branches_5[2])],
        0.3447946513849093,
    )
    total_error = 0.0
    for ix in 1:6, jx in 1:6
        total_error += abs(L5NS[ix_to_tuple_map[ix], ix_to_tuple_map[jx]] .- Lodf_5[ix, jx])
    end
    @test isapprox(sum(total_error), 0.0, atol = 1e-3)

    # A and PTDF case
    A = IncidenceMatrix(sys5)
    P5 = PTDF(sys5)
    L5NS_from_ptdf = LODF(A, P5)
    L5NS_from_ptdf2 = LODF(A, P5; linear_solver = "Dense")
    @test getindex(L5NS_from_ptdf, "5", "6") - -0.3071 <= 1e-4
    @test getindex(L5NS_from_ptdf2, "5", "6") - -0.3071 <= 1e-4
    total_error = 0.0
    total_error2 = 0.0
    for ix in 1:6, jx in 1:6
        total_error +=
            abs(L5NS_from_ptdf[ix_to_tuple_map[ix], ix_to_tuple_map[jx]] .- Lodf_5[ix, jx])
        total_error2 +=
            abs(L5NS_from_ptdf2[ix_to_tuple_map[ix], ix_to_tuple_map[jx]] .- Lodf_5[ix, jx])
    end
    @test isapprox(sum(total_error), 0.0, atol = 1e-3)
    @test isapprox(sum(total_error2), 0.0, atol = 1e-3)

    if PowerNetworkMatrices._has_mkl_pardiso_ext()
        L5NS_from_ptdf3 = LODF(A, P5; linear_solver = "MKLPardiso")
        @test getindex(L5NS_from_ptdf3, "5", "6") - -0.3071 <= 1e-4
        total_error3 = 0.0
        for ix in 1:6, jx in 1:6
            total_error3 +=
                abs(
                    L5NS_from_ptdf3[ix_to_tuple_map[ix], ix_to_tuple_map[jx]] .-
                    Lodf_5[ix, jx],
                )
        end
        @test isapprox(sum(total_error3), 0.0, atol = 1e-3)
    end

    if PowerNetworkMatrices._has_apple_accelerate_backend()
        L5NS_from_ptdf4 = LODF(A, P5; linear_solver = "AppleAccelerate")
        @test getindex(L5NS_from_ptdf4, "5", "6") - -0.3071 <= 1e-4
        total_error4 = 0.0
        for ix in 1:6, jx in 1:6
            total_error4 +=
                abs(
                    L5NS_from_ptdf4[ix_to_tuple_map[ix], ix_to_tuple_map[jx]] .-
                    Lodf_5[ix, jx],
                )
        end
        @test isapprox(sum(total_error4), 0.0, atol = 1e-3)
    end

    # A, ABA, and BA case
    ABA = ABA_Matrix(sys5; factorize = true)
    BA = BA_Matrix(sys5)
    L5NS_from_ba_aba = LODF(A, ABA, BA)
    @test getindex(L5NS_from_ba_aba, "5", "6") - -0.3071 <= 1e-4
    total_error = 0.0
    for ix in 1:6, jx in 1:6
        total_error += abs(
            L5NS_from_ba_aba[ix_to_tuple_map[ix], ix_to_tuple_map[jx]] .- Lodf_5[ix, jx],
        )
    end
    @test isapprox(sum(total_error), 0.0, atol = 1e-3)

    for i in axes(Lodf_5, 1)
        @test isapprox(
            [L5NS[ix_to_tuple_map[i], ix_to_tuple_map[x]] for x in 1:6],
            Lodf_5[i, :],
            atol = 1e-3,
        )
        @test isapprox(
            [L5NS_from_ptdf[ix_to_tuple_map[i], ix_to_tuple_map[x]] for x in 1:6],
            Lodf_5[i, :],
            atol = 1e-3,
        )
        @test isapprox(
            [L5NS_from_ba_aba[ix_to_tuple_map[i], ix_to_tuple_map[x]] for x in 1:6],
            Lodf_5[i, :],
            atol = 1e-3,
        )
    end

    # test if error is thrown in case other linear solvers are called
    @test_throws ErrorException LODF(A, ABA, BA; linear_solver = "Dense")
    @test_throws r"needs a factorized ABA" LODF(A, ABA_Matrix(sys5), BA)

    @test_throws ErrorException LODF(A, P5; linear_solver = "XXX")
    @test_throws ErrorException LODF(sys5; linear_solver = "XXX")
    @test isapprox(
        LODF(sys5; linear_solver = "Dense", tol = eps()).data,
        LODF(sys5; linear_solver = "KLU", tol = eps()).data;
        atol = 1e-10,
    )

    # test if error is thrown in case `tol` is defined in PTDF
    P5 = PTDF(sys5; tol = 1e-3)
    @test_logs (
        :warn,
        "The argument `tol` in the PTDF matrix was set to a value different than the default one.\nThe resulting LODF can include unexpected rounding errors.\n",
    ) match_mode = :any LODF(A, P5)

    # get 14 bus system
    buses_14 = nodes14()
    branches_14 = branches14(buses_14)
    sys_14 = System(100.0)
    for b in buses_14
        add_component!(sys_14, b)
    end
    for br in branches_14
        add_component!(sys_14, br)
    end
    L14 = LODF(sys_14)
    @test isapprox(maximum(L14.data), maximum(Lodf_14), atol = 1e-3)
end

@testset "Test sparse LODF matrix" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ix_to_tuple_map =
        Dict(1 => (1, 2), 2 => (1, 4), 3 => (1, 5), 4 => (2, 3), 5 => (3, 4), 6 => (4, 5))

    # basic case: system
    L5NS_1 = LODF(sys5; tol = 0.4)

    # A and PTDF case
    a_matrix = IncidenceMatrix(sys5)
    ptdf_matrix = PTDF(sys5)
    L5NS_3 = LODF(a_matrix, ptdf_matrix; tol = 0.4)

    # A, ABA, and BA case
    aba_matrix = ABA_Matrix(sys5; factorize = true)
    ba_matrix = BA_Matrix(sys5)
    L5NS_4 = LODF(a_matrix, aba_matrix, ba_matrix; tol = 0.4)

    # reference value
    ref_sparse_Lodf_5 = deepcopy(Lodf_5')
    ref_sparse_Lodf_5[abs.(ref_sparse_Lodf_5) .< PNM.get_tol(L5NS_1)] .= 0

    L5NS_1_rearranged = zeros(size(L5NS_1.data))
    L5NS_3_rearranged = zeros(size(L5NS_3.data))
    L5NS_4_rearranged = zeros(size(L5NS_4.data))
    for ix in 1:size(L5NS_1_rearranged)[1], jx in 1:size(L5NS_1_rearranged)[2]
        L5NS_1_rearranged[ix, jx] = L5NS_1[ix_to_tuple_map[jx], ix_to_tuple_map[ix]]
        L5NS_3_rearranged[ix, jx] = L5NS_3[ix_to_tuple_map[jx], ix_to_tuple_map[ix]]
        L5NS_4_rearranged[ix, jx] = L5NS_4[ix_to_tuple_map[jx], ix_to_tuple_map[ix]]
    end
    # tests
    @test isapprox(L5NS_1_rearranged, ref_sparse_Lodf_5, atol = 1e-3)
    @test isapprox(L5NS_1_rearranged, L5NS_3_rearranged, atol = 1e-3)
    @test isapprox(L5NS_1_rearranged, L5NS_4_rearranged, atol = 1e-3)
end

@testset "Test LODF getindex and get_lodf_data" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")

    # get the LODF matrix
    lodf_ = LODF(sys)

    # test get_lodf_data
    lodf_t = lodf_.data'
    @test isapprox(lodf_t, get_lodf_data(lodf_))

    # test getindex
    for name1 in PNM.get_arc_axis(lodf_)       # selected line
        for name2 in PNM.get_arc_axis(lodf_)   # outage line
            i = lodf_.lookup[1][name1]
            j = lodf_.lookup[2][name2]
            element_1 = lodf_[name1, name2]
            element_2 = lodf_[i, j]
            element_3 = lodf_t[i, j]
            @test isapprox(element_1, element_2, atol = 1e-5)
            @test isapprox(element_1, element_3, atol = 1e-5)
            @test isapprox(element_2, element_3, atol = 1e-5)
        end
    end
end

@testset "LODF: negative-reactance outage matches the flow change it predicts" begin
    # A series capacitor drives H[e,e] = PTDF·A[e,e] above 1, so the LODF denominator
    # 1 - H[e,e] is negative rather than vanishing; it must not be clamped.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = PSY.get_component(Line, sys, "Line10")
    PSY.set_r!(line, 0.0 * PSY.SU)
    PSY.set_x!(line, -0.1 * PSY.SU)
    pre = PTDF(sys)
    lodf = LODF(sys)
    lodf_from_ptdf = LODF(IncidenceMatrix(sys), PTDF(sys; tol = eps()); tol = eps())
    vlodf = VirtualLODF(sys)
    e = PNM.get_arc_tuple(line, PNM.get_network_reduction_data(pre))
    @test PNM.get_PTDF_A_diag(PNM.get_core(vlodf))[PNM.get_arc_lookup(vlodf)[e]] > 1

    PSY.set_available!(line, false)
    post = PTDF(sys)
    PSY.set_available!(line, true)
    buses = PNM.get_bus_axis(pre)
    injection = randn(MersenneTwister(1), length(buses))
    injection .-= sum(injection) / length(injection)
    flow(ptdf, arc) = sum(ptdf[arc, bus] * injection[i] for (i, bus) in enumerate(buses))
    for l in PNM.get_arc_axis(post)
        expected = (flow(post, l) - flow(pre, l)) / flow(pre, e)
        @test isapprox(lodf[l, e], expected; atol = 1e-8)
        @test isapprox(lodf_from_ptdf[l, e], expected; atol = 1e-8)
        @test isapprox(vlodf[l, :][PNM.get_arc_lookup(vlodf)[e]], expected; atol = 1e-8)
    end

    # Partial outage: halving the susceptance is doubling the reactance.
    e_ix = PNM.get_arc_lookup(vlodf)[e]
    b_e = PNM._get_arc_susceptances(vlodf)[e_ix]
    full = PNM.get_partial_lodf_row(vlodf, e_ix, -b_e)
    half = PNM.get_partial_lodf_row(vlodf, e_ix, -b_e / 2)
    PSY.set_x!(line, -0.2 * PSY.SU)
    halved = PTDF(sys)
    for l in PNM.get_arc_axis(post)
        l_ix = PNM.get_arc_lookup(vlodf)[l]
        @test isapprox(full[l_ix], lodf[l, e]; atol = 1e-8)
        expected = (flow(halved, l) - flow(pre, l)) / flow(pre, e)
        @test isapprox(half[l_ix], expected; atol = 1e-8)
    end
end
