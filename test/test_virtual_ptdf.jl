# if it fails, we don't want the terminal to be flooded with errors, therefore failfast=true
@testset "Test Virtual PTDF matrices" for solver in
                                          ("KLU", "AppleAccelerate")
    if !PowerNetworkMatrices._has_apple_accelerate_ext() && solver == "AppleAccelerate"
        @info "Skipped AppleAccelerate tests (extension not loaded)"
        continue
    end
    sys = PSB.build_system(
        PSB.PSYTestSystems,
        "tamu_ACTIVSg2000_sys",
    )
    ptdf_complete = PTDF(sys; linear_solver = solver)
    ptdf_virtual = VirtualPTDF(sys; linear_solver = solver)

    for i in axes(ptdf_complete, 2)
        virtual = ptdf_virtual[i, :]
        for j in axes(ptdf_complete, 1)
            # check values using PTDFs axes
            @test isapprox(ptdf_complete[i, j], ptdf_virtual[i, j]; atol = 1e-10)
        end
    end
    # Check the cache is populated
    @test length(ptdf_virtual.cache) == length(ptdf_virtual.axes[1])
end

@testset "Test Virtual PTDF matrices with tolerance" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    ptdf_reference = deepcopy(S14_slackB1)
    ix_to_arc_map = Dict()
    ix_to_bus_map = Dict()
    for (ix, l) in enumerate(S14_slackB1_branch_axis)
        ix_to_arc_map[ix] = PNM.get_arc_tuple(get_component(ACBranch, sys, l))
    end
    for (ix, b) in enumerate(S14_slackB1_bus_axis)
        ix_to_bus_map[ix] = b
    end
    ptdf_reference[abs.(ptdf_reference) .<= 1e-2] .= 0
    ptdf_virtual_with_tol = VirtualPTDF(sys; tol = 1e-2)
    for ix in 1:size(ptdf_reference)[1]
        row_map = [ix_to_bus_map[x] for x in 1:size(ptdf_reference)[2]]
        row_pnm = [ptdf_virtual_with_tol[ix_to_arc_map[ix], x] for x in row_map]
        row_ref = ptdf_reference[ix, :]
        @test isapprox(row_pnm, row_ref, atol = 1e-3)
    end
    @test isapprox(
        sum(abs.(ptdf_reference[17, :])),
        sum(abs.(ptdf_virtual_with_tol[ix_to_arc_map[17], :])),
        atol = 1e-4)
end

@testset "Test Virtual PTDF matrices for 10 bus system with 2 reference buses" begin
    # get system
    sys = PSB.build_system(PSISystems, "2Area 5 Bus System")   # get the system composed by 2 5-bus ones connected by a DC line
    # get PTDF matrix with KLU as reference
    ptdf_complete = PTDF(sys; linear_solver = "KLU")
    # check VirtualPTDF rows with the ones from KLU
    ptdf_virtual = VirtualPTDF(sys)
    for i in axes(ptdf_complete, 2)
        comp = ptdf_complete[i, :]
        virtual = ptdf_virtual[i, :]
        # check values using PTDFs axes
        @test isapprox(comp, virtual; atol = 1e-10)
    end

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
            ptdf_virtual[arc_pair[1], bus_pair[1]],
            ptdf_virtual[arc_pair[2], bus_pair[2]],
            atol = 1e-6,
        )
    end
end

@testset "Test Virtual PTDF cache" begin
    RTS = build_system(PSITestSystems, "test_RTS_GMLC_sys")
    #Get Arcs from ACTransmission components (arc for HVDC is not included in matrices)
    arc_tuples =
        unique([PNM.get_arc_tuple(br) for br in get_components(ACTransmission, RTS)])
    persist_arcs = arc_tuples[1:10]

    vptdf = VirtualPTDF(RTS; max_cache_size = 1, persistent_arcs = persist_arcs)
    for l in arc_tuples
        @test size(vptdf[l, :]) == (73,)
    end

    for l in persist_arcs
        @test vptdf.lookup[1][l] ∈ keys(vptdf.cache.temp_cache)
    end
end

@testset "Test Virtual PTDF with distributed slack" begin
    # get 5 bus system
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    buscount = length(PSY.get_available_components(PSY.ACBus, sys))
    dist_slack_factor = 1 / buscount
    dist_slack = Dict(i => dist_slack_factor for i in 1:buscount)
    # compute full PTDF
    ptdf = PTDF(sys; dist_slack = dist_slack)
    # compute each row of the virtual PTDF and compare values
    vptdf = VirtualPTDF(sys; dist_slack = dist_slack)
    for row in 1:size(ptdf.data, 2)
        # evaluate the column (just needs one element)
        vptdf[row, 1]
        @test isapprox(vptdf.cache[row], ptdf[row, :], atol = 1e-5)
    end
end

@testset "Test Virtual PTDF matrix with distributed bus and with 2 reference buses" begin
    # check if an error is correctly thrown
    # 2 reference bus system
    sys = PSB.build_system(PSISystems, "2Area 5 Bus System")
    buscount = length(PSY.get_available_components(PSY.ACBus, sys))
    dist_slack_factor = 1 / buscount
    dist_slack = Dict(i => dist_slack_factor for i in 1:buscount)
    ptdf_1 = VirtualPTDF(sys; dist_slack = dist_slack)
    @test_throws ErrorException ptdf_1[1, 1]

    # entry in dist_slack dict not belonging to system/matrix
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buscount = length(PSY.get_available_components(PSY.ACBus, sys5)) + 1
    dist_slack_factor = 1 / buscount
    dist_slack = Dict(i => dist_slack_factor for i in 1:buscount)
    @test_throws IS.InvalidValue VirtualPTDF(
        sys5;
        dist_slack = dist_slack,
    )
end

@testset "Test Virtual PTDF auxiliary functions" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    # test empty cache
    vptdf = VirtualPTDF(sys)
    @test isempty(vptdf)

    # test full cache
    vptdf[2, 1]
    @test !isempty(vptdf)

    # check if error is correctly thrown
    @test_throws ErrorException vptdf[1, 1] = 1

    # get the rows and full PTDF matrix, test get_ptdf_data
    ptdf = PTDF(sys)
    for i in PNM.get_arc_axis(vptdf)
        for j in PNM.get_bus_axis(vptdf)
            vptdf[i, j]
        end
    end
    dict_ = Dict()
    for (n, i) in enumerate(PNM.get_arc_axis(vptdf))
        dict_[n] = vptdf.cache[n]
    end
    @test get_ptdf_data(vptdf) == dict_

    # test get axes values
    arc_tuples = [PNM.get_arc_tuple(arc) for arc in get_components(Arc, sys)]
    @test setdiff(PNM.get_arc_axis(vptdf), arc_tuples) == []
    @test setdiff(
        PNM.get_bus_axis(vptdf),
        PSY.get_number.(PSY.get_available_components(ACBus, sys)),
    ) == String[]

    # test show
    test_value = false
    try
        show(vptdf)
        test_value = true
    catch err
        if err isa Exception
            test_value = false
        end
    end
    @test test_value
end

@testset "Test Virtual PTDF with Apple Accelerate" begin
    if !PowerNetworkMatrices._has_apple_accelerate_ext()
        @info "Skipped AppleAccelerate tests (extension not loaded)"
    else
        # Test VirtualPTDF with Apple Accelerate solver
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")

        # Create VirtualPTDF with AppleAccelerate
        vptdf_aa = VirtualPTDF(sys; linear_solver = "AppleAccelerate")

        # Verify the factorization type is AAFactorization
        @test contains(string(typeof(vptdf_aa.K)), "AAFactorization")

        # Create reference VirtualPTDF with KLU
        vptdf_klu = VirtualPTDF(sys; linear_solver = "KLU")

        # Compare results between AppleAccelerate and KLU
        for arc in PNM.get_arc_axis(vptdf_aa)[1:5]  # Test a subset of arcs
            row_aa = vptdf_aa[arc, :]
            row_klu = vptdf_klu[arc, :]
            @test isapprox(row_aa, row_klu, atol = 1e-10)
        end

        # Test with tolerance
        vptdf_aa_tol = VirtualPTDF(sys; linear_solver = "AppleAccelerate", tol = 1e-2)
        @test contains(string(typeof(vptdf_aa_tol.K)), "AAFactorization")

        # Test with distributed slack
        buscount = length(PSY.get_available_components(PSY.ACBus, sys))
        dist_slack_factor = 1 / buscount
        dist_slack = Dict(i => dist_slack_factor for i in 1:buscount)
        vptdf_aa_slack =
            VirtualPTDF(sys; linear_solver = "AppleAccelerate", dist_slack = dist_slack)
        @test contains(string(typeof(vptdf_aa_slack.K)), "AAFactorization")

        # Compare with KLU for distributed slack case
        vptdf_klu_slack = VirtualPTDF(sys; linear_solver = "KLU", dist_slack = dist_slack)
        test_arc = first(PNM.get_arc_axis(vptdf_aa_slack))
        vptdf_aa_slack[test_arc, 1]  # Trigger computation
        vptdf_klu_slack[test_arc, 1]  # Trigger computation
        @test isapprox(vptdf_aa_slack.cache[1], vptdf_klu_slack.cache[1], atol = 1e-10)

        # Test cache functionality with AppleAccelerate
        arc_tuples = [PNM.get_arc_tuple(br) for br in get_components(ACTransmission, sys)]
        persist_arcs = arc_tuples[1:min(5, length(arc_tuples))]
        vptdf_aa_cache = VirtualPTDF(
            sys;
            linear_solver = "AppleAccelerate",
            max_cache_size = 1,
            persistent_arcs = persist_arcs,
        )

        for l in persist_arcs
            @test size(vptdf_aa_cache[l, :]) == (length(PNM.get_bus_axis(vptdf_aa_cache)),)
            @test vptdf_aa_cache.lookup[1][l] ∈ keys(vptdf_aa_cache.cache.temp_cache)
        end
    end
end
