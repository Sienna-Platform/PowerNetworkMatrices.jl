@testset "Test Virtual LODF matrices" begin
    sys = PSB.build_system(
        PSB.PSYTestSystems,
        "tamu_ACTIVSg2000_sys",
    )
    vlodf = VirtualLODF(sys)
    LODF_ref = LODF(sys)

    # data
    for (idx, name) in enumerate(vlodf.axes[1])
        # compare
        @test isapprox(vlodf[name, :], LODF_ref[idx, :], atol = 1e-6)
    end

    # check dicionary with rows
    data_dict = get_lodf_data(vlodf)
    for i in 1:length(vlodf.axes[1])
        # compare
        @test isapprox(data_dict[i], LODF_ref[i, :], atol = 1e-6)
    end

    # check the getindex function works properly
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf5 = VirtualLODF(sys5)
    ix_to_arc_map = Dict(
        ix => PNM.get_arc_tuple(get_component(ACBranch, sys5, br_name)) for
        (ix, br_name) in enumerate(Lodf_5_branch_axis)
    )
    for i in 1:size(vlodf5, 1)
        for j in 1:size(vlodf5, 2)
            # get the data
            @test isapprox(
                vlodf5[ix_to_arc_map[i], ix_to_arc_map[j]],
                Lodf_5[i, j],
                atol = 1e-3,
            )
        end
    end
end

@testset "Test Virtual LODF functions" begin
    # get system
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses_5 = nodes5()
    branches_5 = branches5(buses_5)
    vlodf = VirtualLODF(sys5)
    # properties
    @test size(vlodf) == (length(branches_5), length(branches_5))
    @test length(PNM.get_arc_axis(vlodf)) == length(branches_5)
end

@testset "Test Virtual LODF matrices with tolerance" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    lodf_reference = deepcopy(Lodf_14)
    ix_to_arc_map = Dict()
    for (ix, l) in enumerate(Lodf_14_branch_axis)
        ix_to_arc_map[ix] = PNM.get_arc_tuple(get_component(ACBranch, sys, l))
    end
    lodf_reference[abs.(lodf_reference) .<= 1e-2] .= 0
    lodf_virtual_with_tol = VirtualLODF(sys; tol = 1e-2)
    for ix in 1:size(lodf_reference)[1]
        row_map = [ix_to_arc_map[x] for x in 1:size(lodf_reference)[2]]
        row_pnm = [lodf_virtual_with_tol[ix_to_arc_map[ix], x] for x in row_map]
        row_ref = lodf_reference[ix, :]
        @test isapprox(row_pnm, row_ref, atol = 1e-3)
    end

    @test isapprox(
        sum(abs.(lodf_reference[1, :])),
        sum(abs.(lodf_virtual_with_tol[ix_to_arc_map[1], :])),
        atol = 1e-5)
end

@testset "Test Virtual LODF matrices for 10 bus system with 2 reference buses" begin
    # get system
    sys = PSB.build_system(PSISystems, "2Area 5 Bus System")   # get the system composed by 2 5-bus ones connected by a DC line
    # get PTDF matrix with KLU as reference
    lodf_complete = LODF(sys; linear_solver = "KLU")
    # check VirtualPTDF rows with the ones from KLU
    lodf_virtual = VirtualLODF(sys)
    for i in axes(lodf_complete, 2)
        comp = lodf_complete[i, :]
        virtual = lodf_virtual[i, :]
        # check values using PTDFs axes
        @test isapprox(comp, virtual; atol = 1e-10)
    end

    # check submatrices: since connected by a single bus, areas must have the same numbers
    corresponding_arcs = [
        ((1, 2), (6, 7)),
        ((4, 5), (9, 10)),
        ((1, 5), (6, 10)),
        ((1, 4), (6, 9)),
        ((3, 4), (8, 9)),
        ((2, 3), (7, 8)),
    ]
    for arc_pair_a in corresponding_arcs, arc_pair_b in corresponding_arcs
        @test isapprox(
            lodf_virtual[arc_pair_a[1], arc_pair_b[1]],
            lodf_virtual[arc_pair_a[2], arc_pair_b[2]],
            atol = 1e-6,
        )
    end
end

@testset "Test virtual LODF cache" begin
    RTS = build_system(PSITestSystems, "test_RTS_GMLC_sys")
    #Get Arcs from ACTransmission components (arc for HVDC is not included in matrices)
    arc_tuples =
        unique([PNM.get_arc_tuple(br) for br in get_components(ACTransmission, RTS)])
    persist_arcs = arc_tuples[1:10]

    vlodf = VirtualLODF(RTS; max_cache_size = 1, persistent_arcs = persist_arcs)

    for l in arc_tuples
        @test size(vlodf[l, :]) == (108,)
    end

    for l in persist_arcs
        @test vlodf.lookup[1][l] ∈ keys(vlodf.cache.temp_cache)
    end
end

@testset "Test Virtual LODF auxiliary functions" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    # test isempty when VirtualLODF is created (cache must be empty)
    vlodf = VirtualLODF(sys)
    @test isempty(vlodf) == true

    # test eachindex and axes
    @test length(eachindex(vlodf)) ==
          length(axes(vlodf)[1]) * length(axes(vlodf)[2])

    # check if error is correctly thrown
    @test_throws ErrorException vlodf[1, 1] = 1

    # test show
    test_value = false
    try
        show(vlodf)
        test_value = true
    catch err
        if err isa Exception
            test_value = false
        end
    end
    @test test_value
end

@testset "Test Virtual LODF with Apple Accelerate" begin
    if !PNM._has_apple_accelerate_backend()
        @info "Skipped AppleAccelerate VirtualLODF tests (backend unavailable on this platform)"
    else
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")

        vlodf_aa = VirtualLODF(sys; linear_solver = "AppleAccelerate")
        vlodf_klu = VirtualLODF(sys; linear_solver = "KLU")

        @test contains(string(typeof(vlodf_aa.K)), "AAFactorCache")
        @test vlodf_klu.K isa PNM.KLULinSolveCache{Float64}

        arc_axis = PNM.get_arc_axis(vlodf_aa)
        @test arc_axis == PNM.get_arc_axis(vlodf_klu)
        for arc in arc_axis
            row_aa = vlodf_aa[arc, :]
            row_klu = vlodf_klu[arc, :]
            @test isapprox(row_aa, row_klu, atol = 1e-9)
        end

        # macOS default should resolve to AppleAccelerate.
        vlodf_default = VirtualLODF(sys)
        @test contains(string(typeof(vlodf_default.K)), "AAFactorCache")
    end
end

@testset "_get_PTDF_A_diag: matches reference implementation" begin
    # Reference: full-bus dot product. Slow, unambiguously correct.
    # Captures the return value of `_solve_factorization` to stay
    # backend-agnostic (mirrors the fast path).
    function _reference_ptdf_a_diag(K, BA, A, ref_bus_positions::Set{Int})
        n_branches = size(BA, 2)
        n_buses = size(BA, 1)
        valid_ix = setdiff(1:n_buses, ref_bus_positions)
        n_valid = length(valid_ix)
        diag_ = zeros(n_branches)
        ba_col = zeros(n_valid)
        ptdf_row = zeros(n_buses)
        for i in 1:n_branches
            fill!(ba_col, 0.0)
            for idx in 1:n_valid
                bus_idx = valid_ix[idx]
                ba_col[idx] = BA[bus_idx, i]
            end
            lin_solve = PNM._solve_factorization(K, ba_col)
            fill!(ptdf_row, 0.0)
            for idx in 1:n_valid
                ptdf_row[valid_ix[idx]] = lin_solve[idx]
            end
            for j in 1:n_buses
                diag_[i] += ptdf_row[j] * A[i, j]
            end
        end
        return diag_
    end

    backends = String["KLU"]
    if PNM._has_apple_accelerate_backend()
        push!(backends, "AppleAccelerate")
    end

    for case in ("c_sys5", "c_sys14")
        sys = PSB.build_system(PSB.PSITestSystems, case)
        A = PNM.IncidenceMatrix(sys)
        BA = PNM.BA_Matrix(sys)
        ref_pos = Set(PNM.get_ref_bus_position(A))
        ABA = PNM.calculate_ABA_matrix(A.data, BA.data, ref_pos)

        for solver_name in backends
            solver = PNM.resolve_linear_solver(solver_name)
            K = PNM._create_factorization(solver, ABA)
            valid_ix = setdiff(1:size(BA.data, 1), ref_pos)
            b2v = PNM._build_bus_to_valid_idx(size(BA.data, 1), valid_ix)
            fast = PNM._get_PTDF_A_diag(K, BA.data, A.data, valid_ix, b2v)
            ref = _reference_ptdf_a_diag(K, BA.data, A.data, ref_pos)
            @test fast ≈ ref atol = 1e-12 rtol = 0
        end
    end
end
