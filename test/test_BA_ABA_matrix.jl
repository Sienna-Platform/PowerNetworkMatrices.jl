@testset "Test A, BA, and ABA matrix creation" begin
    # test on 5 and 14 bus system, and RTS.
    for (category, name) in [
        (PSITestSystems, "c_sys5"),
        (PSITestSystems, "c_sys14"),
        (PSISystems, "RTS_GMLC_DA_sys"),
    ]
        sys = PSB.build_system(category, name)
        # at first let's see if factorization flag works
        ABA_no_lu = ABA_Matrix(sys)
        @test isnothing(ABA_no_lu.K)
        # check the is_factorized function
        @test is_factorized(ABA_no_lu) == false
        # factorize the ABA matrix
        ABA_no_lu = PNM.factorize(ABA_no_lu)
        @test is_factorized(ABA_no_lu) == true
        # get the ABA matrix with the current method
        ABA_lu = ABA_Matrix(sys; factorize = true)
        # check the is_factorized function
        @test is_factorized(ABA_no_lu) == true
        # evaluate if the ABA matrix is correct
        A = IncidenceMatrix(sys)
        BA = BA_Matrix(sys)
        ABA_2 = transpose(A.data) * BA.data'
        @test isapprox(
            ABA_lu.data,
            ABA_2[
                setdiff(1:end, PNM.get_ref_bus_position(A)),
                setdiff(1:end, PNM.get_ref_bus_position(A)),
            ],
            atol = 1e-8,
        )

        # evaluate if the LU factorization evaluates a correct PTDF matrix
        ptdf_1 = PTDF(sys)
        Ix = Matrix(1.0I, size(ABA_lu.data, 1), size(ABA_lu.data, 1))
        ABA_inv = ABA_lu.K \ Ix
        ptdf_2 = BA.data[setdiff(1:end, PNM.get_ref_bus_position(A)), :]' * ABA_inv
        @test isapprox(
            ptdf_1.data[setdiff(1:end, PNM.get_ref_bus_position(A)), :],
            ptdf_2',
            atol = 1e-8,
        )
    end
end

@testset "Test BA and ABA matrix indexing" begin
    sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")

    # get the matrices
    ba = BA_Matrix(sys)
    aba = ABA_Matrix(sys)

    # check if indexing for the BA is correct (row and column indices)
    # ba matrix is stored as transposed
    for i in 1:size(ba, 2)
        for j in 1:size(ba, 1)
            @test isapprox(ba[i, j], ba.data[j, i])
        end
    end
    # check if indexing for the BA is correct (line names and bus numbers)
    lookup1 = PNM.get_lookup(ba)
    for i in axes(ba, 2)
        for j in axes(ba, 1)
            @test isapprox(ba[i, j], ba.data[lookup1[1][j], lookup1[2][i]])
        end
    end

    # check indexing for the ABA matrix
    lookup2 = aba.lookup[1]
    for i in axes(aba, 1)
        @test aba[i, :] == aba.data[lookup2[i], :]
    end

    # test if error is correctly thrown when ref bus is called
    buses = get_components(PSY.ACBus, sys)
    rb = [bus for bus in buses if get_bustype(bus) == PSY.ACBusTypes.REF][1]
    @test_throws ErrorException aba[rb, :]
end

@testset "Test show for A, BA and ABA matrix" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    a = IncidenceMatrix(sys)
    ba = BA_Matrix(sys)
    aba = ABA_Matrix(sys)

    for mat in [a, ba, aba]
        test_value = false
        try
            show(@eval $mat)
            test_value = true
        catch err
            @error err
            test_value = false
        end
        @test test_value
    end
end
