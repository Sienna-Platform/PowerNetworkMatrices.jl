@testset "Test powerflow matrix types" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    @testset "DC_ABA_Matrix_Factorized" begin
        M = ABA_Matrix(sys; factorize = true)
        @test M isa PNM.DC_ABA_Matrix_Factorized
        # PowerFlows uses the alias as an invariant type parameter of PowerFlowData.
        @test isconcretetype(PNM.DC_ABA_Matrix_Factorized)
    end

    @testset "DC_ABA_Matrix_Unfactorized" begin
        M = ABA_Matrix(sys; factorize = false)
        @test M isa PNM.DC_ABA_Matrix_Unfactorized
    end

    @testset "DC_PTDF_Matrix" begin
        if PNM._has_apple_accelerate_backend()
            @test PTDF(sys; linear_solver = "AppleAccelerate") isa PNM.DC_PTDF_Matrix
        end
        @test PTDF(sys; linear_solver = "KLU") isa PNM.DC_PTDF_Matrix
        @test PTDF(sys; linear_solver = "Dense") isa PNM.DC_PTDF_Matrix
    end

    @testset "DC_vPTDF_Matrix" begin
        if PNM._has_apple_accelerate_backend()
            @test VirtualPTDF(sys; linear_solver = "AppleAccelerate") isa
                  PNM.DC_vPTDF_Matrix
        end
        @test VirtualPTDF(sys; linear_solver = "KLU") isa PNM.DC_vPTDF_Matrix
    end

    @testset "DC_BA_Matrix" begin
        M = BA_Matrix(sys)
        @test M isa PNM.DC_BA_Matrix
    end

    @testset "AC_Ybus_Matrix" begin
        M = Ybus(sys)
        @test M isa PNM.AC_Ybus_Matrix
    end
end
