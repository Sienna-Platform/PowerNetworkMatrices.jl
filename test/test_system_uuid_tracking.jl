@testset "System UUID tracking in VirtualPTDF" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys_uuid = IS.get_uuid(sys)

    # UUID is stored when constructing from system
    vptdf = VirtualPTDF(sys)
    @test get_system_uuid(vptdf) == sys_uuid

    # UUID is nothing when constructing from Ybus (no system available)
    ybus = Ybus(sys)
    vptdf_ybus = VirtualPTDF(ybus)
    @test get_system_uuid(vptdf_ybus) === nothing
end

@testset "System UUID tracking in VirtualMODF" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys_uuid = IS.get_uuid(sys)

    vmodf = VirtualMODF(sys)
    @test get_system_uuid(vmodf) == sys_uuid
end

@testset "Default get_system_uuid returns nothing" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    ybus = Ybus(sys)
    @test get_system_uuid(ybus) === nothing

    ptdf = PTDF(sys)
    @test get_system_uuid(ptdf) === nothing
end

@testset "System UUID validation in NetworkModification" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    sys14 = PSB.build_system(PSB.PSITestSystems, "c_sys14")

    # VirtualPTDF built from sys5
    vptdf = VirtualPTDF(sys5)

    # Validation passes for matching system
    PNM._validate_system_uuid(vptdf, sys5)

    # Validation fails for mismatched system
    @test_throws ErrorException PNM._validate_system_uuid(vptdf, sys14)

    # Validation is a no-op when matrix does not track UUID
    ybus = Ybus(sys5)
    PNM._validate_system_uuid(ybus, sys14)  # should not throw
end
