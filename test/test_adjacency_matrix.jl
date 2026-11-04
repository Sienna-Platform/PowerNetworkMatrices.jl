@testset "Test connected components from AdjacencyMatrix" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    M = AdjacencyMatrix(sys5)
    subnetworks = find_subnetworks(M)
    @test length(subnetworks) == 1

    sys10 = PSB.build_system(PSISystems, "2Area 5 Bus System")
    M = AdjacencyMatrix(sys10)
    subnetworks_m = find_subnetworks(M)
    @test length(subnetworks_m) == 2
    @test issubset([6, 1], keys(subnetworks_m))

    subnetworks_sys = find_subnetworks(sys10)
    @test issubset([4, 9], keys(subnetworks_sys))
end
