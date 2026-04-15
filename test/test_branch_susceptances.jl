@testset "Branch susceptances by arc" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)

    branch_susceptances = vlodf.branch_susceptances_by_arc
    @test length(branch_susceptances) == size(vlodf, 1)

    for (j, bs) in enumerate(branch_susceptances)
        if !isempty(bs)
            @test isapprox(sum(bs), vlodf.arc_susceptances[j], atol = 1e-10)
        end
        if length(bs) == 1
            @test isapprox(bs[1], vlodf.arc_susceptances[j], atol = 1e-10)
        end
    end
end

@testset "VirtualMODF has branch susceptances by arc" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    branch_susceptances = vmodf.branch_susceptances_by_arc
    n_arcs = length(PNM.get_arc_axis(vmodf))
    @test length(branch_susceptances) == n_arcs

    for (j, bs) in enumerate(branch_susceptances)
        if !isempty(bs)
            @test isapprox(sum(bs), vmodf.arc_susceptances[j], atol = 1e-10)
        end
    end
end
