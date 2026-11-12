@testset "Test VirtualPTDF with radial lines" begin
    # get the system
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    # get the PTDF matrix for reference
    ptdf_rad = PTDF(sys; network_reductions = NetworkReduction[RadialReduction()])
    vptdf_rad = VirtualPTDF(sys; network_reductions = NetworkReduction[RadialReduction()])

    for i in axes(ptdf_rad, 2)
        virtual = vptdf_rad[i, :]
        for j in axes(ptdf_rad, 1)
            # check values using PTDFs axes
            @test isapprox(ptdf_rad[i, j], vptdf_rad[i, j]; atol = 1e-10)
        end
    end
    # Check the cache is populated
    @test length(vptdf_rad.cache) == length(vptdf_rad.axes[1])
    @test length(vptdf_rad.cache[1]) == length(vptdf_rad.axes[2])
end

@testset "Test VirtualPTDF with radial lines and distributed slack" begin
    # check if VirtualPTDF have same values as PTDF row-wise
    # sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")   # ! remove
    buscount = length(PSY.get_available_components(PSY.ACBus, sys))
    dist_slack_factor = 1 / buscount #* ones(buscount)
    dist_slack = Dict{Int, Float64}()
    for i in 1:buscount
        dist_slack[i] = dist_slack_factor
    end
    ptdf_rad = PTDF(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        dist_slack = dist_slack,
    )
    vptdf_rad = VirtualPTDF(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        dist_slack = dist_slack,
    )
    for i in axes(ptdf_rad, 2)
        virtual = vptdf_rad[i, :]
        for j in axes(ptdf_rad, 1)
            # check values using PTDFs axes
            @test isapprox(ptdf_rad[i, j], vptdf_rad[i, j]; atol = 1e-10)
        end
    end
end
