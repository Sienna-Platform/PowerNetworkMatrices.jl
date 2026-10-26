@testset "PlannedOutage is not registered as a contingency" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    unplanned = PSY.GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 10.0,
        outage_transition_probability = 0.9,
        monitored_components = lines,
    )
    planned = PSY.PlannedOutage(; outage_schedule = "maintenance")
    PSY.add_supplemental_attribute!(sys, lines[1], unplanned)
    PSY.add_supplemental_attribute!(sys, lines[2], planned)

    vmodf = VirtualMODF(sys)
    registered = PNM.get_registered_contingencies(vmodf)
    @test haskey(registered, IS.get_id(unplanned))
    @test !haskey(registered, IS.get_id(planned))
end
