function _uniform_dist_slack(sys)
    buses = [PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)]
    return Dict(b => 1.0 / length(buses) for b in buses)
end

@testset "Post-modification rows honor distributed slack" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    dist = _uniform_dist_slack(sys)
    vptdf = VirtualPTDF(sys; dist_slack = dist)
    line = first(sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name))
    wf = compute_woodbury_factors(vptdf, NetworkModification(vptdf, line))

    sys_ref = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    PSY.set_available!(PSY.get_component(PSY.Line, sys_ref, PSY.get_name(line)), false)
    ptdf_ref = PTDF(sys_ref; dist_slack = _uniform_dist_slack(sys_ref))
    ref_arc_lookup = PNM.get_arc_lookup(ptdf_ref)
    ref_bus_lookup = PNM.get_bus_lookup(ptdf_ref)
    for (m, arc) in enumerate(PNM.get_arc_axis(vptdf))
        haskey(ref_arc_lookup, arc) || continue
        row = apply_woodbury_correction(vptdf, m, wf)
        for (b, bus) in enumerate(PNM.get_bus_axis(vptdf))
            @test isapprox(
                row[b],
                ptdf_ref[ref_arc_lookup[arc], ref_bus_lookup[bus]];
                atol = 1e-6,
            )
        end
    end
end

@testset "Distributed slack with an islanding modification errors" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys; dist_slack = _uniform_dist_slack(sys))
    islanding_found = false
    for line in sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
        mod = NetworkModification(vptdf, line)
        isempty(mod.arc_modifications) && continue
        wf = compute_woodbury_factors(vptdf, mod)
        wf.is_islanding || continue
        @test_throws ErrorException apply_woodbury_correction(vptdf, 1, wf)
        islanding_found = true
        break
    end
    @test islanding_found
end
