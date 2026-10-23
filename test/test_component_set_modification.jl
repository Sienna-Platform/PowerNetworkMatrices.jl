import Logging

function _rows_match_rebuild(vptdf, mod, sys_ref)
    wf = compute_woodbury_factors(vptdf, mod)
    ptdf_ref = PTDF(sys_ref)
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
    return
end

@testset "Component-set modification matches a rebuild" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    vptdf = VirtualPTDF(sys)
    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    chosen = lines[1:2]
    mod = NetworkModification(vptdf, sys, chosen)
    sys_ref = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for l in chosen
        PSY.set_available!(PSY.get_component(PSY.Line, sys_ref, PSY.get_name(l)), false)
    end
    _rows_match_rebuild(vptdf, mod, sys_ref)
end

@testset "Duplicates collapse instead of double-counting" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    vptdf = VirtualPTDF(sys)
    line = first(sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name))
    @test NetworkModification(vptdf, sys, [line, line]) ==
          NetworkModification(vptdf, sys, [line])
    merged = NetworkModification(
        "summed",
        vcat(
            collect(NetworkModification(vptdf, line).arc_modifications),
            collect(NetworkModification(vptdf, line).arc_modifications),
        ),
    )
    @test merged != NetworkModification(vptdf, sys, [line])
end

@testset "Series-chain segments resolve together" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nr = PNM.get_network_reduction_data(vptdf)
    chain = first(values(nr.series_branch_map))
    segments = [seg for seg in chain if seg isa PSY.ACTransmission]
    @assert length(segments) >= 2
    together = NetworkModification(vptdf, sys, segments[1:2])
    @test length(together.arc_modifications) == 1
    @test together.arc_modifications[1].delta_b ≈
          NetworkModification(vptdf, sys, segments[1:1]).arc_modifications[1].delta_b
end

@testset "Unavailable components are skipped" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    PSY.set_available!(lines[1], false)
    vptdf = VirtualPTDF(sys)
    mod = @test_logs min_level = Logging.Info NetworkModification(
        vptdf,
        sys,
        [lines[1], lines[2]],
    )
    @test mod == NetworkModification(vptdf, sys, [lines[2]])
    only_unavailable =
        @test_logs min_level = Logging.Info NetworkModification(vptdf, sys, [lines[1]])
    @test isempty(only_unavailable.arc_modifications)
    @test isempty(only_unavailable.shunt_modifications)
end
