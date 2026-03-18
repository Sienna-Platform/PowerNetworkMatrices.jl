@testset "VirtualMODF construction" begin
    # Test construction with a simple system (no outages)
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    # Should construct without error, but have no contingencies
    @test isempty(vmodf)
    @test length(get_registered_contingencies(vmodf)) == 0

    # Check basic properties
    n_arcs = length(PNM.get_arc_axis(vmodf))
    @test n_arcs > 0
    @test length(vmodf.arc_susceptances) == n_arcs
    @test length(vmodf.PTDF_A_diag) == n_arcs
end

@testset "VirtualMODF: N-1 post-contingency PTDF matches PTDF + LODF correction" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    ptdf_ref = PTDF(sys5)
    vmodf = VirtualMODF(sys5)

    n_arcs = size(vlodf, 1)

    for e in 1:n_arcs
        b_e = vmodf.arc_susceptances[e]
        ctg_uuid = Base.UUID(UInt128(e))
        ctg = ContingencySpec(ctg_uuid, "outage_arc_$e", [BranchModification(e, -b_e)])
        vmodf.contingency_cache[ctg_uuid] = ctg

        for m in 1:n_arcs
            # Post-contingency PTDF from VirtualMODF
            modf_row = PNM._compute_modf_entry(vmodf, m, ctg)

            # Expected: pre_ptdf[m,:] + LODF[m,e] * pre_ptdf[e,:]
            expected = ptdf_ref[m, :] .+ vlodf[m, e] .* ptdf_ref[e, :]
            @test isapprox(modf_row, expected, atol = 1e-6)
        end
        # Clean Woodbury cache between arcs to avoid stale entries
        empty!(vmodf.woodbury_cache)
    end
end
