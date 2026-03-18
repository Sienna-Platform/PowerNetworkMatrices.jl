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

@testset "VirtualMODF: getindex by ContingencySpec" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    # Register a manual contingency
    e = 1
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid = Base.UUID(UInt128(999))
    ctg = ContingencySpec(ctg_uuid, "test_outage", [BranchModification(e, -b_e)])
    vmodf.contingency_cache[ctg_uuid] = ctg

    # Query by integer monitored index + ContingencySpec
    # Row length equals the number of buses in VirtualMODF's bus axis (non-reference buses)
    n_buses = length(vmodf.axes[2])
    row1 = vmodf[1, ctg]
    @test length(row1) == n_buses

    # Second query should hit cache
    row2 = vmodf[1, ctg]
    @test row1 == row2

    # Different monitored arc, same contingency — should reuse Woodbury factors
    row3 = vmodf[2, ctg]
    @test length(row3) == n_buses
    @test row3 != row1  # Different monitored arcs give different rows
end

@testset "VirtualMODF: getindex by arc tuple" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    e = 1
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid = Base.UUID(UInt128(998))
    ctg = ContingencySpec(ctg_uuid, "test_outage_tuple", [BranchModification(e, -b_e)])
    vmodf.contingency_cache[ctg_uuid] = ctg

    # Query using arc tuple
    arc_tuple = vmodf.axes[1][1]
    row = vmodf[arc_tuple, ctg]
    # Tuple indexing should give the same result as integer indexing
    row_by_idx = vmodf[1, ctg]
    @test length(row) == length(row_by_idx)
    @test isapprox(row, row_by_idx, atol = 1e-14)
end

@testset "VirtualMODF: setindex! throws error" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)
    @test_throws ErrorException vmodf[1, 1] = 1.0
end
