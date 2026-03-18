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

@testset "VirtualMODF: cache management" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    # Register a contingency and compute a row
    e = 1
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid = Base.UUID(UInt128(500))
    ctg = ContingencySpec(ctg_uuid, "cache_test", [BranchModification(e, -b_e)])
    vmodf.contingency_cache[ctg_uuid] = ctg

    _ = vmodf[1, ctg]  # Triggers computation + caching

    @test !isempty(vmodf.woodbury_cache)
    @test haskey(vmodf.row_caches, ctg_uuid)  # UUID key

    # clear_caches! should clear Woodbury and row caches but keep contingencies
    PNM.clear_caches!(vmodf)
    @test isempty(vmodf.woodbury_cache)
    @test isempty(vmodf.row_caches)
    @test !isempty(vmodf.contingency_cache)

    # clear_all_caches! should clear everything
    _ = vmodf[1, ctg]  # Recompute
    PNM.clear_all_caches!(vmodf)
    @test isempty(vmodf.contingency_cache)
    @test isempty(vmodf.woodbury_cache)
    @test isempty(vmodf.row_caches)
end

@testset "VirtualMODF: show and auxiliary functions" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    # Test show on empty vmodf
    io = IOBuffer()
    show(io, MIME"text/plain"(), vmodf)
    output_empty = String(take!(io))
    @test length(output_empty) > 0

    # Test show on non-empty vmodf (register a contingency)
    e = 1
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid_show = Base.UUID(UInt128(9999))
    ctg_show = ContingencySpec(ctg_uuid_show, "show_test", [BranchModification(e, -b_e)])
    vmodf.contingency_cache[ctg_uuid_show] = ctg_show
    io2 = IOBuffer()
    show(io2, MIME"text/plain"(), vmodf)
    output_nonempty = String(take!(io2))
    @test occursin("registered contingencies", output_nonempty)

    # Test size (n_arcs × n_buses)
    n_arcs = length(PNM.get_arc_axis(vmodf))
    n_buses = length(vmodf.axes[2])
    @test size(vmodf) == (n_arcs, n_buses)

    # Test isempty (c_sys5 has no outages; vmodf was built fresh above)
    vmodf_empty = VirtualMODF(sys5)
    @test isempty(vmodf_empty)
end

@testset "VirtualMODF: N-1 public getindex accuracy" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    ptdf_ref = PTDF(sys5)
    vmodf = VirtualMODF(sys5)

    n_arcs = size(vlodf, 1)
    n_buses = length(vmodf.axes[2])

    # Register a single-arc full outage
    e = 1
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid = Base.UUID(UInt128(8888))
    ctg = ContingencySpec(ctg_uuid, "public_api_test", [BranchModification(e, -b_e)])
    vmodf.contingency_cache[ctg_uuid] = ctg

    # Verify all monitored arcs through the public getindex API
    for m in 1:n_arcs
        row = vmodf[m, ctg]  # public API — goes through RowCache
        expected = ptdf_ref[m, :] .+ vlodf[m, e] .* ptdf_ref[e, :]
        @test length(row) == n_buses
        @test isapprox(row, expected, atol = 1e-6)
    end

    # Second query should hit cache and return identical values
    row_cached = vmodf[1, ctg]
    row_fresh = vmodf[1, ctg]
    @test row_cached == row_fresh
end

@testset "VirtualMODF: Woodbury cache reuse" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    e = 1
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid = Base.UUID(UInt128(700))
    ctg = ContingencySpec(ctg_uuid, "reuse_test", [BranchModification(e, -b_e)])
    vmodf.contingency_cache[ctg_uuid] = ctg

    # First query: computes Woodbury factors + row
    row1 = vmodf[1, ctg]
    @test haskey(vmodf.woodbury_cache, ctg_uuid)  # UUID key

    # Second query with different monitored arc: reuses Woodbury
    row2 = vmodf[2, ctg]
    # Both rows should be cached now
    @test haskey(vmodf.row_caches, ctg_uuid)  # UUID key
    cache = vmodf.row_caches[ctg_uuid]
    @test haskey(cache, 1)
    @test haskey(cache, 2)

    # Woodbury factors should be computed exactly once for this contingency
    @test length(vmodf.woodbury_cache) == 1
end
