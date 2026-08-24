using Random

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
        ctg_id = e
        ctg = ContingencySpec(
            ctg_id,
            NetworkModification("outage_arc_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_id] = ctg

        for m in 1:n_arcs
            # Post-contingency PTDF from VirtualMODF
            modf_row = PNM._compute_modf_entry(vmodf, m, ctg.modification)

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
    ctg_id = 999
    ctg = ContingencySpec(
        ctg_id,
        NetworkModification("test_outage", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[ctg_id] = ctg

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
    ctg_id = 998
    ctg = ContingencySpec(
        ctg_id,
        NetworkModification("test_outage_tuple", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[ctg_id] = ctg

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
    ctg_id = 500
    ctg = ContingencySpec(
        ctg_id,
        NetworkModification("cache_test", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[ctg_id] = ctg

    _ = vmodf[1, ctg]  # Triggers computation + caching

    @test !isempty(vmodf.woodbury_cache)
    @test !isempty(vmodf.row_caches)

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
    ctg_uuid_show = 9999
    ctg_show = ContingencySpec(
        ctg_uuid_show,
        NetworkModification("show_test", [ArcModification(e, -b_e)]),
    )
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
    ctg_id = 8888
    ctg = ContingencySpec(
        ctg_id,
        NetworkModification("public_api_test", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[ctg_id] = ctg

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
    ctg_id = 700
    ctg = ContingencySpec(
        ctg_id,
        NetworkModification("reuse_test", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[ctg_id] = ctg

    # First query: computes Woodbury factors + row
    row1 = vmodf[1, ctg]
    @test !isempty(vmodf.woodbury_cache)

    # Second query with different monitored arc: reuses Woodbury
    row2 = vmodf[2, ctg]
    # Both rows should be cached now
    @test !isempty(vmodf.row_caches)
    cache = vmodf.row_caches[ctg.modification]
    @test haskey(cache, 1)
    @test haskey(cache, 2)

    # Woodbury factors should be computed exactly once for this contingency
    @test length(vmodf.woodbury_cache) == 1
end
@testset "Compare MODF entries with and without degree-2 reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    valid_outage_branches = get_available_components(
        x -> !(
            typeof(x) <: Union{ThreeWindingTransformer, DiscreteControlledACBranch} ||
            _is_phase_shifting_2w(x)
        ),
        ACTransmission,
        sys,
    )
    for branch in valid_outage_branches
        outage = GeometricDistributionForcedOutage(;
            mean_time_to_recovery = 0.0,
            outage_transition_probability = 0.0,
        )
        add_supplemental_attribute!(sys, branch, outage)
    end
    vmodf = VirtualMODF(sys)
    bus_lookup = PNM.get_bus_lookup(vmodf)
    nrd = vmodf.network_reduction_data
    vmodf_d2 = VirtualMODF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    bus_lookup_d2 = PNM.get_bus_lookup(vmodf_d2)
    nrd_d2 = vmodf_d2.network_reduction_data
    # Compare results for arcs that are present in the reduced system.
    arcs_to_compare = vcat(
        collect(keys(nrd_d2.direct_branch_map)),
        collect(keys(nrd_d2.parallel_branch_map)),
    )
    # Compare results for buses that are present in the reduced system
    buses_to_compare = collect(keys(nrd_d2.bus_reduction_map))
    for branch in valid_outage_branches
        outage = get_supplemental_attributes(branch)[1]
        ctg_id = IS.get_uuid(outage)
        # Skip branches not registered as contingencies in either system
        haskey(get_registered_contingencies(vmodf), ctg_id) || continue
        haskey(get_registered_contingencies(vmodf_d2), ctg_id) || continue
        ctg = get_registered_contingencies(vmodf)[ctg_id]
        ctg_d2 = get_registered_contingencies(vmodf_d2)[ctg_id]
        for arc in arcs_to_compare
            ix_arc = PNM.get_arc_lookup(vmodf)[arc]
            ix_arc_d2 = PNM.get_arc_lookup(vmodf_d2)[arc]
            row_d2 = vmodf_d2[ix_arc_d2, ctg_d2]
            row = vmodf[ix_arc, ctg]
            for bus in buses_to_compare
                ix_bus = PNM.get_bus_index(bus, bus_lookup, nrd)
                ix_bus_d2 = PNM.get_bus_index(bus, bus_lookup_d2, nrd_d2)
                @test isapprox(row_d2[ix_bus_d2], row[ix_bus], atol = 1e-6)
            end
        end
    end
end

@testset "VirtualMODF: BA/A sign convention consistency (issue #278)" begin
    # Regression test for Sienna-Platform/PowerNetworkMatrices.jl#278.
    #
    # The issue is an inconsistent branch orientation between BA[:, m] and
    # A[m, :]: BA[:, m] can correspond to -b_m * A[m, :] even when b_m itself
    # is not negative. The old code used A[m, :] directly for the ν_m⊤·Z term
    # in the Woodbury correction, producing incorrect post-contingency PTDF rows.
    # The fix derives the orientation from BA[:, m] / b instead of A[m, :],
    # so _compute_modf_row no longer depends on reading A.
    #
    # To test: flip A[m, :] for a monitored arc to simulate the opposite
    # orientation. Since the fix only uses BA / b (not A) in
    # _compute_modf_row, the result must still match the PTDF+LODF identity.
    # If A were still used, this flip would cause errors.
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    ptdf_ref = PTDF(sys5)
    vlodf = VirtualLODF(sys5)
    vmodf = VirtualMODF(sys5)
    n_arcs = size(vlodf, 1)

    # Flip A[flip_arc, :] to simulate the opposite sign convention.
    # A is CSC (arcs × buses), so iterate over columns to find entries in the row.
    flip_arc = 2
    A_rows = SparseArrays.rowvals(vmodf.A)
    A_vals = SparseArrays.nonzeros(vmodf.A)
    for col in 1:size(vmodf.A, 2)
        for idx in SparseArrays.nzrange(vmodf.A, col)
            if A_rows[idx] == flip_arc
                A_vals[idx] *= Int8(-1)
            end
        end
    end

    # Confirm A[flip_arc,:] now has opposite sign from BA[:,flip_arc]/b
    b_flip = vmodf.arc_susceptances[flip_arc]
    for bus_idx in vmodf.valid_ix
        ba_val = vmodf.BA[bus_idx, flip_arc] / b_flip
        a_val = Float64(vmodf.A[flip_arc, bus_idx])
        if a_val != 0
            @test sign(ba_val) != sign(a_val)
        end
    end

    # Trip a different arc as contingency and monitor the flipped arc.
    # The Woodbury factors for the contingency arc are unaffected by the A flip
    # (K_mat only reads AZ at contingency arc indices, not at flip_arc).
    # With the fix, _compute_modf_row uses BA/b for zm_Z, so the flipped A
    # is never read → result must match the PTDF+LODF identity.
    for e in 1:n_arcs
        e == flip_arc && continue
        b_e = vmodf.arc_susceptances[e]
        ctg_id = 30000 + e
        ctg = ContingencySpec(
            ctg_id,
            NetworkModification("sign_ctg_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_id] = ctg

        modf_row = PNM._compute_modf_entry(vmodf, flip_arc, ctg.modification)
        expected = ptdf_ref[flip_arc, :] .+ vlodf[flip_arc, e] .* ptdf_ref[e, :]
        @test isapprox(modf_row, expected, atol = 1e-6)
        empty!(vmodf.woodbury_cache)
    end
end

@testset "test delta b positive for registered outages" begin
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    for branch in get_components(ACTransmission, sys)
        _is_phase_shifting_2w(branch) && continue
        outage = GeometricDistributionForcedOutage(;
            mean_time_to_recovery = 0.0,
            outage_transition_probability = 0.0,
        )
        add_supplemental_attribute!(sys, branch, outage)
    end
    vmodf = VirtualMODF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    for branch in get_components(ACTransmission, sys)
        !has_supplemental_attributes(branch) && continue
        outage = get_supplemental_attributes(branch)[1]
        ctg_id = IS.get_uuid(outage)
        ctg = get_registered_contingencies(vmodf)[ctg_id]
        @test ctg.modification.arc_modifications[1].delta_b <= 0.0
    end
end

# Helper for the parallel-safety testsets below: registers a fixed set of line
# outages on c_sys14 and returns (sys, line_names).
function _build_c_sys14_with_outages()
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line_names = ["Line1", "Line2", "Line9", "Line10", "Line12"]
    for line_name in line_names
        line = PSY.get_component(PSY.ACTransmission, sys, line_name)
        outage = PSY.GeometricDistributionForcedOutage(;
            mean_time_to_recovery = 10,
            outage_transition_probability = 0.9999,
        )
        PSY.add_supplemental_attribute!(sys, line, outage)
    end
    return sys, line_names
end

@testset "VirtualMODF concurrent getindex across different contingencies matches serial baseline" begin
    # Mirrors the access pattern PowerSimulations uses in
    # `add_post_contingency_flow_expressions!`: many concurrent tasks query
    # `vmodf[arc, contingency_spec]` across DIFFERENT contingencies. With the
    # single-cache solver + `_LIBKLU_LOCK`, all libklu work serializes; this
    # test confirms the result is still correct under that serialization,
    # including the double-checked-insert path on `woodbury_cache` /
    # `row_caches` when two threads race on a first-time query.
    # Skipped under JULIA_NUM_THREADS=1 because @threads :dynamic degenerates
    # to serial there and the test reduces to a tautology.
    if Threads.nthreads() < 2
        @info "Skipping: requires Threads.nthreads() ≥ 2 to exercise concurrent getindex."
        return
    end

    sys, _ = _build_c_sys14_with_outages()
    vmodf = PowerNetworkMatrices.VirtualMODF(sys)
    registered = PowerNetworkMatrices.get_registered_contingencies(vmodf)
    @test !isempty(registered)
    ctgs = collect(values(registered))
    arc_axis = PowerNetworkMatrices.get_arc_axis(vmodf)
    @test !isempty(arc_axis)

    # Cap arc count so test runtime stays bounded if c_sys14's arc list grows
    # in the future. With 5 contingencies, 20 arcs gives 100 work items per
    # iteration — enough parallelism on a 4-core CI without dominating runtime.
    n_arcs = min(length(arc_axis), 20)
    work = [(arc, ctg) for arc in arc_axis[1:n_arcs] for ctg in ctgs]
    Random.shuffle!(Random.MersenneTwister(42), work)

    # Serial baseline.
    serial = [copy(vmodf[a, c]) for (a, c) in work]

    # Five iterations of parallel access, each starting from a clean cache,
    # to flush scheduling-dependent races.
    for iter in 1:5
        PowerNetworkMatrices.clear_caches!(vmodf)
        parallel = Vector{Vector{Float64}}(undef, length(work))
        Threads.@threads :dynamic for i in eachindex(work)
            a, c = work[i]
            parallel[i] = copy(vmodf[a, c])
        end
        for i in eachindex(work)
            @test parallel[i] ≈ serial[i]
        end
    end
end

@testset "VirtualMODF with Apple Accelerate backend matches KLU" begin
    if !PNM._has_apple_accelerate_backend()
        @info "Skipped AppleAccelerate VirtualMODF tests (backend unavailable on this platform)"
    else
        sys, _ = _build_c_sys14_with_outages()

        vmodf_aa = VirtualMODF(sys; linear_solver = "AppleAccelerate")
        vmodf_klu = VirtualMODF(sys; linear_solver = "KLU")

        # Factorization should be the AA cache type.
        @test contains(string(typeof(vmodf_aa.K)), "AAFactorCache")
        @test vmodf_klu.K isa PNM.KLULinSolveCache{Float64}

        registered_aa = get_registered_contingencies(vmodf_aa)
        registered_klu = get_registered_contingencies(vmodf_klu)
        @test !isempty(registered_aa)
        @test keys(registered_aa) == keys(registered_klu)

        # Compare post-contingency rows for every registered contingency
        # against the KLU build, sweeping all monitored arcs.
        arc_axis = PNM.get_arc_axis(vmodf_aa)
        @test arc_axis == PNM.get_arc_axis(vmodf_klu)
        for (uuid, ctg_aa) in registered_aa
            ctg_klu = registered_klu[uuid]
            for arc in arc_axis
                row_aa = vmodf_aa[arc, ctg_aa]
                row_klu = vmodf_klu[arc, ctg_klu]
                @test isapprox(row_aa, row_klu, atol = 1e-9)
            end
        end
    end
end

@testset "VirtualMODF concurrent getindex on the SAME (arc, ctg) is consistent" begin
    # Complements the previous testset: there, each (arc, ctg) pair appears
    # once in the work list, so only the `woodbury_cache` first-call race is
    # exercised. Here, many tasks race on the SAME (arc, ctg), which
    # exercises the row-cache population path serialized by `solver_lock`.
    if Threads.nthreads() < 2
        @info "Skipping: requires Threads.nthreads() ≥ 2 to exercise concurrent getindex."
        return
    end

    sys, _ = _build_c_sys14_with_outages()
    vmodf = PowerNetworkMatrices.VirtualMODF(sys)
    registered = PowerNetworkMatrices.get_registered_contingencies(vmodf)
    arc_axis = PowerNetworkMatrices.get_arc_axis(vmodf)
    arc = first(arc_axis)
    ctg = first(values(registered))

    serial_value = copy(vmodf[arc, ctg])

    # 64 tasks racing on one cache slot.
    n_tasks = 64
    for iter in 1:5
        PowerNetworkMatrices.clear_caches!(vmodf)
        parallel = Vector{Vector{Float64}}(undef, n_tasks)
        Threads.@threads :dynamic for i in 1:n_tasks
            parallel[i] = copy(vmodf[arc, ctg])
        end
        for i in 1:n_tasks
            @test parallel[i] ≈ serial_value
        end
    end
end

@testset "VirtualMODF: N-2 post-contingency PTDF matches N-2 LODF correction" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    ptdf_ref = PTDF(sys5)
    vmodf = VirtualMODF(sys5)

    n_arcs = size(vlodf, 1)

    # N-2 expected row from pre-contingency PTDF + LODF values.
    # Derived from Woodbury/Sherman-Morrison on the 2×2 susceptance update:
    #   denom = 1 - LODF[e1,e2]*LODF[e2,e1]
    #   eff1  = (LODF[m,e1] + LODF[e2,e1]*LODF[m,e2]) / denom
    #   eff2  = (LODF[e1,e2]*LODF[m,e1] + LODF[m,e2]) / denom
    function n2_lodf_expected(m, e1, e2)
        Lm1 = vlodf[m, e1]
        Lm2 = vlodf[m, e2]
        L12 = vlodf[e1, e2]
        L21 = vlodf[e2, e1]
        denom = 1 - L12 * L21
        eff1 = (Lm1 + L21 * Lm2) / denom
        eff2 = (L12 * Lm1 + Lm2) / denom
        return ptdf_ref[m, :] .+ eff1 .* ptdf_ref[e1, :] .+ eff2 .* ptdf_ref[e2, :]
    end

    # Bridge arcs (PTDF_A_diag ≈ 1) produce N-1 islanding; skip them.
    is_bridge(e) = abs(vmodf.PTDF_A_diag[e]) >= 1.0 - 1e-6

    # Two non-bridge arcs can still form an N-2 island (bridge pair) where
    # removing both disconnects the network. This is detected by
    # L12*L21 ≈ 1 (the 2×2 Woodbury denominator collapses to zero).
    # In that regime VirtualMODF uses a pseudoinverse while the closed-form
    # LODF formula is undefined; skip those pairs.
    n2_is_islanding(e1, e2) = abs(1 - vlodf[e1, e2] * vlodf[e2, e1]) < 1e-6

    ctg_counter = 10_000
    for e1 in 1:n_arcs
        is_bridge(e1) && continue
        for e2 in (e1 + 1):n_arcs
            is_bridge(e2) && continue
            n2_is_islanding(e1, e2) && continue

            b_e1 = vmodf.arc_susceptances[e1]
            b_e2 = vmodf.arc_susceptances[e2]

            ctg_id = ctg_counter
            ctg_counter += 1
            ctg = ContingencySpec(
                ctg_id,
                NetworkModification(
                    "n2_outage_$(e1)_$(e2)",
                    [ArcModification(e1, -b_e1), ArcModification(e2, -b_e2)],
                ),
            )
            vmodf.contingency_cache[ctg_id] = ctg

            for m in 1:n_arcs
                modf_row = PNM._compute_modf_entry(vmodf, m, ctg.modification)
                expected = n2_lodf_expected(m, e1, e2)
                @test isapprox(modf_row, expected; atol = 1e-6)
            end

            # Clear Woodbury cache between contingencies to avoid stale entries.
            empty!(vmodf.woodbury_cache)
        end
    end
end

@testset "VirtualMODF: PTDF_A_diag is lazy, logs on first access, caches" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)
    n_arcs = length(PNM.get_arc_axis(vmodf))

    # `PTDF_A_diag` is lazily stored on the shared core; read it directly off
    # the core's field, which bypasses the wrapper's `getproperty` hook under
    # test (the core is a plain struct, so `.PTDF_A_diag` is a direct field read).
    @test isempty(PNM.get_core(vmodf).PTDF_A_diag)

    diag1 = @test_logs (:info, r"Computing.*PTDF_A_diag.*first access") (
        :info, r"Computed.*PTDF_A_diag",
    ) vmodf.PTDF_A_diag
    @test length(diag1) == n_arcs
    @test !isempty(PNM.get_core(vmodf).PTDF_A_diag)

    # Second read is a cache hit: same identity, no further logs.
    diag2 = @test_logs min_level = Logging.Info vmodf.PTDF_A_diag
    @test diag2 === diag1
    @test PNM.get_PTDF_A_diag(vmodf) === diag1

    # Cross-check against VirtualLODF (which still computes eagerly).
    vlodf = VirtualLODF(sys5)
    @test diag1 ≈ vlodf.PTDF_A_diag atol = 1e-10
end

@testset "VirtualMODF: N-2 non-bridge islanding pair — connected subnetwork matches rebuilt PTDF" begin
    # Real-world flowgate scenario: two individually non-critical lines (neither is
    # a bridge on its own) whose simultaneous loss isolates bus 222 in the RTS-GMLC
    # network.  Lines "B34" (221–222) and "B30" (217–222) are the only connections
    # to bus 222; removing both makes it electrically isolated.
    #
    # The 2×2 Woodbury matrix is singular (L12·L21 ≈ 1), so VirtualMODF falls back
    # to its pseudoinverse path.  We verify correctness against a freshly built PTDF
    # with both lines disabled, skipping the isolated bus column because the
    # pseudoinverse does not force that column to zero.
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    vmodf = VirtualMODF(sys)

    arc_ax = PNM.get_arc_axis(vmodf)
    bus_ax = PNM.get_bus_axis(vmodf)
    arc_lookup = PNM.get_arc_lookup(vmodf)
    n_arcs = length(arc_ax)

    line_b34 = PSY.get_component(PSY.ACBranch, sys, "B34")
    line_b30 = PSY.get_component(PSY.ACBranch, sys, "B30")

    function arc_idx_for_line(line)
        arc = PSY.get_arc(line)
        return arc_lookup[(arc.from.number, arc.to.number)]
    end

    e1 = arc_idx_for_line(line_b34)  # arc (221, 222)
    e2 = arc_idx_for_line(line_b30)  # arc (217, 222)

    b_e1 = vmodf.arc_susceptances[e1]
    b_e2 = vmodf.arc_susceptances[e2]
    ctg_id = 88_000
    ctg = ContingencySpec(
        ctg_id,
        NetworkModification(
            "n2_island_B34_B30",
            [ArcModification(e1, -b_e1), ArcModification(e2, -b_e2)],
        ),
    )
    vmodf.contingency_cache[ctg_id] = ctg

    # Rebuild the PTDF with both lines disabled — gold-standard ground truth.
    sys_mod = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    for name in ("B34", "B30")
        PSY.set_available!(PSY.get_component(PSY.ACBranch, sys_mod, name), false)
    end
    ptdf_rebuilt = PTDF(sys_mod)
    rebuilt_arc_lookup = PNM.get_arc_lookup(ptdf_rebuilt)
    rebuilt_bus_lookup = PNM.get_bus_lookup(ptdf_rebuilt)

    # Connected buses: every bus appearing in at least one surviving arc.
    # Bus 222 is absent because both its lines are outaged.
    connected_buses = Set{Int}()
    for (fb, tb) in keys(rebuilt_arc_lookup)
        push!(connected_buses, fb)
        push!(connected_buses, tb)
    end

    for m in 1:n_arcs
        modf_row = vmodf[m, ctg]

        # Pseudoinverse must produce a finite result even for islanding events.
        @test all(isfinite, modf_row)

        monitored_arc = arc_ax[m]

        # Outaged arcs: full loss of flow sensitivity → zero row.
        if !haskey(rebuilt_arc_lookup, monitored_arc)
            @test all(abs.(modf_row) .< 1e-8)
            continue
        end

        # Surviving arcs: match rebuilt PTDF for every electrically connected bus.
        rebuilt_m = rebuilt_arc_lookup[monitored_arc]
        for (b_idx, bus_num) in enumerate(bus_ax)
            bus_num ∈ connected_buses || continue
            @test isapprox(
                modf_row[b_idx],
                ptdf_rebuilt[rebuilt_m, rebuilt_bus_lookup[bus_num]];
                atol = 1e-6,
            )
        end
    end

    PNM.clear_caches!(vmodf)
end

@testset "VirtualMODF: KLU and AppleAccelerate backend parity" begin
    if !PNM._has_apple_accelerate_backend()
        @info "Skipped VirtualMODF AA/KLU parity (backend unavailable on this platform)"
        return
    end

    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    vmodf_klu = VirtualMODF(sys; linear_solver = "KLU")
    vmodf_aa = VirtualMODF(sys; linear_solver = "AppleAccelerate")

    @test vmodf_klu.K isa PNM.KLULinSolveCache{Float64}
    @test contains(string(typeof(vmodf_aa.K)), "AAFactorCache")

    # Trigger the lazy PTDF_A_diag on both backends.
    diag_klu = vmodf_klu.PTDF_A_diag
    diag_aa = vmodf_aa.PTDF_A_diag
    @test length(diag_klu) == length(diag_aa)
    @test isapprox(diag_aa, diag_klu, atol = 1e-9)

    # Register the same N-1 contingency on both and compare one MODF row.
    e = 1
    b_e = vmodf_klu.arc_susceptances[e]
    ctg_id = 424242
    mod = NetworkModification("aa_parity_outage", [ArcModification(e, -b_e)])
    ctg = ContingencySpec(ctg_id, mod)
    vmodf_klu.contingency_cache[ctg_id] = ctg
    vmodf_aa.contingency_cache[ctg_id] = ctg

    row_klu = vmodf_klu[2, ctg]
    row_aa = vmodf_aa[2, ctg]
    @test isapprox(row_aa, row_klu, atol = 1e-9)
end

@testset "VirtualMODF: sparsification error is not amplified by the Woodbury correction" begin
    # R2: the cutoff is applied to the FINAL row, after the exact Woodbury solve
    # (Z is built from exact factorization solves, never sparsified rows), so the
    # error stays bounded by the cutoff however near-critical the contingency is.
    # Moving sparsification ahead of the solve would amplify it by ||W_inv|| (~190x).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")

    function _register_ctg!(v, arcs, ctgint)
        mods = [ArcModification(e, -v.arc_susceptances[e]) for e in arcs]
        u = ctgint
        ctg = ContingencySpec(u, NetworkModification("r2_$(ctgint)", mods))
        v.contingency_cache[u] = ctg
        return ctg
    end

    # arc 9 -> islanding bridge (pinv branch, infinite amplification);
    # arcs [1, 19] -> near-singular W (||W_inv||_2 ~ 190), with tau large enough
    # to actually drop entries so the bound is exercised rather than vacuous.
    for (arcs, uid, tau) in (([9], 1, 1e-3), ([1, 19], 2, 5e-2))
        vexact = VirtualMODF(sys; tol = eps())
        vtau = VirtualMODF(sys; tol = tau)
        n_arcs = length(PNM.get_arc_axis(vexact))

        ce = _register_ctg!(vexact, arcs, uid)
        ct = _register_ctg!(vtau, arcs, uid)

        wf = PNM._get_woodbury_factors(vexact, ce.modification)
        amplification = Inf
        if !wf.is_islanding
            amplification = opnorm(wf.W_inv, 2)
        end

        max_err = 0.0
        n_dropped = 0
        for m in 1:n_arcs
            exact_row = vexact[m, ce.modification]
            sparse_row = vtau[m, ct.modification]
            max_err = max(max_err, maximum(abs, sparse_row .- exact_row))
            n_dropped += count(iszero, sparse_row) - count(iszero, exact_row)
        end

        @test amplification > 50.0   # contingency is genuinely near-critical
        @test max_err <= tau         # yet terminal-cutoff error stays below the cutoff
        # [1, 19] actually drops entries, so the bound above is non-vacuous.
        if length(arcs) == 2
            @test n_dropped > 0
            @test max_err > 0.5 * tau
        end
    end
end
