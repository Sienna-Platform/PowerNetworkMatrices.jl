@testset "compute_aba_delta: N-1 branch outages match rebuilt ABA" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)
    vptdf = VirtualPTDF(sys)

    for branch in get_components(
        x -> !(typeof(x) <: Union{PhaseShiftingTransformer, DiscreteControlledACBranch}),
        ACTransmission, sys,
    )
        mod_new = NetworkModification(vptdf, branch)
        result_new = apply_aba_modification(aba, mod_new)

        # Reference: disable branch and rebuild
        set_available!(branch, false)
        aba_ref = ABA_Matrix(sys)
        set_available!(branch, true)

        @test size(result_new) == size(aba_ref.data)
        @test isapprox(Matrix(result_new), Matrix(aba_ref.data), atol = 1e-8)
    end
end

@testset "compute_aba_delta: N-2 outage via shared Outage matches rebuilt ABA" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)
    vptdf = VirtualPTDF(sys)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")

    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, line1, outage)
    add_supplemental_attribute!(sys, line2, outage)

    mod = NetworkModification(vptdf, sys, outage)
    result = apply_aba_modification(aba, mod)

    set_available!(line1, false)
    set_available!(line2, false)
    aba_ref = ABA_Matrix(sys)

    @test isapprox(Matrix(result), Matrix(aba_ref.data), atol = 1e-8)
end

@testset "compute_aba_delta: delta + U Woodbury identity holds" begin
    # Verify that the returned `delta` matches `U * Diagonal(delta_b) * U'`.
    # This is the structural guarantee Woodbury relies on.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)
    vptdf = VirtualPTDF(sys)

    for branch in get_components(
        x -> !(typeof(x) <: Union{PhaseShiftingTransformer, DiscreteControlledACBranch}),
        ACTransmission, sys,
    )
        mod = NetworkModification(vptdf, branch)
        d = compute_aba_delta(aba, mod)
        # reconstruction: U * diag(Δb) * Uᵀ must equal the sparse delta
        # SparseMatrixCSC * Diagonal * SparseMatrixCSC has no dispatch in v1.12; densify first.
        # LinearAlgebra is imported via `import LinearAlgebra: I` in the tests
        # harness; access Diagonal through PNM.LinearAlgebra for portability.
        UD = Matrix(d.U) * PNM.LinearAlgebra.Diagonal(d.delta_b)
        recon = UD * transpose(Matrix(d.U))
        @test isapprox(Matrix(d.delta), recon, atol = 1e-12)
    end
end

@testset "ABA Woodbury: N-1 matches refactor over multi-RHS" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)
    vptdf = VirtualPTDF(sys)
    K = PNM.KLU.klu(aba.data)

    n_valid = size(aba.data, 1)
    P = randn(n_valid, 3)  # 3 timesteps

    for branch in get_components(
        x -> !(typeof(x) <: Union{PhaseShiftingTransformer, DiscreteControlledACBranch}),
        ACTransmission, sys,
    )
        mod = NetworkModification(vptdf, branch)
        d = compute_aba_delta(aba, mod)
        M = size(d.U, 2)
        M == 0 && continue

        # Reference via refactor
        ABA_k = aba.data + d.delta
        θ_ref = Matrix(ABA_k) \ P

        # Woodbury
        Z_buf = Matrix{Float64}(undef, n_valid, M)
        wf = compute_aba_woodbury_factors(K, d.U, d.delta_b, d.arc_indices, Z_buf)
        θ_wb = Matrix(aba.data) \ P  # base θ
        scratch = Matrix{Float64}(undef, M, 3)
        apply_aba_woodbury_correction!(θ_wb, d.U, wf, scratch)

        @test isapprox(θ_wb, θ_ref, atol = 1e-8, rtol = 1e-10)
    end
end

@testset "ABA Woodbury: N-2 shared-Outage matches refactor" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)
    vptdf = VirtualPTDF(sys)
    K = PNM.KLU.klu(aba.data)

    line1 = get_component(Line, sys, "1")
    line2 = get_component(Line, sys, "2")
    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, line1, outage)
    add_supplemental_attribute!(sys, line2, outage)

    mod = NetworkModification(vptdf, sys, outage)
    d = compute_aba_delta(aba, mod)
    M = size(d.U, 2)
    @test M == 2

    n_valid = size(aba.data, 1)
    P = randn(n_valid, 3)

    ABA_k = aba.data + d.delta
    θ_ref = Matrix(ABA_k) \ P

    Z_buf = Matrix{Float64}(undef, n_valid, M)
    wf = compute_aba_woodbury_factors(K, d.U, d.delta_b, d.arc_indices, Z_buf)
    θ_wb = Matrix(aba.data) \ P
    scratch = Matrix{Float64}(undef, M, 3)
    apply_aba_woodbury_correction!(θ_wb, d.U, wf, scratch)

    @test isapprox(θ_wb, θ_ref, atol = 1e-8, rtol = 1e-10)
    @test wf.is_islanding == false
end

@testset "Woodbury W inversion: relative tolerance catches large-norm singular W" begin
    # A matrix with huge norm and exact singularity must be flagged even
    # though the fixed absolute tolerance (1e-10) would miss it if the
    # machine-precision residual on the determinant scales with ‖W‖₁².
    # Val(1): w = 0 but with a large relative scale.
    W1 = fill(0.0, 1, 1)
    inv1, isl1 = PNM._invert_woodbury_W(W1, Val(1))
    @test isl1 == true
    @test inv1[1, 1] == 0.0

    # Val(1): large nonzero w should NOT be flagged as islanding.
    W1b = fill(1e6, 1, 1)
    _, isl1b = PNM._invert_woodbury_W(W1b, Val(1))
    @test isl1b == false

    # Val(2): singular (rank-1) W with huge entries — det is O(ε * ‖W‖₁²)
    # from floating-point noise; relative tolerance must catch this.
    W2 = [1e8 2e8; 1e8 2e8]  # rank 1 exactly
    _, isl2 = PNM._invert_woodbury_W(W2, Val(2))
    @test isl2 == true

    # Val(2): well-conditioned large W should not be flagged.
    W2b = [1e8 0.0; 0.0 1e8]
    _, isl2b = PNM._invert_woodbury_W(W2b, Val(2))
    @test isl2b == false

    # Val(M>2): tiny pivot within a large-norm matrix.
    W3 = [1e6 0.0 0.0; 0.0 1e6 0.0; 0.0 0.0 1e-20]
    _, isl3 = PNM._invert_woodbury_W(W3, Val(3))
    @test isl3 == true
end

@testset "compute_aba_delta: rejects off-nominal tap transformers" begin
    # Use the PSS/e 14-bus reduction test system (has genuine TapTransformers).
    sys = PSB.build_system(PSB.PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    tapped_candidates = collect(get_components(PSY.TapTransformer, sys))
    if isempty(tapped_candidates)
        @info "No TapTransformer in fixture; tap-guard test skipped."
    else
        # Pick one, force tap off-nominal, rebuild ABA to reflect it.
        tapped = first(tapped_candidates)
        PSY.set_tap!(tapped, 1.05)
        aba = ABA_Matrix(sys)
        vptdf = VirtualPTDF(sys)
        mod = NetworkModification(vptdf, tapped)
        @test_throws ArgumentError compute_aba_delta(aba, mod)
    end
end

@testset "compute_aba_delta: rejects arc between two reference buses" begin
    # Construct a ref-ref arc synthetically — such arcs don't appear in
    # connected single-subnetwork systems. We fabricate an ArcModification
    # that points at an existing arc and temporarily make both endpoints
    # look like ref buses by wrapping the ABA in a modified lookup.
    # Easier: directly test the error branch by calling the helper with
    # a concocted NetworkModification whose arc_indices happen to resolve
    # to ref-ref. We emulate this using a mocked out-of-range bus lookup.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)
    nr = PNM.get_network_reduction_data(aba)
    arc_ax = PNM.get_arc_axis(nr)

    # Pick any arc; then construct an ABA-like matrix with empty bus_lookup so
    # that every arc appears ref-ref. This exercises the error path.
    broken_aba = PNM.ABA_Matrix(
        aba.data,
        aba.axes,
        (Dict{Int, Int}(), Dict{Int, Int}()),  # empty lookup → every bus ref
        aba.subnetwork_axes,
        aba.ref_bus_position,
        aba.K,
        aba.network_reduction_data,
    )
    # Take the first Line as the modification target
    line = first(
        get_components(
            x -> !(typeof(x) <: Union{PhaseShiftingTransformer, DiscreteControlledACBranch}),
            ACTransmission,
            sys,
        ),
    )
    vptdf = VirtualPTDF(sys)
    mod = NetworkModification(vptdf, line)
    @test_throws ArgumentError compute_aba_delta(broken_aba, mod)
end

@testset "compute_aba_delta: skips shunt modifications for DC" begin
    # Shunt admittance changes do not affect DC susceptance. Verify they are
    # silently skipped and surfaced in `skipped_shunts`.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    aba = ABA_Matrix(sys)

    shunt_mod = PNM.ShuntModification(1, PNM.YBUS_ELTYPE(0.1 + 0.2im))
    mod = NetworkModification(
        "shunt_only",
        PNM.ArcModification[],
        [shunt_mod],
        false,
    )

    d = compute_aba_delta(aba, mod)
    @test d.skipped_shunts == 1
    @test size(d.U, 2) == 0
    @test isempty(d.delta_b)
    @test nnz(d.delta) == 0
end
