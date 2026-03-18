@testset "Partial LODF: full outage matches standard LODF column" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    lodf = LODF(sys5)
    n_arcs = size(vlodf, 1)

    for e in 1:n_arcs
        b_e = vlodf.arc_susceptances[e]
        arc_e = vlodf.axes[1][e]
        # Full outage: delta_b = -b_e should match the LODF column for outage arc e.
        # get_partial_lodf_row returns LODF[ℓ, e] for all monitoring arcs ℓ.
        partial_row = PNM.get_partial_lodf_row(vlodf, e, -b_e)
        standard_col = [lodf[arc_ℓ, arc_e] for arc_ℓ in lodf.axes[1]]
        @test isapprox(partial_row, standard_col, atol = 1e-10)
    end
end

@testset "Partial LODF: half susceptance change differs from full outage" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    lodf = LODF(sys5)

    # For a partial change (e.g., halving the susceptance), the result
    # should differ from the full outage LODF column.
    e = 1
    b_e = vlodf.arc_susceptances[e]
    arc_e = vlodf.axes[1][e]
    partial_row = PNM.get_partial_lodf_row(vlodf, e, -b_e / 2.0)
    standard_col = [lodf[arc_ℓ, arc_e] for arc_ℓ in lodf.axes[1]]
    @test !isapprox(partial_row, standard_col; atol = 1e-3)

    # The self-element should not be -1.0 (only full outage gives -1.0).
    @test partial_row[e] != -1.0
end

@testset "Partial LODF: zero change gives zero redistribution" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    e = 1
    # delta_b = 0 means no change, LODF should be all zeros.
    partial_row = PNM.get_partial_lodf_row(vlodf, e, 0.0)
    @test isapprox(partial_row, zeros(size(vlodf, 1)), atol = 1e-14)
end

@testset "Partial LODF: indexed by arc tuple" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    lodf = LODF(sys5)

    arc_tuple = vlodf.axes[1][1]
    b_e = vlodf.arc_susceptances[1]
    # Test that arc-tuple indexing gives the same result as integer indexing.
    partial_row_int = PNM.get_partial_lodf_row(vlodf, 1, -b_e)
    partial_row_arc = PNM.get_partial_lodf_row(vlodf, arc_tuple, -b_e)
    @test isapprox(partial_row_int, partial_row_arc, atol = 1e-14)

    # Also verify it matches the LODF column.
    standard_col = [lodf[arc_ℓ, arc_tuple] for arc_ℓ in lodf.axes[1]]
    @test isapprox(partial_row_arc, standard_col, atol = 1e-10)
end

@testset "Partial LODF: half-susceptance matches rebuilt ground truth" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    e = 1  # test arc index
    b_e = vlodf.arc_susceptances[e]

    # Compute partial LODF for halving arc e's susceptance.
    partial_row = PNM.get_partial_lodf_row(vlodf, e, -b_e / 2.0)

    # Ground truth: directly apply the Sherman-Morrison formula using a fresh linear solve,
    # independent of the internal work buffers of vlodf.
    # This validates that _getindex_partial correctly implements the formula:
    #   partial_lodf[ℓ] = α · (b_ℓ / b_e) · H[ℓ,e] / (1 - α · H[e,e])
    # where α = -delta_b / b_e = 1/2 and H[ℓ,e] = (A·(ABA)⁻¹·BA)[ℓ,e].
    n_buses = size(vlodf.A, 2)
    n_valid = length(vlodf.valid_ix)

    # Extract the BA column for arc e at non-reference bus indices.
    ba_col_e = [vlodf.BA[vlodf.valid_ix[i], e] for i in 1:n_valid]

    # Solve ABA x = ba_col_e using the factorization (fresh copy to avoid aliasing).
    lin_solve = vlodf.K \ copy(ba_col_e)

    # Map the solution back to full bus space.
    temp_full = zeros(n_buses)
    for i in 1:n_valid
        temp_full[vlodf.valid_ix[i]] = lin_solve[i]
    end

    # H_col[ℓ] = (A · ABA⁻¹ · BA[:,e])[ℓ] for all arcs ℓ.
    H_col = vlodf.A * temp_full

    # Apply Sherman-Morrison scaling: α = 1/2, denom = 1 - α·H[e,e].
    alpha = 0.5
    H_ee = vlodf.PTDF_A_diag[e]
    denom = 1.0 - alpha * H_ee
    gt_row = (alpha / (denom * b_e)) .* (vlodf.arc_susceptances .* H_col)
    # Self-element: the formula gives a finite value (not -1.0) for a partial change.

    @test isapprox(partial_row, gt_row, atol = 1e-12)
end
