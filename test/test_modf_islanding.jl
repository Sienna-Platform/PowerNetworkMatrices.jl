@testset "MODF islanding: M=2 one bridge + one non-bridge" begin
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vmodf = VirtualMODF(sys)

    # Arc 1 is a bridge (PTDF_A_diag ≈ 1.0), arc 5 is not
    e_bridge = 1
    e_other = 5
    b_bridge = vmodf.arc_susceptances[e_bridge]
    b_other = vmodf.arc_susceptances[e_other]
    @test abs(vmodf.PTDF_A_diag[e_bridge] - 1.0) < 1e-6  # confirm bridge

    ctg_uuid = Base.UUID(UInt128(hash((:island_m2, e_bridge, e_other))))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification(
            "island_m2",
            [ArcModification(e_bridge, -b_bridge), ArcModification(e_other, -b_other)],
        ),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    # Monitor a non-bridge, non-contingency arc in the main connected component.
    # Arc 7 (109,110) has PTDF_A_diag ≈ 0.0085, well connected.
    monitored = 7
    row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)

    # The row must NOT be all zeros — the non-bridge outage still redistributes flow
    @test !all(x -> abs(x) < 1e-10, row)

    # The row should have finite, bounded values
    @test all(isfinite, row)
end

@testset "MODF islanding: M=3 two bridges + one non-bridge" begin
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vmodf = VirtualMODF(sys)

    # Arcs 1, 3 are bridges; arc 5 is not
    e1, e2, e3 = 1, 3, 5
    b1 = vmodf.arc_susceptances[e1]
    b2 = vmodf.arc_susceptances[e2]
    b3 = vmodf.arc_susceptances[e3]
    @test abs(vmodf.PTDF_A_diag[e1] - 1.0) < 1e-6
    @test abs(vmodf.PTDF_A_diag[e2] - 1.0) < 1e-6

    ctg_uuid = Base.UUID(UInt128(hash((:island_m3, e1, e2, e3))))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification(
            "island_m3",
            [
                ArcModification(e1, -b1),
                ArcModification(e2, -b2),
                ArcModification(e3, -b3),
            ],
        ),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    # Monitor arc 7 — well-connected, not part of contingency
    monitored = 7
    row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)

    # Must not throw SingularException (current behavior) and must not be all-zeros
    @test !all(x -> abs(x) < 1e-10, row)
    @test all(isfinite, row)
end

@testset "MODF islanding: M=1 single bridge arc" begin
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vmodf = VirtualMODF(sys)

    # Arc 1 is a bridge — full outage completely islands a subnetwork
    e_bridge = 1
    b_bridge = vmodf.arc_susceptances[e_bridge]
    @test abs(vmodf.PTDF_A_diag[e_bridge] - 1.0) < 1e-6

    ctg_uuid = Base.UUID(UInt128(hash((:island_m1, e_bridge))))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification(
            "island_m1",
            [ArcModification(e_bridge, -b_bridge)],
        ),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    # For M=1 bridge, W_inv = pinv([0]) = [0], so correction is zero.
    # The result is b_mon_post * z_m = the pre-contingency PTDF row.
    monitored = 7  # (109,110) — well inside the main grid
    row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)

    @test all(isfinite, row)

    # With pinv, correction vanishes → result equals pre-contingency PTDF row
    ptdf_ref = PTDF(sys)
    @test isapprox(row, ptdf_ref[monitored, :]; atol = 1e-6)
end

@testset "MODF non-islanding: pinv changes do not affect normal path" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys)
    ptdf_ref = PTDF(sys)
    vmodf = VirtualMODF(sys)

    n_arcs = size(vlodf, 1)

    # c_sys5 has no bridge arcs — all contingencies are non-islanding
    for e in 1:n_arcs
        @test abs(vmodf.PTDF_A_diag[e]) < 1.0 - 1e-6
    end

    # Verify PTDF+LODF identity for every arc as contingency
    for e in 1:n_arcs
        b_e = vmodf.arc_susceptances[e]
        ctg_uuid = Base.UUID(UInt128(hash((:regression, e))))
        ctg = ContingencySpec(
            ctg_uuid,
            NetworkModification("regression_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_uuid] = ctg

        for m in 1:n_arcs
            modf_row = PNM._compute_modf_entry(vmodf, m, ctg.modification)
            expected = ptdf_ref[m, :] .+ vlodf[m, e] .* ptdf_ref[e, :]
            @test isapprox(modf_row, expected; atol = 1e-6)
        end
        empty!(vmodf.woodbury_cache)
    end
end

@testset "MODF islanding: M=2 two bridges, fully null W" begin
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vmodf = VirtualMODF(sys)

    # Arcs 1 and 3 are both bridges
    e1, e2 = 1, 3
    b1 = vmodf.arc_susceptances[e1]
    b2 = vmodf.arc_susceptances[e2]
    @test abs(vmodf.PTDF_A_diag[e1] - 1.0) < 1e-6
    @test abs(vmodf.PTDF_A_diag[e2] - 1.0) < 1e-6

    ctg_uuid = Base.UUID(UInt128(hash((:island_2bridge, e1, e2))))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification(
            "island_2bridge",
            [ArcModification(e1, -b1), ArcModification(e2, -b2)],
        ),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    monitored = 7
    row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)

    # pinv(zeros(2,2)) = zeros(2,2) → correction vanishes
    @test all(isfinite, row)
    ptdf_ref = PTDF(sys)
    @test isapprox(row, ptdf_ref[monitored, :]; atol = 1e-6)
end
