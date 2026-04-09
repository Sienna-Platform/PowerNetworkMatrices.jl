# Find a bridge arc (PTDF_A_diag ≈ 1.0) not in `exclude`.
function _find_bridge_arc(vmodf; exclude = Set{Int}())
    for e in eachindex(vmodf.PTDF_A_diag)
        if abs(vmodf.PTDF_A_diag[e] - 1.0) < 1e-6 && !(e in exclude)
            return e
        end
    end
    error("No bridge arc found")
end

# Find a non-bridge arc (PTDF_A_diag well below 1.0) not in `exclude`.
function _find_non_bridge_arc(vmodf; exclude = Set{Int}())
    for e in eachindex(vmodf.PTDF_A_diag)
        if abs(vmodf.PTDF_A_diag[e]) < 1.0 - 1e-6 && !(e in exclude)
            return e
        end
    end
    error("No non-bridge arc found")
end

@testset "MODF islanding" begin
    sys14 = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vmodf = VirtualMODF(sys14)
    ptdf_ref = PTDF(sys14)

    # Compute shared arc indices once — deterministic since PTDF_A_diag is fixed
    e_bridge1 = _find_bridge_arc(vmodf)
    e_bridge2 = _find_bridge_arc(vmodf; exclude = Set([e_bridge1]))
    e_other = _find_non_bridge_arc(vmodf; exclude = Set([e_bridge1, e_bridge2]))
    monitored =
        _find_non_bridge_arc(vmodf; exclude = Set([e_bridge1, e_bridge2, e_other]))

    @testset "M=1 single bridge arc" begin
        b = vmodf.arc_susceptances[e_bridge1]
        ctg = ContingencySpec(
            Base.UUID(UInt128(50001)),
            NetworkModification("island_m1", [ArcModification(e_bridge1, -b)]),
        )
        vmodf.contingency_cache[ctg.uuid] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test all(isfinite, row)
        @test isapprox(row, ptdf_ref[monitored, :]; atol = 1e-6)
        PNM.clear_caches!(vmodf)
    end

    @testset "M=2 one bridge + one non-bridge" begin
        b_br = vmodf.arc_susceptances[e_bridge1]
        b_ot = vmodf.arc_susceptances[e_other]
        ctg = ContingencySpec(
            Base.UUID(UInt128(50002)),
            NetworkModification(
                "island_m2",
                [ArcModification(e_bridge1, -b_br), ArcModification(e_other, -b_ot)],
            ),
        )
        vmodf.contingency_cache[ctg.uuid] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test !all(x -> abs(x) < 1e-10, row)
        @test all(isfinite, row)
        PNM.clear_caches!(vmodf)
    end

    @testset "M=2 two bridges, fully null W" begin
        b1 = vmodf.arc_susceptances[e_bridge1]
        b2 = vmodf.arc_susceptances[e_bridge2]
        ctg = ContingencySpec(
            Base.UUID(UInt128(50003)),
            NetworkModification(
                "island_2bridge",
                [ArcModification(e_bridge1, -b1), ArcModification(e_bridge2, -b2)],
            ),
        )
        vmodf.contingency_cache[ctg.uuid] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test all(isfinite, row)
        @test isapprox(row, ptdf_ref[monitored, :]; atol = 1e-6)
        PNM.clear_caches!(vmodf)
    end

    @testset "M=3 two bridges + one non-bridge" begin
        b1 = vmodf.arc_susceptances[e_bridge1]
        b2 = vmodf.arc_susceptances[e_bridge2]
        b3 = vmodf.arc_susceptances[e_other]
        ctg = ContingencySpec(
            Base.UUID(UInt128(50004)),
            NetworkModification(
                "island_m3",
                [
                    ArcModification(e_bridge1, -b1),
                    ArcModification(e_bridge2, -b2),
                    ArcModification(e_other, -b3),
                ],
            ),
        )
        vmodf.contingency_cache[ctg.uuid] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test !all(x -> abs(x) < 1e-10, row)
        @test all(isfinite, row)
        PNM.clear_caches!(vmodf)
    end
end

@testset "MODF non-islanding: pinv changes do not affect normal path" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys)
    ptdf_ref = PTDF(sys)
    vmodf = VirtualMODF(sys)

    n_arcs = size(vlodf, 1)

    for e in 1:n_arcs
        @test abs(vmodf.PTDF_A_diag[e]) < 1.0 - 1e-6
    end

    for e in 1:n_arcs
        b_e = vmodf.arc_susceptances[e]
        ctg_uuid = Base.UUID(UInt128(60000 + e))
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
        PNM.clear_caches!(vmodf)
    end
end
