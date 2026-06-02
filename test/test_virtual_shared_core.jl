# Equivalence tests for the shared `VirtualFactorCore`: building a VirtualPTDF,
# VirtualMODF, and VirtualLODF on one shared core must produce the same numbers
# as building each independently, while factorizing only once.

@testset "Virtual matrices: shared VirtualFactorCore" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    # Independently constructed objects (each builds its own factorization).
    vptdf_i = VirtualPTDF(sys)
    vmodf_i = VirtualMODF(sys; automatically_register_outages = false)
    vlodf_i = VirtualLODF(sys)

    # Shared: build one core via a VirtualPTDF from a Ybus, then wrap it.
    vptdf_s = VirtualPTDF(Ybus(sys))
    vmodf_s = VirtualMODF(vptdf_s, sys; automatically_register_outages = false)
    vlodf_s = VirtualLODF(vptdf_s)

    @testset "single shared factorization" begin
        core = PNM.get_core(vptdf_s)
        @test PNM.get_core(vmodf_s) === core
        @test PNM.get_core(vlodf_s) === core
        # Same underlying factorization and matrices (identity, not just ==).
        @test vmodf_s.K === core.K
        @test vlodf_s.K === core.K
        @test vmodf_s.BA === core.BA
        @test vlodf_s.A === core.A
    end

    arc_ax = PNM.get_arc_axis(vptdf_s)
    bus_ax = PNM.get_bus_axis(vptdf_s)

    @testset "PTDF rows match independent build" begin
        for arc in arc_ax, bus in bus_ax
            @test isapprox(vptdf_s[arc, bus], vptdf_i[arc, bus]; atol = 1e-10)
        end
    end

    @testset "LODF rows match independent build" begin
        l_arc_ax = PNM.get_arc_axis(vlodf_s)
        for sel in l_arc_ax, out in l_arc_ax
            @test isapprox(vlodf_s[sel, out], vlodf_i[sel, out]; atol = 1e-10)
        end
    end

    @testset "PTDF_A_diag is shared and consistent" begin
        # Building vlodf_s forced the raw diagonal onto the shared core.
        @test !isempty(PNM.get_core(vptdf_s).PTDF_A_diag)
        @test PNM.get_PTDF_A_diag(vmodf_s) ≈ PNM.get_PTDF_A_diag(vmodf_i) atol = 1e-10
        # MODF reads the same cached vector that LODF populated on the core.
        @test PNM.get_PTDF_A_diag(vmodf_s) === PNM.get_core(vptdf_s).PTDF_A_diag
    end

    @testset "MODF post-contingency rows match independent build" begin
        e = 1
        b_e = vmodf_s.arc_susceptances[e]
        mod = NetworkModification("shared_core_eq", [ArcModification(e, -b_e)])
        for monitored in (2, 3)
            @test isapprox(
                vmodf_s[monitored, mod],
                vmodf_i[monitored, mod];
                atol = 1e-10,
            )
        end
    end
end
