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

@testset "VirtualFactorCore: concurrent first access publishes a complete vector" begin
    # A concurrent reader sees either not-ready or the complete vector.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    core = PNM.get_core(VirtualPTDF(Ybus(sys)))
    reference_diag = PNM._get_PTDF_A_diag(
        core.K,
        core.BA,
        core.A,
        core.valid_ix,
        core.bus_to_valid_idx,
    )

    n_tasks = 8
    diags = Vector{Vector{Float64}}(undef, n_tasks)
    susceptances = Vector{Vector{Vector{Float64}}}(undef, n_tasks)
    tasks = map(1:n_tasks) do i
        Threads.@spawn begin
            diags[i] = PNM.get_PTDF_A_diag(core)
            susceptances[i] = PNM.get_branch_susceptances_by_arc(core)
        end
    end
    foreach(wait, tasks)

    for i in 1:n_tasks
        @test diags[i] === core.PTDF_A_diag
        @test susceptances[i] === core.branch_susceptances_by_arc
        @test isapprox(diags[i], reference_diag; atol = 1e-12)
        @test all(w -> all(isfinite, w), susceptances[i])
    end
    @test length(core.PTDF_A_diag) == length(PNM.get_arc_axis(core))
    @test length(core.branch_susceptances_by_arc) == length(PNM.get_arc_axis(core))
end

@testset "VirtualFactorCore from arrays" begin
    @testset "size checks and cutoff" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        ybus = Ybus(sys)
        core = PNM.VirtualFactorCore(ybus)
        ref = Set(PNM.get_ref_bus_position(ybus))
        live = PNM.get_arc_live_members(ybus)
        cat = PNM.get_branch_catalog(ybus)
        build(BA, A, ref, live) = PNM.VirtualFactorCore(
            BA, A, ref, core.axes, core.subnetwork_axes, live, cat,
        )
        @test_throws ErrorException build(core.BA[:, 1:(end - 1)], core.A, ref, live)
        @test_throws ErrorException build(core.BA, core.A[1:(end - 1), :], ref, live)
        @test_throws ErrorException build(core.BA, core.A, ref, live[1:(end - 1)])
        @test_throws ErrorException build(core.BA, core.A, Set([0]), live)
        @test_throws ErrorException build(
            core.BA,
            core.A,
            Set([size(core.BA, 1) + 1]),
            live,
        )
        b = core.arc_susceptances .* core.arc_susceptance_signs
        from_b = PNM.VirtualFactorCore(b, ref, core.axes, core.subnetwork_axes, live, cat)
        @test from_b.BA == core.BA
        @test from_b.A == core.A
        @test_throws ErrorException PNM.VirtualFactorCore(
            b[1:(end - 1)], ref, core.axes, core.subnetwork_axes, live, cat,
        )
        cut = PNM.VirtualFactorCore(
            core.BA, core.A, ref, core.axes, core.subnetwork_axes, live, cat;
            tol = 1e-9,
        )
        @test PNM.get_tol(cut) == 1e-9
        @test PNM.get_tol_spec(cut) == 1e-9
    end

    @testset "the vector constructor checks the network before it factorizes" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        ybus = Ybus(sys)
        core = PNM.VirtualFactorCore(ybus)
        ref = Set(PNM.get_ref_bus_position(ybus))
        live = PNM.get_arc_live_members(ybus)
        cat = PNM.get_branch_catalog(ybus)
        arc_ax, bus_ax = core.axes
        subs = core.subnetwork_axes
        b = core.arc_susceptances .* core.arc_susceptance_signs
        build(b, ref, axes, subs, live) =
            PNM.VirtualFactorCore(b, ref, axes, subs, live, cat; linear_solver = "KLU")
        rebuilt = build(b, ref, core.axes, subs, live)
        @test rebuilt.arc_susceptances .* rebuilt.arc_susceptance_signs == b

        @test_throws "reference bus positions for" build(
            b,
            Set{Int}(),
            core.axes,
            subs,
            live,
        )
        @test_throws "reference bus positions for" build(
            b, Set([1, 4]), core.axes, subs, live,
        )
        @test_throws "not the reference bus of its subnetwork" build(
            b, Set([2]), core.axes, subs, live,
        )

        key = only(keys(subs))
        arcs, buses = subs[key]
        @test_throws "is in no subnetwork" build(
            b, ref, core.axes,
            Dict(key => (arcs, buses[1:(end - 1)])), live,
        )
        @test_throws "outside the bus axis" build(
            b, ref, core.axes,
            Dict(key => (arcs, vcat(buses, [999]))), live,
        )
        @test_throws "in more than one subnetwork" build(
            b, ref, core.axes,
            Dict(key => (arcs, vcat(buses, [buses[end]]))), live,
        )
        @test_throws "is in no subnetwork" build(
            b, ref, core.axes, Dict(key => (arcs[1:(end - 1)], buses)), live,
        )
        @test_throws "The arc $(first(arcs)) is in more than one subnetwork" build(
            b, ref, core.axes, Dict(key => (vcat(arcs, [first(arcs)]), buses)), live,
        )

        isolated = bus_ax[findfirst(!=(key), bus_ax)]
        cut_b = copy(b)
        for (ix, arc) in enumerate(arc_ax)
            isolated in arc && (cut_b[ix] = 0.0)
        end
        @test_throws "not one connected component" build(cut_b, ref, core.axes, subs, live)

        loop_ax = copy(arc_ax)
        loop_ax[1] = (arc_ax[1][1], arc_ax[1][1])
        @test_throws "is a self-loop" build(b, ref, (loop_ax, bus_ax), subs, live)
        dup_ax = copy(arc_ax)
        dup_ax[2] = dup_ax[1]
        @test_throws "arc axis has the entry" build(b, ref, (dup_ax, bus_ax), subs, live)
        dup_bus = copy(bus_ax)
        dup_bus[2] = dup_bus[1]
        @test_throws "bus axis has the entry" build(b, ref, (arc_ax, dup_bus), subs, live)

        huge = copy(b)
        huge[1] = 1e308
        @test_throws "magnitude above" build(huge, ref, core.axes, subs, live)
        negative_live = copy(live)
        negative_live[1] = -1
        @test_throws "has arc_live_members -1, below 1" build(
            b, ref, core.axes, subs, negative_live,
        )
    end

    @testset "second reference bus, shared bus, shared arc, endpoint outside" begin
        sys = build_multi_island_composite_arc_system()
        ybus = Ybus(sys)
        core = PNM.VirtualFactorCore(ybus)
        ref = Set(PNM.get_ref_bus_position(ybus))
        live = PNM.get_arc_live_members(ybus)
        cat = PNM.get_branch_catalog(ybus)
        arc_ax, bus_ax = core.axes
        subs = core.subnetwork_axes
        @test length(subs) == 2
        b = core.arc_susceptances .* core.arc_susceptance_signs
        build(ref, subs) = PNM.VirtualFactorCore(
            b, ref, core.axes, subs, live, cat; linear_solver = "KLU",
        )
        @test build(ref, subs) isa PNM.VirtualFactorCore
        key1, key2 = collect(keys(subs))
        arcs1, buses1 = subs[key1]
        arcs2, buses2 = subs[key2]
        pos(bus) = findfirst(==(bus), bus_ax)
        other_bus = first(setdiff(buses1, [key1]))
        @test_throws "has a second reference bus" build(
            Set([pos(key1), pos(other_bus)]), subs,
        )
        @test_throws "is in more than one subnetwork" build(
            ref, Dict(key1 => (arcs1, buses1), key2 => (arcs2, vcat(buses2, [other_bus]))),
        )
        @test_throws r"The arc .* (is in more than one subnetwork|has an endpoint outside it)" build(
            ref,
            Dict(key1 => (arcs1, buses1), key2 => (vcat(arcs2, [first(arcs1)]), buses2)),
        )
        @test_throws "has an endpoint outside it" build(
            ref,
            Dict(
                key1 => (vcat(arcs1, [first(arcs2)]), buses1),
                key2 => (arcs2[2:end], buses2),
            ),
        )
    end

    @testset "factorization check on the file path" begin
        solvers = ["KLU"]
        PNM._has_apple_accelerate_backend() && push!(solvers, "AppleAccelerateLU")
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        ybus = Ybus(sys)
        core = PNM.VirtualFactorCore(ybus)
        ref = Set(PNM.get_ref_bus_position(ybus))
        live = PNM.get_arc_live_members(ybus)
        cat = PNM.get_branch_catalog(ybus)
        good = core.arc_susceptances .* core.arc_susceptance_signs
        chain_axes = (
            [(1, 2), (2, 3), (3, 4)],
            [1, 2, 3, 4],
        )
        chain_subs = Dict(1 => (chain_axes[1], chain_axes[2]))
        chain(b; solver) = PNM.VirtualFactorCore(
            b, Set([1]), chain_axes, chain_subs, [1, 1, 1], cat; linear_solver = solver,
        )
        @test PNM._singular_column_text(0) == ""
        @test PNM._singular_column_text(3) == " (column 3)"
        for solver in solvers
            @test chain([1.0, 1.0, 1.0]; solver) isa PNM.VirtualFactorCore
            # Zero-impedance substitute (b = 1e6) beside normal arcs: the limit scales with |ABA||x|.
            @test chain([1.0, 1e6, 1.0]; solver) isa PNM.VirtualFactorCore
            # 1 + 1e-17 rounds to 1: the two buses behind the weak arc form a singular block.
            @test_throws "ABA matrix of the factor file" chain([1.0, 1e-17, 1.0]; solver)
            # Non-integer values and a meshed island: a rounding-level pivot, no exact zero.
            mesh_axes = (
                [(1, 2), (2, 3), (3, 4), (4, 5), (3, 5)],
                [1, 2, 3, 4, 5],
            )
            mesh_subs = Dict(1 => (mesh_axes[1], mesh_axes[2]))
            mesh(b) = PNM.VirtualFactorCore(
                b, Set([1]), mesh_axes, mesh_subs, ones(Int, 5), cat;
                linear_solver = solver,
            )
            @test mesh([10.3, 37.7, 21.9, 14.1, 55.3]) isa PNM.VirtualFactorCore
            @test_throws "ABA matrix of the factor file" mesh([
                10.3,
                1e-17,
                37.7,
                21.9,
                14.1,
            ])
            @test_throws "The ABA matrix is empty: every bus is a reference bus." PNM.VirtualFactorCore(
                Float64[], Set([1]), (Tuple{Int, Int}[], [1]),
                Dict(1 => (Tuple{Int, Int}[], [1])), Int[], cat;
                linear_solver = solver,
            )
            @test PNM.VirtualFactorCore(
                good, ref, core.axes, core.subnetwork_axes, live, cat;
                linear_solver = solver,
            ) isa PNM.VirtualFactorCore
        end
    end
end
