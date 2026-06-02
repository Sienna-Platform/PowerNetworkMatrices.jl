@testset "populate_cache VirtualPTDF: batched solve matches lazy getindex" begin
    for solver in ("KLU", "AppleAccelerate")
        if !PowerNetworkMatrices._has_apple_accelerate_backend() &&
           solver == "AppleAccelerate"
            @info "Skipped AppleAccelerate populate_cache tests (backend unavailable)"
            continue
        end
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        v_lazy = VirtualPTDF(sys; linear_solver = solver)
        v_pop = VirtualPTDF(sys; linear_solver = solver)

        arc_lookup = PNM.get_arc_lookup(v_pop)
        arc_tuples = PNM.get_arc_axis(v_pop)[1:min(6, length(arc_lookup))]

        @test populate_cache(v_pop, arc_tuples) === nothing

        for a in arc_tuples
            idx = arc_lookup[a]
            # Row is resident and pinned after population
            @test haskey(PNM.get_cache(v_pop).temp_cache, idx)
            @test idx ∈ PNM.get_cache(v_pop).persistent_cache_keys
            # Batched multi-RHS solve must agree with one-at-a-time lazy compute
            @test isapprox(v_pop[a, :], v_lazy[a, :]; atol = 1e-10)
        end
    end
end

@testset "populate_cache VirtualPTDF: integer indices populate the cache" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    v_lazy = VirtualPTDF(sys)
    v_pop = VirtualPTDF(sys)

    arc_lookup = PNM.get_arc_lookup(v_pop)
    arcs = PNM.get_arc_axis(v_pop)[1:4]
    idxs = [arc_lookup[a] for a in arcs]

    populate_cache(v_pop, idxs)
    for a in arcs
        idx = arc_lookup[a]
        @test idx ∈ PNM.get_cache(v_pop).persistent_cache_keys
        @test isapprox(PNM.get_cache(v_pop).temp_cache[idx], v_lazy[a, :]; atol = 1e-10)
    end
end

@testset "populate_cache VirtualPTDF: pins previously-lazy rows" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    v = VirtualPTDF(sys)
    arc = PNM.get_arc_axis(v)[1]
    idx = PNM.get_arc_lookup(v)[arc]
    # Warm a row lazily (not pinned yet)
    _ = v[arc, :]
    @test haskey(PNM.get_cache(v).temp_cache, idx)
    @test idx ∉ PNM.get_cache(v).persistent_cache_keys
    # populate_cache must pin it without recomputing incorrectly
    populate_cache(v, [arc])
    @test idx ∈ PNM.get_cache(v).persistent_cache_keys
end

@testset "populate_cache VirtualLODF: batched solve matches lazy getindex" begin
    for solver in ("KLU", "AppleAccelerate")
        if !PowerNetworkMatrices._has_apple_accelerate_backend() &&
           solver == "AppleAccelerate"
            @info "Skipped AppleAccelerate populate_cache tests (backend unavailable)"
            continue
        end
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        v_lazy = VirtualLODF(sys; linear_solver = solver)
        v_pop = VirtualLODF(sys; linear_solver = solver)

        arc_lookup = PNM.get_arc_lookup(v_pop)
        arc_tuples = PNM.get_arc_axis(v_pop)[1:min(6, length(arc_lookup))]

        populate_cache(v_pop, arc_tuples)
        for a in arc_tuples
            idx = arc_lookup[a]
            @test haskey(PNM.get_cache(v_pop).temp_cache, idx)
            @test idx ∈ PNM.get_cache(v_pop).persistent_cache_keys
            @test isapprox(v_pop[a, :], v_lazy[a, :]; atol = 1e-10)
            # Self-element convention preserved by the batched path
            @test v_pop[a, a] == -1.0
        end
    end
end

@testset "populate_cache VirtualMODF: batched contingencies match lazy getindex" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    v_lazy = VirtualMODF(sys5)
    v_pop = VirtualMODF(sys5)

    n_arcs = size(v_pop, 1)
    monitored = collect(1:min(4, n_arcs))

    ctgs = ContingencySpec[]
    for e in 1:3
        b_e = PNM._get_arc_susceptances(v_pop)[e]
        uuid = Base.UUID(UInt128(7000 + e))
        ctg = ContingencySpec(
            uuid,
            NetworkModification("populate_ctg_$e", [ArcModification(e, -b_e)]),
        )
        PNM.get_contingency_cache(v_pop)[uuid] = ctg
        PNM.get_contingency_cache(v_lazy)[uuid] = ctg
        push!(ctgs, ctg)
    end

    @test populate_cache(v_pop, ctgs; monitored = monitored) === nothing

    for ctg in ctgs
        mod = ctg.modification
        @test haskey(PNM.get_woodbury_cache(v_pop), mod)
        @test haskey(PNM.get_row_caches(v_pop), mod)
        rc = PNM.get_row_caches(v_pop)[mod]
        for m in monitored
            @test haskey(rc, m)
            @test m ∈ rc.persistent_cache_keys
            @test isapprox(v_pop[m, ctg], v_lazy[m, ctg]; atol = 1e-8)
        end
    end
end

@testset "populate_cache VirtualMODF: tuple-monitored and outage resolution errors" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vmodf = VirtualMODF(sys5)

    e = 1
    b_e = PNM._get_arc_susceptances(vmodf)[e]
    uuid = Base.UUID(UInt128(7100))
    ctg = ContingencySpec(
        uuid,
        NetworkModification("populate_tuple_ctg", [ArcModification(e, -b_e)]),
    )
    PNM.get_contingency_cache(vmodf)[uuid] = ctg

    # Monitor by arc bus-pair tuple
    mon_tuple = PNM.get_arc_axis(vmodf)[2]
    populate_cache(vmodf, [ctg]; monitored = [mon_tuple])
    m_idx = PNM.get_arc_lookup(vmodf)[mon_tuple]
    @test haskey(PNM.get_row_caches(vmodf)[ctg.modification], m_idx)

    # Unregistered UUID must error clearly
    @test_throws ErrorException populate_cache(
        vmodf,
        [Base.UUID(UInt128(99999))];
        monitored = [1],
    )
end
