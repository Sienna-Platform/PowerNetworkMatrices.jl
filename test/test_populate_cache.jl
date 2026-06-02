@testset "populate_cache VirtualPTDF: batched solve matches lazy getindex" for solver in
                                                                              ("KLU", "AppleAccelerate")
    if !PowerNetworkMatrices._has_apple_accelerate_backend() && solver == "AppleAccelerate"
        @info "Skipped AppleAccelerate populate_cache tests (backend unavailable)"
        continue
    end
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    v_lazy = VirtualPTDF(sys; linear_solver = solver)
    v_pop = VirtualPTDF(sys; linear_solver = solver)

    n_arcs = length(v_pop.axes[1])
    arc_idxs = collect(1:min(6, n_arcs))

    @test populate_cache(v_pop, arc_idxs) === nothing

    for i in arc_idxs
        # Row is resident and pinned after population
        @test haskey(v_pop.cache.temp_cache, i)
        @test i ∈ v_pop.cache.persistent_cache_keys
        # Batched multi-RHS solve must agree with one-at-a-time lazy compute
        @test isapprox(v_pop[i, :], v_lazy[i, :]; atol = 1e-10)
    end
end

@testset "populate_cache VirtualPTDF: tuple and string identifiers" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    v_lazy = VirtualPTDF(sys)
    v_pop = VirtualPTDF(sys)

    arc_tuples = collect(keys(v_pop.lookup[1]))[1:4]
    populate_cache(v_pop, arc_tuples)
    for a in arc_tuples
        idx = v_pop.lookup[1][a]
        @test idx ∈ v_pop.cache.persistent_cache_keys
        @test isapprox(v_pop[a, :], v_lazy[a, :]; atol = 1e-10)
    end
end

@testset "populate_cache VirtualPTDF: pins previously-lazy rows" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    v = VirtualPTDF(sys)
    # Warm a row lazily (not pinned yet)
    _ = v[1, :]
    @test haskey(v.cache.temp_cache, 1)
    @test 1 ∉ v.cache.persistent_cache_keys
    # populate_cache must pin it without recomputing incorrectly
    populate_cache(v, [1, 2])
    @test 1 ∈ v.cache.persistent_cache_keys
    @test 2 ∈ v.cache.persistent_cache_keys
end

@testset "populate_cache VirtualLODF: batched solve matches lazy getindex" for solver in
                                                                              ("KLU", "AppleAccelerate")
    if !PowerNetworkMatrices._has_apple_accelerate_backend() && solver == "AppleAccelerate"
        @info "Skipped AppleAccelerate populate_cache tests (backend unavailable)"
        continue
    end
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    v_lazy = VirtualLODF(sys; linear_solver = solver)
    v_pop = VirtualLODF(sys; linear_solver = solver)

    n_arcs = length(v_pop.axes[1])
    arc_idxs = collect(1:min(6, n_arcs))

    populate_cache(v_pop, arc_idxs)
    for i in arc_idxs
        @test haskey(v_pop.cache.temp_cache, i)
        @test i ∈ v_pop.cache.persistent_cache_keys
        @test isapprox(v_pop[i, :], v_lazy[i, :]; atol = 1e-10)
        # Self-element convention preserved by the batched path
        @test v_pop[i, i] == -1.0
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
        b_e = v_pop.arc_susceptances[e]
        uuid = Base.UUID(UInt128(7000 + e))
        ctg = ContingencySpec(
            uuid,
            NetworkModification("populate_ctg_$e", [ArcModification(e, -b_e)]),
        )
        v_pop.contingency_cache[uuid] = ctg
        v_lazy.contingency_cache[uuid] = ctg
        push!(ctgs, ctg)
    end

    @test populate_cache(v_pop, ctgs; monitored = monitored) === nothing

    for ctg in ctgs
        mod = ctg.modification
        @test haskey(v_pop.woodbury_cache, mod)
        @test haskey(v_pop.row_caches, mod)
        rc = v_pop.row_caches[mod]
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
    b_e = vmodf.arc_susceptances[e]
    uuid = Base.UUID(UInt128(7100))
    ctg = ContingencySpec(
        uuid,
        NetworkModification("populate_tuple_ctg", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[uuid] = ctg

    # Monitor by arc bus-pair tuple
    mon_tuple = vmodf.axes[1][2]
    populate_cache(vmodf, [ctg]; monitored = [mon_tuple])
    m_idx = vmodf.lookup[1][mon_tuple]
    @test haskey(vmodf.row_caches[ctg.modification], m_idx)

    # Unregistered UUID must error clearly
    @test_throws ErrorException populate_cache(
        vmodf,
        [Base.UUID(UInt128(99999))];
        monitored = [1],
    )
end
