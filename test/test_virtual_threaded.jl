@testset "VirtualMODF parallel getindex via @spawn" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    ptdf_ref = PTDF(sys5)
    nworkers = max(2, min(Threads.nthreads(), 4))
    vmodf = VirtualMODF(sys5; nworkers = nworkers)
    @test PNM.nworkers(vmodf) == nworkers

    n_arcs = size(vlodf, 1)
    for e in 1:n_arcs
        b_e = vmodf.arc_susceptances[e]
        ctg_uuid = Base.UUID(UInt128(e))
        ctg = ContingencySpec(
            ctg_uuid,
            NetworkModification("outage_arc_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_uuid] = ctg
    end

    # Build (monitored, contingency) work items.
    work = Tuple{Int, Base.UUID}[]
    for e in 1:n_arcs, m in 1:n_arcs
        push!(work, (m, Base.UUID(UInt128(e))))
    end

    # Reference: serial computation of each row.
    serial_results = Dict{Tuple{Int, Base.UUID}, Vector{Float64}}()
    for (m, uuid) in work
        ctg = vmodf.contingency_cache[uuid]
        serial_results[(m, uuid)] = vmodf[m, ctg.modification]
    end

    # Reset caches and recompute in parallel via @spawn.
    PNM.clear_caches!(vmodf)
    futures = map(work) do (m, uuid)
        Threads.@spawn begin
            ctg = vmodf.contingency_cache[uuid]
            vmodf[m, ctg.modification]
        end
    end
    parallel_results = Dict{Tuple{Int, Base.UUID}, Vector{Float64}}()
    for ((m, uuid), fut) in zip(work, futures)
        parallel_results[(m, uuid)] = fetch(fut)
    end

    @test length(parallel_results) == length(serial_results)
    for k in keys(serial_results)
        @test isapprox(parallel_results[k], serial_results[k], atol = 1e-9)
    end
end

@testset "VirtualPTDF threaded getindex (single-task @spawn)" begin
    # VirtualPTDF is not yet pool-backed; concurrent getindex on the same
    # instance races on its shared work_ba_col/temp_data scratch and the
    # underlying KLULinSolveCache. This test exercises the API via @spawn
    # but funnels work through one task at a time so it stays correct.
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys5)
    arcs = vptdf.axes[1]

    serial = [copy(vptdf[arc, :]) for arc in arcs]

    # New instance to clear the row cache.
    vptdf2 = VirtualPTDF(sys5)
    parallel = Vector{Vector{Float64}}(undef, length(arcs))
    fut = Threads.@spawn for i in eachindex(arcs)
        parallel[i] = copy(vptdf2[arcs[i], :])
    end
    fetch(fut)

    for i in eachindex(arcs)
        @test isapprox(parallel[i], serial[i], atol = 1e-10)
    end
end

@testset "VirtualLODF threaded getindex (single-task @spawn)" begin
    # Same caveat as VirtualPTDF: not yet pool-backed. Funnel through one task.
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    arcs = vlodf.axes[1]

    serial = [copy(vlodf[arc, :]) for arc in arcs]

    vlodf2 = VirtualLODF(sys5)
    parallel = Vector{Vector{Float64}}(undef, length(arcs))
    fut = Threads.@spawn for i in eachindex(arcs)
        parallel[i] = copy(vlodf2[arcs[i], :])
    end
    fetch(fut)

    for i in eachindex(arcs)
        @test isapprox(parallel[i], serial[i], atol = 1e-10)
    end
end

@testset "VirtualMODF: parallel islanding contingencies on RTS keep pool healthy" begin
    # Bridge-arc outages disconnect the network and exercise the
    # `_compute_modf_entry` → `_compute_woodbury_factors` → `_solve_factorization`
    # path that crashed PowerSimulations on Windows. With the pool, a
    # `Threads.@spawn` per (monitored, contingency) work item must not corrupt
    # any worker's factorization — i.e. `n_valid(pool) == nworkers` after.
    rts = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    nw = max(2, min(Threads.nthreads(), 4))
    vmodf = VirtualMODF(rts; nworkers = nw)
    @test PNM.n_valid(vmodf.K) == nw

    # Pick up to 4 bridge arcs (PTDF_A_diag ≈ 1.0 ⇒ removal islands the network).
    bridge_arcs = Int[]
    for e in eachindex(vmodf.PTDF_A_diag)
        if abs(vmodf.PTDF_A_diag[e] - 1.0) < 1e-6
            push!(bridge_arcs, e)
            length(bridge_arcs) >= 4 && break
        end
    end
    @test !isempty(bridge_arcs)

    island_mods = map(bridge_arcs) do e
        b_e = vmodf.arc_susceptances[e]
        NetworkModification("rts_island_arc_$(e)", [ArcModification(e, -b_e)])
    end

    n_arcs = length(vmodf.axes[1])
    monitored_set = collect(1:min(n_arcs, 40))
    work = [(m, mod) for mod in island_mods for m in monitored_set]

    futures = map(work) do (m, mod)
        Threads.@spawn vmodf[m, mod]
    end
    results = [fetch(f) for f in futures]

    # Each row must be finite — the Woodbury kernel must handle bridge outages
    # without producing NaN/Inf or letting a libklu solve crash a worker.
    for r in results
        @test all(isfinite, r)
    end

    # The headline assertion: every worker still holds a valid factorization
    # after the parallel islanding workload. A regression that lets a singular
    # solve corrupt a worker's numeric handle (the original PSI Windows crash
    # mode) would surface here as `n_valid(vmodf.K) < nw`.
    @test PNM.n_valid(vmodf.K) == nw

    # Pool keeps serving solves for non-islanding contingencies.
    PNM.clear_caches!(vmodf)
    benign_arc = findfirst(d -> abs(d) < 0.5, vmodf.PTDF_A_diag)
    @test benign_arc !== nothing
    b_benign = vmodf.arc_susceptances[benign_arc]
    benign_mod = NetworkModification(
        "rts_benign_outage",
        [ArcModification(benign_arc, -0.1 * b_benign)],
    )
    benign_result = vmodf[1, benign_mod]
    @test all(isfinite, benign_result)
    @test PNM.n_valid(vmodf.K) == nw
end
