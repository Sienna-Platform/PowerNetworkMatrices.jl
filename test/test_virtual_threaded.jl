@testset "VirtualMODF parallel getindex via @spawn" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
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

    work = Tuple{Int, Base.UUID}[]
    for e in 1:n_arcs, m in 1:n_arcs
        push!(work, (m, Base.UUID(UInt128(e))))
    end

    serial_results = Dict{Tuple{Int, Base.UUID}, Vector{Float64}}()
    for (m, uuid) in work
        ctg = vmodf.contingency_cache[uuid]
        serial_results[(m, uuid)] = vmodf[m, ctg.modification]
    end

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

@testset "VirtualPTDF concurrent getindex matches serial baseline" begin
    # Pool-backed VirtualPTDF must produce identical results regardless of
    # whether rows are queried serially or under contention. With per-worker
    # scratch and a cache_lock, parallel queries cannot corrupt each other.
    rts = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    nworkers = max(2, min(Threads.nthreads(), 4))
    vptdf_serial = VirtualPTDF(rts; nworkers = nworkers)
    arcs = collect(vptdf_serial.axes[1])
    n_query = min(length(arcs), 80)
    query_arcs = arcs[1:n_query]

    serial = [copy(vptdf_serial[a, :]) for a in query_arcs]

    vptdf_par = VirtualPTDF(rts; nworkers = nworkers)
    parallel = Vector{Vector{Float64}}(undef, n_query)
    Threads.@threads :dynamic for i in 1:n_query
        parallel[i] = copy(vptdf_par[query_arcs[i], :])
    end

    for i in 1:n_query
        @test isapprox(parallel[i], serial[i], atol = 1e-9)
    end
end

@testset "VirtualLODF concurrent getindex matches serial baseline" begin
    # Pool-backed VirtualLODF: same correctness guarantee as VirtualPTDF.
    rts = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    nworkers = max(2, min(Threads.nthreads(), 4))
    vlodf_serial = VirtualLODF(rts; nworkers = nworkers)
    arcs = collect(vlodf_serial.axes[1])
    n_query = min(length(arcs), 80)
    query_arcs = arcs[1:n_query]

    serial = [copy(vlodf_serial[a, :]) for a in query_arcs]

    vlodf_par = VirtualLODF(rts; nworkers = nworkers)
    parallel = Vector{Vector{Float64}}(undef, n_query)
    Threads.@threads :dynamic for i in 1:n_query
        parallel[i] = copy(vlodf_par[query_arcs[i], :])
    end

    for i in 1:n_query
        @test isapprox(parallel[i], serial[i], atol = 1e-9)
    end
end

@testset "VirtualLODF concurrent get_partial_lodf_row matches serial baseline" begin
    # _getindex_partial used to be marked NOT thread-safe; pool migration
    # makes it parallel-safe. Verify under real contention with non-bridge
    # arcs (bridge arcs island the network and produce NaN by design).
    rts = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    nworkers = max(2, min(Threads.nthreads(), 4))
    vlodf_serial = VirtualLODF(rts; nworkers = nworkers)
    n_arcs = size(vlodf_serial, 1)

    arc_indices = Int[]
    for i in 1:n_arcs
        if abs(vlodf_serial.PTDF_A_diag[i] - 1.0) >= 1e-3
            push!(arc_indices, i)
            length(arc_indices) >= 60 && break
        end
    end
    @test !isempty(arc_indices)

    serial = Vector{Vector{Float64}}(undef, length(arc_indices))
    for (k, i) in enumerate(arc_indices)
        b = vlodf_serial.arc_susceptances[i]
        serial[k] = PNM.get_partial_lodf_row(vlodf_serial, i, -b)
    end

    vlodf_par = VirtualLODF(rts; nworkers = nworkers)
    parallel = Vector{Vector{Float64}}(undef, length(arc_indices))
    Threads.@threads :dynamic for k in eachindex(arc_indices)
        i = arc_indices[k]
        b = vlodf_par.arc_susceptances[i]
        parallel[k] = PNM.get_partial_lodf_row(vlodf_par, i, -b)
    end

    for k in eachindex(arc_indices)
        @test isapprox(parallel[k], serial[k], atol = 1e-9)
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

    for r in results
        @test all(isfinite, r)
    end

    @test PNM.n_valid(vmodf.K) == nw

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

@testset "VirtualMODF concurrent getindex on RTS matches serial baseline" begin
    # Replaces the @spawn-per-item c_sys5 case with real contention on a
    # bigger system. Threads.@threads :dynamic schedules work across
    # multiple OS threads simultaneously, exercising the pool under load.
    rts = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    nw = max(2, min(Threads.nthreads(), 4))
    vmodf_serial = VirtualMODF(rts; nworkers = nw)

    n_arcs = length(vmodf_serial.axes[1])
    contingency_arcs = Int[]
    for e in eachindex(vmodf_serial.PTDF_A_diag)
        # skip arcs that island the network (PTDF_A_diag ≈ 1.0)
        if abs(vmodf_serial.PTDF_A_diag[e] - 1.0) >= 1e-3
            push!(contingency_arcs, e)
            length(contingency_arcs) >= 8 && break
        end
    end
    @test !isempty(contingency_arcs)

    mods = map(contingency_arcs) do e
        b_e = vmodf_serial.arc_susceptances[e]
        NetworkModification("rts_outage_$(e)", [ArcModification(e, -b_e)])
    end
    monitored_set = collect(1:min(n_arcs, 40))
    work = [(m, mod) for mod in mods for m in monitored_set]

    serial = [vmodf_serial[m, mod] for (m, mod) in work]

    vmodf_par = VirtualMODF(rts; nworkers = nw)
    parallel = Vector{Vector{Float64}}(undef, length(work))
    Threads.@threads :dynamic for i in eachindex(work)
        m, mod = work[i]
        parallel[i] = vmodf_par[m, mod]
    end

    for i in eachindex(work)
        @test isapprox(parallel[i], serial[i], atol = 1e-9)
    end
    @test PNM.n_valid(vmodf_par.K) == nw
end

@testset "VirtualMODF clear_caches! racing with getindex stays correct" begin
    # `clear_caches!` empties Woodbury and row caches under their respective
    # locks; concurrent getindex must either see the cached row or
    # recompute it cleanly — never crash and never return non-finite values.
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    nw = max(2, min(Threads.nthreads(), 4))
    vmodf = VirtualMODF(sys5; nworkers = nw)
    n_arcs = length(vmodf.axes[1])

    e = 1
    b_e = vmodf.arc_susceptances[e]
    mod = NetworkModification("clear_race", [ArcModification(e, -b_e)])

    stop = Threads.Atomic{Bool}(false)
    clearer = Threads.@spawn begin
        while !stop[]
            PNM.clear_caches!(vmodf)
            sleep(0.001)
        end
    end

    queries = 0
    for _ in 1:200, m in 1:n_arcs
        row = vmodf[m, mod]
        @test all(isfinite, row)
        queries += 1
    end
    stop[] = true
    fetch(clearer)
    @test queries > 0
    @test PNM.n_valid(vmodf.K) == nw
end
