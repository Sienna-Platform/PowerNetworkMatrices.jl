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
    n_arcs = size(vptdf, 1)

    serial = [copy(vptdf[i, :]) for i in 1:n_arcs]

    # New instance to clear the row cache.
    vptdf2 = VirtualPTDF(sys5)
    parallel = Vector{Vector{Float64}}(undef, n_arcs)
    fut = Threads.@spawn for i in 1:n_arcs
        parallel[i] = copy(vptdf2[i, :])
    end
    fetch(fut)

    for i in 1:n_arcs
        @test isapprox(parallel[i], serial[i], atol = 1e-10)
    end
end

@testset "VirtualLODF threaded getindex (single-task @spawn)" begin
    # Same caveat as VirtualPTDF: not yet pool-backed. Funnel through one task.
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys5)
    n_arcs = size(vlodf, 1)

    serial = [copy(vlodf[i, :]) for i in 1:n_arcs]

    vlodf2 = VirtualLODF(sys5)
    parallel = Vector{Vector{Float64}}(undef, n_arcs)
    fut = Threads.@spawn for i in 1:n_arcs
        parallel[i] = copy(vlodf2[i, :])
    end
    fetch(fut)

    for i in 1:n_arcs
        @test isapprox(parallel[i], serial[i], atol = 1e-10)
    end
end
