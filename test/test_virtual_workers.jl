# Worker cores: per-worker factorizations for parallel virtual-matrix queries (option 1: matrices
# on a worker core; option 2: shared matrices solving on a caller's worker core), and the
# factor-free flow algebra on angles.

function _workers_sys14()
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for name in ["Line1", "Line2", "Line9", "Line10", "Line12"]
        line = PSY.get_component(PSY.ACTransmission, sys, name)
        PSY.add_supplemental_attribute!(
            sys,
            line,
            PSY.FixedForcedOutage(; outage_status = 1.0),
        )
    end
    return sys
end

# A bridge (islanding alone) and an M = 2 modification that islands through the pinv path.
function _workers_island_mods(vmodf)
    diag = PNM.get_PTDF_A_diag(vmodf)
    sus = vmodf.arc_susceptances
    bridges = findall(d -> abs(d - 1.0) < 1e-6, diag)
    other = findfirst(d -> abs(d) < 1.0 - 1e-6, diag)
    e = first(bridges)
    m1 = NetworkModification("bridge", [ArcModification(e, -sus[e])])
    m2 = NetworkModification(
        "bridge_and_line",
        [ArcModification(e, -sus[e]), ArcModification(other, -sus[other])],
    )
    return [m1, m2]
end

@testset "worker_core shares topology, owns its factorization" begin
    sys = _workers_sys14()
    vptdf = VirtualPTDF(sys; linear_solver = "KLU")
    core = PNM.get_core(vptdf)
    w = PNM.worker_core(vptdf)
    @test w.K !== core.K
    @test w.solver_lock !== core.solver_lock
    @test w.temp_data[1] !== core.temp_data[1]
    @test w.BA === core.BA && w.A === core.A && w.valid_ix === core.valid_ix
    @test PNM.worker_core(w).BA === core.BA

    other = PNM.get_core(VirtualPTDF(sys; linear_solver = "KLU"))
    @test_throws ErrorException PNM.get_ptdf_row(vptdf, 1, other)
    @test_throws ErrorException VirtualPTDF(vptdf, other)
end

@testset "worker_core on Apple Accelerate shares the parent's lock and factorization" begin
    if PNM._has_apple_accelerate_backend()
        sys = _workers_sys14()
        vptdf = VirtualPTDF(sys; linear_solver = "AppleAccelerate")
        w = PNM.worker_core(vptdf)
        @test w.solver_lock === PNM.get_core(vptdf).solver_lock
        @test w.K === PNM.get_core(vptdf).K
        vptdf_w = VirtualPTDF(vptdf, w)
        for arc in PNM.get_arc_axis(vptdf)
            @test vptdf_w[arc, :] ≈ vptdf[arc, :] atol = 1e-10
        end
    else
        @info "Skipped Apple Accelerate worker_core test (backend unavailable)"
    end
end

@testset "Option 1 and option 2 equal the shared matrices bitwise on KLU" begin
    sys = _workers_sys14()
    vptdf = VirtualPTDF(sys; linear_solver = "KLU")
    vmodf = VirtualMODF(sys; linear_solver = "KLU")
    arc_axis = PNM.get_arc_axis(vptdf)
    n_arc = length(arc_axis)
    ctgs = collect(values(get_registered_contingencies(vmodf)))
    @test length(ctgs) == 5
    serial_ptdf = [vptdf[arc, :] for arc in arc_axis]
    serial_modf = [vmodf[a, c] for a in 1:n_arc, c in ctgs]

    # Option 1: each worker its own matrices.
    vptdf_w = VirtualPTDF(vptdf, PNM.worker_core(vptdf))
    vmodf_w = VirtualMODF(vmodf, PNM.worker_core(vmodf))
    @test length(get_registered_contingencies(vmodf_w)) == 5
    @test all(vptdf_w[arc_axis[a], :] == serial_ptdf[a] for a in 1:n_arc)
    @test all(
        vmodf_w[a, c] == serial_modf[a, k] for a in 1:n_arc, (k, c) in enumerate(ctgs)
    )

    # Option 2: fresh shared matrices, solves on a worker core.
    vptdf2 = VirtualPTDF(sys; linear_solver = "KLU")
    vmodf2 = VirtualMODF(sys; linear_solver = "KLU")
    ctgs2 = [get_registered_contingencies(vmodf2)[c.id] for c in ctgs]
    s1 = PNM.worker_core(vptdf2)
    s2 = PNM.worker_core(vmodf2)
    @test all(PNM.get_ptdf_row(vptdf2, a, s1) == serial_ptdf[a] for a in 1:n_arc)
    rows =
        [PNM.get_post_modification_ptdf_row(vmodf2, a, c, s2) for a in 1:n_arc, c in ctgs2]
    @test rows == serial_modf
    # The caches filled by option 2 are the matrix's own.
    @test length(PNM.get_woodbury_cache(vmodf2)) == 5
    @test all(
        vmodf2[a, c] == serial_modf[a, k] for a in 1:n_arc, (k, c) in enumerate(ctgs2)
    )
    wf = PNM.get_woodbury_factors(vmodf2, ctgs2[1], s2)
    @test wf === PNM.get_woodbury_cache(vmodf2)[ctgs2[1].modification]
end

# Serial results, then the same queries from concurrent tasks; returns whether they agree bitwise.
function _workers_parallel_agree(sys, ntasks::Int)
    vptdf = VirtualPTDF(sys; linear_solver = "KLU")
    vmodf = VirtualMODF(sys; linear_solver = "KLU")
    n_arc = length(PNM.get_arc_axis(vptdf))
    ctgs = collect(values(get_registered_contingencies(vmodf)))
    work = [(a, c) for a in 1:n_arc for c in eachindex(ctgs)]
    serial = [vmodf[a, ctgs[c]] for (a, c) in work]
    serial_ptdf = [vptdf[arc, :] for arc in PNM.get_arc_axis(vptdf)]
    chunks = [work[i:ntasks:end] for i in 1:ntasks]

    # Option 1.
    one = map(chunks) do chunk
        Threads.@spawn begin
            vm = VirtualMODF(vmodf, PNM.worker_core(vmodf))
            [vm[a, ctgs[c]] for (a, c) in chunk]
        end
    end
    ok1 =
        reduce(vcat, fetch.(one)) == reduce(vcat, [serial[i:ntasks:end] for i in 1:ntasks])

    # Option 2, on fresh shared matrices (caches filled concurrently).
    vptdf2 = VirtualPTDF(sys; linear_solver = "KLU")
    vmodf2 = VirtualMODF(vptdf2, sys)
    ctgs2 = [get_registered_contingencies(vmodf2)[c.id] for c in ctgs]
    two = map(chunks) do chunk
        Threads.@spawn begin
            s = PNM.worker_core(vmodf2)
            rows = [
                PNM.get_post_modification_ptdf_row(vmodf2, a, ctgs2[c], s) for
                (a, c) in chunk
            ]
            ptdf = [copy(PNM.get_ptdf_row(vptdf2, a, s)) for (a, _) in chunk]
            (rows, ptdf)
        end
    end
    res = fetch.(two)
    ok2 =
        reduce(vcat, first.(res)) == reduce(vcat, [serial[i:ntasks:end] for i in 1:ntasks])
    ok3 = all(r[2] == [serial_ptdf[a] for (a, _) in chunks[i]] for (i, r) in enumerate(res))
    return ok1 && ok2 && ok3
end

@testset "Angles and flows equal PTDF and MODF rows dotted with injections" begin
    sys = _workers_sys14()
    vptdf = VirtualPTDF(sys; linear_solver = "KLU")
    vmodf = VirtualMODF(vptdf, sys)
    n_bus = size(vptdf, 1)
    n_arc = size(vptdf, 2)
    arcs = collect(1:n_arc)
    rng = Random.Xoshiro(7)
    P = randn(rng, n_bus, 3)
    P .-= sum(P; dims = 1) ./ n_bus
    p = P[:, 1]
    rows = [PNM._compute_ptdf_row(vptdf, a) for a in arcs]
    θ = zeros(n_bus)
    @test PNM.solve_bus_angles!(θ, vptdf, p) === θ
    f = PNM.arc_flows!(zeros(n_arc), vptdf, θ, arcs)
    @test f ≈ [PNM.LinearAlgebra.dot(r, p) for r in rows] atol = 1e-12
    Θ = PNM.solve_bus_angles!(zeros(n_bus, 3), PNM.worker_core(vptdf), P)
    @test Θ[:, 1] == θ
    F = PNM.arc_flows!(zeros(n_arc, 3), vmodf, Θ, arcs)
    @test F[:, 1] == f
    @test_throws DimensionMismatch PNM.arc_flows!(zeros(n_arc - 1), vptdf, θ, arcs)

    for ctg in values(get_registered_contingencies(vmodf))
        mod = ctg.modification
        wf = PNM.compute_woodbury_factors(vmodf, mod)
        @test !wf.is_islanding
        post = PNM.arc_flows!(zeros(n_arc, 3), vmodf, Θ, arcs, wf)
        for a in arcs
            row = PNM._compute_modf_entry(vmodf, a, mod)
            @test post[a, :] ≈ vec(row' * P) atol = 1e-12
        end
        outaged = mod.arc_modifications[1].arc_index
        @test all(iszero, post[outaged, :])
        w = PNM.worker_core(vmodf)
        @test PNM.apply_woodbury_correction(w, 3, wf) ==
              PNM._compute_modf_entry(vmodf, 3, mod)
    end

    # Islanding: angles of the injections restricted to the monitored arc's island.
    BA = vmodf.BA
    for mod in _workers_island_mods(vmodf)
        wf = PNM.compute_woodbury_factors(PNM.worker_core(vmodf), mod)
        @test wf.is_islanding
        labels = wf.bus_island_labels
        bridge_labels =
            PNM.compute_woodbury_factors(vmodf, mod, PNM.BridgeLabels(BA)).bus_island_labels
        @test (labels .== labels') == (bridge_labels .== bridge_labels')
        for a in arcs
            island =
                labels[PNM.SparseArrays.rowvals(BA)[first(PNM.SparseArrays.nzrange(BA, a))]]
            p_island = p .* (labels .== island)
            θ_island = PNM.solve_bus_angles!(zeros(n_bus), vmodf, p_island)
            flow = PNM.arc_flows!(zeros(1), vmodf, θ_island, [a], wf)[1]
            row = PNM._compute_modf_entry(vmodf, a, mod)
            @test flow ≈ PNM.LinearAlgebra.dot(row, p) atol = 1e-12
        end
    end
end

# Line1 plus a lossy parallel line with a different r and x on the same arc, both under one
# outage: the group susceptance is not the exact sum of its members', so a susceptance test
# cannot tell that the whole arc is out.
@testset "A full outage of a lossy parallel group zeroes the monitored arc" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line1 = PSY.get_component(PSY.Line, sys, "Line1")
    twin = PSY.Line(;
        input_basis = u"CU",
        name = "Line1_twin",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.get_arc(line1),
        r = 0.031,
        x = 0.173,
        b = (from = 0.0, to = 0.0),
        g = (from = 0.0, to = 0.0),
        rating = 100.0,
        angle_limits = (min = -π / 2, max = π / 2),
    )
    PSY.add_component!(sys, twin)
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    PSY.add_supplemental_attribute!(sys, line1, outage)
    PSY.add_supplemental_attribute!(sys, twin, outage)
    vmodf = VirtualMODF(sys; linear_solver = "KLU")
    ctg = only(values(get_registered_contingencies(vmodf)))
    mod = ctg.modification
    e = only(mod.arc_modifications).arc_index
    @test only(mod.arc_modifications).opened == 2
    wf = PNM.compute_woodbury_factors(vmodf, mod)
    @test wf.arc_out == [true]
    b_post = PNM._post_modification_susceptance(vmodf.arc_susceptances, e, wf)
    @test !iszero(b_post)

    @test all(iszero, vmodf[e, ctg])
    n_bus = size(vmodf.BA, 1)
    P = randn(Random.Xoshiro(3), n_bus, 2)
    P .-= sum(P; dims = 1) ./ n_bus
    θ = PNM.solve_bus_angles!(zeros(n_bus, 2), vmodf, P)
    @test all(iszero, PNM.arc_flows!(zeros(1, 2), vmodf, θ, [e], wf))
end

_buffered_flow_allocations(flows, core, θ, arcs, wf, scratch) =
    @allocated PNM.arc_flows!(flows, core, θ, arcs, wf, scratch)

@testset "Buffered Woodbury flows equal the allocating form and allocate nothing" begin
    sys = _workers_sys14()
    vptdf = VirtualPTDF(sys; linear_solver = "KLU")
    vmodf = VirtualMODF(vptdf, sys)
    core = PNM.get_core(vmodf)
    n_bus = size(vptdf, 1)
    arcs = collect(1:size(vptdf, 2))
    P = randn(Random.Xoshiro(11), n_bus, 3)
    Θ = PNM.solve_bus_angles!(zeros(n_bus, 3), vmodf, P)
    rhs = zeros(length(core.valid_ix), 3)
    @test PNM.solve_bus_angles!(zeros(n_bus, 3), vmodf, P, rhs) == Θ
    @test_throws DimensionMismatch PNM.solve_bus_angles!(
        zeros(n_bus, 3),
        vmodf,
        P,
        rhs[:, 1:2],
    )
    θ = Θ[:, 1]
    # Wider than any modification here, so the views are strided as in a reused scratch.
    scratch = PNM.WoodburyFlowScratch(3, 3)
    mods = vcat(
        [c.modification for c in values(get_registered_contingencies(vmodf))],
        _workers_island_mods(vmodf),
    )
    for mod in mods
        wf = PNM.compute_woodbury_factors(core, mod)
        F = PNM.arc_flows!(zeros(length(arcs), 3), core, Θ, arcs, wf)
        @test PNM.arc_flows!(zeros(length(arcs), 3), core, Θ, arcs, wf, scratch) == F
        f = PNM.arc_flows!(zeros(length(arcs)), core, θ, arcs, wf)
        flows = zeros(length(arcs))
        @test PNM.arc_flows!(flows, core, θ, arcs, wf, scratch) == f
        @test f == F[:, 1]
        @test iszero(_buffered_flow_allocations(flows, core, θ, arcs, wf, scratch))
    end
    wf = PNM.compute_woodbury_factors(core, mods[end])
    @test_throws DimensionMismatch PNM.arc_flows!(
        zeros(length(arcs)), core, θ, arcs, wf, PNM.WoodburyFlowScratch(1),
    )
end

@testset "Parallel worker queries equal serial bitwise on KLU" begin
    @test _workers_parallel_agree(_workers_sys14(), 4)

    # The test worker runs one thread; repeat on two real threads in a child process that
    # loads this file's definitions without its testsets.
    script = """
    include($(repr(joinpath(@__DIR__, "includes.jl"))))
    function _definitions_only(ex)
        if Meta.isexpr(ex, :macrocall) && ex.args[1] === Symbol("@testset")
            return nothing
        end
        return ex
    end
    include(_definitions_only, $(repr(@__FILE__)))
    agree = _workers_parallel_agree(_workers_sys14(), 4)
    println("nthreads=", Threads.nthreads(), " agree=", agree)
    """
    project = "--project=$(Base.active_project())"
    cmd = `$(Base.julia_cmd()) -t 2 --startup-file=no $project -e $script`
    output = read(ignorestatus(cmd), String)
    @test occursin("nthreads=2 agree=true", output)
end
