import LinearAlgebra
import SparseArrays

@testset "FactorCore: PTDF row matches a dense solve on a 5-bus case" begin
    bus_ax = [1, 2, 3, 4, 5]
    arc_ax = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    b = [10.0, 5.0, 8.0, 4.0, 6.0, 7.0]
    n_bus = length(bus_ax)
    n_arc = length(arc_ax)
    A = SparseArrays.spzeros(Int8, n_arc, n_bus)
    BA = SparseArrays.spzeros(Float64, n_bus, n_arc)
    for (e, (f, t)) in enumerate(arc_ax)
        A[e, f] = 1
        A[e, t] = -1
        BA[f, e] = b[e]
        BA[t, e] = -b[e]
    end
    subnetwork_axes = Dict(1 => (copy(arc_ax), copy(bus_ax)))

    core = PMC.FactorCore(
        BA,
        A,
        Set([1]),
        (arc_ax, bus_ax),
        subnetwork_axes,
        ones(Int, n_arc);
        network_data = nothing,
        linear_solver = "KLU",
        tol = 0.0,
    )
    @test isnothing(PMC.get_network_data(core))
    @test PMC.get_arc_axis(core) == arc_ax
    @test PMC.get_bus_axis(core) == bus_ax
    @test PMC.get_ref_bus(core) == [1]

    valid = 2:n_bus
    Ad = Matrix{Float64}(A)
    B = Ad' * LinearAlgebra.Diagonal(b) * Ad
    X = zeros(n_bus, n_bus)
    X[valid, valid] = inv(B[valid, valid])
    dense_ptdf = LinearAlgebra.Diagonal(b) * Ad * X

    for e in 1:n_arc
        row = PMC.with_solver(
            core.K, core.work_ba_col, core.temp_data, core.solver_lock,
        ) do K, work_ba_col, temp_data
            copy(PMC.ptdf_row!(temp_data, K, work_ba_col, core, e))
        end
        @test isapprox(row, dense_ptdf[e, :]; atol = 1e-12)
        @test iszero(row[1])
        @test PMC.ptdf_row(core, e) == row
    end
    @test_throws "Arc index 0 is outside the 6 arcs" PMC.ptdf_row(core, 0)
    @test_throws "Arc index 7 is outside the 6 arcs" PMC.ptdf_row(core, n_arc + 1)

    weights = [0.0, 1.0, 1.0, 0.0, 2.0] ./ 4
    for e in 1:n_arc
        expected = dense_ptdf[e, :] .- LinearAlgebra.dot(dense_ptdf[e, :], weights)
        row = PMC.apply_dist_slack!(PMC.ptdf_row(core, e), core, weights)
        @test isapprox(row, expected; atol = 1e-12)
        @test isapprox(LinearAlgebra.dot(row, weights), 0.0; atol = 1e-12)
    end

    # Arcs 4 and 5 are the only paths to the buses 4 and 5.
    kept = PMC.NetworkModification("one arc", [PMC.ArcModification(1, -b[1])])
    wf = PMC.compute_woodbury_factors(core, kept)
    @test !wf.is_islanding
    post = PMC.apply_woodbury_correction(core, 2, wf)
    expected = post .- LinearAlgebra.dot(post, weights)
    @test PMC.apply_dist_slack!(post, core, weights, wf) == expected
    cut = PMC.NetworkModification(
        "cut",
        [PMC.ArcModification(4, -b[4]), PMC.ArcModification(5, -b[5])],
        PMC.ShuntModification[],
        true,
    )
    wf_cut = PMC.compute_woodbury_factors(core, cut)
    @test wf_cut.is_islanding
    @test_throws "islands the network" PMC.apply_dist_slack!(
        zeros(n_bus),
        core,
        weights,
        wf_cut,
    )

    diag_ref = [dense_ptdf[e, arc_ax[e][1]] - dense_ptdf[e, arc_ax[e][2]] for e in 1:n_arc]
    @test isapprox(PMC.get_PTDF_A_diag(core), diag_ref; atol = 1e-12)
end

@testset "FactorCore: distributed slack needs one reference bus" begin
    arc_ax = [(1, 2), (3, 4)]
    bus_ax = [1, 2, 3, 4]
    core = PMC.FactorCore(
        [1.0, 1.0],
        Set([1, 3]),
        (arc_ax, bus_ax),
        Dict(1 => ([(1, 2)], [1, 2]), 3 => ([(3, 4)], [3, 4])),
        [1, 1];
        network_data = nothing,
        linear_solver = "KLU",
        tol = 0.0,
    )
    @test_throws "multiple reference buses" PMC.apply_dist_slack!(
        zeros(4),
        core,
        fill(0.25, 4),
    )
end

@testset "FactorCore: arc-susceptance constructor accepts verify_factorization" begin
    bus_ax = [1, 2, 3, 4]
    arc_ax = [(1, 2), (2, 3), (3, 4), (1, 4)]
    b = [10.0, 5.0, 8.0, 4.0]
    subnetwork_axes = Dict(1 => (copy(arc_ax), copy(bus_ax)))
    build(; kwargs...) = PMC.FactorCore(
        b,
        Set([1]),
        (arc_ax, bus_ax),
        subnetwork_axes,
        ones(Int, length(arc_ax));
        network_data = nothing,
        linear_solver = "KLU",
        tol = 0.0,
        kwargs...,
    )
    default_core = build()
    for verify in (false, true)
        core = build(; verify_factorization = verify)
        for e in eachindex(arc_ax)
            @test PMC.ptdf_row(core, e) == PMC.ptdf_row(default_core, e)
        end
    end
end

@testset "FactorCore: verify_factorization reaches the check" begin
    arc_ax = [(1, 2), (2, 3), (3, 2)]
    bus_ax = [1, 2, 3]
    build(verify::Bool) = PMC.FactorCore(
        [1.0, 5.0, -5.0],
        Set([1]),
        (arc_ax, bus_ax),
        Dict(1 => (copy(arc_ax), copy(bus_ax))),
        ones(Int, 3);
        network_data = nothing,
        linear_solver = "KLU",
        tol = 0.0,
        verify_factorization = verify,
    )
    @test_throws "arc susceptances" build(true)
    err = try
        build(false)
        nothing
    catch e
        e
    end
    @test !occursin("arc susceptances", sprint(showerror, err))
end
