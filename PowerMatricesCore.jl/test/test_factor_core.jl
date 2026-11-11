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
    end

    diag_ref = [dense_ptdf[e, arc_ax[e][1]] - dense_ptdf[e, arc_ax[e][2]] for e in 1:n_arc]
    @test isapprox(PMC.get_PTDF_A_diag(core), diag_ref; atol = 1e-12)
end
