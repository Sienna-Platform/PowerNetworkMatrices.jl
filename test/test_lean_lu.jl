import SparseArrays
import LinearAlgebra
import Random

# Structurally nonsymmetric, diagonally dominant, irreducible (has a full cycle).
function _lean_test_matrix(rng, n, ::Type{Ti}) where {Ti}
    A =
        SparseArrays.sprand(rng, n, n, 4 / n) +
        SparseArrays.spdiagm(1 => ones(n - 1), -(n - 1) => ones(1)) +
        SparseArrays.spdiagm(0 => fill(10.0, n))
    return SparseArrays.SparseMatrixCSC{Float64, Ti}(A)
end

function _lean_plan(A; btf = 0)
    cache = PNM.KLULinSolveCache(A)
    cache.common[].btf = btf
    PNM.full_factor!(cache, A)
    return PNM.LeanLUPlan(cache, A), cache
end

@testset "LeanLU: refactor + solve match KLU ($Ti)" for Ti in (Int32, Int64)
    rng = Random.MersenneTwister(1)
    n = 300
    A = _lean_test_matrix(rng, n, Ti)
    plan, cache = _lean_plan(A)
    ws = PNM.LeanLUWorkspace(plan)
    @test 0 < plan.rcond0 <= 1
    @test PNM.lean_solve! === PNM.lean_ldiv!

    b = randn(rng, n)
    @test PNM.lean_refactor!(ws, plan, SparseArrays.nonzeros(A)) ≈ 1.0
    @test PNM.lean_ldiv!(ws, plan, copy(b)) ≈ A \ b

    # New values, same pattern: lean agrees with a KLU refactor + solve.
    for _ in 1:5
        A2 = copy(A)
        SparseArrays.nonzeros(A2) .*= 1 .+ 0.2 .* randn(rng, SparseArrays.nnz(A2))
        @test PNM.lean_refactor!(ws, plan, SparseArrays.nonzeros(A2)) >= 1e-3
        x = PNM.lean_ldiv!(ws, plan, copy(b))
        PNM.numeric_refactor!(cache, A2)
        @test x ≈ PNM.solve!(cache, copy(b)) rtol = 1e-10
        @test all(iszero, ws.x)
    end
end

@testset "LeanLU: one shared plan, concurrent tasks" begin
    rng = Random.MersenneTwister(2)
    n = 200
    A = _lean_test_matrix(rng, n, Int64)
    plan, _ = _lean_plan(A)
    nsets = 64
    vals = [
        SparseArrays.nonzeros(A) .* (1 .+ 0.1 .* randn(rng, SparseArrays.nnz(A)))
        for _ in 1:nsets
    ]
    b = randn(rng, n)

    serial = map(vals) do v
        ws = PNM.LeanLUWorkspace(plan)
        PNM.lean_refactor!(ws, plan, v)
        PNM.lean_ldiv!(ws, plan, copy(b))
    end
    threaded = Vector{Vector{Float64}}(undef, nsets)
    Threads.@threads for i in 1:nsets
        ws = PNM.LeanLUWorkspace(plan)
        PNM.lean_refactor!(ws, plan, vals[i])
        threaded[i] = PNM.lean_ldiv!(ws, plan, copy(b))
    end
    @test threaded == serial
    for i in 1:nsets
        Ai = copy(A)
        SparseArrays.nonzeros(Ai) .= vals[i]
        @test serial[i] ≈ Ai \ b
    end
end

@testset "LeanLU: rejection and errors" begin
    rng = Random.MersenneTwister(3)
    n = 50
    A = _lean_test_matrix(rng, n, Int64)
    plan, cache = _lean_plan(A)
    ws = PNM.LeanLUWorkspace(plan)

    # Zero pivots report 0 and leave the scratch clean.
    @test PNM.lean_refactor!(ws, plan, zeros(SparseArrays.nnz(A))) == 0.0
    @test all(iszero, ws.x)
    @test PNM.lean_refactor!(ws, plan, fill(NaN, SparseArrays.nnz(A))) == 0.0
    @test all(iszero, ws.x)

    @test_throws DimensionMismatch PNM.lean_refactor!(ws, plan, zeros(3))
    @test_throws DimensionMismatch PNM.lean_ldiv!(ws, plan, zeros(3))
    other, _ = _lean_plan(_lean_test_matrix(rng, n + 1, Int64))
    @test_throws DimensionMismatch PNM.lean_ldiv!(ws, other, zeros(n + 1))
    @test_throws ArgumentError PNM.LeanLUPlan(PNM.KLULinSolveCache(A), A)

    # Reducible matrix: BTF leaves off-diagonal-block entries, so no flat LU plan.
    U = SparseArrays.spdiagm(0 => fill(2.0, 5), 1 => ones(4))
    @test_throws ArgumentError _lean_plan(U; btf = 1)
    planU, _ = _lean_plan(U; btf = 0)
    wsU = PNM.LeanLUWorkspace(planU)
    PNM.lean_refactor!(wsU, planU, SparseArrays.nonzeros(U))
    @test PNM.lean_ldiv!(wsU, planU, ones(5)) ≈ U \ ones(5)

    # Uncoupled blocks are fine with BTF on.
    D = SparseArrays.blockdiag(A, A)
    planD, _ = _lean_plan(D; btf = 1)
    wsD = PNM.LeanLUWorkspace(planD)
    PNM.lean_refactor!(wsD, planD, SparseArrays.nonzeros(D))
    @test PNM.lean_ldiv!(wsD, planD, ones(2n)) ≈ D \ ones(2n)

    # A must have the pattern the cache factored.
    A2 = copy(A)
    A2[1, n] += 1.0
    A2[1, n] == 1.0 || (A2[2, n] = 1.0)
    @test_throws ArgumentError PNM.LeanLUPlan(cache, A2)
end

@testset "LeanLU: refactor and solve allocate nothing" begin
    rng = Random.MersenneTwister(4)
    A = _lean_test_matrix(rng, 100, Int64)
    plan, _ = _lean_plan(A)
    ws = PNM.LeanLUWorkspace(plan)
    Ax = SparseArrays.nonzeros(A)
    b = randn(rng, 100)
    PNM.lean_refactor!(ws, plan, Ax)
    PNM.lean_ldiv!(ws, plan, b)
    @test (@allocated PNM.lean_refactor!(ws, plan, Ax)) == 0
    @test (@allocated PNM.lean_ldiv!(ws, plan, b)) == 0
end

@testset "LeanLUCache: KLU first, then lean ($Ti)" for Ti in (Int32, Int64)
    rng = Random.MersenneTwister(5)
    n = 200
    A = _lean_test_matrix(rng, n, Ti)
    b = randn(rng, n)
    c = PNM.LeanLUCache(A)
    @test c isa PNM.LinearSolverCache
    PNM.full_factor!(c, A)
    @test PNM.is_factored(c)
    @test !c.lean_factored
    @test PNM.solve!(c, copy(b)) ≈ A \ b

    for _ in 1:3
        A2 = copy(A)
        SparseArrays.nonzeros(A2) .*= 1 .+ 0.2 .* randn(rng, SparseArrays.nnz(A2))
        PNM.numeric_refactor!(c, A2)
        @test c.lean_factored
        @test PNM.solve!(c, copy(b)) ≈ A2 \ b
        B = randn(rng, n, 3)
        @test PNM.solve!(c, copy(B)) ≈ A2 \ B
        @test PNM.solve_w_refinement(c, A2, b) ≈ A2 \ b
    end
end

@testset "LeanLUCache: rejected pivots fall back to KLU until symbolic_factor!" begin
    A1 = SparseArrays.sparse([10.0 1.0; 1.0 10.0])
    A2 = SparseArrays.sparse([1e-12 1.0; 1.0 1.0])  # frozen pivot A2[1,1] is tiny
    b = [1.0, 2.0]
    c = PNM.LeanLUCache(A1)
    PNM.full_factor!(c, A1)
    PNM.numeric_refactor!(c, A1)
    @test c.lean_factored

    PNM.numeric_refactor!(c, A2)
    @test !c.lean_factored
    @test !c.use_lean
    @test PNM.solve!(c, copy(b)) ≈ Matrix(A2) \ b
    PNM.numeric_refactor!(c, A1)
    @test !c.lean_factored
    @test PNM.solve!(c, copy(b)) ≈ Matrix(A1) \ b

    PNM.symbolic_factor!(c, A1)
    PNM.numeric_refactor!(c, A1)
    PNM.numeric_refactor!(c, A1)
    @test c.lean_factored

    Z = copy(A1)
    SparseArrays.nonzeros(Z) .= 0.0
    @test_throws LinearAlgebra.SingularException PNM.numeric_refactor!(c, Z)
    @test_throws ArgumentError PNM.numeric_refactor!(
        c,
        SparseArrays.sparse(1.0 * LinearAlgebra.I, 2, 2),
    )
end

@testset "LeanLUCache: lean refactor and solve allocate nothing" begin
    rng = Random.MersenneTwister(6)
    A = _lean_test_matrix(rng, 100, Int64)
    b = randn(rng, 100)
    c = PNM.full_factor!(PNM.LeanLUCache(A), A)
    PNM.numeric_refactor!(c, A)
    PNM.solve!(c, b)
    @test (@allocated PNM.numeric_refactor!(c, A)) == 0
    @test (@allocated PNM.solve!(c, b)) == 0
end
