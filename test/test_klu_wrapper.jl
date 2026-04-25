@testset "KLU wrapper: real round-trip and refactor" begin
    n = 50
    rng_vals = collect(1.0:n)
    A = SparseArrays.spdiagm(0 => rng_vals .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    x = collect(1.0:n)
    b = A * x

    cache = PNM.klu_factorize(A)
    @test PNM.is_factored(cache)
    @test size(cache) == (n, n)

    y = copy(b)
    PNM.solve!(cache, y)
    @test isapprox(y, x, atol = 1e-10)

    # Refactor with new values, same pattern.
    A2 = SparseArrays.spdiagm(0 => rng_vals .+ 2.0,
        1 => fill(0.2, n - 1), -1 => fill(0.2, n - 1))
    x2 = randn(n)
    b2 = A2 * x2
    PNM.numeric_refactor!(cache, A2)
    y2 = copy(b2)
    PNM.solve!(cache, y2)
    @test isapprox(y2, x2, atol = 1e-9)

    # Pattern change should be rejected with check_pattern=true.
    A3 = copy(A2)
    A3[2, n] = 1.0
    @test_throws ArgumentError PNM.numeric_refactor!(cache, A3)
end

@testset "KLU wrapper: solve_sparse! matches dense path" begin
    n, m = 30, 40
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)

    # Build a deliberately sparse RHS.
    rows = vcat([1, 5, 12, n], [3, 9, n - 1])
    cols = vcat(fill(2, 4), fill(7, 3))
    vals = [1.0, -2.0, 3.5, -0.5, 2.0, -1.0, 1.5]
    B = SparseArrays.sparse(rows, cols, vals, n, m)

    out = PNM.solve_sparse(cache, B)
    Bdense = Matrix(B)
    PNM.solve!(cache, Bdense)
    @test isapprox(out, Bdense, atol = 1e-10)
end

@testset "KLU wrapper: solve_sparse! zeros empty columns" begin
    n = 20
    A = SparseArrays.spdiagm(0 => fill(2.0, n), 1 => fill(-1.0, n - 1),
        -1 => fill(-1.0, n - 1))
    cache = PNM.klu_factorize(A)
    rows = [3, 7]
    cols = [3, 3]
    vals = [1.0, 2.0]
    B = SparseArrays.sparse(rows, cols, vals, n, 5)

    out = PNM.solve_sparse(cache, B)
    @test all(==(0.0), out[:, 1])
    @test all(==(0.0), out[:, 2])
    @test all(==(0.0), out[:, 4])
    @test all(==(0.0), out[:, 5])
    @test !all(==(0.0), out[:, 3])

    Bdense = Matrix(B)
    PNM.solve!(cache, Bdense)
    @test isapprox(out, Bdense, atol = 1e-12)
end

@testset "KLU wrapper: backslash" begin
    n = 25
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)
    x = randn(n)
    b = A * x
    @test isapprox(cache \ b, x, atol = 1e-10)
end

@testset "KLU wrapper: pool basic" begin
    n = 30
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 4)
    @test PNM.nworkers(pool) == 4
    @test size(pool) == (n, n)

    x = randn(n)
    b = A * x
    result = PNM.with_worker(pool) do cache, idx
        @test 1 <= idx <= 4
        y = copy(b)
        PNM.solve!(cache, y)
        return y
    end
    @test isapprox(result, x, atol = 1e-10)
end

@testset "KLU wrapper: pool concurrent solves" begin
    n = 60
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 2.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    nw = max(2, min(Threads.nthreads(), 4))
    pool = PNM.KLULinSolvePool(A; nworkers = nw)

    nrhs = 32
    Xs = [randn(n) for _ in 1:nrhs]
    Bs = [A * x for x in Xs]
    Ys = Vector{Vector{Float64}}(undef, nrhs)

    Threads.@threads for k in 1:nrhs
        PNM.with_worker(pool) do cache, _idx
            y = copy(Bs[k])
            PNM.solve!(cache, y)
            Ys[k] = y
        end
    end

    for k in 1:nrhs
        @test isapprox(Ys[k], Xs[k], atol = 1e-9)
    end
end

@testset "KLU wrapper: pool numeric_refactor!" begin
    n = 20
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 2)

    A2 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 5.0,
        1 => fill(0.2, n - 1), -1 => fill(0.2, n - 1))
    PNM.numeric_refactor!(pool, A2)

    x = randn(n)
    b = A2 * x
    y = PNM.with_worker(pool) do cache, _idx
        out = copy(b)
        PNM.solve!(cache, out)
        return out
    end
    @test isapprox(y, x, atol = 1e-9)
end

@testset "KLU wrapper: ComplexF64 path" begin
    n = 12
    A = SparseArrays.spdiagm(
        0 => ComplexF64.(collect(1.0:n) .+ 1.0im),
        1 => fill(0.1 + 0.0im, n - 1),
        -1 => fill(0.1 + 0.0im, n - 1),
    )
    x = ComplexF64.(randn(n) .+ 1im .* randn(n))
    b = A * x

    cache = PNM.klu_factorize(A)
    y = copy(b)
    PNM.solve!(cache, y)
    @test isapprox(y, x, atol = 1e-10)

    # Sparse RHS path on the complex cache.
    rows = [1, n]
    cols = [1, 2]
    vals = ComplexF64.([1.0 + 0.5im, 2.0 - 0.3im])
    B = SparseArrays.sparse(rows, cols, vals, n, 3)
    out = PNM.solve_sparse(cache, B)
    Bdense = Matrix(B)
    PNM.solve!(cache, Bdense)
    @test isapprox(out, Bdense, atol = 1e-10)
end

@testset "KLU wrapper: solve_sparse! into a view" begin
    n = 15
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 5.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)
    nrhs = 6
    B = SparseArrays.sprand(n, nrhs, 0.2)

    full = zeros(n + 4, nrhs)
    PNM.solve_sparse!(cache, B; out = view(full, 3:(n + 2), :))

    Bdense = Matrix(B)
    PNM.solve!(cache, Bdense)
    @test isapprox(full[3:(n + 2), :], Bdense, atol = 1e-10)
    @test all(==(0.0), full[1:2, :])
    @test all(==(0.0), full[(n + 3):end, :])
end
