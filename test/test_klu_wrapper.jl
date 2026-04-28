import SparseArrays

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

@testset "KLU wrapper: solve_sparse matches dense path" begin
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

@testset "KLU wrapper: solve_sparse zeros empty columns" begin
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

@testset "KLU wrapper: solve_sparse! block chunking matches monolithic solve" begin
    # Many sparse columns with small block must give the same answer as a
    # large block: defends the bottleneck-preservation contract from the
    # original plan (n × block working set, not n × nrhs).
    n = 80
    nrhs = 250
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 5.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)
    B = SparseArrays.sprand(n, nrhs, 0.05)

    Xbig = PNM.solve_sparse(cache, B; block = 256)
    Xsmall = PNM.solve_sparse(cache, B; block = 7)
    @test isapprox(Xbig, Xsmall, atol = 1e-12)

    Bdense = Matrix(B)
    PNM.solve!(cache, Bdense)
    @test isapprox(Xsmall, Bdense, atol = 1e-10)
end

@testset "KLU wrapper: warm solve! / numeric_refactor! are non-allocating" begin
    n = 40
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)
    b = randn(n)

    # Warm-up to compile.
    y = copy(b)
    PNM.solve!(cache, y)
    A2 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 2.0,
        1 => fill(0.2, n - 1), -1 => fill(0.2, n - 1))
    PNM.numeric_refactor!(cache, A2)

    # Hot path: solve! mutates the buffer in place; no allocations.
    y .= b
    alloc_solve = @allocated PNM.solve!(cache, y)
    @test alloc_solve == 0

    alloc_refactor = @allocated PNM.numeric_refactor!(cache, A2)
    @test alloc_refactor == 0
end

@testset "KLU wrapper: solve_sparse! warm working set is bounded" begin
    n = 50
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)
    B = SparseArrays.sprand(n, 100, 0.05)
    out = Matrix{Float64}(undef, n, 100)

    PNM.solve_sparse!(cache, B; out = out, block = 32)
    # Warm call must not scale with `n * nrhs`; bound is well below that.
    alloc_warm = @allocated PNM.solve_sparse!(cache, B; out = out, block = 32)
    @test alloc_warm < n * size(B, 2) * sizeof(Float64) ÷ 4
end

@testset "KLU wrapper: pool refactor round-trip preserves parallel correctness" begin
    n = 40
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    nw = max(2, min(Threads.nthreads(), 4))
    pool = PNM.KLULinSolvePool(A; nworkers = nw)

    # Refactor with new values and verify every worker sees the new factor by
    # forcing concurrent solves across all workers.
    A2 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 7.0,
        1 => fill(0.3, n - 1), -1 => fill(0.3, n - 1))
    PNM.numeric_refactor!(pool, A2)

    nrhs = 4 * nw
    Xs = [randn(n) for _ in 1:nrhs]
    Bs = [A2 * x for x in Xs]
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

@testset "KLU wrapper: pool all-fail refactor surfaces error and blocks acquire" begin
    n = 10
    A = SparseArrays.spdiagm(0 => fill(2.0, n), 1 => fill(0.1, n - 1),
        -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 2)
    @test PNM.n_valid(pool) == 2

    # Singular A: every worker fails. No auto-reset (would just fail again).
    Asingular = SparseArrays.spdiagm(0 => zeros(n), 1 => fill(0.1, n - 1),
        -1 => fill(0.1, n - 1))
    @test_throws Exception PNM.numeric_refactor!(pool, Asingular)
    @test PNM.n_valid(pool) == 0

    # Acquire is rejected so callers can't deadlock or get stale workers.
    @test_throws ErrorException PNM.with_worker(pool) do _cache, _idx
        nothing
    end
end

@testset "KLU wrapper: reset! recovers a pool whose workers all have failed factorizations" begin
    n = 10
    A = SparseArrays.spdiagm(0 => fill(2.0, n), 1 => fill(0.1, n - 1),
        -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 2)

    Asingular = SparseArrays.spdiagm(0 => zeros(n), 1 => fill(0.1, n - 1),
        -1 => fill(0.1, n - 1))
    @test_throws Exception PNM.numeric_refactor!(pool, Asingular)
    @test PNM.n_valid(pool) == 0

    # Recovery: caller supplies a known-good matrix.
    Agood = SparseArrays.spdiagm(0 => fill(3.0, n), 1 => fill(0.2, n - 1),
        -1 => fill(0.2, n - 1))
    PNM.reset!(pool, Agood)
    @test PNM.n_valid(pool) == 2

    x = randn(n)
    b = Agood * x
    y = PNM.with_worker(pool) do cache, _idx
        out = copy(b)
        PNM.solve!(cache, out)
        return out
    end
    @test isapprox(y, x, atol = 1e-9)
end

@testset "KLU wrapper: pool degraded mode (≤ 50% failed)" begin
    n = 12
    A = SparseArrays.spdiagm(0 => fill(2.0, n), 1 => fill(0.1, n - 1),
        -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 4)

    # Manually break 1 of 4 workers (25% failure, below the 50% threshold).
    # Finalizing nulls its symbolic handle; the next refactor on this worker
    # throws "call symbolic_factor!" without touching any others.
    Base.finalize(pool.workers[1])

    A2 = SparseArrays.spdiagm(0 => fill(3.0, n), 1 => fill(0.2, n - 1),
        -1 => fill(0.2, n - 1))
    @test_throws Exception PNM.numeric_refactor!(pool, A2)
    # Survivors stay valid; the failed worker is held out.
    @test PNM.n_valid(pool) == 3

    # Pool keeps serving solves through survivors.
    x = randn(n)
    b = A2 * x
    y = PNM.with_worker(pool) do cache, _idx
        out = copy(b)
        PNM.solve!(cache, out)
        return out
    end
    @test isapprox(y, x, atol = 1e-9)

    # `reset!` brings the broken worker back.
    PNM.reset!(pool, A2)
    @test PNM.n_valid(pool) == 4
end

@testset "KLU wrapper: pool auto-reset (> 50% failed but not all)" begin
    n = 10
    A = SparseArrays.spdiagm(0 => fill(2.0, n), 1 => fill(0.1, n - 1),
        -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 4)

    # Break 3 of 4 workers (75% failure, above the 50% threshold).
    for w in 1:3
        Base.finalize(pool.workers[w])
    end

    A2 = SparseArrays.spdiagm(0 => fill(5.0, n), 1 => fill(0.3, n - 1),
        -1 => fill(0.3, n - 1))
    # Auto-reset succeeds because A2 is well-formed; full_factor! on every
    # worker (including the survivor) restores a uniform good state.
    PNM.numeric_refactor!(pool, A2)
    @test PNM.n_valid(pool) == 4

    x = randn(n)
    b = A2 * x
    y = PNM.with_worker(pool) do cache, _idx
        out = copy(b)
        PNM.solve!(cache, out)
        return out
    end
    @test isapprox(y, x, atol = 1e-9)
end

@testset "KLU wrapper: pool numeric_refactor! blocks for in-flight worker" begin
    # Regression for the non-blocking-drain bug: with `_drain_available!`
    # using `isready`, a worker checked out at the moment of the refactor
    # would be silently skipped and continue serving solves with the stale
    # factorization. The fix makes the drain block on `take!` until every
    # valid worker is back in the pool's hand, so this test:
    #   1. holds a worker via `with_worker` from a spawned task,
    #   2. fires `numeric_refactor!` with a different matrix from another
    #      task and asserts that it has not completed while the holder is
    #      still in the critical section,
    #   3. releases the holder, drives parallel solves, and verifies every
    #      result is consistent with the new matrix (not the old).
    n = 30
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    pool = PNM.KLULinSolvePool(A; nworkers = 2)

    holder_started = Base.Event()
    holder_release = Base.Event()
    holder = Threads.@spawn PNM.with_worker(pool) do _cache, _idx
        notify(holder_started)
        wait(holder_release)
    end
    wait(holder_started)

    A2 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 5.0,
        1 => fill(0.2, n - 1), -1 => fill(0.2, n - 1))
    refactor_task = Threads.@spawn PNM.numeric_refactor!(pool, A2)

    # Give the refactor task time to drain available workers and block on
    # `take!` for the held one. Without the blocking drain it would have
    # completed by now.
    sleep(0.2)
    @test !istaskdone(refactor_task)

    notify(holder_release)
    fetch(holder)
    fetch(refactor_task)

    # Drive enough parallel solves that every worker — including the
    # previously-held one — is exercised. With the bug, that worker would
    # still hold A's factor and produce wrong answers against A2 RHS.
    nrhs = 32
    Xs = [randn(n) for _ in 1:nrhs]
    Bs = [A2 * x for x in Xs]
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
