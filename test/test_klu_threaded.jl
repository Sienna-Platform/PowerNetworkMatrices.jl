function _klu_small(n = 6)
    return SparseArrays.spdiagm(
        0 => collect(1.0:n) .+ 2.0,
        1 => fill(0.3, n - 1),
        -1 => fill(0.2, n - 1),
    )
end

@testset "Re-entry is allowed" begin
    A = _klu_small()
    b = ones(6)
    cache = PNM.KLULinSolveCache(A)
    PNM.full_factor!(cache, A)
    @test iszero(cache.owner[])
    PNM.full_refactor!(cache, A)
    @test iszero(cache.owner[])
    cache2 = PNM.klu_factorize(A)
    @test iszero(cache2.owner[])
    @test cache2 \ b ≈ A \ b
    @test iszero(cache2.owner[])
end

@testset "Foreign owner raises" begin
    A = _klu_small()
    b = ones(6)
    cache = PNM.klu_factorize(A)
    cache.owner[] = UInt(1)
    @test_throws ErrorException PNM.solve!(cache, copy(b))
    @test_throws ErrorException PNM.numeric_refactor!(cache, A)
    @test_throws ErrorException PNM.full_factor!(cache, A)
    cache.owner[] = UInt(0)
    @test PNM.solve!(cache, copy(b)) ≈ A \ b
end

@testset "Owner released on throw" begin
    A = SparseArrays.sparse([1, 2, 1, 2], [1, 1, 2, 2], [2.0, 1.0, 1.0, 2.0])
    cache = PNM.klu_factorize(A)
    A0 = copy(A)
    A0.nzval[1] = 0.0
    @test_throws PNM.LinearAlgebra.SingularException PNM.numeric_refactor!(cache, A0)
    @test iszero(cache.owner[])
end

@testset "Distinct caches in parallel" begin
    if Threads.nthreads() == 1
        @test_skip false
    else
        function _digest(h::UInt64, x)
            for v in x
                h = hash(v, h)
            end
            return h
        end

        function _klu_task(k::Int, Ti::Type, base, niter::Int)
            rng = Random.Xoshiro(k)
            A = SparseArrays.SparseMatrixCSC{Float64, Ti}(base)
            n = size(A, 1)
            v0 = copy(SparseArrays.nonzeros(A))
            cache = PNM.klu_factorize(A)
            h = UInt64(0)
            for it in 1:niter
                if iszero(it % 25)
                    cache = PNM.klu_factorize(A)
                    k == 1 && GC.gc()
                end
                SparseArrays.nonzeros(A) .= v0 .* (1 .+ 0.01 .* rand(rng, length(v0)))
                PNM.numeric_refactor!(cache, A)
                x = PNM.solve!(cache, rand(rng, n))
                y = PNM.tsolve!(cache, rand(rng, n))
                B = SparseArrays.sprand(rng, n, 4, 0.001)
                Z = PNM.solve_sparse(cache, B)
                h = _digest(h, x)
                h = _digest(h, y)
                h = _digest(h, Z)
            end
            return h
        end

        m = 100
        T = SparseArrays.spdiagm(
            -1 => fill(-1.0, m - 1),
            0 => fill(2.0, m),
            1 => fill(-1.0, m - 1),
        )
        I_m = SparseArrays.sparse(I, m, m)
        base = kron(I_m, T) + kron(T, I_m) + 0.5 * SparseArrays.sparse(I, m * m, m * m)
        ntasks = 2 * Threads.nthreads()
        niter = 200
        function index_type(k)
            if isodd(k)
                return Int32
            end
            return Int64
        end
        serial = [_klu_task(k, index_type(k), base, niter) for k in 1:ntasks]
        tasks = [Threads.@spawn _klu_task(k, index_type(k), base, niter) for k in 1:ntasks]
        @test fetch.(tasks) == serial
    end
end
