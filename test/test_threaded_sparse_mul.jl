@testset "Threaded Sparse Multiplication" begin
    import SparseArrays
    import LinearAlgebra

    # Build a representative sparse matrix (like a power system incidence/admittance matrix)
    n = 50
    A_dense = zeros(Float64, n, n)
    # Tridiagonal-like sparse pattern (common in power grids)
    for i in 1:n
        A_dense[i, i] = 4.0
        if i > 1
            A_dense[i, i - 1] = -1.0
        end
        if i < n
            A_dense[i, i + 1] = -1.0
        end
    end
    A_sparse = SparseArrays.sparse(A_dense)
    x = rand(Float64, n)

    @testset "threaded_mul! computes y = A * x" begin
        y_expected = A_dense * x
        y = zeros(Float64, n)
        PNM.threaded_mul!(y, A_sparse, x)
        @test isapprox(y, y_expected; atol = 1e-12)
    end

    @testset "threaded_tmul! computes y = A' * x" begin
        y_expected = A_dense' * x
        y = zeros(Float64, n)
        PNM.threaded_tmul!(y, A_sparse, x)
        @test isapprox(y, y_expected; atol = 1e-12)
    end

    @testset "threaded_mul! with non-square matrix" begin
        m, k = 40, 60
        B_dense = Matrix(SparseArrays.sprand(Float64, m, k, 0.1))
        B_sparse = SparseArrays.sparse(B_dense)
        xk = rand(Float64, k)
        y_expected = B_dense * xk
        y = zeros(Float64, m)
        PNM.threaded_mul!(y, B_sparse, xk)
        @test isapprox(y, y_expected; atol = 1e-12)
    end

    @testset "threaded_tmul! with non-square matrix" begin
        m, k = 40, 60
        B_dense = Matrix(SparseArrays.sprand(Float64, m, k, 0.1))
        B_sparse = SparseArrays.sparse(B_dense)
        xm = rand(Float64, m)
        y_expected = B_dense' * xm
        y = zeros(Float64, k)
        PNM.threaded_tmul!(y, B_sparse, xm)
        @test isapprox(y, y_expected; atol = 1e-12)
    end

    @testset "threaded_mul! with Int8 sparse matrix (incidence-like)" begin
        # Incidence matrices in this codebase are SparseMatrixCSC{Int8, Int}
        nbus = 30
        nline = 45
        I_vals = Int8[]
        J_vals = Int[]
        V_vals = Int8[]
        for line in 1:nline
            from = rand(1:nbus)
            to = rand(setdiff(1:nbus, from))
            push!(I_vals, Int8(from))
            push!(J_vals, line)
            push!(V_vals, Int8(1))
            push!(I_vals, Int8(to))
            push!(J_vals, line)
            push!(V_vals, Int8(-1))
        end
        A_inc = SparseArrays.sparse(Int.(I_vals), J_vals, V_vals, nbus, nline)
        xline = rand(Float64, nline)
        y_expected = Float64.(Matrix(A_inc)) * xline
        y = zeros(Float64, nbus)
        PNM.threaded_mul!(y, A_inc, xline)
        @test isapprox(y, y_expected; atol = 1e-12)
    end

    @testset "threaded_sparse_dense_mul! computes Y = A * X" begin
        m, k, p = 30, 40, 20
        A_sd = SparseArrays.sprand(Float64, m, k, 0.15)
        X_sd = rand(Float64, k, p)
        Y_expected = Matrix(A_sd) * X_sd
        Y = zeros(Float64, m, p)
        PNM.threaded_sparse_dense_mul!(Y, A_sd, X_sd)
        @test isapprox(Y, Y_expected; atol = 1e-12)
    end

    @testset "threaded_sparse_dense_mul! with Int8 incidence matrix" begin
        # Matches LODF pattern: Int8 incidence × Float64 PTDF
        nbus = 20
        nline = 30
        I_vals = Int[]
        J_vals = Int[]
        V_vals = Int8[]
        for line in 1:nline
            from = rand(1:nbus)
            to = rand(setdiff(1:nbus, from))
            push!(I_vals, from)
            push!(J_vals, line)
            push!(V_vals, Int8(1))
            push!(I_vals, to)
            push!(J_vals, line)
            push!(V_vals, Int8(-1))
        end
        A_inc = SparseArrays.sparse(I_vals, J_vals, V_vals, nbus, nline)
        ptdf_mock = rand(Float64, nbus, nline)
        Y_expected = Float64.(Matrix(A_inc)) * ptdf_mock
        Y = zeros(Float64, nline, nline)
        # Note: a * ptdf where a is (nline x nbus) incidence, ptdf is (nbus x nline)
        # For this test use the actual dimensions
        A_t = SparseArrays.sparse(J_vals, I_vals, V_vals, nline, nbus)
        Y_expected2 = Float64.(Matrix(A_t)) * ptdf_mock
        Y2 = zeros(Float64, nline, nline)
        PNM.threaded_sparse_dense_mul!(Y2, A_t, ptdf_mock)
        @test isapprox(Y2, Y_expected2; atol = 1e-12)
    end

    @testset "above-threshold matrices exercise threaded paths" begin
        # Matrices above THREADED_MUL_MIN_SIZE (1000) will use the threaded
        # branches when Threads.nthreads() > 1.  When CI runs single-threaded
        # they fall back to the scalar loop, which uses identical arithmetic.
        big_n = PNM.THREADED_MUL_MIN_SIZE + 1
        A_big = SparseArrays.sprandn(Float64, big_n, big_n, 5.0 / big_n)
        x_big = rand(Float64, big_n)

        y_expected = Matrix(A_big) * x_big
        y = zeros(Float64, big_n)
        PNM.threaded_mul!(y, A_big, x_big)
        @test isapprox(y, y_expected; atol = 1e-10)

        yt_expected = Matrix(A_big)' * x_big
        yt = zeros(Float64, big_n)
        PNM.threaded_tmul!(yt, A_big, x_big)
        @test isapprox(yt, yt_expected; atol = 1e-10)

        p_big = PNM.THREADED_MUL_MIN_SIZE + 1
        X_big = rand(Float64, big_n, p_big)
        Y_expected = Matrix(A_big) * X_big
        Y_big = zeros(Float64, big_n, p_big)
        PNM.threaded_sparse_dense_mul!(Y_big, A_big, X_big)
        @test isapprox(Y_big, Y_expected; atol = 1e-10)
    end

    @testset "dimension mismatch errors" begin
        @test_throws AssertionError PNM.threaded_mul!(zeros(n + 1), A_sparse, x)
        @test_throws AssertionError PNM.threaded_mul!(zeros(n), A_sparse, zeros(n + 1))
        @test_throws AssertionError PNM.threaded_tmul!(zeros(n + 1), A_sparse, x)
        @test_throws AssertionError PNM.threaded_tmul!(zeros(n), A_sparse, zeros(n + 1))
    end
end
