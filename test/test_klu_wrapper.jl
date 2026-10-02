import SparseArrays
import LinearAlgebra
import Random

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

@testset "KLU wrapper: tsolve! solves Aᵀ x = b" begin
    n = 30
    # Asymmetric matrix so transposing actually exercises the transpose path.
    A = SparseArrays.spdiagm(
        0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.5, n - 1),
        -1 => fill(0.1, n - 1),
    )
    cache = PNM.klu_factorize(A)

    # Single RHS.
    x = randn(n)
    b = transpose(A) * x
    y = copy(b)
    PNM.tsolve!(cache, y)
    @test isapprox(y, x, atol = 1e-10)

    # Multiple RHS, dense matrix path.
    X = randn(n, 4)
    B = transpose(A) * X
    Y = copy(B)
    PNM.tsolve!(cache, Y)
    @test isapprox(Y, X, atol = 1e-10)
end

@testset "KLU wrapper: reuse_symbolic=false rebuilds analysis on refactor" begin
    n = 25
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    x = collect(1.0:n)
    b = A * x

    cache = PNM.klu_factorize(A; reuse_symbolic = false)
    @test PNM.is_factored(cache)
    y = copy(b)
    PNM.solve!(cache, y)
    @test isapprox(y, x, atol = 1e-10)

    # A pattern change is allowed because each refactor reruns
    # `symbolic_factor!` from scratch.
    A2 = copy(A)
    A2[1, n] = 0.25
    x2 = randn(n)
    b2 = A2 * x2
    PNM.full_refactor!(cache, A2)
    y2 = copy(b2)
    PNM.solve!(cache, y2)
    @test isapprox(y2, x2, atol = 1e-9)
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
    PNM.solve_sparse!(cache, B, view(full, 3:(n + 2), :))

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

    PNM.solve_sparse!(cache, B, out; block = 32)
    # Warm call must not scale with `n * nrhs`; bound is well below that.
    alloc_warm = @allocated PNM.solve_sparse!(cache, B, out; block = 32)
    @test alloc_warm < n * size(B, 2) * sizeof(Float64) ÷ 4
end

# ---------------------------------------------------------------------------
# Int32 index-type path
# ---------------------------------------------------------------------------

@testset "KLU wrapper: Int32 real round-trip and refactor" begin
    n = 50
    rng_vals = collect(1.0:n)
    A64 = SparseArrays.spdiagm(0 => rng_vals .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    A = SparseArrays.SparseMatrixCSC{Float64, Int32}(A64)
    x = collect(1.0:n)
    b = A64 * x

    cache = PNM.klu_factorize(A)
    @test PNM.is_factored(cache)
    @test cache isa PNM.KLUWrapper.KLULinSolveCache{Float64, Int32}
    @test size(cache) == (n, n)

    y = copy(b)
    PNM.solve!(cache, y)
    @test isapprox(y, x, atol = 1e-10)

    # Refactor with new values, same pattern.
    A2_64 = SparseArrays.spdiagm(0 => rng_vals .+ 2.0,
        1 => fill(0.2, n - 1), -1 => fill(0.2, n - 1))
    A2 = SparseArrays.SparseMatrixCSC{Float64, Int32}(A2_64)
    x2 = randn(n)
    b2 = A2_64 * x2
    PNM.numeric_refactor!(cache, A2)
    y2 = copy(b2)
    PNM.solve!(cache, y2)
    @test isapprox(y2, x2, atol = 1e-9)
end

@testset "KLU wrapper: Int32 vs Int64 solves agree bit-for-bit" begin
    # KLU is deterministic; identical entries and identical permutation give
    # identical solves regardless of the C entry point's index width.
    Random.seed!(0xC0FFEE)
    n = 200
    A64 = SparseArrays.spdiagm(0 => 4.0 .+ rand(n), 1 => 0.1 .* rand(n - 1),
        -1 => 0.1 .* rand(n - 1))
    A32 = SparseArrays.SparseMatrixCSC{Float64, Int32}(A64)

    c64 = PNM.klu_factorize(A64)
    c32 = PNM.klu_factorize(A32)
    @test c64 isa PNM.KLUWrapper.KLULinSolveCache{Float64, Int64}
    @test c32 isa PNM.KLUWrapper.KLULinSolveCache{Float64, Int32}

    b = randn(n)
    y64 = copy(b)
    y32 = copy(b)
    PNM.solve!(c64, y64)
    PNM.solve!(c32, y32)
    @test y64 == y32   # bit-equal
end

@testset "KLU wrapper: Int32 solve_sparse! agrees with dense reference" begin
    Random.seed!(0xBEEF)
    n = 80
    A_dense = Matrix(SparseArrays.sprandn(n, n, 0.1)) + 4 * LinearAlgebra.I
    A = SparseArrays.SparseMatrixCSC{Float64, Int32}(SparseArrays.sparse(A_dense))
    cache = PNM.klu_factorize(A)

    B_dense = SparseArrays.sprandn(n, 20, 0.1)
    B = SparseArrays.SparseMatrixCSC{Float64, Int32}(B_dense)

    out = PNM.solve_sparse(cache, B)
    @test isapprox(out, A_dense \ Matrix(B_dense); atol = 1e-9)
end

@testset "KLU wrapper: Int32 numeric_refactor! is allocation-free" begin
    n = 100
    A64 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.05, n - 1), -1 => fill(0.05, n - 1))
    A = SparseArrays.SparseMatrixCSC{Float64, Int32}(A64)
    cache = PNM.klu_factorize(A)

    A2_64 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.5,
        1 => fill(0.07, n - 1), -1 => fill(0.07, n - 1))
    A2 = SparseArrays.SparseMatrixCSC{Float64, Int32}(A2_64)

    # Warm up specialization on the Int32 path.
    PNM.numeric_refactor!(cache, A2)
    alloc = @allocated PNM.numeric_refactor!(cache, A2)
    @test alloc == 0
end

# ---------------------------------------------------------------------------
# Performance-knob surface
# ---------------------------------------------------------------------------

@testset "KLU wrapper: condest! works on both index types" begin
    n = 60
    A64 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    A32 = SparseArrays.SparseMatrixCSC{Float64, Int32}(A64)

    for cache in (PNM.klu_factorize(A64), PNM.klu_factorize(A32))
        b = randn(n)
        y = copy(b)
        PNM.solve!(cache, y)
        @test isapprox(y, A64 \ b, atol = 1e-10)

        c = PNM.KLUWrapper.condest!(cache)
        @test c > 0
    end
end

@testset "KLU wrapper: same-nnz pattern mismatch errors and leaves the cache intact" begin
    n = 30
    D = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0)
    one_at(i, j) = SparseArrays.sparse([i], [j], [0.5], n, n)
    A = D + one_at(1, n)
    for Ti in (Int64, Int32)
        At = SparseArrays.SparseMatrixCSC{Float64, Ti}(A)
        cache = PNM.klu_factorize(At)
        colptr = copy(cache.colptr)
        rowval = copy(cache.rowval)
        # Same nnz: a different row in the same column, then a different column.
        for B in (D + one_at(2, n), D + one_at(n, 1))
            Bt = SparseArrays.SparseMatrixCSC{Float64, Ti}(B)
            @test SparseArrays.nnz(Bt) == SparseArrays.nnz(At)
            @test_throws ArgumentError PNM.numeric_refactor!(cache, Bt)
            @test_throws ArgumentError PNM.KLUWrapper.symbolic_refactor!(cache, Bt)
            @test cache.colptr == colptr
            @test cache.rowval == rowval
        end
        PNM.numeric_refactor!(cache, At)
        y = ones(n)
        PNM.solve!(cache, y)
        @test isapprox(y, A \ ones(n); atol = 1e-12)
    end
end

@testset "KLU wrapper: snapshot_values = false" begin
    n = 40
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    A2 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 2.0,
        1 => fill(0.2, n - 1), -1 => fill(0.2, n - 1))
    for Ti in (Int64, Int32)
        At = SparseArrays.SparseMatrixCSC{Float64, Ti}(A)
        A2t = SparseArrays.SparseMatrixCSC{Float64, Ti}(A2)
        on = PNM.klu_factorize(At)
        off = PNM.klu_factorize(At; snapshot_values = false)
        PNM.numeric_refactor!(on, A2t)
        PNM.numeric_refactor!(off, A2t)
        @test isempty(off.nzval)
        b = randn(n)
        y_on = copy(b)
        y_off = copy(b)
        PNM.solve!(on, y_on)
        PNM.solve!(off, y_off)
        @test y_on == y_off
        @test (@allocated PNM.numeric_refactor!(off, A2t)) == 0
        # Snapshot-dependent paths error loudly; the explicit-matrix condest! works.
        @test_throws ErrorException PNM.KLUWrapper.condest!(off)
        @test_throws ErrorException PNM.KLUWrapper._recover_factorization!(off)
        @test PNM.KLUWrapper.condest!(off, A2t) == PNM.KLUWrapper.condest!(on)
        @test PNM.KLUWrapper.condest!(on, A2t) == PNM.KLUWrapper.condest!(on)
        @test_throws ArgumentError PNM.KLUWrapper.condest!(
            off,
            SparseArrays.SparseMatrixCSC{Float64, Ti}(
                A2 + SparseArrays.sparse([1], [n], [1.0], n, n),
            ),
        )
    end
end

# ---------------------------------------------------------------------------
# Iterative refinement
# ---------------------------------------------------------------------------

@testset "KLU wrapper: solve_w_refinement matches direct solve on well-conditioned A" begin
    Random.seed!(2)
    n = 40
    A = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.1, n - 1), -1 => fill(0.1, n - 1))
    cache = PNM.klu_factorize(A)

    x_true = randn(n)
    b = A * x_true
    X = PNM.solve_w_refinement(cache, A, b)
    @test isapprox(X, x_true, atol = 1e-12)
end

@testset "KLU wrapper: solve_w_refinement! recovers ill-conditioned solve" begin
    # Build a noticeably ill-conditioned tridiagonal so a single back-solve
    # leaves residual headroom that refinement actually closes.
    Random.seed!(3)
    n = 50
    A = SparseArrays.spdiagm(0 => fill(1.0, n), 1 => fill(1.0 - 1e-10, n - 1))
    cache = PNM.klu_factorize(A)

    x_true = randn(n)
    b = A * x_true
    X = zeros(n)
    PNM.solve_w_refinement!(cache, A, X, b; tol = 1e-12)
    @test isapprox(X, x_true, atol = 1e-8)
end

@testset "KLU wrapper: solve_w_refinement requires a factored cache" begin
    n = 10
    A = SparseArrays.spdiagm(0 => 1.0:n)
    cache = PNM.KLUWrapper.KLULinSolveCache(A)  # not factored
    b = randn(n)
    @test_throws ErrorException PNM.solve_w_refinement(cache, A, b)
end

@testset "KLU wrapper: solve_w_refinement works on Int32 cache" begin
    Random.seed!(4)
    n = 30
    A64 = SparseArrays.spdiagm(0 => collect(1.0:n) .+ 1.0,
        1 => fill(0.05, n - 1), -1 => fill(0.05, n - 1))
    A = SparseArrays.SparseMatrixCSC{Float64, Int32}(A64)
    cache = PNM.klu_factorize(A)
    x_true = randn(n)
    b = A64 * x_true
    X = PNM.solve_w_refinement(cache, A, b)
    @test isapprox(X, x_true, atol = 1e-10)
end
