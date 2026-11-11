import LinearAlgebra
import SparseArrays
import Random

const _AA_TEST_SEED = 0xABCDEF

function _random_unsymmetric(n::Int; density::Float64 = 0.04)
    M = SparseArrays.sprandn(n, n, density)
    return SparseArrays.sparse(M + (2 * n) * LinearAlgebra.I)
end

@testset "AccelerateWrapper: complex LU matches KLU" begin
    PNM._has_apple_accelerate_backend() || return
    rng = Random.MersenneTwister(_AA_TEST_SEED)
    Y = _complex_ybus_block()
    n = size(Y, 1)
    cache = PNM.AccelerateWrapper.aa_factorize(Y)
    @test typeof(cache) == PNM.AAFactorCache{ComplexF64}
    @test eltype(cache) == ComplexF64
    @test PNM.AccelerateWrapper.is_factored(cache)

    klu = PNM.klu_factorize(Y)
    b = randn(rng, ComplexF64, n)
    x_klu = copy(b)
    PNM.solve!(klu, x_klu)
    x_aa = copy(b)
    PNM.AccelerateWrapper.solve!(cache, x_aa)
    @test isapprox(x_aa, x_klu; rtol = 1e-10)
    @test isapprox(cache \ b, x_klu; rtol = 1e-10)

    B = randn(rng, ComplexF64, n, 3)
    X = copy(B)
    PNM.AccelerateWrapper.solve!(cache, X)
    @test isapprox(Y * X, B; rtol = 1e-10)
end

@testset "AccelerateWrapper: complex numeric_refactor! keeps the analysis" begin
    PNM._has_apple_accelerate_backend() || return
    Y = _complex_ybus_block()
    cache = PNM.AccelerateWrapper.aa_factorize(Y)
    Y2 = copy(Y)
    SparseArrays.nonzeros(Y2) .*= (1.0 + 0.5im)
    PNM.AccelerateWrapper.numeric_refactor!(cache, Y2)
    b = ones(ComplexF64, size(Y, 1))
    x = copy(b)
    PNM.AccelerateWrapper.solve!(cache, x)
    @test isapprox(Y2 * x, b; rtol = 1e-10)

    Y3 = copy(Y)
    SparseArrays.nonzeros(Y3)[1] = 0
    SparseArrays.dropzeros!(Y3)
    @test_throws ArgumentError PNM.AccelerateWrapper.numeric_refactor!(cache, Y3)
end

@testset "AAFactorCache: generic factor functions and factorization_ok" begin
    PNM._has_apple_accelerate_backend() || return
    for Y in (_random_unsymmetric(30), _complex_ybus_block())
        T = eltype(Y)
        cache = PNM.make_linear_solver_cache(PNM.AppleAccelerateLUSolver(), Y)
        @test cache isa PNM.AAFactorCache{T}
        PNM.full_factor!(cache, Y)
        @test PNM.is_factored(cache)
        PNM.numeric_refactor!(cache, Y)
        b = ones(T, size(Y, 1))
        x = copy(b)
        PNM.solve!(cache, x)
        @test isapprox(Y * x, b; rtol = 1e-9)
        @test PNM.factorization_ok(cache, Y)
    end

    # The probe sees a singular matrix that libSparse factors without an error.
    L = _floating_complex_ring(4)
    flagged = try
        cache = PNM.make_linear_solver_cache(PNM.AppleAccelerateLUSolver(), L)
        PNM.full_factor!(cache, L)
        !PNM.factorization_ok(cache, L)
    catch e
        e isa LinearAlgebra.SingularException
    end
    @test flagged
end
