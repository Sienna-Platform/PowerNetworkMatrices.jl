# MKL runs on x86_64 Linux and Windows only. The numeric tests run on those CI
# runners and skip on Apple Silicon.
import SparseArrays
import LinearAlgebra
import Random

_mkl_ready() = PNM._has_mkl_pardiso_ext() && Pardiso.mkl_is_available()

function _pardiso_real_block()
    return ABA_Matrix(PSB.build_system(PSB.PSITestSystems, "c_sys14")).data
end

@testset "PardisoLinSolveCache: needs the Pardiso extension" begin
    if !PNM._has_mkl_pardiso_ext()
        A = SparseArrays.sparse(1.0 * LinearAlgebra.I, 3, 3)
        @test_throws ErrorException PNM.PardisoLinSolveCache(A)
    end
end

@testset "PardisoLinSolveCache: real and complex solves match KLU" begin
    if _mkl_ready()
        rng = Random.MersenneTwister(0x5A)
        for M in (_pardiso_real_block(), _complex_ybus_block())
            T = eltype(M)
            cache = PNM.PardisoLinSolveCache(M)
            @test typeof(cache) == PNM.PardisoLinSolveCache{T}
            @test !PNM.is_factored(cache)
            PNM.full_factor!(cache, M)
            @test PNM.is_factored(cache)
            klu = PNM.klu_factorize(M)
            b = randn(rng, T, size(M, 1))
            x = copy(b)
            PNM.solve!(cache, x)
            x_klu = copy(b)
            PNM.solve!(klu, x_klu)
            @test isapprox(x, x_klu; rtol = 1e-9)
            B = randn(rng, T, size(M, 1), 3)
            X = copy(B)
            PNM.solve!(cache, X)
            @test isapprox(M * X, B; rtol = 1e-9)
        end
    else
        @info "Skipped MKL Pardiso numeric tests (MKL not available)"
    end
end

@testset "PardisoLinSolveCache: refactor, patterns, Int32 indices, deepcopy" begin
    if _mkl_ready()
        Y = _complex_ybus_block()
        cache = PNM.PardisoLinSolveCache(Y)
        PNM.full_factor!(cache, Y)
        # Same object, new values: the GA solver refactors this way.
        SparseArrays.nonzeros(Y) .*= (1.0 + 0.5im)
        PNM.numeric_refactor!(cache, Y)
        b = ones(ComplexF64, size(Y, 1))
        x = copy(b)
        PNM.solve!(cache, x)
        @test isapprox(Y * x, b; rtol = 1e-9)

        Y2 = copy(Y)
        SparseArrays.nonzeros(Y2)[1] = 0
        SparseArrays.dropzeros!(Y2)
        @test_throws ArgumentError PNM.numeric_refactor!(cache, Y2)

        A32 = SparseArrays.SparseMatrixCSC{Float64, Int32}(_pardiso_real_block())
        c32 = PNM.PardisoLinSolveCache(A32)
        PNM.full_factor!(c32, A32)
        b32 = ones(size(A32, 1))
        x32 = copy(b32)
        PNM.solve!(c32, x32)
        @test isapprox(A32 * x32, b32; rtol = 1e-9)

        @test_throws ErrorException deepcopy(cache)
    end
end

@testset "PardisoLinSolveCache: a floating complex island is detectable" begin
    if _mkl_ready()
        L = _floating_complex_ring(4)
        v = ComplexF64.(1:4)
        detected = try
            cache = PNM.PardisoLinSolveCache(L)
            PNM.full_factor!(cache, L)
            x = L * v
            PNM.solve!(cache, x)
            !(all(isfinite, x) &&
                LinearAlgebra.norm(x - v) <= 1e-6 * LinearAlgebra.norm(v))
        catch e
            typeof(e) == Pardiso.PardisoException
        end
        @test detected
    end
end
