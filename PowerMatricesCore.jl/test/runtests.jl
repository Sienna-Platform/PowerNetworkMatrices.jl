using Test
using PowerMatricesCore

const TEST_DIR = @__DIR__

@testset "PowerMatricesCore" begin
    for f in sort!(readdir(TEST_DIR))
        if startswith(f, "test_") && endswith(f, ".jl")
            @testset "$f" begin
                include(joinpath(TEST_DIR, f))
            end
        end
    end
end
