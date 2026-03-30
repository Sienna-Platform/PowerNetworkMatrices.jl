"""
Benchmark script for threaded vs non-threaded sparse matrix multiplication.

Run with:
    julia --threads=auto test/performance/benchmark_threaded_mul.jl

This script benchmarks the `threaded_sparse_mul` function against Julia's
built-in sparse × dense multiplication to quantify the performance benefit
of multi-threading for the key operations in LODF/PTDF computations.
"""

using SparseArrays
using LinearAlgebra
using Random
using Statistics

# Include only the threaded_sparse_mul module directly for isolated benchmarking
include(joinpath(@__DIR__, "..", "..", "src", "threaded_sparse_mul.jl"))

using .Main: threaded_mul!, threaded_sparse_mul

function create_incidence_matrix(n_buses::Int, n_branches::Int)
    # Create a realistic incidence matrix (each branch connects exactly 2 buses)
    I_idx = Int[]
    J_idx = Int[]
    V_val = Int8[]
    rng = Random.MersenneTwister(42)
    for branch in 1:n_branches
        from_bus = rand(rng, 1:n_buses)
        to_bus = rand(rng, setdiff(1:n_buses, from_bus))
        push!(I_idx, branch)
        push!(J_idx, from_bus)
        push!(V_val, Int8(1))
        push!(I_idx, branch)
        push!(J_idx, to_bus)
        push!(V_val, Int8(-1))
    end
    return sparse(I_idx, J_idx, V_val, n_branches, n_buses)
end

function benchmark_sparse_mul(;
    sizes = [(100, 150), (500, 750), (1000, 1500), (2000, 3000), (5000, 7500)],
    n_warmup = 3,
    n_trials = 10,
)
    println("=" ^ 80)
    println("Sparse Matrix × Dense Matrix Benchmark (A * PTDF pattern)")
    println("Julia threads: $(Threads.nthreads())")
    println("=" ^ 80)
    println()

    for (n_buses, n_branches) in sizes
        A = create_incidence_matrix(n_buses, n_branches)
        # PTDF is (n_buses × n_branches) — this is the pattern in LODF: a * ptdf
        X = randn(n_buses, n_branches)
        Y_ref = zeros(n_branches, n_branches)
        Y_threaded = zeros(n_branches, n_branches)

        # Warmup
        for _ in 1:n_warmup
            mul!(Y_ref, A, X)
            fill!(Y_threaded, 0.0)
            threaded_mul!(Y_threaded, A, X)
        end

        # Verify correctness
        fill!(Y_threaded, 0.0)
        threaded_mul!(Y_threaded, A, X)
        mul!(Y_ref, A, X)
        max_err = maximum(abs.(Y_threaded .- Y_ref))
        @assert max_err < 1e-10 "Correctness check failed! Max error: $max_err"

        # Benchmark standard mul!
        t_standard = zeros(n_trials)
        for i in 1:n_trials
            t_standard[i] = @elapsed mul!(Y_ref, A, X)
        end

        # Benchmark threaded_mul!
        t_threaded = zeros(n_trials)
        for i in 1:n_trials
            fill!(Y_threaded, 0.0)
            t_threaded[i] = @elapsed threaded_mul!(Y_threaded, A, X)
        end

        med_std = median(t_standard)
        med_thr = median(t_threaded)
        speedup = med_std / med_thr

        println("Size: A=$(size(A)), X=$(size(X))")
        println("  Standard mul!:    $(round(med_std * 1000; digits=3)) ms (median)")
        println("  Threaded mul!:    $(round(med_thr * 1000; digits=3)) ms (median)")
        println("  Speedup:          $(round(speedup; digits=2))x")
        println("  Max error:        $max_err")
        println()
    end

    println("=" ^ 80)
    println("Sparse Matrix × Dense Vector Benchmark (A * temp_data pattern)")
    println("=" ^ 80)
    println()

    for (n_buses, n_branches) in sizes
        A = create_incidence_matrix(n_buses, n_branches)
        x = randn(n_buses)
        y_ref = zeros(n_branches)
        y_threaded = zeros(n_branches)

        # Warmup
        for _ in 1:n_warmup
            mul!(y_ref, A, x)
            fill!(y_threaded, 0.0)
            threaded_mul!(y_threaded, A, x)
        end

        # Verify correctness
        fill!(y_threaded, 0.0)
        threaded_mul!(y_threaded, A, x)
        mul!(y_ref, A, x)
        max_err = maximum(abs.(y_threaded .- y_ref))
        @assert max_err < 1e-10 "Correctness check failed! Max error: $max_err"

        # Benchmark
        t_standard = zeros(n_trials)
        for i in 1:n_trials
            t_standard[i] = @elapsed mul!(y_ref, A, x)
        end

        t_threaded = zeros(n_trials)
        for i in 1:n_trials
            fill!(y_threaded, 0.0)
            t_threaded[i] = @elapsed threaded_mul!(y_threaded, A, x)
        end

        med_std = median(t_standard)
        med_thr = median(t_threaded)
        speedup = med_std / med_thr

        println("Size: A=$(size(A)), x=$(length(x))")
        println("  Standard mul!:    $(round(med_std * 1000; digits=3)) ms (median)")
        println("  Threaded mul!:    $(round(med_thr * 1000; digits=3)) ms (median)")
        println("  Speedup:          $(round(speedup; digits=2))x")
        println("  Max error:        $max_err")
        println()
    end
end

benchmark_sparse_mul()
