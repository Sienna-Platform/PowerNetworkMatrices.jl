# How to Choose a Linear Solver

This guide helps you select the appropriate linear solver for your network matrix computations.

## Available Solvers

`PowerNetworkMatrices.jl` supports three linear solver methods:

 1. **KLU** (default) - Sparse solver using KLU factorization
 2. **Dense** - Dense matrix operations
 3. **MKLPardiso** - Intel's MKL Pardiso solver (Intel systems only)

## Choosing the Right Solver

### Use KLU When:

  - Working with typical power systems (most cases)
  - System size is medium to large (> 100 buses)
  - You want good performance without special dependencies
  - Running on any platform (Linux, macOS, Windows)

Use [`PTDF`](@ref) with the default KLU solver:

```julia
ptdf_matrix = PTDF(sys)  # KLU is the default
# or explicitly:
ptdf_matrix = PTDF(sys; linear_solver = "KLU")
```

### Use Dense When:

  - System is very small (< 30 buses)
  - You're debugging or validating results
  - Matrix operations are simple and small-scale

Specify the Dense solver explicitly:

```julia
ptdf_matrix = PTDF(sys; linear_solver = "Dense")
```

### Use MKLPardiso When:

  - You have Intel processors
  - Running on Linux or Windows (not available on macOS)
  - Maximum performance is critical
  - Working with very large systems (> 1000 buses)

Specify the MKLPardiso solver:

```julia
ptdf_matrix = PTDF(sys; linear_solver = "MKLPardiso")
```

## Performance Considerations

### System Size

| Buses   | Recommended Solver |
|:------- |:------------------ |
| < 30    | Dense or KLU       |
| 30-1000 | KLU                |
| > 1000  | KLU or MKLPardiso  |

### Platform Availability

| Solver     | Linux | Windows | macOS |
|:---------- |:----- |:------- |:----- |
| KLU        | ✓     | ✓       | ✓     |
| Dense      | ✓     | ✓       | ✓     |
| MKLPardiso | ✓     | ✓       | ✗     |

## Switching Solvers

You can easily switch between solvers to compare performance:

```julia
using BenchmarkTools

# Benchmark KLU
@btime ptdf_klu = PTDF($sys; linear_solver = "KLU")

# Benchmark Dense
@btime ptdf_dense = PTDF($sys; linear_solver = "Dense")

# Benchmark MKLPardiso (if available)
@btime ptdf_mkl = PTDF($sys; linear_solver = "MKLPardiso")
```

## Troubleshooting

### MKLPardiso Not Available

If you get an error when using MKLPardiso:

 1. Verify you're on Linux or Windows (not macOS)
 2. Check that you have Intel processors
 3. Ensure MKL dependencies are installed

Fall back to KLU if MKLPardiso is unavailable.

## Related Topics

  - [How to Compute Network Matrices](@ref) - Learn how to use these solvers
  - [PTDF matrix](@ref) - Detailed walkthrough with solver comparisons
