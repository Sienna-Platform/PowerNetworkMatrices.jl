# # How to Choose a Linear Solver

# This guide helps you select the appropriate linear solver for your network matrix computations.

# ## Available Solvers

# Pass the solver name as the `linear_solver` keyword to any matrix constructor
# (`PTDF`, `LODF`, `ABA_Matrix`, …). `PowerNetworkMatrices.jl` supports four:
#
#  1. **`"KLU"`** - sparse KLU factorization. Always available (built-in
#     `KLUWrapper` submodule); the default off Apple hardware.
#  2. **`"AppleAccelerateLU"`** - sparse LU via Apple's Accelerate/libSparse.
#     Always compiled in (built-in `AccelerateWrapper` submodule) but
#     runtime-gated to macOS 15.5+ on Apple hardware, where it is the default.
#  3. **`"MKLPardiso"`** - Intel's MKL Pardiso. A weak-dependency package
#     extension: only loaded once you also add and import `Pardiso.jl`.
#  4. **`"Dense"`** - dense matrix operations, for small or debugging cases.
#
# The default is platform-dependent: `AppleAccelerateLU` on macOS 15.5+ (Apple
# hardware), `KLU` everywhere else. KLU and Apple Accelerate are always-present
# submodules — only MKL Pardiso is an optional extension.

# The examples below use a small test system loaded with `PowerSystemCaseBuilder`:

using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Choosing the Right Solver

# ### Use KLU When:
#
#   - Working with typical power systems (most cases)
#   - System size is medium to large (> 100 buses)
#   - You want good performance without special dependencies
#   - Running on any platform (Linux, macOS, Windows)
#
# Use [`PTDF`](@ref) with the KLU solver (the default off Apple hardware):

ptdf_matrix = PTDF(sys)  # platform default
## or explicitly:
ptdf_matrix = PTDF(sys; linear_solver = "KLU");

# ### Use Apple Accelerate When:
#
#   - Running on Apple-silicon macOS 15.5 or newer
#   - You want the platform-tuned sparse LU (it is the default there)
#
# Select it explicitly with:

ptdf_matrix = PTDF(sys; linear_solver = "AppleAccelerateLU");

# ### Use Dense When:
#
#   - System is very small (< 30 buses)
#   - You're debugging or validating results
#   - Matrix operations are simple and small-scale
#
# Specify the Dense solver explicitly:

ptdf_matrix = PTDF(sys; linear_solver = "Dense");

# ### Use MKLPardiso When:
#
#   - You have Intel processors
#   - Running on Linux or Windows (not available on Apple silicon)
#   - Maximum performance is critical
#   - Working with very large systems (> 1000 buses)
#
# MKLPardiso lives in a weak-dependency package extension, so first add and
# import `Pardiso.jl` to load it, then request the solver:

# ```julia
# using Pardiso   # loads the MKLPardisoExt extension
# ptdf_matrix = PTDF(sys; linear_solver = "MKLPardiso")
# ```

# ## Performance Considerations

# ### System Size

# | Buses   | Recommended Solver |
# |:------- |:------------------ |
# | < 30    | Dense or KLU       |
# | 30-1000 | KLU                |
# | > 1000  | KLU or MKLPardiso  |

# ### Platform Availability

# | Solver            | Linux | Windows | macOS             |
# |:----------------- |:----- |:------- |:----------------- |
# | KLU               | ✓     | ✓       | ✓                 |
# | AppleAccelerateLU | ✗     | ✗       | ✓ (Apple, 15.5+)  |
# | Dense             | ✓     | ✓       | ✓                 |
# | MKLPardiso        | ✓     | ✓       | ✗                 |
#
# `AppleAccelerateLU` needs no extra package — it is built in. Only
# `MKLPardiso` requires installing and importing `Pardiso.jl`.

# ## Switching Solvers

# You can easily switch between solvers to compare performance:

# ```julia
# using BenchmarkTools
#
# # Benchmark KLU
# @btime ptdf_klu = PTDF($sys; linear_solver = "KLU")
#
# # Benchmark Dense
# @btime ptdf_dense = PTDF($sys; linear_solver = "Dense")
#
# # Benchmark MKLPardiso (if available)
# @btime ptdf_mkl = PTDF($sys; linear_solver = "MKLPardiso")
# ```

# ## Troubleshooting

# ### MKLPardiso Not Available

# If you get an error when using MKLPardiso:
#
#  1. Confirm you have added and imported `Pardiso.jl` (the extension only
#     loads once `Pardiso` is available)
#  2. Verify you're on Linux or Windows (not macOS)
#  3. Check that you have Intel processors
#
# Fall back to KLU if MKLPardiso is unavailable — it is always present.

# ## Related Topics
#
#   - [How to Compute Network Matrices](@ref) - use these solvers to build matrices
#   - Reference: [Tolerance and solver settings](../reference/tolerance_and_solvers.md) -
#     the full solver-type and tolerance API
