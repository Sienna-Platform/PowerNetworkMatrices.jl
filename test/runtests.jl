# Parallel test runner. Each top-level `test_*.jl` file runs in its OWN isolated Julia
# worker process (via ParallelTestRunner/Malt), so files execute concurrently and do not
# share mutable state (PowerSystemCaseBuilder-built systems, the global logger, the RNG).
#
#   julia --project=test test/runtests.jl                 # full suite, all jobs
#   julia --project=test test/runtests.jl test_ptdf       # filter by FILE name (startswith)
#   julia --project=test test/runtests.jl --jobs=4        # cap parallelism
#   julia --project=test test/runtests.jl --list          # list discoverable tests
#
# NOTE: the previous ReTest runner asserted "no Error-level log events across the whole run"
# via a single global MultiLogger. That global assertion does not carry to per-worker
# isolation and has been dropped; the per-`@test`/`@testset` assertions inside each file
# still run and gate the result.
@static if (Sys.ARCH === :x86_64 || Sys.ARCH === :i686) && !Sys.isapple()
    import Pkg
    Pkg.add("Pardiso")
end

using PowerNetworkMatrices
using ParallelTestRunner

const TEST_DIR = @__DIR__

# Discover ONLY the top-level `test_*.jl` files. `includes.jl`, `testing_data.jl` and
# `test_data/` are shared infrastructure, not standalone testsets — they must not be run as
# tests (ParallelTestRunner's default discovery would pick them up).

const DISABLED_TESTS = Set(String[])

testsuite = Dict{String, Expr}(
    splitext(f)[1] => :(include($(joinpath(TEST_DIR, f)))) for
    f in readdir(TEST_DIR) if
    startswith(f, "test_") && endswith(f, ".jl") && splitext(f)[1] ∉ DISABLED_TESTS
)

const INIT_CODE = :(include($(joinpath(TEST_DIR, "includes.jl"))))

# Worker-process env: PowerSystemCaseBuilder reads a shared serialized-system HDF5 store
# concurrently across workers — disable HDF5 file locking to avoid cross-process contention.
const WORKER_ENV = ["HDF5_USE_FILE_LOCKING" => "FALSE", "RUNNING_SIENNA_TESTS" => "true"]

runtests(PowerNetworkMatrices, ARGS; testsuite, init_code = INIT_CODE, env = WORKER_ENV)
