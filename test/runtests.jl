# Each test_*.jl runs in its own worker (ParallelTestRunner); files share nothing.
#
#   julia --project=test test/runtests.jl                 # full suite, all jobs
#   julia --project=test test/runtests.jl test_ptdf       # filter by FILE name (startswith)
#   julia --project=test test/runtests.jl --jobs=4        # cap parallelism
#   julia --project=test test/runtests.jl --list          # list discoverable tests
@static if (Sys.ARCH === :x86_64 || Sys.ARCH === :i686) && !Sys.isapple()
    import Pkg
    Pkg.add("Pardiso")
end

using PowerNetworkMatrices
using ParallelTestRunner
import PowerSystemCaseBuilder as PSB

const TEST_DIR = @__DIR__

const DISABLED_TESTS = Set(String[])

testsuite = Dict{String, Expr}(
    splitext(f)[1] => :(include($(joinpath(TEST_DIR, f)))) for
    f in readdir(TEST_DIR) if
    startswith(f, "test_") && endswith(f, ".jl") && splitext(f)[1] ∉ DISABLED_TESTS
)

const INIT_CODE = :(include($(joinpath(TEST_DIR, "includes.jl"))))

# Worker-process env: PowerSystemCaseBuilder reads a shared serialized-system HDF5 store
# concurrently across workers — disable HDF5 file locking to avoid cross-process contention.
# ParallelTestRunner already pins Julia, OpenBLAS and BLAS to one thread per worker but not
# Accelerate, whose libSparse dispatch pool has been seen to wedge a worker on macOS.
const WORKER_ENV = [
    "HDF5_USE_FILE_LOCKING" => "FALSE",
    "RUNNING_SIENNA_TESTS" => "true",
    "VECLIB_MAXIMUM_THREADS" => "1",
]

# A cold PSB cache means every worker misses `is_serialized` and they race to write the same
# bundle directory, which `PSY.to_file` does not do atomically. Running serially populates it
# safely; once warm this costs nothing. An explicit `--jobs` wins, because a second `--jobs`
# in ARGS would survive `extract_flag!` and then be read as a test-name filter.
function _psb_cache_is_cold()
    if !isdir(PSB.SERIALIZED_DIR)
        return true
    end
    return isempty(readdir(PSB.SERIALIZED_DIR))
end

if _psb_cache_is_cold() && !any(startswith("--jobs"), ARGS)
    @info "PowerSystemCaseBuilder cache is empty; building it serially before testing."
    push!(ARGS, "--jobs=1")
end

runtests(PowerNetworkMatrices, ARGS; testsuite, init_code = INIT_CODE, env = WORKER_ENV)
