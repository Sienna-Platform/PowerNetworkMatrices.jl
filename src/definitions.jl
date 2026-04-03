const YBUS_ELTYPE = ComplexF32

const SUPPORTED_LINEAR_SOLVERS = ["KLU", "MKLPardiso", "AppleAccelerate", "Dense"]

const KiB = 1024
const MiB = KiB * KiB
const GiB = MiB * KiB
const MAX_CACHE_SIZE_MiB = 100
const ROW_PERSISTENT_CACHE_WARN = 1 * GiB
const ZERO_IMPEDANCE_LINE_REACTANCE_THRESHOLD = 1e-3
const LODF_ENTRY_TOLERANCE = 1e-6
const MODF_ISLANDING_TOLERANCE = 1e-10

DEFAULT_LODF_CHUNK_SIZE = 18_000

SKIP_PARALLEL_REDUCTION_TYPES = [
    PSY.PhaseShiftingTransformer,
    ThreeWindingTransformerWinding{PSY.PhaseShiftingTransformer3W},
]

# Singleton types for linear solver dispatch, enabling compile-time method resolution.
abstract type LinearSolverType end
struct KLUSolver <: LinearSolverType end
struct DenseSolver <: LinearSolverType end
struct MKLPardisoSolver <: LinearSolverType end
struct AppleAccelerateSolver <: LinearSolverType end

const LINEAR_SOLVER_MAP = Dict{String, LinearSolverType}(
    "KLU" => KLUSolver(),
    "Dense" => DenseSolver(),
    "MKLPardiso" => MKLPardisoSolver(),
    "AppleAccelerate" => AppleAccelerateSolver(),
)

function resolve_linear_solver(s::String)
    haskey(LINEAR_SOLVER_MAP, s) ||
        error("Unsupported linear solver: $s. Supported: $(keys(LINEAR_SOLVER_MAP))")
    return LINEAR_SOLVER_MAP[s]
end
