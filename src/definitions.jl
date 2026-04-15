const YBUS_ELTYPE = ComplexF32

const KiB = 1024
const MiB = KiB * KiB
const GiB = MiB * KiB
const MAX_CACHE_SIZE_MiB = 100
const ROW_PERSISTENT_CACHE_WARN = 1 * GiB
const ZERO_IMPEDANCE_LINE_REACTANCE_THRESHOLD = 1e-3
const LODF_ENTRY_TOLERANCE = 1e-6
const MODF_ISLANDING_TOLERANCE = 1e-10
const MODF_ISLANDING_REL_TOL = 1e-12
const YBUS_DELTA_TOL = 1e-10

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

const SUPPORTED_LINEAR_SOLVERS = ("KLU", "MKLPardiso", "AppleAccelerate", "Dense")

@inline function resolve_linear_solver(s::String)
    s == "KLU" && return KLUSolver()
    s == "Dense" && return DenseSolver()
    s == "MKLPardiso" && return MKLPardisoSolver()
    s == "AppleAccelerate" && return AppleAccelerateSolver()
    error("Unsupported linear solver: $s. Supported: $SUPPORTED_LINEAR_SOLVERS")
end
