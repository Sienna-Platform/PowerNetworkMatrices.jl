const YBUS_ELTYPE = ComplexF32

const KiB = 1024
const MiB = KiB * KiB
const GiB = MiB * KiB
const MAX_CACHE_SIZE_MiB = 100
const ROW_PERSISTENT_CACHE_WARN = 1 * GiB
const ZERO_IMPEDANCE_BRANCH_YBUS_SUSCEPTANCE_THRESHOLD = 1e4
const ZERO_IMPEDANCE_X_EPSILON = 1e-6
const LODF_ENTRY_TOLERANCE = 1e-6
const MODF_ISLANDING_TOLERANCE = 1e-10
const YBUS_DELTA_TOL = 1e-10
# At or above this bus count a default `AutoTolerance` sparsifies the on-demand
# (virtual) matrices; below it, AutoTolerance is a no-op so small systems are
# returned exactly. Sparsification only pays off — and is only wanted — at scale.
const AUTO_TOLERANCE_BUS_LIMIT = 2000

DEFAULT_LODF_CHUNK_SIZE = 18_000

# Phase shifting is a per-circuit data property, so this is data-driven via
# `PSY.is_phase_shifting` rather than a type list. Non-transformer branches never shift;
# `ThreeWindingTransformerCircuit` is covered by the `PSY.is_phase_shifting` method it
# defines for itself.
_is_phase_shifting(::PSY.ACTransmission) = false
_is_phase_shifting(
    t::Union{
        PSY.TwoWindingTransformer,
        PSY.ThreeWindingTransformer,
        ThreeWindingTransformerCircuit,
    },
) = PSY.is_phase_shifting(t)

# Singleton types for linear solver dispatch, enabling compile-time method resolution.
abstract type LinearSolverType end
struct KLUSolver <: LinearSolverType end
struct DenseSolver <: LinearSolverType end
struct MKLPardisoSolver <: LinearSolverType end
struct AppleAccelerateLUSolver <: LinearSolverType end

const SUPPORTED_LINEAR_SOLVERS =
    ("KLU", "MKLPardiso", "AppleAccelerateLU", "Dense")

@inline function resolve_linear_solver(s::String)
    s == "KLU" && return KLUSolver()
    s == "Dense" && return DenseSolver()
    s == "MKLPardiso" && return MKLPardisoSolver()
    s == "AppleAccelerateLU" && return AppleAccelerateLUSolver()
    s == "AppleAccelerate" && return AppleAccelerateLUSolver()
    error("Unsupported linear solver: $s. Supported: $SUPPORTED_LINEAR_SOLVERS")
end
