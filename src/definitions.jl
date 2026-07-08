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

# Star-leg reactance floor (pu, system base). A star leg whose reactance derives to ~0 from
# the pairwise impedances is floored to this value. Recovered verbatim from the pre-refactor
# PowerFlowFileParser parse-time rule (`ZERO_IMPEDANCE_REACTANCE_THRESHOLD`, per PSS/E Data
# Formats §1.15.1 "Three-Winding Transformer Notes"). The old parser floored the SYSTEM-BASE
# star reactance before re-converting to device base; the SU star reactance the wrapper
# derives is the equivalent quantity, so the floor is applied there.
const STAR_LEG_REACTANCE_FLOOR = 1e-4

# Detection tolerance for the flooring above; matches the old parser's `atol = eps(Float32)`.
const STAR_LEG_ZERO_REACTANCE_ATOL = eps(Float32)

# A phase-shifting branch must not be folded into a parallel-equivalent group: the parallel
# susceptance model cannot represent a per-branch phase shift. Phase shifting is a per-winding
# data property in the new transformer model (no distinct PST type remains), so the skip is
# data-driven via `PSY.is_phase_shifting` rather than a type list.
_skip_parallel_reduction(::PSY.ACTransmission) = false
_skip_parallel_reduction(b::PSY.TwoWindingTransformer) = PSY.is_phase_shifting(b)
_skip_parallel_reduction(w::ThreeWindingTransformerWinding) =
    PSY.is_phase_shifting(w.winding)

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
