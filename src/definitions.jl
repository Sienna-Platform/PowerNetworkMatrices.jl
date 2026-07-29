const YBUS_ELTYPE = ComplexF32

# Cached two-port `(Y11, Y12, Y21, Y22)` of a reduction aggregate. Deliberately ComplexF64, not
# YBUS_ELTYPE: `ybus_branch_entries` hands back ComplexF64 for parallel groups but ComplexF32 for
# series chains (`_build_chain_ybus` assembles at Ybus storage precision), so the declared field
# type is the one explicit place that conversion happens. It is also load-bearing — the
# π-recovery representability test resolves at 1e-9, which Float32 cannot. Four numbers per
# aggregate, so this costs nothing against the sparse Ybus that YBUS_ELTYPE exists to shrink.
const CACHED_TWO_PORT = NTuple{4, ComplexF64}
const EMPTY_TWO_PORT =
    (zero(ComplexF64), zero(ComplexF64), zero(ComplexF64), zero(ComplexF64))

"""
An aggregate of branches occupying a single retained arc after a network reduction: a
`BranchesParallel`/`MixedBranchesParallel` group or a `BranchesSeries` chain.

Aggregates subtype `PSY.ACTransmission`, so without this intermediate layer they match blanket
`::PSY.ACTransmission` methods that assume a single physical branch — silently, in the cases
that return a value rather than erroring. Dispatch on this type whenever a method needs the
reduction-aware `(segment, nr)` form instead.
"""
abstract type AbstractReductionAggregate <: PSY.ACTransmission end

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
