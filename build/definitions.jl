const SUPPORTED_LINEAR_SOLVERS = ["KLU", "MKLPardiso", "AppleAccelerate", "Dense"]

const KiB = 1024
const MiB = KiB * KiB
const GiB = MiB * KiB
const MAX_CACHE_SIZE_MiB = 100
const ROW_PERSISTENT_CACHE_WARN = 1 * GiB
const ZERO_IMPEDANCE_LINE_REACTANCE_THRESHOLD = 1e-3
const LODF_ENTRY_TOLERANCE = 1e-6

DEFAULT_LODF_CHUNK_SIZE = 18_000

SKIP_PARALLEL_REDUCTION_TYPES = [
    PSY.PhaseShiftingTransformer,
    ThreeWindingTransformerWinding{PSY.PhaseShiftingTransformer3W},
]
