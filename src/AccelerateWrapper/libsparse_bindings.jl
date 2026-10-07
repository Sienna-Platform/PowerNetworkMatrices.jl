# Direct bindings into Apple's libSparse.dylib (part of the Accelerate
# framework). Only the LU entry points used by PowerNetworkMatrices are
# wrapped, for Float64 and ComplexF64. No Float32 variants, no QR or
# Cholesky-AtA variants. Mangled names match what AppleAccelerate.jl uses; see
# `/Library/Developer/CommandLineTools/SDKs/MacOSX*.sdk/System/Library/
# Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/
# Versions/A/Headers/Sparse/Solve.h` for the C declarations.
#
# The complex C structs (`SparseMatrixStructureComplex`,
# `SparseOpaqueFactorization_Complex_Double`, `DenseMatrix_Complex_Double`,
# `DenseVector_Complex_Double`) have the same size and field offsets as the
# real structs. Only the attribute bitfield type is different, and it is also
# 4 bytes. Thus one Julia struct serves both element types, and the element
# type selects the mangled symbol.

const LIBSPARSE =
    "/System/Library/Frameworks/Accelerate.framework/Versions/A/" *
    "Frameworks/vecLib.framework/libSparse.dylib"

# Only `SparseFactorizationLU` is used by PNM (the documented default,
# "currently LU with TPP" — threshold partial pivoting, provably stable).
# The other LU codes are listed for documentation. All LU codes require
# macOS 15.5+. QR = 40 and CholeskyAtA = 41 are intentionally not wrapped
# (see file header). The LDLT/Cholesky family (codes 0–4) is not wrapped
# because all PNM workloads use the general LU path.
@enum SparseFactorization_t::UInt8 begin
    # QR = 40, CholeskyAtA = 41 — intentionally not wrapped (see file header).
    SparseFactorizationLU = 80
    SparseFactorizationLUUnpivoted = 81
    SparseFactorizationLUSPP = 82
    SparseFactorizationLUTPP = 83
end

@enum SparseOrder_t::UInt8 begin
    SparseOrderDefault = 0
    SparseOrderUser = 1
    SparseOrderAMD = 2
    SparseOrderMetis = 3
    SparseOrderCOLAMD = 4
end

@enum SparseScaling_t::UInt8 begin
    SparseScalingDefault = 0
    SparseScalingUser = 1
    SparseScalingEquilibriationInf = 2
end

@enum SparseStatus_t::Int32 begin
    SparseStatusOk = 0
    SparseStatusFailed = -1
    SparseMatrixIsSingular = -2
    SparseInternalError = -3
    SparseParameterError = -4
    SparseStatusReleased = -2147483647
end

@enum SparseControl_t::UInt32 begin
    SparseDefaultControl = 0
end

# `SparseAttributes_t` is a packed bitfield in C; Julia can't express that
# cleanly, so we model it as `Cuint` and assemble the bits ourselves.
const att_type = Cuint
const ATT_ORDINARY = att_type(0)

struct SparseMatrixStructure
    rowCount::Cint
    columnCount::Cint
    columnStarts::Ptr{Clong}
    rowIndices::Ptr{Cint}
    attributes::att_type
    blockSize::UInt8
end

struct SparseNumericFactorOptions
    control::SparseControl_t
    scalingMethod::SparseScaling_t
    scaling::Ptr{Cvoid}
    pivotTolerance::Float64
    zeroTolerance::Float64
end

# Defaults match Apple's SolveImplementation.h. The scaling method is the
# meaningful knob — `SparseScalingEquilibriationInf` on the LU path is a ~4×
# speedup with no correctness impact for well-conditioned ABA-like inputs.
function SparseNumericFactorOptions(scaling::SparseScaling_t)
    return SparseNumericFactorOptions(
        SparseDefaultControl,
        scaling,
        C_NULL,
        0.01,
        eps(Cdouble) * 1e-4,
    )
end

struct SparseSymbolicFactorOptions
    control::SparseControl_t
    orderMethod::SparseOrder_t
    order::Ptr{Cvoid}
    ignoreRowsAndColumns::Ptr{Cvoid}
    malloc::Ptr{Cvoid}
    free::Ptr{Cvoid}
    reportError::Ptr{Cvoid}
end

# No Julia code may sit behind these callbacks. libSparse calls them from its
# dispatch worker threads while the calling Julia thread is blocked inside a
# non-GC-safe ccall; a Julia callback adopts the worker, allocates, triggers
# GC, and `jl_gc_wait_for_the_world` then waits forever on the blocked caller.
# So malloc/free are libc's own symbols, and `reportError` is libc `puts`
# (ABI-compatible with `void (*)(const char *)` on every macOS target): it
# prints the message and returns, and the caller turns the returned status
# into a Julia exception. `C_NULL` is not an option for `reportError` —
# libSparse then halts the process with `__builtin_trap()`.
function SparseSymbolicFactorOptions()
    return SparseSymbolicFactorOptions(
        SparseDefaultControl,
        SparseOrderDefault,
        C_NULL,
        C_NULL,
        cglobal(:malloc),
        cglobal(:free),
        cglobal(:puts),
    )
end

struct DenseVector_t{T <: Union{Float64, ComplexF64}}
    count::Cint
    data::Ptr{T}
end

struct DenseMatrix_t{T <: Union{Float64, ComplexF64}}
    rowCount::Cint
    columnCount::Cint
    columnStride::Cint
    attributes::att_type
    data::Ptr{T}
end

struct SparseMatrix_t{T <: Union{Float64, ComplexF64}}
    structure::SparseMatrixStructure
    data::Ptr{T}
end

struct SparseOpaqueSymbolicFactorization
    status::SparseStatus_t
    rowCount::Cint
    columnCount::Cint
    attributes::att_type
    blockSize::UInt8
    type::SparseFactorization_t
    factorization::Ptr{Cvoid}
    workspaceSize_Float::Csize_t
    workspaceSize_Double::Csize_t
    factorSize_Float::Csize_t
    factorSize_Double::Csize_t
end

# Placeholder used to initialize `AAFactorCache.symbolic` before the first
# factor. Marked released so any accidental cleanup is a no-op.
function _null_symbolic()
    return SparseOpaqueSymbolicFactorization(
        SparseStatusReleased,
        0,
        0,
        ATT_ORDINARY,
        0,
        SparseFactorizationLU,
        C_NULL,
        0,
        0,
        0,
        0,
    )
end

struct SparseOpaqueFactorization_t
    status::SparseStatus_t
    attributes::att_type
    symbolicFactorization::SparseOpaqueSymbolicFactorization
    userFactorStorage::Bool
    numericFactorization::Ptr{Cvoid}
    solveWorkspaceRequiredStatic::Csize_t
    solveWorkspaceRequiredPerRHS::Csize_t
end

function _null_factorization()
    return SparseOpaqueFactorization_t(
        SparseStatusReleased,
        ATT_ORDINARY,
        _null_symbolic(),
        false,
        C_NULL,
        0,
        0,
    )
end

# Build the Apple-side dense views at the ccall boundary. `StridedMatrix`'s
# first-dimension stride must be 1 (the libSparse contract); we assert at the
# call site, not here.
function _dense_matrix(B::StridedMatrix{T}) where {T <: Union{Float64, ComplexF64}}
    return DenseMatrix_t{T}(
        Cint(size(B, 1)),
        Cint(size(B, 2)),
        Cint(stride(B, 2)),
        ATT_ORDINARY,
        pointer(B),
    )
end

function _dense_vector(b::StridedVector{T}) where {T <: Union{Float64, ComplexF64}}
    return DenseVector_t{T}(Cint(length(b)), pointer(b))
end

# --- ccalls -----------------------------------------------------------------
#
# Mangled symbol names come from the C++ ABI of libSparse. They are stable on
# the system framework and match what AppleAccelerate.jl binds. If Apple
# breaks them in a future macOS release, the failure will be loud (dlopen of
# a missing symbol at first call), which is the behavior we want.

# Symbolic-only factor: analyzes the pattern, returns an opaque symbolic
# factor. Can back many numeric factors on the same pattern.
function _sparse_symbolic_factor(
    ::Type{Float64},
    ftype::SparseFactorization_t,
    structure::SparseMatrixStructure,
    sym_opts::SparseSymbolicFactorOptions,
)::SparseOpaqueSymbolicFactorization
    return @ccall LIBSPARSE._Z12SparseFactorh21SparseMatrixStructure27SparseSymbolicFactorOptions(
        ftype::Cuint,
        structure::SparseMatrixStructure,
        sym_opts::SparseSymbolicFactorOptions,
    )::SparseOpaqueSymbolicFactorization
end

function _sparse_symbolic_factor(
    ::Type{ComplexF64},
    ftype::SparseFactorization_t,
    structure::SparseMatrixStructure,
    sym_opts::SparseSymbolicFactorOptions,
)::SparseOpaqueSymbolicFactorization
    return @ccall LIBSPARSE._Z12SparseFactorh28SparseMatrixStructureComplex27SparseSymbolicFactorOptions(
        ftype::Cuint,
        structure::SparseMatrixStructure,
        sym_opts::SparseSymbolicFactorOptions,
    )::SparseOpaqueSymbolicFactorization
end

# Numeric factor on top of an existing symbolic factor. Reusable: the
# symbolic handle is not consumed.
function _sparse_numeric_factor(
    symbolic::SparseOpaqueSymbolicFactorization,
    matrix::SparseMatrix_t{Float64},
    num_opts::SparseNumericFactorOptions,
)::SparseOpaqueFactorization_t
    return @ccall LIBSPARSE._Z12SparseFactor33SparseOpaqueSymbolicFactorization19SparseMatrix_Double26SparseNumericFactorOptions(
        symbolic::SparseOpaqueSymbolicFactorization,
        matrix::SparseMatrix_t{Float64},
        num_opts::SparseNumericFactorOptions,
    )::SparseOpaqueFactorization_t
end

function _sparse_numeric_factor(
    symbolic::SparseOpaqueSymbolicFactorization,
    matrix::SparseMatrix_t{ComplexF64},
    num_opts::SparseNumericFactorOptions,
)::SparseOpaqueFactorization_t
    return @ccall LIBSPARSE._Z12SparseFactor33SparseOpaqueSymbolicFactorization27SparseMatrix_Complex_Double26SparseNumericFactorOptions(
        symbolic::SparseOpaqueSymbolicFactorization,
        matrix::SparseMatrix_t{ComplexF64},
        num_opts::SparseNumericFactorOptions,
    )::SparseOpaqueFactorization_t
end

# Workspace-aware solve overloads (libSparse, macOS 10.13+). The factor
# exposes `solveWorkspaceRequiredStatic + nrhs * solveWorkspaceRequiredPerRHS`
# bytes of scratch it needs per call; supplying a reusable buffer eliminates
# the implicit malloc/free that the no-workspace variants perform internally.
# On AA_LU at 10k nodes that buffer is ~234 KiB / RHS — substantial per-call
# churn when the no-workspace path is used in a tight row loop. The workspace
# pointer must be 16-byte aligned; Julia's `Vector{Float64}` data satisfies
# this for any non-trivial size.
function _sparse_solve_matrix_ws!(
    factor::SparseOpaqueFactorization_t,
    B::DenseMatrix_t{Float64},
    workspace::Ptr{Cvoid},
)
    @ccall LIBSPARSE._Z11SparseSolve32SparseOpaqueFactorization_Double18DenseMatrix_DoublePv(
        factor::SparseOpaqueFactorization_t,
        B::DenseMatrix_t{Float64},
        workspace::Ptr{Cvoid},
    )::Cvoid
    return nothing
end

function _sparse_solve_matrix_ws!(
    factor::SparseOpaqueFactorization_t,
    B::DenseMatrix_t{ComplexF64},
    workspace::Ptr{Cvoid},
)
    @ccall LIBSPARSE._Z11SparseSolve40SparseOpaqueFactorization_Complex_Double26DenseMatrix_Complex_DoublePv(
        factor::SparseOpaqueFactorization_t,
        B::DenseMatrix_t{ComplexF64},
        workspace::Ptr{Cvoid},
    )::Cvoid
    return nothing
end

function _sparse_solve_vector_ws!(
    factor::SparseOpaqueFactorization_t,
    b::DenseVector_t{Float64},
    workspace::Ptr{Cvoid},
)
    @ccall LIBSPARSE._Z11SparseSolve32SparseOpaqueFactorization_Double18DenseVector_DoublePv(
        factor::SparseOpaqueFactorization_t,
        b::DenseVector_t{Float64},
        workspace::Ptr{Cvoid},
    )::Cvoid
    return nothing
end

function _sparse_solve_vector_ws!(
    factor::SparseOpaqueFactorization_t,
    b::DenseVector_t{ComplexF64},
    workspace::Ptr{Cvoid},
)
    @ccall LIBSPARSE._Z11SparseSolve40SparseOpaqueFactorization_Complex_Double26DenseVector_Complex_DoublePv(
        factor::SparseOpaqueFactorization_t,
        b::DenseVector_t{ComplexF64},
        workspace::Ptr{Cvoid},
    )::Cvoid
    return nothing
end

# Bytes of workspace required to solve with `nrhs` right-hand sides.
@inline function _solve_workspace_bytes(
    factor::SparseOpaqueFactorization_t,
    nrhs::Integer,
)
    return Int(factor.solveWorkspaceRequiredStatic) +
           Int(nrhs) * Int(factor.solveWorkspaceRequiredPerRHS)
end

# Frees the libSparse-side numeric / symbolic storage attached to an opaque
# factor. Idempotent: a second call with a `SparseStatusReleased` handle is
# a no-op on libSparse's side.
function _sparse_cleanup_factor!(::Type{Float64}, factor::SparseOpaqueFactorization_t)
    @ccall LIBSPARSE._Z13SparseCleanup32SparseOpaqueFactorization_Double(
        factor::SparseOpaqueFactorization_t,
    )::Cvoid
    return nothing
end

function _sparse_cleanup_factor!(::Type{ComplexF64}, factor::SparseOpaqueFactorization_t)
    @ccall LIBSPARSE._Z13SparseCleanup40SparseOpaqueFactorization_Complex_Double(
        factor::SparseOpaqueFactorization_t,
    )::Cvoid
    return nothing
end

function _sparse_cleanup_symbolic!(symbolic::SparseOpaqueSymbolicFactorization)
    @ccall LIBSPARSE._Z13SparseCleanup33SparseOpaqueSymbolicFactorization(
        symbolic::SparseOpaqueSymbolicFactorization,
    )::Cvoid
    return nothing
end

# Translate libSparse status codes into Julia exceptions. Singular and
# parameter-error are the most common; the rest fall through to a generic
# `error`.
function _libsparse_throw(status::SparseStatus_t, op::AbstractString)
    status == SparseMatrixIsSingular &&
        throw(LinearAlgebra.SingularException(0))
    status == SparseParameterError &&
        throw(ArgumentError("libSparse $(op) failed: parameter error"))
    status == SparseInternalError &&
        error("libSparse $(op) failed: internal error")
    status == SparseStatusFailed &&
        error("libSparse $(op) failed")
    return error("libSparse $(op) failed: status=$(Int(status))")
end
