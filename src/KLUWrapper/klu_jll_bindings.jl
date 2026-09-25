# Bindings into libklu (SuiteSparse_jll), covering both index-type
# entry-point families:
#   - `klu_l_*` / `klu_zl_*` for `SuiteSparse_long` (Int64) indices
#   - `klu_*`   / `klu_z_*`  for `int`              (Int32) indices
#
# Each ccall is wrapped in `@klu_lock` so that all libklu activity in the
# process serializes through `_LIBKLU_LOCK`; see it in `KLUWrapper.jl` for the
# evidence. The `klu_*_free_*` calls are the exception: they run from GC
# finalizers, which cannot wait on a lock (a contended wait throws "task switch
# not allowed from inside gc finalizer", leaking the handle and eventually
# killing the process), and a free touches only the handle it releases.

import LinearAlgebra
import SuiteSparse_jll: libklu

# ---------------------------------------------------------------------------
# Status codes and shared error helper
# ---------------------------------------------------------------------------

const KLU_OK = 0
const KLU_SINGULAR = 1
const KLU_OUT_OF_MEMORY = -2
const KLU_INVALID = -3
const KLU_TOO_LARGE = -4

# ---------------------------------------------------------------------------
# Int64 (SuiteSparse_long) family
# ---------------------------------------------------------------------------

# Layout matches `klu_l_common` in upstream `klu.h`. Must stay in sync.
mutable struct KluLCommon
    tol::Cdouble
    memgrow::Cdouble
    initmem_amd::Cdouble
    initmem::Cdouble
    maxwork::Cdouble
    btf::Cint
    ordering::Cint
    scale::Cint
    user_order::Ptr{Cvoid}
    user_data::Ptr{Cvoid}
    halt_if_singular::Cint
    status::Cint
    nrealloc::Cint
    structural_rank::Int64
    numerical_rank::Int64
    singular_col::Int64
    noffdiag::Int64
    flops::Cdouble
    rcond::Cdouble
    condest::Cdouble
    rgrowth::Cdouble
    work::Cdouble
    memusage::Csize_t
    mempeak::Csize_t
    KluLCommon() = new()
end

# Opaque handles. Empty structs let the ccall signatures stay explicit.
mutable struct KluLSymbolic end
mutable struct KluLNumeric end

const SymbolicPtr = Ptr{KluLSymbolic}
const NumericPtr = Ptr{KluLNumeric}

klu_l_defaults!(common::Ref{KluLCommon}) =
    @klu_lock ccall((:klu_l_defaults, libklu), Cint, (Ptr{KluLCommon},), common)

function klu_l_analyze(
    n::Int64,
    ap::Ptr{Int64},
    ai::Ptr{Int64},
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_analyze, libklu),
        SymbolicPtr,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{KluLCommon}),
        n, ap, ai, common,
    )
end

function klu_l_free_symbolic!(
    symbolic_ref::Ref{SymbolicPtr},
    common::Ref{KluLCommon},
)
    return ccall(
        (:klu_l_free_symbolic, libklu),
        Cint,
        (Ptr{SymbolicPtr}, Ptr{KluLCommon}),
        symbolic_ref, common,
    )
end

function klu_l_factor(
    ap::Ptr{Int64},
    ai::Ptr{Int64},
    ax::Ptr{Cdouble},
    symbolic::SymbolicPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_factor, libklu),
        NumericPtr,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, SymbolicPtr, Ptr{KluLCommon}),
        ap, ai, ax, symbolic, common,
    )
end

function klu_l_refactor(
    ap::Ptr{Int64},
    ai::Ptr{Int64},
    ax::Ptr{Cdouble},
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_refactor, libklu),
        Cint,
        (
            Ptr{Int64},
            Ptr{Int64},
            Ptr{Cdouble},
            SymbolicPtr,
            NumericPtr,
            Ptr{KluLCommon},
        ),
        ap, ai, ax, symbolic, numeric, common,
    )
end

function klu_l_solve(
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    ldim::Int64,
    nrhs::Int64,
    b::Ptr{Cdouble},
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_solve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{Cdouble}, Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_l_tsolve(
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    ldim::Int64,
    nrhs::Int64,
    b::Ptr{Cdouble},
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_tsolve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{Cdouble}, Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_l_free_numeric!(
    numeric_ref::Ref{NumericPtr},
    common::Ref{KluLCommon},
)
    return ccall(
        (:klu_l_free_numeric, libklu),
        Cint,
        (Ptr{NumericPtr}, Ptr{KluLCommon}),
        numeric_ref, common,
    )
end

# Performance knobs / diagnostics — Int64 family.

# klu_l_sort sorts the columns of the L and U factors in place. KLU's
# numeric phase stores L/U columns in arbitrary order; sorting once after
# the factor improves cache locality on every subsequent solve. Cheap
# (O(nnz_factor)) and idempotent. Call once after the first numeric
# factor; refactor preserves the sort.
function klu_l_sort(
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_sort, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Ptr{KluLCommon}),
        symbolic, numeric, common,
    )
end

# klu_l_condest computes a 1-norm condition number estimate, populating
# `common.condest`. Costs roughly two extra solves. Useful for iterative
# refinement (informs tolerance choice) and as a diagnostic for near-singular
# matrices. Caller reads result from `cache.common[].condest`.
function klu_l_condest(
    ap::Ptr{Int64},
    ax::Ptr{Cdouble},
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_condest, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Cdouble}, SymbolicPtr, NumericPtr, Ptr{KluLCommon}),
        ap, ax, symbolic, numeric, common,
    )
end

# klu_l_rcond fills `common.rcond` with the cheap diagonal-ratio
# reciprocal condition estimate (min(|diag(U)|)/max(|diag(U)|)). Faster
# than condest but less reliable.
function klu_l_rcond(
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_l_rcond, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Ptr{KluLCommon}),
        symbolic, numeric, common,
    )
end

# Complex / Int64
function klu_zl_factor(
    ap::Ptr{Int64},
    ai::Ptr{Int64},
    ax::Ptr{ComplexF64},
    symbolic::SymbolicPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_zl_factor, libklu),
        NumericPtr,
        (Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}, SymbolicPtr, Ptr{KluLCommon}),
        ap, ai, ax, symbolic, common,
    )
end

function klu_zl_refactor(
    ap::Ptr{Int64},
    ai::Ptr{Int64},
    ax::Ptr{ComplexF64},
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_zl_refactor, libklu),
        Cint,
        (
            Ptr{Int64},
            Ptr{Int64},
            Ptr{ComplexF64},
            SymbolicPtr,
            NumericPtr,
            Ptr{KluLCommon},
        ),
        ap, ai, ax, symbolic, numeric, common,
    )
end

function klu_zl_solve(
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    ldim::Int64,
    nrhs::Int64,
    b::Ptr{ComplexF64},
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_zl_solve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{ComplexF64}, Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_zl_tsolve(
    symbolic::SymbolicPtr,
    numeric::NumericPtr,
    ldim::Int64,
    nrhs::Int64,
    b::Ptr{ComplexF64},
    conj_solve::Cint,
    common::Ref{KluLCommon},
)
    return @klu_lock ccall(
        (:klu_zl_tsolve, libklu),
        Cint,
        (
            SymbolicPtr,
            NumericPtr,
            Int64,
            Int64,
            Ptr{ComplexF64},
            Cint,
            Ptr{KluLCommon},
        ),
        symbolic, numeric, ldim, nrhs, b, conj_solve, common,
    )
end

function klu_zl_free_numeric!(
    numeric_ref::Ref{NumericPtr},
    common::Ref{KluLCommon},
)
    return ccall(
        (:klu_zl_free_numeric, libklu),
        Cint,
        (Ptr{NumericPtr}, Ptr{KluLCommon}),
        numeric_ref, common,
    )
end

# ---------------------------------------------------------------------------
# Int32 (int) family — mirror of the Int64 set above
# ---------------------------------------------------------------------------

# Layout matches `klu_common` (Int32 path) in upstream `klu.h`. Differs from
# `KluLCommon` only in the four rank/diag fields, which are `int` instead of
# `SuiteSparse_long`.
mutable struct KluCommon
    tol::Cdouble
    memgrow::Cdouble
    initmem_amd::Cdouble
    initmem::Cdouble
    maxwork::Cdouble
    btf::Cint
    ordering::Cint
    scale::Cint
    user_order::Ptr{Cvoid}
    user_data::Ptr{Cvoid}
    halt_if_singular::Cint
    status::Cint
    nrealloc::Cint
    structural_rank::Cint
    numerical_rank::Cint
    singular_col::Cint
    noffdiag::Cint
    flops::Cdouble
    rcond::Cdouble
    condest::Cdouble
    rgrowth::Cdouble
    work::Cdouble
    memusage::Csize_t
    mempeak::Csize_t
    KluCommon() = new()
end

mutable struct KluSymbolic end
mutable struct KluNumeric end

const SymbolicPtr32 = Ptr{KluSymbolic}
const NumericPtr32 = Ptr{KluNumeric}

klu_defaults!(common::Ref{KluCommon}) =
    @klu_lock ccall((:klu_defaults, libklu), Cint, (Ptr{KluCommon},), common)

function klu_analyze(
    n::Cint,
    ap::Ptr{Cint},
    ai::Ptr{Cint},
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_analyze, libklu),
        SymbolicPtr32,
        (Cint, Ptr{Cint}, Ptr{Cint}, Ptr{KluCommon}),
        n, ap, ai, common,
    )
end

function klu_free_symbolic!(
    symbolic_ref::Ref{SymbolicPtr32},
    common::Ref{KluCommon},
)
    return ccall(
        (:klu_free_symbolic, libklu),
        Cint,
        (Ptr{SymbolicPtr32}, Ptr{KluCommon}),
        symbolic_ref, common,
    )
end

function klu_factor(
    ap::Ptr{Cint},
    ai::Ptr{Cint},
    ax::Ptr{Cdouble},
    symbolic::SymbolicPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_factor, libklu),
        NumericPtr32,
        (Ptr{Cint}, Ptr{Cint}, Ptr{Cdouble}, SymbolicPtr32, Ptr{KluCommon}),
        ap, ai, ax, symbolic, common,
    )
end

function klu_refactor(
    ap::Ptr{Cint},
    ai::Ptr{Cint},
    ax::Ptr{Cdouble},
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_refactor, libklu),
        Cint,
        (
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Cdouble},
            SymbolicPtr32,
            NumericPtr32,
            Ptr{KluCommon},
        ),
        ap, ai, ax, symbolic, numeric, common,
    )
end

function klu_solve(
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    ldim::Cint,
    nrhs::Cint,
    b::Ptr{Cdouble},
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_solve, libklu),
        Cint,
        (SymbolicPtr32, NumericPtr32, Cint, Cint, Ptr{Cdouble}, Ptr{KluCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_tsolve(
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    ldim::Cint,
    nrhs::Cint,
    b::Ptr{Cdouble},
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_tsolve, libklu),
        Cint,
        (SymbolicPtr32, NumericPtr32, Cint, Cint, Ptr{Cdouble}, Ptr{KluCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_free_numeric!(
    numeric_ref::Ref{NumericPtr32},
    common::Ref{KluCommon},
)
    return ccall(
        (:klu_free_numeric, libklu),
        Cint,
        (Ptr{NumericPtr32}, Ptr{KluCommon}),
        numeric_ref, common,
    )
end

# Performance knobs / diagnostics — Int32 family.

function klu_sort(
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_sort, libklu),
        Cint,
        (SymbolicPtr32, NumericPtr32, Ptr{KluCommon}),
        symbolic, numeric, common,
    )
end

function klu_condest(
    ap::Ptr{Cint},
    ax::Ptr{Cdouble},
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_condest, libklu),
        Cint,
        (Ptr{Cint}, Ptr{Cdouble}, SymbolicPtr32, NumericPtr32, Ptr{KluCommon}),
        ap, ax, symbolic, numeric, common,
    )
end

function klu_rcond(
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_rcond, libklu),
        Cint,
        (SymbolicPtr32, NumericPtr32, Ptr{KluCommon}),
        symbolic, numeric, common,
    )
end

# Complex / Int32
function klu_z_factor(
    ap::Ptr{Cint},
    ai::Ptr{Cint},
    ax::Ptr{ComplexF64},
    symbolic::SymbolicPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_z_factor, libklu),
        NumericPtr32,
        (Ptr{Cint}, Ptr{Cint}, Ptr{ComplexF64}, SymbolicPtr32, Ptr{KluCommon}),
        ap, ai, ax, symbolic, common,
    )
end

function klu_z_refactor(
    ap::Ptr{Cint},
    ai::Ptr{Cint},
    ax::Ptr{ComplexF64},
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_z_refactor, libklu),
        Cint,
        (
            Ptr{Cint},
            Ptr{Cint},
            Ptr{ComplexF64},
            SymbolicPtr32,
            NumericPtr32,
            Ptr{KluCommon},
        ),
        ap, ai, ax, symbolic, numeric, common,
    )
end

function klu_z_solve(
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    ldim::Cint,
    nrhs::Cint,
    b::Ptr{ComplexF64},
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_z_solve, libklu),
        Cint,
        (
            SymbolicPtr32,
            NumericPtr32,
            Cint,
            Cint,
            Ptr{ComplexF64},
            Ptr{KluCommon},
        ),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_z_tsolve(
    symbolic::SymbolicPtr32,
    numeric::NumericPtr32,
    ldim::Cint,
    nrhs::Cint,
    b::Ptr{ComplexF64},
    conj_solve::Cint,
    common::Ref{KluCommon},
)
    return @klu_lock ccall(
        (:klu_z_tsolve, libklu),
        Cint,
        (
            SymbolicPtr32,
            NumericPtr32,
            Cint,
            Cint,
            Ptr{ComplexF64},
            Cint,
            Ptr{KluCommon},
        ),
        symbolic, numeric, ldim, nrhs, b, conj_solve, common,
    )
end

function klu_z_free_numeric!(
    numeric_ref::Ref{NumericPtr32},
    common::Ref{KluCommon},
)
    return ccall(
        (:klu_z_free_numeric, libklu),
        Cint,
        (Ptr{NumericPtr32}, Ptr{KluCommon}),
        numeric_ref, common,
    )
end

# ---------------------------------------------------------------------------
# Shared error throw — dispatches on the common-struct width
# ---------------------------------------------------------------------------

function klu_throw(common::Union{KluLCommon, KluCommon}, op::AbstractString)
    s = common.status
    s == KLU_SINGULAR &&
        throw(LinearAlgebra.SingularException(Int(common.singular_col + 1)))
    s == KLU_OUT_OF_MEMORY && throw(OutOfMemoryError())
    s == KLU_INVALID && throw(ArgumentError("KLU $(op) failed: invalid argument"))
    s == KLU_TOO_LARGE && throw(OverflowError("KLU $(op) failed: integer overflow"))
    return error("KLU $(op) failed: status=$(s)")
end
