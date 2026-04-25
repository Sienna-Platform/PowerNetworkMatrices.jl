# Direct ccall bindings into libklu (provided by SuiteSparse_jll).
#
# This file intentionally only declares what the cache needs:
#   - klu_l_defaults / klu_l_analyze / klu_l_factor / klu_l_refactor / klu_l_solve
#   - klu_zl_factor  / klu_zl_refactor / klu_zl_solve
#   - klu_l_tsolve   / klu_zl_tsolve
#   - klu_l_free_symbolic / klu_l_free_numeric / klu_zl_free_numeric
#
# The Int32 entry points (klu_*) are not bound: PowerNetworkMatrices only
# constructs sparse matrices with `Int` indices, which on supported Julia
# (>= 1.10, 64-bit) is `Int64`, so the SuiteSparse_long (`klu_l_*`) variants
# are sufficient. A startup assertion guards against 32-bit hosts.

import SuiteSparse_jll: libklu

# klu_l_common mirrors `klu_common` for SuiteSparse_long. Layout copied from
# KLU.jl's wrappers.jl — kept in sync there with the upstream `klu.h`.
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

# Symbolic and numeric handles are opaque to us; we only ever pass the pointer
# back to libklu. Empty mutable structs make the ccall signature explicit at
# the call sites without forcing us to mirror the full C struct layouts.
mutable struct KluLSymbolic end
mutable struct KluLNumeric end

const SymbolicPtr = Ptr{KluLSymbolic}
const NumericPtr = Ptr{KluLNumeric}

# --- common ---------------------------------------------------------------

klu_l_defaults!(common::Ref{KluLCommon}) =
    ccall((:klu_l_defaults, libklu), Cint, (Ptr{KluLCommon},), common)

# --- symbolic -------------------------------------------------------------

function klu_l_analyze(n::Int64, ap::Ptr{Int64}, ai::Ptr{Int64},
    common::Ref{KluLCommon})
    return ccall(
        (:klu_l_analyze, libklu),
        SymbolicPtr,
        (Int64, Ptr{Int64}, Ptr{Int64}, Ptr{KluLCommon}),
        n, ap, ai, common,
    )
end

function klu_l_free_symbolic!(symbolic_ref::Ref{SymbolicPtr},
    common::Ref{KluLCommon})
    return ccall(
        (:klu_l_free_symbolic, libklu),
        Cint,
        (Ptr{SymbolicPtr}, Ptr{KluLCommon}),
        symbolic_ref, common,
    )
end

# --- numeric (real) -------------------------------------------------------

function klu_l_factor(ap::Ptr{Int64}, ai::Ptr{Int64}, ax::Ptr{Cdouble},
    symbolic::SymbolicPtr, common::Ref{KluLCommon})
    return ccall(
        (:klu_l_factor, libklu),
        NumericPtr,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, SymbolicPtr, Ptr{KluLCommon}),
        ap, ai, ax, symbolic, common,
    )
end

function klu_l_refactor(ap::Ptr{Int64}, ai::Ptr{Int64}, ax::Ptr{Cdouble},
    symbolic::SymbolicPtr, numeric::NumericPtr, common::Ref{KluLCommon})
    return ccall(
        (:klu_l_refactor, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Int64}, Ptr{Cdouble}, SymbolicPtr, NumericPtr,
            Ptr{KluLCommon}),
        ap, ai, ax, symbolic, numeric, common,
    )
end

function klu_l_solve(symbolic::SymbolicPtr, numeric::NumericPtr,
    ldim::Int64, nrhs::Int64, b::Ptr{Cdouble}, common::Ref{KluLCommon})
    return ccall(
        (:klu_l_solve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{Cdouble}, Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_l_tsolve(symbolic::SymbolicPtr, numeric::NumericPtr,
    ldim::Int64, nrhs::Int64, b::Ptr{Cdouble}, common::Ref{KluLCommon})
    return ccall(
        (:klu_l_tsolve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{Cdouble}, Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_l_free_numeric!(numeric_ref::Ref{NumericPtr},
    common::Ref{KluLCommon})
    return ccall(
        (:klu_l_free_numeric, libklu),
        Cint,
        (Ptr{NumericPtr}, Ptr{KluLCommon}),
        numeric_ref, common,
    )
end

# --- numeric (complex) ----------------------------------------------------

function klu_zl_factor(ap::Ptr{Int64}, ai::Ptr{Int64}, ax::Ptr{ComplexF64},
    symbolic::SymbolicPtr, common::Ref{KluLCommon})
    return ccall(
        (:klu_zl_factor, libklu),
        NumericPtr,
        (Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}, SymbolicPtr, Ptr{KluLCommon}),
        ap, ai, ax, symbolic, common,
    )
end

function klu_zl_refactor(ap::Ptr{Int64}, ai::Ptr{Int64}, ax::Ptr{ComplexF64},
    symbolic::SymbolicPtr, numeric::NumericPtr, common::Ref{KluLCommon})
    return ccall(
        (:klu_zl_refactor, libklu),
        Cint,
        (Ptr{Int64}, Ptr{Int64}, Ptr{ComplexF64}, SymbolicPtr, NumericPtr,
            Ptr{KluLCommon}),
        ap, ai, ax, symbolic, numeric, common,
    )
end

function klu_zl_solve(symbolic::SymbolicPtr, numeric::NumericPtr,
    ldim::Int64, nrhs::Int64, b::Ptr{ComplexF64}, common::Ref{KluLCommon})
    return ccall(
        (:klu_zl_solve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{ComplexF64},
            Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, common,
    )
end

function klu_zl_tsolve(symbolic::SymbolicPtr, numeric::NumericPtr,
    ldim::Int64, nrhs::Int64, b::Ptr{ComplexF64}, conj_solve::Cint,
    common::Ref{KluLCommon})
    return ccall(
        (:klu_zl_tsolve, libklu),
        Cint,
        (SymbolicPtr, NumericPtr, Int64, Int64, Ptr{ComplexF64}, Cint,
            Ptr{KluLCommon}),
        symbolic, numeric, ldim, nrhs, b, conj_solve, common,
    )
end

function klu_zl_free_numeric!(numeric_ref::Ref{NumericPtr},
    common::Ref{KluLCommon})
    return ccall(
        (:klu_zl_free_numeric, libklu),
        Cint,
        (Ptr{NumericPtr}, Ptr{KluLCommon}),
        numeric_ref, common,
    )
end

# --- KLU status decoding --------------------------------------------------

# Status codes from klu.h
const KLU_OK = 0
const KLU_SINGULAR = 1
const KLU_OUT_OF_MEMORY = -2
const KLU_INVALID = -3
const KLU_TOO_LARGE = -4

function klu_status_string(status::Integer)
    status == KLU_OK && return "OK"
    status == KLU_SINGULAR && return "matrix is singular"
    status == KLU_OUT_OF_MEMORY && return "out of memory"
    status == KLU_INVALID && return "invalid argument"
    status == KLU_TOO_LARGE && return "integer overflow"
    return "unknown KLU status $(status)"
end

function klu_throw(common::KluLCommon, op::AbstractString)
    return error("KLU $(op) failed: $(klu_status_string(common.status)) " *
                 "(status=$(common.status))")
end
