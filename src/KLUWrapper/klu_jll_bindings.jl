# Bindings into libklu (SuiteSparse_jll), restricted to the SuiteSparse_long
# (`klu_l_*` and `klu_zl_*`) entry points used by KLULinSolveCache.

import LinearAlgebra
import SuiteSparse_jll: libklu

"""
    @klu_gc_preserve vars... body

Windows-only: expands to `GC.@preserve vars... body`. On every other
platform it expands to `body` unchanged (zero runtime cost).

The `klu_l_*` / `klu_zl_*` ccalls receive raw `Ptr{Cdouble}` / `Ptr{Int64}`
arguments derived from `pointer(some_julia_array)`. Julia's ccall lowering
roots arguments declared as `Ref{T}` or `Array{T}`, but it does not root
the underlying array when the ccall signature is `Ptr{T}` and only the
already-computed pointer is passed — the array is then reclaimable by GC
during the C call.

Symptom that motivated the macro: under the parallel `KLULinSolvePool`
path (`Threads.@spawn`-ed PTDF row solves driven by PowerSimulations'
`add_expressions!`), Windows surfaces `KLU_INVALID` from `klu_l_solve`,
which from KLU's source means a NULL `Symbolic` / `Numeric` / `B`. The
per-worker pool design rules out NULL `Symbolic` / `Numeric` (each worker
owns its own factorization), leaving the RHS `B` (or the cache whose
finalizer frees the libklu numeric handle) being reclaimed mid-ccall as
the remaining explanation. Mac (AppleAccelerate) and the Linux test paths
have not exhibited this — GC scheduling and safepoint placement during
the ccall differ across OS/threading-model combinations — so the
preservation is gated to Windows to avoid noise on platforms where it
isn't load-bearing.

Each ccall site lists the Julia-managed values whose lifetime must span
the C call (e.g., the RHS array and the cache).
"""
macro klu_gc_preserve(args...)
    length(args) >= 1 ||
        throw(ArgumentError("@klu_gc_preserve requires at least one expression"))
    vars = args[1:(end - 1)]
    body = args[end]
    @static if Sys.iswindows()
        if isempty(vars)
            return esc(body)
        else
            return esc(:($(GC).@preserve $(vars...) $body))
        end
    else
        return esc(body)
    end
end

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
    ccall((:klu_l_defaults, libklu), Cint, (Ptr{KluLCommon},), common)

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

# Status codes from klu.h.
const KLU_OK = 0
const KLU_SINGULAR = 1
const KLU_OUT_OF_MEMORY = -2
const KLU_INVALID = -3
const KLU_TOO_LARGE = -4

function klu_throw(common::KluLCommon, op::AbstractString)
    s = common.status
    s == KLU_SINGULAR &&
        throw(LinearAlgebra.SingularException(Int(common.singular_col + 1)))
    s == KLU_OUT_OF_MEMORY && throw(OutOfMemoryError())
    s == KLU_INVALID && throw(ArgumentError("KLU $(op) failed: invalid argument"))
    s == KLU_TOO_LARGE && throw(OverflowError("KLU $(op) failed: integer overflow"))
    return error("KLU $(op) failed: status=$(s)")
end
