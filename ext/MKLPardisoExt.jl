module MKLPardisoExt

import PowerNetworkMatrices as PNM
using Pardiso
import SparseArrays
import SparseArrays: SparseMatrixCSC, getcolptr, rowvals
import LinearAlgebra

"""
Function for internal use only.

Computes the PTDF matrix by means of the MKL Pardiso for dense matrices.

# Arguments
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence Matrix
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix
- `ref_bus_positions::Set{Int}`:
        vector containing the indexes of the reference slack buses.
- `dist_slack::Vector{Float64}`:
        vector containing the weights for the distributed slacks.
"""
function PNM._calculate_PTDF_matrix_MKLPardiso(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64})
    linecount = size(BA, 2)
    buscount = size(BA, 1)

    ABA = PNM.calculate_ABA_matrix(A, BA, ref_bus_positions)
    @assert LinearAlgebra.issymmetric(ABA)
    ps = Pardiso.MKLPardisoSolver()
    Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM)
    Pardiso.pardisoinit(ps)
    # Pardiso.set_msglvl!(ps, Pardiso.MESSAGE_LEVEL_ON)
    defaults = Pardiso.get_iparms(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    for (ix, v) in enumerate(defaults[2:end])
        Pardiso.set_iparm!(ps, ix + 1, v)
    end
    Pardiso.set_iparm!(ps, 2, 2)
    Pardiso.set_iparm!(ps, 59, 2)
    Pardiso.set_iparm!(ps, 6, 1)
    Pardiso.set_iparm!(ps, 12, 1)
    Pardiso.set_iparm!(ps, 11, 0)
    Pardiso.set_iparm!(ps, 13, 0)
    Pardiso.set_iparm!(ps, 32, 1)

    # initialize matrices for evaluation
    valid_ix = setdiff(1:buscount, ref_bus_positions)
    PTDFm_t = zeros(buscount, linecount)

    full_BA = Matrix(BA[valid_ix, :])
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    elseif isempty(dist_slack) && length(ref_bus_positions) != buscount
        Pardiso.pardiso(ps, PTDFm_t[valid_ix, :], ABA, full_BA)
        PTDFm_t[valid_ix, :] = full_BA
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps)
        return PTDFm_t
    elseif length(dist_slack) == buscount
        @info "Distributed bus"
        Pardiso.pardiso(ps, PTDFm_t[valid_ix, :], ABA, full_BA)
        PTDFm_t[valid_ix, :] = full_BA
        Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
        Pardiso.pardiso(ps)
        slack_array = dist_slack / sum(dist_slack)
        slack_array = reshape(slack_array, 1, buscount)
        return PTDFm_t - ones(buscount, 1) * (slack_array * PTDFm_t)
    else
        error("Distributed bus specification doesn't match the number of buses.")
    end
    return
end

_pardiso_matrix_type(::Type{Float64}) = Pardiso.REAL_NONSYM
_pardiso_matrix_type(::Type{ComplexF64}) = Pardiso.COMPLEX_NONSYM

# The order is necessary: matrix type, then init (it sets the defaults for that type),
# then the iparm changes, then the transpose flag.
function _init_pardiso!(ps, ::Type{T}) where {T}
    Pardiso.set_matrixtype!(ps, _pardiso_matrix_type(T))
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 8, 2)
    # Pardiso reads CSR and Julia stores CSC. For MKL, `fix_iparm!(ps, :N)` sets
    # iparm[12] = 2, a plain transpose (not conjugate), so Pardiso solves A·x = b for
    # real and complex A.
    Pardiso.fix_iparm!(ps, :N)
    return ps
end

function PNM._make_pardiso_cache(
    A::SparseMatrixCSC{T},
) where {T <: Union{Float64, ComplexF64}}
    if !Pardiso.mkl_is_available()
        error(
            "MKLPardiso backend selected but MKL is not available on this platform. " *
            "MKL Pardiso requires x86_64 Linux or Windows; it is unavailable on Apple " *
            "Silicon.",
        )
    end
    ps = Pardiso.MKLPardisoSolver()
    _init_pardiso!(ps, T)
    cache = PNM.PardisoLinSolveCache{T}(
        ps, A, Int[], Int[], false, T[], Matrix{T}(undef, 0, 0), false,
    )
    finalizer(_finalize_pardiso_cache, cache)
    return cache
end

# Frees the native MKL handle once. Not safe to call from a finalizer; see
# `_finalize_pardiso_cache`.
function _release!(c::PNM.PardisoLinSolveCache)
    c.released && return c
    # Set first: a failed RELEASE_ALL then leaks once, but never frees twice.
    c.released = true
    Pardiso.set_phase!(c.ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(c.ps)
    return c
end

# A finalizer runs inside the GC and must not block. RELEASE_ALL can deadlock there,
# so the release runs in a task. `errormonitor` logs a failure of that task.
function _finalize_pardiso_cache(c::PNM.PardisoLinSolveCache)
    c.released && return
    errormonitor(@async _release!(c))
    return
end

function _same_pattern(cache::PNM.PardisoLinSolveCache, A::SparseMatrixCSC)
    return size(A, 1) == length(cache.colptr) - 1 &&
           getcolptr(A) == cache.colptr &&
           rowvals(A) == cache.rowval
end

function PNM.symbolic_factor!(
    cache::PNM.PardisoLinSolveCache{T},
    A::SparseMatrixCSC{T},
) where {T}
    cache.A = A
    cache.colptr = Vector{Int}(getcolptr(A))
    cache.rowval = Vector{Int}(rowvals(A))
    Pardiso.set_phase!(cache.ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(cache.ps, cache.A, T[])
    cache.is_factored = false
    return cache
end

function PNM.numeric_refactor!(
    cache::PNM.PardisoLinSolveCache{T},
    A::SparseMatrixCSC{T},
) where {T}
    if !_same_pattern(cache, A)
        throw(
            ArgumentError(
                "Cannot numeric_refactor!: the matrix has a different sparsity pattern " *
                "than the last symbolic_factor!. Call full_factor! instead.",
            ),
        )
    end
    cache.A = A
    Pardiso.set_phase!(cache.ps, Pardiso.NUM_FACT)
    Pardiso.pardiso(cache.ps, cache.A, T[])
    cache.is_factored = true
    return cache
end

function PNM.full_factor!(
    cache::PNM.PardisoLinSolveCache{T},
    A::SparseMatrixCSC{T},
) where {T}
    PNM.symbolic_factor!(cache, A)
    return PNM.numeric_refactor!(cache, A)
end

# Pardiso solves out of place. The persistent scratch buffers keep the solves free of
# allocations on the Julia side.
function PNM.solve!(cache::PNM.PardisoLinSolveCache{T}, b::StridedVector{T}) where {T}
    cache.is_factored || error("PardisoLinSolveCache: call full_factor! before solve!.")
    Pardiso.set_phase!(cache.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    if length(cache.scratch) != length(b)
        resize!(cache.scratch, length(b))
    end
    Pardiso.pardiso(cache.ps, cache.scratch, cache.A, b)
    copyto!(b, cache.scratch)
    return b
end

function PNM.solve!(cache::PNM.PardisoLinSolveCache{T}, B::StridedMatrix{T}) where {T}
    cache.is_factored || error("PardisoLinSolveCache: call full_factor! before solve!.")
    Pardiso.set_phase!(cache.ps, Pardiso.SOLVE_ITERATIVE_REFINE)
    if size(cache.scratch_mat) != size(B)
        cache.scratch_mat = Matrix{T}(undef, size(B))
    end
    Pardiso.pardiso(cache.ps, cache.scratch_mat, cache.A, B)
    copyto!(B, cache.scratch_mat)
    return B
end

end # module
