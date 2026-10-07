"""
    PardisoLinSolveCache{T}

An MKL Pardiso factorization of a square sparse matrix with element type `T`
(`Float64` or `ComplexF64`). The `MKLPardisoExt` extension implements
`symbolic_factor!`, `numeric_refactor!`, `full_factor!`, and `solve!` for it.
Load `Pardiso` to use them. MKL Pardiso runs on x86_64 Linux and Windows only.

The cache keeps a reference to the matrix of the last factorization, because
Pardiso reads the matrix again in the solve phase for iterative refinement.
`numeric_refactor!` requires the sparsity pattern of the last
`symbolic_factor!`, and throws an `ArgumentError` for a different pattern.
"""
mutable struct PardisoLinSolveCache{T <: Union{Float64, ComplexF64}} <: LinearSolverCache
    # A `Pardiso.MKLPardisoSolver`. Its type is not available without the extension.
    ps::Any
    # The index type is free: PowerFlows factors Int32-indexed Jacobians.
    A::SparseArrays.SparseMatrixCSC{T}
    colptr::Vector{Int}
    rowval::Vector{Int}
    is_factored::Bool
    scratch::Vector{T}
    scratch_mat::Matrix{T}
end

"""
    PardisoLinSolveCache(A::SparseMatrixCSC{T}) -> PardisoLinSolveCache{T}

Build an MKL Pardiso cache for `A` without a factorization. Call `full_factor!`
before `solve!`. Throws when the `Pardiso` package is not loaded or when MKL is
not available on this platform.
"""
function PardisoLinSolveCache(
    A::SparseArrays.SparseMatrixCSC{T},
) where {T <: Union{Float64, ComplexF64}}
    _has_mkl_pardiso_ext() || error(_mkl_pardiso_install_error())
    return _make_pardiso_cache(A)
end

function _make_pardiso_cache end

is_factored(cache::PardisoLinSolveCache) = cache.is_factored

function Base.deepcopy_internal(::PardisoLinSolveCache, ::IdDict)
    return error(
        "deepcopy of a PardisoLinSolveCache is unsafe: the copy would share one native " *
        "MKL Pardiso factorization with the original. Two caches on one factorization " *
        "corrupt it, and the copy would use freed memory after the original is collected. " *
        "Share the owning matrix by reference, or build a new factorization.",
    )
end
