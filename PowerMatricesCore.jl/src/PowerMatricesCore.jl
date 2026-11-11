"""
The DC network factorization, the PTDF and Woodbury row kernels, and the factor-file core
arrays. The package has no dependency on PowerSystems.
"""
module PowerMatricesCore

using DocStringExtensions
import SparseArrays
import SparseArrays: rowvals, nzrange
import LinearAlgebra
import LinearAlgebra: ldiv!, mul!, I, dot

"""Supertype of the cached sparse-factorization backends (`KLULinSolveCache`,
`AAFactorCache`, and `PardisoLinSolveCache`).
Defined ahead of the wrapper submodules so their cache structs can subtype it."""
abstract type LinearSolverCache end

include("definitions.jl")
include("KLUWrapper/KLUWrapper.jl")
import .KLUWrapper:
    KLULinSolveCache,
    klu_factorize,
    symbolic_factor!,
    symbolic_refactor!,
    numeric_refactor!,
    full_factor!,
    full_refactor!,
    solve!,
    tsolve!,
    solve_sparse!,
    solve_sparse,
    condest!,
    is_factored

include("AccelerateWrapper/AccelerateWrapper.jl")
import .AccelerateWrapper: AAFactorCache, aa_factorize

include("linalg_settings.jl")
include("pardiso_cache.jl")
include("solver_dispatch.jl")
include("iterative_refinement.jl")

@template DEFAULT = """
                    $(SIGNATURES)
                    $(DOCSTRING)
                    """

include("linear_solver_generic.jl")
include("aba_matrix.jl")
include("common.jl")
include("auto_tolerance.jl")
include("virtual_factor_helpers.jl")
include("row_cache.jl")
include("connectivity.jl")
include("modf_definitions.jl")
include("factor_core.jl")
include("bridges.jl")
include("woodbury_kernel.jl")
include("virtual_workers.jl")

# Forward declaration for the function that `MKLPardisoExt` defines.
function _calculate_PTDF_matrix_MKLPardiso end

end
