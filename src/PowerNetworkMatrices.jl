module PowerNetworkMatrices

export ABA_Matrix
export AdjacencyMatrix
export ArcAdmittanceMatrix
export BA_Matrix
export ArcModification
export ShuntModification
export ContingencySpec
export NetworkModification
export DegreeTwoReduction
export IncidenceMatrix
export LODF
export NetworkReduction
export NetworkReductionData
export PTDF
export RadialReduction
export VirtualLODF
export VirtualMODF
export VirtualPTDF
export WardReduction
export Ybus
export apply_ybus_modification
export compute_ybus_delta
export DC_ABA_Matrix_Factorized
export DC_ABA_Matrix_Unfactorized
export DC_PTDF_Matrix
export DC_vPTDF_Matrix
export DC_BA_Matrix
export AC_Ybus_Matrix
export YBUS_ELTYPE
export get_sum_of_max_rating
export get_single_element_contingency_rating
export get_impedance_averaged_rating

export apply_woodbury_correction
export clear_all_caches!
export clear_caches!
export compute_woodbury_factors
export factorize
export get_post_modification_ptdf_row
export get_system_uuid
export is_factorized

export depth_first_search
export find_subnetworks
export from_hdf5
export get_bus_reduction_map
export get_lodf_data
export get_network_reduction_data
export get_partial_lodf_row
export get_ptdf_data
export get_registered_contingencies
export get_reductions
export get_ward_reduction
export iterative_union_find
export to_hdf5
export validate_connectivity

using DocStringExtensions
import InfrastructureSystems as IS
import PowerSystems as PSY
import PowerSystems: ACBusTypes

import DataStructures
import DataStructures: SortedDict
import SparseArrays
import SparseArrays: rowvals, nzrange
import HDF5
import LinearAlgebra
import LinearAlgebra: BLAS.gemm
import LinearAlgebra: ldiv!, mul!, I, dot
import LinearAlgebra: LAPACK.getrf!, LAPACK.getrs!
import Preferences

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
    n_valid,
    is_factored

include("AccelerateWrapper/AccelerateWrapper.jl")
import .AccelerateWrapper: AAFactorCache, aa_factorize, aa_spmm!, aa_spmv!

include("linalg_settings.jl")
include("solver_dispatch.jl")
include("iterative_refinement.jl")

function __init__()
    something(get_linalg_backend_check(), false) && check_linalg_backend()
end

@template DEFAULT = """
                    $(SIGNATURES)
                    $(DOCSTRING)
                    """

# network calculations
include("PowerNetworkMatrix.jl")
include("ThreeWindingTransformerWinding.jl")
include("definitions.jl")
include("EquivalentBranch.jl")
include("BranchesSeries.jl")
include("BranchesParallel.jl")
include("NetworkReduction.jl")
include("radial_reduction.jl")
include("degree_two_reduction.jl")
include("ward_reduction.jl")
# ZIBR's spec marker must precede ReductionContainer (which dispatches
# `validate_reduction_type` on it). Its `get_reduction(::Ybus, ...)` lives in
# `apply_zero_impedance_reduction.jl`, included after `Ybus.jl` below.
include("zero_impedance_branch_reduction.jl")
include("ReductionContainer.jl")
include("NetworkReductionData.jl")
include("ArcAdmittanceMatrix.jl")
include("YbusACBranches.jl")
include("Ybus.jl")
include("apply_zero_impedance_reduction.jl")
include("IncidenceMatrix.jl")
include("reduction_helpers.jl")
include("AdjacencyMatrix.jl")
include("connectivity_checks.jl")
include("subnetworks.jl")
include("common.jl")
include("BA_ABA_matrices.jl")
include("ptdf_calculations.jl")
include("row_cache.jl")
include("virtual_ptdf_calculations.jl")
include("PowerflowMatrixTypes.jl")
include("lodf_calculations.jl")
include("virtual_lodf_calculations.jl")
include("modf_definitions.jl")
include("network_modification.jl")
include("woodbury_kernel.jl")
include("virtual_ptdf_modification.jl")
include("modf_reduction_consistency.jl")
include("virtual_modf_calculations.jl")
include("system_utils.jl")
include("serialization.jl")

# Forward declarations for symbols still defined inside package extensions.
# AppleAccelerate-related functions live in `src/` now (no extension needed)
# and are not redeclared here.
function _calculate_PTDF_matrix_MKLPardiso end
function _calculate_LODF_matrix_MKLPardiso end
function _pardiso_sequential_LODF! end
function _pardiso_single_LODF! end

end
