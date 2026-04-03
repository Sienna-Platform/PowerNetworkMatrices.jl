module PowerNetworkMatrices

export ABA_Matrix
export AdjacencyMatrix
export ArcAdmittanceMatrix
export BA_Matrix
export ArcModification
export ContingencySpec
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
export DC_ABA_Matrix_Factorized
export DC_ABA_Matrix_Unfactorized
export DC_PTDF_Matrix
export DC_vPTDF_Matrix
export DC_BA_Matrix
export AC_Ybus_Matrix
export YBUS_ELTYPE

export clear_all_caches!
export clear_caches!
export factorize
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
import KLU: klu
import KLU
import LinearAlgebra
import LinearAlgebra: BLAS.gemm
import LinearAlgebra: ldiv!, mul!, I, dot
import LinearAlgebra: LAPACK.getrf!, LAPACK.getrs!
import Preferences

include("linalg_settings.jl")

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
include("ReductionContainer.jl")
include("NetworkReductionData.jl")
include("ArcAdmittanceMatrix.jl")
include("YbusACBranches.jl")
include("Ybus.jl")
include("IncidenceMatrix.jl")
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
include("virtual_modf_calculations.jl")
include("system_utils.jl")
include("serialization.jl")

# Declare functions that will be defined by extensions
# These need to be declared so extensions can extend them
function _calculate_PTDF_matrix_MKLPardiso end
function _calculate_PTDF_matrix_AppleAccelerate end
function _calculate_LODF_matrix_MKLPardiso end
function _calculate_LODF_matrix_AppleAccelerate end
function _pardiso_sequential_LODF! end
function _pardiso_single_LODF! end
function _create_apple_accelerate_factorization end

end
