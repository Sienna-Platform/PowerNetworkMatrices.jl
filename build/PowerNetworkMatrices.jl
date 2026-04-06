module PowerNetworkMatrices

export ABA_Matrix
export BA_Matrix
export factorize
export find_subnetworks
export from_hdf5
export get_ptdf_data
export get_lodf_data
export get_bus_reduction_map
export get_ward_reduction
export get_reductions
export get_network_reduction_data
export IncidenceMatrix
export AdjacencyMatrix
export is_factorized
export LODF
export PTDF
export NetworkReduction
export NetworkReductionData
export RadialReduction
export DegreeTwoReduction
export WardReduction

export depth_first_search
export iterative_union_find
export to_hdf5
export validate_connectivity
export VirtualLODF
export VirtualPTDF
export Ybus
export ArcAdmittanceMatrix
export DC_ABA_Matrix_Factorized
export DC_ABA_Matrix_Unfactorized
export DC_PTDF_Matrix
export DC_vPTDF_Matrix
export DC_BA_Matrix
export AC_Ybus_Matrix

using DocStringExtensions
import InfrastructureSystems as IS
import PowerSystems as PSY
import PowerSystems: ACBusTypes

# TODO make public so users can check for solver availability?
# Check if MKL/Pardiso extension is available at runtime
# Extensions are loaded when the trigger packages (MKL, Pardiso) are loaded
function _has_mkl_pardiso_ext()
    ext = Base.get_extension(@__MODULE__, :MKLPardisoExt)
    return !isnothing(ext)
end

# Check if AppleAccelerate extension is available at runtime  
function _has_apple_accelerate_ext()
    ext = Base.get_extension(@__MODULE__, :AppleAccelerateExt)
    return !isnothing(ext)
end

# _create_apple_accelerate_factorization is defined in ext/AppleAccelerateExt.jl
# when AppleAccelerate package is loaded

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

function __init__()
    # Suggest optimal BLAS backend.
    blas_config = lowercase(string(LinearAlgebra.BLAS.get_config()))
    if Sys.iswindows() && !contains(blas_config, "mkl")
        @info """For faster dense matrix operations, consider using MKL:
                   pkg> add MKL
                   using MKL  # before any matrix operations
                 Sparse factorization still uses KLU (recommended)."""
    elseif Sys.isapple() && !contains(blas_config, "accelerate")
        @info """For faster dense matrix operations, consider using AppleAccelerate:
                   pkg> add AppleAccelerate
                   using AppleAccelerate  # before any matrix operations
                 Sparse factorization still uses KLU (recommended)."""
    end
end

end
