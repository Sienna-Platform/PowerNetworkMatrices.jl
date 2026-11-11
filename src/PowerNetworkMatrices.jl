module PowerNetworkMatrices

export ABA_Matrix
export BranchArcTable
export get_branch_kind
export get_branch_id
export get_branch_parent
export get_branch_sign
export get_branch_susceptance
export get_branch_alpha
export get_branch_arc_count
export branch_flows
export LinearSolverCache
export AutoTolerance
export discover_data_precision
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
export ZeroImpedanceBranchReduction
export apply_ybus_modification
export apply_ybus_modification!
export restore_ybus_modification!
export compute_ybus_delta
export get_applied_reductions
export get_irreducible_bus_numbers
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
export equivalent_branch
export arc_equivalent_branch
export branch_admittance
export get_effective_series_susceptance
export reduced_arc_admittance
export three_winding_arcs
export branch_flow_limits

export apply_woodbury_correction
export clear_all_caches!
export clear_caches!
export compute_woodbury_factors
export factorize
export get_post_modification_ptdf_row
export BranchOutageTable
export BranchReasonTable
export get_outage_rows
export get_outage_ids
export get_outage_opens_arc
export get_outage_member
export from_hdf5
export get_reduction_codes
export no_arc_reason
export no_arc_surviving_bus
export NO_ARC_HAS_ARC
export NO_ARC_RADIAL
export NO_ARC_ZERO_IMPEDANCE
export NO_ARC_WARD
export NO_ARC_NOT_IN_NETWORK
export get_system_uuid
export is_factorized
export populate_cache
export to_hdf5

export find_subnetworks
export get_bus_reduction_map
export get_lodf_data
export get_network_reduction_data
export get_partial_lodf_row
export get_ptdf_data
export get_ptdf_row
export get_registered_contingencies
export get_reductions
export iterative_union_find
export validate_connectivity

using DocStringExtensions
import InfrastructureSystems as IS
import PowerSystems as PSY
using PowerSystems: @u_str, PerUnit
import PowerSystems: ACBusTypes

import PowerMatricesCore as PMC
import DataStructures
import DataStructures: SortedDict
import HDF5
import SparseArrays
import SparseArrays: rowvals, nzrange
import LinearAlgebra
import LinearAlgebra: ldiv!, mul!, I, dot

import PowerMatricesCore:
    LinearSolverCache,
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
    is_factored,
    AAFactorCache,
    aa_factorize,
    YBUS_ELTYPE,
    KiB,
    MiB,
    GiB,
    MAX_CACHE_SIZE_MiB,
    ROW_PERSISTENT_CACHE_WARN,
    MODF_ISLANDING_TOLERANCE,
    AUTO_TOLERANCE_BUS_LIMIT,
    LinearSolverType,
    KLUSolver,
    DenseSolver,
    MKLPardisoSolver,
    AppleAccelerateLUSolver,
    SUPPORTED_LINEAR_SOLVERS,
    resolve_linear_solver,
    _has_mkl_pardiso_ext,
    _mkl_pardiso_install_error,
    _AA_MIN_MACOS,
    _macos_product_version,
    _macos_at_least,
    _has_apple_accelerate_backend,
    _apple_accelerate_unavailable_error,
    _default_linear_solver,
    PardisoLinSolveCache,
    _make_pardiso_cache,
    with_solver,
    make_linear_solver_cache,
    SINGULAR_PROBE_RTOL,
    factorization_ok,
    DEFAULT_REFINEMENT_MAX_ITER,
    solve_w_refinement,
    solve_w_refinement!,
    calculate_ABA_matrix,
    _calculate_PTDF_matrix_MKLPardiso,
    AutoTolerance,
    discover_data_precision,
    DEFAULT_AUTO_TOLERANCE,
    MIN_RELATIVE_TOLERANCE,
    MAX_RELATIVE_TOLERANCE,
    AbsoluteCutoff,
    RelativeCutoff,
    SparsificationCutoff,
    apply_cutoff,
    cutoff_value,
    _validate_data_precision,
    _sigfigs,
    _quantile_sorted,
    _resolve_delta,
    _relative_alpha,
    _hager_invnorm1,
    _condition_estimate,
    _log_condition_estimate,
    _dense_tol,
    _resolve_virtual_cutoff,
    RowCacheValue,
    RowCache,
    set_persistent_row!,
    pin_row!,
    purge_one!,
    check_cache_size!,
    _pin!,
    _cached_row,
    ArcModification,
    ShuntModification,
    NetworkModification,
    ContingencySpec,
    WoodburyFactors,
    _merge_arc_modifications,
    _merge_shunt_modifications,
    get_representative,
    union_sets!,
    _build_bus_to_valid_idx,
    sparsify,
    _create_factorization,
    _solve_factorization,
    _solve_ba_column!,
    _ba_column_susceptance,
    _extract_arc_susceptances,
    _get_PTDF_A_diag

const KLUWrapper = PMC.KLUWrapper
const AccelerateWrapper = PMC.AccelerateWrapper

@template DEFAULT = """
                    $(SIGNATURES)
                    $(DOCSTRING)
                    """

# network calculations
include("PowerNetworkMatrix.jl")
include("ThreeWindingTransformerCircuit.jl")
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
include("BranchCatalog.jl")
include("ArcAdmittanceMatrix.jl")
include("YbusACBranches.jl")
include("Ybus.jl")
include("apply_zero_impedance_reduction.jl")
include("IncidenceMatrix.jl")
include("reduction_helpers.jl")
include("irreducible_bus_selection.jl")
include("AdjacencyMatrix.jl")
include("connectivity_checks.jl")
include("bridges.jl")
include("subnetworks.jl")
include("common.jl")
include("BranchAdmittance.jl")
include("BA_ABA_matrices.jl")
include("virtual_factor_helpers.jl")
include("virtual_factor_core.jl")
include("ptdf_calculations.jl")
include("virtual_ptdf_calculations.jl")
include("PowerflowMatrixTypes.jl")
include("lodf_calculations.jl")
include("virtual_lodf_calculations.jl")
include("network_modification.jl")
include("virtual_ptdf_modification.jl")
include("modf_reduction_consistency.jl")
include("virtual_modf_calculations.jl")
include("woodbury_kernel.jl")
include("populate_cache.jl")
include("virtual_workers.jl")
include("system_utils.jl")
include("serialization.jl")
include("factor_file_io.jl")

"""
    to_hdf5(mat, filename; kwargs...)

Write the factor data of a `VirtualPTDF` or `VirtualMODF` to an HDF5 file.
"""
function to_hdf5 end

"""
    from_hdf5(T, filename; kwargs...)

Read a matrix or a table from a factor file.
"""
function from_hdf5 end

end
