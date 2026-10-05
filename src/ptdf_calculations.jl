"""
Structure containing the Power Transfer Distribution Factor (PTDF) matrix and related power system data.

The PTDF matrix contains sensitivity coefficients that quantify how power injections at buses
affect the power flows on transmission lines. Each element PTDF[i,j] represents the incremental
change in flow on line i due to a unit power injection at bus j, under DC power flow assumptions.

# Fields
- `data::M <: AbstractArray{Float64, 2}`:
        The PTDF matrix data stored in transposed form for computational efficiency.
        Element (i,j) represents the sensitivity of line j flow to bus i injection
- `axes::Ax`:
        Tuple containing (bus_numbers, branch_identifiers) for matrix dimensions
- `lookup::L <: NTuple{2, Dict}`:
        Tuple of dictionaries providing fast lookup from bus/branch identifiers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`:
        Mapping from reference bus numbers to their corresponding subnetwork axes
- `tol::Float64`:
        Tolerance threshold used for matrix sparsification (elements below this value are dropped)
- `branch_catalog::BranchCatalog`:
        Container for network reduction information applied during matrix construction

# Notes
- Stored transposed (bus × arc); `ptdf[bus, arc]` is the sensitivity of the arc flow to a bus injection.
- Elements below `tol` are dropped when sparsified.
- Valid under DC power flow assumptions.
"""
struct PTDF{Ax, L <: NTuple{2, Dict}, M <: AbstractArray{Float64, 2}} <:
       PowerNetworkMatrix{Float64}
    data::M
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    tol::Float64
    branch_catalog::BranchCatalog
end

get_bus_axis(M::PTDF) = M.axes[1]
get_bus_lookup(M::PTDF) = M.lookup[1]
get_arc_axis(M::PTDF) = M.axes[2]
get_arc_lookup(M::PTDF) = M.lookup[2]

stores_transpose(::PTDF) = true

function _buildptdf_from_matrices(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{T, Int} where {T <: Union{Float32, Float64}},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64},
    ::KLUSolver)
    return _calculate_PTDF_matrix_KLU(A, BA, ref_bus_positions, dist_slack)
end

function _buildptdf_from_matrices(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{T, Int} where {T <: Union{Float32, Float64}},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64},
    ::DenseSolver)
    return _calculate_PTDF_matrix_DENSE(A, BA, ref_bus_positions, dist_slack)
end

function _buildptdf_from_matrices(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{T, Int} where {T <: Union{Float32, Float64}},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64},
    ::MKLPardisoSolver)
    _has_mkl_pardiso_ext() || error(_mkl_pardiso_install_error())
    return _calculate_PTDF_matrix_MKLPardiso(A, BA, ref_bus_positions, dist_slack)
end

function _buildptdf_from_matrices(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{T, Int} where {T <: Union{Float32, Float64}},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64},
    ::AppleAccelerateLUSolver)
    _has_apple_accelerate_backend() || error(_apple_accelerate_unavailable_error())
    return _calculate_PTDF_matrix_AppleAccelerate(A, BA, ref_bus_positions, dist_slack)
end

"""
Function for internal use only.

Computes the PTDF matrix by factorizing ABA with `factorize` and solving for the BA columns
with `solve_columns!` (KLU or AppleAccelerate).

# Arguments
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence Matrix
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix
- `ref_bus_positions::Set{Int}`:
        vector containing the indexes of the reference slack buses.
- `dist_slack::Vector{Float64}`:
        vector containing the weights for the distributed slacks.
- `factorize`, `solve_columns!`: sparse factorization and in-place solve of the chosen backend.
"""
function _calculate_PTDF_matrix_sparse(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64},
    factorize,
    solve_columns!)
    linecount = size(BA, 2)
    buscount = size(BA, 1)
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    end
    if !isempty(dist_slack) && length(dist_slack) != buscount
        error("Distributed bus specification doesn't match the number of buses.")
    end
    length(ref_bus_positions) < buscount || error(
        "All buses are reference buses; PTDF is not defined.",
    )

    ABA = calculate_ABA_matrix(A, BA, ref_bus_positions)
    cache = factorize(ABA)
    valid_ix = setdiff(1:buscount, ref_bus_positions)
    PTDFm_t = zeros(buscount, linecount)
    solve_columns!(cache, BA[valid_ix, :], view(PTDFm_t, valid_ix, :))

    isempty(dist_slack) && return PTDFm_t

    @info "Distributed bus"
    slack_array = reshape(dist_slack ./ sum(dist_slack), 1, buscount)
    return PTDFm_t .- (slack_array * PTDFm_t)
end

function _calculate_PTDF_matrix_KLU(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64})
    return _calculate_PTDF_matrix_sparse(
        A, BA, ref_bus_positions, dist_slack, klu_factorize, solve_sparse!)
end

"""
Function for internal use only.

Computes the PTDF matrix by means of the LAPACK and BLAS functions for dense matrices.

# Arguments
- `A::Matrix{Int8}`:
        Incidence Matrix
- `BA::Matrix{T} where {T <: Union{Float32, Float64}}`:
        BA matrix
- `ref_bus_positions::Set{Int}`:
        vector containing the indexes of the reference slack buses.
- `dist_slack::Vector{Float64})`:
        vector containing the weights for the distributed slacks.
"""
function _calculate_PTDF_matrix_DENSE(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64})
    linecount = size(BA, 2)
    buscount = size(BA, 1)
    # Use dense calculation of ABA
    valid_ixs = setdiff(1:buscount, ref_bus_positions)
    ABA = Matrix(calculate_ABA_matrix(A, BA, ref_bus_positions))
    PTDFm_t = zeros(buscount, linecount)
    ABA_lu = LinearAlgebra.lu(ABA)
    BA = Matrix(BA[valid_ixs, :])
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    elseif isempty(dist_slack) && length(ref_bus_positions) < buscount
        PTDFm_t[valid_ixs, :] = ABA_lu \ BA
        return PTDFm_t
    elseif length(dist_slack) == buscount
        @info "Distributed bus"
        PTDFm_t[valid_ixs, :] = ABA_lu \ BA
        slack_array = reshape(dist_slack / sum(dist_slack), 1, buscount)
        return PTDFm_t .- (slack_array * PTDFm_t)
    else
        error("Distributed bus specification doesn't match the number of buses.")
    end

    return
end

# _calculate_PTDF_matrix_MKLPardiso is defined in ext/MKLPardisoExt.jl
# when Pardiso package is loaded

@static if Sys.isapple()
    """
    Function for internal use only.

    Computes the PTDF matrix using the internal Apple Accelerate backend
    (`AccelerateWrapper`). Available only on macOS — non-Apple callers are
    rejected by `_buildptdf_from_matrices` before reaching this entry.

    # Arguments
    - `A::SparseArrays.SparseMatrixCSC{Int8, Int}`: Incidence Matrix
    - `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`: BA matrix
    - `ref_bus_positions::Set{Int}`: indexes of reference slack buses
    - `dist_slack::Vector{Float64}`: distributed-slack weights
    """
    function _calculate_PTDF_matrix_AppleAccelerate(
        A::SparseArrays.SparseMatrixCSC{Int8, Int},
        BA::SparseArrays.SparseMatrixCSC{Float64, Int},
        ref_bus_positions::Set{Int},
        dist_slack::Vector{Float64},
    )
        return _calculate_PTDF_matrix_sparse(
            A, BA, ref_bus_positions, dist_slack,
            AccelerateWrapper.aa_factorize, AccelerateWrapper.solve_sparse!)
    end
end

"""
    PTDF(sys::PSY.System; dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(), linear_solver = _default_linear_solver(), tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE, network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct a Power Transfer Distribution Factor (PTDF) matrix from a PowerSystems.System by computing
the sensitivity of transmission line flows to bus power injections. This is the primary constructor
for PTDF analysis starting from system data.

# Arguments
- `sys::PSY.System`: The power system from which to construct the PTDF matrix

# Keyword Arguments
- `dist_slack::Dict{Int, Float64} = Dict{Int, Float64}()`:
        Dictionary mapping bus numbers to distributed slack weights for realistic slack modeling.
        Empty dictionary uses single slack bus (default behavior)
- `linear_solver::String = _default_linear_solver()`:
        Linear solver algorithm for matrix computations. Options: "KLU", "Dense", "MKLPardiso", "AppleAccelerateLU"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance. A `Float64` drops elements below it; the default
        [`AutoTolerance`](@ref) leaves this dense matrix exact.
- `network_reductions::Vector{NetworkReduction} = NetworkReduction[]`:
        Vector of network reduction algorithms to apply before matrix construction
- `include_constant_impedance_loads::Bool=true`:
        Whether to include constant impedance loads as shunt admittances in the network model
- Additional keyword arguments are passed to the underlying matrix constructors

# Returns
- `PTDF`: The constructed PTDF matrix structure containing:
  - Bus-to-impedance-arc injection sensitivity coefficients
  - Network topology information and reference bus identification
  - Sparsification tolerance and computational metadata

# Notes
- `dist_slack` weights are normalized to sum to 1.0; they require a single reference bus.
- Sparsification with `tol > eps()` reduces memory usage.
- Valid under DC power flow assumptions.
"""
function PTDF(sys::PSY.System;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    kwargs...,
)
    ybus = Ybus(
        sys;
        kwargs...,
    )
    return PTDF(ybus; dist_slack = dist_slack, linear_solver = linear_solver, tol = tol)
end

"""
    PTDF(ybus::Ybus; dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(), linear_solver = _default_linear_solver(), tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE, network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct a Power Transfer Distribution Factor (PTDF) matrix from existing Ybus matrix.
This constructor is more efficient when the prerequisite matrices are already available and provides
direct control over the underlying matrix computations.

# Arguments
- `ybus::Ybus`: The power system Ybus matrix from which to construct the PTDF matrix

# Keyword Arguments
- `dist_slack::Dict{Int, Float64} = Dict{Int, Float64}()`:
        Dictionary mapping bus numbers to distributed slack weights for realistic slack modeling.
        Empty dictionary uses single slack bus (default behavior)
- `linear_solver::String = _default_linear_solver()`:
        Linear solver algorithm for matrix computations. Options: "KLU", "Dense", "MKLPardiso", "AppleAccelerateLU"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance. A `Float64` drops elements below it; the default
        [`AutoTolerance`](@ref) leaves this dense matrix exact.

# Returns
- `PTDF`: The constructed PTDF matrix structure containing:
  - Bus-to-impedance-arc injection sensitivity coefficients
  - Network topology information and reference bus identification
  - Sparsification tolerance and computational metadata

# Notes
- `dist_slack` weights are normalized to sum to 1.0; they require a single reference bus.
- Sparsification with `tol > eps()` reduces memory usage.
- Valid under DC power flow assumptions.
"""
function PTDF(ybus::Ybus;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    A = IncidenceMatrix(ybus)
    BA = BA_Matrix(ybus)
    return PTDF(
        A,
        BA;
        dist_slack = dist_slack,
        linear_solver = linear_solver,
        tol = tol,
    )
end

"""
    PTDF(A::IncidenceMatrix, BA::BA_Matrix; dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(), linear_solver = _default_linear_solver(), tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE)

Construct a Power Transfer Distribution Factor (PTDF) matrix from existing incidence and BA matrices.
This constructor is more efficient when the prerequisite matrices are already available and provides
direct control over the underlying matrix computations.

# Arguments
- `A::IncidenceMatrix`: The incidence matrix containing bus-branch connectivity information
- `BA::BA_Matrix`: The branch susceptance weighted incidence matrix (B × A)

# Keyword Arguments
- `dist_slack::Dict{Int, Float64} = Dict{Int, Float64}()`:
        Dictionary mapping bus numbers to distributed slack participation factors.
        Empty dictionary uses single slack bus (reference bus from matrices)
- `linear_solver::String = _default_linear_solver()`:
        Linear solver algorithm for matrix computations. Options: "KLU", "Dense", "MKLPardiso", "AppleAccelerateLU"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance. A `Float64` drops elements below it; the default
        [`AutoTolerance`](@ref) leaves this dense matrix exact.

# Returns
- `PTDF`: The constructed PTDF matrix structure with injection-to-flow sensitivity coefficients

# Notes
- `A` and `BA` must share the same network reductions.
- `dist_slack` bus numbers must exist in the matrices; weights are normalized to sum to 1.0.
- An `AutoTolerance` is a no-op here: the dense PTDF stays a dense `Matrix{Float64}`. Pass a `Float64` `tol` to sparsify explicitly, or use `VirtualPTDF` at scale.
"""
function PTDF(
    A::IncidenceMatrix,
    BA::BA_Matrix;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    dist_slack_vector = if !(isempty(dist_slack))
        redistribute_dist_slack(dist_slack, A, get_network_reduction_data(A))
    else
        Float64[]
    end
    solver = resolve_linear_solver(linear_solver)
    if !isequal(get_network_reduction_data(A), get_network_reduction_data(BA))
        error("A and BA matrices have non-equivalent network reductions.")
    end
    S = _buildptdf_from_matrices(
        A.data,
        BA.data,
        Set(get_ref_bus_position(BA)),
        dist_slack_vector,
        solver,
    )
    tol_value = _dense_tol(tol)
    if tol_value > eps()
        S = sparsify(S, tol_value)
    end
    return PTDF(
        S,
        BA.axes,
        BA.lookup,
        BA.subnetwork_axes,
        tol_value,
        get_branch_catalog(BA),
    )
end

##############################################################################
########################### Auxiliary functions ##############################
##############################################################################

function Base.getindex(A::PTDF, branch_name::String, bus)
    multiplier, arc = get_branch_multiplier(A, branch_name)
    i, j = to_index(A, bus, arc)
    return A.data[i, j] * multiplier
end

# PTDF stores the transposed matrix. Overload indexing and how data is exported.
function Base.getindex(A::PTDF, arc, bus)
    i, j = to_index(A, bus, arc)
    return A.data[i, j]
end

function Base.getindex(
    A::PTDF,
    line_number::Union{Int, Colon},
    bus_number::Union{Int, Colon},
)
    return A.data[bus_number, line_number]
end

"""
    get_ptdf_data(ptdf::PTDF)

Extract the PTDF matrix data in the standard orientation (non-transposed).

# Arguments
- `ptdf::PTDF`: The PTDF structure from which to extract data

# Returns
- `AbstractArray{Float64, 2}`: The PTDF matrix data with standard orientation
"""
function get_ptdf_data(ptdf::PTDF)
    return transpose(ptdf.data)
end

function get_tol(ptdf::PTDF)
    return ptdf.tol
end

function redistribute_dist_slack(
    dist_slack::Dict{Int, Float64},
    A::IncidenceMatrix,
    nr::NetworkReductionData,
)
    dist_slack_vector = zeros(length(A.axes[2]))
    for (bus_no, dist_slack_factor) in dist_slack
        bus_no_ = get(nr.reverse_bus_search_map, bus_no, bus_no)
        if !haskey(A.lookup[2], bus_no_)
            throw(
                IS.InvalidValue(
                    "Bus number $bus_no_ not found in the incidence matrix. Correct your slack distribution specification.",
                ),
            )
        end
        ix = A.lookup[2][bus_no_]
        dist_slack_vector[ix] += dist_slack_factor
    end
    return dist_slack_vector
end
