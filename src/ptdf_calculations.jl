"""
Structure containing the Power Transfer Distribution Factor (PTDF) matrix and related power system data.

The PTDF matrix contains sensitivity coefficients that quantify how power injections at buses
affect the power flows on transmission lines. Each element ``\\mathrm{PTDF}[i,j]`` represents the
incremental change in flow on line ``i`` due to a unit power injection at bus ``j``, under DC power
flow assumptions.

# Fields
- `data::M <: AbstractArray{Float64, 2}`:
        The PTDF matrix data stored in transposed form for computational efficiency.
        Element (i,j) represents the sensitivity of line j flow to bus i injection
- `axes::Ax`:
        Tuple containing `(bus_numbers, branch_identifiers)` for matrix dimensions
- `lookup::L <: NTuple{2, Dict}`:
        Tuple of dictionaries providing fast lookup from bus/branch identifiers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`:
        Mapping from reference bus numbers to their corresponding subnetwork axes
- `tol::Base.RefValue{Float64}`:
        Tolerance threshold used for matrix sparsification (elements below this value are dropped)
- `network_reduction_data::NetworkReductionData`:
        Container for network reduction information applied during matrix construction

# Mathematical Properties
- **Matrix Form**: ``\\mathrm{PTDF}[i,j] = \\partial f_i / \\partial P_j`` where ``f_i`` is flow on line ``i`` and ``P_j`` is injection at bus ``j``
- **Dimensions**: `(n_buses × n_arcs)` for all buses and impedance arcs
- **Linear Superposition**: ``f_i = \\sum_j \\mathrm{PTDF}[i,j] \\, P_j`` over all injections ``P_j``
- **Physical Meaning**: Values represent the fraction of bus injection that flows through each line
- **Reference Bus**: Rows corresponding to reference buses are typically zero

# Applications
- **Power Flow Analysis**: Rapid calculation of line flows for given injection patterns
- **Sensitivity Studies**: Evaluate impact of generation/load changes on transmission flows
- **Congestion Management**: Identify lines affected by specific injection changes
- **Market Analysis**: Support nodal pricing and transmission rights calculations
- **Planning Studies**: Assess transmission utilization under various scenarios

# Computational Features
- **Matrix Storage**: Stored in transposed form (bus × branch) for efficient computation
- **Sparsification**: Small elements removed based on tolerance to reduce memory usage
- **Reference Bus Handling**: Reference bus injections automatically handled in calculations
- **Distributed Slack**: Supports distributed slack bus configurations for improved realism

# Usage Notes
- Access via `ptdf[bus, line]` returns the sensitivity coefficient
- Matrix indexing uses bus numbers and branch identifiers
- Sparsification improves memory efficiency but may introduce small numerical errors
- Results valid under DC power flow assumptions (neglects voltage magnitudes and reactive power)
- Reference bus choice affects the specific values but not the relative sensitivities
"""
struct PTDF{Ax, L <: NTuple{2, Dict}, M <: AbstractArray{Float64, 2}} <:
       PowerNetworkMatrix{Float64}
    data::M
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    network_reduction_data::NetworkReductionData
end

get_axes(M::PTDF) = M.axes
get_lookup(M::PTDF) = M.lookup
get_ref_bus(M::PTDF) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::PTDF) = [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::PTDF) = M.network_reduction_data
get_bus_axis(M::PTDF) = M.axes[1]
get_bus_lookup(M::PTDF) = M.lookup[1]
get_arc_axis(M::PTDF) = M.axes[2]
get_arc_lookup(M::PTDF) = M.lookup[2]

stores_transpose(::PTDF) = true

"""
Deserialize a PTDF from an HDF5 file.

# Arguments
- `filename::AbstractString`: File containing a serialized PTDF.
"""
PTDF(filename::AbstractString) = from_hdf5(PTDF, filename)

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

Computes the PTDF matrix by means of the KLU.LU factorization for sparse matrices.

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
function _calculate_PTDF_matrix_KLU(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64})
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
    cache = klu_factorize(ABA)
    valid_ix = setdiff(1:buscount, ref_bus_positions)
    PTDFm_t = zeros(buscount, linecount)
    solve_sparse!(cache, BA[valid_ix, :], view(PTDFm_t, valid_ix, :))

    isempty(dist_slack) && return PTDFm_t

    @info "Distributed bus"
    slack_array = reshape(dist_slack ./ sum(dist_slack), 1, buscount)
    return PTDFm_t .- (slack_array * PTDFm_t)
end

function _binfo_check(binfo::Int)
    if binfo != 0
        if binfo < 0
            error("Illegal Argument in Inputs")
        elseif binfo > 0
            error("Singular value in factorization. Possibly there is an islanded bus")
        else
            @assert false
        end
    end
    return
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
    (ABA, bipiv, binfo) = getrf!(ABA)
    _binfo_check(binfo)
    BA = Matrix(BA[valid_ixs, :])
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    elseif isempty(dist_slack) && length(ref_bus_positions) < buscount
        getrs!('N', ABA, bipiv, BA)
        PTDFm_t[valid_ixs, :] = BA
        return PTDFm_t
    elseif length(dist_slack) == buscount
        @info "Distributed bus"
        getrs!('N', ABA, bipiv, BA)
        PTDFm_t[valid_ixs, :] = BA
        slack_array = dist_slack / sum(dist_slack)
        slack_array = reshape(slack_array, 1, buscount)
        return PTDFm_t -
               gemm('N', 'N', ones(buscount, 1), gemm('N', 'N', slack_array, PTDFm_t))
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
    rejected by `_buildptdf_from_matrices` before reaching this entry. Shape
    mirrors `_calculate_PTDF_matrix_KLU`: factor ABA via LU, then solve
    `ABA · X = BA[valid_ix, :]` via the block-packed `solve_sparse!`.

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
        cache = AccelerateWrapper.aa_factorize(ABA)
        valid_ix = setdiff(1:buscount, ref_bus_positions)
        PTDFm_t = zeros(buscount, linecount)
        AccelerateWrapper.solve_sparse!(
            cache,
            BA[valid_ix, :],
            view(PTDFm_t, valid_ix, :),
        )

        isempty(dist_slack) && return PTDFm_t

        @info "Distributed bus"
        slack_array = reshape(dist_slack ./ sum(dist_slack), 1, buscount)
        return PTDFm_t .- (slack_array * PTDFm_t)
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
        Linear solver algorithm for matrix computations. Options: "KLU", "AppleAccelerateLU", "Dense", "MKLPardiso"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance for dropping small matrix elements to reduce memory usage.
        A `Float64` applies a fixed absolute cutoff at any size; an [`AutoTolerance`](@ref)
        (the default) applies a relative per-row cutoff on large virtual matrices only.
- `network_reductions::Vector{NetworkReduction} = NetworkReduction[]`:
        Vector of network reduction algorithms to apply before matrix construction
- `include_constant_impedance_loads::Bool=true`:
        Whether to include constant impedance loads as shunt admittances in the network model
- `subnetwork_algorithm=iterative_union_find`:
        Algorithm used for identifying electrical islands and connected components
- Additional keyword arguments are passed to the underlying matrix constructors

# Returns
- `PTDF`: The constructed PTDF matrix structure containing:
  - Bus-to-impedance-arc injection sensitivity coefficients
  - Network topology information and reference bus identification
  - Sparsification tolerance and computational metadata

# Construction Process
1. **Ybus Construction**: Creates system admittance matrix with specified reductions
2. **Incidence Matrix**: Builds bus-branch connectivity matrix A
3. **BA Matrix**: Computes branch susceptance weighted incidence matrix
4. **PTDF Computation**: Calculates power transfer distribution factors using ``A^\\top B A``
5. **Distributed Slack**: Applies distributed slack correction if specified
6. **Sparsification**: Removes small elements based on tolerance threshold

# Distributed Slack Configuration
- **Single Slack**: Empty `dist_slack` dictionary uses conventional single slack bus
- **Distributed Slack**: Dictionary maps bus numbers to participation factors
- **Normalization**: Participation factors automatically normalized to sum to 1.0
- **Physical Meaning**: Distributed slack better represents generator response to load changes

# Linear Solver Options
- **"KLU"**: Sparse LU factorization (default off Apple hardware, recommended for most cases)
- **"AppleAccelerateLU"**: Apple Accelerate sparse LU (default on macOS 15.5+ Apple hardware)
- **"Dense"**: Dense matrix operations (faster for small systems, higher memory usage)
- **"MKLPardiso"**: Intel MKL Pardiso solver (requires MKL library, best for very large systems)

# Mathematical Foundation
The PTDF matrix is computed as
```math
\\mathrm{PTDF} = (A^\\top B A)^{-1} A^\\top B,
```
where ``A`` is the incidence matrix (the [`IncidenceMatrix`](@ref)) and ``B`` the branch
susceptance matrix (see [`BA_Matrix`](@ref)).

# Notes
- Results are valid under DC power flow assumptions (linear approximation)
- Reference bus selection affects specific values but not relative sensitivities
- Sparsification with `tol > eps()` can significantly reduce memory usage
- Network reductions improve computational efficiency for large systems
- Distributed slack provides more realistic representation of system response
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
        Linear solver algorithm for matrix computations. Options: "KLU", "AppleAccelerateLU", "Dense", "MKLPardiso"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance for dropping small matrix elements to reduce memory usage.
        A `Float64` applies a fixed absolute cutoff at any size; an [`AutoTolerance`](@ref)
        (the default) applies a relative per-row cutoff on large virtual matrices only.

# Returns
- `PTDF`: The constructed PTDF matrix structure containing:
  - Bus-to-impedance-arc injection sensitivity coefficients
  - Network topology information and reference bus identification
  - Sparsification tolerance and computational metadata

# Construction Process
1. **Incidence Matrix**: Builds bus-branch connectivity matrix A (from Ybus matrix)
2. **BA Matrix**: Computes branch susceptance weighted incidence matrix
3. **PTDF Computation**: Calculates power transfer distribution factors using ``A^\\top B A``
4. **Distributed Slack**: Applies distributed slack correction if specified
5. **Sparsification**: Removes small elements based on tolerance threshold

# Distributed Slack Configuration
- **Single Slack**: Empty `dist_slack` dictionary uses conventional single slack bus
- **Distributed Slack**: Dictionary maps bus numbers to participation factors
- **Normalization**: Participation factors automatically normalized to sum to 1.0
- **Physical Meaning**: Distributed slack better represents generator response to load changes

# Linear Solver Options
- **"KLU"**: Sparse LU factorization (default off Apple hardware, recommended for most cases)
- **"AppleAccelerateLU"**: Apple Accelerate sparse LU (default on macOS 15.5+ Apple hardware)
- **"Dense"**: Dense matrix operations (faster for small systems, higher memory usage)
- **"MKLPardiso"**: Intel MKL Pardiso solver (requires MKL library, best for very large systems)

# Mathematical Foundation
The PTDF matrix is computed as
```math
\\mathrm{PTDF} = (A^\\top B A)^{-1} A^\\top B,
```
where ``A`` is the incidence matrix (the [`IncidenceMatrix`](@ref)) and ``B`` the branch
susceptance matrix (see [`BA_Matrix`](@ref)).

# Notes
- Results are valid under DC power flow assumptions (linear approximation)
- Reference bus selection affects specific values but not relative sensitivities
- Sparsification with `tol > eps()` can significantly reduce memory usage
- Network reductions improve computational efficiency for large systems
- Distributed slack provides more realistic representation of system response
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
        Linear solver algorithm for matrix computations. Options: "KLU", "AppleAccelerateLU", "Dense", "MKLPardiso"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance for dropping small matrix elements to reduce memory usage.
        A `Float64` applies a fixed absolute cutoff at any size; an [`AutoTolerance`](@ref)
        (the default) applies a relative per-row cutoff on large virtual matrices only.

# Returns
- `PTDF`: The constructed PTDF matrix structure with injection-to-flow sensitivity coefficients

# Mathematical Computation
The PTDF matrix is computed using the relationship
```math
\\mathrm{PTDF} = (A^\\top B A)^{-1} A^\\top B,
```
where:
- ``A`` is the incidence matrix representing bus-branch connectivity
- ``B`` is the diagonal susceptance matrix (embedded in BA matrix)
- The computation involves solving the ABA linear system for efficiency

# Distributed Slack Handling
- **Single Slack**: Uses reference bus identified from input matrices
- **Distributed Slack**: Applies participation factor corrections to final PTDF
- **Automatic Processing**: Dictionary converted to vector form matching matrix dimensions
- **Validation**: Ensures distributed slack bus numbers exist in the network
- **Normalization**: Participation factors automatically normalized to maintain power balance

# Network Consistency Requirements
- **Reduction Compatibility**: Both input matrices must have equivalent network reduction states
- **Reference Alignment**: BA matrix reference buses determine the PTDF reference framework
- **Topology Consistency**: Matrices must represent the same network topology

# Performance Considerations
- **Matrix Reuse**: More efficient when A and BA matrices are already computed
- **Memory Management**: Sparsification reduces storage requirements significantly
- **Solver Selection**: KLU recommended for sparse systems, Dense for small networks
- **Computational Efficiency**: Avoids redundant system matrix construction

# Error Handling and Validation
- **Matrix Compatibility**: Validates that A and BA have consistent network reductions
- **Slack Validation**: Checks that distributed slack buses exist in the matrix structure
- **Solver Validation**: Ensures selected linear solver is supported and available
- **Numerical Stability**: Handles singular systems and provides informative error messages

# Usage Recommendations
- **Preferred Method**: Use when incidence and BA matrices are already available
- **Repeated Calculations**: Ideal for multiple PTDF computations with different slack configurations
- **Large Systems**: Consider sparsification for memory efficiency
- **Distributed Slack**: Provides more realistic modeling of generator response to load changes
"""
# Build the dense PTDF matrix S and resolve its sparsification tolerance. A
# numeric tol is an absolute cutoff; an AutoTolerance is a no-op (`_dense_tol`),
# since the dense PTDF is the small-system path and must stay a dense
# `Matrix{Float64}` (preserving `DC_PTDF_Matrix` and downstream dispatch). Pass a
# `Float64` tol to sparsify a dense PTDF explicitly, or use VirtualPTDF at scale.
function _dense_ptdf_with_tol(
    tol::Union{Float64, AutoTolerance},
    solver::LinearSolverType,
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    dist_slack::Vector{Float64},
)
    S = _buildptdf_from_matrices(A, BA, ref_bus_positions, dist_slack, solver)
    return (S, _dense_tol(tol))
end

function PTDF(
    A::IncidenceMatrix,
    BA::BA_Matrix;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    dist_slack_vector = if !(isempty(dist_slack))
        redistribute_dist_slack(dist_slack, A, A.network_reduction_data)
    else
        Float64[]
    end
    solver = resolve_linear_solver(linear_solver)
    if !isequal(A.network_reduction_data, BA.network_reduction_data)
        error("A and BA matrices have non-equivalent network reductions.")
    end
    axes = BA.axes
    lookup = BA.lookup
    A_matrix = A.data
    subnetwork_axes = BA.subnetwork_axes
    ref_bus_positions = get_ref_bus_position(BA)
    S, tol_value = _dense_ptdf_with_tol(
        tol,
        solver,
        A_matrix,
        BA.data,
        Set(ref_bus_positions),
        dist_slack_vector,
    )
    if tol_value > eps()
        return PTDF(
            sparsify(S, tol_value),
            axes,
            lookup,
            subnetwork_axes,
            Ref(tol_value),
            BA.network_reduction_data,
        )
    else
        return PTDF(
            S,
            axes,
            lookup,
            subnetwork_axes,
            Ref(tol_value),
            BA.network_reduction_data,
        )
    end
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
