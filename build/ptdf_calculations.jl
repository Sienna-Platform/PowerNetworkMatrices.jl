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
- `tol::Base.RefValue{Float64}`:
        Tolerance threshold used for matrix sparsification (elements below this value are dropped)
- `network_reduction_data::NetworkReductionData`:
        Container for network reduction information applied during matrix construction

# Mathematical Properties
- **Matrix Form**: PTDF[i,j] = ∂f_i/∂P_j where f_i is flow on line i, P_j is injection at bus j
- **Dimensions**: (n_buses × n_arcs) for all buses and impedance arcs
- **Linear Superposition**: Total flow = Σ(PTDF[i,j] × P_j) for all injections P_j
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
    linear_solver::String)
    if linear_solver == "KLU"
        PTDFm = _calculate_PTDF_matrix_KLU(A, BA, ref_bus_positions, dist_slack)
    elseif linear_solver == "Dense"
        # Convert SparseMatrices to Dense
        PTDFm = _calculate_PTDF_matrix_DENSE(
            A,
            BA,
            ref_bus_positions,
            dist_slack,
        )
    elseif linear_solver == "MKLPardiso"
        if !_has_mkl_pardiso_ext()
            error(
                "The MKL/Pardiso extension is not available. Install MKL and Pardiso packages: using Pkg; Pkg.add([\"MKL\", \"Pardiso\"])",
            )
        end
        PTDFm =
            _calculate_PTDF_matrix_MKLPardiso(A, BA, ref_bus_positions, dist_slack)
    elseif linear_solver == "AppleAccelerate"
        if !_has_apple_accelerate_ext()
            error(
                "AppleAccelerate extension is not available. This solver is only available on macOS. Install AppleAccelerate: using Pkg; Pkg.add(\"AppleAccelerate\")",
            )
        end
        PTDFm =
            _calculate_PTDF_matrix_AppleAccelerate(A, BA, ref_bus_positions, dist_slack)
    end

    return PTDFm
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

    ABA = calculate_ABA_matrix(A, BA, ref_bus_positions)
    K = klu(ABA)
    # initialize matrices for evaluation
    valid_ix = setdiff(1:buscount, ref_bus_positions)
    PTDFm_t = zeros(buscount, linecount)
    copyto!(PTDFm_t, BA)
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    elseif isempty(dist_slack) && length(ref_bus_positions) < buscount
        PTDFm_t[valid_ix, :] = KLU.solve!(K, PTDFm_t[valid_ix, :])
        PTDFm_t[collect(ref_bus_positions), :] .= 0.0
        return PTDFm_t
    elseif length(dist_slack) == buscount
        @info "Distributed bus"
        PTDFm_t[valid_ix, :] = KLU.solve!(K, PTDFm_t[valid_ix, :])
        PTDFm_t[collect(ref_bus_positions), :] .= 0.0
        slack_array = dist_slack / sum(dist_slack)
        slack_array = reshape(slack_array, 1, buscount)
        return PTDFm_t .- (slack_array * PTDFm_t)
    else
        error("Distributed bus specification doesn't match the number of buses.")
    end

    return
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
# when MKL and Pardiso packages are loaded

# _calculate_PTDF_matrix_AppleAccelerate is defined in ext/AppleAccelerateExt.jl
# when AppleAccelerate package is loaded

"""
    PTDF(sys::PSY.System; dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(), linear_solver = "KLU", tol::Float64 = eps(), network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct a Power Transfer Distribution Factor (PTDF) matrix from a PowerSystems.System by computing
the sensitivity of transmission line flows to bus power injections. This is the primary constructor
for PTDF analysis starting from system data.

# Arguments
- `sys::PSY.System`: The power system from which to construct the PTDF matrix

# Keyword Arguments
- `dist_slack::Dict{Int, Float64} = Dict{Int, Float64}()`:
        Dictionary mapping bus numbers to distributed slack weights for realistic slack modeling.
        Empty dictionary uses single slack bus (default behavior)
- `linear_solver::String = "KLU"`:
        Linear solver algorithm for matrix computations. Options: "KLU", "Dense", "MKLPardiso"
- `tol::Float64 = eps()`:
        Sparsification tolerance for dropping small matrix elements to reduce memory usage
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
4. **PTDF Computation**: Calculates power transfer distribution factors using A^T × B^(-1) × A
5. **Distributed Slack**: Applies distributed slack correction if specified
6. **Sparsification**: Removes small elements based on tolerance threshold

# Distributed Slack Configuration
- **Single Slack**: Empty `dist_slack` dictionary uses conventional single slack bus
- **Distributed Slack**: Dictionary maps bus numbers to participation factors
- **Normalization**: Participation factors automatically normalized to sum to 1.0
- **Physical Meaning**: Distributed slack better represents generator response to load changes

# Linear Solver Options
- **"KLU"**: Sparse LU factorization (default, recommended for most cases)
- **"Dense"**: Dense matrix operations (faster for small systems, higher memory usage)
- **"MKLPardiso"**: Intel MKL Pardiso solver (requires MKL library, best for very large systems)

# Mathematical Foundation
The PTDF matrix is computed as:
```
PTDF = A^T × (A^T × B × A)^(-1) × A^T × B
```
where A is the incidence matrix and B is the susceptance matrix.

# Notes
- Results are valid under DC power flow assumptions (linear approximation)
- Reference bus selection affects specific values but not relative sensitivities
- Sparsification with `tol > eps()` can significantly reduce memory usage
- Network reductions improve computational efficiency for large systems
- Distributed slack provides more realistic representation of system response
"""
function PTDF(sys::PSY.System;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = "KLU",
    tol::Float64 = eps(),
    kwargs...,
)
    Ymatrix = Ybus(
        sys;
        kwargs...,
    )
    A = IncidenceMatrix(Ymatrix)
    BA = BA_Matrix(Ymatrix)
    return PTDF(
        A,
        BA;
        dist_slack = dist_slack,
        linear_solver = linear_solver,
        tol = tol,
    )
end

"""
    PTDF(A::IncidenceMatrix, BA::BA_Matrix; dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(), linear_solver = "KLU", tol::Float64 = eps())

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
- `linear_solver::String = "KLU"`:
        Linear solver algorithm for matrix computations. Options: "KLU", "Dense", "MKLPardiso"
- `tol::Float64 = eps()`:
        Sparsification tolerance for dropping small matrix elements to reduce memory usage

# Returns
- `PTDF`: The constructed PTDF matrix structure with injection-to-flow sensitivity coefficients

# Mathematical Computation
The PTDF matrix is computed using the relationship:
```
PTDF = (A^T × B × A)^(-1) × A^T × B
```
where:
- A is the incidence matrix representing bus-branch connectivity
- B is the diagonal susceptance matrix (embedded in BA matrix)
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
function PTDF(
    A::IncidenceMatrix,
    BA::BA_Matrix;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = "KLU",
    tol::Float64 = eps(),
)
    if !(isempty(dist_slack))
        dist_slack = redistribute_dist_slack(dist_slack, A, A.network_reduction_data)
    else
        dist_slack = Float64[]
    end
    validate_linear_solver(linear_solver)
    if !isequal(A.network_reduction_data, BA.network_reduction_data)
        error("A and BA matrices have non-equivalent network reductions.")
    end
    axes = BA.axes
    lookup = BA.lookup
    A_matrix = A.data
    subnetwork_axes = BA.subnetwork_axes
    ref_bus_positions = get_ref_bus_position(BA)
    S = _buildptdf_from_matrices(
        A_matrix,
        BA.data,
        Set(ref_bus_positions),
        dist_slack,
        linear_solver,
    )
    if tol > eps()
        return PTDF(
            sparsify(S, tol),
            axes,
            lookup,
            subnetwork_axes,
            Ref(tol),
            BA.network_reduction_data,
        )
    else
        return PTDF(
            S,
            axes,
            lookup,
            subnetwork_axes,
            Ref(tol),
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
