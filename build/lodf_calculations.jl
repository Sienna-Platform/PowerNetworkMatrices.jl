"""
Structure containing the Line Outage Distribution Factor (LODF) matrix and related power system data.

The LODF matrix contains sensitivity coefficients that quantify how the outage of one transmission 
line affects the power flows on all other lines in the system. Each element LODF[i,j] represents 
the change in flow on line i when line j is taken out of service, normalized by the pre-outage 
flow on line j.

# Fields
- `data::M <: AbstractArray{Float64, 2}`:
        The LODF matrix data stored in transposed form for computational efficiency. 
        Element (i,j) represents the sensitivity of line j flow to line i outage
- `axes::Ax`:
        Tuple of identical branch/arc identifier vectors for both matrix dimensions
- `lookup::L <: NTuple{2, Dict}`:
        Tuple of identical dictionaries providing fast lookup from branch identifiers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`:
        Mapping from reference bus numbers to their corresponding subnetwork branch axes
- `tol::Base.RefValue{Float64}`:
        Tolerance threshold used for matrix sparsification (elements below this value are dropped)
- `network_reduction_data::NetworkReductionData`:
        Container for network reduction information applied during matrix construction

# Mathematical Properties
- **Matrix Form**: LODF[i,j] = ∂f_i/∂P_j where f_i is flow on line i, P_j is injection change due to line j outage
- **Dimensions**: (n_branches × n_branches) for all transmission lines in the system
- **Diagonal Elements**: Always -1 (100% flow reduction on the outaged line itself)
- **Symmetry**: Generally non-symmetric matrix reflecting directional flow sensitivities
- **Physical Meaning**: Values represent fraction of pre-outage flow that redistributes to other lines

# Applications
- **Contingency Analysis**: Evaluate impact of single line outages on system flows
- **Security Assessment**: Identify critical transmission bottlenecks and vulnerable lines
- **System Planning**: Analyze network robustness and redundancy requirements
- **Real-time Operations**: Support operator decision-making for preventive/corrective actions

# Computational Notes
- **Storage**: Matrix stored in transposed form for efficient column-wise access patterns
- **Sparsification**: Small elements removed based on tolerance to reduce memory usage
- **Linear Approximation**: Based on DC power flow assumptions (neglects voltage magnitudes and reactive power)
- **Single Contingencies**: Designed for single line outage analysis (N-1 contingencies)

# Usage Notes
- Access via `lodf[monitored_line, outaged_line]` returns sensitivity coefficient
- Diagonal elements are always -1.0 representing complete flow loss on outaged line
- Matrix sparsification improves performance but may introduce small numerical errors
- Results valid under DC power flow assumptions and normal operating conditions
"""
struct LODF{Ax, L <: NTuple{2, Dict}, M <: AbstractArray{Float64, 2}} <:
       PowerNetworkMatrix{Float64}
    data::M
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    network_reduction_data::NetworkReductionData
end
stores_transpose(::LODF) = true

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
    linear_solver::String = "KLU",
)
    if linear_solver == "KLU"
        lodf_t = _calculate_LODF_matrix_KLU(a, ptdf)
    elseif linear_solver == "Dense"
        lodf_t = _calculate_LODF_matrix_DENSE(a, ptdf)
    elseif linear_solver == "MKLPardiso"
        if !_has_mkl_pardiso_ext()
            error(
                "The MKL/Pardiso extension is not available. Install MKL and Pardiso packages: using Pkg; Pkg.add([\"MKL\", \"Pardiso\"])",
            )
        end
        lodf_t = _calculate_LODF_matrix_MKLPardiso(a, ptdf)
    elseif linear_solver == "AppleAccelerate"
        if !_has_apple_accelerate_ext()
            error(
                "AppleAccelerate extension is not available. This solver is only available on macOS. Install AppleAccelerate: using Pkg; Pkg.add(\"AppleAccelerate\")",
            )
        end
        lodf_t = _calculate_LODF_matrix_AppleAccelerate(a, ptdf)
    end
    return lodf_t
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    k::KLU.KLUFactorization{Float64, Int},
    ba::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    linear_solver::String,
)
    if linear_solver == "KLU"
        lodf_t = _calculate_LODF_matrix_KLU(a, k, ba, ref_bus_positions)
    else
        error("Other linear solvers are not implemented.")
    end
    return lodf_t
end

function _calculate_LODF_matrix_KLU(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    k::KLU.KLUFactorization{Float64, Int},
    ba::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
)
    linecount = size(ba, 2)
    # get inverse of aba
    first_ = zeros(size(a, 2), size(a, 1))
    valid_ix = setdiff(1:size(a, 2), ref_bus_positions)
    copyto!(first_, transpose(a))
    first_[valid_ix, :] = KLU.solve!(k, first_[valid_ix, :])
    first_[collect(ref_bus_positions), :] .= 0.0
    ptdf_denominator = first_' * ba

    m_I = Int[]
    m_V = Float64[]
    for iline in 1:linecount
        if (1.0 - ptdf_denominator[iline, iline]) < LODF_ENTRY_TOLERANCE
            push!(m_I, iline)
            push!(m_V, 1.0)
        else
            push!(m_I, iline)
            push!(m_V, 1 - ptdf_denominator[iline, iline])
        end
    end
    Dem_LU = klu(SparseArrays.sparse(m_I, m_I, m_V))
    KLU.solve!(Dem_LU, ptdf_denominator)
    ptdf_denominator[SparseArrays.diagind(ptdf_denominator)] .= -1.0
    return ptdf_denominator
end

function _calculate_LODF_matrix_KLU(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    linecount = size(ptdf, 2)
    ptdf_denominator_t = a * ptdf
    m_I = Int[]
    m_V = Float64[]
    for iline in 1:linecount
        if (1.0 - ptdf_denominator_t[iline, iline]) < LODF_ENTRY_TOLERANCE
            push!(m_I, iline)
            push!(m_V, 1.0)
        else
            push!(m_I, iline)
            push!(m_V, 1 - ptdf_denominator_t[iline, iline])
        end
    end
    Dem_LU = klu(SparseArrays.sparse(m_I, m_I, m_V))
    lodf_t = Dem_LU \ ptdf_denominator_t
    lodf_t[SparseArrays.diagind(lodf_t)] .= -1.0

    return lodf_t
end

function _calculate_LODF_matrix_DENSE(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    linecount = size(ptdf, 2)
    ptdf_denominator_t = a * ptdf
    m_V = Float64[]
    for iline in 1:linecount
        if (1.0 - ptdf_denominator_t[iline, iline]) < LODF_ENTRY_TOLERANCE
            push!(m_V, 1.0)
        else
            push!(m_V, 1.0 - ptdf_denominator_t[iline, iline])
        end
    end
    (mV, bipiv, binfo) = getrf!(Matrix(LinearAlgebra.diagm(m_V)))
    _binfo_check(binfo)
    getrs!('N', mV, bipiv, ptdf_denominator_t)
    ptdf_denominator_t[LinearAlgebra.diagind(ptdf_denominator_t)] .= -1.0
    return ptdf_denominator_t
end

# _pardiso_sequential_LODF!, _pardiso_single_LODF!, _calculate_LODF_matrix_MKLPardiso
# are defined in ext/MKLPardisoExt.jl when MKL and Pardiso packages are loaded

# _calculate_LODF_matrix_AppleAccelerate is defined in ext/AppleAccelerateExt.jl
# when AppleAccelerate package is loaded

"""
    LODF(sys::PSY.System; linear_solver::String = "KLU", tol::Float64 = eps(), network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct a Line Outage Distribution Factor (LODF) matrix from a PowerSystems.System by computing 
the sensitivity of line flows to single line outages. This is the primary constructor for LODF 
analysis starting from system data.

# Arguments
- `sys::PSY.System`: The power system from which to construct the LODF matrix

# Keyword Arguments
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
- `LODF`: The constructed LODF matrix structure containing:
  - Line-to-line outage sensitivity coefficients
  - Network topology information and branch identifiers
  - Sparsification tolerance and computational metadata

# Construction Process
1. **Ybus Construction**: Creates system admittance matrix with specified reductions
2. **Incidence Matrix**: Builds bus-branch connectivity matrix A
3. **BA Matrix**: Computes branch susceptance weighted incidence matrix
4. **PTDF Calculation**: Derives power transfer distribution factors
5. **LODF Computation**: Calculates line outage distribution factors from PTDF
6. **Sparsification**: Applies tolerance threshold to reduce matrix density

# Linear Solver Options
- **"KLU"**: Sparse LU factorization (default, recommended for most cases)
- **"Dense"**: Dense matrix operations (faster for small systems)
- **"MKLPardiso"**: Intel MKL Pardiso solver (requires MKL, best for very large systems)

# Mathematical Foundation
The LODF matrix is computed using the relationship:
```
LODF = (A * PTDF) / (1 - diag(A * PTDF))
```
where A is the incidence matrix and PTDF is the power transfer distribution factor matrix.

# Notes
- Sparsification with `tol > eps()` can significantly reduce memory usage
- Network reductions can improve computational efficiency for large systems
- Results are valid under DC power flow assumptions (linear approximation)
- Diagonal elements are always -1.0 representing complete flow loss on outaged lines
- For very large systems, consider using "MKLPardiso" solver with appropriate chunk size
"""
function LODF(
    sys::PSY.System;
    linear_solver::String = "KLU",
    tol::Float64 = eps(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    Ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    A = IncidenceMatrix(Ymatrix)
    BA = BA_Matrix(Ymatrix)
    ptdf = PTDF(A, BA)
    return LODF(A, ptdf; linear_solver = linear_solver, tol = tol, kwargs...)
end

"""
    LODF(A::IncidenceMatrix, PTDFm::PTDF; linear_solver::String = "KLU", tol::Float64 = eps())

Construct a Line Outage Distribution Factor (LODF) matrix from existing incidence and PTDF matrices.
This constructor is more efficient when the prerequisite matrices are already available.

# Arguments
- `A::IncidenceMatrix`: The incidence matrix containing bus-branch connectivity information
- `PTDFm::PTDF`: The power transfer distribution factor matrix (should be non-sparsified for accuracy)

# Keyword Arguments
- `linear_solver::String = "KLU"`: 
        Linear solver algorithm for matrix computations. Options: "KLU", "Dense", "MKLPardiso"
- `tol::Float64 = eps()`: 
        Sparsification tolerance for the LODF matrix (not applied to input PTDF)

# Returns
- `LODF`: The constructed LODF matrix structure with line outage sensitivity coefficients

# Mathematical Computation
The LODF matrix is computed using the formula:
```
LODF = (A * PTDF) / (1 - diag(A * PTDF))
```
where:
- A is the incidence matrix representing bus-branch connectivity
- PTDF contains power transfer distribution factors
- The denominator (1 - diagonal terms) accounts for the outaged line's own flow

# Important Notes
- **PTDF Sparsification**: The input PTDF matrix should be non-sparsified (constructed with default tolerance) to avoid accuracy issues
- **Tolerance Application**: The `tol` parameter only affects LODF sparsification, not the input PTDF
- **Network Consistency**: Both input matrices must have equivalent network reduction states
- **Diagonal Elements**: Automatically set to -1.0 representing complete flow loss on outaged lines

# Performance Considerations
- **Matrix Validation**: Warns if input PTDF was sparsified and converts to dense format for accuracy
- **Memory Usage**: Sparsification with `tol > eps()` can significantly reduce memory requirements
- **Computational Efficiency**: More efficient than system-based constructor when matrices exist

# Error Handling
- Validates that incidence and PTDF matrices have consistent network reduction data
- Issues warnings if sparsified PTDF matrices are used (potential accuracy issues)
- Supports automatic conversion of sparse PTDF to dense format when necessary

# Linear Solver Selection
- **"KLU"**: Recommended for most applications (sparse, numerically stable)
- **"Dense"**: Faster for smaller systems but higher memory usage
- **"MKLPardiso"**: Best performance for very large systems (requires MKL library)
"""
function LODF(
    A::IncidenceMatrix,
    PTDFm::PTDF;
    linear_solver::String = "KLU",
    tol::Float64 = eps(),
)
    validate_linear_solver(linear_solver)
    subnetwork_axes = make_arc_arc_subnetwork_axes(A)

    if PTDFm.tol.x > 1e-15
        warn_msg = string(
            "The argument `tol` in the PTDF matrix was set to a value different than the default one.\n",
            "The resulting LODF can include unexpected rounding errors.\n",
        )
        @warn(warn_msg)
        PTDFm_data = Matrix(PTDFm.data)
    else
        PTDFm_data = PTDFm.data
    end

    if !isequal(A.network_reduction_data, PTDFm.network_reduction_data)
        error("A and PTDF matrices have non-equivalent network reductions.")
    end
    ax_ref = make_ax_ref(get_arc_axis(A))

    if tol > eps()
        lodf_t = _buildlodf(A.data, PTDFm_data, linear_solver)
        return LODF(
            sparsify(lodf_t, tol),
            (get_arc_axis(A), get_arc_axis(A)),
            (ax_ref, ax_ref),
            subnetwork_axes,
            Ref(tol),
            A.network_reduction_data,
        )
    end
    return LODF(
        _buildlodf(A.data, PTDFm_data, linear_solver),
        (get_arc_axis(A), get_arc_axis(A)),
        (ax_ref, ax_ref),
        subnetwork_axes,
        Ref(tol),
        A.network_reduction_data,
    )
end

"""
    LODF(A::IncidenceMatrix, ABA::ABA_Matrix, BA::BA_Matrix; linear_solver::String = "KLU", tol::Float64 = eps())

Construct a Line Outage Distribution Factor (LODF) matrix from incidence, ABA, and BA matrices.
This constructor provides direct control over the underlying matrix computations and is most
efficient when the prerequisite matrices with factorization are already available.

# Arguments
- `A::IncidenceMatrix`: The incidence matrix containing bus-branch connectivity information
- `ABA::ABA_Matrix`: The bus susceptance matrix (A^T * B * A), preferably with KLU factorization
- `BA::BA_Matrix`: The branch susceptance weighted incidence matrix (B * A)

# Keyword Arguments
- `linear_solver::String = "KLU"`: 
        Linear solver algorithm for matrix computations. Currently only "KLU" is supported
- `tol::Float64 = eps()`: 
        Sparsification tolerance for dropping small matrix elements

# Returns
- `LODF`: The constructed LODF matrix structure with line outage sensitivity coefficients

# Mathematical Computation
This method computes LODF using the factorized form:
```
LODF = (A * ABA^(-1) * BA) / (1 - diag(A * ABA^(-1) * BA))
```
where:
- A is the incidence matrix
- ABA^(-1) uses the factorized form from the ABA matrix (requires `ABA.K` to be factorized)
- BA is the susceptance-weighted incidence matrix

# Requirements and Limitations
- **Factorization Required**: The ABA matrix should be pre-factorized (contains KLU factorization) for efficiency
- **Single Slack Bus**: This method does not support distributed slack bus configurations
- **Network Consistency**: All three input matrices must have equivalent network reduction states
- **Solver Limitation**: Currently only supports "KLU" linear solver

# Performance Advantages
- **Pre-factorization**: Leverages existing KLU factorization in ABA matrix for maximum efficiency
- **Direct Computation**: Avoids intermediate PTDF calculation, reducing computational steps
- **Memory Efficient**: Works directly with sparse matrix structures throughout computation
- **Numerical Stability**: Uses numerically stable KLU solver for matrix operations

# Error Handling
- Validates network reduction consistency across all three input matrices
- Raises error if matrices have mismatched reduction states
- Validates linear solver selection (currently only "KLU" supported)

# Usage Recommendations
- Use this constructor when you have pre-computed and factorized matrices available
- Ensure ABA matrix is factorized using `factorize(ABA)` or constructed with `factorize=true`
- For systems with distributed slack, use the PTDF-based constructor instead
- Most efficient option for repeated LODF computations on the same network topology
"""
function LODF(
    A::IncidenceMatrix,
    ABA::ABA_Matrix,
    BA::BA_Matrix;
    linear_solver::String = "KLU",
    tol::Float64 = eps(),
)
    if !(
        isequal(A.network_reduction_data, BA.network_reduction_data) &&
        isequal(BA.network_reduction_data, ABA.network_reduction_data)
    )
        error(
            "Mismatch in `NetworkReduction`, A, BA, and ABA matrices must be computed with the same network reduction.",
        )
    end
    validate_linear_solver(linear_solver)
    subnetwork_axes = make_arc_arc_subnetwork_axes(A)
    ax_ref = make_ax_ref(get_arc_axis(A))
    if tol > eps()
        lodf_t =
            _buildlodf(A.data, ABA.K, BA.data, Set(get_ref_bus_position(A)), linear_solver)
        return LODF(
            sparsify(lodf_t, tol),
            (get_arc_axis(A), get_arc_axis(A)),
            (ax_ref, ax_ref),
            subnetwork_axes,
            Ref(tol),
            A.network_reduction_data,
        )
    end
    return LODF(
        _buildlodf(A.data, ABA.K, BA.data, Set(get_ref_bus_position(A)), linear_solver),
        (get_arc_axis(A), get_arc_axis(A)),
        (ax_ref, ax_ref),
        subnetwork_axes,
        Ref(tol),
        A.network_reduction_data,
    )
end

############################################################
# auxiliary functions for getting data from LODF structure #
############################################################

# NOTE: the LODF matrix is saved as transposed!

function Base.getindex(A::LODF, selected_branch_name::String, outage_branch_name::String)
    multiplier_selected, arc_selected = get_branch_multiplier(A, selected_branch_name)
    multiplier_outage, arc_outage = get_branch_multiplier(A, outage_branch_name)
    i, j = to_index(A, arc_outage, arc_selected)
    return A.data[i, j] * multiplier_selected * multiplier_outage
end

function Base.getindex(A::LODF, selected_arc, outage_arc)
    i, j = to_index(A, outage_arc, selected_arc)
    return A.data[i, j]
end

function Base.getindex(
    A::LODF,
    selected_line_number::Union{Int, Colon},
    outage_line_number::Union{Int, Colon},
)
    return A.data[outage_line_number, selected_line_number]
end

"""
    get_lodf_data(lodf::LODF)

Extract the LODF matrix data in the standard orientation (non-transposed).

# Arguments
- `lodf::LODF`: The LODF structure from which to extract data

# Returns
- `AbstractArray{Float64, 2}`: The LODF matrix data with standard orientation
"""
function get_lodf_data(lodf::LODF)
    return transpose(lodf.data)
end

function get_arc_axis(lodf::LODF)
    return lodf.axes[1]
end

function get_tol(lodf::LODF)
    return lodf.tol
end
