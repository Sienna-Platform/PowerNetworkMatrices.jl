"""
Structure containing the Line Outage Distribution Factor (LODF) matrix and related power system data.

The LODF matrix contains sensitivity coefficients that quantify how the outage of one transmission
line affects the power flows on all other lines in the system. Each element ``\\mathrm{LODF}[i,j]``
represents the change in flow on line ``i`` when line ``j`` is taken out of service, normalized by
the pre-outage flow on line ``j``.

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
- **Matrix Form**: ``\\mathrm{LODF}[i,j] = \\partial f_i / \\partial P_j`` where ``f_i`` is flow on line ``i`` and ``P_j`` is the injection change due to the outage of line ``j``
- **Dimensions**: `(n_branches × n_branches)` for all transmission lines in the system
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

get_axes(M::LODF) = M.axes
get_lookup(M::LODF) = M.lookup
get_ref_bus(M::LODF) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::LODF) = [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::LODF) = M.network_reduction_data
get_arc_lookup(M::LODF) = M.lookup[1]
stores_transpose(::LODF) = true

# --- Demand-matrix short-circuit ---------------------------------------------
#
# The LODF computation builds a *diagonal* "demand" matrix `D = diag(m_V)`
# where `m_V[i] = 1 - PTDF·A[i, i]` (clamped to 1.0 at `LODF_ENTRY_TOLERANCE`
# to avoid divide-by-zero when an outage islands the line). The original
# code factored `D` and ran a triangular solve `D · X = ptdf_denominator`;
# that's a `factor + back-solve` over a diagonal, which collapses to
# element-wise row scaling. KLU's BTF short-circuits this internally so the
# overhead was modest; AA's libSparse and LAPACK's `getrf!`/`getrs!` do
# not, so the previous code was 3–5× slower on AA and order-of-magnitude
# slower on DENSE than necessary. Replace both with a direct row scaling.

function _build_lodf_demand(ptdf_denominator::AbstractMatrix{Float64}, linecount::Int)
    m_V = Vector{Float64}(undef, linecount)
    @inbounds for i in 1:linecount
        d = 1.0 - ptdf_denominator[i, i]
        m_V[i] = d < LODF_ENTRY_TOLERANCE ? 1.0 : d
    end
    return m_V
end

function _apply_lodf_demand!(M::AbstractMatrix{Float64}, m_V::Vector{Float64})
    IS.@assert_op size(M, 1) == length(m_V)
    IS.@assert_op size(M, 1) == size(M, 2)
    # `inv_dem .* M` mirrors what the triangular solve did internally —
    # one reciprocal per row, then a row-wise multiply. The broadcast
    # `M .*= inv_dem` scales each row `i` by `inv_dem[i]` because the
    # length-n vector broadcasts down the first dimension.
    inv_dem = 1.0 ./ m_V
    M .*= inv_dem
    M[SparseArrays.diagind(M)] .= -1.0
    return M
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
    ::KLUSolver,
)
    return _calculate_LODF_matrix_KLU(a, ptdf)
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
    ::DenseSolver,
)
    return _calculate_LODF_matrix_DENSE(a, ptdf)
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
    ::MKLPardisoSolver,
)
    _has_mkl_pardiso_ext() || error(_mkl_pardiso_install_error())
    return _calculate_LODF_matrix_MKLPardiso(a, ptdf)
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
    ::AppleAccelerateLUSolver,
)
    _has_apple_accelerate_backend() || error(_apple_accelerate_unavailable_error())
    return _calculate_LODF_matrix_AppleAccelerate(a, ptdf)
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    k::KLULinSolveCache{Float64},
    ba::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    ::KLUSolver,
)
    return _calculate_LODF_matrix_KLU(a, k, ba, ref_bus_positions)
end

function _buildlodf(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    k::KLULinSolveCache{Float64},
    ba::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
    ::LinearSolverType,
)
    return error("Only KLU solver is implemented for this LODF construction path.")
end

function _calculate_LODF_matrix_KLU(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    k::KLULinSolveCache{Float64},
    ba::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
)
    linecount = size(ba, 2)
    valid_ix = setdiff(1:size(a, 2), ref_bus_positions)
    a_t_valid = SparseArrays.SparseMatrixCSC(transpose(a))[valid_ix, :]
    first_ = zeros(size(a, 2), size(a, 1))
    solve_sparse!(k, a_t_valid, view(first_, valid_ix, :))
    ptdf_denominator = first_' * ba

    m_V = _build_lodf_demand(ptdf_denominator, linecount)
    _apply_lodf_demand!(ptdf_denominator, m_V)
    return ptdf_denominator
end

function _calculate_LODF_matrix_KLU(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    linecount = size(ptdf, 2)
    ptdf_denominator_t = a * ptdf
    m_V = _build_lodf_demand(ptdf_denominator_t, linecount)
    lodf_t = copy(ptdf_denominator_t)
    _apply_lodf_demand!(lodf_t, m_V)
    return lodf_t
end

function _calculate_LODF_matrix_DENSE(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    linecount = size(ptdf, 2)
    ptdf_denominator_t = a * ptdf
    m_V = _build_lodf_demand(ptdf_denominator_t, linecount)
    _apply_lodf_demand!(ptdf_denominator_t, m_V)
    return ptdf_denominator_t
end

# _pardiso_sequential_LODF!, _pardiso_single_LODF!, _calculate_LODF_matrix_MKLPardiso
# are defined in ext/MKLPardisoExt.jl when the Pardiso package is loaded

@static if Sys.isapple()
    """
    Function for internal use only.

    Computes the LODF matrix using the internal Apple Accelerate backend
    (`AccelerateWrapper`). Available only on macOS. Shape mirrors
    `_calculate_LODF_matrix_KLU(a, ptdf)` exactly: factor the diagonal "demand"
    matrix ``\\mathrm{diag}(1 - A \\, \\mathrm{PTDF})`` and solve in place against
    ``A \\, \\mathrm{PTDF}``.

    # Arguments
    - `a::SparseArrays.SparseMatrixCSC{Int8, Int}`: Incidence Matrix
    - `ptdf::Matrix{Float64}`: PTDF matrix
    """
    function _calculate_LODF_matrix_AppleAccelerate(
        a::SparseArrays.SparseMatrixCSC{Int8, Int},
        ptdf::Matrix{Float64},
    )
        linecount = size(ptdf, 2)
        ptdf_denominator_t = a * ptdf
        m_V = _build_lodf_demand(ptdf_denominator_t, linecount)
        _apply_lodf_demand!(ptdf_denominator_t, m_V)
        return ptdf_denominator_t
    end
end

# Numeric/default tol: original PTDF-based route, unchanged behavior.
function _lodf_from_system(
    tol::Float64,
    A::IncidenceMatrix,
    BA::BA_Matrix,
    Ymatrix::Ybus,
    linear_solver::String,
)
    # Keep the intermediate PTDF dense (tol = eps()); the from-PTDF LODF needs an
    # unsparsified PTDF for accuracy, and only the LODF itself is sparsified.
    ptdf = PTDF(A, BA; linear_solver = linear_solver, tol = eps())
    return LODF(A, ptdf; linear_solver = linear_solver, tol = tol)
end

# AutoTolerance: build a factorized ABA so conditioning is available, then use
# the KLU-only ABA/BA constructor.
function _lodf_from_system(
    spec::AutoTolerance,
    A::IncidenceMatrix,
    BA::BA_Matrix,
    Ymatrix::Ybus,
    ::String,
)
    ABA = ABA_Matrix(Ymatrix; factorize = true)
    return LODF(A, ABA, BA; tol = spec)
end

"""
    LODF(sys::PSY.System; linear_solver::String = _default_linear_solver(), tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE, network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct a Line Outage Distribution Factor (LODF) matrix from a PowerSystems.System by computing
the sensitivity of line flows to single line outages. This is the primary constructor for LODF
analysis starting from system data.

# Arguments
- `sys::PSY.System`: The power system from which to construct the LODF matrix

# Keyword Arguments
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
- **"KLU"**: Sparse LU factorization (default off Apple hardware, recommended for most cases)
- **"AppleAccelerateLU"**: Apple Accelerate sparse LU (default on macOS 15.5+ Apple hardware)
- **"Dense"**: Dense matrix operations (faster for small systems)
- **"MKLPardiso"**: Intel MKL Pardiso solver (requires MKL, best for very large systems)

# Mathematical Foundation
With ``H = A \\, \\mathrm{PTDF}``, the sensitivity of monitored line ``\\ell`` to the outage
of line ``e`` is
```math
\\mathrm{LODF}[\\ell, e] = \\frac{H[\\ell, e]}{1 - H[e, e]}
```
where ``A`` is the incidence matrix and ``\\mathrm{PTDF}`` is the power transfer distribution factor matrix.

# Notes
- Sparsification with `tol > eps()` can significantly reduce memory usage
- Network reductions can improve computational efficiency for large systems
- Results are valid under DC power flow assumptions (linear approximation)
- Diagonal elements are always -1.0 representing complete flow loss on outaged lines
- For very large systems, consider using "MKLPardiso" solver with appropriate chunk size
"""
function LODF(
    sys::PSY.System;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    Ymatrix = Ybus(sys; network_reductions = network_reductions, kwargs...)
    A = IncidenceMatrix(Ymatrix)
    BA = BA_Matrix(Ymatrix)
    # Numeric tol keeps the PTDF route (unchanged); an AutoTolerance needs ABA for
    # conditioning, so route it through the factorized-ABA constructor.
    return _lodf_from_system(tol, A, BA, Ymatrix, linear_solver)
end

"""
    LODF(A::IncidenceMatrix, PTDFm::PTDF; linear_solver::String = _default_linear_solver(), tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE)

Construct a Line Outage Distribution Factor (LODF) matrix from existing incidence and PTDF matrices.
This constructor is more efficient when the prerequisite matrices are already available.

# Arguments
- `A::IncidenceMatrix`: The incidence matrix containing bus-branch connectivity information
- `PTDFm::PTDF`: The power transfer distribution factor matrix (should be non-sparsified for accuracy)

# Keyword Arguments
- `linear_solver::String = _default_linear_solver()`:
        Linear solver algorithm for matrix computations. Options: "KLU", "AppleAccelerateLU", "Dense", "MKLPardiso"
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance for the LODF matrix (not applied to input PTDF)

# Returns
- `LODF`: The constructed LODF matrix structure with line outage sensitivity coefficients

# Mathematical Computation
With ``H = A \\, \\mathrm{PTDF}``, the sensitivity of monitored line ``\\ell`` to the outage
of line ``e`` is
```math
\\mathrm{LODF}[\\ell, e] = \\frac{H[\\ell, e]}{1 - H[e, e]}
```
where:
- ``A`` is the incidence matrix (the [`IncidenceMatrix`](@ref)) representing bus-branch connectivity
- ``\\mathrm{PTDF}`` contains power transfer distribution factors
- The denominator ``1 - H[e,e]`` accounts for the outaged line's own flow

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
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    solver = resolve_linear_solver(linear_solver)
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

    tol_value = _dense_tol(tol)
    if tol_value > eps()
        lodf_t = _buildlodf(A.data, PTDFm_data, solver)
        return LODF(
            _sparsify_lodf(lodf_t, tol_value),
            (get_arc_axis(A), get_arc_axis(A)),
            (ax_ref, ax_ref),
            subnetwork_axes,
            Ref(tol_value),
            A.network_reduction_data,
        )
    end
    return LODF(
        _buildlodf(A.data, PTDFm_data, solver),
        (get_arc_axis(A), get_arc_axis(A)),
        (ax_ref, ax_ref),
        subnetwork_axes,
        Ref(tol_value),
        A.network_reduction_data,
    )
end

"""
    LODF(A::IncidenceMatrix, ABA::ABA_Matrix, BA::BA_Matrix; linear_solver::String = "KLU", tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE)

Construct a Line Outage Distribution Factor (LODF) matrix from incidence, ABA, and BA matrices.
This constructor provides direct control over the underlying matrix computations and is most
efficient when the prerequisite matrices with factorization are already available.

# Arguments
- `A::IncidenceMatrix`: The incidence matrix containing bus-branch connectivity information
- `ABA::ABA_Matrix`: The bus susceptance matrix ``A^\\top B A``, preferably with KLU factorization
- `BA::BA_Matrix`: The branch susceptance weighted incidence matrix ``B A``

# Keyword Arguments
- `linear_solver::String = "KLU"`:
        This constructor is intentionally KLU-only because `ABA.K` is always a
        KLU factorization. The keyword is kept for API consistency; passing any
        other value will error.
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance for dropping small matrix elements

# Returns
- `LODF`: The constructed LODF matrix structure with line outage sensitivity coefficients

# Mathematical Computation
This method computes LODF using the factorized form ``H = A\\, \\mathrm{ABA}^{-1} \\mathrm{BA}``,
```math
\\mathrm{LODF}[\\ell, e] = \\frac{H[\\ell, e]}{1 - H[e, e]}
```
where:
- ``A`` is the incidence matrix (the [`IncidenceMatrix`](@ref))
- ``\\mathrm{ABA}^{-1}`` uses the factorized form from the [`ABA_Matrix`](@ref) (requires `ABA.K` to be factorized)
- ``\\mathrm{BA}`` is the susceptance-weighted incidence matrix (the [`BA_Matrix`](@ref))

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
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    # NOTE: ABA.K is always a KLU factorization, so this constructor is
    # KLU-only regardless of the `linear_solver` argument. The kwarg is kept
    # for API consistency; passing anything other than "KLU" will error in
    # `_buildlodf`.
    if !(
        isequal(A.network_reduction_data, BA.network_reduction_data) &&
        isequal(BA.network_reduction_data, ABA.network_reduction_data)
    )
        error(
            "Mismatch in `NetworkReduction`, A, BA, and ABA matrices must be computed with the same network reduction.",
        )
    end
    solver = resolve_linear_solver(linear_solver)
    subnetwork_axes = make_arc_arc_subnetwork_axes(A)
    ax_ref = make_ax_ref(get_arc_axis(A))
    lodf_t = _buildlodf(A.data, ABA.K, BA.data, Set(get_ref_bus_position(A)), solver)
    tol_value = _dense_tol(tol)
    if tol_value > eps()
        lodf_t = _sparsify_lodf(lodf_t, tol_value)
    end
    return LODF(
        lodf_t,
        (get_arc_axis(A), get_arc_axis(A)),
        (ax_ref, ax_ref),
        subnetwork_axes,
        Ref(tol_value),
        A.network_reduction_data,
    )
end

# The LODF diagonal is structurally -1.0 (complete flow loss on the outaged arc).
# A tol >= 1.0 (large meshed network where scale = max|LODF| > 1) would let
# `droptol!` remove it. Rather than sparsify and patch each dropped diagonal back
# (N CSC insertions, each O(nnz)), zero the dense diagonal so `droptol!` only
# touches off-diagonals, then re-add the structural -I in a single
# sparse-plus-UniformScaling merge.
function _sparsify_lodf(lodf_t::Matrix{Float64}, tol::Float64)
    lodf_t[LinearAlgebra.diagind(lodf_t)] .= 0.0
    return sparsify(lodf_t, tol) - LinearAlgebra.I
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
