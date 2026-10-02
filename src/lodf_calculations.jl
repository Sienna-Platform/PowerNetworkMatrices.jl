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
- `tol::Float64`:
        Tolerance threshold used for matrix sparsification (elements below this value are dropped)
- `branch_catalog::BranchCatalog`:
        Container for network reduction information applied during matrix construction

# Notes
- Stored transposed; `lodf[monitored, outaged]` is the flow change on the monitored arc per unit pre-outage flow on the outaged arc.
- The diagonal is always -1.0.
- Elements below `tol` are dropped when sparsified.
- Valid under DC power flow assumptions.
"""
struct LODF{Ax, L <: NTuple{2, Dict}, M <: AbstractArray{Float64, 2}} <:
       PowerNetworkMatrix{Float64}
    data::M
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    tol::Float64
    branch_catalog::BranchCatalog
end

# Arc-indexed: no get_bus_lookup(M::LODF) exists, so this throws MethodError on any call.
# Pre-existing; kept here so LODF does not silently inherit the bus-indexed generic method.
get_ref_bus_position(M::LODF) = [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
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
    ::LinearSolverType,
)
    return _calculate_LODF_matrix(a, ptdf)
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

function _buildlodf(
    ::SparseArrays.SparseMatrixCSC{Int8, Int},
    ::AAFactorCache,
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
    ::Set{Int},
    ::LinearSolverType,
)
    return error(
        "LODF(A, ABA, BA) needs an ABA factorized with KLU; this ABA was factorized with " *
        "AppleAccelerate. Build it with `ABA_Matrix(...; factorize = true, " *
        "linear_solver = \"KLU\")`, or use `LODF(sys; linear_solver = \"AppleAccelerateLU\")`.",
    )
end

function _buildlodf(
    ::SparseArrays.SparseMatrixCSC{Int8, Int},
    ::Nothing,
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
    ::Set{Int},
    ::LinearSolverType,
)
    return error(
        "LODF(A, ABA, BA) needs a factorized ABA; build it with " *
        "`ABA_Matrix(...; factorize = true)`.",
    )
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

function _calculate_LODF_matrix(
    a::SparseArrays.SparseMatrixCSC{Int8, Int},
    ptdf::Matrix{Float64},
)
    ptdf_denominator_t = a * ptdf
    m_V = _build_lodf_demand(ptdf_denominator_t, size(ptdf, 2))
    _apply_lodf_demand!(ptdf_denominator_t, m_V)
    return ptdf_denominator_t
end

# Numeric tol: the PTDF-based route.
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
        Solver for the intermediate PTDF when `tol` is a `Float64`. An `AutoTolerance` builds
        from a KLU-factorized ABA instead, so the solver is only checked for validity there
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance. A `Float64` drops elements below it; the default
        [`AutoTolerance`](@ref) leaves this dense matrix exact.
- `network_reductions::Vector{NetworkReduction} = NetworkReduction[]`:
        Vector of network reduction algorithms to apply before matrix construction
- `include_constant_impedance_loads::Bool=true`:
        Whether to include constant impedance loads as shunt admittances in the network model
- Additional keyword arguments are passed to the underlying matrix constructors

# Returns
- `LODF`: The constructed LODF matrix structure containing:
  - Line-to-line outage sensitivity coefficients
  - Network topology information and branch identifiers
  - Sparsification tolerance and computational metadata

# Notes
- Sparsification with `tol > eps()` can significantly reduce memory usage
- Network reductions can improve computational efficiency for large systems
- Results are valid under DC power flow assumptions (linear approximation)
- Diagonal elements are always -1.0 representing complete flow loss on outaged lines
"""
function LODF(
    sys::PSY.System;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    resolve_linear_solver(linear_solver)
    Ymatrix = Ybus(sys; network_reductions = network_reductions, kwargs...)
    A = IncidenceMatrix(Ymatrix)
    BA = BA_Matrix(Ymatrix)
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
        Checked against the supported solvers but otherwise unused: this route only rescales
        the PTDF, so no factorization is involved
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance for the LODF matrix (not applied to input PTDF). A `Float64`
        drops elements below it; the default [`AutoTolerance`](@ref) leaves the LODF exact.

# Returns
- `LODF`: The constructed LODF matrix structure with line outage sensitivity coefficients

# Notes
- The input PTDF should be non-sparsified (default `tol`); a sparsified PTDF triggers a warning and is densified.
- `tol` only sparsifies the LODF, not the input PTDF.
- `A` and `PTDFm` must share the same network reductions.
- The diagonal is set to -1.0.
"""
function LODF(
    A::IncidenceMatrix,
    PTDFm::PTDF;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    solver = resolve_linear_solver(linear_solver)
    subnetwork_axes = make_arc_arc_subnetwork_axes(A)

    if get_tol(PTDFm) > 1e-15
        warn_msg = string(
            "The argument `tol` in the PTDF matrix was set to a value different than the default one.\n",
            "The resulting LODF can include unexpected rounding errors.\n",
        )
        @warn(warn_msg)
        PTDFm_data = Matrix(PTDFm.data)
    else
        PTDFm_data = PTDFm.data
    end

    if !isequal(get_network_reduction_data(A), get_network_reduction_data(PTDFm))
        error("A and PTDF matrices have non-equivalent network reductions.")
    end
    ax_ref = make_ax_ref(get_arc_axis(A))

    tol_value = _dense_tol(tol)
    lodf_t = _buildlodf(A.data, PTDFm_data, solver)
    if tol_value > eps()
        lodf_t = _sparsify_lodf(lodf_t, tol_value)
    end
    return LODF(
        lodf_t,
        (get_arc_axis(A), get_arc_axis(A)),
        (ax_ref, ax_ref),
        subnetwork_axes,
        tol_value,
        get_branch_catalog(A),
    )
end

"""
    LODF(A::IncidenceMatrix, ABA::ABA_Matrix, BA::BA_Matrix; linear_solver::String = "KLU", tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE)

Construct a Line Outage Distribution Factor (LODF) matrix from incidence, ABA, and BA matrices.
This constructor provides direct control over the underlying matrix computations and is most
efficient when the prerequisite matrices with factorization are already available.

# Arguments
- `A::IncidenceMatrix`: The incidence matrix containing bus-branch connectivity information
- `ABA::ABA_Matrix`: The bus susceptance matrix (A^T * B * A), preferably with KLU factorization
- `BA::BA_Matrix`: The branch susceptance weighted incidence matrix (B * A)

# Keyword Arguments
- `linear_solver::String = "KLU"`:
        This constructor needs `ABA.K` to be a KLU factorization
        (`ABA_Matrix(...; factorize = true, linear_solver = "KLU")`); an
        AppleAccelerate-factorized ABA raises an error.
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Sparsification tolerance. A `Float64` drops elements below it; the default
        [`AutoTolerance`](@ref) leaves this dense matrix exact.

# Returns
- `LODF`: The constructed LODF matrix structure with line outage sensitivity coefficients

# Notes
- `ABA` must be KLU-factorized.
- Single slack bus only; use the PTDF-based constructor for distributed slack.
- `A`, `BA`, and `ABA` must share the same network reductions.
"""
function LODF(
    A::IncidenceMatrix,
    ABA::ABA_Matrix,
    BA::BA_Matrix;
    linear_solver::String = "KLU",
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
    # NOTE: this constructor needs `ABA.K` to be a KLU factorization; an
    # AppleAccelerate-factorized ABA raises an error in `_buildlodf`
    # regardless of the `linear_solver` argument passed here.
    if !(
        isequal(get_network_reduction_data(A), get_network_reduction_data(BA)) &&
        isequal(get_network_reduction_data(BA), get_network_reduction_data(ABA))
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
        tol_value,
        get_branch_catalog(A),
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
