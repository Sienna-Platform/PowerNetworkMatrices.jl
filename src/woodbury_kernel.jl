"""
Shared Woodbury matrix identity kernel for computing post-modification
network sensitivity factors. Used by both VirtualPTDF and VirtualMODF.

Implements van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹
"""

"""
    _woodbury_tol_scale(W_mat, power) -> Float64

Relative-plus-absolute scale tolerance for detecting near-singularity of W.

Returns `max(MODF_ISLANDING_TOLERANCE, MODF_ISLANDING_REL_TOL * ‖W‖₁^power)`.
The absolute floor dominates when ‖W‖₁ is small or zero (so a fully-null W
still triggers islanding detection). The relative term dominates on
large-magnitude W where floating-point noise on pivots/determinants grows
with ‖W‖₁.

The `power` argument matches the "units" of what the caller compares:
  - `power = 1` for pivot/entry comparisons (Val{1}, Val{M>2} LU pivots)
  - `power = 2` for determinant comparisons (Val{2})
"""
@inline function _woodbury_tol_scale(W_mat::Matrix{Float64}, power::Int)
    n = LinearAlgebra.opnorm(W_mat, 1)
    rel_scale = MODF_ISLANDING_REL_TOL * n^power
    return max(MODF_ISLANDING_TOLERANCE, rel_scale)
end

"""
    _invert_woodbury_W(W_mat, ::Val{M}) -> (W_inv::Matrix{Float64}, is_islanding::Bool)

Invert the M×M Woodbury W matrix. Dispatches on `Val{M}` so the compiler
can specialize each case. Analytical formulas for M=1 and M=2 avoid LU
factorization overhead. Falls back to LU for M > 2. Near-singularity is
detected with a relative-plus-absolute tolerance (see `_woodbury_tol_scale`);
the `pinv` fallback uses the absolute floor for its `atol` to stay robust
regardless of W's scale.
"""
function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    ::Val{1},
)::Tuple{Matrix{Float64}, Bool}
    w = W_mat[1, 1]
    tol = _woodbury_tol_scale(W_mat, 1)
    is_island = abs(w) < tol
    W_inv = Matrix{Float64}(undef, 1, 1)
    if is_island
        W_inv[1, 1] = 0.0
    else
        W_inv[1, 1] = 1.0 / w
    end
    return W_inv, is_island
end

function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    ::Val{2},
)::Tuple{Matrix{Float64}, Bool}
    a, b, c, d = W_mat[1, 1], W_mat[1, 2], W_mat[2, 1], W_mat[2, 2]
    det_W = a * d - b * c
    # det has units of ‖W‖₁²; relative scale must too, so power=2.
    tol = _woodbury_tol_scale(W_mat, 2)
    is_island = abs(det_W) < tol
    if is_island
        W_inv = LinearAlgebra.pinv(W_mat; atol = MODF_ISLANDING_TOLERANCE)
    else
        inv_det = 1.0 / det_W
        W_inv = Matrix{Float64}(undef, 2, 2)
        W_inv[1, 1] = d * inv_det
        W_inv[1, 2] = -b * inv_det
        W_inv[2, 1] = -c * inv_det
        W_inv[2, 2] = a * inv_det
    end
    return W_inv, is_island
end

function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    ::Val{M},
)::Tuple{Matrix{Float64}, Bool} where {M}
    W_lu = LinearAlgebra.lu(W_mat; check = false)
    tol = _woodbury_tol_scale(W_mat, 1)
    is_island = any(i -> abs(W_lu.U[i, i]) < tol, 1:M)
    W_inv =
        if is_island
            LinearAlgebra.pinv(W_mat; atol = MODF_ISLANDING_TOLERANCE)
        else
            LinearAlgebra.inv(W_lu)
        end
    return W_inv, is_island
end

# --- Accessor functions for Woodbury kernel ---
# VirtualPTDF accessors (VirtualMODF accessors defined in virtual_modf_calculations.jl)

_get_K(m::VirtualPTDF) = m.K
_get_BA(m::VirtualPTDF) = m.BA
_get_arc_susceptances(m::VirtualPTDF) = m.arc_susceptances
_get_valid_ix(m::VirtualPTDF) = m.valid_ix
_get_temp_data(m::VirtualPTDF) = m.temp_data
_get_work_ba_col(m::VirtualPTDF) = m.work_ba_col

"""
    _compute_woodbury_factors(mat, modifications) -> WoodburyFactors

Compute the Woodbury correction factors for a set of arc modifications.
Implements van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹U (A⁻¹ + U⊤B_r⁻¹U)⁻¹ U⊤B_r⁻¹

where U = [ν_{e1} ... ν_{eM}] and A = diag(Δb₁, ..., Δb_M).

The expensive part (M KLU solves + M×M factorization) is shared
across all monitored arcs for a given modification set.

!!! warning
    Not thread-safe. Mutates scratch vectors in `mat`. Do not call
    concurrently on the same VirtualPTDF/VirtualMODF instance.
"""
function _compute_woodbury_factors(
    mat::PowerNetworkMatrix,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    M = length(modifications)
    K = _get_K(mat)
    BA = _get_BA(mat)
    arc_sus = _get_arc_susceptances(mat)
    valid_ix = _get_valid_ix(mat)
    temp_data = _get_temp_data(mat)
    work_ba_col = _get_work_ba_col(mat)
    n_bus = length(temp_data)

    arc_indices = Vector{Int}(undef, M)
    delta_b_vec = Vector{Float64}(undef, M)
    for (j, mod) in enumerate(modifications)
        arc_indices[j] = mod.arc_index
        delta_b_vec[j] = mod.delta_b
    end

    # Compute Z[:,j] = B⁻¹ν_j for each modified arc
    Z = Matrix{Float64}(undef, n_bus, M)

    for (j, mod) in enumerate(modifications)
        e = mod.arc_index
        b_e = arc_sus[e]

        @inbounds for i in eachindex(valid_ix)
            work_ba_col[i] = BA[valid_ix[i], e]
        end
        lin_solve = _solve_factorization(K, work_ba_col)

        fill!(view(Z, :, j), 0.0)
        @inbounds for i in eachindex(valid_ix)
            Z[valid_ix[i], j] = lin_solve[i] / b_e
        end
    end

    # K_mat[i,j] = ν_i⊤ B⁻¹ ν_j
    # Use BA[:,arc]/b instead of A[arc,:] for consistent sign convention (issue #278).
    # Iterate sparse BA columns (typically 2 nonzeros per arc).
    ba_nzv = SparseArrays.nonzeros(BA)
    ba_rv = SparseArrays.rowvals(BA)
    K_mat = zeros(M, M)
    for i in 1:M
        e_i = arc_indices[i]
        b_i = arc_sus[e_i]
        for j in 1:M
            val = 0.0
            @inbounds for nz_idx in nzrange(BA, e_i)
                row = ba_rv[nz_idx]
                val += (ba_nzv[nz_idx] / b_i) * Z[row, j]
            end
            K_mat[i, j] = val
        end
    end

    # W = diag(1/Δb) + K_mat
    W_mat = LinearAlgebra.diagm(1.0 ./ delta_b_vec) + K_mat

    # Pre-invert W (Val dispatch lets the compiler specialize M=1,2)
    W_inv, is_island = _invert_woodbury_W(W_mat, Val(M))

    if is_island
        @debug "Contingency islands the network; using pinv-based Woodbury correction."
    end

    return WoodburyFactors(Z, W_inv, arc_indices, delta_b_vec, is_island)
end

"""
    _apply_woodbury_correction(mat, monitored_idx, wf) -> Vector{Float64}

Compute the post-modification PTDF row for a monitored arc using
precomputed Woodbury factors.

Post-modification PTDF: `PTDF_m[mon,:] = b_mon_post · ν_mon⊤ · B_m⁻¹`
Computed as: `b_mon_post · (z_m - Z · W⁻¹ · (ν_mon⊤ · Z))`

!!! warning
    Not thread-safe. Mutates scratch vectors in `mat`. Do not call
    concurrently on the same VirtualPTDF/VirtualMODF instance.
"""
function _apply_woodbury_correction(
    mat::PowerNetworkMatrix,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    K = _get_K(mat)
    BA = _get_BA(mat)
    arc_sus = _get_arc_susceptances(mat)
    valid_ix = _get_valid_ix(mat)
    temp_data = _get_temp_data(mat)
    work_ba_col = _get_work_ba_col(mat)
    n_bus = length(temp_data)

    M = length(wf.arc_indices)

    # Effective susceptance of monitored arc after modifications
    b_mon = arc_sus[monitored_idx]
    for (j, idx) in enumerate(wf.arc_indices)
        if idx == monitored_idx
            b_mon += wf.delta_b[j]
        end
    end
    if abs(b_mon) < eps()
        return zeros(n_bus)
    end

    # z_m = B⁻¹ν_m / b_mon_pre via KLU solve on BA column
    b_mon_pre = arc_sus[monitored_idx]
    @inbounds for i in eachindex(valid_ix)
        work_ba_col[i] = BA[valid_ix[i], monitored_idx]
    end
    lin_solve = _solve_factorization(K, work_ba_col)

    # Build z_m into temp_data
    fill!(temp_data, 0.0)
    @inbounds for i in eachindex(valid_ix)
        temp_data[valid_ix[i]] = lin_solve[i] / b_mon_pre
    end

    # ν_m⊤ · Z  (1 × M vector)
    # Use BA[:,m]/b instead of A[m,:] for consistent sign convention (issue #278).
    ba_nzv = SparseArrays.nonzeros(BA)
    ba_rv = SparseArrays.rowvals(BA)
    zm_Z = zeros(M)
    @inbounds for nz_idx in nzrange(BA, monitored_idx)
        row = ba_rv[nz_idx]
        coeff = ba_nzv[nz_idx] / b_mon_pre
        for j in 1:M
            zm_Z[j] += coeff * wf.Z[row, j]
        end
    end

    # Woodbury correction: temp_data -= Z · (W⁻¹ · zm_Z)
    correction_coeff = wf.W_inv * zm_Z
    LinearAlgebra.mul!(temp_data, wf.Z, correction_coeff, -1.0, 1.0)

    # Post-modification PTDF row = b_mon_post · (z_m - correction)
    temp_data .*= b_mon
    return copy(temp_data)
end

# -----------------------------------------------------------------------------
# ABA-domain Woodbury (angle solve)
# -----------------------------------------------------------------------------
#
# For a DC contingency that modifies arcs `j ∈ 1..M` with susceptance deltas
# `Δb_j`, the post-contingency ABA is
#
#     ABAₖ = ABA + Uᵀ · diag(Δb) · U
#
# where `U` is the reduced arc incidence matrix restricted to the modified arcs
# (columns ±1 on from/to positions, ref bus rows dropped). By the Woodbury
# identity,
#
#     θₖ = θ_base  -  Z · W⁻¹ · (Uᵀ · θ_base)
#     Z  = ABA⁻¹ · U
#     W  = diag(1/Δb) + Uᵀ · ABA⁻¹ · U   (M × M, small)
#
# Unlike `_compute_woodbury_factors` above (which is PTDF-flavored and mutates
# matrix-owned scratch), the functions below take the KLU factorization and
# scratch buffers as explicit arguments. Safe for concurrent use when each
# caller owns its own (factorization, scratch) tuple.

"""
    compute_aba_woodbury_factors(K, U, delta_b, arc_indices, Z_buf) -> WoodburyFactors

Compute Woodbury correction factors for a DC (ABA-domain) contingency.

# Arguments
- `K::KLU.KLUFactorization{Float64, Int}` — factorization of the base ABA matrix.
- `U::SparseMatrixCSC{Float64, Int}` — reduced arc incidence for modified arcs,
  shape `n_valid × M`. Produced by [`compute_aba_delta`](@ref).
- `delta_b::Vector{Float64}` — susceptance deltas, length `M`.
- `arc_indices::Vector{Int}` — arc indices for the modified arcs, length `M`.
- `Z_buf::Matrix{Float64}` — pre-allocated buffer of shape `n_valid × M`. On
  return holds `Z = K \\\\ U` and is referenced by `WoodburyFactors.Z` (no
  copy). Caller must NOT mutate or reuse `Z_buf` while `wf` is live.

# Thread safety
Safe for concurrent use as long as `K` is not mutated, `U` / `delta_b` /
`arc_indices` are not written elsewhere, and each task owns its `Z_buf`.
"""
function compute_aba_woodbury_factors(
    K::KLU.KLUFactorization{Float64, Int},
    U::SparseArrays.SparseMatrixCSC{Float64, Int},
    delta_b::Vector{Float64},
    arc_indices::Vector{Int},
    Z_buf::Matrix{Float64},
)::WoodburyFactors
    n_valid, M = size(U)
    @assert size(Z_buf) == (n_valid, M) "Z_buf shape mismatch: expected $((n_valid, M)), got $(size(Z_buf))"
    @assert length(delta_b) == M
    @assert length(arc_indices) == M

    fill!(Z_buf, 0.0)
    U_rv = SparseArrays.rowvals(U)
    U_nz = SparseArrays.nonzeros(U)
    @inbounds for j in 1:M
        for nz in SparseArrays.nzrange(U, j)
            Z_buf[U_rv[nz], j] = U_nz[nz]
        end
    end
    LinearAlgebra.ldiv!(K, Z_buf)

    K_mat = zeros(M, M)
    @inbounds for i in 1:M
        for nz in SparseArrays.nzrange(U, i)
            row = U_rv[nz]
            coeff = U_nz[nz]
            for j in 1:M
                K_mat[i, j] += coeff * Z_buf[row, j]
            end
        end
    end
    W_mat = Matrix(LinearAlgebra.Diagonal(1.0 ./ delta_b)) + K_mat
    W_inv, is_island = _invert_woodbury_W(W_mat, Val(M))
    if is_island
        @debug "ABA Woodbury: contingency islands the network; using pinv-based correction."
    end
    # WoodburyFactors.Z aliases Z_buf (see docstring); no copy.
    return WoodburyFactors(Z_buf, W_inv, arc_indices, delta_b, is_island)
end

"""
    apply_aba_woodbury_correction!(θ, U, wf, scratch_MxT) -> θ

In-place post-contingency angle correction via Woodbury. `θ` enters as the base
solution `ABA \\\\ P` (n_valid × n_ts); on return it holds the post-contingency
angles `θₖ`. Handles the multi-RHS case natively — each column of `θ` is one
timestep, and all timesteps are corrected in a single matmul pass.

# Arguments
- `θ::Matrix{Float64}` — `n_valid × n_ts`. In: base angles. Out: post-contingency angles.
- `U::SparseMatrixCSC{Float64, Int}` — reduced arc incidence for the modified arcs.
- `wf::WoodburyFactors` — precomputed factors for this contingency.
- `scratch_MxT::Matrix{Float64}` — pre-allocated `M × n_ts` scratch buffer.
  Mutated. Caller owns it.

# Thread safety
Safe for concurrent use: only writes `θ` and `scratch_MxT`, both caller-owned.
"""
function apply_aba_woodbury_correction!(
    θ::Matrix{Float64},
    U::SparseArrays.SparseMatrixCSC{Float64, Int},
    wf::WoodburyFactors,
    scratch_MxT::Matrix{Float64},
)
    n_valid, n_ts = size(θ)
    M = length(wf.delta_b)
    @assert size(scratch_MxT) == (M, n_ts) "scratch_MxT shape mismatch"
    @assert size(U) == (n_valid, M)
    @assert size(wf.Z) == (n_valid, M)

    # scratch := Uᵀ · θ   (M × n_ts)
    LinearAlgebra.mul!(scratch_MxT, transpose(U), θ)
    # correction := W⁻¹ · scratch  (in place, via temp product since W_inv is dense M×M)
    # Reuse scratch by doing scratch_new = W_inv * scratch. Allocate a tiny temp:
    correction = wf.W_inv * scratch_MxT
    # θ .-= Z * correction
    LinearAlgebra.mul!(θ, wf.Z, correction, -1.0, 1.0)
    return θ
end
