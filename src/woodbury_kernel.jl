"""
Shared Woodbury matrix identity kernel for computing post-modification
network sensitivity factors. Used by both VirtualPTDF and VirtualMODF.

Implements van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹
"""

# --- Accessor functions for Woodbury kernel ---
# VirtualPTDF accessors (VirtualMODF accessors defined in virtual_modf_calculations.jl)

_get_K(m::VirtualPTDF) = m.K
_get_BA(m::VirtualPTDF) = m.BA
_get_A(m::VirtualPTDF) = m.A
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
    modifications::Vector{ArcModification},
)::WoodburyFactors
    M = length(modifications)
    K = _get_K(mat)
    BA = _get_BA(mat)
    A = _get_A(mat)
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

    # K_mat[i,j] = ν_i⊤ B⁻¹ ν_j — compute only needed rows via sparse dot products
    # instead of full A * Z (avoids allocating n_arcs × M dense matrix)
    K_mat = zeros(M, M)
    for j in 1:M, i in 1:M
        K_mat[i, j] = dot(view(A, arc_indices[i], :), view(Z, :, j))
    end

    # W = diag(1/Δb) + K_mat
    W_mat = LinearAlgebra.diagm(1.0 ./ delta_b_vec) + K_mat

    # Pre-invert W
    W_inv, is_island = _invert_woodbury_W(W_mat, M)

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
    A = _get_A(mat)
    arc_sus = _get_arc_susceptances(mat)
    valid_ix = _get_valid_ix(mat)
    temp_data = _get_temp_data(mat)
    work_ba_col = _get_work_ba_col(mat)
    n_bus = length(temp_data)

    if wf.is_islanding
        @warn "Contingency islands the network. Returning zeros."
        return zeros(n_bus)
    end

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
    zm_Z = zeros(M)
    for j in 1:M
        zm_Z[j] = dot(view(A, monitored_idx, :), view(wf.Z, :, j))
    end

    # Woodbury correction: temp_data -= Z · (W⁻¹ · zm_Z)
    correction_coeff = wf.W_inv * zm_Z
    LinearAlgebra.mul!(temp_data, wf.Z, correction_coeff, -1.0, 1.0)

    # Post-modification PTDF row = b_mon_post · (z_m - correction)
    temp_data .*= b_mon
    return copy(temp_data)
end
