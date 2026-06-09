"""
Shared Woodbury matrix identity kernel for computing post-modification
network sensitivity factors. Used by both VirtualPTDF and VirtualMODF.

Implements van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹
"""

"""
    _invert_woodbury_W(W_mat, ::Val{M}) -> (W_inv::Matrix{Float64}, is_islanding::Bool)

Invert the M×M Woodbury W matrix. Dispatches on `Val{M}` so the compiler
can specialize each case. Analytical formulas for M=1 and M=2 avoid LU
factorization overhead. Falls back to LU for M > 2.
"""
function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    ::Val{1},
)::Tuple{Matrix{Float64}, Bool}
    w = W_mat[1, 1]
    is_island = abs(w) < MODF_ISLANDING_TOLERANCE
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
    is_island = abs(det_W) < MODF_ISLANDING_TOLERANCE
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
    is_island = any(i -> abs(W_lu.U[i, i]) < MODF_ISLANDING_TOLERANCE, 1:M)
    W_inv =
        if is_island
            LinearAlgebra.pinv(W_mat; atol = MODF_ISLANDING_TOLERANCE)
        else
            LinearAlgebra.inv(W_lu)
        end
    return W_inv, is_island
end

_get_BA(m::VirtualPTDF) = m.BA
_get_arc_susceptances(m::VirtualPTDF) = m.arc_susceptances
_get_valid_ix(m::VirtualPTDF) = m.valid_ix

# --- Islanding zero-forcing (shared by VirtualPTDF and VirtualMODF) -----------
# When a contingency disconnects the network, the post-contingency sensitivity of
# a monitored arc to an injection in a *different* component is exactly zero, but
# the pinv-based Woodbury correction leaves the stale pre-contingency value there.
# These helpers identify the disconnected buses by connectivity and zero them.

# Connected-component label (a representative bus position) of every bus in the
# post-contingency network: the base topology minus the arcs this modification
# fully outages (post-contingency susceptance ≈ 0). A partially-reduced arc still
# connects its buses, so it is kept. Reuses the package union-find over each kept
# arc's endpoints (the ≤ 2 nonzeros of its `BA` column).
function _post_contingency_bus_labels(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    modifications::Tuple{Vararg{ArcModification}},
    n_bus::Int,
)::Vector{Int}
    removed = Set{Int}()
    for m in modifications
        if abs(arc_sus[m.arc_index] + m.delta_b) < MODF_ISLANDING_TOLERANCE
            push!(removed, m.arc_index)
        end
    end

    rv = SparseArrays.rowvals(BA)
    uf = collect(1:n_bus)
    for e in 1:size(BA, 2)
        e in removed && continue
        # Union the arc's endpoints (a BA column has 2 nonzeros; chain if more).
        rng = SparseArrays.nzrange(BA, e)
        @inbounds for k in first(rng):(last(rng) - 1)
            union_sets!(uf, rv[k], rv[k + 1])
        end
    end
    @inbounds for i in 1:n_bus
        uf[i] = get_representative(uf, i)
    end
    return uf
end

# Force entries of buses disconnected from the monitored arc to exactly zero.
# `labels` is empty for connected contingencies, making this a no-op.
function _zero_islanded_entries!(
    row::Vector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    monitored_idx::Int,
    labels::Vector{Int},
)
    isempty(labels) && return row
    # The monitored arc's component is identified by either of its endpoint buses.
    rv = SparseArrays.rowvals(BA)
    rng = SparseArrays.nzrange(BA, monitored_idx)
    isempty(rng) && return row
    monitored_label = labels[rv[first(rng)]]
    @inbounds for j in eachindex(row)
        if labels[j] != monitored_label
            row[j] = 0.0
        end
    end
    return row
end

"""
    _compute_woodbury_factors_impl(K, work_ba_col, temp_data, BA, arc_sus,
                                   valid_ix, modifications) -> WoodburyFactors

Pure-data Woodbury factor computation. Mutates `work_ba_col` and
`temp_data`. The caller is responsible for exclusive access to those
buffers; in `Virtual{PTDF, MODF}` this is provided by holding
`solver_lock` via `with_solver` for the duration of the call.
"""
function _compute_woodbury_factors_impl(
    K,
    work_ba_col::Vector{Float64},
    temp_data::Vector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    valid_ix::Vector{Int},
    bus_to_valid_idx::Vector{Int},
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    M = length(modifications)
    n_bus = length(temp_data)

    arc_indices = Vector{Int}(undef, M)
    delta_b_vec = Vector{Float64}(undef, M)
    for (j, mod) in enumerate(modifications)
        arc_indices[j] = mod.arc_index
        delta_b_vec[j] = mod.delta_b
    end

    # Compute Z[:,j] = B⁻¹ν_j for each modified arc
    Z = Matrix{Float64}(undef, n_bus, M)
    ba_rv_outer = SparseArrays.rowvals(BA)
    ba_nz_outer = SparseArrays.nonzeros(BA)

    for (j, mod) in enumerate(modifications)
        e = mod.arc_index
        b_e = arc_sus[e]

        # Sparse-only extraction of BA[:, e] into work_ba_col.
        fill!(work_ba_col, 0.0)
        @inbounds for k in SparseArrays.nzrange(BA, e)
            valid_i = bus_to_valid_idx[ba_rv_outer[k]]
            valid_i > 0 || continue
            work_ba_col[valid_i] = ba_nz_outer[k]
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

    # Label post-contingency components only when islanding, so the correction can
    # force entries of disconnected buses to exactly zero (see _zero_islanded_entries!).
    labels = Int[]
    if is_island
        @debug "Contingency islands the network; using pinv-based Woodbury correction."
        labels = _post_contingency_bus_labels(BA, arc_sus, modifications, n_bus)
    end

    return WoodburyFactors(Z, W_inv, arc_indices, delta_b_vec, is_island, labels)
end

"""
    _apply_woodbury_correction_impl(K, work_ba_col, temp_data, BA, arc_sus,
                                    valid_ix, monitored_idx, wf) -> Vector{Float64}

Pure-data Woodbury correction. Mutates `work_ba_col` and `temp_data`; the
caller owns exclusive access to those buffers.
"""
function _apply_woodbury_correction_impl(
    K,
    work_ba_col::Vector{Float64},
    temp_data::Vector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    valid_ix::Vector{Int},
    bus_to_valid_idx::Vector{Int},
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
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

    # z_m = B⁻¹ν_m / b_mon_pre via sparse-only BA-column extraction + solve.
    b_mon_pre = arc_sus[monitored_idx]
    fill!(work_ba_col, 0.0)
    ba_rv_mon = SparseArrays.rowvals(BA)
    ba_nz_mon = SparseArrays.nonzeros(BA)
    @inbounds for k in SparseArrays.nzrange(BA, monitored_idx)
        valid_i = bus_to_valid_idx[ba_rv_mon[k]]
        valid_i > 0 || continue
        work_ba_col[valid_i] = ba_nz_mon[k]
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
    # Under islanding, force buses disconnected from the monitored arc to exactly
    # zero (no-op when not islanding: `bus_island_labels` is empty).
    _zero_islanded_entries!(temp_data, BA, monitored_idx, wf.bus_island_labels)
    return copy(temp_data)
end

# Outer dispatchers: VirtualPTDF and VirtualMODF both acquire a solver and
# matched per-worker scratch via `with_solver` / `with_worker`. The
# VirtualMODF methods are defined in virtual_modf_calculations.jl alongside
# the struct.

function _compute_woodbury_factors(
    mat::VirtualPTDF,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    return with_solver(
        mat.K, mat.work_ba_col, mat.temp_data, mat.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        _compute_woodbury_factors_impl(
            K_solver, work_ba_col, temp_data,
            mat.BA, mat.arc_susceptances, mat.valid_ix, mat.bus_to_valid_idx,
            modifications,
        )
    end
end

function _apply_woodbury_correction(
    mat::VirtualPTDF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return with_solver(
        mat.K, mat.work_ba_col, mat.temp_data, mat.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        _apply_woodbury_correction_impl(
            K_solver, work_ba_col, temp_data,
            mat.BA, mat.arc_susceptances, mat.valid_ix, mat.bus_to_valid_idx,
            monitored_idx, wf,
        )
    end
end
