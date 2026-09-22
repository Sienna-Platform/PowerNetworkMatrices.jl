"""
Shared Woodbury matrix identity kernel for computing post-modification
network sensitivity factors. Used by both VirtualPTDF and VirtualMODF.

Implements van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹
"""

"""
    _invert_woodbury_W(W_mat, M) -> (W_inv::Matrix{Float64}, is_islanding::Bool)

Invert the M×M Woodbury W matrix. Analytical formulas for M=1 and M=2 avoid
LU factorization overhead; M > 2 falls back to LU. `M` is a plain `Int`: it comes
from a runtime contingency size, so a `Val{M}` would buy a dynamic dispatch and a
fresh specialization per distinct size rather than a compile-time constant.
"""
function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    M::Int,
)::Tuple{Matrix{Float64}, Bool}
    if iszero(M)
        # A contingency whose branch was eliminated by the zero-impedance reduction
        # resolves to no arc modifications, so W is 0×0. LAPACK's getri! (reached via
        # `inv`) rejects a 0×0 argument ("invalid argument #6"), so return the empty
        # inverse directly. The correction is then nil and the unmodified base PTDF
        # row survives. Registered outages can no longer reach this state
        # (`_validate_transmission_survived` rejects them); a hand-built modification can.
        return Matrix{Float64}(undef, 0, 0), false
    elseif M == 1
        w = W_mat[1, 1]
        is_island = abs(w) < MODF_ISLANDING_TOLERANCE
        W_inv = Matrix{Float64}(undef, 1, 1)
        if is_island
            W_inv[1, 1] = 0.0
        else
            W_inv[1, 1] = 1.0 / w
        end
        return W_inv, is_island
    elseif M == 2
        a, b, c, d = W_mat[1, 1], W_mat[1, 2], W_mat[2, 1], W_mat[2, 2]
        det_W = a * d - b * c
        is_island = abs(det_W) < MODF_ISLANDING_TOLERANCE
        if is_island
            return LinearAlgebra.pinv(W_mat; atol = MODF_ISLANDING_TOLERANCE), is_island
        end
        inv_det = 1.0 / det_W
        W_inv = Matrix{Float64}(undef, 2, 2)
        W_inv[1, 1] = d * inv_det
        W_inv[1, 2] = -b * inv_det
        W_inv[2, 1] = -c * inv_det
        W_inv[2, 2] = a * inv_det
        return W_inv, is_island
    end
    W_lu = LinearAlgebra.lu(W_mat; check = false)
    is_island = any(i -> abs(W_lu.U[i, i]) < MODF_ISLANDING_TOLERANCE, 1:M)
    if is_island
        return LinearAlgebra.pinv(W_mat; atol = MODF_ISLANDING_TOLERANCE), is_island
    end
    return LinearAlgebra.inv(W_lu), is_island
end

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
    _woodbury_factors_from_Z(Z, BA, arc_sus, modifications) -> WoodburyFactors

Assemble the Woodbury factors from an already-resolved `Z`, whose column `j` is
`B⁻¹ν_j` for the `j`-th modified arc in full-bus space. The callers differ only
in how they obtain `Z`: one solve per arc in the kernel path, a lookup into the
batched pre-contingency solves in `populate_cache`.
"""
function _woodbury_factors_from_Z(
    Z::Matrix{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    M = length(modifications)
    n_bus = size(Z, 1)

    arc_indices = Vector{Int}(undef, M)
    delta_b_vec = Vector{Float64}(undef, M)
    for (j, mod) in enumerate(modifications)
        arc_indices[j] = mod.arc_index
        delta_b_vec[j] = mod.delta_b
    end

    # K_mat[i,j] = ν_i⊤ B⁻¹ ν_j
    # Use BA[:,arc]/b instead of A[arc,:] for consistent sign convention.
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
    W_inv, is_island = _invert_woodbury_W(W_mat, M)

    # Label post-contingency components only when islanding, so the correction can
    # force entries of disconnected buses to exactly zero (see _zero_islanded_entries!).
    labels = Int[]
    if is_island
        @debug "Contingency islands the network; using pinv-based Woodbury correction."
        labels = _post_contingency_bus_labels(BA, arc_sus, modifications, n_bus)
    end

    return WoodburyFactors(Z, W_inv, arc_indices, delta_b_vec, is_island, labels)
end

# Susceptance of the monitored arc once the modifications are applied.
function _post_modification_susceptance(
    arc_sus::Vector{Float64},
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Float64
    b_mon = arc_sus[monitored_idx]
    for (j, idx) in enumerate(wf.arc_indices)
        if idx == monitored_idx
            b_mon += wf.delta_b[j]
        end
    end
    return b_mon
end

"""
    _woodbury_correction!(z_m, BA, b_mon_pre, b_mon_post, monitored_idx, wf) -> Vector{Float64}

Turn `z_m` — `B⁻¹ν_m / b_mon_pre` in full-bus space — into the post-modification
PTDF row of the monitored arc, in place. The callers differ only in how they
obtain `z_m`: a solve in the kernel path, a lookup into the batched
pre-contingency solves in `populate_cache`.
"""
function _woodbury_correction!(
    z_m::Vector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    b_mon_pre::Float64,
    b_mon_post::Float64,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    M = length(wf.arc_indices)

    # ν_m⊤ · Z  (1 × M vector)
    # Use BA[:,m]/b instead of A[m,:] for consistent sign convention.
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

    # Woodbury correction: z_m -= Z · (W⁻¹ · zm_Z), then scale by b_mon_post.
    correction_coeff = wf.W_inv * zm_Z
    LinearAlgebra.mul!(z_m, wf.Z, correction_coeff, -1.0, 1.0)
    z_m .*= b_mon_post
    # Under islanding, force buses disconnected from the monitored arc to exactly
    # zero (no-op when not islanding: `bus_island_labels` is empty).
    _zero_islanded_entries!(z_m, BA, monitored_idx, wf.bus_island_labels)
    return z_m
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

    # Compute Z[:,j] = B⁻¹ν_j for each modified arc
    Z = Matrix{Float64}(undef, n_bus, M)

    for (j, mod) in enumerate(modifications)
        e = mod.arc_index
        b_e = arc_sus[e]

        lin_solve = _solve_ba_column!(K, work_ba_col, BA, bus_to_valid_idx, e)

        fill!(view(Z, :, j), 0.0)
        @inbounds for i in eachindex(valid_ix)
            Z[valid_ix[i], j] = lin_solve[i] / b_e
        end
    end

    return _woodbury_factors_from_Z(Z, BA, arc_sus, modifications)
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

    b_mon = _post_modification_susceptance(arc_sus, monitored_idx, wf)
    if abs(b_mon) < eps()
        return zeros(n_bus)
    end

    # z_m = B⁻¹ν_m / b_mon_pre.
    b_mon_pre = arc_sus[monitored_idx]
    lin_solve = _solve_ba_column!(K, work_ba_col, BA, bus_to_valid_idx, monitored_idx)

    fill!(temp_data, 0.0)
    @inbounds for i in eachindex(valid_ix)
        temp_data[valid_ix[i]] = lin_solve[i] / b_mon_pre
    end

    _woodbury_correction!(temp_data, BA, b_mon_pre, b_mon, monitored_idx, wf)
    return copy(temp_data)
end

# Outer dispatchers. The single implementation operates on the shared
# `VirtualFactorCore`; the `Virtual{PTDF, MODF}` wrappers forward to it (the
# VirtualMODF forwards live in virtual_modf_calculations.jl). Both acquire the
# solver and the single scratch slot via `with_solver`.

function _compute_woodbury_factors(
    core::VirtualFactorCore,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        _compute_woodbury_factors_impl(
            K_solver, work_ba_col, temp_data,
            core.BA, core.arc_susceptances, core.valid_ix, core.bus_to_valid_idx,
            modifications,
        )
    end
end

function _apply_woodbury_correction(
    core::VirtualFactorCore,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        _apply_woodbury_correction_impl(
            K_solver, work_ba_col, temp_data,
            core.BA, core.arc_susceptances, core.valid_ix, core.bus_to_valid_idx,
            monitored_idx, wf,
        )
    end
end

# VirtualPTDF forwards to the shared core method.
function _compute_woodbury_factors(
    mat::VirtualPTDF,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    return _compute_woodbury_factors(get_core(mat), modifications)
end

function _apply_woodbury_correction(
    mat::VirtualPTDF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return _apply_woodbury_correction(get_core(mat), monitored_idx, wf)
end
