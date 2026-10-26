"""
    _invert_woodbury_W(W_mat, M, signed_delta_b) -> (W_inv::Matrix{Float64}, is_islanding::Bool)

Invert the M×M Woodbury W; closed form for M ≤ 2, LU otherwise.

Singularity is judged on `S = diag(signed_delta_b)·W`, whose diagonal is `1 + Δb·K`. `S` is
independent of the arcs' impedance scale and exactly singular for a true island, whereas `W`
itself shrinks like `x²` for very low-impedance arcs and would be mistaken for singular.
"""
function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    M::Int,
    signed_delta_b::Vector{Float64},
)::Tuple{Matrix{Float64}, Bool}
    if iszero(M)
        # M = 0 (hand-built modifications only; registration rejects it): getri! rejects 0×0.
        return Matrix{Float64}(undef, 0, 0), false
    elseif M == 1
        w = W_mat[1, 1]
        is_island = abs(w * signed_delta_b[1]) < MODF_ISLANDING_TOLERANCE
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
        is_island =
            abs(det_W * signed_delta_b[1] * signed_delta_b[2]) < MODF_ISLANDING_TOLERANCE
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
    scale = LinearAlgebra.Diagonal(signed_delta_b)
    S_lu = LinearAlgebra.lu(scale * W_mat; check = false)
    is_island = any(i -> abs(S_lu.U[i, i]) < MODF_ISLANDING_TOLERANCE, 1:M)
    if is_island
        return LinearAlgebra.pinv(W_mat; atol = MODF_ISLANDING_TOLERANCE), is_island
    end
    # W = D⁻¹S, so W⁻¹ = S⁻¹D.
    return LinearAlgebra.inv(S_lu) * scale, is_island
end

# --- Islanding zero-forcing (shared by VirtualPTDF and VirtualMODF) -----------
# When a contingency disconnects the network, the post-contingency sensitivity of
# a monitored arc to an injection in a *different* component is exactly zero, but
# the pinv-based Woodbury correction leaves the stale pre-contingency value there.
# These helpers identify the disconnected buses by connectivity and zero them.

_removes_arc(arc_sus::Vector{Float64}, m::ArcModification) =
    abs(arc_sus[m.arc_index] + m.delta_b) < MODF_ISLANDING_TOLERANCE

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
        if _removes_arc(arc_sus, m)
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

# Labeler for `_woodbury_factors_from_Z`: a modification that fully outages exactly one bridge of
# the base `BA` graph copies the base component labels and relabels the bridge's far side, O(n_bus)
# instead of a union-find over every arc; any other modification runs
# `_post_contingency_bus_labels`. The partition is the union-find's, the representatives are not:
# base labels are DFS roots and a far side takes its child endpoint, never a root.
struct BridgeLabels
    tree::BridgeTree
    base::Vector{Int}
end

function BridgeLabels(BA::SparseArrays.SparseMatrixCSC{Float64, Int})
    rv = SparseArrays.rowvals(BA)
    edges = Vector{Tuple{Int, Int}}(undef, size(BA, 2))
    for e in 1:size(BA, 2)
        rng = SparseArrays.nzrange(BA, e)
        length(rng) <= 2 ||
            error("BA column $e has $(length(rng)) nonzeros; an arc has at most two.")
        # A column with fewer than two nonzeros connects nothing: a self-loop, which
        # `find_bridges` ignores.
        if length(rng) == 2
            edges[e] = (rv[first(rng)], rv[last(rng)])
        else
            edges[e] = (1, 1)
        end
    end
    n_bus = size(BA, 1)
    tree = find_bridges(n_bus, edges)
    base = zeros(Int, n_bus)
    i = 1
    while i <= n_bus
        root = tree.preorder[i]
        for p in i:tree.last[root]
            base[tree.preorder[p]] = root
        end
        i = tree.last[root] + 1
    end
    return BridgeLabels(tree, base)
end

function (bl::BridgeLabels)(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    modifications::Tuple{Vararg{ArcModification}},
    n_bus::Int,
)::Vector{Int}
    removed = 0
    n_removed = 0
    for m in modifications
        if _removes_arc(arc_sus, m)
            removed = m.arc_index
            n_removed += 1
        end
    end
    if !isone(n_removed) || !is_bridge(bl.tree, removed)
        return _post_contingency_bus_labels(BA, arc_sus, modifications, n_bus)
    end
    labels = copy(bl.base)
    far = bl.tree.far_end[removed]
    for b in bridge_far_side(bl.tree, removed)
        labels[b] = far
    end
    return labels
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

# Sign of an arc's DC susceptance: BA holds it at the arc's from bus, the bus `A` marks +1.
function _arc_susceptance_sign(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    e::Int,
)::Float64
    rng = SparseArrays.nzrange(BA, e)
    isempty(rng) && return 1.0
    p = first(rng)
    return sign(SparseArrays.nonzeros(BA)[p]) * A[e, SparseArrays.rowvals(BA)[p]]
end

"""
    _woodbury_factors_from_Z(Z, BA, signs, arc_sus, modifications, arc_out[, labeler]) -> WoodburyFactors

Assemble the Woodbury factors from an already-resolved `Z`, whose column `j` is
`B⁻¹ν_j` for the `j`-th modified arc in full-bus space. The callers differ only
in how they obtain `Z`: one solve per arc in the kernel path, a lookup into the
batched pre-contingency solves in `populate_cache`. `signs[e]` is the sign of arc `e`'s DC
susceptance (`VirtualFactorCore.arc_susceptance_signs`). Islanding labels come from
`labeler(BA, arc_sus, modifications, n_bus)`, e.g. a `BridgeLabels`.
"""
function _woodbury_factors_from_Z(
    Z::Matrix{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    signs::Vector{Float64},
    arc_sus::Vector{Float64},
    modifications::Tuple{Vararg{ArcModification}},
    arc_out::Vector{Bool},
    labeler::F = _post_contingency_bus_labels,
)::WoodburyFactors where {F}
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

    # W = diag(1/Δb) + K_mat. Z and K use ν = BA[:, e] / |b_e| = sign(b_e) · incidence, so the
    # rank-one term must carry the signed change sign(b_e) · Δ|b|: the magnitude-space Δb alone
    # adds a negative-reactance arc a second time instead of removing it.
    signed_delta_b = [signs[arc_indices[j]] * delta_b_vec[j] for j in 1:M]
    W_mat = LinearAlgebra.diagm(1.0 ./ signed_delta_b) + K_mat
    W_inv, is_island = _invert_woodbury_W(W_mat, M, signed_delta_b)

    # Label post-contingency components only when islanding, so the correction can
    # force entries of disconnected buses to exactly zero (see _zero_islanded_entries!).
    labels = Int[]
    if is_island
        @debug "Contingency islands the network; using pinv-based Woodbury correction."
        labels = labeler(BA, arc_sus, modifications, n_bus)
    end

    return WoodburyFactors(Z, W_inv, arc_indices, delta_b_vec, arc_out, is_island, labels)
end

# zm_Z = ν_mᵀ Z[:, eachindex(zm_Z)], ν_m = BA[:, m] / b_pre (BA's sign convention, not A's).
function _monitored_Z!(
    zm_Z::AbstractVector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    m::Int,
    b_pre::Float64,
    Z::Matrix{Float64},
)
    ba_nzv = SparseArrays.nonzeros(BA)
    ba_rv = SparseArrays.rowvals(BA)
    fill!(zm_Z, 0.0)
    @inbounds for nz_idx in SparseArrays.nzrange(BA, m)
        row = ba_rv[nz_idx]
        coeff = ba_nzv[nz_idx] / b_pre
        for j in eachindex(zm_Z)
            zm_Z[j] += coeff * Z[row, j]
        end
    end
    return zm_Z
end

# Per modified arc: true when the modification opens every member. Counts members instead of
# comparing susceptances, because the group susceptance is not the exact sum of its members'.
function _arcs_fully_opened(
    core::VirtualFactorCore,
    modifications::Tuple{Vararg{ArcModification}},
)::Vector{Bool}
    nr = get_network_reduction_data(core)
    arc_ax = get_arc_axis(core)
    return Bool[
        m.opened > 0 && m.opened >= _arc_member_count(nr, arc_ax[m.arc_index]) for
        m in modifications
    ]
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

# True when the modifications take the monitored arc out of service: all its members are open,
# or the post-modification susceptance is exactly zero (a DC-only `ArcModification`).
function _monitored_arc_out(
    arc_sus::Vector{Float64},
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Bool
    for (j, idx) in enumerate(wf.arc_indices)
        if idx == monitored_idx && wf.arc_out[j]
            return true
        end
    end
    return iszero(_post_modification_susceptance(arc_sus, monitored_idx, wf))
end

"""
    _woodbury_correction!(z_m, BA, b_mon_pre, b_mon_post, monitored_idx, wf) -> Vector{Float64}

Turn `z_m` — `B⁻¹ν_m / b_mon_pre` in full-bus space — into the post-modification
PTDF row of the monitored arc, in place.
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
    return _woodbury_correction!(
        z_m,
        zeros(M),
        zeros(M),
        BA,
        b_mon_pre,
        b_mon_post,
        monitored_idx,
        wf,
    )
end

"""
    _woodbury_correction!(z_m, zm_Z, coeff, BA, b_mon_pre, b_mon_post, monitored_idx, wf) -> Vector{Float64}

Non-allocating form: `zm_Z` and `coeff` are caller-owned buffers of length at least the
number of modified arcs; only their first `M` entries are written.
"""
function _woodbury_correction!(
    z_m::Vector{Float64},
    zm_Z_buf::Vector{Float64},
    coeff_buf::Vector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    b_mon_pre::Float64,
    b_mon_post::Float64,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    M = length(wf.arc_indices)
    zm_Z = view(zm_Z_buf, 1:M)
    correction_coeff = view(coeff_buf, 1:M)

    _monitored_Z!(zm_Z, BA, monitored_idx, b_mon_pre, wf.Z)

    # Woodbury correction: z_m -= Z · (W⁻¹ · zm_Z), then scale by b_mon_post.
    LinearAlgebra.mul!(correction_coeff, wf.W_inv, zm_Z)
    LinearAlgebra.mul!(z_m, wf.Z, correction_coeff, -1.0, 1.0)
    z_m .*= b_mon_post
    # Under islanding, force buses disconnected from the monitored arc to exactly
    # zero (no-op when not islanding: `bus_island_labels` is empty).
    _zero_islanded_entries!(z_m, BA, monitored_idx, wf.bus_island_labels)
    return z_m
end

# Solves and the scratch slot go through `with_solver`, which holds the core's `solver_lock`.
"""
    _compute_woodbury_factors(mat, modifications[, labeler]) -> WoodburyFactors

Woodbury factors `Z[:,j] = B⁻¹ν_j` for each modified arc, on a `VirtualPTDF`, a `VirtualMODF`
or a `VirtualFactorCore` (solving on that core's factorization).
"""
_compute_woodbury_factors(
    mat::Union{VirtualPTDF, VirtualMODF},
    modifications::Tuple{Vararg{ArcModification}},
) = _compute_woodbury_factors(get_core(mat), modifications)

function _compute_woodbury_factors(
    core::VirtualFactorCore,
    modifications::Tuple{Vararg{ArcModification}},
    labeler::F = _post_contingency_bus_labels,
)::WoodburyFactors where {F}
    return with_solver(
        core.K,
        core.work_ba_col,
        core.temp_data,
        core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        Z = Matrix{Float64}(undef, length(temp_data), length(modifications))
        for (j, mod) in enumerate(modifications)
            e = mod.arc_index
            lin_solve =
                _solve_ba_column!(K_solver, work_ba_col, core.BA, core.bus_to_valid_idx, e)
            _gather_to_buses!(
                view(Z, :, j),
                core.valid_ix,
                lin_solve,
                core.arc_susceptances[e],
            )
        end
        return _woodbury_factors_from_Z(
            Z,
            core.BA,
            core.arc_susceptance_signs,
            core.arc_susceptances,
            modifications,
            _arcs_fully_opened(core, modifications),
            labeler,
        )
    end
end

_apply_woodbury_correction(
    mat::VirtualMODF,
    monitored_idx::Int,
    wf::WoodburyFactors,
) = _apply_woodbury_correction(get_core(mat), monitored_idx, wf)

function _apply_woodbury_correction(
    vptdf::VirtualPTDF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)
    row = _apply_woodbury_correction(get_core(vptdf), monitored_idx, wf)
    return _apply_dist_slack!(row, vptdf, wf)
end

function _apply_dist_slack!(row::Vector{Float64}, vptdf::VirtualPTDF, wf::WoodburyFactors)
    _use_dist_slack(vptdf) || return row
    if wf.is_islanding
        error(
            "Distributed slack is not supported for a modification that islands the network.",
        )
    end
    # A distributed slack withdraws each injection in proportion to the slack weights, which
    # shifts every entry of a sensitivity row by the row's weighted mean.
    row .-= LinearAlgebra.dot(row, get_dist_slack_normalized(vptdf))
    return row
end

_apply_woodbury_correction(
    core::VirtualFactorCore,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64} = only(_woodbury_rows(core, monitored_idx, (wf,)))

"""
    _woodbury_rows(core, monitored_idx, factors) -> Vector{Vector{Float64}}

Rows of the monitored arc on several networks from one solve of its `BA` column:
`rows[i]` is the post-modification row under `factors[i]`, or the base row (reference-bus
slack) when `factors[i]` is `nothing`. Each row equals what the single-modification path
computes, bit for bit: the base row is the plain solve, and every modification corrects its
own copy of `B⁻¹ν_m / b_mon_pre`. No solve runs when every requested row is the zero row of
an arc taken out of service.
"""
function _woodbury_rows(
    core::VirtualFactorCore,
    monitored_idx::Int,
    factors,
)::Vector{Vector{Float64}}
    arc_sus = core.arc_susceptances
    n = length(factors)
    rows = Vector{Vector{Float64}}(undef, n)
    max_modified = 0
    for wf in factors
        isnothing(wf) && continue
        max_modified = max(max_modified, length(wf.arc_indices))
    end
    zm_Z_buf = Vector{Float64}(undef, max_modified)
    coeff_buf = Vector{Float64}(undef, max_modified)
    return with_solver(
        core.K,
        core.work_ba_col,
        core.temp_data,
        core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        n_bus = length(temp_data)
        needs_base = false
        needs_scaled = false
        for wf in factors
            if isnothing(wf)
                needs_base = true
            elseif !_monitored_arc_out(arc_sus, monitored_idx, wf)
                needs_scaled = true
            end
        end
        b_mon_pre = arc_sus[monitored_idx]
        base_row = Float64[]
        if needs_base || needs_scaled
            lin_solve = _solve_ba_column!(
                K_solver,
                work_ba_col,
                core.BA,
                core.bus_to_valid_idx,
                monitored_idx,
            )
            if needs_base
                base_row = _gather_to_buses!(
                    Vector{Float64}(undef, n_bus),
                    core.valid_ix,
                    lin_solve,
                )
            end
            # z_m = B⁻¹ν_m / b_mon_pre, shared by every modification.
            needs_scaled &&
                _gather_to_buses!(temp_data, core.valid_ix, lin_solve, b_mon_pre)
        end
        base_used = false
        for (i, wf) in enumerate(factors)
            if isnothing(wf)
                # Every base-network slot gets its own row; only the first takes the buffer.
                if base_used
                    rows[i] = copy(base_row)
                else
                    rows[i] = base_row
                end
                base_used = true
            elseif _monitored_arc_out(arc_sus, monitored_idx, wf)
                rows[i] = zeros(n_bus)
            else
                b_mon = _post_modification_susceptance(arc_sus, monitored_idx, wf)
                rows[i] = _woodbury_correction!(
                    copy(temp_data),
                    zm_Z_buf,
                    coeff_buf,
                    core.BA,
                    b_mon_pre,
                    b_mon,
                    monitored_idx,
                    wf,
                )
            end
        end
        return rows
    end
end
