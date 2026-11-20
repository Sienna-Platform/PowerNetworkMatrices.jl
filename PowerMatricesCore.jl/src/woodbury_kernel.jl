"""
    _invert_woodbury_W(W_mat, M, signed_delta_b) -> (W_inv::Matrix{Float64}, is_islanding::Bool)

Invert the M×M Woodbury W; closed form for M ≤ 2, LU of `S = diag(signed_delta_b)·W` otherwise.

The modification islands the network when a pivot of the partially pivoted LU of `S` is below
`MODF_ISLANDING_TOLERANCE`. `S` has the diagonal `1 + Δb·K`. For an opened arc of reactance
`x`, this entry is `x/(x + X_alt)`, where `X_alt` is the reactance of the other paths between
the buses of the arc. Thus the pivots of `S` scale with `x/X_alt` for a low-impedance arc,
rather than like `x²` as the pivots of `W` do. For an island, `W_inv` is a pseudo-inverse.
"""
function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    M::Int,
    signed_delta_b::Vector{Float64},
)::Tuple{Matrix{Float64}, Bool}
    if iszero(M)
        # getri! rejects a 0×0 matrix.
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
        # The pivots of the LU are p1 = max(|S11|, |S21|) and p2 = |det(S)| / p1. A test on
        # det(S) alone flags two low-impedance arcs that are not bridges as an island: each
        # pivot is small, so the product is below the tolerance.
        p1 = max(abs(signed_delta_b[1] * a), abs(signed_delta_b[2] * c))
        abs_det_S = abs(det_W * signed_delta_b[1] * signed_delta_b[2])
        is_island =
            p1 < MODF_ISLANDING_TOLERANCE || abs_det_S < MODF_ISLANDING_TOLERANCE * p1
        if is_island
            return _island_pinv(W_mat, signed_delta_b), is_island
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
    is_island = any(i -> abs(S_lu.factors[i, i]) < MODF_ISLANDING_TOLERANCE, 1:M)
    if is_island
        return _island_pinv(W_mat, signed_delta_b), is_island
    end
    # W = D⁻¹S, so W⁻¹ = S⁻¹D.
    return LinearAlgebra.inv(S_lu) * scale, is_island
end

# Pseudo-inverse of a singular W, cut on S = D·W for the reason the detection uses S: an
# absolute cutoff on W itself also drops the directions of very low-impedance arcs.
function _island_pinv(W_mat::Matrix{Float64}, signed_delta_b::Vector{Float64})
    scale = LinearAlgebra.Diagonal(signed_delta_b)
    return LinearAlgebra.pinv(scale * W_mat; atol = MODF_ISLANDING_TOLERANCE) * scale
end

# --- Islanding zero-forcing (shared by the PTDF and MODF corrections) ---------
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
    modifications::Vector{ArcModification},
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
    modifications::Vector{ArcModification},
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

"""
    _changed_modifications(modifications) -> Vector{Int}

Positions in `modifications` of the arcs whose DC susceptance changes. Only these arcs enter
the Woodbury factors. An arc with `delta_b == 0`, such as an arc with no susceptance, or a
parallel group whose only remaining member has zero reactance when that member opens, leaves
`B` unchanged, and its `1/Δb` term in `W` would be infinite.

A `NetworkModification` keeps these entries, because their `delta_y*` and `opened` fields can
still change the Ybus: an outage of a line with `r > 0` and `x = 0` is one example. The AC
consumers `compute_ybus_delta`, `apply_ybus_modification` and the AC workspaces of
PowerTransmissionSecurityAnalysis read these fields.
"""
_changed_modifications(modifications::Vector{ArcModification}) =
    [j for (j, mod) in enumerate(modifications) if !iszero(mod.delta_b)]

# ν = BA[:, e] / b_e is 0/0 for an arc with no DC susceptance, e.g. a parallel group whose
# reactances cancel, so every solve for it holds NaN.
function _check_changed_arcs_have_susceptance(
    arc_sus::Vector{Float64},
    modifications::Vector{ArcModification},
)
    for k in _changed_modifications(modifications)
        e = modifications[k].arc_index
        iszero(arc_sus[e]) && error(
            "Arc $e has zero DC susceptance, but its modification changes it by " *
            "$(modifications[k].delta_b). The Woodbury update cannot add susceptance to it.",
        )
    end
    return nothing
end

"""
    _woodbury_factors_from_Z(Z, basis, BA, signs, arc_sus, modifications, arc_out[, labeler]) -> WoodburyFactors

Assemble the arc-form Woodbury factors from an already-resolved `Z`; `_arc_form_factors` is
the caller. Column `j` of `Z` is `B⁻¹ν` in full-bus space for the `j`-th arc of
`_changed_modifications(modifications)`, and `basis` is the arc-form basis of those arcs.
`arc_out` has one entry per modification. `signs[e]` is the sign of arc `e`'s DC susceptance
(`FactorCore.arc_susceptance_signs`). Islanding labels come from
`labeler(BA, arc_sus, modifications, n_bus)`, e.g. a `BridgeLabels`, which sees every
modification.
"""
function _woodbury_factors_from_Z(
    Z::Matrix{Float64},
    basis::WoodburyBasis,
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    signs::Vector{Float64},
    arc_sus::Vector{Float64},
    modifications::Vector{ArcModification},
    arc_out::Vector{Bool},
    labeler::F = _post_contingency_bus_labels,
)::WoodburyFactors where {F}
    n_bus = size(Z, 1)
    kept = _changed_modifications(modifications)
    M = length(kept)
    _check_changed_arcs_have_susceptance(arc_sus, modifications)
    size(Z, 2) == M ||
        error("Z has $(size(Z, 2)) columns; expected $M, one per changed arc.")

    arc_indices = Vector{Int}(undef, M)
    delta_b_vec = Vector{Float64}(undef, M)
    for (j, k) in enumerate(kept)
        e = modifications[k].arc_index
        arc_indices[j] = e
        delta_b_vec[j] = modifications[k].delta_b
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

    return WoodburyFactors(
        Z,
        W_inv,
        arc_indices,
        delta_b_vec,
        arc_out[kept],
        is_island,
        labels,
        basis,
    )
end

"""
    _woodbury_rank(wf::WoodburyFactors) -> Int

Number of columns of the Woodbury update `wf`, the size of every per-column buffer.
"""
_woodbury_rank(wf::WoodburyFactors) = size(wf.W_inv, 1)

# The arc-form factors of `basis`. They island, with `post_labels` as the island labels,
# when `W` is singular or `post_labels` puts a bus outside the islands of the reference
# buses `ref_positions`.
function _arc_form_factors(
    Z::Matrix{Float64},
    basis::WoodburyBasis,
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    signs::Vector{Float64},
    arc_sus::Vector{Float64},
    modifications::Vector{ArcModification},
    arc_out::Vector{Bool},
    post_labels::Vector{Int},
    ref_positions::Vector{Int},
)::WoodburyFactors
    wf = _woodbury_factors_from_Z(
        Z,
        basis,
        BA,
        signs,
        arc_sus,
        modifications,
        arc_out,
        (_, _, _, _) -> post_labels,
    )
    (wf.is_islanding || all(_live_buses(post_labels, ref_positions, size(Z, 1)))) &&
        return wf
    return WoodburyFactors(
        wf.Z,
        wf.W_inv,
        wf.arc_indices,
        wf.delta_b,
        wf.arc_out,
        true,
        post_labels,
        wf.basis,
    )
end

"""
    _woodbury_factors_from_basis(Z, basis, BA, signs, arc_sus, modifications, arc_out, post_labels, ref_positions, arc_fallback) -> WoodburyFactors

Woodbury factors of `modifications` on the base network from `Z = B⁻¹U`, `U` the columns of
`basis`. An arc-form basis assembles exactly as `_woodbury_factors_from_Z`. Otherwise
`K[i, j] = scale_i (Z[p_i, j] − Z[q_i, j])`, `S = I + D K` and `W⁻¹ = S⁻¹ D` by LU; the
modification islands when `post_labels` puts a bus outside the islands of the reference buses
`ref_positions`, and then carries `post_labels`. When `S` has a pivot below
`MODF_ISLANDING_TOLERANCE`, the result is `arc_fallback()`, the arc-form factors.
"""
function _woodbury_factors_from_basis(
    Z::Matrix{Float64},
    basis::WoodburyBasis,
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    signs::Vector{Float64},
    arc_sus::Vector{Float64},
    modifications::Vector{ArcModification},
    arc_out::Vector{Bool},
    post_labels::Vector{Int},
    ref_positions::Vector{Int},
    arc_fallback::F,
)::WoodburyFactors where {F}
    if basis.arc_form
        return _arc_form_factors(
            Z,
            basis,
            BA,
            signs,
            arc_sus,
            modifications,
            arc_out,
            post_labels,
            ref_positions,
        )
    end
    r = length(basis)
    K_mat = Matrix{Float64}(undef, r, r)
    for j in 1:r, i in 1:r
        K_mat[i, j] = basis.scale[i] * (Z[basis.p[i], j] - Z[basis.q[i], j])
    end
    W_inv = Matrix{Float64}(undef, 0, 0)
    if r > 0
        S_lu = LinearAlgebra.lu(LinearAlgebra.I + basis.D * K_mat; check = false)
        any(i -> abs(S_lu.factors[i, i]) < MODF_ISLANDING_TOLERANCE, 1:r) &&
            return arc_fallback()
        W_inv = S_lu \ basis.D
    end

    kept = _changed_modifications(modifications)
    arc_indices = Int[modifications[k].arc_index for k in kept]
    delta_b = Float64[modifications[k].delta_b for k in kept]
    is_island = !all(_live_buses(post_labels, ref_positions, size(Z, 1)))
    return WoodburyFactors(
        Z,
        W_inv,
        arc_indices,
        delta_b,
        arc_out[kept],
        is_island,
        is_island ? post_labels : Int[],
        basis,
    )
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

# True when `m` opens every member of its arc. Counts members instead of comparing
# susceptances, because the group susceptance is not the exact sum of its members'.
_opens_every_member(core::FactorCore, m::ArcModification) =
    m.opened > 0 && m.opened >= core.arc_live_members[m.arc_index]

# Per modified arc: true when the modification opens every member.
function _arcs_fully_opened(
    core::FactorCore,
    modifications::Vector{ArcModification},
)::Vector{Bool}
    return Bool[_opens_every_member(core, m) for m in modifications]
end

# The entry of `modifications` on arc `arc_index`, or `nothing`. A `NetworkModification` keeps
# its arc modifications sorted by arc index, so this is a binary search.
function _arc_modification_at(
    modifications::Vector{ArcModification},
    arc_index::Int,
)::Union{Nothing, ArcModification}
    lo, hi = 1, length(modifications)
    while lo <= hi
        mid = (lo + hi) >>> 1
        m = modifications[mid]
        m.arc_index == arc_index && return m
        if m.arc_index < arc_index
            lo = mid + 1
        else
            hi = mid - 1
        end
    end
    return nothing
end

# The modification a row is judged on: the one taking the base network to the row's network,
# as its arc modifications or as the Woodbury factors assembled for it on the base network.
const _PostModification = Union{Vector{ArcModification}, WoodburyFactors}

# Susceptance change of arc `m` under `post`, or `nothing` when `post` leaves it unchanged.
function _delta_b_at(post::Vector{ArcModification}, m::Int)::Union{Nothing, Float64}
    mod = _arc_modification_at(post, m)
    return isnothing(mod) ? nothing : mod.delta_b
end

function _delta_b_at(wf::WoodburyFactors, m::Int)::Union{Nothing, Float64}
    j = findfirst(==(m), wf.arc_indices)
    return isnothing(j) ? nothing : wf.delta_b[j]
end

# True when `post` opens every member of arc `m`.
function _opens_every_member_at(
    core::FactorCore,
    post::Vector{ArcModification},
    m::Int,
)::Bool
    mod = _arc_modification_at(post, m)
    isnothing(mod) && return false
    return _opens_every_member(core, mod)
end

function _opens_every_member_at(::FactorCore, wf::WoodburyFactors, m::Int)::Bool
    j = findfirst(==(m), wf.arc_indices)
    return !isnothing(j) && wf.arc_out[j]
end

"""
    _post_modification_susceptance(arc_sus, m, post) -> Float64

Susceptance of arc `m` on the network `post` takes the base network to: its base susceptance
plus the change `post` makes on it, in the base arc's magnitude frame.
"""
function _post_modification_susceptance(
    arc_sus::Vector{Float64},
    m::Int,
    post::_PostModification,
)::Float64
    b = arc_sus[m]
    delta_b = _delta_b_at(post, m)
    if !isnothing(delta_b)
        b += delta_b
    end
    return b
end

"""
    _post_modification_arc_out(arc_sus, core, m, post) -> Bool

Whether arc `m` is out of service on the network `post` takes the base network to: `post`
opens every member, or leaves the arc with exactly zero susceptance (a DC-only
`ArcModification`). Members are counted because the summed `delta_b` of a lossy parallel
group does not cancel the arc susceptance exactly.
"""
function _post_modification_arc_out(
    arc_sus::Vector{Float64},
    core::FactorCore,
    m::Int,
    post::_PostModification,
)::Bool
    _opens_every_member_at(core, post, m) && return true
    return iszero(_post_modification_susceptance(arc_sus, m, post))
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
    M = _woodbury_rank(wf)
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

Non-allocating form: `zm_Z` and `coeff` are caller-owned buffers of length at least
`_woodbury_rank(wf)`; only their first `M` entries are written.
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
    M = _woodbury_rank(wf)
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

function _check_arc_indices(
    core::FactorCore,
    modifications::Vector{ArcModification},
)
    for mod in modifications
        _check_arc_index(core, mod.arc_index)
    end
    return
end

# Column `j` of the result is `B⁻¹ν_j` for the `j`-th arc of `_changed_modifications`, in
# full-bus space, from one base solve per arc. Call inside `with_solver`, which owns `K_solver` and `work_ba_col`.
function _base_woodbury_columns(
    core::FactorCore,
    K_solver,
    work_ba_col::Vector{Float64},
    n_bus::Int,
    modifications::Vector{ArcModification},
)::Matrix{Float64}
    kept = _changed_modifications(modifications)
    Z = Matrix{Float64}(undef, n_bus, length(kept))
    for (j, k) in enumerate(kept)
        e = modifications[k].arc_index
        lin_solve =
            _solve_ba_column!(K_solver, work_ba_col, core.BA, core.bus_to_valid_idx, e)
        _gather_to_buses!(view(Z, :, j), core.valid_ix, lin_solve, core.arc_susceptances[e])
    end
    return Z
end

# Solve `B x = scale · (e_p − e_q)` on the valid buses; a reference bus takes no entry.
# Capture the return: only KLU solves in place.
function _solve_node_pair!(
    K,
    work::Vector{Float64},
    bus_to_valid_idx::Vector{Int},
    p::Int,
    q::Int,
    scale::Float64,
)
    fill!(work, 0.0)
    valid_p = bus_to_valid_idx[p]
    valid_p > 0 && (work[valid_p] = scale)
    valid_q = bus_to_valid_idx[q]
    valid_q > 0 && (work[valid_q] = -scale)
    return _solve_factorization(K, work)
end

# Column `j` of the result is `B⁻¹u_j` for column `u_j` of `basis`, in full-bus space, from one
# base solve per column; an arc-form basis solves each modified arc as
# `_base_woodbury_columns`. Call inside `with_solver`, which owns `K_solver` and `work_ba_col`.
function _basis_columns(
    core::FactorCore,
    K_solver,
    work_ba_col::Vector{Float64},
    n_bus::Int,
    basis::WoodburyBasis,
    modifications::Vector{ArcModification},
)::Matrix{Float64}
    basis.arc_form &&
        return _base_woodbury_columns(core, K_solver, work_ba_col, n_bus, modifications)
    Z = Matrix{Float64}(undef, n_bus, length(basis))
    for j in 1:length(basis)
        lin_solve = _solve_node_pair!(
            K_solver,
            work_ba_col,
            core.bus_to_valid_idx,
            basis.p[j],
            basis.q[j],
            basis.scale[j],
        )
        _gather_to_buses!(view(Z, :, j), core.valid_ix, lin_solve)
    end
    return Z
end

"""
    _base_network_factors(core, modifications, labeler, columns, arc_columns) -> WoodburyFactors

Woodbury factors of `modifications` on the base network over its minimal-rank basis, built
from the island labels `labeler(BA, arc_sus, modifications, n_bus)` of the modified network.
`columns(basis)` gives `B⁻¹U` for a basis and `arc_columns()` the arc-form `B⁻¹U` of every
modification, used when the node-pair basis is singular.
"""
function _base_network_factors(
    core::FactorCore,
    modifications::Vector{ArcModification},
    labeler::L,
    columns::C,
    arc_columns::A,
)::WoodburyFactors where {L, C, A}
    BA = core.BA
    signs = core.arc_susceptance_signs
    arc_sus = core.arc_susceptances
    _check_changed_arcs_have_susceptance(arc_sus, modifications)
    post_labels = labeler(BA, arc_sus, modifications, size(BA, 1))
    basis = _woodbury_basis(core, ArcModification[], Int[], modifications, post_labels)
    arc_out = _arcs_fully_opened(core, modifications)
    ref_positions = _ref_bus_positions(core)
    arc_fallback =
        () -> _arc_form_factors(
            arc_columns(),
            _arc_form_basis(core, modifications),
            BA,
            signs,
            arc_sus,
            modifications,
            arc_out,
            post_labels,
            ref_positions,
        )
    return _woodbury_factors_from_basis(
        columns(basis),
        basis,
        BA,
        signs,
        arc_sus,
        modifications,
        arc_out,
        post_labels,
        ref_positions,
        arc_fallback,
    )
end

"""
    _compute_woodbury_factors(core, modifications[, labeler]) -> WoodburyFactors

Woodbury factors of `modifications` over their minimal-rank basis, solved on `core`'s
factorization. The island labels of the modified network come from
`labeler(BA, arc_sus, modifications, n_bus)`. Solves and the scratch slot go through
`with_solver`, which holds the core's `solver_lock`.
"""
function _compute_woodbury_factors(
    core::FactorCore,
    modifications::Vector{ArcModification},
    labeler::F = _post_contingency_bus_labels,
)::WoodburyFactors where {F}
    _check_arc_indices(core, modifications)
    return with_solver(
        core.K,
        core.work_ba_col,
        core.temp_data,
        core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        n_bus = length(temp_data)
        return _base_network_factors(
            core,
            modifications,
            labeler,
            basis ->
                _basis_columns(core, K_solver, work_ba_col, n_bus, basis, modifications),
            () -> _base_woodbury_columns(core, K_solver, work_ba_col, n_bus, modifications),
        )
    end
end

# The row of the monitored arc on the network whose base-network factors are `wf`, judged on
# the modification `wf` was assembled for.
function _apply_woodbury_correction(
    core::FactorCore,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    plan = _base_topology_row_plan(core, monitored_idx, wf, wf)
    return only(_rows_from_plans(core, monitored_idx, (plan,)))
end

"""
    _outside_reference_islands(core, wf, m) -> Bool

Whether monitored arc `m` lies outside the islands holding a reference bus on the network whose
factors are `wf`: `wf` islands and the label of `m`'s from bus is no reference bus's label.
"""
function _outside_reference_islands(
    core::FactorCore,
    wf::WoodburyFactors,
    m::Int,
)::Bool
    wf.is_islanding || return false
    labels = wf.bus_island_labels
    label = labels[first(_arc_ends(core, m))]
    n_ref = length(core.bus_to_valid_idx) - length(core.valid_ix)
    for j in 1:n_ref
        labels[_reference_position(core.valid_ix, j)] == label && return false
    end
    return true
end

# Position of the `j`-th reference bus in ascending order, without allocating. The ascending
# valid positions `valid` skip exactly the reference positions, so `valid[k] - k` reference
# positions precede `valid[k]`; the `j`-th one follows the `k - 1` valid positions before the
# first `k` with `valid[k] - k >= j`.
function _reference_position(valid::Vector{Int}, j::Int)::Int
    lo, hi = 1, length(valid) + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        if valid[mid] - mid >= j
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo - 1 + j
end

# --- Row plans: the one path from a monitored arc's base solve to its post-modification rows ---

@enum _RowKind _ZERO_ROW _BASE_ROW _CORRECTED_ROW

"""
    _RowPlan

How one row is computed from the monitored arc's base solve. `_ZERO_ROW` is all zeros and
`_BASE_ROW` the base solve itself. `_CORRECTED_ROW` is the base solve corrected by the
Woodbury factors `correction` and scaled by `b_post`.
"""
struct _RowPlan
    kind::_RowKind
    correction::Union{Nothing, WoodburyFactors}
    b_post::Float64
end

_RowPlan(kind::_RowKind) = _RowPlan(kind, nothing, 0.0)

"""
    _base_topology_row_plan(core, m, post, wf) -> _RowPlan

Plan of monitored arc `m`'s row on the network the modification `post` takes the base network
to, with `wf` its Woodbury factors on the base network, or `nothing` for the base network
itself. The arc is judged out of service, and its susceptance taken, on `post`. An arc out of
service has the zero row, including an arc with no susceptance on the base network, whose
`BA` column is empty, and so has an arc outside the islands of `post` that hold a reference
bus. Factors with no columns, such as those of a shunt-only modification, still take the
correction, which scales the base solve by `b_post / b_pre`.
"""
function _base_topology_row_plan(
    core::FactorCore,
    m::Int,
    post::_PostModification,
    wf::Union{Nothing, WoodburyFactors},
)::_RowPlan
    arc_sus = core.arc_susceptances
    _post_modification_arc_out(arc_sus, core, m, post) && return _RowPlan(_ZERO_ROW)
    isnothing(wf) && return _RowPlan(_BASE_ROW)
    _outside_reference_islands(core, wf, m) && return _RowPlan(_ZERO_ROW)
    return _RowPlan(_CORRECTED_ROW, wf, _post_modification_susceptance(arc_sus, m, post))
end

"""
    _rows_from_plans(core, m, plans) -> Vector{Vector{Float64}}

Rows of monitored arc `m`, one per plan, from one solve of its base column on `core`, before
any cutoff. No solve runs when every plan is a zero row. Every factor must be resolved before
this call, since computing factors runs solves of its own.
"""
function _rows_from_plans(core::FactorCore, m::Int, plans)::Vector{Vector{Float64}}
    needs_base = any(p -> p.kind == _BASE_ROW, plans)
    needs_scaled = any(p -> p.kind == _CORRECTED_ROW, plans)
    b_mon_pre = core.arc_susceptances[m]
    base_row, z_base = with_solver(
        core.K,
        core.work_ba_col,
        core.temp_data,
        core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        n_bus = length(temp_data)
        base_row = Float64[]
        z_base = Float64[]
        if needs_base || needs_scaled
            lin_solve = _solve_ba_column!(
                K_solver,
                work_ba_col,
                core.BA,
                core.bus_to_valid_idx,
                m,
            )
            if needs_base
                base_row = _gather_to_buses!(
                    Vector{Float64}(undef, n_bus),
                    core.valid_ix,
                    lin_solve,
                )
            end
            # z_base = B⁻¹ν_m, ν_m normalized by the base susceptance.
            if needs_scaled
                z_base = _gather_to_buses!(
                    Vector{Float64}(undef, n_bus),
                    core.valid_ix,
                    lin_solve,
                    b_mon_pre,
                )
            end
        end
        return base_row, z_base
    end
    return _evaluate_row_plans(core, m, plans, base_row, z_base)
end

"""
    _evaluate_row_plans(core, m, plans, base_row, z_base) -> Vector{Vector{Float64}}

Rows of monitored arc `m`, one per plan, from its base row `base_row` and `z_base = B⁻¹ν_m`
(`ν_m` normalized by the base susceptance). This is the only place a row is zeroed for an arc
out of service or scaled by its post-modification susceptance. The first base-row plan takes
`base_row` itself; every other row is a fresh vector.
"""
function _evaluate_row_plans(
    core::FactorCore,
    m::Int,
    plans,
    base_row::Vector{Float64},
    z_base::Vector{Float64},
)::Vector{Vector{Float64}}
    max_modified = 0
    for p in plans
        isnothing(p.correction) ||
            (max_modified = max(max_modified, _woodbury_rank(p.correction)))
    end
    zm_Z_buf = Vector{Float64}(undef, max_modified)
    coeff_buf = Vector{Float64}(undef, max_modified)
    b_mon_pre = core.arc_susceptances[m]
    n_bus = size(core.BA, 1)
    correct(z, b_post, wf) = _woodbury_correction!(
        copy(z), zm_Z_buf, coeff_buf, core.BA, b_mon_pre, b_post, m, wf,
    )
    rows = Vector{Vector{Float64}}(undef, length(plans))
    base_used = false
    for (i, plan) in enumerate(plans)
        if plan.kind == _ZERO_ROW
            rows[i] = zeros(n_bus)
        elseif plan.kind == _BASE_ROW
            rows[i] = base_used ? copy(base_row) : base_row
            base_used = true
        else
            rows[i] = correct(z_base, plan.b_post, plan.correction)
        end
    end
    return rows
end
