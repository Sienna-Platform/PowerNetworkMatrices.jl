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
        # M = 0 (no arc changes its susceptance): getri! rejects 0×0.
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

Only arcs whose DC susceptance changes enter the factors. An arc with `delta_b == 0`, such as
an arc with no susceptance, or a parallel group whose only remaining member has zero
reactance when that member opens, leaves `B` unchanged, and its `1/Δb` term in `W` would be
infinite. Its column of `Z` and its entry of `arc_out` are dropped; the labeler still sees
every modification.
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
    n_bus = size(Z, 1)
    kept = [j for (j, mod) in enumerate(modifications) if !iszero(mod.delta_b)]
    M = length(kept)
    if M < length(modifications)
        Z = Z[:, kept]
        arc_out = arc_out[kept]
    end

    arc_indices = Vector{Int}(undef, M)
    delta_b_vec = Vector{Float64}(undef, M)
    for (j, k) in enumerate(kept)
        arc_indices[j] = modifications[k].arc_index
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

# True when `m` opens every member of its arc. Counts members instead of comparing
# susceptances, because the group susceptance is not the exact sum of its members'.
_opens_every_member(nr::NetworkReductionData, arc_ax, m::ArcModification) =
    m.opened > 0 && m.opened >= _arc_member_count(nr, arc_ax[m.arc_index])

# Per modified arc: true when the modification opens every member.
function _arcs_fully_opened(
    core::VirtualFactorCore,
    modifications::Tuple{Vararg{ArcModification}},
)::Vector{Bool}
    nr = get_network_reduction_data(core)
    arc_ax = get_arc_axis(core)
    return Bool[_opens_every_member(nr, arc_ax, m) for m in modifications]
end

# The entry of `modifications` on arc `arc_index`, or `nothing`. A `NetworkModification` keeps
# its arc modifications sorted by arc index, so this is a binary search.
function _arc_modification_at(
    modifications::Tuple{Vararg{ArcModification}},
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

# The modification a row is judged on: the one taking the base network to the row's network
# (K ∪ c for a post-contingency row), as its arc modifications or as the Woodbury factors
# assembled for it on the base network.
const _PostModification = Union{Tuple{Vararg{ArcModification}}, WoodburyFactors}

# Susceptance change of arc `m` under `post`, or `nothing` when `post` leaves it unchanged.
function _delta_b_at(post::Tuple{Vararg{ArcModification}}, m::Int)::Union{Nothing, Float64}
    mod = _arc_modification_at(post, m)
    return isnothing(mod) ? nothing : mod.delta_b
end

function _delta_b_at(wf::WoodburyFactors, m::Int)::Union{Nothing, Float64}
    j = findfirst(==(m), wf.arc_indices)
    return isnothing(j) ? nothing : wf.delta_b[j]
end

# True when `post` opens every member of arc `m`.
function _opens_every_member_at(
    core::VirtualFactorCore,
    post::Tuple{Vararg{ArcModification}},
    m::Int,
)::Bool
    mod = _arc_modification_at(post, m)
    isnothing(mod) && return false
    return _opens_every_member(get_network_reduction_data(core), get_arc_axis(core), mod)
end

function _opens_every_member_at(::VirtualFactorCore, wf::WoodburyFactors, m::Int)::Bool
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
    core::VirtualFactorCore,
    m::Int,
    post::_PostModification,
)::Bool
    _opens_every_member_at(core, post, m) && return true
    return iszero(_post_modification_susceptance(arc_sus, m, post))
end

# Susceptance of arc `m` on K ∪ c. The tier-2 modification alone would miss the change K
# already made to a partly out arc.
_post_contingency_susceptance(
    arc_sus::Vector{Float64},
    m::Int,
    net::PostContingencyNetwork,
) = _post_modification_susceptance(arc_sus, m, net.post_contingency.arc_modifications)

# Whether arc `m` is out of service on K ∪ c.
_post_contingency_arc_out(
    arc_sus::Vector{Float64},
    core::VirtualFactorCore,
    m::Int,
    net::PostContingencyNetwork,
) = _post_modification_arc_out(arc_sus, core, m, net.post_contingency.arc_modifications)

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

# Column `j` of the result is `B⁻¹ν_j` for the `j`-th modified arc, in full-bus space, from
# one base solve per arc. Call inside `with_solver`, which owns `K_solver` and `work_ba_col`.
function _base_woodbury_columns(
    core::VirtualFactorCore,
    K_solver,
    work_ba_col::Vector{Float64},
    n_bus::Int,
    modifications::Tuple{Vararg{ArcModification}},
)::Matrix{Float64}
    Z = Matrix{Float64}(undef, n_bus, length(modifications))
    for (j, mod) in enumerate(modifications)
        e = mod.arc_index
        lin_solve =
            _solve_ba_column!(K_solver, work_ba_col, core.BA, core.bus_to_valid_idx, e)
        _gather_to_buses!(view(Z, :, j), core.valid_ix, lin_solve, core.arc_susceptances[e])
    end
    return Z
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
        return _woodbury_factors_from_Z(
            _base_woodbury_columns(
                core,
                K_solver,
                work_ba_col,
                length(temp_data),
                modifications,
            ),
            core.BA,
            core.arc_susceptance_signs,
            core.arc_susceptances,
            modifications,
            _arcs_fully_opened(core, modifications),
            labeler,
        )
    end
end

"""
    _tier2_factors(core, topology_wf, net) -> WoodburyFactors

Woodbury factors of the tier-2 modification `net.tier2` on a topology K other than the base
network, with no solve beyond one base solve per arc Δ modifies. Column `j` of `Z` is
`B_K⁻¹ν_j` for the `j`-th such arc: its base solve corrected by the topology's factors
`topology_wf` with the final scaling left at 1. `ν_j` keeps the base normalization, so K
enters only through `Z`. Whether each arc is out and the island labels are taken on K ∪ c,
from `net.post_contingency`.
"""
function _tier2_factors(
    core::VirtualFactorCore,
    topology_wf::WoodburyFactors,
    net::PostContingencyNetwork,
)::WoodburyFactors
    modifications = net.tier2.arc_modifications
    arc_sus = core.arc_susceptances
    BA = core.BA
    Z = with_solver(
        core.K,
        core.work_ba_col,
        core.temp_data,
        core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        return _base_woodbury_columns(
            core,
            K_solver,
            work_ba_col,
            length(temp_data),
            modifications,
        )
    end
    column = Vector{Float64}(undef, size(Z, 1))
    for (j, mod) in enumerate(modifications)
        e = mod.arc_index
        copyto!(column, view(Z, :, j))
        _woodbury_correction!(column, BA, arc_sus[e], 1.0, e, topology_wf)
        Z[:, j] .= column
    end
    post_modifications = net.post_contingency.arc_modifications
    arc_out = Bool[
        _opens_every_member_at(core, post_modifications, mod.arc_index) for
        mod in modifications
    ]
    labeler =
        (BA_, arc_sus_, _, n_bus) ->
            _post_contingency_bus_labels(BA_, arc_sus_, post_modifications, n_bus)
    return _woodbury_factors_from_Z(
        Z,
        BA,
        core.arc_susceptance_signs,
        arc_sus,
        modifications,
        arc_out,
        labeler,
    )
end

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

# The row of the monitored arc on the network whose base-network factors are `wf`, judged on
# the modification `wf` was assembled for.
function _apply_woodbury_correction(
    core::VirtualFactorCore,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    plan = _base_topology_row_plan(core, monitored_idx, wf, wf)
    return only(_rows_from_plans(core, monitored_idx, (plan,)))
end

# --- Row plans: the one path from a monitored arc's base solve to its post-modification rows ---

@enum _RowKind _ZERO_ROW _BASE_ROW _TOPOLOGY_ROW _TIER2_ROW

# How one row is computed from the monitored arc's base solve. `_TOPOLOGY_ROW` is the tier-1
# row on K with factors `topology` and susceptance `b_post`; `_TIER2_ROW` corrects `B_K⁻¹ν_m`
# (the base solve itself when `topology` is `nothing`) with the factors `tier2` and scales by
# `b_post`.
struct _RowPlan
    kind::_RowKind
    topology::Union{Nothing, WoodburyFactors}
    tier2::Union{Nothing, WoodburyFactors}
    b_post::Float64
end

_RowPlan(kind::_RowKind) = _RowPlan(kind, nothing, nothing, 0.0)

"""
    _base_topology_row_plan(core, m, post, wf) -> _RowPlan

Plan of monitored arc `m`'s row on the network the modification `post` takes the base network
to, with `wf` its Woodbury factors on the base network, or `nothing` for the base network
itself. The arc is judged out of service, and its susceptance taken, on `post`. An arc out of
service has the zero row, including an arc with no susceptance on the base network, whose
`BA` column is empty. Factors with no columns, such as those of a shunt-only modification,
still take the correction, which scales the base solve by `b_post / b_pre`.
"""
function _base_topology_row_plan(
    core::VirtualFactorCore,
    m::Int,
    post::_PostModification,
    wf::Union{Nothing, WoodburyFactors},
)::_RowPlan
    arc_sus = core.arc_susceptances
    _post_modification_arc_out(arc_sus, core, m, post) && return _RowPlan(_ZERO_ROW)
    isnothing(wf) && return _RowPlan(_BASE_ROW)
    return _RowPlan(
        _TIER2_ROW,
        nothing,
        wf,
        _post_modification_susceptance(arc_sus, m, post),
    )
end

"""
    _rows_from_plans(core, m, plans) -> Vector{Vector{Float64}}

Rows of monitored arc `m`, one per plan, from one solve of its base column on `core`, before
any cutoff. No solve runs when every plan is a zero row. Every factor must be resolved before
this call, since computing factors runs solves of its own.
"""
function _rows_from_plans(core::VirtualFactorCore, m::Int, plans)::Vector{Vector{Float64}}
    needs_base = any(p -> p.kind == _BASE_ROW, plans)
    needs_scaled = any(p -> p.kind == _TOPOLOGY_ROW || p.kind == _TIER2_ROW, plans)
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
    core::VirtualFactorCore,
    m::Int,
    plans,
    base_row::Vector{Float64},
    z_base::Vector{Float64},
)::Vector{Vector{Float64}}
    max_modified = 0
    for p in plans, wf in (p.topology, p.tier2)
        isnothing(wf) || (max_modified = max(max_modified, length(wf.arc_indices)))
    end
    zm_Z_buf = Vector{Float64}(undef, max_modified)
    coeff_buf = Vector{Float64}(undef, max_modified)
    b_mon_pre = core.arc_susceptances[m]
    n_bus = size(core.BA, 1)
    correct(z, b_post, wf) = _woodbury_correction!(
        copy(z), zm_Z_buf, coeff_buf, core.BA, b_mon_pre, b_post, m, wf,
    )
    # B_K⁻¹ν_m per distinct topology; equal topologies share one cached factors object.
    z_topology = IdDict{WoodburyFactors, Vector{Float64}}()
    rows = Vector{Vector{Float64}}(undef, length(plans))
    base_used = false
    for (i, plan) in enumerate(plans)
        if plan.kind == _ZERO_ROW
            rows[i] = zeros(n_bus)
        elseif plan.kind == _BASE_ROW
            rows[i] = base_used ? copy(base_row) : base_row
            base_used = true
        elseif plan.kind == _TOPOLOGY_ROW
            rows[i] = correct(z_base, plan.b_post, plan.topology)
        else
            topology_wf = plan.topology
            z_k = if isnothing(topology_wf)
                z_base
            else
                get!(() -> correct(z_base, 1.0, topology_wf), z_topology, topology_wf)
            end
            rows[i] = correct(z_k, plan.b_post, plan.tier2)
        end
    end
    return rows
end
