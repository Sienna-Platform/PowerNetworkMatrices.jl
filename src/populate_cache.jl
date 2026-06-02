# Bulk cache population for the lazy virtual network matrices.
#
# `getindex` on a `VirtualPTDF` / `VirtualLODF` / `VirtualMODF` computes one row
# at a time, issuing a single right-hand-side (RHS) linear solve per row through
# the ABA factorization. When the optimization-problem build queries many rows
# (every monitored branch, every contingency), those solves dominate.
#
# `populate_cache` amortizes that cost: it gathers the requested rows, builds a
# single sparse RHS matrix of the corresponding `BA` columns, and solves them
# all in one batched call via the backend's `solve_sparse!` (KLU or Apple
# Accelerate). A batched solve reuses the factorization's working set across
# columns and issues one libklu/libSparse call per block of 64 instead of one
# per row, which is markedly faster on large systems. The resulting rows are
# pinned in the row cache so later `getindex` calls are pure cache hits.

# --- Backend dispatch: batched (multi-RHS) sparse solve --------------------

# KLU's `solve_sparse!` is imported unqualified into PNM; Apple Accelerate's is
# reached through the `AccelerateWrapper` module (mirrors `ptdf_calculations.jl`).
_solve_multi_rhs!(
    K::KLULinSolveCache{Float64},
    B::SparseArrays.SparseMatrixCSC{Float64, Int},
    out::Matrix{Float64},
) = solve_sparse!(K, B, out)

_solve_multi_rhs!(
    K::AAFactorCache,
    B::SparseArrays.SparseMatrixCSC{Float64, Int},
    out::Matrix{Float64},
) = AccelerateWrapper.solve_sparse!(K, B, out)

# Generic fallback for any other factorization backend: solve column-by-column.
function _solve_multi_rhs!(K, B::SparseArrays.SparseMatrixCSC{Float64, Int}, out::Matrix{Float64})
    n = size(B, 1)
    col = zeros(n)
    @inbounds for j in axes(B, 2)
        fill!(col, 0.0)
        for p in SparseArrays.nzrange(B, j)
            col[SparseArrays.rowvals(B)[p]] = SparseArrays.nonzeros(B)[p]
        end
        _solve_factorization(K, col)
        copyto!(view(out, :, j), col)
    end
    return out
end

"""
    _solve_arc_columns(mat, arc_rows) -> Matrix{Float64}

Solve `ABA · X = BA[valid_ix, arc_rows]` for all requested arcs at once,
returning the `(n_valid × length(arc_rows))` dense solution. The caller must
hold `mat.solver_lock` (the batched solve mutates the factorization's scratch).
"""
function _solve_arc_columns(mat, arc_rows::Vector{Int})
    valid_ix = mat.valid_ix
    B = mat.BA[valid_ix, arc_rows]
    out = Matrix{Float64}(undef, length(valid_ix), length(arc_rows))
    _solve_multi_rhs!(mat.K, B, out)
    return out
end

# --- Component resolution --------------------------------------------------

# Resolve a user-supplied component to an integer arc/row index. Accepts the
# same identifiers as `getindex`: an integer index, an arc bus-pair tuple, or a
# branch name.
_resolve_arc_index(::PowerNetworkMatrix, c::Integer) = Int(c)
_resolve_arc_index(mat::PowerNetworkMatrix, c::Tuple{Int, Int}) = get_arc_lookup(mat)[c]
function _resolve_arc_index(mat::PowerNetworkMatrix, c::AbstractString)
    _, arc = get_branch_multiplier(mat, String(c))
    return get_arc_lookup(mat)[arc]
end

# --- VirtualPTDF -----------------------------------------------------------

"""
    populate_cache(vptdf::VirtualPTDF, components) -> Nothing

Precompute and pin the PTDF rows for an iterable of `components`, using a single
batched multi-RHS solve instead of one solve per row. `components` may mix
integer arc indices, arc bus-pair tuples `(from, to)`, and branch-name strings.

Populated rows are added to the cache's persistent set, so subsequent
`vptdf[component, :]` queries are cache hits and are never evicted by later lazy
lookups. Use this before an optimization-problem build to amortize the linear
solves over the monitored branch set.

$(TYPEDSIGNATURES)
"""
function populate_cache(vptdf::VirtualPTDF, components)
    rows = unique(Int[_resolve_arc_index(vptdf, c) for c in components])
    isempty(rows) && return nothing

    buscount = size(vptdf, 1)
    ref_bus_positions = get_ref_bus_position(vptdf)
    if !isempty(vptdf.dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    end
    use_dist_slack = length(vptdf.dist_slack) == buscount
    if !use_dist_slack && !isempty(vptdf.dist_slack)
        error("Distributed bus specification doesn't match the number of buses.")
    end
    tol = get_tol(vptdf)

    # Only solve rows not already resident; existing rows are pinned below.
    new_rows = @lock vptdf.cache_lock Int[r for r in rows if !haskey(vptdf.cache, r)]

    if !isempty(new_rows)
        sol = @lock vptdf.solver_lock _solve_arc_columns(vptdf, new_rows)
        valid_ix = vptdf.valid_ix
        @lock vptdf.cache_lock begin
            full = zeros(buscount)
            for (j, r) in enumerate(new_rows)
                haskey(vptdf.cache, r) && continue  # lost a race; keep the winner
                fill!(full, 0.0)
                @inbounds for i in eachindex(valid_ix)
                    full[valid_ix[i]] = sol[i, j]
                end
                if use_dist_slack
                    full .-= dot(full, vptdf.dist_slack_normalized)
                end
                stored = tol > eps() ? sparsify(full, tol) : copy(full)
                set_persistent_row!(vptdf.cache, r, stored)
            end
        end
    end

    @lock vptdf.cache_lock begin
        for r in rows
            pin_row!(vptdf.cache, r)
        end
        warn_if_over_capacity(vptdf.cache)
    end
    return nothing
end

# --- VirtualLODF -----------------------------------------------------------

"""
    populate_cache(vlodf::VirtualLODF, components) -> Nothing

Precompute and pin the LODF rows for an iterable of `components` (integer arc
indices, arc bus-pair tuples, or branch-name strings) using a single batched
multi-RHS solve. The post-contingency scaling `(A · B⁻¹ · BA) .* inv_PTDF_A_diag`
is applied to all requested rows at once via one sparse-dense product.

Populated rows are pinned in the cache so later `vlodf[component, :]` queries
are cache hits.

$(TYPEDSIGNATURES)
"""
function populate_cache(vlodf::VirtualLODF, components)
    rows = unique(Int[_resolve_arc_index(vlodf, c) for c in components])
    isempty(rows) && return nothing
    tol = get_tol(vlodf)
    n_bus = length(vlodf.temp_data[1])

    new_rows = @lock vlodf.cache_lock Int[r for r in rows if !haskey(vlodf.cache, r)]

    if !isempty(new_rows)
        sol = @lock vlodf.solver_lock _solve_arc_columns(vlodf, new_rows)
        valid_ix = vlodf.valid_ix
        # Scatter each solved column back to full-bus space, then apply the
        # LODF map to every column at once: L = (A · Tmp) .* inv_PTDF_A_diag.
        tmp = zeros(n_bus, length(new_rows))
        @inbounds for j in eachindex(new_rows), i in eachindex(valid_ix)
            tmp[valid_ix[i], j] = sol[i, j]
        end
        lodf_cols = vlodf.A * tmp                  # (n_arcs × length(new_rows))
        lodf_cols .*= vlodf.inv_PTDF_A_diag        # broadcast per-arc scaling down columns
        @inbounds for (j, r) in enumerate(new_rows)
            lodf_cols[r, j] = -1.0                  # self-element convention
        end
        @lock vlodf.cache_lock begin
            for (j, r) in enumerate(new_rows)
                haskey(vlodf.cache, r) && continue
                row = lodf_cols[:, j]
                stored = tol > eps() ? sparsify(row, tol) : row
                set_persistent_row!(vlodf.cache, r, stored)
            end
        end
    end

    @lock vlodf.cache_lock begin
        for r in rows
            pin_row!(vlodf.cache, r)
        end
        warn_if_over_capacity(vlodf.cache)
    end
    return nothing
end

# --- VirtualMODF -----------------------------------------------------------

# Resolve a contingency identifier to the `NetworkModification` used as the
# cache key. Accepts a modification, a `ContingencySpec`, a registered
# `PSY.Outage`, or a registered outage UUID.
_resolve_modification(::VirtualMODF, mod::NetworkModification) = mod
_resolve_modification(::VirtualMODF, ctg::ContingencySpec) = ctg.modification
function _resolve_modification(vmodf::VirtualMODF, outage::PSY.Outage)
    return _resolve_modification(vmodf, IS.get_uuid(outage))
end
function _resolve_modification(vmodf::VirtualMODF, uuid::Base.UUID)
    haskey(vmodf.contingency_cache, uuid) || error(
        "Contingency (UUID=$uuid) is not registered. Construct the VirtualMODF " *
        "with the system containing this outage, or pass the NetworkModification " *
        "/ ContingencySpec directly.",
    )
    return vmodf.contingency_cache[uuid].modification
end

"""
    _woodbury_factors_from_base(base_full, BA, arc_sus, modifications, n_bus) -> WoodburyFactors

Assemble Woodbury factors from precomputed pre-contingency solves. `base_full`
maps each modified arc index to `B⁻¹ · BA[:, arc]` scattered to full-bus space.
Identical math to `_compute_woodbury_factors_impl` but with the per-arc libklu
solves replaced by dictionary lookups (the batched solve already paid for them).
"""
function _woodbury_factors_from_base(
    base_full::Dict{Int, Vector{Float64}},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    modifications::Tuple{Vararg{ArcModification}},
    n_bus::Int,
)::WoodburyFactors
    M = length(modifications)
    arc_indices = Vector{Int}(undef, M)
    delta_b_vec = Vector{Float64}(undef, M)
    for (j, mod) in enumerate(modifications)
        arc_indices[j] = mod.arc_index
        delta_b_vec[j] = mod.delta_b
    end

    # Z[:, j] = B⁻¹ ν_j = (B⁻¹ BA[:, e_j]) / b_{e_j}
    Z = Matrix{Float64}(undef, n_bus, M)
    for j in 1:M
        b_e = arc_sus[arc_indices[j]]
        col = base_full[arc_indices[j]]
        @inbounds for i in 1:n_bus
            Z[i, j] = col[i] / b_e
        end
    end

    # K_mat[i, j] = ν_i⊤ B⁻¹ ν_j, iterating the sparse BA columns of arc e_i.
    ba_nzv = SparseArrays.nonzeros(BA)
    ba_rv = SparseArrays.rowvals(BA)
    K_mat = zeros(M, M)
    for i in 1:M
        e_i = arc_indices[i]
        b_i = arc_sus[e_i]
        for j in 1:M
            val = 0.0
            @inbounds for nz_idx in SparseArrays.nzrange(BA, e_i)
                val += (ba_nzv[nz_idx] / b_i) * Z[ba_rv[nz_idx], j]
            end
            K_mat[i, j] = val
        end
    end

    W_mat = LinearAlgebra.diagm(1.0 ./ delta_b_vec) + K_mat
    W_inv, is_island = _invert_woodbury_W(W_mat, Val(M))
    return WoodburyFactors(Z, W_inv, arc_indices, delta_b_vec, is_island)
end

"""
    _woodbury_correction_from_base(base_full, BA, arc_sus, monitored_idx, wf, n_bus) -> Vector{Float64}

Post-modification PTDF row for `monitored_idx` from precomputed pre-contingency
solves. Mirrors `_apply_woodbury_correction_impl`, reusing `base_full[monitored_idx]`
(contingency-independent) instead of solving.
"""
function _woodbury_correction_from_base(
    base_full::Dict{Int, Vector{Float64}},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    monitored_idx::Int,
    wf::WoodburyFactors,
    n_bus::Int,
)::Vector{Float64}
    M = length(wf.arc_indices)

    b_mon = arc_sus[monitored_idx]
    for (j, idx) in enumerate(wf.arc_indices)
        idx == monitored_idx && (b_mon += wf.delta_b[j])
    end
    abs(b_mon) < eps() && return zeros(n_bus)

    b_mon_pre = arc_sus[monitored_idx]
    temp = base_full[monitored_idx] ./ b_mon_pre   # z_m (fresh vector; base_full untouched)

    ba_nzv = SparseArrays.nonzeros(BA)
    ba_rv = SparseArrays.rowvals(BA)
    zm_Z = zeros(M)
    @inbounds for nz_idx in SparseArrays.nzrange(BA, monitored_idx)
        coeff = ba_nzv[nz_idx] / b_mon_pre
        row = ba_rv[nz_idx]
        for j in 1:M
            zm_Z[j] += coeff * wf.Z[row, j]
        end
    end

    correction_coeff = wf.W_inv * zm_Z
    LinearAlgebra.mul!(temp, wf.Z, correction_coeff, -1.0, 1.0)
    temp .*= b_mon
    return temp
end

"""
    populate_cache(vmodf::VirtualMODF, contingencies; monitored) -> Nothing

Precompute and pin the post-contingency PTDF rows for an iterable of
`contingencies`, each evaluated over the user-supplied `monitored` arc set, using
batched multi-RHS solves.

`contingencies` may mix `NetworkModification`, `ContingencySpec`, registered
`PSY.Outage`, and registered outage UUIDs. `monitored` is an iterable of arc
identifiers (integer indices, bus-pair tuples, or branch names).

The acceleration comes from a single batched solve over the union of all arcs
referenced — every contingency's modified arcs plus every monitored arc. Because
the pre-contingency solve for a monitored arc is contingency-independent, it is
computed once and reused across all contingencies; the Woodbury factors per
contingency are then assembled from these precomputed solves with no further
linear solves. This replaces the `O(n_contingencies × (M + n_monitored))`
one-at-a-time solves of repeated `getindex` with one batched solve over the
distinct arcs.

Results are written into the per-contingency row caches (and Woodbury cache) and
pinned, so subsequent `vmodf[monitored, contingency]` queries are cache hits.

$(TYPEDSIGNATURES)
"""
function populate_cache(vmodf::VirtualMODF, contingencies; monitored)
    mods = unique(NetworkModification[_resolve_modification(vmodf, c) for c in contingencies])
    mon_idx = unique(Int[_resolve_arc_index(vmodf, m) for m in monitored])
    (isempty(mods) || isempty(mon_idx)) && return nothing

    n_bus = length(vmodf.temp_data[1])
    tol = get_tol(vmodf)
    BA = vmodf.BA
    arc_sus = vmodf.arc_susceptances

    @lock vmodf.solver_lock begin
        # Union of all arcs needing a pre-contingency solve: monitored arcs and
        # every contingency's modified arcs.
        arc_set = Set{Int}(mon_idx)
        for mod in mods
            for am in mod.arc_modifications
                push!(arc_set, am.arc_index)
            end
        end
        all_arcs = collect(arc_set)

        # One batched solve for B⁻¹ BA[:, arc] over the distinct arcs.
        sol = _solve_arc_columns(vmodf, all_arcs)
        valid_ix = vmodf.valid_ix
        base_full = Dict{Int, Vector{Float64}}()
        sizehint!(base_full, length(all_arcs))
        for (j, arc) in enumerate(all_arcs)
            full = zeros(n_bus)
            @inbounds for i in eachindex(valid_ix)
                full[valid_ix[i]] = sol[i, j]
            end
            base_full[arc] = full
        end

        for mod in mods
            wf = get!(vmodf.woodbury_cache, mod) do
                _woodbury_factors_from_base(base_full, BA, arc_sus, mod.arc_modifications, n_bus)
            end
            rc = get!(vmodf.row_caches, mod) do
                RowCache(vmodf.max_cache_size_bytes, Set{Int}(), n_bus * sizeof(Float64))
            end
            for m in mon_idx
                if haskey(rc, m)
                    pin_row!(rc, m)
                    continue
                end
                row = _woodbury_correction_from_base(base_full, BA, arc_sus, m, wf, n_bus)
                stored = tol > eps() ? sparsify(row, tol) : row
                set_persistent_row!(rc, m, stored)
            end
            warn_if_over_capacity(rc)
        end
    end
    return nothing
end
