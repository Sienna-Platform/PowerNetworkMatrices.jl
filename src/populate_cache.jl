# Bulk cache population for the lazy virtual network matrices.
#
# `getindex` on a `VirtualPTDF` / `VirtualLODF` / `VirtualMODF` computes one row
# at a time, issuing a single right-hand-side (RHS) linear solve per row through
# the shared `VirtualFactorCore` ABA factorization. When the optimization-problem
# build queries many rows (every monitored branch, every contingency), those
# solves dominate.
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

# Generic fallback for any other factorization backend: solve column-by-column
# through the single-RHS `_solve_factorization` seam (the documented extension
# point for new solver backends). Errors clearly when the backend implements
# neither a batched `solve_sparse!` nor `_solve_factorization`, instead of
# surfacing a raw MethodError.
function _solve_multi_rhs!(
    K,
    B::SparseArrays.SparseMatrixCSC{Float64, Int},
    out::Matrix{Float64},
)
    n = size(B, 1)
    col = zeros(n)
    applicable(_solve_factorization, K, col) || error(
        "Factorization backend $(typeof(K)) supports neither a batched " *
        "`solve_sparse!` nor a single-RHS `_solve_factorization`; extend one of " *
        "them to use `populate_cache` with this backend.",
    )
    @inbounds for j in axes(B, 2)
        fill!(col, 0.0)
        for p in SparseArrays.nzrange(B, j)
            col[SparseArrays.rowvals(B)[p]] = SparseArrays.nonzeros(B)[p]
        end
        # Capture the return: KLU/Accelerate solve in place and return `col`,
        # but a future backend may return a fresh vector, so read the result.
        result = _solve_factorization(K, col)
        copyto!(view(out, :, j), result)
    end
    return out
end

"""
    _solve_arc_columns(core, arc_rows) -> Matrix{Float64}

Solve `ABA · X = BA[valid_ix, arc_rows]` for all requested arcs at once,
returning the `(n_valid × length(arc_rows))` dense solution. The caller must
hold `core.solver_lock` (the batched solve mutates the factorization's scratch).
"""
function _solve_arc_columns(core::VirtualFactorCore, arc_rows::Vector{Int})
    valid_ix = core.valid_ix
    B = core.BA[valid_ix, arc_rows]
    out = Matrix{Float64}(undef, length(valid_ix), length(arc_rows))
    _solve_multi_rhs!(core.K, B, out)
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
solves over the monitored branch set. Pinned rows count against the cache
capacity, so populating more rows than `max_cache_size` holds is an error.

$(TYPEDSIGNATURES)
"""
function populate_cache(vptdf::VirtualPTDF, components)
    rows = unique(Int[_resolve_arc_index(vptdf, c) for c in components])
    isempty(rows) && return nothing

    core = get_core(vptdf)
    cache = get_cache(vptdf)
    cache_lock = get_cache_lock(vptdf)
    dist_slack_normalized = get_dist_slack_normalized(vptdf)
    buscount = size(core.BA, 1)
    use_dist_slack = _use_dist_slack(vptdf)
    cutoff = get_cutoff(core)

    # Only solve rows not already resident; existing rows are pinned below.
    new_rows = @lock cache_lock Int[r for r in rows if !haskey(cache, r)]

    if !isempty(new_rows)
        sol = @lock core.solver_lock _solve_arc_columns(core, new_rows)
        valid_ix = core.valid_ix
        # Build each row outside the cache lock (the scatter, dist-slack, and
        # sparsify dominate); take cache_lock only for the double-check + insert,
        # matching the `cached_row_lookup` pattern. `apply_cutoff` returns `full`
        # unchanged when the cutoff is a no-op, so allocate a fresh `full` per row
        # to avoid aliasing the same buffer across stored rows.
        stored_rows = Vector{RowCacheValue}(undef, length(new_rows))
        for (j, _) in enumerate(new_rows)
            full = zeros(buscount)
            @inbounds for i in eachindex(valid_ix)
                full[valid_ix[i]] = sol[i, j]
            end
            if use_dist_slack
                full .-= dot(full, dist_slack_normalized)
            end
            stored_rows[j] = apply_cutoff(cutoff, full)
        end
        @lock cache_lock begin
            for (j, r) in enumerate(new_rows)
                haskey(cache, r) && continue  # lost a race; keep the winner
                set_persistent_row!(cache, r, stored_rows[j])
            end
        end
    end

    @lock cache_lock begin
        for r in rows
            pin_row!(cache, r)
        end
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
are cache hits. Pinned rows count against the cache capacity, so populating more
rows than `max_cache_size` holds is an error.

$(TYPEDSIGNATURES)
"""
function populate_cache(vlodf::VirtualLODF, components)
    rows = unique(Int[_resolve_arc_index(vlodf, c) for c in components])
    isempty(rows) && return nothing

    core = get_core(vlodf)
    cache = get_cache(vlodf)
    cache_lock = get_cache_lock(vlodf)
    inv_PTDF_A_diag = get_inv_PTDF_A_diag(vlodf)
    cutoff = get_cutoff(core)
    n_bus = length(core.temp_data[1])

    new_rows = @lock cache_lock Int[r for r in rows if !haskey(cache, r)]

    if !isempty(new_rows)
        sol = @lock core.solver_lock _solve_arc_columns(core, new_rows)
        valid_ix = core.valid_ix
        # Scatter each solved column back to full-bus space, then apply the
        # LODF map to every column at once: L = (A · Tmp) .* inv_PTDF_A_diag.
        tmp = zeros(n_bus, length(new_rows))
        @inbounds for j in eachindex(new_rows), i in eachindex(valid_ix)
            tmp[valid_ix[i], j] = sol[i, j]
        end
        lodf_cols = core.A * tmp                   # (n_arcs × length(new_rows))
        lodf_cols .*= inv_PTDF_A_diag              # broadcast per-arc scaling down columns
        @inbounds for (j, r) in enumerate(new_rows)
            lodf_cols[r, j] = -1.0                  # self-element convention
        end
        # Build each row outside the cache lock; take cache_lock only to insert.
        stored_rows = Vector{RowCacheValue}(undef, length(new_rows))
        for j in eachindex(new_rows)
            row = lodf_cols[:, j]
            stored_rows[j] = apply_cutoff(cutoff, row)
        end
        @lock cache_lock begin
            for (j, r) in enumerate(new_rows)
                haskey(cache, r) && continue
                set_persistent_row!(cache, r, stored_rows[j])
            end
        end
    end

    @lock cache_lock begin
        for r in rows
            pin_row!(cache, r)
        end
    end
    return nothing
end

# --- VirtualMODF -----------------------------------------------------------

# Resolve a contingency identifier to the `NetworkModification` used as the
# cache key. Accepts a modification, a `ContingencySpec`, a registered
# `PSY.Outage`, or a registered outage id.
_resolve_modification(::VirtualMODF, mod::NetworkModification) = mod
_resolve_modification(::VirtualMODF, ctg::ContingencySpec) = ctg.modification
function _resolve_modification(vmodf::VirtualMODF, outage::PSY.Outage)
    return _resolve_modification(vmodf, IS.get_id(outage))
end
function _resolve_modification(vmodf::VirtualMODF, id::Int)
    contingency_cache = get_contingency_cache(vmodf)
    haskey(contingency_cache, id) || error(
        "Contingency (id=$id) is not registered. Construct the VirtualMODF " *
        "with the system containing this outage, or pass the NetworkModification " *
        "/ ContingencySpec directly.",
    )
    return contingency_cache[id].modification
end

# Resolve a monitored arc identifier to its row index. Tuples go through
# `_monitored_arc_index` so an arc that was reduced away yields VirtualMODF's
# descriptive error instead of a raw KeyError.
_resolve_monitored_index(::VirtualMODF, m::Integer) = Int(m)
_resolve_monitored_index(vmodf::VirtualMODF, m::Tuple{Int, Int}) =
    _monitored_arc_index(vmodf, m)

"""
    _woodbury_factors_from_base(base_full, BA, arc_sus, modifications, n_bus) -> WoodburyFactors

Assemble Woodbury factors from precomputed pre-contingency solves. `base_full`
maps each modified arc index to `B⁻¹ · BA[:, arc]` scattered to full-bus space,
so the per-arc libklu solves of `_compute_woodbury_factors_impl` become
dictionary lookups; the shared kernel does the rest.
"""
function _woodbury_factors_from_base(
    base_full::Dict{Int, Vector{Float64}},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    modifications::Tuple{Vararg{ArcModification}},
    n_bus::Int,
)::WoodburyFactors
    # Z[:, j] = B⁻¹ ν_j = (B⁻¹ BA[:, e_j]) / b_{e_j}
    Z = Matrix{Float64}(undef, n_bus, length(modifications))
    for (j, mod) in enumerate(modifications)
        b_e = arc_sus[mod.arc_index]
        col = base_full[mod.arc_index]
        @inbounds for i in 1:n_bus
            Z[i, j] = col[i] / b_e
        end
    end
    return _woodbury_factors_from_Z(Z, BA, arc_sus, modifications)
end

"""
    _woodbury_correction_from_base(base_full, BA, arc_sus, monitored_idx, wf, n_bus) -> Vector{Float64}

Post-modification PTDF row for `monitored_idx`, reusing the contingency-independent
`base_full[monitored_idx]` instead of solving. The correction itself — including
the islanding zero-out — is the shared kernel's.
"""
function _woodbury_correction_from_base(
    base_full::Dict{Int, Vector{Float64}},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_sus::Vector{Float64},
    monitored_idx::Int,
    wf::WoodburyFactors,
    n_bus::Int,
)::Vector{Float64}
    b_mon = _post_modification_susceptance(arc_sus, monitored_idx, wf)
    abs(b_mon) < eps() && return zeros(n_bus)

    b_mon_pre = arc_sus[monitored_idx]
    z_m = base_full[monitored_idx] ./ b_mon_pre   # fresh vector; base_full untouched
    return _woodbury_correction!(z_m, BA, b_mon_pre, b_mon, monitored_idx, wf)
end

"""
    populate_cache(vmodf::VirtualMODF, contingencies; monitored) -> Nothing

Precompute and pin the post-contingency PTDF rows for an iterable of
`contingencies`, each evaluated over the user-supplied `monitored` arc set, using
batched multi-RHS solves.

`contingencies` may mix `NetworkModification`, `ContingencySpec`, registered
`PSY.Outage`, and registered outage UUIDs. `monitored` is an iterable of arc
identifiers (integer indices or bus-pair tuples).

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
    core = get_core(vmodf)
    row_caches = get_row_caches(vmodf)
    woodbury_cache = get_woodbury_cache(vmodf)
    max_bytes = get_max_cache_size_bytes(vmodf)
    n_bus = length(core.temp_data[1])
    cutoff = get_cutoff(core)
    BA = core.BA
    arc_sus = core.arc_susceptances

    @lock core.solver_lock begin
        # Resolve under the lock: `_resolve_modification` reads `contingency_cache`
        # for UUID/Outage inputs, so resolving here (not before the lock) matches
        # `getindex` and avoids racing with `clear_all_caches!`.
        mods = unique(
            NetworkModification[_resolve_modification(vmodf, c) for c in contingencies],
        )
        mon_idx = unique(Int[_resolve_monitored_index(vmodf, m) for m in monitored])
        (isempty(mods) || isempty(mon_idx)) && return nothing

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
        sol = _solve_arc_columns(core, all_arcs)
        valid_ix = core.valid_ix
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
            wf = get!(woodbury_cache, mod) do
                _woodbury_factors_from_base(
                    base_full,
                    BA,
                    arc_sus,
                    mod.arc_modifications,
                    n_bus,
                )
            end
            rc = get!(row_caches, mod) do
                RowCache(max_bytes, Set{Int}(), n_bus * sizeof(Float64))
            end
            for m in mon_idx
                if haskey(rc, m)
                    pin_row!(rc, m)
                    continue
                end
                row = _woodbury_correction_from_base(base_full, BA, arc_sus, m, wf, n_bus)
                stored = apply_cutoff(cutoff, row)
                set_persistent_row!(rc, m, stored)
            end
        end
    end
    return nothing
end
