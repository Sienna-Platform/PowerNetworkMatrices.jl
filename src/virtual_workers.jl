# Per-worker factorizations for parallel queries on the virtual matrices. A `worker_core` shares
# a core's read-only topology but owns its factorization, scratch and lock, so workers solve
# without contending on the shared `solver_lock`. Every function here takes any object with a
# `get_core` and solves on that core.

get_core(core::VirtualFactorCore) = core

# Apple Accelerate calls have segfaulted when concurrent, even on distinct factorizations, so a
# worker core on that backend shares its parent's lock: workers serialize. KLU's distinct caches
# are safe to use concurrently, so each worker gets its own lock.
_worker_lock(::KLULinSolveCache{Float64}, ::ReentrantLock) = ReentrantLock()
_worker_lock(::AAFactorCache, parent::ReentrantLock) = parent

_factorize_like(
    ::KLULinSolveCache{Float64},
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
) =
    _create_factorization(KLUSolver(), ABA)
_factorize_like(::AAFactorCache, ABA::SparseArrays.SparseMatrixCSC{Float64, Int}) =
    _create_factorization(AppleAccelerateLUSolver(), ABA)

"""
    worker_core(mat) -> VirtualFactorCore

A new core sharing the read-only topology of `mat`'s core (a `VirtualPTDF`, `VirtualMODF` or
`VirtualFactorCore`) with its own factorization of the same ABA matrix, its own solve scratch
and its own `solver_lock`. On KLU the factorization is deterministic, so every solve on the
worker equals the same solve on the original bit for bit, and workers solve concurrently. On
Apple Accelerate the worker shares the parent's lock, so its solves serialize with the parent's
and every other worker's. The lazy `PTDF_A_diag` and branch susceptances start empty.
"""
function worker_core(mat)
    core = get_core(mat)
    n_bus = size(core.BA, 1)
    ref = Set{Int}(setdiff(1:n_bus, core.valid_ix))
    ABA = calculate_ABA_matrix(core.A, core.BA, ref)
    lock = _worker_lock(core.K, core.solver_lock)
    K = @lock lock _factorize_like(core.K, ABA)
    return VirtualFactorCore(
        K,
        core.BA,
        core.A,
        core.arc_susceptances,
        core.axes,
        core.lookup,
        core.valid_ix,
        core.bus_to_valid_idx,
        core.subnetwork_axes,
        core.tol,
        core.branch_catalog,
        [zeros(n_bus)],
        [zeros(length(core.valid_ix))],
        lock,
        core.system_uuid,
        Float64[],
        Threads.Atomic{Bool}(false),
        Vector{Vector{Float64}}(),
        Threads.Atomic{Bool}(false),
    )
end

function _check_worker_core(core::VirtualFactorCore, solver::VirtualFactorCore)
    solver.BA === core.BA || error(
        "The solver core does not share this matrix's topology; build it with " *
        "worker_core(matrix).",
    )
    return
end

# --- Option 1: matrices on a worker core ---

"""
    VirtualPTDF(vptdf::VirtualPTDF, core::VirtualFactorCore; max_cache_size) -> VirtualPTDF

`vptdf` (same distributed slack) on `core`, a [`worker_core`](@ref) of it, with an empty row
cache: rows equal `vptdf`'s, solved on `core`'s factorization.
"""
function VirtualPTDF(
    vptdf::VirtualPTDF,
    core::VirtualFactorCore;
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
)
    _check_worker_core(get_core(vptdf), core)
    cache = _persistent_row_cache(
        max_cache_size, core.lookup, Tuple{Int, Int}[], size(core.BA, 1),
    )
    return VirtualPTDF(
        core,
        get_dist_slack(vptdf),
        get_dist_slack_normalized(vptdf),
        cache,
        ReentrantLock(),
    )
end

"""
    VirtualMODF(vmodf::VirtualMODF, core::VirtualFactorCore) -> VirtualMODF

`vmodf` (a copy of its registered contingencies, same cache bound) on `core`, a
[`worker_core`](@ref) of it, with empty Woodbury and row caches.
"""
function VirtualMODF(vmodf::VirtualMODF, core::VirtualFactorCore)
    parent = get_core(vmodf)
    _check_worker_core(parent, core)
    registered = @lock parent.solver_lock copy(get_registered_contingencies(vmodf))
    return VirtualMODF(
        core,
        registered,
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache{RowCacheValue}}(),
        get_max_cache_size_bytes(vmodf),
    )
end

# --- Option 2: shared matrices, caller's factorization ---

"""
    get_ptdf_row(vptdf::VirtualPTDF, arc, solver::VirtualFactorCore)

`get_ptdf_row(vptdf, arc)` (`arc` an index or a bus pair), solving a cache miss on `solver`, a
[`worker_core`](@ref) of `vptdf`, instead of under `vptdf`'s `solver_lock`. The row cache is
shared and guarded by `cache_lock`; the returned row is the cache's own, read-only.
"""
function get_ptdf_row(
    vptdf::VirtualPTDF,
    arc::Union{Int, Tuple{Int, Int}},
    solver::VirtualFactorCore,
)
    _check_worker_core(get_core(vptdf), solver)
    row = _resolve_arc_index(vptdf, arc)
    return _cached_row(
        get_cache(vptdf), get_cache_lock(vptdf), row, get_cutoff(vptdf),
    ) do
        _compute_ptdf_row(vptdf, row, solver)
    end
end

# The modification of `contingency` and its row cache, created on first use. Caller holds the
# core's `solver_lock`, which guards every VirtualMODF dictionary.
function _modf_row_cache!(vmodf::VirtualMODF, contingency)
    mod = _resolve_modification(vmodf, contingency)
    rc = get!(get_row_caches(vmodf), mod) do
        RowCache(get_max_cache_size_bytes(vmodf), Set{Int}(), size(vmodf)[2] * sizeof(Float64))
    end
    return mod, rc
end

"""
    get_woodbury_factors(vmodf::VirtualMODF, contingency, solver::VirtualFactorCore) -> WoodburyFactors

The cached Woodbury factors of `contingency` (a `NetworkModification`, `ContingencySpec`,
registered `PSY.Outage` or outage id), computed on a miss on `solver`, a [`worker_core`](@ref)
of `vmodf`. The cache is `vmodf`'s own, shared with `vmodf[m, contingency]`; the core's
`solver_lock` is held only around cache reads and writes, never for the solves. Two workers
missing on the same contingency both compute it and the first insert wins.
"""
function get_woodbury_factors(
    vmodf::VirtualMODF,
    contingency,
    solver::VirtualFactorCore,
)
    core = get_core(vmodf)
    _check_worker_core(core, solver)
    cache = get_woodbury_cache(vmodf)
    mod = @lock core.solver_lock begin
        resolved = _resolve_modification(vmodf, contingency)
        haskey(cache, resolved) && return cache[resolved]
        resolved
    end
    wf = _compute_woodbury_factors(solver, mod.arc_modifications)
    return @lock core.solver_lock get!(cache, mod, wf)
end

"""
    get_post_modification_ptdf_row(vmodf::VirtualMODF, monitored, contingency, solver::VirtualFactorCore) -> Vector{Float64}

`vmodf[monitored, contingency]` (a fresh copy), solving every miss on `solver`, a
[`worker_core`](@ref) of `vmodf`: the Woodbury and row caches are shared, and the core's
`solver_lock` is held only around cache reads and writes.
"""
function get_post_modification_ptdf_row(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    contingency,
    solver::VirtualFactorCore,
)
    core = get_core(vmodf)
    _check_worker_core(core, solver)
    m = _resolve_monitored_index(vmodf, monitored)
    mod = @lock core.solver_lock begin
        resolved, rc = _modf_row_cache!(vmodf, contingency)
        haskey(rc, m) && return copy(rc[m])
        resolved
    end
    wf = get_woodbury_factors(vmodf, mod, solver)
    stored = apply_cutoff(get_cutoff(core), _apply_woodbury_correction(solver, m, wf))
    @lock core.solver_lock begin
        _, rc = _modf_row_cache!(vmodf, mod)
        haskey(rc, m) && return copy(rc[m])
        rc[m] = stored
    end
    return copy(stored)
end

# --- Factor-free flow algebra on angles ---

function _check_bus_rows(core::VirtualFactorCore, x::AbstractVecOrMat, name::String)
    size(x, 1) == size(core.BA, 1) || throw(
        DimensionMismatch(
            "$name has $(size(x, 1)) rows; the network has $(size(core.BA, 1)) buses.",
        ),
    )
    return
end

"""
    solve_bus_angles!(θ, mat, p) -> θ

`θ = B⁻¹ p` on the reference-bus-reduced ABA matrix `B` of `mat` (a `VirtualPTDF`,
`VirtualMODF` or `VirtualFactorCore`), in full-bus space with the reference-bus entries 0;
`p` (injections, per bus position) is read at the non-reference buses only. Vectors or
matrices (one column per case). Solves on `mat`'s core under its `solver_lock`; pass a
[`worker_core`](@ref) to solve concurrently. `arc_flows!(…, θ, …)` then gives the flows of `p`
balanced at each island's reference bus, i.e. the PTDF rows dotted with `p`.
"""
function solve_bus_angles!(
    θ::AbstractVector{Float64},
    mat,
    p::AbstractVector{Float64},
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_bus_rows(core, p, "p")
    valid_ix = core.valid_ix
    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, _
        @inbounds for (i, b) in enumerate(valid_ix)
            work_ba_col[i] = p[b]
        end
        lin_solve = _solve_factorization(K_solver, work_ba_col)
        return _gather_to_buses!(θ, valid_ix, lin_solve)
    end
end

solve_bus_angles!(θ::AbstractMatrix{Float64}, mat, p::AbstractMatrix{Float64}) =
    solve_bus_angles!(
        θ, mat, p, Matrix{Float64}(undef, length(get_core(mat).valid_ix), size(p, 2)),
    )

"""
    solve_bus_angles!(θ::AbstractMatrix, mat, p::AbstractMatrix, rhs::Matrix{Float64}) -> θ

The matrix form solving in `rhs`, a caller buffer of size (non-reference buses, columns of `p`).
"""
function solve_bus_angles!(
    θ::AbstractMatrix{Float64},
    mat,
    p::AbstractMatrix{Float64},
    rhs::Matrix{Float64},
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_bus_rows(core, p, "p")
    size(θ, 2) == size(p, 2) ||
        throw(DimensionMismatch("θ has $(size(θ, 2)) columns, p has $(size(p, 2))."))
    valid_ix = core.valid_ix
    size(rhs) == (length(valid_ix), size(p, 2)) || throw(
        DimensionMismatch(
            "rhs is $(size(rhs)); expected ($(length(valid_ix)), $(size(p, 2))).",
        ),
    )
    for t in axes(p, 2), (i, b) in enumerate(valid_ix)
        rhs[i, t] = p[b, t]
    end
    sol = @lock core.solver_lock _solve_factorization(core.K, rhs)
    fill!(θ, 0.0)
    for t in axes(p, 2), (i, b) in enumerate(valid_ix)
        θ[b, t] = sol[i, t]
    end
    return θ
end

# BA[:, arc] ⋅ θ[:, t].
function _ba_dot(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc::Int,
    θ::AbstractVecOrMat{Float64},
    t::Int,
)
    rv = SparseArrays.rowvals(BA)
    nzv = SparseArrays.nonzeros(BA)
    acc = 0.0
    @inbounds for k in SparseArrays.nzrange(BA, arc)
        acc += nzv[k] * θ[rv[k], t]
    end
    return acc
end

function _check_flows(
    flows::AbstractVecOrMat,
    θ::AbstractVecOrMat,
    arcs::AbstractVector{Int},
)
    size(flows, 1) == length(arcs) && size(flows, 2) == size(θ, 2) || throw(
        DimensionMismatch(
            "flows is $(size(flows)); expected ($(length(arcs)), $(size(θ, 2))).",
        ),
    )
    return
end

"""
    arc_flows!(flows, mat, θ, arcs) -> flows
    arc_flows!(flows, mat, θ, arcs, wf::WoodburyFactors) -> flows

Flows on the arc indices `arcs` from angles `θ` (from [`solve_bus_angles!`](@ref); a vector, or
a matrix with one column per case, `flows[q, t]` for arc `arcs[q]`). Without `wf`,
`flows = BA[:, arcs]ᵀ θ`. With `wf` the flows are post-modification by Woodbury:
`flows[q, t] = b_post * (ν_mᵀ θ_t - (ν_mᵀ Z) ⋅ W⁻ᵀ Uᵀ θ_t)`, which for `θ = B⁻¹ p` equals
`vmodf[arcs[q], mod] ⋅ p` before its islanding zero-out (an arc the modification removes reads
exactly 0.0). Under islanding, entries of other islands are not zeroed: pass `θ = B⁻¹ p_I` with
`p` restricted to the arc's island `I` (see `wf.bus_island_labels`). Pure algebra on `BA` and
`wf`, O(length(arcs) · M) per column; no solve, no lock, safe from any number of tasks.
"""
function arc_flows!(
    flows::AbstractVecOrMat{Float64},
    mat,
    θ::AbstractVecOrMat{Float64},
    arcs::AbstractVector{Int},
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_flows(flows, θ, arcs)
    for t in axes(θ, 2), (q, m) in enumerate(arcs)
        flows[q, t] = _ba_dot(core.BA, m, θ, t)
    end
    return flows
end

arc_flows!(
    flows::AbstractVecOrMat{Float64},
    mat,
    θ::AbstractVecOrMat{Float64},
    arcs::AbstractVector{Int},
    wf::WoodburyFactors,
) = arc_flows!(
    flows, mat, θ, arcs, wf, WoodburyFlowScratch(length(wf.arc_indices), size(θ, 2)),
)

"""
    WoodburyFlowScratch(max_modified_arcs, max_columns = 1)

Buffers of the Woodbury [`arc_flows!`](@ref) for modifications of up to `max_modified_arcs`
arcs and angles of up to `max_columns` columns, so a caller reusing one allocates nothing.
"""
struct WoodburyFlowScratch
    u::Matrix{Float64}
    c::Matrix{Float64}
    zm_Z::Vector{Float64}
end

WoodburyFlowScratch(max_modified_arcs::Int, max_columns::Int = 1) = WoodburyFlowScratch(
    Matrix{Float64}(undef, max_modified_arcs, max_columns),
    Matrix{Float64}(undef, max_modified_arcs, max_columns),
    Vector{Float64}(undef, max_modified_arcs),
)

"""
    arc_flows!(flows, mat, θ, arcs, wf::WoodburyFactors, scratch::WoodburyFlowScratch) -> flows

The Woodbury form on caller-owned buffers; allocation-free.
"""
function arc_flows!(
    flows::AbstractVecOrMat{Float64},
    mat,
    θ::AbstractVecOrMat{Float64},
    arcs::AbstractVector{Int},
    wf::WoodburyFactors,
    scratch::WoodburyFlowScratch,
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_flows(flows, θ, arcs)
    BA = core.BA
    arc_sus = core.arc_susceptances
    M = length(wf.arc_indices)
    T = size(θ, 2)
    M <= length(scratch.zm_Z) && T <= size(scratch.u, 2) || throw(
        DimensionMismatch(
            "The WoodburyFlowScratch holds $(length(scratch.zm_Z)) modified arcs and " *
            "$(size(scratch.u, 2)) columns; this call needs $M and $T.",
        ),
    )
    # c[:, t] = W⁻ᵀ Uᵀ θ_t, with (Uᵀ θ)_j = ν_jᵀ θ = BA[:, e_j]ᵀ θ / b_{e_j}. Transposed so the
    # result matches the row form `z_m - Z W⁻¹ (Zᵀ ν_m)` dotted with p in floating point too.
    u = view(scratch.u, 1:M, 1:T)
    for t in 1:T, j in 1:M
        e = wf.arc_indices[j]
        u[j, t] = _ba_dot(BA, e, θ, t) / arc_sus[e]
    end
    c = view(scratch.c, 1:M, 1:T)
    LinearAlgebra.mul!(c, transpose(wf.W_inv), u)
    nzv = SparseArrays.nonzeros(BA)
    rv = SparseArrays.rowvals(BA)
    zm_Z = scratch.zm_Z
    for (q, m) in enumerate(arcs)
        b_post = _post_modification_susceptance(arc_sus, m, wf)
        if abs(b_post) < eps()
            for t in 1:T
                flows[q, t] = 0.0
            end
            continue
        end
        b_pre = arc_sus[m]
        fill!(zm_Z, 0.0)
        @inbounds for k in SparseArrays.nzrange(BA, m)
            coeff = nzv[k] / b_pre
            for j in 1:M
                zm_Z[j] += coeff * wf.Z[rv[k], j]
            end
        end
        for t in 1:T
            acc = _ba_dot(BA, m, θ, t) / b_pre
            for j in 1:M
                acc -= zm_Z[j] * c[j, t]
            end
            flows[q, t] = b_post * acc
        end
    end
    return flows
end

"""
    compute_woodbury_factors(mat, mod[, labeler]) -> WoodburyFactors

Uncached Woodbury factors of `mod`, solved on `mat`'s core (`mat` a `VirtualMODF` or
`VirtualFactorCore`, e.g. a [`worker_core`](@ref); the `VirtualPTDF` form also takes
`labeler`). `labeler(BA, arc_sus, modifications, n_bus)` gives the island labels when `mod`
islands; a `BridgeLabels` gives the same partition as the default in O(n_bus) for a single
bridge, with different representatives.
"""
compute_woodbury_factors(
    mat,
    mod::NetworkModification,
) =
    compute_woodbury_factors(mat, mod, _post_contingency_bus_labels)

function compute_woodbury_factors(
    mat,
    mod::NetworkModification,
    labeler::F,
) where {F}
    return _compute_woodbury_factors(get_core(mat), mod.arc_modifications, labeler)
end

"""
    apply_woodbury_correction(core::VirtualFactorCore, monitored_idx::Int, wf) -> Vector{Float64}

The `VirtualPTDF` form solved on `core`, e.g. a [`worker_core`](@ref).
"""
apply_woodbury_correction(
    core::VirtualFactorCore,
    monitored_idx::Int,
    wf::WoodburyFactors,
) =
    _apply_woodbury_correction(core, monitored_idx, wf)
