get_core(core::VirtualFactorCore) = core

# A worker computes its own lazy branch susceptances, so it does not share the parent's.
worker_network_data(nd::CoreNetworkData) = CoreNetworkData(nd.branch_catalog)

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
    cache = _persistent_row_cache(max_cache_size, core, Tuple{Int, Int}[])
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
    rc = get!(() -> _new_modf_row_cache(vmodf), get_row_caches(vmodf), mod)
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
    wf = _compute_woodbury_factors(solver, _arc_modification_vector(mod))
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
