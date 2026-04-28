"""
Internal pairing of a per-modification `RowCache` with the lock that
guards its reads, writes, and copies. Lets queries against different
modifications run in parallel instead of contending on a single global
lock.
"""
struct _LockedRowCache
    cache::RowCache
    lock::ReentrantLock
end

"""
The Virtual Multiple Outage Distribution Factor (VirtualMODF) structure computes
post-contingency PTDF rows lazily for registered contingencies using the
Woodbury matrix identity (van Dijk et al. Eq. 29).

Contingencies are resolved from PSY.Outage supplemental attributes at construction
time. After registration, the System is not needed for queries.

Caching is two-tiered:
- Woodbury factors (M KLU solves) are cached per contingency
- PTDF rows (1 KLU solve each) are cached per (monitored_arc, contingency) via
  one RowCache per contingency

# Arguments
- `K::KLULinSolvePool{Float64}`:
        Pool of independent ABA factorizations sized to `nworkers`. With
        `nworkers > 1`, multiple threads may call `getindex` concurrently.
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix.
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence matrix.
- `PTDF_A_diag::Vector{Float64}`:
        Raw diagonal elements of PTDF·A (H[e,e] values).
- `arc_susceptances::Vector{Float64}`:
        Effective susceptance for each arc.
- `branch_susceptances_by_arc::Vector{Vector{Float64}}`:
        Per-branch susceptances for each arc. For single-branch arcs, contains
        one element equal to the arc susceptance. For parallel branches, contains
        one entry per branch in the parallel group.
- `dist_slack::Vector{Float64}`:
        Distributed slack bus weights.
- `axes::Ax`:
        Tuple of (arc_axis, bus_axis).
- `lookup::L`:
        Tuple of lookup dictionaries for indexing.
- `valid_ix::Vector{Int}`:
        Indices of non-reference buses.
- `contingency_cache::Dict{Base.UUID, ContingencySpec}`:
        Resolved contingencies keyed by outage UUID.
- `woodbury_cache::Dict{NetworkModification, WoodburyFactors}`:
        Precomputed Woodbury factors keyed by modification.
- `woodbury_inflight::Dict{NetworkModification, Base.Event}`:
        In-flight Woodbury computations, used to deduplicate concurrent
        first-time queries for the same modification. Guarded by
        `woodbury_cache_lock`. The first thread to query a fresh
        modification claims the slot, computes the factors, populates
        `woodbury_cache`, and `notify`s the event; concurrent waiters
        block on the event instead of recomputing.
- `row_caches::Dict{NetworkModification, _LockedRowCache}`:
        One `RowCache` per modification, paired with its own lock. The outer
        `row_caches_lock` only guards inserts/clears of this dict; the per-mod
        lock guards reads/writes of the corresponding cache so that queries
        against different modifications run in parallel.
- `subnetwork_axes::Dict{Int, Ax}`:
        Maps reference bus indices to subnetwork axes.
- `tol::Base.RefValue{Float64}`:
        Tolerance for sparsification.
- `max_cache_size_bytes::Int`:
        Max cache size in bytes per contingency.
- `network_reduction_data::NetworkReductionData`:
        Network reduction mappings for branch resolution.
- `temp_data::Vector{Vector{Float64}}`:
        Per-worker scratch vector of size n_buses (one per pool worker).
- `work_ba_col::Vector{Vector{Float64}}`:
        Per-worker work array for BA column extraction (one per pool worker).
- `system_uuid::Union{Base.UUID, Nothing}`:
        UUID of the system used to construct the matrix, used to validate that
        modification operations are applied to the correct system.
"""
struct VirtualMODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}} <:
       PowerNetworkMatrix{Float64}
    K::KLULinSolvePool{Float64}
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    PTDF_A_diag::Vector{Float64}
    arc_susceptances::Vector{Float64}
    branch_susceptances_by_arc::Vector{Vector{Float64}}
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    contingency_cache::Dict{Base.UUID, ContingencySpec}
    woodbury_cache::Dict{NetworkModification, WoodburyFactors}
    # In-flight set: shares `woodbury_cache_lock` with `woodbury_cache`.
    # Concurrent first-time queries for the same modification observe the
    # event here and `wait` instead of redundantly recomputing.
    woodbury_inflight::Dict{NetworkModification, Base.Event}
    woodbury_cache_lock::ReentrantLock
    row_caches::Dict{NetworkModification, _LockedRowCache}
    row_caches_lock::ReentrantLock
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    max_cache_size_bytes::Int
    network_reduction_data::NetworkReductionData
    temp_data::Vector{Vector{Float64}}
    work_ba_col::Vector{Vector{Float64}}
    system_uuid::Union{Base.UUID, Nothing}
end

# --- Accessors ---

get_axes(M::VirtualMODF) = M.axes
get_lookup(M::VirtualMODF) = M.lookup
get_ref_bus(M::VirtualMODF) = sort!(collect(keys(M.subnetwork_axes)))
get_network_reduction_data(M::VirtualMODF) = M.network_reduction_data
get_arc_lookup(M::VirtualMODF) = M.lookup[1]
get_bus_lookup(M::VirtualMODF) = M.lookup[2]
get_arc_axis(mat::VirtualMODF) = mat.axes[1]
get_bus_axis(mat::VirtualMODF) = mat.axes[2]
get_tol(mat::VirtualMODF) = mat.tol[]
get_system_uuid(M::VirtualMODF) = M.system_uuid

# Woodbury kernel accessors. VirtualMODF holds per-worker scratch and a pool;
# use with_worker(K) and the matching scratch index for thread safety.
_get_BA(m::VirtualMODF) = m.BA
_get_arc_susceptances(m::VirtualMODF) = m.arc_susceptances
_get_valid_ix(m::VirtualMODF) = m.valid_ix

"""
Return the number of pool workers (= max concurrent solves) for `vmodf`.
"""
nworkers(vmodf::VirtualMODF) = nworkers(vmodf.K)

function _compute_woodbury_factors(
    mat::VirtualMODF,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    return with_worker(mat.K) do cache, idx
        _compute_woodbury_factors_impl(
            cache, mat.work_ba_col[idx], mat.temp_data[idx],
            mat.BA, mat.arc_susceptances, mat.valid_ix, modifications,
        )
    end
end

function _apply_woodbury_correction(
    mat::VirtualMODF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return with_worker(mat.K) do cache, idx
        _apply_woodbury_correction_impl(
            cache, mat.work_ba_col[idx], mat.temp_data[idx],
            mat.BA, mat.arc_susceptances, mat.valid_ix, monitored_idx, wf,
        )
    end
end

"""
    get_registered_contingencies(vmodf::VirtualMODF) -> Dict{Base.UUID, ContingencySpec}

Return the cached contingency registrations for inspection.
"""
get_registered_contingencies(vmodf::VirtualMODF) = vmodf.contingency_cache

# --- Base interface ---

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualMODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    print(
        io,
        "VirtualMODF with $(length(array.contingency_cache)) registered contingencies",
    )
    return
end

function Base.isempty(vmodf::VirtualMODF)
    return isempty(vmodf.contingency_cache)
end

Base.size(vmodf::VirtualMODF) = (length(vmodf.axes[1]), length(vmodf.axes[2]))

Base.setindex!(::VirtualMODF, _, idx...) = error("Operation not supported by VirtualMODF")
Base.setindex!(::VirtualMODF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualMODF")

# --- Constructor ---

"""
    VirtualMODF(sys::PSY.System; kwargs...) -> VirtualMODF

Build a VirtualMODF from a PowerSystems System. Automatically registers all
Outage supplemental attributes found in the system.

# Arguments
- `sys::PSY.System`: Power system to build from

# Keyword Arguments
- `dist_slack::Vector{Float64}`: Distributed slack weights (default: empty)
- `tol::Float64`: Tolerance for row sparsification (default: eps())
- `max_cache_size::Int`: Max cache size in MiB per contingency (default: MAX_CACHE_SIZE_MiB)
- `network_reductions::Vector{NetworkReduction}`: Network reductions to apply
"""
function VirtualMODF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    tol::Float64 = eps(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    nworkers::Int = max(1, Threads.nthreads() - 1),
    kwargs...,
)
    if length(dist_slack) != 0
        @info "Distributed bus"
    end

    # Build network matrices (same path as VirtualLODF)
    Ymatrix = Ybus(sys; network_reductions = network_reductions, kwargs...)
    ref_bus_positions = get_ref_bus_position(Ymatrix)
    A = IncidenceMatrix(Ymatrix)

    arc_ax = get_arc_axis(A)
    bus_ax = get_bus_axis(A)
    axes = (arc_ax, bus_ax)
    arc_ax_ref = make_ax_ref(arc_ax)
    bus_ax_ref = make_ax_ref(bus_ax)
    look_up = (arc_ax_ref, bus_ax_ref)
    subnetwork_axes = A.subnetwork_axes

    BA = BA_Matrix(Ymatrix)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    K_pool = KLULinSolvePool(ABA; nworkers = nworkers)

    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)

    # Use one worker for the diagonal precomputation; the pool is sized to
    # `nworkers` solves, so a serial precompute is fine.
    PTDF_A_diag = with_worker(K_pool) do cache, _idx
        _get_PTDF_A_diag(cache, BA.data, A.data, Set(ref_bus_positions))
    end
    arc_susceptances = _extract_arc_susceptances(BA.data)
    branch_susceptances_by_arc = _extract_branch_susceptances_by_arc(
        BA.data, arc_ax, Ymatrix.network_reduction_data)

    temp_data = [zeros(length(bus_ax)) for _ in 1:nworkers]
    work_ba_col = [zeros(length(valid_ix)) for _ in 1:nworkers]
    max_cache_bytes = max_cache_size * MiB

    vmodf = VirtualMODF(
        K_pool,
        BA.data,
        A.data,
        PTDF_A_diag,
        arc_susceptances,
        branch_susceptances_by_arc,
        dist_slack,
        axes,
        look_up,
        valid_ix,
        Dict{Base.UUID, ContingencySpec}(),
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, Base.Event}(),
        ReentrantLock(),
        Dict{NetworkModification, _LockedRowCache}(),
        ReentrantLock(),
        subnetwork_axes,
        Ref(tol),
        max_cache_bytes,
        Ymatrix.network_reduction_data,
        temp_data,
        work_ba_col,
        IS.get_uuid(sys),
    )

    # Auto-register all outage attributes from the system
    _register_outages!(vmodf, sys)

    return vmodf
end

# --- Outage registration ---

"""
    _register_outages!(vmodf, sys)

Bulk-register all Outage supplemental attributes in the system.
Called automatically by the VirtualMODF constructor.

Uses `PSY.get_supplemental_attributes(PSY.Outage, sys)` which accepts
the abstract type and iterates over all concrete subtypes
(PlannedOutage, UnplannedOutage).
"""
function _register_outages!(vmodf::VirtualMODF, sys::PSY.System)
    count = 0
    for outage in PSY.get_supplemental_attributes(PSY.Outage, sys)
        try
            _register_outage!(vmodf, sys, outage)
            count += 1
        catch e
            e isa ErrorException || rethrow()
            @warn "Could not register outage: $(e.msg)"
        end
    end

    if count == 0
        @warn "No outage supplemental attributes found in system. " *
              "VirtualMODF contingency cache is empty."
    else
        @info "Registered $count contingencies from system outage attributes."
    end
    return
end

"""
    _register_outage!(vmodf, sys, outage) -> ContingencySpec

Resolve an Outage supplemental attribute to a ContingencySpec and cache it.
Delegates to `NetworkModification(mat, sys, outage)` for the resolution logic.
"""
function _register_outage!(
    vmodf::VirtualMODF,
    sys::PSY.System,
    outage::PSY.Outage,
)
    outage_uuid = IS.get_uuid(outage)
    if haskey(vmodf.contingency_cache, outage_uuid)
        return vmodf.contingency_cache[outage_uuid]
    end
    mod = NetworkModification(vmodf, sys, outage)
    ctg = ContingencySpec(outage_uuid, mod)
    vmodf.contingency_cache[outage_uuid] = ctg
    return ctg
end

# --- Woodbury factor computation ---

"""
    _get_woodbury_factors(vmodf, mod) -> WoodburyFactors

Compute and cache the Woodbury factors for a network modification. Safe to
call from multiple threads: `vmodf.woodbury_cache_lock` guards both the
cache and an in-flight set, so concurrent first-time queries for the same
modification observe a single shared computation. The first caller claims
the slot, performs the KLU solves, and `notify`s waiters on completion;
others block on the event and pick up the result from the cache. Failure
in the owner is propagated by clearing the in-flight slot, notifying
waiters, and rethrowing — waiters then loop and either see a winner or
become the next owner themselves.
"""
function _get_woodbury_factors(
    vmodf::VirtualMODF,
    mod::NetworkModification,
)
    while true
        cached, event, is_owner = @lock vmodf.woodbury_cache_lock begin
            wf = get(vmodf.woodbury_cache, mod, nothing)
            if wf !== nothing
                (wf, nothing, false)
            else
                pending = get(vmodf.woodbury_inflight, mod, nothing)
                if pending !== nothing
                    (nothing, pending, false)
                else
                    ev = Base.Event()
                    vmodf.woodbury_inflight[mod] = ev
                    (nothing, ev, true)
                end
            end
        end

        cached === nothing || return cached

        if !is_owner
            wait(event)
            continue
        end

        my_event = event
        try
            wf = _compute_woodbury_factors(vmodf, mod.arc_modifications)
            @lock vmodf.woodbury_cache_lock vmodf.woodbury_cache[mod] = wf
            return wf
        finally
            # Only retract the in-flight slot if it still references our
            # event: `clear_all_caches!` plus a subsequent claim by another
            # thread would otherwise let us delete a stranger's event.
            @lock vmodf.woodbury_cache_lock begin
                if get(vmodf.woodbury_inflight, mod, nothing) === my_event
                    delete!(vmodf.woodbury_inflight, mod)
                end
            end
            notify(my_event)
        end
    end
end

"""
    _compute_modf_entry(vmodf, monitored_idx, mod) -> Vector{Float64}

Compute the post-modification PTDF row for a monitored arc under the given modification.
Gets or computes Woodbury factors, then applies the Woodbury correction.

For N-1 contingencies, the result satisfies:
    post_ptdf[mon, :] = pre_ptdf[mon, :] + LODF[mon, e] * pre_ptdf[e, :]

!!! warning
    Not thread-safe. Mutates scratch vectors in `vmodf`.
"""
function _compute_modf_entry(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    mod::NetworkModification,
)::Vector{Float64}
    wf = _get_woodbury_factors(vmodf, mod)
    return _apply_woodbury_correction(vmodf, monitored_idx, wf)
end

# --- Row cache management ---

"""
    _get_or_create_row_cache(vmodf, mod) -> _LockedRowCache

Get or create the per-modification `RowCache` paired with its dedicated
lock. The outer `vmodf.row_caches_lock` is held only briefly to look up
or insert the entry; callers use the returned per-mod lock for all
hit/miss/store operations on the cache so that queries against different
modifications do not contend on a single global lock.
"""
function _get_or_create_row_cache(vmodf::VirtualMODF, mod::NetworkModification)
    @lock vmodf.row_caches_lock begin
        existing = get(vmodf.row_caches, mod, nothing)
        existing === nothing || return existing
        row_size = length(vmodf.temp_data[1]) * sizeof(Float64)
        entry = _LockedRowCache(
            RowCache(vmodf.max_cache_size_bytes, Set{Int}(), row_size),
            ReentrantLock(),
        )
        vmodf.row_caches[mod] = entry
        return entry
    end
end

# --- getindex: by integer monitored index + NetworkModification ---

"""
Get the post-modification PTDF row for monitored arc `monitored_idx` under `mod`.
Uses per-modification RowCache for LRU-eviction caching.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    mod::NetworkModification,
)
    entry = _get_or_create_row_cache(vmodf, mod)
    row_cache = entry.cache
    cache_lock = entry.lock

    # Copy outside the lock — copies are O(n_buses) and would otherwise
    # block other readers on this contingency.
    cached = @lock cache_lock get(row_cache.temp_cache, monitored_idx, nothing)
    cached === nothing || return copy(cached)

    row = _compute_modf_entry(vmodf, monitored_idx, mod)
    stored = get_tol(vmodf) > eps() ? sparsify(row, get_tol(vmodf)) : row

    cached_row = @lock cache_lock begin
        existing = get(row_cache.temp_cache, monitored_idx, nothing)
        if existing === nothing
            row_cache[monitored_idx] = stored
            stored
        else
            existing
        end
    end
    return copy(cached_row)
end

"""
Arc-tuple indexed version of getindex for VirtualMODF with NetworkModification.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Tuple{Int, Int},
    mod::NetworkModification,
)
    m_idx = vmodf.lookup[1][monitored]
    return vmodf[m_idx, mod]
end

# --- getindex: by ContingencySpec (delegates to NetworkModification) ---

"""
Get the post-contingency PTDF row for monitored arc under a ContingencySpec.
Delegates to the NetworkModification-based getindex.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    contingency::ContingencySpec,
)
    return vmodf[monitored_idx, contingency.modification]
end

function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Tuple{Int, Int},
    contingency::ContingencySpec,
)
    return vmodf[monitored, contingency.modification]
end

# --- getindex: by PSY.Outage (UUID lookup → ContingencySpec → NetworkModification) ---

"""
Get the post-contingency PTDF row for monitored arc `monitored` when outage `outage` trips.
The outage must have been registered at VirtualMODF construction time.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Int,
    outage::PSY.Outage,
)
    outage_uuid = IS.get_uuid(outage)
    # Read under `woodbury_cache_lock` to pair with the locked `empty!` in
    # `clear_all_caches!` — without it, a concurrent clear could rehash the
    # underlying Dict mid-lookup.
    ctg = @lock vmodf.woodbury_cache_lock begin
        if !haskey(vmodf.contingency_cache, outage_uuid)
            error(
                "Outage (UUID=$outage_uuid) is not registered. " *
                "Construct VirtualMODF with the system containing this outage.",
            )
        end
        vmodf.contingency_cache[outage_uuid]
    end
    return vmodf[monitored, ctg.modification]
end

"""
Arc-tuple indexed version of getindex by PSY.Outage.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Tuple{Int, Int},
    outage::PSY.Outage,
)
    m_idx = vmodf.lookup[1][monitored]
    return vmodf[m_idx, outage]
end

"""
    clear_caches!(vmodf::VirtualMODF)

Clear Woodbury and row caches. Does NOT clear the contingency registration
cache — registered outages remain valid and can be queried again.
"""
function clear_caches!(vmodf::VirtualMODF)
    @lock vmodf.woodbury_cache_lock empty!(vmodf.woodbury_cache)
    @lock vmodf.row_caches_lock empty!(vmodf.row_caches)
    return
end

"""
    clear_all_caches!(vmodf::VirtualMODF)

Clear all caches including contingency registrations. After calling this function,
the `VirtualMODF` object is effectively empty and cannot be queried — it has
no registered contingencies. To restore functionality, a new `VirtualMODF` must
be constructed from a `PSY.System`.

Use `clear_caches!` instead to preserve contingency registrations while
freeing computation cache memory.
"""
function clear_all_caches!(vmodf::VirtualMODF)
    # `contingency_cache` is emptied under `woodbury_cache_lock` to pair with
    # the locked read in the `Outage`-keyed `getindex`; without the lock a
    # concurrent query could observe a torn Dict state. The in-flight set is
    # cleared too — owner threads still holding their event reference will
    # skip the now-mismatched delete via the identity check, so dropping
    # entries here can't clobber another thread's claim.
    @lock vmodf.woodbury_cache_lock begin
        empty!(vmodf.contingency_cache)
        empty!(vmodf.woodbury_cache)
        empty!(vmodf.woodbury_inflight)
    end
    @lock vmodf.row_caches_lock empty!(vmodf.row_caches)
    return
end
