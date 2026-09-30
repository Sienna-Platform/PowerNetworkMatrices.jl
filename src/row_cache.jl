# Value type stored per cached row: a dense row, or a sparsified row when a
# tolerance is applied.
const RowCacheValue = Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}

"""
Structure used for saving the rows of the Virtual PTDF and LODF matrix.

# Arguments
- `temp_cache::Dict{Int, Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}`:
        Dictionary saving the row of the PTDF/LODF matrix
- `persistent_cache_keys::Set{Int}`:
        Set listing the rows to keep in `temp_cache`
- `max_num_keys::Int`
        Defines the maximum number of keys saved (rows of the matrix)
- `access_order::Vector{Int}`:
        Vector tracking access order for LRU eviction (most recent at end)
"""
struct RowCache{T <: Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}
    temp_cache::Dict{Int, T}
    persistent_cache_keys::Set{Int}
    max_num_keys::Int
    access_order::Vector{Int}
end

"""
Structure used for saving the rows of the Virtual PTDF and LODF matrix.

# Arguments
- `max_cache_size::Int`
        Defines the maximum allowed cache size (rows*row_size).
- `persistent_rows::Set{Int}`:
        Set listing the rows to keep in `temp_cache`.
- `row_size`
        Defines the size of the single row to store.
"""
function RowCache(max_cache_size::Int, persistent_rows::Set{Int}, row_size)
    persistent_data_size = (length(persistent_rows) + 1) * row_size
    if persistent_data_size > max_cache_size
        error(
            "The required cache size for the persisted row is larger than the max cache size. Persistent data size = $(persistent_data_size), max cache size = $(max_cache_size)",
        )
    else
        @debug "required cache for persisted values = $((length(persistent_rows) + 1)*row_size). Max cache specification = $(max_cache_size)"
    end
    max_num_keys = max(length(persistent_rows) + 1, floor(Int, max_cache_size / row_size))
    return RowCache(
        sizehint!(
            Dict{Int, Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}(),
            max_num_keys,
        ),
        # Copy: the cache mutates this set (pinning, and `empty!`), and the caller's own
        # set must not move under it.
        copy(persistent_rows),
        max_num_keys,
        sizehint!(Vector{Int}(), max_num_keys),
    )
end

"""
Check if cache is empty.
"""
function Base.isempty(cache::RowCache)
    return isempty(cache.temp_cache)
end

"""
Erases the cache, pinned rows included, returning it to its constructed capacity.
"""
function Base.empty!(cache::RowCache)
    empty!(cache.temp_cache)
    empty!(cache.access_order)
    empty!(cache.persistent_cache_keys)
    return
end

"""
Checks if `key` is present as a key of the dictionary in `cache`

# Arguments
- `cache::RowCache`:
        cache where data is stored.
- `key::Int`:
        row number (corresponds to the enumerated branch index).
"""
function Base.haskey(cache::RowCache, key::Int)
    return haskey(cache.temp_cache, key)
end

"""
Allocates vector as row of the matrix saved in cache.

# Arguments
- `cache::RowCache`:
        cache where the row vector is going to be saved
- `val::Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}`:
        vector to be saved
- `key::Int`:
        row number (corresponding to the enumerated branch index) related to the input row vector
"""
function Base.setindex!(
    cache::RowCache{T},
    val::T,
    key::Int,
) where {T <: Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}
    # check size of the stored elements. If exceeding the limit, then one
    # element not belonging to the `persistent_cache_keys` is removed.
    check_cache_size!(cache; new_add = true)
    cache.temp_cache[key] = val
    # Update access order for LRU tracking
    push!(cache.access_order, key)
    return
end

"""
Gets the row of the stored matrix in cache.

# Arguments
- `cache::RowCache`:
        cache where the row vector is going to be saved
- `key::Int`:
        row number (corresponding to the enumerated branch index) related to the row vector.
"""
function Base.getindex(
    cache::RowCache,
    key::Int,
)
    return cache.temp_cache[key]
end

# One slot stays evictable so check_cache_size! can always make room (the constructor's
# length + 1 <= max_num_keys bound).
function _pin!(cache::RowCache, key::Int)
    key in cache.persistent_cache_keys && return
    if length(cache.persistent_cache_keys) >= cache.max_num_keys - 1
        error(
            "Cannot pin row $key: $(length(cache.persistent_cache_keys)) pinned rows " *
            "already fill max_num_keys = $(cache.max_num_keys) less one evictable slot. " *
            "Increase max_cache_size or pin fewer rows.",
        )
    end
    push!(cache.persistent_cache_keys, key)
    return
end

"""
Stores `val` for `key` and pins `key` so it is never evicted by LRU.

Used by `populate_cache` to bulk-fill rows computed via multi-RHS solves and
guarantee they stay warm for later queries. Pinned rows count against the cache
capacity: a pin that would leave no evictable slot errors.

# Arguments
- `cache::RowCache`:
        cache where the row vector is stored and pinned.
- `key::Int`:
        row number (enumerated branch index) for the row vector.
- `val`:
        the row vector (dense `Vector{Float64}` or sparsified `SparseVector{Float64}`).
"""
function set_persistent_row!(
    cache::RowCache{T},
    key::Int,
    val::T,
) where {T <: Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}
    # Make room before pinning: a `_pin!` that throws must not leave a key in
    # `persistent_cache_keys` with no row behind it.
    is_new = !haskey(cache.temp_cache, key)
    if is_new
        check_cache_size!(cache; new_add = true)
    end
    _pin!(cache, key)
    if is_new
        push!(cache.access_order, key)
    end
    cache.temp_cache[key] = val
    return
end

"""
Pin an already-stored `key` so a row populated lazily is also protected from
eviction. No-op for keys absent from `temp_cache`; errors when the pin would
consume the cache's last evictable slot.
"""
function pin_row!(cache::RowCache, key::Int)
    haskey(cache.temp_cache, key) || return
    _pin!(cache, key)
    return
end

"""
Shows the number of rows stored in cache
"""
function Base.length(cache::RowCache)
    return length(cache.temp_cache)
end

"""
Deletes a row from the stored matrix in cache not belonging to the
persistent_cache_keys set. Uses LRU (Least Recently Used) eviction strategy
based on access_order tracking.
"""
function purge_one!(cache::RowCache)
    # Use LRU eviction: find oldest non-persistent key
    for i in 1:length(cache.access_order)
        k = cache.access_order[i]
        if k ∉ cache.persistent_cache_keys && haskey(cache.temp_cache, k)
            deleteat!(cache.access_order, i)
            delete!(cache.temp_cache, k)
            return
        end
    end
    return
end

"""
Check saved rows in cache and delete one not belonging to `persistent_cache_keys`.
Errors when every row is pinned.
"""
function check_cache_size!(cache::RowCache; new_add::Bool = false)
    limit = cache.max_num_keys - Int(new_add)
    length(cache.temp_cache) > limit || return
    @info "Maximum memory reached, removing rows from cache (not belonging to `persistent_cache_keys`)." maxlog =
        1
    purge_one!(cache)
    if length(cache.temp_cache) > limit
        error(
            "RowCache holds $(length(cache.temp_cache)) rows at capacity " *
            "max_num_keys = $(cache.max_num_keys) and every one of them is pinned, so " *
            "no row can be evicted. Increase `max_cache_size` or pin fewer rows.",
        )
    end
    return
end

"""
    _cached_row(compute_row, cache, cache_lock, row, cutoff) -> RowCacheValue

Return the cached row object for `row` itself (no copy), computing, applying `cutoff` to, and
inserting it on a miss. Acquires `cache_lock` to test for a hit, runs `compute_row` outside the
lock (solves dominate the cost), then takes the lock again to insert; a concurrent producer that
wins the insert race wins, and every caller gets the winner's stored row.
"""
function _cached_row(
    compute_row,
    cache::RowCache,
    cache_lock::ReentrantLock,
    row::Int,
    cutoff::SparsificationCutoff,
)
    @lock cache_lock begin
        haskey(cache, row) && return cache.temp_cache[row]
    end
    stored = apply_cutoff(cutoff, compute_row())
    @lock cache_lock begin
        haskey(cache, row) && return cache.temp_cache[row]
        cache[row] = stored
        return stored
    end
end
