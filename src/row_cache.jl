"""
Structure used for saving the rows of the Virtual PTDF and LODF matrix.

# Arguments
- `temp_cache::Dict{Int, Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}`:
        Dictionary saving the row of the PTDF/LODF matrix
- `persistent_cache_keys::Set{Int}`:
        Set listing the rows to keep in `temp_cache`
- `max_cache_size::Int`
        Defines the maximum allowed cache size (rows*row_size)
- `max_num_keys::Int`
        Defines the maximum number of keys saved (rows of the matrix)
- `access_order::Vector{Int}`:
        Vector tracking access order for LRU eviction (most recent at end)
"""
struct RowCache{T <: Union{Vector{Float64}, SparseArrays.SparseVector{Float64}}}
    temp_cache::Dict{Int, T}
    persistent_cache_keys::Set{Int}
    max_cache_size::Int
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
        persistent_rows,
        max_cache_size,
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
Erases the cache.
"""
function Base.empty!(cache::RowCache)
    isempty(cache.temp_cache) && return
    if !isempty(cache.persistent_cache_keys)
        @warn("Calling empty! will delete entries for the persistent rows")
    end
    empty!(cache.temp_cache)
    empty!(cache.access_order)
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

"""
Stores `val` for `key` and pins `key` so it is never evicted by LRU.

Used by `populate_cache` to bulk-fill rows computed via multi-RHS solves and
guarantee they stay warm for later queries. Unlike `setindex!`, this never
triggers `purge_one!`: pinned rows are added to `persistent_cache_keys`, and a
cache populated beyond `max_num_keys` simply grows (callers should warn via
[`warn_if_over_capacity`](@ref)).

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
    if !haskey(cache.temp_cache, key)
        push!(cache.access_order, key)
    end
    cache.temp_cache[key] = val
    push!(cache.persistent_cache_keys, key)
    return
end

"""
Pin an already-stored `key` so a row populated lazily is also protected from
eviction. No-op for keys absent from `temp_cache`.
"""
function pin_row!(cache::RowCache, key::Int)
    haskey(cache.temp_cache, key) && push!(cache.persistent_cache_keys, key)
    return
end

"""
    warn_if_over_capacity(cache::RowCache)

Emit a single warning when the cache holds more rows than `max_num_keys`. This
happens when `populate_cache` pins more rows than the configured
`max_cache_size` can hold; the pinned rows remain resident (never evicted), so
the only risk is higher memory use.
"""
function warn_if_over_capacity(cache::RowCache)
    if length(cache.temp_cache) > cache.max_num_keys
        @warn "populate_cache pinned $(length(cache.temp_cache)) rows, exceeding " *
              "the cache capacity (max_num_keys = $(cache.max_num_keys)). Pinned " *
              "rows are not evicted; increase `max_cache_size` to avoid memory " *
              "pressure." maxlog = 1
    end
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
    # Fallback: if access_order is out of sync, use old method
    for k in keys(cache.temp_cache)
        if k ∉ cache.persistent_cache_keys
            delete!(cache.temp_cache, k)
            break
        end
    end
    return
end

"""
Check saved rows in cache and delete one not belonging to `persistent_cache_keys`.
"""
function check_cache_size!(cache::RowCache; new_add::Bool = false)
    if new_add
        v = 1
    else
        v = 0
    end
    if length(cache.temp_cache) > cache.max_num_keys - v
        @info "Maximum memory reached, removing rows from cache (not belonging to `persistent_cache_keys`)." maxlog =
            1
        purge_one!(cache)
    end
    return
end

"""
    cached_row_lookup(compute_row, cache, cache_lock, row, column, tol) -> value

Shared cache-fast-path / compute / double-checked-insert pattern used by
`VirtualPTDF._getindex` and `VirtualLODF._getindex`. Acquires `cache_lock`
to test for a hit, runs `compute_row` outside the lock on a miss
(KLU solves dominate the cost), then takes the lock again to insert. A
concurrent producer that wins the insert race wins; the other side
returns the winner's row.

`compute_row` is the first positional argument so callers can pass the
miss-path computation as a `do … end` block:

```julia
return cached_row_lookup(
    vlodf.cache, vlodf.cache_lock, row, column, get_tol(vlodf),
) do
    _compute_lodf_row(vlodf, row)
end
```
"""
function cached_row_lookup(
    compute_row,
    cache::RowCache,
    cache_lock::ReentrantLock,
    row::Int,
    column::Union{Int, Colon},
    tol::Float64,
)
    @lock cache_lock begin
        haskey(cache, row) && return cache.temp_cache[row][column]
    end
    row_data = compute_row()
    stored = tol > eps() ? sparsify(row_data, tol) : row_data
    @lock cache_lock begin
        haskey(cache, row) && return cache.temp_cache[row][column]
        cache[row] = stored
        return cache[row][column]
    end
end
