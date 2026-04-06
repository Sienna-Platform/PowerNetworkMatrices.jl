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
