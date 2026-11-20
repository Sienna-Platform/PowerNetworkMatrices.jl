"""
    Tier2Cache

The caches of post-contingency networks on topologies other than the base network, and the
base solves their tier-2 factors share. The Woodbury factors and rows of modifications from
the base network, those of topologies and of contingencies on the base topology, live beside
it in the caller's tier-1 caches, a `Dict{NetworkModification, WoodburyFactors}` and a
`Dict{NetworkModification, RowCache{RowCacheValue}}`, which the functions on a `Tier2Cache`
take as arguments.

Every function on a `Tier2Cache` requires the caller to hold the `solver_lock` of the
`FactorCore` the cache serves; the lock guards the cache and the tier-1 caches together.

# Fields
- `tier2_factor_cache::Dict{PostContingencyNetwork, WoodburyFactors}`:
        Tier-2 Woodbury factors of each post-contingency network on a topology other than
        the base network. A contingency that changes arcs both inside and outside the
        islands holding a reference bus of a topology that islands the network has its
        factors stored under its narrowed network, the contingency without its arcs outside
        those islands. Those on the base network live in the tier-1 factor cache, under
        their tier-2 modification.
- `tier2_row_caches::Dict{PostContingencyNetwork, RowCache{RowCacheValue}}`:
        One `RowCache` per post-contingency network on a topology other than the base
        network; those on the base network use the tier-1 row caches.
- `column_cache::Dict{Tuple{Int, Int}, Vector{Float64}}`:
        Base solves `B⁻¹(e_p − e_q)` keyed by the bus-position pair `(p, q)`, `p < q`,
        shared by every network's tier-2 factors.
- `column_last_use::Dict{Tuple{Int, Int}, Int}`:
        For each key of `column_cache`, the value of `column_clock` at its last lookup.
- `column_clock::Base.RefValue{Int}`:
        Count of column cache lookups and insertions; it orders `column_last_use`.
- `column_cache_bytes::Base.RefValue{Int}`:
        Bytes held by `column_cache`.
- `max_column_cache_bytes::Int`:
        Bound on `column_cache_bytes`; the least recently used columns are evicted to keep
        it.
- `column_hits::Base.RefValue{Int}`, `column_misses::Base.RefValue{Int}`:
        Lookups of `column_cache` served from it and solved, since the last
        [`clear_caches!`](@ref).
"""
struct Tier2Cache
    tier2_factor_cache::Dict{PostContingencyNetwork, WoodburyFactors}
    tier2_row_caches::Dict{PostContingencyNetwork, RowCache{RowCacheValue}}
    column_cache::Dict{Tuple{Int, Int}, Vector{Float64}}
    column_last_use::Dict{Tuple{Int, Int}, Int}
    column_clock::Base.RefValue{Int}
    column_cache_bytes::Base.RefValue{Int}
    max_column_cache_bytes::Int
    column_hits::Base.RefValue{Int}
    column_misses::Base.RefValue{Int}
end

"""
    Tier2Cache(max_column_cache_bytes::Int) -> Tier2Cache

Empty caches whose base solves are bounded by `max_column_cache_bytes`.
"""
Tier2Cache(max_column_cache_bytes::Int) = Tier2Cache(
    Dict{PostContingencyNetwork, WoodburyFactors}(),
    Dict{PostContingencyNetwork, RowCache{RowCacheValue}}(),
    Dict{Tuple{Int, Int}, Vector{Float64}}(),
    Dict{Tuple{Int, Int}, Int}(),
    Ref(0),
    Ref(0),
    max_column_cache_bytes,
    Ref(0),
    Ref(0),
)

get_tier2_factor_cache(c::Tier2Cache) = c.tier2_factor_cache
get_tier2_row_caches(c::Tier2Cache) = c.tier2_row_caches
get_column_cache(c::Tier2Cache) = c.column_cache
get_column_last_use(c::Tier2Cache) = c.column_last_use
get_column_clock(c::Tier2Cache) = c.column_clock
get_column_cache_bytes(c::Tier2Cache) = c.column_cache_bytes
get_max_column_cache_bytes(c::Tier2Cache) = c.max_column_cache_bytes
get_column_hits(c::Tier2Cache) = c.column_hits
get_column_misses(c::Tier2Cache) = c.column_misses

# --- Column cache ---

"""
    _cached_column!(cache::Tier2Cache, core::FactorCore, p, q) -> Vector{Float64}

The base solve `B⁻¹(e_a − e_b)` for `(a, b) = (min(p, q), max(p, q))` in full-bus space,
from the column cache or solved on `core` and cached on a miss. For `p > q` this is
`−B⁻¹(e_p − e_q)`; the caller negates it. The returned vector is the cached one and must not
be mutated. Inserting evicts the least recently used columns to keep the cache within
`max_column_cache_bytes`; a column larger than the bound is returned without being cached.
The caller holds `core.solver_lock`.
"""
function _cached_column!(cache::Tier2Cache, core::FactorCore, p::Int, q::Int)
    key = (min(p, q), max(p, q))
    column = get(get_column_cache(cache), key, nothing)
    if !isnothing(column)
        get_column_hits(cache)[] += 1
        _touch_column!(cache, key)
        return column
    end
    get_column_misses(cache)[] += 1
    column = with_solver(
        core.K,
        core.work_ba_col,
        core.temp_data,
        core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        lin_solve = _solve_node_pair!(
            K_solver,
            work_ba_col,
            core.bus_to_valid_idx,
            key[1],
            key[2],
            1.0,
        )
        return _gather_to_buses!(
            Vector{Float64}(undef, length(temp_data)),
            core.valid_ix,
            lin_solve,
        )
    end
    _insert_column!(cache, key, column)
    return column
end

function _insert_column!(cache::Tier2Cache, key::Tuple{Int, Int}, column::Vector{Float64})
    columns = get_column_cache(cache)
    last_use = get_column_last_use(cache)
    bytes = get_column_cache_bytes(cache)
    limit = get_max_column_cache_bytes(cache)
    column_bytes = sizeof(column)
    column_bytes > limit && return
    while bytes[] + column_bytes > limit
        oldest = _least_recently_used(last_use)
        delete!(last_use, oldest)
        bytes[] -= sizeof(pop!(columns, oldest))
    end
    columns[key] = column
    _touch_column!(cache, key)
    bytes[] += column_bytes
    return
end

function _touch_column!(cache::Tier2Cache, key::Tuple{Int, Int})
    clock = get_column_clock(cache)
    clock[] += 1
    get_column_last_use(cache)[key] = clock[]
    return
end

# Eviction follows a miss, so this linear scan costs far less than the sparse solve before it.
function _least_recently_used(last_use::Dict{Tuple{Int, Int}, Int})
    oldest = (0, 0)
    oldest_use = typemax(Int)
    for (key, use) in last_use
        if use < oldest_use
            oldest = key
            oldest_use = use
        end
    end
    return oldest
end

# --- Tier-1 factors and rows ---

"""
    _tier1_factors!(woodbury_cache, core, mod) -> WoodburyFactors

The Woodbury factors of `mod` on the base network, from `woodbury_cache` or computed on
`core` and cached on a miss. The caller holds `core.solver_lock`; the solves re-enter it.
"""
function _tier1_factors!(
    woodbury_cache::Dict{NetworkModification, WoodburyFactors},
    core::FactorCore,
    mod::NetworkModification,
)::WoodburyFactors
    # The do-block form: the three-argument `get!` evaluates its default eagerly, which would
    # run the solves on every hit.
    return get!(woodbury_cache, mod) do
        _compute_woodbury_factors(core, _arc_modification_vector(mod))
    end
end

# Plan of monitored arc `m`'s row under `mod`, from the base network. The modification's
# factors are cached even when this row is the zero row, since its other rows need them. The
# base network needs no correction, so it has no factors.
function _modification_row_plan(
    woodbury_cache::Dict{NetworkModification, WoodburyFactors},
    core::FactorCore,
    m::Int,
    mod::NetworkModification,
)::_RowPlan
    wf = mod == _BASE_NETWORK ? nothing : _tier1_factors!(woodbury_cache, core, mod)
    return _base_topology_row_plan(core, m, _arc_modification_vector(mod), wf)
end

# A row cache for one network's rows: one bus-long row per monitored arc.
function _network_row_cache(core::FactorCore, max_cache_size_bytes::Int)
    return RowCache(
        max_cache_size_bytes,
        Set{Int}(),
        length(get_bus_axis(core)) * sizeof(Float64),
        length(get_arc_axis(core)),
    )
end

# --- Post-contingency rows ---

function _cached_row_at(caches::Dict, key, m::Int)
    rc = get(caches, key, nothing)
    (isnothing(rc) || !haskey(rc, m)) && return nothing
    return rc[m]
end

# `net`'s rows live, on the base topology, in the tier-1 row cache of its tier-2 modification;
# otherwise in the network's own tier-2 row cache.
function _cached_post_contingency_row(
    cache::Tier2Cache,
    row_caches::Dict{NetworkModification, RowCache{RowCacheValue}},
    net::PostContingencyNetwork,
    m::Int,
)
    _on_base_topology(net) && return _cached_row_at(row_caches, net.tier2, m)
    return _cached_row_at(get_tier2_row_caches(cache), net, m)
end

function _post_contingency_row_cache!(
    cache::Tier2Cache,
    row_caches::Dict{NetworkModification, RowCache{RowCacheValue}},
    core::FactorCore,
    max_cache_size_bytes::Int,
    net::PostContingencyNetwork,
)
    new_cache = () -> _network_row_cache(core, max_cache_size_bytes)
    _on_base_topology(net) && return get!(new_cache, row_caches, net.tier2)
    return get!(new_cache, get_tier2_row_caches(cache), net)
end

"""
    _arcs_outside_reference_islands(core, topology_wf, net) -> Set{Int}

On a topology K whose factors island the network, the arcs whose susceptance the contingency
changes that lie outside the islands of K holding a reference bus. Such an arc joins only
buses K cut off: taking it out changes no row of an arc inside those islands, and rows of
arcs outside them are zero. Arcs with no susceptance change never enter the Woodbury update.
Arcs are placed by their ends, so an arc with no susceptance, whose BA column is empty, is
placed by the incidence matrix.
"""
function _arcs_outside_reference_islands(
    core::FactorCore,
    topology_wf::WoodburyFactors,
    net::PostContingencyNetwork,
)::Set{Int}
    outside = Set{Int}()
    labels = topology_wf.bus_island_labels
    reference = Set{Int}(labels[p] for p in _ref_bus_positions(core))
    for mod in net.tier2.arc_modifications
        iszero(mod.delta_b) && continue
        e = mod.arc_index
        any(b -> labels[b] in reference, _arc_ends(core, e)) || push!(outside, e)
    end
    return outside
end

# `net` with each arc in `arcs` as K leaves it: the post-contingency modification takes K's
# entry for the arc, or none, and the tier-2 modification is derived again from it.
function _without_arcs(net::PostContingencyNetwork, arcs::Set{Int})
    post = net.post_contingency
    at_k = Dict(m.arc_index => m for m in net.topology.arc_modifications)
    arc_mods = ArcModification[]
    for m in post.arc_modifications
        if !(m.arc_index in arcs)
            push!(arc_mods, m)
        elseif haskey(at_k, m.arc_index)
            push!(arc_mods, at_k[m.arc_index])
        end
    end
    narrowed = NetworkModification(
        post.label,
        arc_mods,
        ShuntModification[post.shunt_modifications...],
        post.is_islanding,
    )
    return PostContingencyNetwork(
        _Tier2Derived(),
        net.topology,
        narrowed,
        _tier2_modification(narrowed, net.topology),
        net.contingency_id,
    )
end

# Plan of monitored arc `m`'s row on K, which is its row on `net` when the contingency
# changes no arc, or only arcs outside the islands of K holding a reference bus.
function _topology_row_plan(
    arc_sus::Vector{Float64},
    m::Int,
    net::PostContingencyNetwork,
    topology_wf::WoodburyFactors,
)::_RowPlan
    b_topology =
        _post_modification_susceptance(arc_sus, m, _arc_modification_vector(net.topology))
    return _RowPlan(_CORRECTED_ROW, topology_wf, b_topology)
end

# Plan of monitored arc `m`'s row on `net` when the contingency changes no arc of K's islands
# holding a reference bus: zero for an arc outside those islands, else K's row.
function _unchanged_topology_row_plan(
    core::FactorCore,
    arc_sus::Vector{Float64},
    m::Int,
    net::PostContingencyNetwork,
    topology_wf::WoodburyFactors,
)::_RowPlan
    _outside_reference_islands(core, topology_wf, m) && return _RowPlan(_ZERO_ROW)
    return _topology_row_plan(arc_sus, m, net, topology_wf)
end

"""
    _post_contingency_row_plan(cache, core, woodbury_cache, m, net, topology_factors) -> _RowPlan

Plan of monitored arc `m`'s row on `net`, resolving and caching the factors it needs: on the
base topology the plan of the tier-2 modification from the base network; otherwise K's
factors in `woodbury_cache` and the tier-2 factors in `cache`. When K islands the network
and the contingency changes arcs both inside and outside the islands holding a reference
bus, the tier-2 factors are those of its narrowed network (see `_without_arcs`), cached
under it. `topology_factors` memoizes K's factors by object identity within one call, so
networks sharing a topology object hash it once. The caller holds `core.solver_lock`.
"""
function _post_contingency_row_plan(
    cache::Tier2Cache,
    core::FactorCore,
    woodbury_cache::Dict{NetworkModification, WoodburyFactors},
    m::Int,
    net::PostContingencyNetwork,
    topology_factors::IdDict{NetworkModification, WoodburyFactors},
)::_RowPlan
    _on_base_topology(net) &&
        return _modification_row_plan(woodbury_cache, core, m, net.tier2)
    arc_sus = core.arc_susceptances
    _post_contingency_arc_out(arc_sus, core, m, net) && return _RowPlan(_ZERO_ROW)
    topology_wf = get!(topology_factors, net.topology) do
        _tier1_factors!(woodbury_cache, core, net.topology)
    end
    isempty(net.tier2.arc_modifications) &&
        return _unchanged_topology_row_plan(core, arc_sus, m, net, topology_wf)
    live = net
    narrowed = false
    if topology_wf.is_islanding
        outside = _arcs_outside_reference_islands(core, topology_wf, net)
        if !isempty(outside)
            n_changed = count(mod -> !iszero(mod.delta_b), net.tier2.arc_modifications)
            length(outside) == n_changed &&
                return _unchanged_topology_row_plan(core, arc_sus, m, net, topology_wf)
            live = _without_arcs(net, outside)
            narrowed = true
        end
    end
    # The tier-2 factors come before the zero row of an arc K cuts off: building them throws
    # when the contingency reconnects a bus K cuts off, so every row of the network throws.
    tier2_wf = get!(get_tier2_factor_cache(cache), live) do
        _tier2_factors(
            (p, q) -> _cached_column!(cache, core, p, q),
            core,
            topology_wf,
            live.topology,
            live.tier2,
            live.post_contingency,
        )
    end
    # The narrowed network keeps every outage of K, so the buses K cut off stay cut off.
    if narrowed && !tier2_wf.is_islanding
        error(
            "Contingency $(net.contingency_id) on topology \"$(net.topology.label)\" \
            without its arcs outside the islands holding a reference bus leaves no bus cut \
            off, though the topology cuts buses off.",
        )
    end
    _outside_reference_islands(core, topology_wf, m) && return _RowPlan(_ZERO_ROW)
    _outside_reference_islands(core, tier2_wf, m) && return _RowPlan(_ZERO_ROW)
    b_post = _post_contingency_susceptance(arc_sus, m, live)
    # With no columns and no bus dead on K ∪ c (a bus dead on K counts), the row is K's row
    # scaled to K ∪ c. Otherwise it takes the tier-2 row, which zeroes the dead buses.
    if iszero(_woodbury_rank(tier2_wf)) && !tier2_wf.is_islanding
        return _RowPlan(_CORRECTED_ROW, topology_wf, b_post)
    end
    return _RowPlan(_CORRECTED_ROW, topology_wf, tier2_wf, b_post)
end

# Rows of monitored arc `m` on each network in `nets`, before the cutoff, from one solve of
# its base column.
function _post_contingency_rows(
    cache::Tier2Cache,
    core::FactorCore,
    woodbury_cache::Dict{NetworkModification, WoodburyFactors},
    m::Int,
    nets::Vector{PostContingencyNetwork},
)::Vector{Vector{Float64}}
    topology_factors = IdDict{NetworkModification, WoodburyFactors}()
    plans = [
        _post_contingency_row_plan(cache, core, woodbury_cache, m, net, topology_factors)
        for net in nets
    ]
    return _rows_from_plans(core, m, plans)
end

"""
    _post_contingency_ptdf_rows!(cache, core, woodbury_cache, row_caches, max_cache_size_bytes, m, networks) -> Vector{RowCacheValue}

The PTDF rows of monitored arc `m` on each network in `networks`, with `core`'s cutoff
applied, from at most one solve of the arc's base column. Rows are read from and written to
the row caches, `row_caches` for networks on the base topology and `cache` for the others,
whose new row caches hold `max_cache_size_bytes`. Row caches are created only once the rows
exist, so a call that throws leaves none. The caller holds `core.solver_lock` and owns the
returned copies.
"""
function _post_contingency_ptdf_rows!(
    cache::Tier2Cache,
    core::FactorCore,
    woodbury_cache::Dict{NetworkModification, WoodburyFactors},
    row_caches::Dict{NetworkModification, RowCache{RowCacheValue}},
    max_cache_size_bytes::Int,
    m::Int,
    networks::AbstractVector{PostContingencyNetwork},
)::Vector{RowCacheValue}
    # Each distinct network gets one slot; `slot[i]` is the slot of `networks[i]`.
    slot_of = Dict{PostContingencyNetwork, Int}()
    slot = Vector{Int}(undef, length(networks))
    values = RowCacheValue[]
    pending = PostContingencyNetwork[]
    pending_slots = Int[]
    for (i, net) in enumerate(networks)
        k = get!(slot_of, net, length(slot_of) + 1)
        slot[i] = k
        k <= length(values) && continue
        cached = _cached_post_contingency_row(cache, row_caches, net, m)
        if isnothing(cached)
            push!(values, Float64[])
            push!(pending, net)
            push!(pending_slots, k)
        else
            push!(values, cached)
        end
    end
    if !isempty(pending)
        cutoff = get_cutoff(core)
        rows = _post_contingency_rows(cache, core, woodbury_cache, m, pending)
        for (net, k, row) in zip(pending, pending_slots, rows)
            value = apply_cutoff(cutoff, row)
            rc = _post_contingency_row_cache!(
                cache,
                row_caches,
                core,
                max_cache_size_bytes,
                net,
            )
            rc[m] = value
            values[k] = value
        end
    end
    return RowCacheValue[copy(values[k]) for k in slot]
end

# --- Cache statistics ---

_cached_bytes(row::Vector{Float64}) = sizeof(row)
_cached_bytes(row::SparseArrays.SparseVector{Float64}) =
    sizeof(SparseArrays.nonzeros(row)) + sizeof(SparseArrays.nonzeroinds(row))

_cached_bytes(basis::WoodburyBasis) =
    sizeof(basis.p) + sizeof(basis.q) + sizeof(basis.scale) + sizeof(basis.D)

_cached_bytes(wf::WoodburyFactors) =
    sizeof(wf.Z) + sizeof(wf.W_inv) + sizeof(wf.arc_indices) + sizeof(wf.delta_b) +
    sizeof(wf.arc_out) + sizeof(wf.bus_island_labels) + _cached_bytes(wf.basis)

function _factor_statistics(factors::Dict)
    return (;
        entries = length(factors),
        bytes = sum(_cached_bytes, values(factors); init = 0),
    )
end

function _row_statistics(row_caches)
    entries = 0
    bytes = 0
    for caches in row_caches, rc in values(caches), row in values(rc.temp_cache)
        entries += 1
        bytes += _cached_bytes(row)
    end
    return (; entries, bytes)
end

"""
    cache_statistics(cache::Tier2Cache, woodbury_cache, row_caches) -> NamedTuple

Entries and bytes of `cache` and of the tier-1 caches beside it:
- `columns = (; entries, bytes, hits, misses)`: the base solves shared by tier-2 factors,
  with the lookups served from the cache and solved since the last [`clear_caches!`](@ref);
- `tier1_factors = (; entries, bytes)`: `woodbury_cache`, the Woodbury factors on the base
  network of topologies and of contingencies on the base topology;
- `tier2_factors = (; entries, bytes)`: tier-2 factors of post-contingency networks on other
  topologies;
- `rows = (; entries, bytes)`: cached PTDF rows of every network, in `row_caches` and
  `cache`.

Bytes count the arrays held, not container overhead. The caller holds the core's
`solver_lock`.
"""
function cache_statistics(
    cache::Tier2Cache,
    woodbury_cache::Dict{NetworkModification, WoodburyFactors},
    row_caches::Dict{NetworkModification, RowCache{RowCacheValue}},
)
    columns = (;
        entries = length(get_column_cache(cache)),
        bytes = get_column_cache_bytes(cache)[],
        hits = get_column_hits(cache)[],
        misses = get_column_misses(cache)[],
    )
    return (;
        columns,
        tier1_factors = _factor_statistics(woodbury_cache),
        tier2_factors = _factor_statistics(get_tier2_factor_cache(cache)),
        rows = _row_statistics((row_caches, get_tier2_row_caches(cache))),
    )
end

# --- Eviction ---

"""
    clear_caches!(cache::Tier2Cache)

Empty the tier-2 factor, row and column caches and zero the column cache's hit and miss
counts. The caller holds the core's `solver_lock`.
"""
function clear_caches!(cache::Tier2Cache)
    empty!(get_tier2_factor_cache(cache))
    empty!(get_tier2_row_caches(cache))
    empty!(get_column_cache(cache))
    empty!(get_column_last_use(cache))
    get_column_clock(cache)[] = 0
    get_column_cache_bytes(cache)[] = 0
    get_column_hits(cache)[] = 0
    get_column_misses(cache)[] = 0
    return
end

"""
    retain_modifications!(cache::Tier2Cache, in_use)

End a build: evict every tier-2 factor and row, those of the networks in `in_use` included,
and keep the column cache with its hit and miss counts. Tier-2 factors and rows therefore
live for one build, while the base solves serve the next one. The factors of the topologies
in use live in the caller's tier-1 cache, which the caller retains. The caller holds the
core's `solver_lock`.
"""
function retain_modifications!(cache::Tier2Cache, in_use)
    empty!(get_tier2_factor_cache(cache))
    empty!(get_tier2_row_caches(cache))
    return
end
