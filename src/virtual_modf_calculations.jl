"""
The Virtual Multiple Outage Distribution Factor (VirtualMODF) structure computes
post-contingency PTDF rows lazily for registered contingencies using the
Woodbury matrix identity (van Dijk et al. Eq. 29).

Contingencies are resolved from PSY.UnplannedOutage supplemental attributes at construction
time. After registration, the System is not needed for queries.

Internally the factorization and topology live in a private
[`VirtualFactorCore`](@ref) held in the `core` field; the VirtualMODF only adds
the contingency-specific caches. A single core can be shared with a
`VirtualPTDF` / `VirtualLODF` so the factorization is computed once.

Caching is two-tiered:
- Woodbury factors (M KLU solves) are cached per contingency
- PTDF rows (1 KLU solve each) are cached per (monitored_arc, contingency) via
  one RowCache per contingency

A contingency on a topology other than the base network (a [`PostContingencyNetwork`](@ref))
caches its tier-2 factors and rows in their own caches; see
[`get_post_contingency_ptdf_rows`](@ref).

# Thread-safety

Concurrent `getindex` is safe but serialized: the core's `solver_lock` (a
`ReentrantLock`) is held for the full body of `getindex`, `clear_caches!`, and
`clear_all_caches!`, so Dict mutations on the cache structures and the libklu
solves it wraps all run under a single mutex.

# Fields
- `core::VirtualFactorCore`:
        Shared factorization/topology container (see [`VirtualFactorCore`](@ref)).
        `PTDF_A_diag` and `branch_susceptances_by_arc` are computed lazily on the
        core and shared with any other wrapper using the same core.
- `contingency_cache::Dict{Int, ContingencySpec}`:
        Resolved contingencies keyed by outage UUID.
- `woodbury_cache::Dict{NetworkModification, WoodburyFactors}`:
        Precomputed Woodbury factors keyed by modification.
- `row_caches::Dict{NetworkModification, RowCache{RowCacheValue}}`:
        One `RowCache` per modification.
- `tier2_factor_cache::Dict{PostContingencyNetwork, WoodburyFactors}`:
        Tier-2 Woodbury factors of each post-contingency network on a topology other than
        the base network. Those on the base network live in `woodbury_cache`, under their
        tier-2 modification.
- `tier2_row_caches::Dict{PostContingencyNetwork, RowCache{RowCacheValue}}`:
        One `RowCache` per post-contingency network on a topology other than the base
        network; those on the base network use `row_caches`.
- `max_cache_size_bytes::Int`:
        Max cache size in bytes per contingency.
"""
struct VirtualMODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    core::VirtualFactorCore{Ax, L, K}
    contingency_cache::Dict{Int, ContingencySpec}
    woodbury_cache::Dict{NetworkModification, WoodburyFactors}
    row_caches::Dict{NetworkModification, RowCache{RowCacheValue}}
    tier2_factor_cache::Dict{PostContingencyNetwork, WoodburyFactors}
    tier2_row_caches::Dict{PostContingencyNetwork, RowCache{RowCacheValue}}
    max_cache_size_bytes::Int
end

# --- Field forwarding ---

# Forward non-wrapper field reads to the core. The lazy `PTDF_A_diag` /
# `branch_susceptances_by_arc` are special-cased so reads trigger the core's
# lazy compute (and back-compat reads like `vmodf.PTDF_A_diag` keep working).
function Base.getproperty(vmodf::VirtualMODF, name::Symbol)
    if name === :core ||
       name === :contingency_cache ||
       name === :woodbury_cache ||
       name === :row_caches ||
       name === :tier2_factor_cache ||
       name === :tier2_row_caches ||
       name === :max_cache_size_bytes
        return getfield(vmodf, name)
    end
    name === :PTDF_A_diag && return get_PTDF_A_diag(get_core(vmodf))
    name === :branch_susceptances_by_arc &&
        return get_branch_susceptances_by_arc(get_core(vmodf))
    return getproperty(get_core(vmodf), name)
end

# --- Accessors ---

# Field getters. `getfield` is confined to these (and the `getproperty` hook
# above), so the rest of the file reads the wrapper's own fields through proper
# accessors instead of `getfield`.
get_core(M::VirtualMODF) = getfield(M, :core)
get_woodbury_cache(M::VirtualMODF) = getfield(M, :woodbury_cache)
get_row_caches(M::VirtualMODF) = getfield(M, :row_caches)
get_tier2_factor_cache(M::VirtualMODF) = getfield(M, :tier2_factor_cache)
get_tier2_row_caches(M::VirtualMODF) = getfield(M, :tier2_row_caches)
get_max_cache_size_bytes(M::VirtualMODF) = getfield(M, :max_cache_size_bytes)

# Accessors forward to the core; VirtualPTDF and VirtualMODF share them.
for f in (
    :get_axes, :get_lookup, :get_ref_bus, :get_ref_bus_position,
    :get_network_reduction_data, :get_branch_catalog, :get_arc_lookup, :get_bus_lookup,
    :get_arc_axis, :get_bus_axis, :get_tol, :get_cutoff, :get_system_uuid,
    :_get_BA, :_get_arc_susceptances, :_get_valid_ix,
)
    @eval $f(M::Union{VirtualPTDF, VirtualMODF}) = $f(get_core(M))
end

"""
$(TYPEDSIGNATURES)

Return `H[e, e]` for each arc `e`. Triggers the core's lazy compute on first
call and returns the cached vector thereafter.
"""
get_PTDF_A_diag(vmodf::VirtualMODF) = get_PTDF_A_diag(get_core(vmodf))

"""
    get_registered_contingencies(vmodf::VirtualMODF) -> Dict{Int, ContingencySpec}

Return the cached contingency registrations for inspection.
"""
get_registered_contingencies(vmodf::VirtualMODF) = getfield(vmodf, :contingency_cache)

# --- Base interface ---

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualMODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    print(
        io,
        "VirtualMODF with $(length(get_registered_contingencies(array))) registered contingencies",
    )
    return
end

# A VirtualMODF holds no dense data, so the generic `PowerNetworkMatrix` method, which prints
# `data`, does not apply.
function Base.show(io::IO, array::VirtualMODF)
    summary(io, array)
    print(
        io,
        " with $(length(get_registered_contingencies(array))) registered contingencies",
    )
    return
end

function Base.isempty(vmodf::VirtualMODF)
    return isempty(get_registered_contingencies(vmodf))
end

function Base.size(vmodf::VirtualMODF)
    core = get_core(vmodf)
    return (length(core.axes[1]), length(core.axes[2]))
end

Base.setindex!(::VirtualMODF, _, idx...) = error("Operation not supported by VirtualMODF")
Base.setindex!(::VirtualMODF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualMODF")

# --- Constructors ---

"""
    VirtualMODF(sys::PSY.System; kwargs...) -> VirtualMODF

Build a VirtualMODF from a PowerSystems System. Automatically registers all
Unplanned outage supplemental attributes found in the system.

The buses of every outaged component and the components each outage declares
monitored (`get_monitored_components`) are automatically added to the irreducible
set before the base `Ybus` is built. `WardReduction.study_buses` define the
retained network rather than exempting buses, so — when outages are
auto-registered (`automatically_register_outages`) — they are validated against
the contingencies and a branch outside the study area is an error, not silently
covered. This is mandatory: the
ABA/Woodbury solve runs on the reduced network, so a branch in a contingency must
survive every reduction step, including the zero-impedance reduction that is
auto-applied during `Ybus` construction.

# Keyword Arguments
- `linear_solver::String = _default_linear_solver()`: Linear solver for the
        ABA factorization. Options: "KLU", "AppleAccelerate". Defaults to
        "AppleAccelerate" on macOS and "KLU" elsewhere.
- `tol::Union{Float64, AutoTolerance}`: Tolerance for row sparsification.
        A `Float64` applies a fixed absolute cutoff; an [`AutoTolerance`](@ref)
        (the default) applies a relative per-row cutoff so requested columns stay
        sparse on large systems.
- `max_cache_size::Int`: Max cache size in MiB per contingency (default: MAX_CACHE_SIZE_MiB)
- `network_reductions::Vector{NetworkReduction}`: Network reductions to apply
- `irreducible_buses::IS.ComponentSelector`: Components whose buses the network reductions must
        keep (see [`get_irreducible_bus_numbers`](@ref)). Default: `IS.make_selector()`
- `automatically_register_outages::Bool`: Register all system unplanned outage attributes (default: true)
"""
function VirtualMODF(
    sys::PSY.System;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    irreducible_buses::IS.ComponentSelector = IS.make_selector(),
    automatically_register_outages::Bool = true,
    kwargs...,
)
    user_irreducible = get_irreducible_bus_numbers(sys, irreducible_buses)
    resolve_linear_solver(linear_solver)

    # Split so the zero-impedance entry reaches `Ybus` (which applies it first) instead of
    # the manual loop below, which would apply it a second time.
    network_reductions, zero_impedance_reduction =
        _split_zero_impedance_reduction(network_reductions)

    # Outage/monitored buses are auto-protected so contingency branches survive
    # reduction. Collect them from `sys` before building the base Ybus and fold
    # them into the irreducible set so the auto-applied zero-impedance reduction
    # never merges away a monitored/outaged branch endpoint in the first place.
    # Registering an outage with previously-unseen monitored components after
    # construction shifts this set and requires rebuilding the MODF.
    protected_buses = _collect_protected_buses(sys)
    applied_irreducible = union(user_irreducible, protected_buses)

    # Build the base Ybus with the combined irreducible set (zero-impedance reduction
    # auto-applied here honors it); this is the starting point for further reductions.
    Ymatrix = _build_ybus(
        sys,
        applied_irreducible;
        network_reductions = NetworkReduction[zero_impedance_reduction],
        kwargs...,
    )

    # radial/degree-two read the container's irreducible set (already seeded above by
    # `_build_ybus`). Ward's `study_buses` defines the retained network rather
    # than exempting buses, so it is validated against the contingencies — but only the ones
    # this MODF will actually register.
    automatically_register_outages &&
        _validate_ward_contingency_coverage(network_reductions, sys)
    for reduction in network_reductions
        Ymatrix = build_reduced_ybus(Ymatrix, sys, reduction)
    end

    core = VirtualFactorCore(
        Ymatrix;
        linear_solver = linear_solver,
        tol = tol,
        system_uuid = PSY.get_system_uuid(sys),
    )

    return VirtualMODF(
        core,
        sys;
        max_cache_size = max_cache_size,
        automatically_register_outages = automatically_register_outages,
    )
end

"""
    VirtualMODF(core::VirtualFactorCore, sys::PSY.System; kwargs...) -> VirtualMODF

Wrap an existing [`VirtualFactorCore`](@ref) in a VirtualMODF and register the
system's outages, sharing one factorization across multiple virtual matrices.

The core arrives already reduced and factorized, so this path cannot protect the outaged
and monitored buses the way `VirtualMODF(sys; ...)` does. Registration therefore validates
each outage against the core's surviving arcs and errors when a reduction ate one. Reduce
the core with those components in the `irreducible_buses` selector, or build from the system
instead.
"""
function VirtualMODF(
    core::VirtualFactorCore,
    sys::PSY.System;
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    automatically_register_outages::Bool = true,
)
    max_cache_bytes = max_cache_size * MiB
    vmodf = VirtualMODF(
        core,
        Dict{Int, ContingencySpec}(),
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache{RowCacheValue}}(),
        Dict{PostContingencyNetwork, WoodburyFactors}(),
        Dict{PostContingencyNetwork, RowCache{RowCacheValue}}(),
        max_cache_bytes,
    )

    # Auto-register all outage attributes from the system
    automatically_register_outages && _register_all_outages!(vmodf, sys)

    return vmodf
end

"""
    VirtualMODF(vptdf::VirtualPTDF, sys::PSY.System; kwargs...) -> VirtualMODF

Build a VirtualMODF that reuses an existing `VirtualPTDF`'s factorization. The
two objects share the same [`VirtualFactorCore`](@ref), so the ABA matrix is
factorized only once. Same reduction caveat as VirtualMODF(core, sys).
"""
function VirtualMODF(vptdf::VirtualPTDF, sys::PSY.System; kwargs...)
    return VirtualMODF(get_core(vptdf), sys; kwargs...)
end

"""
    _validate_transmission_survived(sys, outage, mod)

For an outage whose ACTransmission components resolved to no arc modifications: throw when
any is in service (a reduction eliminated it, so every query would return the base-case row);
warn when all are out of service (a legitimate no-op). This is the only survive-check
possible on a core that was reduced before outages were known.
"""
function _validate_transmission_survived(
    sys::PSY.System,
    outage::PSY.UnplannedOutage,
    mod::NetworkModification,
)
    isempty(mod.arc_modifications) || return
    transmission =
        collect(
            PSY.get_associated_components(sys, outage; component_type = PSY.ACTransmission),
        )
    isempty(transmission) && return
    # `get_available` is derived for a ThreeWindingTransformer, so this one predicate covers
    # every transmission type without a type check.
    in_service = filter(PSY.get_available, transmission)
    if isempty(in_service)
        @warn "Outage (label=$(mod.label)) references only out-of-service transmission " *
              "component(s) $(join(sort!([PSY.get_name(c) for c in transmission]), ", ")); " *
              "they carry no flow, so this contingency is a no-op and every query returns " *
              "the base-case row." maxlog = 5
        return
    end
    names = sort!([PSY.get_name(c) for c in in_service])
    throw(
        IS.ConflictingInputsError(
            "Outage (label=$(mod.label)) references in-service transmission component(s) \
            $(join(names, ", ")) that a network reduction eliminated; every query would \
            return the base-case row. Build with \
            VirtualMODF(sys; network_reductions = ...) or select them with \
            irreducible_buses.",
        ),
    )
end

# --- Outage registration ---

_warn_or_rethrow_failed_outage(e::ErrorException) =
    @warn "Could not register outage: $(e.msg)"
_warn_or_rethrow_failed_outage(e) = rethrow()

"""
    _register_all_outages!(vmodf, sys)

Bulk-register all unplanned outage supplemental attributes in the system.
Called automatically by the VirtualMODF constructor.
"""
function _register_all_outages!(vmodf::VirtualMODF, sys::PSY.System)
    count = 0
    for outage in PSY.get_supplemental_attributes(PSY.UnplannedOutage, sys)
        try
            _register_outage!(vmodf, sys, outage)
            count += 1
        catch e
            _warn_or_rethrow_failed_outage(e)
        end
    end

    if iszero(count)
        @warn "No outage supplemental attributes found in system. " *
              "VirtualMODF contingency cache is empty."
    else
        @info "Registered $count contingencies from system outage attributes."
    end
    return
end

"""
    _register_outage!(vmodf, sys, outage) -> ContingencySpec

Resolve an unplanned outage supplemental attribute to a ContingencySpec and cache it.
Delegates to `NetworkModification(mat, sys, outage)` for the resolution logic.
"""
function _register_outage!(vmodf::VirtualMODF, sys::PSY.System, outage::PSY.UnplannedOutage)
    contingency_cache = get_registered_contingencies(vmodf)
    outage_id = IS.get_id(outage)
    if haskey(contingency_cache, outage_id)
        @warn "Unplanned outage with UUID $(outage_id) is already registered; skipping."
        return
    end
    mod = NetworkModification(vmodf, sys, outage)
    _validate_transmission_survived(sys, outage, mod)
    contingency_cache[outage_id] = ContingencySpec(outage_id, mod)
    return
end

# --- Woodbury factor computation ---

"""
    _get_woodbury_factors(vmodf, mod) -> WoodburyFactors

Return cached Woodbury factors for a modification, computing them on a miss.
Caller holds the core `solver_lock`; the inner `_compute_woodbury_factors`
re-enters that same lock (it's a `ReentrantLock`).
"""
function _get_woodbury_factors(vmodf::VirtualMODF, mod::NetworkModification)
    # Use the do-block form, NOT `get!(dict, key, default)`: Julia evaluates
    # function arguments eagerly, so the 3-arg form would run the M KLU solves
    # on every call (cache hit included), defeating the cache.
    return get!(get_woodbury_cache(vmodf), mod) do
        _compute_woodbury_factors(vmodf, mod.arc_modifications)
    end
end

"""
    _compute_modf_entry(vmodf, monitored_idx, mod) -> Vector{Float64}

Compute the post-modification PTDF row for a monitored arc under the given modification.

For N-1 contingencies, the result satisfies:
    post_ptdf[mon, :] = pre_ptdf[mon, :] + LODF[mon, e] * pre_ptdf[e, :]
"""
function _compute_modf_entry(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    mod::NetworkModification,
)::Vector{Float64}
    return only(_post_contingency_rows(vmodf, monitored_idx, [_base_topology_network(mod)]))
end

# Per-modification row cache: one n_bus-long row per monitored arc.
_new_modf_row_cache(vmodf::VirtualMODF) =
    RowCache(get_max_cache_size_bytes(vmodf), Set{Int}(), size(vmodf)[2] * sizeof(Float64))

# --- getindex: by integer monitored index + NetworkModification ---

"""
Get the post-modification PTDF row for monitored arc `monitored_idx` under `mod`.
Uses per-modification RowCache for LRU-eviction caching.

Rows after a contingency on a scheduled topology K come from
[`get_post_contingency_ptdf_rows`](@ref), on the networks of
[`get_post_contingency_network`](@ref). Indexing with a hand-built `NetworkModification` for
K ∪ c gives a correct row, but computes the full combined update from the base network and
bypasses the reuse of K's factors.

$(TYPEDSIGNATURES)
"""
function Base.getindex(vmodf::VirtualMODF, monitored_idx::Int, mod::NetworkModification)
    core = get_core(vmodf)
    row_caches = get_row_caches(vmodf)
    cutoff = get_cutoff(vmodf)
    return @lock core.solver_lock begin
        rc = get!(() -> _new_modf_row_cache(vmodf), row_caches, mod)
        if haskey(rc, monitored_idx)
            return copy(rc[monitored_idx])
        end
        row = _compute_modf_entry(vmodf, monitored_idx, mod)
        stored = apply_cutoff(cutoff, row)
        rc[monitored_idx] = stored
        copy(stored)
    end
end

# Row index for a monitored arc, with a clear error when it was reduced away (else
# a raw KeyError). Only the canonical (from, to) orientation is accepted.
function _monitored_arc_index(vmodf::VirtualMODF, monitored::Tuple{Int, Int})
    arc_lookup = get_arc_lookup(vmodf)
    haskey(arc_lookup, monitored) && return arc_lookup[monitored]
    error(
        "Monitored arc $monitored is not present in the reduced network; it was " *
        "likely eliminated by a network reduction. Declare this branch as a " *
        "monitored component on the outage (`get_monitored_components`) so its " *
        "buses are protected from reduction.",
    )
end

"""
Arc-tuple indexed version of getindex for VirtualMODF with NetworkModification.

Rows after a contingency on a scheduled topology K come from
[`get_post_contingency_ptdf_rows`](@ref), on the networks of
[`get_post_contingency_network`](@ref). Indexing with a hand-built `NetworkModification` for
K ∪ c gives a correct row, but computes the full combined update from the base network and
bypasses the reuse of K's factors.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Tuple{Int, Int},
    mod::NetworkModification,
)
    m_idx = _monitored_arc_index(vmodf, monitored)
    return vmodf[m_idx, mod]
end

"""
    get_post_modification_ptdf_rows(vmodf::VirtualMODF, monitored, mods) -> Vector{Union{Vector{Float64}, SparseVector{Float64, Int}}}

The PTDF rows of one monitored arc (an index or a bus pair) under each modification in
`mods`, from a single solve of the arc's base column: `rows[i] == vmodf[monitored, mods[i]]`,
bit for bit. An empty modification gives the base-network row, as in
[`get_ptdf_row`](@ref); a modification taking the monitored arc out of service gives the
zero row. Each modification's Woodbury factors are cached as for `getindex`.

Rows are read from and written to the per-modification row caches exactly as
`vmodf[monitored, mods[i]]` would, so mixing this function with `getindex` sees one cache
state. A modification whose row is cached needs no correction; when every row is cached no
solve runs. The core's `solver_lock` is held for the whole call. The caller owns the
returned copies.
"""
function get_post_modification_ptdf_rows(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    mods::AbstractVector{NetworkModification},
)
    core = get_core(vmodf)
    m = _resolve_monitored_index(vmodf, monitored)
    row_caches = get_row_caches(vmodf)
    cutoff = get_cutoff(vmodf)
    return @lock core.solver_lock begin
        stored = Dict{NetworkModification, RowCacheValue}()
        pending = NetworkModification[]
        seen = Set{NetworkModification}()
        for mod in mods
            mod in seen && continue
            push!(seen, mod)
            rc = get!(() -> _new_modf_row_cache(vmodf), row_caches, mod)
            if haskey(rc, m)
                stored[mod] = rc[m]
            else
                push!(pending, mod)
            end
        end
        if !isempty(pending)
            nets = PostContingencyNetwork[_base_topology_network(mod) for mod in pending]
            for (mod, row) in zip(pending, _post_contingency_rows(vmodf, m, nets))
                value = apply_cutoff(cutoff, row)
                row_caches[mod][m] = value
                stored[mod] = value
            end
        end
        RowCacheValue[copy(stored[mod]) for mod in mods]
    end
end

# --- Post-contingency rows on scheduled topologies (two-tier Woodbury) ---

_on_base_topology(net::PostContingencyNetwork) = net.topology == _BASE_NETWORK

# The network `mod` takes the base network to, as a post-contingency network on the base
# topology. `contingency_id` only labels it: hashing and equality ignore it.
_base_topology_network(mod::NetworkModification, contingency_id::Int = 0) =
    PostContingencyNetwork(_Tier2Derived(), _BASE_NETWORK, mod, mod, contingency_id)

# `net`'s rows live, on the base topology, in the registered contingency's row cache, keyed by
# the tier-2 modification; otherwise in the network's own tier-2 row cache.
function _cached_post_contingency_row(
    vmodf::VirtualMODF,
    net::PostContingencyNetwork,
    m::Int,
)
    _on_base_topology(net) && return _cached_row_at(get_row_caches(vmodf), net.tier2, m)
    return _cached_row_at(get_tier2_row_caches(vmodf), net, m)
end

function _cached_row_at(caches::Dict, key, m::Int)
    rc = get(caches, key, nothing)
    (isnothing(rc) || !haskey(rc, m)) && return nothing
    return rc[m]
end

function _post_contingency_row_cache!(vmodf::VirtualMODF, net::PostContingencyNetwork)
    _on_base_topology(net) &&
        return get!(() -> _new_modf_row_cache(vmodf), get_row_caches(vmodf), net.tier2)
    return get!(() -> _new_modf_row_cache(vmodf), get_tier2_row_caches(vmodf), net)
end

# Resolve, and cache, the factors the row of monitored arc `m` on `net` needs. `topology_factors`
# memoizes the topology factors by object identity within one call, so networks sharing a
# topology object hash it once.
function _post_contingency_row_plan(
    vmodf::VirtualMODF,
    m::Int,
    net::PostContingencyNetwork,
    topology_factors::IdDict{NetworkModification, WoodburyFactors},
)::_RowPlan
    core = get_core(vmodf)
    arc_sus = core.arc_susceptances
    if _on_base_topology(net)
        # The contingency's factors are cached even when this row is the zero row, since its
        # other rows need them. The base network needs no correction, so it has no factors.
        tier2_wf =
            net.tier2 == _BASE_NETWORK ? nothing : _get_woodbury_factors(vmodf, net.tier2)
        return _base_topology_row_plan(
            core,
            m,
            net.post_contingency.arc_modifications,
            tier2_wf,
        )
    end
    _post_contingency_arc_out(arc_sus, core, m, net) && return _RowPlan(_ZERO_ROW)
    topology_wf = get!(topology_factors, net.topology) do
        _get_woodbury_factors(vmodf, net.topology)
    end
    if isempty(net.tier2.arc_modifications) ||
       _contingency_outside_reference_islands(core, topology_wf, net)
        return _RowPlan(
            _TOPOLOGY_ROW,
            topology_wf,
            nothing,
            _post_modification_susceptance(arc_sus, m, net.topology.arc_modifications),
        )
    end
    tier2_wf = get!(get_tier2_factor_cache(vmodf), net) do
        _tier2_factors(core, topology_wf, net)
    end
    return _RowPlan(
        _TIER2_ROW,
        topology_wf,
        tier2_wf,
        _post_contingency_susceptance(arc_sus, m, net),
    )
end

# The islanding rule on a topology whose factors island the network: true when every arc whose
# susceptance the contingency changes lies outside the islands of K holding a reference bus,
# so its rows are K's; false when every such arc lies inside them, so tier 2 applies. A
# contingency on both sides is not covered and throws. Arcs with no susceptance change never
# enter the Woodbury update, and an arc with no susceptance has no BA entries to place it.
function _contingency_outside_reference_islands(
    core::VirtualFactorCore,
    topology_wf::WoodburyFactors,
    net::PostContingencyNetwork,
)::Bool
    topology_wf.is_islanding || return false
    labels = topology_wf.bus_island_labels
    reference = Set{Int}(labels[p] for p in get_ref_bus_position(core))
    rv = SparseArrays.rowvals(core.BA)
    inside = Int[]
    outside = Int[]
    for mod in net.tier2.arc_modifications
        iszero(mod.delta_b) && continue
        e = mod.arc_index
        if any(k -> labels[rv[k]] in reference, SparseArrays.nzrange(core.BA, e))
            push!(inside, e)
        else
            push!(outside, e)
        end
    end
    isempty(outside) && return false
    isempty(inside) && return true
    arc_ax = get_arc_axis(core)
    throw(
        ArgumentError(
            "Post-contingency rows of contingency $(net.contingency_id) on topology \
            \"$(net.topology.label)\" are not supported: the topology islands the network, \
            and the contingency changes arcs $(join(arc_ax[inside], ", ")) inside the \
            islands holding a reference bus and arcs $(join(arc_ax[outside], ", ")) \
            outside them.",
        ),
    )
end

# Rows of monitored arc `m` on each network in `nets`, before the cutoff, from one solve of
# its base column.
function _post_contingency_rows(
    vmodf::VirtualMODF,
    m::Int,
    nets::Vector{PostContingencyNetwork},
)::Vector{Vector{Float64}}
    topology_factors = IdDict{NetworkModification, WoodburyFactors}()
    plans = [_post_contingency_row_plan(vmodf, m, net, topology_factors) for net in nets]
    return _rows_from_plans(get_core(vmodf), m, plans)
end

"""
    get_post_contingency_ptdf_rows(vmodf::VirtualMODF, monitored, networks::AbstractVector{PostContingencyNetwork}) -> Vector{Union{Vector{Float64}, SparseVector{Float64, Int}}}

The PTDF rows of one monitored arc (an index or a bus pair) on each post-contingency network
in `networks`, from a single solve of the arc's base column. Each row is the arc's row on the
network K ∪ c, referenced to the reference buses, with `vmodf`'s cutoff applied.

On a topology K other than the base network, the row is a second Woodbury update K → K ∪ c
of the arc's row on K, with no further solve: the tier-1 correction gives `B_K⁻¹ν` of the arc
once per distinct topology, and each network corrects its own copy with its tier-2 factors
and scales by the arc's susceptance on K ∪ c. On the base topology the row equals
`vmodf[monitored, network.tier2]` bit for bit.

A monitored arc out of service on K ∪ c gives the zero row. When K's factors island the
network and every arc the contingency changes lies outside the islands holding a reference
bus, the row is the arc's row on K. A contingency on an islanded K that changes arcs both
inside and outside those islands throws an `ArgumentError`.

Rows after a contingency on a scheduled topology come from this function. Querying
`vmodf[monitored, mod]` with a hand-built `NetworkModification` for K ∪ c gives a correct
row, but computes the full combined update from the base network and bypasses the reuse of
K's factors that the second update provides.

Rows are read from and written to the row caches, and tier-2 factors are cached, per
network: networks on the base topology share `vmodf`'s contingency caches, the others have
their own; see [`retain_modifications!`](@ref). The core's `solver_lock` is held for the
whole call. The caller owns the returned copies.
"""
function get_post_contingency_ptdf_rows(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    networks::AbstractVector{PostContingencyNetwork},
)
    core = get_core(vmodf)
    m = _resolve_monitored_index(vmodf, monitored)
    cutoff = get_cutoff(vmodf)
    return @lock core.solver_lock begin
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
            cached = _cached_post_contingency_row(vmodf, net, m)
            if isnothing(cached)
                push!(values, Float64[])
                push!(pending, net)
                push!(pending_slots, k)
            else
                push!(values, cached)
            end
        end
        # Row caches are created only once the rows exist, so a call that throws leaves none.
        if !isempty(pending)
            rows = _post_contingency_rows(vmodf, m, pending)
            for (net, k, row) in zip(pending, pending_slots, rows)
                value = apply_cutoff(cutoff, row)
                _post_contingency_row_cache!(vmodf, net)[m] = value
                values[k] = value
            end
        end
        RowCacheValue[copy(values[k]) for k in slot]
    end
end

"""
    get_ptdf_row(vmodf::VirtualMODF, arc::Tuple{Int, Int}) -> Union{Vector{Float64}, SparseVector{Float64, Int}}

The base-network PTDF row of `arc`, solved on the factorization `vmodf` shares with its
[`VirtualPTDF`](@ref) and referenced to the reference buses, with no distributed slack.
Rows are cached in `vmodf` under the empty [`NetworkModification`](@ref); the caller owns
the returned copy. The row is dense unless `vmodf`'s cutoff sparsified it, in which case it
is a `SparseVector`.
"""
function get_ptdf_row(vmodf::VirtualMODF, arc::Tuple{Int, Int})
    return vmodf[_monitored_arc_index(vmodf, arc), _BASE_NETWORK]
end

# --- getindex: by ContingencySpec or PSY.UnplannedOutage (the base-topology network) ---

"""
Get the post-contingency PTDF row for monitored arc `monitored` (an index or a bus pair) under
a `ContingencySpec`: the row on its [`PostContingencyNetwork`](@ref) on the base topology,
from [`get_post_contingency_ptdf_rows`](@ref).

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    contingency::ContingencySpec,
)
    mod = contingency.modification
    net = _base_topology_network(mod, contingency.id)
    return only(get_post_contingency_ptdf_rows(vmodf, monitored, [net]))
end

"""
Get the post-contingency PTDF row for monitored arc `monitored` (an index or a bus pair) when
outage `outage` trips: the row on its [`PostContingencyNetwork`](@ref) on the base topology,
from [`get_post_contingency_ptdf_rows`](@ref). The outage must have been registered at
VirtualMODF construction time.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    outage::PSY.UnplannedOutage,
)
    net = PostContingencyNetwork(vmodf, IS.get_id(outage))
    return only(get_post_contingency_ptdf_rows(vmodf, monitored, [net]))
end

"""
    clear_caches!(vmodf::VirtualMODF)

Clear the Woodbury and row caches, those of post-contingency networks on scheduled
topologies included. Does NOT clear the contingency registration cache — registered outages
remain valid and can be queried again.
"""
function clear_caches!(vmodf::VirtualMODF)
    core = get_core(vmodf)
    @lock core.solver_lock _empty_factor_and_row_caches!(vmodf)
    return
end

function _empty_factor_and_row_caches!(vmodf::VirtualMODF)
    empty!(get_woodbury_cache(vmodf))
    empty!(get_row_caches(vmodf))
    empty!(get_tier2_factor_cache(vmodf))
    empty!(get_tier2_row_caches(vmodf))
    return
end

"""
    retain_modifications!(vmodf::VirtualMODF, keep)

Evict the cached PTDF rows and Woodbury factors of every modification except the base
network and the modifications in `keep`. Modifications are matched by value, so a
modification rebuilt from the same components is retained. The base network is always
retained because every modification's rows are corrected from its rows.

`keep` may also hold [`PostContingencyNetwork`](@ref)s. A retained network keeps its rows
and tier-2 factors, and its topology's rows and factors, so the topology need not be named.
A network on the base topology is its tier-2 modification, the registered contingency's.
Every other network's caches are evicted.

Each modification's row cache is bounded, but the set of modifications is not: a caller that
queries new modifications over the life of `vmodf`, such as a schedule of topologies rebuilt
for each simulation window, calls this to bound the memory held to the modifications it
still needs. The caches of registered contingencies are evicted unless they are in `keep`;
the registrations themselves are kept, so an evicted contingency is recomputed on its next
query.
"""
function retain_modifications!(vmodf::VirtualMODF, keep)
    core = get_core(vmodf)
    @lock core.solver_lock begin
        retained = Set{NetworkModification}([_BASE_NETWORK])
        networks = Set{PostContingencyNetwork}()
        for entry in keep
            _retain!(retained, networks, entry)
        end
        filter!(entry -> first(entry) in retained, get_woodbury_cache(vmodf))
        filter!(entry -> first(entry) in retained, get_row_caches(vmodf))
        filter!(entry -> first(entry) in networks, get_tier2_factor_cache(vmodf))
        filter!(entry -> first(entry) in networks, get_tier2_row_caches(vmodf))
    end
    return
end

_retain!(retained::Set{NetworkModification}, _, mod::NetworkModification) =
    push!(retained, mod)

function _retain!(
    retained::Set{NetworkModification},
    networks::Set{PostContingencyNetwork},
    net::PostContingencyNetwork,
)
    if _on_base_topology(net)
        push!(retained, net.tier2)
    else
        push!(networks, net)
        push!(retained, net.topology)
    end
    return
end

"""
    clear_all_caches!(vmodf::VirtualMODF)

Clear all caches including contingency registrations. After calling this function,
the `VirtualMODF` object is effectively empty and cannot be queried. Use
`clear_caches!` instead to preserve contingency registrations while freeing
computation cache memory.
"""
function clear_all_caches!(vmodf::VirtualMODF)
    core = get_core(vmodf)
    @lock core.solver_lock begin
        empty!(get_registered_contingencies(vmodf))
        _empty_factor_and_row_caches!(vmodf)
    end
    return
end
