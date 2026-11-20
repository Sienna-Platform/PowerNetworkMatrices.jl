"""
The Virtual Multiple Outage Distribution Factor (VirtualMODF) structure computes
post-contingency PTDF rows lazily for registered contingencies using the
Woodbury matrix identity (van Dijk et al. Eq. 29).

Contingencies are resolved from `PSY.UnplannedOutage` supplemental attributes at
construction time. `PSY.PlannedOutage`s are not contingencies and are not registered.
After registration, the System is not needed for queries.

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
- `max_cache_size_bytes::Int`:
        Max cache size in bytes per contingency.
- `tier2::Tier2Cache`:
        Tier-2 factors and rows of post-contingency networks on topologies other than the
        base network, and the base solves their factors share (see [`Tier2Cache`](@ref)).
        Networks on the base topology use `woodbury_cache` and `row_caches`, under their
        tier-2 modification. The core's `solver_lock` guards it with the other caches.
"""
struct VirtualMODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    core::VirtualFactorCore{Ax, L, K}
    contingency_cache::Dict{Int, ContingencySpec}
    woodbury_cache::Dict{NetworkModification, WoodburyFactors}
    row_caches::Dict{NetworkModification, RowCache{RowCacheValue}}
    max_cache_size_bytes::Int
    tier2::Tier2Cache
end

# A VirtualMODF on `core` with the given registrations and empty caches.
function VirtualMODF(
    core::VirtualFactorCore,
    contingency_cache::Dict{Int, ContingencySpec},
    max_cache_size_bytes::Int,
    max_column_cache_bytes::Int,
)
    return VirtualMODF(
        core,
        contingency_cache,
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache{RowCacheValue}}(),
        max_cache_size_bytes,
        Tier2Cache(max_column_cache_bytes),
    )
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
       name === :max_cache_size_bytes ||
       name === :tier2
        return getfield(vmodf, name)
    end
    name === :PTDF_A_diag && return get_PTDF_A_diag(get_core(vmodf))
    name === :branch_susceptances_by_arc &&
        return get_branch_susceptances_by_arc(get_core(vmodf))
    return _core_property(get_core(vmodf), name)
end

# --- Accessors ---

# Field getters. `getfield` is confined to these (and the `getproperty` hook
# above), so the rest of the file reads the wrapper's own fields through proper
# accessors instead of `getfield`.
get_core(M::VirtualMODF) = getfield(M, :core)
get_woodbury_cache(M::VirtualMODF) = getfield(M, :woodbury_cache)
get_row_caches(M::VirtualMODF) = getfield(M, :row_caches)
get_max_cache_size_bytes(M::VirtualMODF) = getfield(M, :max_cache_size_bytes)
get_tier2_cache(M::VirtualMODF) = getfield(M, :tier2)
get_tier2_factor_cache(M::VirtualMODF) = get_tier2_factor_cache(get_tier2_cache(M))
get_tier2_row_caches(M::VirtualMODF) = get_tier2_row_caches(get_tier2_cache(M))
get_max_column_cache_bytes(M::VirtualMODF) = get_max_column_cache_bytes(get_tier2_cache(M))

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
`PSY.UnplannedOutage` supplemental attributes found in the system. Planned outages
are not contingencies and are ignored.

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
- `max_column_cache_bytes::Int`: Bound in bytes on the base solves shared by tier-2 factors
        (default: MAX_COLUMN_CACHE_BYTES, 1 GiB)
- `network_reductions::Vector{NetworkReduction}`: Network reductions to apply
- `irreducible_buses::IS.ComponentSelector`: Components whose buses the network reductions must
        keep (see [`get_irreducible_bus_numbers`](@ref)). Default: `IS.make_selector()`
- `automatically_register_outages::Bool`: Register all system `PSY.UnplannedOutage` attributes (default: true)
- `scheduled_branches`, `closed_switches`: passed to [`Ybus`](@ref), which keeps the buses of
        both lists out of the reductions and includes the closed switches as closed
"""
function VirtualMODF(
    sys::PSY.System;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    max_column_cache_bytes::Int = MAX_COLUMN_CACHE_BYTES,
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
        max_column_cache_bytes = max_column_cache_bytes,
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

`max_cache_size` (MiB per contingency) and `max_column_cache_bytes` are as for
`VirtualMODF(sys; ...)`.
"""
function VirtualMODF(
    core::VirtualFactorCore,
    sys::PSY.System;
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    max_column_cache_bytes::Int = MAX_COLUMN_CACHE_BYTES,
    automatically_register_outages::Bool = true,
)
    vmodf = VirtualMODF(core; max_cache_size, max_column_cache_bytes)

    # Auto-register all outage attributes from the system
    automatically_register_outages && _register_all_outages!(vmodf, sys)

    return vmodf
end

"""
    VirtualMODF(core::VirtualFactorCore; max_cache_size, max_column_cache_bytes) -> VirtualMODF

Wrap `core` in a VirtualMODF with no registered contingencies and empty caches.
"""
function VirtualMODF(
    core::VirtualFactorCore;
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    max_column_cache_bytes::Int = MAX_COLUMN_CACHE_BYTES,
)
    return VirtualMODF(
        core,
        Dict{Int, ContingencySpec}(),
        max_cache_size * MiB,
        max_column_cache_bytes,
    )
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
    _validate_transmission_survived(sys, outage, mod, closed_switches)

For an outage whose ACTransmission components resolved to no arc modifications: throw when
any is in service (a reduction eliminated it, so every query would return the base-case row);
warn when all are out of service (a legitimate no-op). A component is in service when it is
available and, for a switch, closed by its status or named in `closed_switches`. This is the
only survive-check possible on a core that was reduced before outages were known.
"""
function _validate_transmission_survived(
    sys::PSY.System,
    outage::PSY.Outage,
    mod::NetworkModification,
    closed_switches::Set{String},
)
    isempty(mod.arc_modifications) || return
    transmission =
        collect(
            PSY.get_associated_components(sys, outage; component_type = PSY.ACTransmission),
        )
    isempty(transmission) && return
    in_service = filter(c -> _in_service(c, closed_switches), transmission)
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

_warn_or_rethrow_failed_outage(e::Union{ErrorException, UnsupportedOutageError}) =
    @warn "Could not register outage: $(e.msg)"
_warn_or_rethrow_failed_outage(e) = rethrow()

"""
    _register_all_outages!(vmodf, sys)

Bulk-register all `PSY.UnplannedOutage` supplemental attributes in the system.
`PSY.PlannedOutage`s are not contingencies and are skipped.
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
        @warn "No unplanned outage supplemental attributes found in system. " *
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
function _register_outage!(vmodf::VirtualMODF, sys::PSY.System, outage::PSY.Outage)
    contingency_cache = get_registered_contingencies(vmodf)
    outage_id = IS.get_id(outage)
    if haskey(contingency_cache, outage_id)
        @warn "Outage with UUID $(outage_id) is already registered; skipping."
        return
    end
    mod = NetworkModification(vmodf, sys, outage)
    _validate_transmission_survived(sys, outage, mod,
        get_closed_switches(get_reductions(get_network_reduction_data(vmodf))))
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
_get_woodbury_factors(vmodf::VirtualMODF, mod::NetworkModification) =
    _tier1_factors!(get_woodbury_cache(vmodf), get_core(vmodf), mod)

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
    plan = _modf_row_plan(vmodf, monitored_idx, mod)
    return only(_rows_from_plans(get_core(vmodf), monitored_idx, (plan,)))
end

# Plan of monitored arc `m`'s row under `mod`, from the base network.
_modf_row_plan(vmodf::VirtualMODF, m::Int, mod::NetworkModification)::_RowPlan =
    _modification_row_plan(get_woodbury_cache(vmodf), get_core(vmodf), m, mod)

# Per-modification row cache: one n_bus-long row per monitored arc.
_new_modf_row_cache(vmodf::VirtualMODF) =
    _network_row_cache(get_core(vmodf), get_max_cache_size_bytes(vmodf))

# --- getindex: by integer monitored index + NetworkModification ---

"""
Get the post-modification PTDF row for monitored arc `monitored_idx` under `mod`.
Uses per-modification RowCache for LRU-eviction caching.
An arc in an island that holds no base reference bus gets the all-zero row.

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
zero row. An arc in an island that holds no base reference bus gets the all-zero row. Each
modification's Woodbury factors are cached as for `getindex`.

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
            plans = [_modf_row_plan(vmodf, m, mod) for mod in pending]
            for (mod, row) in zip(pending, _rows_from_plans(core, m, plans))
                value = apply_cutoff(cutoff, row)
                row_caches[mod][m] = value
                stored[mod] = value
            end
        end
        RowCacheValue[copy(stored[mod]) for mod in mods]
    end
end

# --- Post-contingency rows on scheduled topologies (two-tier Woodbury) ---

# The network `mod` takes the base network to, as a post-contingency network on the base
# topology. `contingency_id` only labels it: hashing and equality ignore it.
_base_topology_network(mod::NetworkModification, contingency_id::Int = 0) =
    PostContingencyNetwork(_Tier2Derived(), _BASE_NETWORK, mod, mod, contingency_id)

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
bus, the row is the arc's row on K. When it changes arcs both inside and outside those
islands, the rows are those of the contingency without its arcs outside them, which join
only buses K cut off.

Rows after a contingency on a scheduled topology come from this function. Querying
`vmodf[monitored, mod]` with a hand-built `NetworkModification` for K ∪ c gives a correct
row, but computes the full combined update from the base network and bypasses the reuse of
K's factors that the second update provides.

Rows are read from and written to the row caches, and tier-2 factors are cached, per
network: networks on the base topology share `vmodf`'s contingency caches, the others have
their own. These rows and tier-2 factors last until the next
[`retain_modifications!`](@ref), which ends a build. The core's `solver_lock` is held for the
whole call. The caller owns the returned copies.
"""
function get_post_contingency_ptdf_rows(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    networks::AbstractVector{PostContingencyNetwork},
)
    core = get_core(vmodf)
    m = _resolve_monitored_index(vmodf, monitored)
    return @lock core.solver_lock _post_contingency_ptdf_rows!(
        get_tier2_cache(vmodf),
        core,
        get_woodbury_cache(vmodf),
        get_row_caches(vmodf),
        get_max_cache_size_bytes(vmodf),
        m,
        networks,
    )
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

# --- getindex: by ContingencySpec or PSY.Outage (the contingency on the base network) ---

"""
Get the post-contingency PTDF row for monitored arc `monitored` (an index or a bus pair) under
a `ContingencySpec`: `vmodf[monitored, contingency.modification]`, which is also the row on
its [`PostContingencyNetwork`](@ref) on the base topology.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    contingency::ContingencySpec,
)
    return vmodf[monitored, contingency.modification]
end

"""
Get the post-contingency PTDF row for monitored arc `monitored` (an index or a bus pair) when
outage `outage` trips: `vmodf[monitored, mod]` for the outage's registered modification
`mod`, which is also the row on its [`PostContingencyNetwork`](@ref) on the base topology.
The outage must have been registered at VirtualMODF construction time.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Union{Int, Tuple{Int, Int}},
    outage::PSY.Outage,
)
    mod = @lock get_core(vmodf).solver_lock _resolve_modification(vmodf, outage)
    return vmodf[monitored, mod]
end

"""
    clear_caches!(vmodf::VirtualMODF)

Clear the Woodbury, row and column caches, those of post-contingency networks on scheduled
topologies included, and zero the column cache's hit and miss counts. Does NOT clear the
contingency registration cache — registered outages remain valid and can be queried again.
"""
function clear_caches!(vmodf::VirtualMODF)
    core = get_core(vmodf)
    @lock core.solver_lock _empty_factor_and_row_caches!(vmodf)
    return
end

function _empty_factor_and_row_caches!(vmodf::VirtualMODF)
    empty!(get_woodbury_cache(vmodf))
    empty!(get_row_caches(vmodf))
    clear_caches!(get_tier2_cache(vmodf))
    return
end

"""
    cache_statistics(vmodf::VirtualMODF) -> NamedTuple

Entries and bytes of each cache of `vmodf`:
- `columns = (; entries, bytes, hits, misses)`: the base solves shared by tier-2 factors,
  with the lookups served from the cache and solved since the last [`clear_caches!`](@ref);
- `tier1_factors = (; entries, bytes)`: Woodbury factors on the base network, of topologies
  and of contingencies on the base topology;
- `tier2_factors = (; entries, bytes)`: tier-2 factors of post-contingency networks on other
  topologies;
- `rows = (; entries, bytes)`: cached PTDF rows of every network.

Bytes count the arrays held, not container overhead. The core's `solver_lock` is held for
the call.
"""
function cache_statistics(vmodf::VirtualMODF)
    core = get_core(vmodf)
    return @lock core.solver_lock cache_statistics(
        get_tier2_cache(vmodf),
        get_woodbury_cache(vmodf),
        get_row_caches(vmodf),
    )
end

"""
    retain_modifications!(vmodf::VirtualMODF, keep)

End a build: evict the cached rows and tier-2 factors, and the Woodbury factors of every
modification not named in `keep`.

Kept are the Woodbury factors of the base network and of every modification in `keep`,
matched by value, so a modification rebuilt from the same components is retained. `keep` may
also hold [`PostContingencyNetwork`](@ref)s. A retained network keeps its topology's factors,
so the topology need not be named. On the base topology it keeps the factors of its
registered contingency, the network's tier-2 modification. The base-network rows and the
column cache, with its hit and miss counts, are also kept.

Every other row cache is emptied, those of retained modifications and networks included, and
so is the tier-2 factor cache. Rows and tier-2 factors therefore live for one build, while
the factors of the topologies and contingencies still in use, and the base columns, serve the
next one. The set of modifications is not bounded by `vmodf`, so a caller that queries new
modifications over its life, such as a topology sequence rebuilt for each simulation
window, calls this once per build. The registrations are kept, so an evicted contingency is
recomputed on its next query.
"""
function retain_modifications!(vmodf::VirtualMODF, keep)
    core = get_core(vmodf)
    @lock core.solver_lock begin
        retained = Set{NetworkModification}([_BASE_NETWORK])
        for entry in keep
            _retain!(retained, entry)
        end
        filter!(entry -> first(entry) in retained, get_woodbury_cache(vmodf))
        filter!(entry -> first(entry) == _BASE_NETWORK, get_row_caches(vmodf))
        retain_modifications!(get_tier2_cache(vmodf), keep)
    end
    return
end

_retain!(retained::Set{NetworkModification}, mod::NetworkModification) =
    push!(retained, mod)

_retain!(retained::Set{NetworkModification}, net::PostContingencyNetwork) =
    push!(retained, _on_base_topology(net) ? net.tier2 : net.topology)

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
