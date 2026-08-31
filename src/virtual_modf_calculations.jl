"""
The Virtual Multiple Outage Distribution Factor (VirtualMODF) structure computes
post-contingency PTDF rows lazily for registered contingencies using the
Woodbury matrix identity (van Dijk et al. Eq. 29).

Contingencies are resolved from PSY.Outage supplemental attributes at construction
time. After registration, the System is not needed for queries.

Internally the factorization and topology live in a private
[`VirtualFactorCore`](@ref) held in the `core` field; the VirtualMODF only adds
the contingency-specific caches. A single core can be shared with a
`VirtualPTDF` / `VirtualLODF` so the factorization is computed once.

Caching is two-tiered:
- Woodbury factors (M KLU solves) are cached per contingency
- PTDF rows (1 KLU solve each) are cached per (monitored_arc, contingency) via
  one RowCache per contingency

# Thread-safety

Concurrent `getindex` is safe but serialized: the core's `solver_lock` (a
`ReentrantLock`) is held for the full body of `getindex`, `clear_caches!`, and
`clear_all_caches!`, so Dict mutations on the cache structures and the libklu
solves it wraps all run under a single mutex. libklu activity additionally
serializes through the process-wide `_LIBKLU_LOCK`.

# Fields
- `core::VirtualFactorCore`:
        Shared factorization/topology container (see [`VirtualFactorCore`](@ref)).
        `PTDF_A_diag` and `branch_susceptances_by_arc` are computed lazily on the
        core and shared with any other wrapper using the same core.
- `dist_slack::Vector{Float64}`:
        Distributed slack bus weights (retained for API symmetry; not used by the
        Woodbury kernel).
- `contingency_cache::Dict{Int, ContingencySpec}`:
        Resolved contingencies keyed by outage UUID.
- `woodbury_cache::Dict{NetworkModification, WoodburyFactors}`:
        Precomputed Woodbury factors keyed by modification.
- `row_caches::Dict{NetworkModification, RowCache}`:
        One `RowCache` per modification.
- `max_cache_size_bytes::Int`:
        Max cache size in bytes per contingency.
"""
struct VirtualMODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    core::VirtualFactorCore{Ax, L, K}
    dist_slack::Vector{Float64}
    contingency_cache::Dict{Int, ContingencySpec}
    woodbury_cache::Dict{NetworkModification, WoodburyFactors}
    row_caches::Dict{NetworkModification, RowCache}
    max_cache_size_bytes::Int
end

# --- Field forwarding ---

# Forward non-wrapper field reads to the core. The lazy `PTDF_A_diag` /
# `branch_susceptances_by_arc` are special-cased so reads trigger the core's
# lazy compute (and back-compat reads like `vmodf.PTDF_A_diag` keep working).
function Base.getproperty(vmodf::VirtualMODF, name::Symbol)
    if name === :core ||
       name === :dist_slack ||
       name === :contingency_cache ||
       name === :woodbury_cache ||
       name === :row_caches ||
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
get_dist_slack(M::VirtualMODF) = getfield(M, :dist_slack)
get_contingency_cache(M::VirtualMODF) = getfield(M, :contingency_cache)
get_woodbury_cache(M::VirtualMODF) = getfield(M, :woodbury_cache)
get_row_caches(M::VirtualMODF) = getfield(M, :row_caches)
get_max_cache_size_bytes(M::VirtualMODF) = getfield(M, :max_cache_size_bytes)

get_axes(M::VirtualMODF) = get_axes(get_core(M))
get_lookup(M::VirtualMODF) = get_lookup(get_core(M))
get_ref_bus(M::VirtualMODF) = get_ref_bus(get_core(M))
get_ref_bus_position(M::VirtualMODF) = get_ref_bus_position(get_core(M))
get_network_reduction_data(M::VirtualMODF) = get_network_reduction_data(get_core(M))
get_branch_catalog(M::VirtualMODF) = get_branch_catalog(get_core(M))
get_arc_lookup(M::VirtualMODF) = get_arc_lookup(get_core(M))
get_bus_lookup(M::VirtualMODF) = get_bus_lookup(get_core(M))
get_arc_axis(M::VirtualMODF) = get_arc_axis(get_core(M))
get_bus_axis(M::VirtualMODF) = get_bus_axis(get_core(M))
get_tol(M::VirtualMODF) = get_tol(get_core(M))
get_cutoff(M::VirtualMODF) = get_cutoff(get_core(M))
get_system_uuid(M::VirtualMODF) = get_system_uuid(get_core(M))
_get_BA(M::VirtualMODF) = _get_BA(get_core(M))
_get_arc_susceptances(M::VirtualMODF) = _get_arc_susceptances(get_core(M))
_get_valid_ix(M::VirtualMODF) = _get_valid_ix(get_core(M))

"""
$(TYPEDSIGNATURES)

Return `H[e, e]` for each arc `e`. Triggers the core's lazy compute on first
call and returns the cached vector thereafter.
"""
get_PTDF_A_diag(vmodf::VirtualMODF) = get_PTDF_A_diag(get_core(vmodf))

# Woodbury kernel outer dispatchers forward to the shared core method.
function _compute_woodbury_factors(
    mat::VirtualMODF,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    return _compute_woodbury_factors(get_core(mat), modifications)
end

function _apply_woodbury_correction(
    mat::VirtualMODF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return _apply_woodbury_correction(get_core(mat), monitored_idx, wf)
end

"""
    get_registered_contingencies(vmodf::VirtualMODF) -> Dict{Int, ContingencySpec}

Return the cached contingency registrations for inspection.
"""
get_registered_contingencies(vmodf::VirtualMODF) = get_contingency_cache(vmodf)

# --- Base interface ---

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualMODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    print(
        io,
        "VirtualMODF with $(length(get_contingency_cache(array))) registered contingencies",
    )
    return
end

function Base.isempty(vmodf::VirtualMODF)
    return isempty(get_contingency_cache(vmodf))
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
Outage supplemental attributes found in the system.

The buses of every outaged component and the components each outage declares
monitored (`get_monitored_components`) are automatically added to the irreducible
set before the base `Ybus` is built, and (when `network_reductions` are supplied)
any `WardReduction.study_buses` are augmented to match. This is mandatory: the
ABA/Woodbury solve runs on the reduced network, so a branch in a contingency must
survive every reduction step, including the zero-impedance reduction that is
auto-applied during `Ybus` construction.

# Keyword Arguments
- `dist_slack::Vector{Float64}`: Distributed slack weights (default: empty)
- `linear_solver::String = _default_linear_solver()`: Linear solver for the
        ABA factorization. Options: "KLU", "AppleAccelerate". Defaults to
        "AppleAccelerate" on macOS and "KLU" elsewhere.
- `tol::Union{Float64, AutoTolerance}`: Tolerance for row sparsification.
        A `Float64` applies a fixed absolute cutoff; an [`AutoTolerance`](@ref)
        (the default) applies a relative per-row cutoff so requested columns stay
        sparse on large systems.
- `max_cache_size::Int`: Max cache size in MiB per contingency (default: MAX_CACHE_SIZE_MiB)
- `network_reductions::Vector{NetworkReduction}`: Network reductions to apply
- `automatically_register_outages::Bool`: Register all system Outage attributes (default: true)
"""
function VirtualMODF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    irreducible_buses = Set{Int}(),
    automatically_register_outages::Bool = true,
    kwargs...,
)
    if !isempty(dist_slack)
        @info "Distributed bus"
    end
    # Accept any iterable of bus numbers and normalize once, matching `Ybus`.
    irreducible_buses = Set{Int}(irreducible_buses)
    resolve_linear_solver(linear_solver)

    # Split so the zero-impedance entry reaches `Ybus` (which applies it first) instead of
    # the manual loop below, which would apply it a second time.
    network_reductions, zero_impedance_reduction =
        _split_zero_impedance_reduction(network_reductions)

    # Outage/monitored buses are auto-protected so contingency branches survive
    # reduction. Collect them from `sys` before building the base Ybus and fold
    # them into `irreducible_buses` so the auto-applied zero-impedance reduction
    # never merges away a monitored/outaged branch endpoint in the first place.
    # Registering an outage with previously-unseen monitored components after
    # construction shifts this set and requires rebuilding the MODF.
    protected_buses = _collect_protected_buses(sys)
    applied_irreducible = union(irreducible_buses, protected_buses)

    # Build the base Ybus with the combined irreducible set (zero-impedance reduction
    # auto-applied here honors it); this is the starting point for further reductions.
    Ymatrix = Ybus(
        sys;
        irreducible_buses = applied_irreducible,
        network_reductions = NetworkReduction[zero_impedance_reduction],
        kwargs...,
    )

    # radial/degree-two read the container's irreducible set (already seeded above via
    # `Ybus`'s `irreducible_buses`). Ward's `study_buses` defines the retained network rather
    # than exempting buses, so it is validated against the contingencies.
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
        dist_slack = dist_slack,
        max_cache_size = max_cache_size,
        automatically_register_outages = automatically_register_outages,
    )
end

"""
    VirtualMODF(core::VirtualFactorCore, sys::PSY.System; kwargs...) -> VirtualMODF

Wrap an existing [`VirtualFactorCore`](@ref) in a VirtualMODF and register the
system's outages. Use this (or `VirtualMODF(vptdf, sys)`) to share one
factorization across multiple virtual matrices.
"""
function VirtualMODF(
    core::VirtualFactorCore,
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    automatically_register_outages::Bool = true,
)
    max_cache_bytes = max_cache_size * MiB
    vmodf = VirtualMODF(
        core,
        dist_slack,
        Dict{Int, ContingencySpec}(),
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache}(),
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
factorized only once.
"""
function VirtualMODF(vptdf::VirtualPTDF, sys::PSY.System; kwargs...)
    return VirtualMODF(get_core(vptdf), sys; kwargs...)
end

"""
    _warn_if_transmission_dropped(sys, outage, mod)

Warn when an outage references `ACTransmission` components but its modification has
no arc modifications — those branches were eliminated by reduction, so the
contingency would silently return the unmodified base row.
"""
function _warn_if_transmission_dropped(
    sys::PSY.System,
    outage::PSY.Outage,
    mod::NetworkModification,
)
    isempty(mod.arc_modifications) || return
    transmission =
        PSY.get_associated_components(sys, outage; component_type = PSY.ACTransmission)
    isempty(transmission) && return
    @warn "Outage (label=$(mod.label)) references transmission components but " *
          "resolved to no arc modifications; they were eliminated by a network " *
          "reduction. Querying this contingency returns the unmodified PTDF row."
    return
end

# --- Outage registration ---

_warn_or_rethrow_failed_outage(e::ErrorException) =
    @warn "Could not register outage: $(e.msg)"
_warn_or_rethrow_failed_outage(e) = rethrow()

"""
    _register_all_outages!(vmodf, sys)

Bulk-register all Outage supplemental attributes in the system.
Called automatically by the VirtualMODF constructor.
"""
function _register_all_outages!(vmodf::VirtualMODF, sys::PSY.System)
    count = 0
    for outage in PSY.get_supplemental_attributes(PSY.Outage, sys)
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

Resolve an Outage supplemental attribute to a ContingencySpec and cache it.
Delegates to `NetworkModification(mat, sys, outage)` for the resolution logic.
"""
function _register_outage!(vmodf::VirtualMODF, sys::PSY.System, outage::PSY.Outage)
    contingency_cache = get_contingency_cache(vmodf)
    outage_id = IS.get_id(outage)
    if haskey(contingency_cache, outage_id)
        @warn "Outage with UUID $(outage_id) is already registered; skipping."
        return
    end
    mod = NetworkModification(vmodf, sys, outage)
    ctg = ContingencySpec(outage_id, mod)
    contingency_cache[outage_id] = ctg
    _warn_if_transmission_dropped(sys, outage, mod)
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
    wf = _get_woodbury_factors(vmodf, mod)
    return _apply_woodbury_correction(vmodf, monitored_idx, wf)
end

# --- getindex: by integer monitored index + NetworkModification ---

"""
Get the post-modification PTDF row for monitored arc `monitored_idx` under `mod`.
Uses per-modification RowCache for LRU-eviction caching.

$(TYPEDSIGNATURES)
"""
function Base.getindex(vmodf::VirtualMODF, monitored_idx::Int, mod::NetworkModification)
    core = get_core(vmodf)
    row_caches = get_row_caches(vmodf)
    max_bytes = get_max_cache_size_bytes(vmodf)
    cutoff = get_cutoff(vmodf)
    return @lock core.solver_lock begin
        rc = get!(row_caches, mod) do
            row_size = length(core.temp_data[1]) * sizeof(Float64)
            RowCache(max_bytes, Set{Int}(), row_size)
        end
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

# --- getindex: by ContingencySpec (delegates to NetworkModification) ---

"""
Get the post-contingency PTDF row for monitored arc under a ContingencySpec.

$(TYPEDSIGNATURES)
"""
function Base.getindex(vmodf::VirtualMODF, monitored_idx::Int, contingency::ContingencySpec)
    return vmodf[monitored_idx, contingency.modification]
end

function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Tuple{Int, Int},
    contingency::ContingencySpec,
)
    return vmodf[monitored, contingency.modification]
end

# --- getindex: by PSY.Outage (id lookup → ContingencySpec → NetworkModification) ---

"""
Get the post-contingency PTDF row for monitored arc `monitored` when outage `outage` trips.
The outage must have been registered at VirtualMODF construction time.

$(TYPEDSIGNATURES)
"""
function Base.getindex(vmodf::VirtualMODF, monitored::Int, outage::PSY.Outage)
    core = get_core(vmodf)
    contingency_cache = get_contingency_cache(vmodf)
    outage_id = IS.get_id(outage)
    # Pair with the locked `empty!` in `clear_all_caches!`; without it, a
    # concurrent clear could rehash `contingency_cache` mid-lookup.
    ctg = @lock core.solver_lock begin
        if !haskey(contingency_cache, outage_id)
            error(
                "Outage (UUID=$outage_id) is not registered. " *
                "Construct VirtualMODF with the system containing this outage.",
            )
        end
        contingency_cache[outage_id]
    end
    return vmodf[monitored, ctg.modification]
end

"""
Arc-tuple indexed version of getindex by PSY.Outage.

$(TYPEDSIGNATURES)
"""
function Base.getindex(vmodf::VirtualMODF, monitored::Tuple{Int, Int}, outage::PSY.Outage)
    m_idx = _monitored_arc_index(vmodf, monitored)
    return vmodf[m_idx, outage]
end

"""
    clear_caches!(vmodf::VirtualMODF)

Clear Woodbury and row caches. Does NOT clear the contingency registration
cache — registered outages remain valid and can be queried again.
"""
function clear_caches!(vmodf::VirtualMODF)
    core = get_core(vmodf)
    @lock core.solver_lock begin
        empty!(get_woodbury_cache(vmodf))
        empty!(get_row_caches(vmodf))
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
        empty!(get_contingency_cache(vmodf))
        empty!(get_woodbury_cache(vmodf))
        empty!(get_row_caches(vmodf))
    end
    return
end
