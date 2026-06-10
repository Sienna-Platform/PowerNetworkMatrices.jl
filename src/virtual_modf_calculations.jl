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

# Thread-safety

Concurrent `getindex` is safe but serialized: `solver_lock` (a `ReentrantLock`)
is held for the full body of `getindex`, `clear_caches!`, and `clear_all_caches!`,
so Dict mutations on the cache structures and the libklu solves it wraps all
run under a single mutex. libklu activity additionally serializes through the
process-wide `_LIBKLU_LOCK`. Two threads racing on a first-time query for the
same modification serialize on `solver_lock`; the second observes the populated
cache and skips the recomputation.

# Arguments
- `K::KLULinSolveCache{Float64}`:
        ABA factorization (single cache; concurrent `getindex` callers
        serialize through `solver_lock` and `_LIBKLU_LOCK`).
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix.
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence matrix.
- `PTDF_A_diag::Vector{Float64}`:
        Diagonal of `PTDF·A` (H[e,e] values). Lazily populated on the first
        read of `vmodf.PTDF_A_diag`; empty until then.
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
- `bus_to_valid_idx::Vector{Int}`:
        Inverse of `valid_ix`: `bus_to_valid_idx[b]` is the position of bus
        `b` inside `valid_ix`, or 0 if `b` is a reference bus. Lets the
        Woodbury kernel iterate the nonzeros of a `BA` column directly.
- `contingency_cache::Dict{Base.UUID, ContingencySpec}`:
        Resolved contingencies keyed by outage UUID.
- `woodbury_cache::Dict{NetworkModification, WoodburyFactors}`:
        Precomputed Woodbury factors keyed by modification.
- `row_caches::Dict{NetworkModification, RowCache}`:
        One `RowCache` per modification. Mutations are serialized by
        `solver_lock`, the same mutex that wraps the libklu solve, so no
        separate cache lock is needed.
- `subnetwork_axes::Dict{Int, Ax}`:
        Maps reference bus indices to subnetwork axes.
- `tol::Base.RefValue{Float64}`:
        Tolerance for sparsification.
- `max_cache_size_bytes::Int`:
        Max cache size in bytes per contingency.
- `network_reduction_data::NetworkReductionData`:
        Network reduction mappings for branch resolution. The buses retained
        through reduction are the keys of its `bus_reduction_map`; the
        per-reduction `irreducible_buses` field holds only that step's protected
        working set, not the full retained-bus record.
- `temp_data::Vector{Vector{Float64}}`:
        Single-element scratch vector of size n_buses.
- `work_ba_col::Vector{Vector{Float64}}`:
        Single-element work array for BA column extraction.
- `solver_lock::ReentrantLock`:
        Reentrant; held for the duration of `getindex` / `clear_*caches!`.
        Serializes both libklu solves and Dict mutations on the caches;
        combined with `_LIBKLU_LOCK` ensures one libklu call at a time.
- `system_uuid::Union{Base.UUID, Nothing}`:
        UUID of the system used to construct the matrix, used to validate that
        modification operations are applied to the correct system.
"""
struct VirtualMODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    K::K
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    PTDF_A_diag::Vector{Float64}
    arc_susceptances::Vector{Float64}
    branch_susceptances_by_arc::Vector{Vector{Float64}}
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    bus_to_valid_idx::Vector{Int}
    contingency_cache::Dict{Base.UUID, ContingencySpec}
    woodbury_cache::Dict{NetworkModification, WoodburyFactors}
    row_caches::Dict{NetworkModification, RowCache}
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    max_cache_size_bytes::Int
    network_reduction_data::NetworkReductionData
    temp_data::Vector{Vector{Float64}}
    work_ba_col::Vector{Vector{Float64}}
    solver_lock::ReentrantLock
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

function Base.getproperty(vmodf::VirtualMODF, name::Symbol)
    name === :PTDF_A_diag && return _get_ptdf_a_diag_lazy!(vmodf)
    return getfield(vmodf, name)
end

# Use `getfield` for every field read here — `vmodf.PTDF_A_diag` would
# recurse via `getproperty`.
function _get_ptdf_a_diag_lazy!(vmodf::VirtualMODF)
    diag = getfield(vmodf, :PTDF_A_diag)
    !isempty(diag) && return diag
    @lock getfield(vmodf, :solver_lock) begin
        diag = getfield(vmodf, :PTDF_A_diag)
        !isempty(diag) && return diag
        n_arcs = length(getfield(vmodf, :axes)[1])
        @info "Computing VirtualMODF.PTDF_A_diag on first access ($n_arcs arcs)."
        t0 = time_ns()
        K = getfield(vmodf, :K)
        BA = getfield(vmodf, :BA)
        A = getfield(vmodf, :A)
        ref_bus_set = _ref_bus_positions(vmodf)
        new_diag = _get_PTDF_A_diag(K, BA, A, ref_bus_set)
        resize!(diag, length(new_diag))
        copyto!(diag, new_diag)
        elapsed = (time_ns() - t0) / 1e9
        @info "Computed VirtualMODF.PTDF_A_diag in $(round(elapsed; digits = 2)) s (cached)."
        return diag
    end
end

function _ref_bus_positions(vmodf::VirtualMODF)
    n_buses = length(getfield(vmodf, :axes)[2])
    valid_ix = getfield(vmodf, :valid_ix)
    return Set{Int}(setdiff(1:n_buses, valid_ix))
end

"""
$(TYPEDSIGNATURES)

Return `H[e, e]` for each arc `e`. Same as `vmodf.PTDF_A_diag`; both trigger
the lazy compute on first call and return the cached vector thereafter.
"""
get_PTDF_A_diag(vmodf::VirtualMODF) = vmodf.PTDF_A_diag

# Woodbury kernel accessors. The Woodbury kernel always goes through
# `with_solver` so it picks up the `solver_lock` + `_LIBKLU_LOCK` chain.
_get_BA(m::VirtualMODF) = m.BA
_get_arc_susceptances(m::VirtualMODF) = m.arc_susceptances
_get_valid_ix(m::VirtualMODF) = m.valid_ix

function _compute_woodbury_factors(
    mat::VirtualMODF,
    modifications::Tuple{Vararg{ArcModification}},
)::WoodburyFactors
    return with_solver(
        mat.K,
        mat.work_ba_col,
        mat.temp_data,
        mat.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        _compute_woodbury_factors_impl(
            K_solver,
            work_ba_col,
            temp_data,
            mat.BA,
            mat.arc_susceptances,
            mat.valid_ix,
            mat.bus_to_valid_idx,
            modifications,
        )
    end
end

function _apply_woodbury_correction(
    mat::VirtualMODF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return with_solver(
        mat.K,
        mat.work_ba_col,
        mat.temp_data,
        mat.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        _apply_woodbury_correction_impl(
            K_solver,
            work_ba_col,
            temp_data,
            mat.BA,
            mat.arc_susceptances,
            mat.valid_ix,
            mat.bus_to_valid_idx,
            monitored_idx,
            wf,
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

When `network_reductions` are supplied, they are adjusted to retain the buses of
every outaged component and the components each outage declares monitored
(`get_monitored_components`). This is mandatory: the ABA/Woodbury solve runs on the
reduced network, so a branch in a contingency must survive reduction.

# Arguments
- `sys::PSY.System`: Power system to build from

# Keyword Arguments
- `dist_slack::Vector{Float64}`: Distributed slack weights (default: empty)
- `linear_solver::String = _default_linear_solver()`: Linear solver for the
        ABA factorization. Options: "KLU", "AppleAccelerate". Defaults to
        "AppleAccelerate" on macOS and "KLU" elsewhere.
- `tol::Float64`: Tolerance for row sparsification (default: eps())
- `max_cache_size::Int`: Max cache size in MiB per contingency (default: MAX_CACHE_SIZE_MiB)
- `network_reductions::Vector{NetworkReduction}`: Network reductions to apply
- `automatically_register_outages::Bool`: Register all system Outage attributes (default: true)
"""
function VirtualMODF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    linear_solver::String = _default_linear_solver(),
    tol::Float64 = eps(),
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
    solver = resolve_linear_solver(linear_solver)

    # ZIBR is auto-applied during Ybus construction; reject it in user reductions since
    # those are applied manually below.
    for r in network_reductions
        _reject_zibr_in_user_reductions(r)
    end

    # Build the base Ybus once (zero-impedance reduction auto-applied). It supplies the
    # ZI-survivor map for the protection set and is the starting point for the reductions.
    Ymatrix = Ybus(sys; irreducible_buses = irreducible_buses, kwargs...)

    # Outage/monitored buses are auto-protected so contingency branches survive
    # reduction. Registering an outage with previously-unseen monitored components
    # after construction shifts this set and requires rebuilding the MODF.
    protected_buses = _collect_protected_buses(sys, Ymatrix)
    applied_irreducible = union(irreducible_buses, protected_buses)

    # Protect via two channels: radial/degree-two read the container's irreducible set,
    # while Ward reads `study_buses` (augmented separately).
    _inject_protected_buses!(Ymatrix, protected_buses)
    applied_reductions = _augment_ward_reductions(network_reductions, applied_irreducible)
    for reduction in applied_reductions
        Ymatrix = build_reduced_ybus(Ymatrix, sys, reduction)
    end
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
    K = _create_factorization(solver, ABA)

    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)
    bus_to_valid_idx = _build_bus_to_valid_idx(length(bus_ax), valid_ix)

    # Empty: populated lazily on first read of `vmodf.PTDF_A_diag`.
    PTDF_A_diag = Float64[]
    arc_susceptances = _extract_arc_susceptances(BA.data)
    branch_susceptances_by_arc =
        _extract_branch_susceptances_by_arc(BA.data, arc_ax, Ymatrix.network_reduction_data)

    # Single scratch slot — solves serialize via `solver_lock` + `_LIBKLU_LOCK`.
    temp_data = [zeros(length(bus_ax))]
    work_ba_col = [zeros(length(valid_ix))]
    max_cache_bytes = max_cache_size * MiB

    vmodf = VirtualMODF(
        K,
        BA.data,
        A.data,
        PTDF_A_diag,
        arc_susceptances,
        branch_susceptances_by_arc,
        dist_slack,
        axes,
        look_up,
        valid_ix,
        bus_to_valid_idx,
        Dict{Base.UUID, ContingencySpec}(),
        Dict{NetworkModification, WoodburyFactors}(),
        Dict{NetworkModification, RowCache}(),
        subnetwork_axes,
        Ref(tol),
        max_cache_bytes,
        Ymatrix.network_reduction_data,
        temp_data,
        work_ba_col,
        ReentrantLock(),
        IS.get_uuid(sys),
    )

    # Auto-register all outage attributes from the system
    automatically_register_outages && _register_all_outages!(vmodf, sys)

    return vmodf
end

"""
    _warn_if_transmission_dropped(sys, outage, mod)

Warn when an outage references `ACTransmission` components but its modification has
no arc modifications — those branches were eliminated by reduction, so the
contingency would silently return the unmodified base row. Outages touching only
non-network components (generators, loads) resolve empty legitimately and are not
flagged.
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

"""
    _register_all_outages!(vmodf, sys)

Bulk-register all Outage supplemental attributes in the system.
Called automatically by the VirtualMODF constructor.

Uses `PSY.get_supplemental_attributes(PSY.Outage, sys)` which accepts
the abstract type and iterates over all concrete subtypes
(PlannedOutage, UnplannedOutage).
"""
function _register_all_outages!(vmodf::VirtualMODF, sys::PSY.System)
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
    outage_uuid = IS.get_uuid(outage)
    if haskey(vmodf.contingency_cache, outage_uuid)
        @warn "Outage with UUID $(outage_uuid) is already registered; skipping."
        return
    end
    mod = NetworkModification(vmodf, sys, outage)
    ctg = ContingencySpec(outage_uuid, mod)
    vmodf.contingency_cache[outage_uuid] = ctg
    _warn_if_transmission_dropped(sys, outage, mod)
    return
end

# --- Woodbury factor computation ---

"""
    _get_woodbury_factors(vmodf, mod) -> WoodburyFactors

Return cached Woodbury factors for a modification, computing them on a miss.
Caller holds `solver_lock`; the inner `_compute_woodbury_factors` re-enters
that same lock (it's a `ReentrantLock`).
"""
function _get_woodbury_factors(vmodf::VirtualMODF, mod::NetworkModification)
    # Use the do-block form, NOT `get!(dict, key, default)`: Julia evaluates
    # function arguments eagerly, so the 3-arg form would run the M KLU solves
    # on every call (cache hit included), defeating the cache.
    return get!(vmodf.woodbury_cache, mod) do
        _compute_woodbury_factors(vmodf, mod.arc_modifications)
    end
end

"""
    _compute_modf_entry(vmodf, monitored_idx, mod) -> Vector{Float64}

Compute the post-modification PTDF row for a monitored arc under the given modification.
Gets or computes Woodbury factors, then applies the Woodbury correction.

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
    return @lock vmodf.solver_lock begin
        rc = get!(vmodf.row_caches, mod) do
            row_size = length(vmodf.temp_data[1]) * sizeof(Float64)
            RowCache(vmodf.max_cache_size_bytes, Set{Int}(), row_size)
        end
        if haskey(rc, monitored_idx)
            return copy(rc[monitored_idx])
        end
        row = _compute_modf_entry(vmodf, monitored_idx, mod)
        if get_tol(vmodf) > eps()
            stored = sparsify(row, get_tol(vmodf))
        else
            stored = row
        end
        rc[monitored_idx] = stored
        copy(stored)
    end
end

# Row index for a monitored arc, with a clear error when it was reduced away (else
# a raw KeyError). Only the canonical (from, to) orientation is accepted: the row
# comes from `BA[:, idx]`, so the reversed tuple would silently sign-flip it
# (matches VirtualPTDF / VirtualLODF).
function _monitored_arc_index(vmodf::VirtualMODF, monitored::Tuple{Int, Int})
    arc_lookup = vmodf.lookup[1]
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
Delegates to the NetworkModification-based getindex.

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

# --- getindex: by PSY.Outage (UUID lookup → ContingencySpec → NetworkModification) ---

"""
Get the post-contingency PTDF row for monitored arc `monitored` when outage `outage` trips.
The outage must have been registered at VirtualMODF construction time.

$(TYPEDSIGNATURES)
"""
function Base.getindex(vmodf::VirtualMODF, monitored::Int, outage::PSY.Outage)
    outage_uuid = IS.get_uuid(outage)
    # Pair with the locked `empty!` in `clear_all_caches!`; without it, a
    # concurrent clear could rehash `contingency_cache` mid-lookup.
    ctg = @lock vmodf.solver_lock begin
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
    @lock vmodf.solver_lock begin
        empty!(vmodf.woodbury_cache)
        empty!(vmodf.row_caches)
    end
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
    @lock vmodf.solver_lock begin
        empty!(vmodf.contingency_cache)
        empty!(vmodf.woodbury_cache)
        empty!(vmodf.row_caches)
    end
    return
end
