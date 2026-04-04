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
- `K::KLU.KLUFactorization{Float64, Int}`:
        LU factorization of ABA matrix.
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
- `woodbury_cache::Dict{UInt64, WoodburyFactors}`:
        Precomputed Woodbury factors keyed by modification content hash.
- `row_caches::Dict{UInt64, RowCache}`:
        One RowCache per modification, keyed by modification content hash.
- `subnetwork_axes::Dict{Int, Ax}`:
        Maps reference bus indices to subnetwork axes.
- `tol::Base.RefValue{Float64}`:
        Tolerance for sparsification.
- `max_cache_size_bytes::Int`:
        Max cache size in bytes per contingency.
- `network_reduction_data::NetworkReductionData`:
        Network reduction mappings for branch resolution.
- `temp_data::Vector{Float64}`:
        Scratch vector of size n_buses.
- `work_ba_col::Vector{Float64}`:
        Pre-allocated work array for BA column extraction.
"""
struct VirtualMODF{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{Float64}
    K::KLU.KLUFactorization{Float64, Int}
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
    woodbury_cache::Dict{UInt64, WoodburyFactors}
    row_caches::Dict{UInt64, RowCache}
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    max_cache_size_bytes::Int
    network_reduction_data::NetworkReductionData
    temp_data::Vector{Float64}
    work_ba_col::Vector{Float64}
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

# Woodbury kernel accessors
_get_K(m::VirtualMODF) = m.K
_get_BA(m::VirtualMODF) = m.BA
_get_arc_susceptances(m::VirtualMODF) = m.arc_susceptances
_get_valid_ix(m::VirtualMODF) = m.valid_ix
_get_temp_data(m::VirtualMODF) = m.temp_data
_get_work_ba_col(m::VirtualMODF) = m.work_ba_col

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
    # Use A.subnetwork_axes which has Ax == (arc_ax, bus_ax) matching our axes field type
    subnetwork_axes = A.subnetwork_axes

    BA = BA_Matrix(Ymatrix)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    K = klu(ABA)

    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)

    # Compute raw PTDF diagonal and arc susceptances
    PTDF_A_diag = _get_PTDF_A_diag(K, BA.data, A.data, Set(ref_bus_positions))
    arc_susceptances = _extract_arc_susceptances(BA.data)
    branch_susceptances_by_arc = _extract_branch_susceptances_by_arc(
        BA.data, arc_ax, Ymatrix.network_reduction_data)

    temp_data = zeros(length(bus_ax))
    work_ba_col = zeros(length(valid_ix))
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
        Dict{Base.UUID, ContingencySpec}(),
        Dict{UInt64, WoodburyFactors}(),
        Dict{UInt64, RowCache}(),
        subnetwork_axes,
        Ref(tol),
        max_cache_bytes,
        Ymatrix.network_reduction_data,
        temp_data,
        work_ba_col,
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
    _segment_susceptance_after_outage(segment, tripped_set) -> Float64

Compute the remaining susceptance of a series chain segment after removing
tripped components. Dispatches on segment type to handle both single branches
and parallel groups within a series chain.

Returns 0.0 if the segment (or all branches in a parallel group) is fully tripped.
"""
function _segment_susceptance_after_outage(
    segment::PSY.ACTransmission,
    tripped_set::Set{<:PSY.ACTransmission},
)::Float64
    return segment ∈ tripped_set ? 0.0 : get_series_susceptance(segment)
end

function _segment_susceptance_after_outage(
    segment::BranchesParallel,
    tripped_set::Set{<:PSY.ACTransmission},
)::Float64
    b_remaining = 0.0
    for branch in segment.branches
        if branch ∉ tripped_set
            b_remaining += get_series_susceptance(branch)
        end
    end
    return b_remaining
end

"""
    _compute_series_outage_delta_b(series_chain::BranchesSeries, component::PSY.ACTransmission) -> Float64

Compute the change in equivalent arc susceptance when `component` is tripped
from `series_chain`. Delegates to the vector version.
"""
function _compute_series_outage_delta_b(
    series_chain::BranchesSeries,
    component::PSY.ACTransmission,
)::Float64
    return _compute_series_outage_delta_b(series_chain, [component])
end

"""
    _compute_series_outage_delta_b(series_chain::BranchesSeries, tripped::Vector{<:PSY.ACTransmission}) -> Float64

Compute the change in equivalent arc susceptance when multiple components are
simultaneously tripped from a series chain.

For a series chain with segments of susceptance b₁, b₂, ..., bₙ, the equivalent
susceptance is: b_eq = 1 / (1/b₁ + 1/b₂ + ... + 1/bₙ).

Segments can be individual branches or `BranchesParallel` groups. When a tripped
component is inside a parallel group, only that branch's susceptance is removed
from the group — the rest of the parallel group remains in the series chain.

Returns Δb = b_new - b_old (always negative for outages).
If all segments are fully tripped, returns -b_eq (full arc outage).
"""
function _compute_series_outage_delta_b(
    series_chain::BranchesSeries,
    tripped::Vector{<:PSY.ACTransmission},
)::Float64
    b_old = get_series_susceptance(series_chain)
    tripped_set = Set{PSY.ACTransmission}(tripped)
    remaining_inv_sum = 0.0
    chain_broken = false
    for segment in series_chain
        b_seg = _segment_susceptance_after_outage(segment, tripped_set)
        if b_seg == 0.0
            chain_broken = true
            continue
        else
            remaining_inv_sum += 1.0 / b_seg
        end
    end
    if chain_broken
        return -b_old
    end
    b_new = 1.0 / remaining_inv_sum
    return b_new - b_old
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
    _invert_woodbury_W(W_mat, M) -> (W_inv::Matrix{Float64}, is_islanding::Bool)

Invert the M×M Woodbury W matrix. Uses analytical formulas for M ≤ 2
(the common N-1 and N-2 cases) to avoid LU factorization overhead.
Falls back to LU for M > 2.
"""
function _invert_woodbury_W(
    W_mat::Matrix{Float64},
    M::Int,
)::Tuple{Matrix{Float64}, Bool}
    if M == 1
        w = W_mat[1, 1]
        is_island = abs(w) < MODF_ISLANDING_TOLERANCE
        W_inv = Matrix{Float64}(undef, 1, 1)
        W_inv[1, 1] = is_island ? 0.0 : 1.0 / w
        return W_inv, is_island
    elseif M == 2
        a, b, c, d = W_mat[1, 1], W_mat[1, 2], W_mat[2, 1], W_mat[2, 2]
        det_W = a * d - b * c
        is_island = abs(det_W) < MODF_ISLANDING_TOLERANCE
        W_inv = Matrix{Float64}(undef, 2, 2)
        if is_island
            fill!(W_inv, 0.0)
        else
            inv_det = 1.0 / det_W
            W_inv[1, 1] = d * inv_det
            W_inv[1, 2] = -b * inv_det
            W_inv[2, 1] = -c * inv_det
            W_inv[2, 2] = a * inv_det
        end
        return W_inv, is_island
    else
        W_lu = LinearAlgebra.lu(W_mat)
        is_island = any(i -> abs(W_lu.U[i, i]) < MODF_ISLANDING_TOLERANCE, 1:M)
        W_inv = is_island ? zeros(M, M) : LinearAlgebra.inv(W_lu)
        return W_inv, is_island
    end
end

"""
    _get_woodbury_factors(vmodf, mod) -> WoodburyFactors

Compute and cache the Woodbury factors for a network modification.
Delegates to the shared Woodbury kernel `_compute_woodbury_factors`.
Caches by content hash of the modification.

!!! warning
    This function is NOT thread-safe. It mutates `vmodf.work_ba_col` on
    every call. Do not call concurrently on the same `VirtualMODF` instance.
"""
function _get_woodbury_factors(
    vmodf::VirtualMODF,
    mod::NetworkModification,
)
    key = hash(mod) % UInt64
    if haskey(vmodf.woodbury_cache, key)
        return vmodf.woodbury_cache[key]
    end
    wf = _compute_woodbury_factors(vmodf, mod.modifications)
    vmodf.woodbury_cache[key] = wf
    return wf
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
    _get_or_create_row_cache(vmodf, key) -> RowCache

Get or create the per-modification RowCache for the given label key.
"""
function _get_or_create_row_cache(vmodf::VirtualMODF, key::UInt64)
    if !haskey(vmodf.row_caches, key)
        row_size = length(vmodf.temp_data) * sizeof(Float64)
        vmodf.row_caches[key] =
            RowCache(vmodf.max_cache_size_bytes, Set{Int}(), row_size)
    end
    return vmodf.row_caches[key]
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
    key = hash(mod) % UInt64
    cache = _get_or_create_row_cache(vmodf, key)

    if haskey(cache, monitored_idx)
        return copy(cache[monitored_idx])
    end

    row = _compute_modf_entry(vmodf, monitored_idx, mod)

    if get_tol(vmodf) > eps()
        cache[monitored_idx] = sparsify(row, get_tol(vmodf))
    else
        cache[monitored_idx] = row
    end

    return copy(cache[monitored_idx])
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
    if !haskey(vmodf.contingency_cache, outage_uuid)
        error(
            "Outage (UUID=$outage_uuid) is not registered. " *
            "Construct VirtualMODF with the system containing this outage.",
        )
    end
    ctg = vmodf.contingency_cache[outage_uuid]
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
    empty!(vmodf.woodbury_cache)
    empty!(vmodf.row_caches)
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
    empty!(vmodf.contingency_cache)
    empty!(vmodf.woodbury_cache)
    empty!(vmodf.row_caches)
    return
end
