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
- `woodbury_cache::Dict{Base.UUID, WoodburyFactors}`:
        Precomputed Woodbury factors keyed by outage UUID.
- `row_caches::Dict{Base.UUID, RowCache}`:
        One RowCache per contingency, keyed by outage UUID.
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
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    contingency_cache::Dict{Base.UUID, ContingencySpec}
    woodbury_cache::Dict{Base.UUID, WoodburyFactors}
    row_caches::Dict{Base.UUID, RowCache}
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
get_arc_axis(mat::VirtualMODF) = mat.axes[1]
get_tol(mat::VirtualMODF) = mat.tol[]

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

    temp_data = zeros(length(bus_ax))
    work_ba_col = zeros(length(valid_ix))
    max_cache_bytes = max_cache_size * MiB

    vmodf = VirtualMODF(
        K,
        BA.data,
        A.data,
        PTDF_A_diag,
        arc_susceptances,
        dist_slack,
        axes,
        look_up,
        valid_ix,
        Dict{Base.UUID, ContingencySpec}(),
        Dict{Base.UUID, WoodburyFactors}(),
        Dict{Base.UUID, RowCache}(),
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
    _register_outage!(vmodf, sys, outage) -> ContingencySpec

Resolve an Outage supplemental attribute to a ContingencySpec and cache it.

Resolution chain:
1. Get ACTransmission components via `PSY.get_associated_components(sys, outage; component_type=PSY.ACTransmission)`
2. Filter to PSY.ACTransmission components
3. Check NetworkReductionData for double-circuit membership
4. Compute Δb: -b_circuit (parallel) or -b_arc (direct)
5. Merge modifications on the same arc
6. Store ContingencySpec keyed by outage UUID
"""
function _register_outage!(
    vmodf::VirtualMODF,
    sys::PSY.System,
    outage::PSY.Outage,
)
    nr = vmodf.network_reduction_data
    outage_uuid = IS.get_uuid(outage)

    # Already registered?
    if haskey(vmodf.contingency_cache, outage_uuid)
        return vmodf.contingency_cache[outage_uuid]
    end

    # Get associated components directly (returns component objects, not UUIDs)
    associated_components = collect(
        PSY.get_associated_components(sys, outage;
            component_type = PSY.ACTransmission),
    )

    if isempty(associated_components)
        error("Outage has no associated ACTransmission components.")
    end

    # Resolve each component to a BranchModification
    mods = BranchModification[]
    component_names = String[]

    for component in associated_components
        push!(component_names, PSY.get_name(component))

        if haskey(nr.reverse_parallel_branch_map, component)
            # Double circuit: partial susceptance change
            arc_tuple = nr.reverse_parallel_branch_map[component]
            arc_idx = vmodf.lookup[1][arc_tuple]
            b_circuit = PSY.get_series_susceptance(component)
            push!(mods, BranchModification(arc_idx, -b_circuit))

        elseif haskey(nr.reverse_direct_branch_map, component)
            # Single circuit: full outage
            arc_tuple = nr.reverse_direct_branch_map[component]
            arc_idx = vmodf.lookup[1][arc_tuple]
            b_arc = vmodf.arc_susceptances[arc_idx]
            push!(mods, BranchModification(arc_idx, -b_arc))

        elseif haskey(nr.reverse_series_branch_map, component)
            # Branch is part of a series (degree-two) chain: full outage of the series arc
            arc_tuple = nr.reverse_series_branch_map[component]
            arc_idx = vmodf.lookup[1][arc_tuple]
            b_arc = vmodf.arc_susceptances[arc_idx]
            push!(mods, BranchModification(arc_idx, -b_arc))

        else
            # No reduction applied — fall back to direct arc tuple lookup using PSY bus numbers.
            arc = PSY.get_arc(component)
            fr = PSY.get_number(PSY.get_from(arc))
            to = PSY.get_number(PSY.get_to(arc))
            arc_tuple = (fr, to)
            if haskey(vmodf.lookup[1], arc_tuple)
                arc_idx = vmodf.lookup[1][arc_tuple]
                b_arc = vmodf.arc_susceptances[arc_idx]
                push!(mods, BranchModification(arc_idx, -b_arc))
            else
                @warn "Branch $(PSY.get_name(component)) arc ($fr, $to) not found in " *
                      "network matrix lookup. Skipping."
            end
        end
    end

    if isempty(mods)
        error("No valid branch modifications found for outage.")
    end

    # Merge modifications on the same arc
    merged = _merge_modifications(mods)

    ctg_name = isempty(component_names) ? string(outage_uuid) :
               join(component_names, "+")
    ctg = ContingencySpec(outage_uuid, ctg_name, merged)

    # Cache the result
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
    _get_woodbury_factors(vmodf, contingency) -> WoodburyFactors

Compute and cache the Woodbury factors for a contingency.
Implements van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹U (A⁻¹ + U⊤B_r⁻¹U)⁻¹ U⊤B_r⁻¹

where U = [ν_{e1} ... ν_{eM}] and A = diag(Δb₁, ..., Δb_M).

The expensive part (M KLU solves + M×M factorization) is shared
across all monitored arcs for this contingency.

!!! warning
    This function is NOT thread-safe. It mutates `vmodf.work_ba_col` on
    every call. Do not call concurrently on the same `VirtualMODF` instance.
"""
function _get_woodbury_factors(
    vmodf::VirtualMODF,
    contingency::ContingencySpec,
)
    if haskey(vmodf.woodbury_cache, contingency.uuid)
        return vmodf.woodbury_cache[contingency.uuid]
    end

    mods = contingency.modifications
    M = length(mods)
    n_bus = length(vmodf.temp_data)

    branch_indices = [mod.arc_index for mod in mods]
    delta_b_vec = [mod.delta_b for mod in mods]

    # Compute Z[:,j] = B⁻¹ν_j for each modified arc, where ν_j = BA[:,e] / b_e
    # is the unscaled incidence direction; Z[:,j] = B⁻¹ A[e,:]⊤
    Z = Matrix{Float64}(undef, n_bus, M)

    for (j, mod) in enumerate(mods)
        e = mod.arc_index
        b_e = vmodf.arc_susceptances[e]

        @inbounds for i in eachindex(vmodf.valid_ix)
            vmodf.work_ba_col[i] = vmodf.BA[vmodf.valid_ix[i], e]
        end
        lin_solve = KLU.solve!(vmodf.K, vmodf.work_ba_col)

        fill!(view(Z, :, j), 0.0)
        @inbounds for i in eachindex(vmodf.valid_ix)
            Z[vmodf.valid_ix[i], j] = lin_solve[i] / b_e
        end
    end

    # K_mat[i,j] = ν_i⊤ B⁻¹ ν_j  (using A * Z to compute all rows efficiently)
    # A is (n_arcs × n_bus), so (A * Z) is (n_arcs × M).
    AZ = vmodf.A * Z  # sparse-dense product: n_arcs × M
    K_mat = zeros(M, M)
    for j in 1:M, i in 1:M
        K_mat[i, j] = AZ[branch_indices[i], j]
    end

    # W = diag(1/Δb) + K_mat = A⁻¹ + U⊤B⁻¹U
    W_mat = LinearAlgebra.diagm(1.0 ./ delta_b_vec) + K_mat

    # Pre-invert W: analytical for M ≤ 2 (avoids LU overhead for the common N-1/N-2 cases),
    # LU-based for M > 2.
    W_inv, is_island = _invert_woodbury_W(W_mat, M)

    wf = WoodburyFactors(Z, W_inv, branch_indices, delta_b_vec, is_island)
    vmodf.woodbury_cache[contingency.uuid] = wf
    return wf
end

"""
    _compute_modf_row(vmodf, monitored_idx, wf) -> Vector{Float64}

Compute the post-contingency PTDF row for a monitored arc.
Uses Woodbury correction: PTDF_m[mon,:] = b_mon_post · ν_mon⊤ · B_m⁻¹

The row is computed as: b_mon_post · (z_m - Z · W⁻¹ · (ν_mon⊤ · Z))
where z_m = B⁻¹ν_mon / b_mon_pre (one additional KLU solve).

!!! warning
    This function is NOT thread-safe. It mutates `vmodf.work_ba_col` on
    every call. Do not call concurrently on the same `VirtualMODF` instance.
"""
function _compute_modf_row(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    n_bus = length(vmodf.temp_data)

    if wf.is_islanding
        @warn "Contingency islands the network. Returning zeros."
        return zeros(n_bus)
    end

    M = length(wf.branch_indices)

    # Effective susceptance of monitored arc after modifications
    b_mon = vmodf.arc_susceptances[monitored_idx]
    for (j, idx) in enumerate(wf.branch_indices)
        if idx == monitored_idx
            b_mon += wf.delta_b[j]
        end
    end
    if abs(b_mon) < eps()
        return zeros(n_bus)
    end

    # z_m = B⁻¹ν_m / b_mon_pre via KLU solve on BA column
    b_mon_pre = vmodf.arc_susceptances[monitored_idx]
    @inbounds for i in eachindex(vmodf.valid_ix)
        vmodf.work_ba_col[i] = vmodf.BA[vmodf.valid_ix[i], monitored_idx]
    end
    lin_solve = KLU.solve!(vmodf.K, vmodf.work_ba_col)

    # Build z_m directly into vmodf.temp_data to avoid allocation
    fill!(vmodf.temp_data, 0.0)
    @inbounds for i in eachindex(vmodf.valid_ix)
        vmodf.temp_data[vmodf.valid_ix[i]] = lin_solve[i] / b_mon_pre
    end

    # ν_m⊤ · Z  (1 × M vector — small, ok to allocate)
    zm_Z = zeros(M)
    for j in 1:M
        zm_Z[j] = dot(view(vmodf.A, monitored_idx, :), view(wf.Z, :, j))
    end

    # Woodbury correction: W⁻¹ · zm_Z, then subtract correction from temp_data in place
    correction_coeff = wf.W_inv * zm_Z
    for j in 1:M
        c = correction_coeff[j]
        @inbounds for n in eachindex(vmodf.temp_data)
            vmodf.temp_data[n] -= c * wf.Z[n, j]
        end
    end

    # Post-contingency PTDF row = b_mon_post · (z_m - correction), now in temp_data
    return b_mon .* vmodf.temp_data
end

"""
    _compute_modf_entry(vmodf, monitored_idx, contingency) -> Vector{Float64}

Compute the post-contingency PTDF row for a monitored arc under the given contingency.
Gets or computes Woodbury factors, then computes the post-contingency PTDF row.

For N-1 contingencies, the result satisfies:
    post_ptdf[mon, :] = pre_ptdf[mon, :] + LODF[mon, e] * pre_ptdf[e, :]
"""
function _compute_modf_entry(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    contingency::ContingencySpec,
)::Vector{Float64}
    wf = _get_woodbury_factors(vmodf, contingency)
    return _compute_modf_row(vmodf, monitored_idx, wf)
end

# --- Row cache management ---

"""
    _get_or_create_row_cache(vmodf, ctg_uuid) -> RowCache

Get or create the per-contingency RowCache for the given UUID.
"""
function _get_or_create_row_cache(vmodf::VirtualMODF, ctg_uuid::Base.UUID)
    if !haskey(vmodf.row_caches, ctg_uuid)
        row_size = length(vmodf.temp_data) * sizeof(Float64)
        vmodf.row_caches[ctg_uuid] =
            RowCache(vmodf.max_cache_size_bytes, Set{Int}(), row_size)
    end
    return vmodf.row_caches[ctg_uuid]
end

# --- getindex: by integer monitored index + ContingencySpec ---

"""
Get the post-contingency PTDF row for monitored arc `monitored_idx` under `contingency`.
Uses per-contingency RowCache for LRU-eviction caching.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    contingency::ContingencySpec,
)
    cache = _get_or_create_row_cache(vmodf, contingency.uuid)

    if haskey(cache, monitored_idx)
        return copy(cache[monitored_idx])
    end

    row = _compute_modf_entry(vmodf, monitored_idx, contingency)

    if get_tol(vmodf) > eps()
        cache[monitored_idx] = sparsify(row, get_tol(vmodf))
    else
        cache[monitored_idx] = copy(row)
    end

    return copy(cache[monitored_idx])
end

# --- getindex: by arc tuple + ContingencySpec ---

"""
Arc-tuple indexed version of getindex for VirtualMODF.

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vmodf::VirtualMODF,
    monitored::Tuple{Int, Int},
    contingency::ContingencySpec,
)
    m_idx = vmodf.lookup[1][monitored]
    return vmodf[m_idx, contingency]
end

# --- getindex: by Int + PSY.Outage (UUID lookup) ---

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
    return vmodf[monitored, ctg]
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

"""
Merge BranchModifications that target the same arc index.
"""
function _merge_modifications(mods::Vector{BranchModification})
    by_arc = Dict{Int, Float64}()
    for m in mods
        by_arc[m.arc_index] = get(by_arc, m.arc_index, 0.0) + m.delta_b
    end
    return [BranchModification(idx, db) for (idx, db) in by_arc]
end
