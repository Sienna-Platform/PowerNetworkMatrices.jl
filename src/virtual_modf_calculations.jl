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
- `woodbury_cache::Dict{Base.UUID, W}`:
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
struct VirtualMODF{Ax, L <: NTuple{2, Dict}, W <: WoodburyFactors} <: PowerNetworkMatrix{Float64}
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
    woodbury_cache::Dict{Base.UUID, W}
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

    WF_concrete = WoodburyFactors{LinearAlgebra.LU{Float64, Matrix{Float64}, Vector{Int}}}
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
        Dict{Base.UUID, WF_concrete}(),
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
