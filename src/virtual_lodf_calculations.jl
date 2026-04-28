"""
The Virtual Line Outage Distribution Factor (VirtualLODF) structure gathers
the rows of the LODF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualLODF is initialized with no row stored.

The VirtualLODF struct is indexed using branch names.

# Thread-safety

`VirtualLODF` is parallel-safe for concurrent `getindex`: the factorization
is held in a `KLULinSolvePool`, scratch arrays are sized per-worker, and the
row cache is guarded by a `ReentrantLock`. The same applies to
`get_partial_lodf_row`.

# Arguments
- `K::KLULinSolvePool{Float64}`:
        Pool of independent ABA factorizations, one per worker.
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix.
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence matrix.
- `inv_PTDF_A_diag::Vector{Float64}`:
        Vector contiaining the element-wise reciprocal of the diagonal elements
        coming from multuiplying the PTDF matrix with th Incidence matrix
- `PTDF_A_diag::Vector{Float64}`:
        Raw diagonal elements of the PTDF·A product (H[e,e] values), before
        tolerance clamping. Used for partial susceptance change computations.
- `arc_susceptances::Vector{Float64}`:
        Effective susceptance for each arc, extracted from the BA matrix.
        For arc j, this is the absolute value of the first nonzero in BA column j.
        BA columns always have the structure [+b, -b] (from-bus and to-bus entries),
        so both nonzeros have the same magnitude.
- `branch_susceptances_by_arc::Vector{Vector{Float64}}`:
        Per-branch susceptances for each arc. For single-branch arcs, contains
        one element equal to the arc susceptance. For parallel branches, contains
        one entry per branch in the parallel group.
- `ref_bus_positions::Set{Int}`:
        Vector containing the indexes of the rows of the transposed BA matrix
        corresponding to the reference buses.
- `dist_slack::Vector{Float64}`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `axes<:NTuple{2, Dict}`:
        Tuple containing two vectors showing the branch names.
- `lookup<:NTuple{2, Dict}`:
        Tuple containing two dictionaries, mapping the branches names
        the enumerated row indexes indexes.
- `valid_ix::Vector{Int}`:
        Vector containing the row/columns indices of matrices related the buses
        which are not slack ones.
- `temp_data::Vector{Vector{Float64}}`:
        Per-worker temporary vector for internal use (one per pool worker).
- `cache::RowCache`:
        Cache where LODF rows are stored.
- `cache_lock::ReentrantLock`:
        Guards `cache` reads/writes for parallel `getindex` callers.
- `subnetworks::Dict{Int, Set{Int}}`:
        Dictionary containing the subsets of buses defining the different subnetwork of the system.
- `tol::Base.RefValue{Float64}`:
        Tolerance related to scarification and values to drop.
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `work_ba_col::Vector{Vector{Float64}}`:
        Per-worker BA-column scratch buffer (one per pool worker).
"""
struct VirtualLODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}} <:
       PowerNetworkMatrix{Float64}
    K::KLULinSolvePool{Float64}
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    inv_PTDF_A_diag::Vector{Float64}
    PTDF_A_diag::Vector{Float64}
    arc_susceptances::Vector{Float64}
    branch_susceptances_by_arc::Vector{Vector{Float64}}
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    temp_data::Vector{Vector{Float64}}
    cache::RowCache
    cache_lock::ReentrantLock
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    network_reduction_data::NetworkReductionData
    work_ba_col::Vector{Vector{Float64}}
end

get_axes(M::VirtualLODF) = M.axes
get_lookup(M::VirtualLODF) = M.lookup
get_ref_bus(M::VirtualLODF) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::VirtualLODF) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::VirtualLODF) = M.network_reduction_data
get_arc_lookup(M::VirtualLODF) = M.lookup[1]

"""
Number of pool workers (= max concurrent solves) backing `vlodf`.
"""
nworkers(vlodf::VirtualLODF) = nworkers(vlodf.K)

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualLODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

function _get_PTDF_A_diag(
    K::KLULinSolveCache{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    ref_bus_positions::Set{Int},
)
    n_branches = size(BA, 2)
    n_buses = size(BA, 1)
    diag_ = zeros(n_branches)

    # Pre-compute valid indices (non-reference buses)
    valid_ix = setdiff(1:n_buses, ref_bus_positions)
    n_valid = length(valid_ix)

    # Pre-allocate work arrays for efficiency
    ba_col = zeros(n_valid)
    ptdf_row = zeros(n_buses)

    # For each branch, compute PTDF row and dot with incidence column
    for i in 1:n_branches
        # Extract BA column for valid indices
        fill!(ba_col, 0.0)
        for idx in 1:n_valid
            bus_idx = valid_ix[idx]
            ba_col[idx] = BA[bus_idx, i]
        end

        solve!(K, ba_col)

        # Map back to full bus indices
        fill!(ptdf_row, 0.0)
        for idx in 1:n_valid
            ptdf_row[valid_ix[idx]] = ba_col[idx]
        end

        # Compute diagonal element: sum of PTDF[i,j] * A[i,j] for all buses j
        for j in 1:n_buses
            diag_[i] += ptdf_row[j] * A[i, j]
        end
    end
    return diag_
end

"""
Extract the effective susceptance for each arc from the BA matrix.
For arc j, the susceptance is the absolute value of the first nonzero in BA column j.
BA columns always have the structure [+b, -b] (from-bus and to-bus entries),
so both nonzeros have the same magnitude.
"""
function _extract_arc_susceptances(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
)::Vector{Float64}
    n_arcs = size(BA, 2)
    b = Vector{Float64}(undef, n_arcs)
    nzv = SparseArrays.nonzeros(BA)
    for j in 1:n_arcs
        rng = nzrange(BA, j)
        b[j] = isempty(rng) ? 0.0 : abs(nzv[first(rng)])
    end
    return b
end

"""
    _extract_branch_susceptances_by_arc(BA, arc_ax, nr_data) -> Vector{Vector{Float64}}

Extract per-branch susceptances for each arc. For arcs with a single branch,
returns a one-element vector equal to the arc susceptance. For arcs with
parallel branches (double circuits), returns one entry per branch. For arcs
with series-reduced branches (D2 reduction), returns one entry per segment.

This enables single-branch contingencies on parallel and series-reduced arcs.
"""
function _extract_branch_susceptances_by_arc(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_ax::Vector{Tuple{Int, Int}},
    nr_data::NetworkReductionData,
)::Vector{Vector{Float64}}
    n_arcs = size(BA, 2)
    nzv = SparseArrays.nonzeros(BA)
    result = Vector{Vector{Float64}}(undef, n_arcs)

    for j in 1:n_arcs
        arc = arc_ax[j]
        rng = nzrange(BA, j)
        arc_b = isempty(rng) ? 0.0 : abs(nzv[first(rng)])

        if haskey(nr_data.parallel_branch_map, arc)
            bp = nr_data.parallel_branch_map[arc]
            result[j] = Float64[
                get_series_susceptance(branch) for branch in bp.branches
            ]
        elseif haskey(nr_data.series_branch_map, arc)
            bs = nr_data.series_branch_map[arc]
            result[j] = Float64[
                get_series_susceptance(segment) for segment in bs
            ]
        else
            result[j] = [arc_b]
        end
    end

    return result
end

"""
Builds the Virtual LODF matrix from a system. The return is a VirtualLODF
struct with an empty cache.

# Arguments
- `sys::PSY.System`:
        PSY system for which the matrix is constructed

# Keyword Arguments
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `nworkers::Int`:
        Number of parallel workers in the underlying KLU pool. Defaults to
        `max(1, Threads.nthreads() - 1)`.
- `kwargs...`:
        other keyword arguments used by VirtualPTDF
"""
function VirtualLODF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    tol::Float64 = eps(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    nworkers::Int = max(1, Threads.nthreads() - 1),
    kwargs...,
)
    if length(dist_slack) != 0
        @info "Distributed bus"
    end
    Ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    ref_bus_positions = get_ref_bus_position(Ymatrix)
    A = IncidenceMatrix(Ymatrix)
    arc_ax = get_arc_axis(A)
    axes = (arc_ax, arc_ax)
    arc_ax_ref = make_ax_ref(arc_ax)
    look_up = (arc_ax_ref, arc_ax_ref)
    subnetwork_axes = make_arc_arc_subnetwork_axes(A)
    BA = BA_Matrix(Ymatrix)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    K_pool = KLULinSolvePool(ABA; nworkers = nworkers)
    bus_ax = get_bus_axis(A)

    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)
    # PTDF diagonal precompute is single-threaded; route through one pool
    # worker so the pool stays the only owner of the factorization.
    PTDF_diag = with_worker(K_pool) do cache, _idx
        _get_PTDF_A_diag(cache, BA.data, A.data, Set(ref_bus_positions))
    end
    PTDF_A_diag_raw = copy(PTDF_diag)
    arc_susceptances = _extract_arc_susceptances(BA.data)
    branch_susceptances_by_arc = _extract_branch_susceptances_by_arc(
        BA.data, arc_ax, Ymatrix.network_reduction_data)
    PTDF_diag[PTDF_diag .> 1 - LODF_ENTRY_TOLERANCE] .= 0.0

    if isempty(persistent_arcs)
        empty_cache =
            RowCache(max_cache_size * MiB, Set{Int}(), length(bus_ax) * sizeof(Float64))
    else
        init_persistent_dict = Set{Int}(A.lookup[1][k] for k in persistent_arcs)
        empty_cache =
            RowCache(
                max_cache_size * MiB,
                init_persistent_dict,
                length(bus_ax) * sizeof(Float64),
            )
    end

    n_scratch = nworkers
    temp_data = [zeros(length(bus_ax)) for _ in 1:n_scratch]
    work_ba_col = [zeros(length(valid_ix)) for _ in 1:n_scratch]

    return VirtualLODF(
        K_pool,
        BA.data,
        A.data,
        1.0 ./ (1.0 .- PTDF_diag),
        PTDF_A_diag_raw,
        arc_susceptances,
        branch_susceptances_by_arc,
        dist_slack,
        axes,
        look_up,
        valid_ix,
        temp_data,
        empty_cache,
        ReentrantLock(),
        subnetwork_axes,
        Ref(tol),
        Ymatrix.network_reduction_data,
        work_ba_col,
    )
end

# Overload Base functions

"""
Checks if the any of the fields of VirtualLODF is empty.
"""
function Base.isempty(vlodf::VirtualLODF)
    for name in fieldnames(typeof(vlodf))
        # note: impossible to define empty KLU field
        if name != :K && isempty(getfield(vlodf, name))
            @debug "Field " * string(name) * " is empty."
            return true
        end
    end
    return false
end

"""
Shows the size of the whole LODF matrix, not the number of rows stored.
"""
Base.size(vlodf::VirtualLODF) = (size(vlodf.BA, 2), size(vlodf.BA, 2))

"""
Gives the cartesian indexes of the LODF matrix.
"""
Base.eachindex(vlodf::VirtualLODF) = CartesianIndices(size(vlodf))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualLODF) = "VirtualLODF"
end

# Compute the LODF row for `row` using exclusive per-worker scratch. Pure
# computation: no cache reads/writes, no tolerance application.
function _compute_lodf_row(vlodf::VirtualLODF, row::Int)::Vector{Float64}
    return with_worker(vlodf.K) do cache, idx
        work_ba_col = vlodf.work_ba_col[idx]
        temp_data = vlodf.temp_data[idx]
        @inbounds for i in eachindex(vlodf.valid_ix)
            work_ba_col[i] = vlodf.BA[vlodf.valid_ix[i], row]
        end
        solve!(cache, work_ba_col)

        fill!(temp_data, 0.0)
        @inbounds for i in eachindex(vlodf.valid_ix)
            temp_data[vlodf.valid_ix[i]] = work_ba_col[i]
        end

        lodf_row = (vlodf.A * temp_data) .* vlodf.inv_PTDF_A_diag
        lodf_row[row] = -1.0
        return lodf_row
    end
end

function _getindex(
    vlodf::VirtualLODF,
    row::Int,
    column::Union{Int, Colon},
)
    return cached_row_lookup(
        vlodf.cache, vlodf.cache_lock, row, column, get_tol(vlodf),
    ) do
        _compute_lodf_row(vlodf, row)
    end
end

"""
Gets the value of the element of the LODF matrix given the row and column indices
corresponding to the selected and outage branch respectively. If `column` is a Colon then
the entire row is returned.

# Arguments
- `vlodf::VirtualLODF`:
        VirtualLODF struct where to evaluate and store the row values.
- `row`:
        selected line name
- `column`:
        outage line name. If `Colon` then get the values of the whole row.
"""
function Base.getindex(vlodf::VirtualLODF, row, column)
    row_, column_ = to_index(vlodf, row, column)
    return _getindex(vlodf, row_, column_)
end

# Define for ambiguity resolution
function Base.getindex(vlodf::VirtualLODF, row::Integer, column::Integer)
    return _getindex(vlodf, row, column)
end

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualLODF, _, idx...) = error("Operation not supported by VirtualLODF")

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualLODF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualLODF")

"""
    get_lodf_data(mat::VirtualLODF) -> Dict{Int, Vector{Float64}}

Get the cached LODF row data from a [`VirtualLODF`](@ref) matrix.

Unlike [`get_lodf_data(::LODF)`](@ref), which returns a dense matrix, this returns
a dictionary mapping row indices to lazily computed row vectors.

# Arguments
- `mat::VirtualLODF`: The virtual LODF matrix

# Returns
- `Dict{Int, Vector{Float64}}`: Cached row data keyed by row index
"""
get_lodf_data(mat::VirtualLODF) = mat.cache.temp_cache

function get_arc_axis(mat::VirtualLODF)
    return mat.axes[1]
end

""" Gets the tolerance used for sparsifying the rows of the VirtualLODF matrix"""
function get_tol(mat::VirtualLODF)
    return mat.tol[]
end

"""
    _getindex_partial(vlodf, arc_idx, delta_b) -> Vector{Float64}

Compute the partial LODF column for a susceptance change `delta_b` on arc `arc_idx`.

Parallel-safe: routes the solve through `with_worker(vlodf.K)` so each
caller acquires exclusive per-worker scratch.

Uses the Sherman-Morrison (matrix inversion lemma) formula derived from DC power flow
sensitivity analysis. For a change Δb in the susceptance of arc e, the change in flow
on monitoring arc ℓ per unit pre-change flow on arc e is:

    partial_LODF[ℓ, e] = α · (b_ℓ / b_e) · H[ℓ,e] / (1 - α · H[e,e])

where:
- α = -Δb / b_e   (positive for outage/decrease, negative for increase)
- H[ℓ, e] = (A · (ABA)⁻¹ · BA)[ℓ, e] = b_e · C[e, ℓ]  (computed via KLU solve)
- b_ℓ = susceptance of monitoring arc ℓ
- H[e,e] = PTDF_A_diag[e]

When `delta_b = -b_e` (full outage), α = 1 and this reduces to the standard LODF column:
    LODF[ℓ, e] = b_ℓ · C[e, ℓ] / (1 - H[e,e])
When `delta_b = 0`, returns zeros (no change).
The self-element (ℓ = e) is overridden to -1.0 for full outage per standard LODF convention.
"""
function _getindex_partial(
    vlodf::VirtualLODF,
    arc_idx::Int,
    delta_b::Float64,
)::Vector{Float64}
    n_arcs = size(vlodf.BA, 2)

    # Zero change means zero redistribution.
    if abs(delta_b) < eps()
        return zeros(n_arcs)
    end

    b_arc = vlodf.arc_susceptances[arc_idx]
    if b_arc == 0.0
        return zeros(n_arcs)
    end

    return with_worker(vlodf.K) do cache, idx
        work_ba_col = vlodf.work_ba_col[idx]
        temp_data = vlodf.temp_data[idx]

        # Steps 1-2: Compute B⁻¹(b_e · ν_e) via KLU solve.
        @inbounds for i in eachindex(vlodf.valid_ix)
            work_ba_col[i] = vlodf.BA[vlodf.valid_ix[i], arc_idx]
        end
        solve!(cache, work_ba_col)

        # Step 3: Map solution back to full bus space.
        fill!(temp_data, 0.0)
        @inbounds for i in eachindex(vlodf.valid_ix)
            temp_data[vlodf.valid_ix[i]] = work_ba_col[i]
        end

        # Step 4: H_col[ℓ] = b_e · C[e,ℓ] for all monitoring arcs ℓ.
        H_col = vlodf.A * temp_data

        # Step 5: Scalar denominator: 1 - α · H[e,e].
        # α = -Δb / b_e (positive for outage/decrease, negative for increase).
        H_ee = vlodf.PTDF_A_diag[arc_idx]
        alpha = -delta_b / b_arc
        denom = 1.0 - alpha * H_ee

        # Step 6: Partial LODF column: scale by b_ℓ/b_e to convert from C[e,ℓ] to b_ℓ·C[e,ℓ].
        # partial_lodf[ℓ] = α · b_ℓ/b_e · H_col[ℓ] / denom
        #                 = α · b_ℓ · C[e,ℓ] / (1 - α · H_ee)
        partial_lodf =
            (alpha / (denom * b_arc)) .* (vlodf.arc_susceptances .* H_col)

        # By convention, the outaged arc's own redistribution factor is -1.0 for a full
        # outage: the arc carries -100% of its own pre-contingency flow post-outage.
        # The raw formula gives α·H[e,e]/denom for the self-element, which is
        # b_e·C[e,e]/(1-b_e·C[e,e]) = H_ee/(1-H_ee) ≠ -1 in general.
        if abs(delta_b + b_arc) < eps() * b_arc
            partial_lodf[arc_idx] = -1.0
        end

        return partial_lodf
    end
end

"""
    get_partial_lodf_row(vlodf::VirtualLODF, arc_idx::Int, delta_b::Float64) -> Vector{Float64}

Compute the LODF row for a partial susceptance change `delta_b` on arc `arc_idx`.

For a full outage, set `delta_b = -arc_susceptance`. For a single circuit outage
on a double-circuit arc with total susceptance `b_total`, set `delta_b = -b_circuit`.

# Arguments
- `vlodf::VirtualLODF`: VirtualLODF matrix
- `arc_idx::Int`: Arc index (integer)
- `delta_b::Float64`: Change in susceptance (negative for outage)

# Returns
- `Vector{Float64}`: LODF row of length n_arcs

$(TYPEDSIGNATURES)
"""
function get_partial_lodf_row(
    vlodf::VirtualLODF,
    arc_idx::Int,
    delta_b::Float64,
)
    return _getindex_partial(vlodf, arc_idx, delta_b)
end

"""
    get_partial_lodf_row(vlodf::VirtualLODF, arc::Tuple{Int, Int}, delta_b::Float64) -> Vector{Float64}

Arc-tuple indexed version of [`get_partial_lodf_row`](@ref).

$(TYPEDSIGNATURES)
"""
function get_partial_lodf_row(
    vlodf::VirtualLODF,
    arc::Tuple{Int, Int},
    delta_b::Float64,
)
    arc_idx = vlodf.lookup[1][arc]
    return _getindex_partial(vlodf, arc_idx, delta_b)
end
