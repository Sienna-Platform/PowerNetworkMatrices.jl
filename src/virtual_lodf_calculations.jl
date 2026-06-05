"""
The Virtual Line Outage Distribution Factor (VirtualLODF) structure gathers
the rows of the LODF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualLODF is initialized with no row stored.

The VirtualLODF struct is indexed using branch names.

# Thread-safety

Concurrent `getindex` (and `get_partial_lodf_row`) is safe but serialized:
every libklu solve runs under `_LIBKLU_LOCK` (process-wide) and the per-cache
`solver_lock`, and the row cache is guarded by `cache_lock`. Multi-threaded
callers can issue requests concurrently; the libklu work runs one at a time.

# Arguments
- `K`:
        ABA factorization cache (`KLULinSolveCache{Float64}` or `AAFactorization`).
        Solves serialize via the locks above.
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
- `bus_to_valid_idx::Vector{Int}`:
        Inverse of `valid_ix`: `bus_to_valid_idx[b]` is the position of bus
        `b` inside `valid_ix`, or 0 if `b` is a reference bus. Lets the hot
        path iterate the nonzeros of a `BA` column directly.
- `temp_data::Vector{Vector{Float64}}`:
        Single-element scratch vector kept as a `Vector{Vector{Float64}}` for
        uniform `with_solver` callback signatures.
- `cache::RowCache`:
        Cache where LODF rows are stored.
- `cache_lock::ReentrantLock`:
        Guards `cache` reads/writes for parallel `getindex` callers.
- `subnetworks::Dict{Int, Set{Int}}`:
        Dictionary containing the subsets of buses defining the different subnetwork of the system.
- `tol::SparsificationCutoff`:
        Resolved per-row sparsification rule (absolute cutoff for a `Float64`
        `tol`, relative cutoff for an [`AutoTolerance`](@ref)).
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `work_ba_col::Vector{Vector{Float64}}`:
        Single-element BA-column scratch buffer.
"""
struct VirtualLODF{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    K::K
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
    bus_to_valid_idx::Vector{Int}
    temp_data::Vector{Vector{Float64}}
    cache::RowCache
    cache_lock::ReentrantLock
    subnetwork_axes::Dict{Int, Ax}
    tol::SparsificationCutoff
    network_reduction_data::NetworkReductionData
    work_ba_col::Vector{Vector{Float64}}
    # Serializes solves on this cache. `with_solver` always acquires it,
    # for both the KLU and AppleAccelerate backends.
    solver_lock::ReentrantLock
end

get_axes(M::VirtualLODF) = M.axes
get_lookup(M::VirtualLODF) = M.lookup
get_ref_bus(M::VirtualLODF) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::VirtualLODF) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::VirtualLODF) = M.network_reduction_data
get_arc_lookup(M::VirtualLODF) = M.lookup[1]

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualLODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

"""
    _get_PTDF_A_diag(K, BA, A, ref_bus_positions) -> Vector{Float64}

Compute `diag(PTDF · A)`. Each row of `A` has exactly two nonzeros (+1 at the
from-bus, -1 at the to-bus), so the per-arc dot product reduces to two indexed
reads into the solved PTDF row after a one-time transpose of `A`.
"""
function _get_PTDF_A_diag(
    K,
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    ref_bus_positions::Set{Int},
)
    n_branches = size(BA, 2)
    n_buses = size(BA, 1)
    diag_ = zeros(n_branches)

    valid_ix = setdiff(1:n_buses, ref_bus_positions)
    n_valid = length(valid_ix)
    bus_to_valid_idx = _build_bus_to_valid_idx(n_buses, valid_ix)

    # Per-arc (from_valid, to_valid) via one transpose of A; 0 = ref bus.
    A_T = SparseArrays.sparse(transpose(A))
    arc_from_valid = Vector{Int}(undef, n_branches)
    arc_to_valid = Vector{Int}(undef, n_branches)
    at_rv = SparseArrays.rowvals(A_T)
    at_nz = SparseArrays.nonzeros(A_T)
    for i in 1:n_branches
        f_valid = 0
        t_valid = 0
        @inbounds for k in SparseArrays.nzrange(A_T, i)
            bus_ix = at_rv[k]
            v = at_nz[k]
            valid_i = bus_to_valid_idx[bus_ix]
            if v > 0
                f_valid = valid_i
            elseif v < 0
                t_valid = valid_i
            end
        end
        arc_from_valid[i] = f_valid
        arc_to_valid[i] = t_valid
    end

    ba_col = zeros(n_valid)
    ba_rv = SparseArrays.rowvals(BA)
    ba_nz = SparseArrays.nonzeros(BA)

    for i in 1:n_branches
        fill!(ba_col, 0.0)
        @inbounds for k in SparseArrays.nzrange(BA, i)
            valid_i = bus_to_valid_idx[ba_rv[k]]
            valid_i > 0 || continue
            ba_col[valid_i] = ba_nz[k]
        end

        # Read PTDF row from the returned buffer — backend-agnostic
        # (KLU mutates `ba_col` and returns it; other backends may
        # return a fresh vector, so capture the return value).
        lin_solve = _solve_factorization(K, ba_col)

        # H[e,e] = ptdf[from] - ptdf[to]; ref-bus entries are 0.
        f = arc_from_valid[i]
        t = arc_to_valid[i]
        v_f = if f > 0
            lin_solve[f]
        else
            0.0
        end
        v_t = if t > 0
            lin_solve[t]
        else
            0.0
        end
        @inbounds diag_[i] = v_f - v_t
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
- `linear_solver::String = _default_linear_solver()`: Linear solver for the
        ABA factorization. Options: "KLU", "AppleAccelerate". Defaults to
        "AppleAccelerate" on macOS and "KLU" elsewhere.
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `kwargs...`:
        other keyword arguments used by VirtualPTDF
"""
function VirtualLODF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    if length(dist_slack) != 0
        @info "Distributed bus"
    end
    solver = resolve_linear_solver(linear_solver)
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
    K = _create_factorization(solver, ABA)
    bus_ax = get_bus_axis(A)

    # Resolve the per-row sparsification rule once: a Float64 becomes an absolute
    # cutoff, an AutoTolerance a relative cutoff (κ logged via the reused K).
    cutoff = _resolve_virtual_cutoff(tol, K, ABA, SparseArrays.nonzeros(BA.data))

    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)
    # PTDF diagonal precompute runs serially on the dispatcher thread.
    PTDF_diag = _get_PTDF_A_diag(K, BA.data, A.data, Set(ref_bus_positions))
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

    # Single scratch slot — solves serialize via `solver_lock` + `_LIBKLU_LOCK`.
    temp_data = [zeros(length(bus_ax))]
    work_ba_col = [zeros(length(valid_ix))]
    bus_to_valid_idx = _build_bus_to_valid_idx(length(bus_ax), valid_ix)

    return VirtualLODF(
        K,
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
        bus_to_valid_idx,
        temp_data,
        empty_cache,
        ReentrantLock(),
        subnetwork_axes,
        cutoff,
        Ymatrix.network_reduction_data,
        work_ba_col,
        ReentrantLock(),
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
    return with_solver(
        vlodf.K, vlodf.work_ba_col, vlodf.temp_data, vlodf.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        # Sparse-only extraction: iterate BA[:, row] non-zeros (typically
        # 2 per arc) instead of scanning the full bus axis.
        fill!(work_ba_col, 0.0)
        BA = vlodf.BA
        bus_to_valid_idx = vlodf.bus_to_valid_idx
        ba_rv = SparseArrays.rowvals(BA)
        ba_nz = SparseArrays.nonzeros(BA)
        @inbounds for k in SparseArrays.nzrange(BA, row)
            valid_i = bus_to_valid_idx[ba_rv[k]]
            valid_i > 0 || continue
            work_ba_col[valid_i] = ba_nz[k]
        end
        lin_solve = _solve_factorization(K_solver, work_ba_col)

        fill!(temp_data, 0.0)
        @inbounds for i in eachindex(vlodf.valid_ix)
            temp_data[vlodf.valid_ix[i]] = lin_solve[i]
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
        vlodf.cache, vlodf.cache_lock, row, column, vlodf.tol,
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

""" Gets the tolerance used for sparsifying the rows of the VirtualLODF matrix.
Returns the absolute cutoff for a `Float64` `tol`, or the relative fraction for
an `AutoTolerance`."""
function get_tol(mat::VirtualLODF)
    return cutoff_value(mat.tol)
end

"""
    _getindex_partial(vlodf, arc_idx, delta_b) -> Vector{Float64}

Compute the partial LODF column for a susceptance change `delta_b` on arc `arc_idx`.

Concurrent callers serialize on `vlodf.solver_lock` and `_LIBKLU_LOCK`.

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

    return with_solver(
        vlodf.K, vlodf.work_ba_col, vlodf.temp_data, vlodf.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        # Steps 1-2: Compute B⁻¹(b_e · ν_e) via sparse-only BA-column
        # extraction + solve.
        fill!(work_ba_col, 0.0)
        BA = vlodf.BA
        bus_to_valid_idx = vlodf.bus_to_valid_idx
        ba_rv = SparseArrays.rowvals(BA)
        ba_nz = SparseArrays.nonzeros(BA)
        @inbounds for k in SparseArrays.nzrange(BA, arc_idx)
            valid_i = bus_to_valid_idx[ba_rv[k]]
            valid_i > 0 || continue
            work_ba_col[valid_i] = ba_nz[k]
        end
        lin_solve = _solve_factorization(K_solver, work_ba_col)

        # Step 3: Map solution back to full bus space.
        fill!(temp_data, 0.0)
        @inbounds for i in eachindex(vlodf.valid_ix)
            temp_data[vlodf.valid_ix[i]] = lin_solve[i]
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
