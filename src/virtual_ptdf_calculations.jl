"""
The Virtual Power Transfer Distribution Factor (VirtualPTDF) structure gathers
the rows of the PTDF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualPTDF is initialized with no row stored.

The VirtualPTDF is indexed using branch names and bus numbers as for the PTDF
matrix.

# Thread-safety

Concurrent `getindex` is safe but serialized: every libklu solve is wrapped
by `_LIBKLU_LOCK` (process-wide) and the per-cache `solver_lock` here, and
the row cache is guarded by `cache_lock`. Multiple threads can call
`getindex` simultaneously; their libklu work runs one at a time, while the
JuMP-side work (in callers) parallelizes freely.

# Arguments
- `K`:
        LU factorization of the ABA matrix. A `KLULinSolveCache{Float64}` for
        the default KLU solver, or an `AccelerateWrapper.AAFactorCache` when the
        AppleAccelerate backend is selected on macOS.
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix
- `ref_bus_positions::Set{Int}`:
        Vector containing the indexes of the columns of the BA matrix corresponding
        to the reference buses
- `dist_slack::Vector{Float64}`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `axes<:NTuple{2, Dict}`:
        Tuple containing two vectors: the first one showing the branches names,
        the second showing the buses numbers. There is no link between the
        order of the vector of the branches names and the way the PTDF rows are
        stored in the cache.
- `lookup<:NTuple{2, Dict}`:
        Tuple containing two dictionaries, mapping the branches
        and buses with their enumerated indexes. The branch indexes refer to
        the key of the cache dictionary. The bus indexes refer to the position
        of the elements in the PTDF row stored.
- `temp_data::Vector{Vector{Float64}}`:
        Single-element vector holding a temporary buffer for internal use.
        Kept as `Vector{Vector{Float64}}` so the dispatch on
        `_solve_factorization` stays uniform across backends.
- `valid_ix::Vector{Int}`:
        Vector containing the row/columns indices of matrices related the buses
        which are not slack ones.
- `bus_to_valid_idx::Vector{Int}`:
        Inverse of `valid_ix`: `bus_to_valid_idx[b]` is the position of bus
        `b` inside `valid_ix`, or 0 if `b` is a reference bus. Lets the hot
        path iterate the nonzeros of a `BA` column instead of scanning the
        full bus axis.
- `cache::RowCache`:
        Cache where PTDF rows are stored.
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
- `solver_lock::ReentrantLock`:
        Serializes solves on this cache. Combined with `_LIBKLU_LOCK` at the
        libklu boundary, ensures one solve at a time per cache.
- `system_uuid::Union{Base.UUID, Nothing}`:
        UUID of the system used to construct the matrix, used to validate that
        modification operations are applied to the correct system. `nothing` when
        constructed from a Ybus without an associated system.
"""
struct VirtualPTDF{Ax, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    K::K
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    arc_susceptances::Vector{Float64}
    dist_slack::Vector{Float64}
    dist_slack_normalized::Vector{Float64}
    axes::Ax
    lookup::L
    temp_data::Vector{Vector{Float64}}
    valid_ix::Vector{Int}
    bus_to_valid_idx::Vector{Int}
    cache::RowCache
    cache_lock::ReentrantLock
    subnetwork_axes::Dict{Int, Ax}
    tol::SparsificationCutoff
    network_reduction_data::NetworkReductionData
    work_ba_col::Vector{Vector{Float64}}
    solver_lock::ReentrantLock
    system_uuid::Union{Base.UUID, Nothing}
end

get_axes(M::VirtualPTDF) = M.axes
get_lookup(M::VirtualPTDF) = M.lookup
get_ref_bus(M::VirtualPTDF) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::VirtualPTDF) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::VirtualPTDF) = M.network_reduction_data
get_bus_lookup(M::VirtualPTDF) = M.lookup[2]
get_arc_lookup(M::VirtualPTDF) = M.lookup[1]
get_system_uuid(M::VirtualPTDF) = M.system_uuid

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualPTDF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

"""
Builds the Virtual PTDF matrix from a system. The return is a VirtualPTDF
struct with an empty cache.

# Arguments
- `sys::PSY.System`:
        PSY system for which the matrix is constructed

# Keyword Arguments
- `dist_slack::Dict{Int, Float64} = Dict{Int, Float64}()`:
        Dictionary of weights to be used as distributed slack bus.
        The distributed slack dictionary must have the same number of entries as the number of buses.
- `linear_solver::String = _default_linear_solver()`:
        Linear solver to use for factorization. Options: "KLU", "AppleAccelerateLU".
        Defaults to "AppleAccelerateLU" on macOS 15.5+ and "KLU" elsewhere.
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Tolerance related to sparsification and values to drop. A `Float64` applies a
        fixed absolute cutoff; an [`AutoTolerance`](@ref) (the default) applies a
        relative per-row cutoff so requested rows stay sparse on large systems.
- `max_cache_size::Int`:
        max cache size in MiB (initialized as `MAX_CACHE_SIZE_MiB`).
- `persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}()`:
        arcs to be evaluated as soon as the VirtualPTDF is created (initialized as empty vector of tuples).
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `kwargs...`:
        other keyword arguments used by VirtualPTDF
"""
function VirtualPTDF(
    sys::PSY.System;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    resolve_linear_solver(linear_solver)
    Ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    VirtualPTDF(
        Ymatrix;
        dist_slack = dist_slack,
        linear_solver = linear_solver,
        tol = tol,
        max_cache_size = max_cache_size,
        persistent_arcs = persistent_arcs,
        system_uuid = IS.get_uuid(sys),
    )
end

# Factorization dispatch methods for VirtualPTDF solver selection.
function _create_factorization(
    ::KLUSolver,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    return klu_factorize(ABA)
end

function _create_factorization(
    ::AppleAccelerateLUSolver,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    _has_apple_accelerate_backend() || error(_apple_accelerate_unavailable_error())
    return AccelerateWrapper.aa_factorize(ABA)
end

function _create_factorization(
    ::LinearSolverType,
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    return error(
        "Only KLU and AppleAccelerateLU solvers are supported for VirtualPTDF factorization.",
    )
end

"""
Builds the Virtual PTDF matrix from a Ybus matrix. This constructor is more efficient when the prerequisite Ybus
matrix is already available and provides direct control over the underlying matrix computations (including network reductions).
The return is a VirtualPTDF struct with an empty cache.

# Arguments
- `ybus::Ybus`: Ybus matrix from which the matrix is constructed

# Keyword Arguments
- `dist_slack::Dict{Int, Float64} = Dict{Int, Float64}()`:
        Dictionary of weights to be used as distributed slack bus.
        The distributed slack dictionary must have the same number of entries as the number of buses.
- `linear_solver::String = _default_linear_solver()`:
        Linear solver to use for factorization. Options: "KLU", "AppleAccelerateLU".
        Defaults to "AppleAccelerateLU" on macOS 15.5+ and "KLU" elsewhere.
- `tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE`:
        Tolerance related to sparsification and values to drop. A `Float64` applies a
        fixed absolute cutoff; an [`AutoTolerance`](@ref) (the default) applies a
        relative per-row cutoff so requested rows stay sparse on large systems.
- `max_cache_size::Int`:
        max cache size in MiB (initialized as `MAX_CACHE_SIZE_MiB`).
- `persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}()`:
        arcs to be evaluated as soon as the VirtualPTDF is created (initialized as empty vector of tuples).
"""
function VirtualPTDF(
    ybus::Ybus;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    system_uuid::Union{Base.UUID, Nothing} = nothing,
)
    solver = resolve_linear_solver(linear_solver)
    ref_bus_positions = get_ref_bus_position(ybus)
    A = IncidenceMatrix(ybus)
    if !(isempty(dist_slack))
        dist_slack_vector = redistribute_dist_slack(dist_slack, A, A.network_reduction_data)
    else
        dist_slack_vector = Float64[]
    end
    BA = BA_Matrix(ybus)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    bus_ax = get_bus_axis(A)
    axes = A.axes
    look_up = A.lookup
    subnetwork_axes = A.subnetwork_axes
    if length(subnetwork_axes) > 1
        @info "Network is not connected, using subnetworks"
    end

    if isempty(persistent_arcs)
        empty_cache =
            RowCache(max_cache_size * MiB, Set{Int}(), length(bus_ax) * sizeof(Float64))
    else
        init_persistent_dict = Set{Int}(look_up[1][k] for k in persistent_arcs)
        empty_cache =
            RowCache(
                max_cache_size * MiB,
                init_persistent_dict,
                length(bus_ax) * sizeof(Float64),
            )
    end

    # Create factorization based on solver type dispatch.
    K = _create_factorization(solver, ABA)

    # Resolve the per-row sparsification rule once: a Float64 becomes an absolute
    # cutoff, an AutoTolerance a relative cutoff (κ logged via the reused K).
    cutoff = _resolve_virtual_cutoff(tol, K, ABA, SparseArrays.nonzeros(BA.data))

    # Pre-compute normalized distributed slack for efficiency
    if !isempty(dist_slack_vector)
        dist_slack_normalized = dist_slack_vector / sum(dist_slack_vector)
    else
        dist_slack_normalized = Float64[]
    end

    # Single scratch slot — solves serialize through `solver_lock` +
    # `_LIBKLU_LOCK`, so per-worker scratch is unnecessary. Kept as a
    # `Vector{Vector{Float64}}` so `with_solver`'s callback signature
    # stays uniform across solver backends.
    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)
    bus_to_valid_idx = _build_bus_to_valid_idx(length(bus_ax), valid_ix)
    temp_data = [zeros(length(bus_ax))]
    work_ba_col = [zeros(length(valid_ix))]

    arc_susceptances = _extract_arc_susceptances(BA.data)

    return VirtualPTDF(
        K,
        BA.data,
        A.data,
        arc_susceptances,
        dist_slack_vector,
        dist_slack_normalized,
        axes,
        look_up,
        temp_data,
        valid_ix,
        bus_to_valid_idx,
        empty_cache,
        ReentrantLock(),
        subnetwork_axes,
        cutoff,
        ybus.network_reduction_data,
        work_ba_col,
        ReentrantLock(),
        system_uuid,
    )
end

# Overload Base functions

"""
Checks if the any of the fields of VirtualPTDF is empty.
"""
function Base.isempty(vptdf::VirtualPTDF)
    for name in fieldnames(typeof(vptdf))
        if name == :dist_slack && !isempty(getfield(vptdf, name))
            @debug "Field dist_slack has default value: " *
                   string(getfield(vptdf, name)) * "."
            return false
        elseif (name in [:cache, :radial_branches]) && !isempty(getfield(vptdf, name))
            @debug "Field " * string(name) * " not defined."
            return false
        end
    end
    return true
end

"""
Gives the size of the whole PTDF matrix, not the number of rows stored.
"""
Base.size(vptdf::VirtualPTDF) = size(vptdf.BA)

"""
Gives the cartesian indexes of the PTDF matrix (same as the BA one).
"""
Base.eachindex(vptdf::VirtualPTDF) = CartesianIndices(size(vptdf.BA))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualPTDF) = "VirtualPTDF"
end

# Helper function to solve with different factorization types. Both
# overloads solve in place (zero-allocation hot path). The KLU and Apple
# Accelerate backends are the only solvers supported here; adding a new
# backend requires extending this method.
function _solve_factorization(K::KLULinSolveCache{Float64}, b::Vector{Float64})
    solve!(K, b)
    return b
end

function _solve_factorization(K::AAFactorCache, b::Vector{Float64})
    AccelerateWrapper.solve!(K, b)
    return b
end

function _compute_ptdf_row(vptdf::VirtualPTDF, row::Int)::Vector{Float64}
    buscount = size(vptdf, 1)
    ref_bus_positions = get_ref_bus_position(vptdf)
    if !isempty(vptdf.dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    end
    use_dist_slack = length(vptdf.dist_slack) == buscount
    if !use_dist_slack && !isempty(vptdf.dist_slack)
        error("Distributed bus specification doesn't match the number of buses.")
    end

    return with_solver(
        vptdf.K, vptdf.work_ba_col, vptdf.temp_data, vptdf.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        # Extract BA[:, row] non-zeros into work_ba_col at non-ref-bus
        # positions. Iterates only the nonzeros of the BA column (typically
        # 2 per arc) instead of scanning the full bus axis and bisecting
        # the CSC for each entry.
        fill!(work_ba_col, 0.0)
        BA = vptdf.BA
        bus_to_valid_idx = vptdf.bus_to_valid_idx
        ba_rv = SparseArrays.rowvals(BA)
        ba_nz = SparseArrays.nonzeros(BA)
        @inbounds for k in SparseArrays.nzrange(BA, row)
            valid_i = bus_to_valid_idx[ba_rv[k]]
            valid_i > 0 || continue
            work_ba_col[valid_i] = ba_nz[k]
        end
        lin_solve = _solve_factorization(K_solver, work_ba_col)
        fill!(temp_data, 0.0)
        valid_ix = vptdf.valid_ix
        @inbounds for i in eachindex(valid_ix)
            temp_data[valid_ix[i]] = lin_solve[i]
        end
        if use_dist_slack
            adjustment = dot(temp_data, vptdf.dist_slack_normalized)
            return temp_data .- adjustment
        end
        return copy(temp_data)
    end
end

function _getindex(
    vptdf::VirtualPTDF,
    row::Int,
    column::Union{Int, Colon},
)
    return cached_row_lookup(
        vptdf.cache, vptdf.cache_lock, row, column, vptdf.tol,
    ) do
        _compute_ptdf_row(vptdf, row)
    end
end

function Base.getindex(vptdf::VirtualPTDF, branch_name::String, bus)
    multiplier, arc = get_branch_multiplier(vptdf, branch_name)
    row_, column_ = to_index(vptdf, arc, bus)
    return _getindex(vptdf, row_, column_) * multiplier
end

"""
Gets the value of the element of the PTDF matrix given the row and column indices
corresponding to the branch and buses one respectively. If `column` is a Colon then
the entire row is returned.

# Arguments
- `vptdf::VirtualPTDF`:
        VirtualPTDF struct where to evaluate and store the row values.
- `row`:
        Branch index.
- `column`:
        Bus index. If Colon then get the values of the whole row.
"""
function Base.getindex(vptdf::VirtualPTDF, row, column)
    row_, column_ = to_index(vptdf, row, column)
    return _getindex(vptdf, row_, column_)
end

# Define for ambiguity resolution
function Base.getindex(vptdf::VirtualPTDF, row::Integer, column::Integer)
    return _getindex(vptdf, row, column)
end

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualPTDF, _, idx...) = error("Operation not supported by VirtualPTDF")

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualPTDF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualPTDF")

"""
    get_ptdf_data(mat::VirtualPTDF) -> Dict{Int, Vector{Float64}}

Get the cached PTDF row data from a [`VirtualPTDF`](@ref) matrix.

Unlike [`get_ptdf_data(::PTDF)`](@ref), which returns a dense matrix, this returns
a dictionary mapping row indices to lazily computed row vectors.

# Arguments
- `mat::VirtualPTDF`: The virtual PTDF matrix

# Returns
- `Dict{Int, Vector{Float64}}`: Cached row data keyed by row index
"""
get_ptdf_data(mat::VirtualPTDF) = mat.cache.temp_cache

function get_arc_axis(ptdf::VirtualPTDF)
    return ptdf.axes[1]
end

function get_bus_axis(ptdf::VirtualPTDF)
    return ptdf.axes[2]
end

""" Gets the tolerance used for sparsifying the rows of the VirtualPTDF matrix.
Returns the absolute cutoff for a `Float64` `tol`, or the relative fraction for
an `AutoTolerance`."""
function get_tol(vptdf::VirtualPTDF)
    return cutoff_value(vptdf.tol)
end
