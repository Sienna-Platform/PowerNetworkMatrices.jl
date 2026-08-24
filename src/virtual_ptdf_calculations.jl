"""
The Virtual Power Transfer Distribution Factor (VirtualPTDF) structure gathers
the rows of the PTDF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualPTDF is initialized with no row stored.

The VirtualPTDF is indexed using branch names and bus numbers as for the PTDF
matrix.

Internally the factorization and topology live in a private
[`VirtualFactorCore`](@ref) held in the `core` field; the VirtualPTDF only adds
the PTDF-specific state (the row cache and the distributed-slack vectors). A
single core can be shared with a `VirtualMODF` / `VirtualLODF` so the
factorization is computed once.

# Thread-safety

Concurrent `getindex` is safe but serialized: every libklu solve is wrapped
by `_LIBKLU_LOCK` (process-wide) and the core's `solver_lock`, and the row
cache is guarded by `cache_lock`. Multiple threads can call `getindex`
simultaneously; their libklu work runs one at a time, while the JuMP-side work
(in callers) parallelizes freely.

# Fields
- `core::VirtualFactorCore`:
        Shared factorization/topology container (see [`VirtualFactorCore`](@ref)).
- `dist_slack::Vector{Float64}`:
        Vector of weights to be used as distributed slack bus.
- `dist_slack_normalized::Vector{Float64}`:
        Pre-normalized distributed slack weights.
- `cache::RowCache`:
        Cache where PTDF rows are stored.
- `cache_lock::ReentrantLock`:
        Guards `cache` reads/writes for parallel `getindex` callers.

The shared factorization, topology, subnetwork axes, network reduction data, and
the resolved sparsification cutoff all live in the [`VirtualFactorCore`](@ref).
"""
struct VirtualPTDF{Ax, L <: NTuple{2, Dict}, K} <:
       PowerNetworkMatrix{Float64}
    core::VirtualFactorCore{Ax, L, K}
    dist_slack::Vector{Float64}
    dist_slack_normalized::Vector{Float64}
    cache::RowCache
    cache_lock::ReentrantLock
end

# Forward any non-wrapper field read to the core, so the public field-access
# surface (`vptdf.BA`, `vptdf.K`, `vptdf.axes`, ...) keeps working unchanged.
function Base.getproperty(vptdf::VirtualPTDF, name::Symbol)
    if name === :core ||
       name === :dist_slack ||
       name === :dist_slack_normalized ||
       name === :cache ||
       name === :cache_lock
        return getfield(vptdf, name)
    end
    return getproperty(get_core(vptdf), name)
end

# Field getters. `getfield` is confined to these (and the `getproperty` hook
# above), so the rest of the file reads the wrapper's own fields through proper
# accessors instead of `getfield`.
get_core(M::VirtualPTDF) = getfield(M, :core)
get_cache(M::VirtualPTDF) = getfield(M, :cache)
get_cache_lock(M::VirtualPTDF) = getfield(M, :cache_lock)
get_dist_slack(M::VirtualPTDF) = getfield(M, :dist_slack)
get_dist_slack_normalized(M::VirtualPTDF) = getfield(M, :dist_slack_normalized)

# Accessors forward to the core.
get_axes(M::VirtualPTDF) = get_axes(get_core(M))
get_lookup(M::VirtualPTDF) = get_lookup(get_core(M))
get_ref_bus(M::VirtualPTDF) = get_ref_bus(get_core(M))
get_ref_bus_position(M::VirtualPTDF) = get_ref_bus_position(get_core(M))
get_network_reduction_data(M::VirtualPTDF) = get_network_reduction_data(get_core(M))
get_branch_catalog(M::VirtualPTDF) = get_branch_catalog(get_core(M))
get_bus_lookup(M::VirtualPTDF) = get_bus_lookup(get_core(M))
get_arc_lookup(M::VirtualPTDF) = get_arc_lookup(get_core(M))
get_system_uuid(M::VirtualPTDF) = get_system_uuid(get_core(M))
get_arc_axis(M::VirtualPTDF) = get_arc_axis(get_core(M))
get_bus_axis(M::VirtualPTDF) = get_bus_axis(get_core(M))
get_tol(M::VirtualPTDF) = get_tol(get_core(M))
get_cutoff(M::VirtualPTDF) = get_cutoff(get_core(M))
_get_BA(M::VirtualPTDF) = _get_BA(get_core(M))
_get_arc_susceptances(M::VirtualPTDF) = _get_arc_susceptances(get_core(M))
_get_valid_ix(M::VirtualPTDF) = _get_valid_ix(get_core(M))

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualPTDF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

"""
    redistribute_dist_slack(dist_slack, core::VirtualFactorCore) -> Vector{Float64}

Map a per-bus distributed-slack weight dictionary onto the core's bus axis,
accounting for network reductions.
"""
function redistribute_dist_slack(
    dist_slack::Dict{Int, Float64},
    core::VirtualFactorCore,
)
    dist_slack_vector = zeros(length(core.axes[2]))
    nr = get_network_reduction_data(core)
    bus_lookup = core.lookup[2]
    for (bus_no, dist_slack_factor) in dist_slack
        bus_no_ = get(nr.reverse_bus_search_map, bus_no, bus_no)
        if !haskey(bus_lookup, bus_no_)
            throw(
                IS.InvalidValue(
                    "Bus number $bus_no_ not found in the incidence matrix. Correct your slack distribution specification.",
                ),
            )
        end
        dist_slack_vector[bus_lookup[bus_no_]] += dist_slack_factor
    end
    return dist_slack_vector
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
- `linear_solver::String = _default_linear_solver()`:
        Linear solver to use for factorization. Options: "KLU", "AppleAccelerateLU".
- `tol::Float64 = eps()`:
        Tolerance related to sparsification and values to drop.
- `max_cache_size::Int`:
        max cache size in MiB (initialized as MAX_CACHE_SIZE_MiB).
- `persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}()`:
        arcs to be evaluated as soon as the VirtualPTDF is created.
- `network_reductions::Vector{NetworkReduction}`:
        Network reductions applied when computing the matrix.
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
    return VirtualPTDF(
        Ymatrix;
        dist_slack = dist_slack,
        linear_solver = linear_solver,
        tol = tol,
        max_cache_size = max_cache_size,
        persistent_arcs = persistent_arcs,
        system_uuid = PSY.get_system_uuid(sys),
    )
end

"""
Builds the Virtual PTDF matrix from a Ybus matrix. The return is a VirtualPTDF
struct with an empty cache.
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
    core = VirtualFactorCore(
        ybus;
        linear_solver = linear_solver,
        tol = tol,
        system_uuid = system_uuid,
    )
    return VirtualPTDF(
        core;
        dist_slack = dist_slack,
        max_cache_size = max_cache_size,
        persistent_arcs = persistent_arcs,
    )
end

"""
Builds a VirtualPTDF that wraps an existing [`VirtualFactorCore`](@ref). Use this
to share one factorization between a VirtualPTDF and a VirtualMODF/VirtualLODF.
"""
function VirtualPTDF(
    core::VirtualFactorCore;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
)
    bus_ax = core.axes[2]
    look_up = core.lookup
    if isempty(dist_slack)
        dist_slack_vector = Float64[]
    else
        dist_slack_vector = redistribute_dist_slack(dist_slack, core)
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

    if !isempty(dist_slack_vector)
        dist_slack_normalized = dist_slack_vector / sum(dist_slack_vector)
    else
        dist_slack_normalized = Float64[]
    end

    return VirtualPTDF(
        core,
        dist_slack_vector,
        dist_slack_normalized,
        empty_cache,
        ReentrantLock(),
    )
end

# Overload Base functions

"""
Checks if the VirtualPTDF holds any stored state (a populated cache or a
distributed-slack specification).
"""
function Base.isempty(vptdf::VirtualPTDF)
    isempty(get_dist_slack(vptdf)) || return false
    isempty(get_cache(vptdf)) || return false
    return true
end

"""
Gives the size of the whole PTDF matrix, not the number of rows stored.
"""
Base.size(vptdf::VirtualPTDF) = size(get_core(vptdf).BA)

"""
Gives the cartesian indexes of the PTDF matrix (same as the BA one).
"""
Base.eachindex(vptdf::VirtualPTDF) = CartesianIndices(size(vptdf))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualPTDF) = "VirtualPTDF"
end

function _compute_ptdf_row(vptdf::VirtualPTDF, row::Int)::Vector{Float64}
    core = get_core(vptdf)
    dist_slack = get_dist_slack(vptdf)
    dist_slack_normalized = get_dist_slack_normalized(vptdf)
    buscount = size(core.BA, 1)
    ref_bus_positions = get_ref_bus_position(core)
    if !isempty(dist_slack) && length(ref_bus_positions) != 1
        error(
            "Distributed slack is not supported for systems with multiple reference buses.",
        )
    end
    use_dist_slack = length(dist_slack) == buscount
    if !use_dist_slack && !isempty(dist_slack)
        error("Distributed bus specification doesn't match the number of buses.")
    end

    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        # Extract BA[:, row] non-zeros into work_ba_col at non-ref-bus
        # positions. Iterates only the nonzeros of the BA column (typically
        # 2 per arc) instead of scanning the full bus axis.
        fill!(work_ba_col, 0.0)
        BA = core.BA
        bus_to_valid_idx = core.bus_to_valid_idx
        ba_rv = SparseArrays.rowvals(BA)
        ba_nz = SparseArrays.nonzeros(BA)
        @inbounds for k in SparseArrays.nzrange(BA, row)
            valid_i = bus_to_valid_idx[ba_rv[k]]
            valid_i > 0 || continue
            work_ba_col[valid_i] = ba_nz[k]
        end
        lin_solve = _solve_factorization(K_solver, work_ba_col)
        fill!(temp_data, 0.0)
        valid_ix = core.valid_ix
        @inbounds for i in eachindex(valid_ix)
            temp_data[valid_ix[i]] = lin_solve[i]
        end
        if use_dist_slack
            adjustment = dot(temp_data, dist_slack_normalized)
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
        get_cache(vptdf), get_cache_lock(vptdf), row, column, get_cutoff(vptdf),
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

Returns a dictionary mapping row indices to lazily computed row vectors.
"""
get_ptdf_data(mat::VirtualPTDF) = get_cache(mat).temp_cache
