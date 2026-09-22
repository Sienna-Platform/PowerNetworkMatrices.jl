"""
The Virtual Line Outage Distribution Factor (VirtualLODF) structure gathers
the rows of the LODF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualLODF is initialized with no row stored.

The VirtualLODF struct is indexed using branch names.

Internally the factorization and topology live in a private
[`VirtualFactorCore`](@ref) held in the `core` field; the VirtualLODF only adds
the LODF-specific state (the inverse PTDF·A diagonal, the row cache, and its
arc×arc axes/lookup). A single core can be shared with a `VirtualPTDF` /
`VirtualMODF` so the factorization (and the `PTDF_A_diag`) is computed once.

# Thread-safety

Concurrent `getindex` (and `get_partial_lodf_row`) is safe but serialized:
every libklu solve runs under `_LIBKLU_LOCK` (process-wide) and the core's
`solver_lock`, and the row cache is guarded by `cache_lock`.

# Fields
- `core::VirtualFactorCore`:
        Shared factorization/topology container (see [`VirtualFactorCore`](@ref)).
- `inv_PTDF_A_diag::Vector{Float64}`:
        Element-wise reciprocal `1 / (1 - H[e,e])` (with `H[e,e]` clamped by
        `LODF_ENTRY_TOLERANCE`). The raw, unclamped `H[e,e]` lives on the core as
        `PTDF_A_diag`.
- `dist_slack::Vector{Float64}`:
        Distributed slack weights (retained for API symmetry).
- `axes<:NTuple{2, Vector}`:
        Tuple of two vectors of branch (arc) identifiers (row and column).
- `lookup<:NTuple{2, Dict}`:
        Tuple of two dictionaries mapping arc identifiers to row/column indices.
- `subnetwork_axes::Dict{Int, Ax}`:
        Per-reference-bus subnetwork axes in arc×arc form.
- `cache::RowCache{RowCacheValue}`:
        Cache where LODF rows are stored.
- `cache_lock::ReentrantLock`:
        Guards `cache` reads/writes for parallel `getindex` callers.

The shared factorization, network reduction data, and the resolved sparsification
cutoff all live in the [`VirtualFactorCore`](@ref).
"""
struct VirtualLODF{
    Ax <: NTuple{2, Vector},
    L <: NTuple{2, Dict},
    C <: VirtualFactorCore,
} <: PowerNetworkMatrix{Float64}
    core::C
    inv_PTDF_A_diag::Vector{Float64}
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    cache::RowCache{RowCacheValue}
    cache_lock::ReentrantLock
end

# Forward non-wrapper field reads to the core. `axes`/`lookup`/`subnetwork_axes`
# are wrapper fields (arc×arc form, distinct from the core's arc×bus form), so
# they resolve via `getfield`. `PTDF_A_diag`/`branch_susceptances_by_arc` map to
# the core's lazy getters.
function Base.getproperty(vlodf::VirtualLODF, name::Symbol)
    if name === :core ||
       name === :inv_PTDF_A_diag ||
       name === :dist_slack ||
       name === :axes ||
       name === :lookup ||
       name === :subnetwork_axes ||
       name === :cache ||
       name === :cache_lock
        return getfield(vlodf, name)
    end
    name === :PTDF_A_diag && return get_PTDF_A_diag(get_core(vlodf))
    name === :branch_susceptances_by_arc &&
        return get_branch_susceptances_by_arc(get_core(vlodf))
    return getproperty(get_core(vlodf), name)
end

# Field getters. `getfield` is confined to these (and the `getproperty` hook
# above), so the rest of the file reads the wrapper's own fields through proper
# accessors instead of `getfield`.
get_core(M::VirtualLODF) = getfield(M, :core)
get_inv_PTDF_A_diag(M::VirtualLODF) = getfield(M, :inv_PTDF_A_diag)
get_dist_slack(M::VirtualLODF) = getfield(M, :dist_slack)
get_subnetwork_axes(M::VirtualLODF) = getfield(M, :subnetwork_axes)
get_cache(M::VirtualLODF) = getfield(M, :cache)
get_cache_lock(M::VirtualLODF) = getfield(M, :cache_lock)

# `axes`/`lookup` are wrapper fields (arc×arc form, distinct from the core's
# arc×bus form), so these accessors read them directly.
get_axes(M::VirtualLODF) = getfield(M, :axes)
get_lookup(M::VirtualLODF) = getfield(M, :lookup)
get_ref_bus(M::VirtualLODF) = sort!(collect(keys(get_subnetwork_axes(M))))
# Arc-indexed wrapper: no get_bus_lookup(M::VirtualLODF) exists, so this throws MethodError
# on any call. Pre-existing; see the LODF note in lodf_calculations.jl.
get_ref_bus_position(M::VirtualLODF) =
    [get_bus_lookup(M)[x] for x in keys(get_subnetwork_axes(M))]
get_network_reduction_data(M::VirtualLODF) = get_network_reduction_data(get_core(M))
get_branch_catalog(M::VirtualLODF) = get_branch_catalog(get_core(M))
get_arc_lookup(M::VirtualLODF) = get_lookup(M)[1]
get_system_uuid(M::VirtualLODF) = get_system_uuid(get_core(M))
_get_BA(M::VirtualLODF) = _get_BA(get_core(M))
_get_arc_susceptances(M::VirtualLODF) = _get_arc_susceptances(get_core(M))
_get_valid_ix(M::VirtualLODF) = _get_valid_ix(get_core(M))

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualLODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

"""
Builds the Virtual LODF matrix from a system. The return is a VirtualLODF
struct with an empty cache.

# Arguments
- `sys::PSY.System`:
        PSY system for which the matrix is constructed

# Keyword Arguments
- `linear_solver::String = _default_linear_solver()`: Linear solver for the
        ABA factorization.
- `network_reductions::Vector{NetworkReduction}`:
        Network reductions applied when computing the matrix.
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
    resolve_linear_solver(linear_solver)
    Ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    core = VirtualFactorCore(
        Ymatrix;
        linear_solver = linear_solver,
        tol = tol,
        system_uuid = PSY.get_system_uuid(sys),
    )
    return VirtualLODF(
        core;
        dist_slack = dist_slack,
        max_cache_size = max_cache_size,
        persistent_arcs = persistent_arcs,
    )
end

# A bridge arc islands the network when it is outaged, so `1 - H[e,e]` collapses and the
# LODF scaling is undefined. Both LODF paths clamp such a diagonal to zero — a denominator
# of exactly 1 — instead of dividing by a vanishing number.
function _clamped_ptdf_a_diag(h_ee::Float64)
    if h_ee > 1 - LODF_ENTRY_TOLERANCE
        return 0.0
    end
    return h_ee
end

"""
Builds a VirtualLODF that wraps an existing [`VirtualFactorCore`](@ref). Forces
the core's `PTDF_A_diag` computation (needed for the LODF scaling) and derives
the clamped `inv_PTDF_A_diag`.
"""
function VirtualLODF(
    core::VirtualFactorCore;
    dist_slack::Vector{Float64} = Float64[],
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
)
    # Force the (shared, cached) raw H[e,e] on the core; clamp a copy for the
    # LODF scaling so the core's raw value stays available to MODF/PTDF.
    clamped = _clamped_ptdf_a_diag.(get_PTDF_A_diag(core))
    inv_PTDF_A_diag = 1.0 ./ (1.0 .- clamped)

    arc_ax = core.axes[1]
    axes = (arc_ax, arc_ax)
    arc_ax_ref = core.lookup[1]
    look_up = (arc_ax_ref, arc_ax_ref)

    # arc×arc subnetwork axes derived from the core's bus-form subnetwork axes.
    subnetwork_axes = Dict{Int, typeof(axes)}()
    for key in keys(core.subnetwork_axes)
        subnetwork_axes[key] = (core.subnetwork_axes[key][1], core.subnetwork_axes[key][1])
    end

    bus_ax = core.axes[2]
    empty_cache = RowCache(
        max_cache_size * MiB,
        Set{Int}(look_up[1][k] for k in persistent_arcs),
        length(bus_ax) * sizeof(Float64),
    )

    return VirtualLODF(
        core,
        inv_PTDF_A_diag,
        dist_slack,
        axes,
        look_up,
        subnetwork_axes,
        empty_cache,
        ReentrantLock(),
    )
end

"""
Builds a VirtualLODF that reuses an existing `VirtualPTDF`'s factorization,
sharing its [`VirtualFactorCore`](@ref).
"""
function VirtualLODF(
    vptdf::VirtualPTDF;
    dist_slack::Vector{Float64} = Float64[],
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
)
    return VirtualLODF(
        get_core(vptdf);
        dist_slack = dist_slack,
        max_cache_size = max_cache_size,
        persistent_arcs = persistent_arcs,
    )
end

# Overload Base functions

"""
Checks if the VirtualLODF holds any stored state.
"""
function Base.isempty(vlodf::VirtualLODF)
    isempty(get_inv_PTDF_A_diag(vlodf)) && return true
    isempty(get_dist_slack(vlodf)) && return true
    isempty(get_axes(vlodf)) && return true
    isempty(get_lookup(vlodf)) && return true
    isempty(get_subnetwork_axes(vlodf)) && return true
    isempty(get_cache(vlodf)) && return true
    return false
end

"""
Shows the size of the whole LODF matrix, not the number of rows stored.
"""
function Base.size(vlodf::VirtualLODF)
    n_arcs = size(get_core(vlodf).BA, 2)
    return (n_arcs, n_arcs)
end

"""
Gives the cartesian indexes of the LODF matrix.
"""
Base.eachindex(vlodf::VirtualLODF) = CartesianIndices(size(vlodf))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualLODF) = "VirtualLODF"
end

# Compute the LODF row for `row`. Pure computation: no cache reads/writes, no
# tolerance application.
function _compute_lodf_row(vlodf::VirtualLODF, row::Int)::Vector{Float64}
    core = get_core(vlodf)
    inv_PTDF_A_diag = get_inv_PTDF_A_diag(vlodf)
    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        lin_solve =
            _solve_ba_column!(K_solver, work_ba_col, core.BA, core.bus_to_valid_idx, row)

        fill!(temp_data, 0.0)
        @inbounds for i in eachindex(core.valid_ix)
            temp_data[core.valid_ix[i]] = lin_solve[i]
        end

        lodf_row = core.A * temp_data
        lodf_row .*= inv_PTDF_A_diag
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
        get_cache(vlodf), get_cache_lock(vlodf), row, column, get_cutoff(vlodf),
    ) do
        _compute_lodf_row(vlodf, row)
    end
end

"""
Gets the value of the element of the LODF matrix given the row and column indices
corresponding to the selected and outage branch respectively. If `column` is a Colon then
the entire row is returned.
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

Get the cached LODF row data from a [`VirtualLODF`](@ref) matrix, mapping row
indices to lazily computed row vectors.
"""
get_lodf_data(mat::VirtualLODF) = get_cache(mat).temp_cache

function get_arc_axis(mat::VirtualLODF)
    return get_axes(mat)[1]
end

""" Gets the tolerance used for sparsifying the rows of the VirtualLODF matrix.
Returns the absolute cutoff for a `Float64` `tol`, or the relative fraction for
an `AutoTolerance`."""
function get_tol(mat::VirtualLODF)
    return get_tol(get_core(mat))
end

get_cutoff(mat::VirtualLODF) = get_cutoff(get_core(mat))

"""
    _getindex_partial(vlodf, arc_idx, delta_b) -> Vector{Float64}

Compute the partial LODF column for a susceptance change `delta_b` on arc `arc_idx`.

Concurrent callers serialize on the core `solver_lock` and `_LIBKLU_LOCK`.

Uses the Sherman-Morrison (matrix inversion lemma) formula:

    partial_LODF[ℓ, e] = α · (b_ℓ / b_e) · H[ℓ,e] / (1 - α · H[e,e])

where α = -Δb / b_e and H[e,e] is `PTDF_A_diag[e]` clamped by
`_clamped_ptdf_a_diag`, the same clamp `inv_PTDF_A_diag` carries. When
`delta_b = -b_e` (full outage) this reduces to the standard LODF column; the
self-element is overridden to -1.0 for a full outage.
"""
function _getindex_partial(
    vlodf::VirtualLODF,
    arc_idx::Int,
    delta_b::Float64,
)::Vector{Float64}
    core = get_core(vlodf)
    n_arcs = size(core.BA, 2)

    # Zero change means zero redistribution.
    if abs(delta_b) < eps()
        return zeros(n_arcs)
    end

    b_arc = core.arc_susceptances[arc_idx]
    if b_arc == 0.0
        return zeros(n_arcs)
    end

    # Raw (unclamped) H[e,e]; cached on the core.
    ptdf_a_diag = get_PTDF_A_diag(core)

    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, temp_data
        # Steps 1-2: Compute B⁻¹(b_e · ν_e).
        lin_solve = _solve_ba_column!(
            K_solver, work_ba_col, core.BA, core.bus_to_valid_idx, arc_idx,
        )

        # Step 3: Map solution back to full bus space.
        fill!(temp_data, 0.0)
        @inbounds for i in eachindex(core.valid_ix)
            temp_data[core.valid_ix[i]] = lin_solve[i]
        end

        # Step 4: H_col[ℓ] = b_e · C[e,ℓ] for all monitoring arcs ℓ.
        H_col = core.A * temp_data

        # Step 5: Scalar denominator: 1 - α · H[e,e].
        H_ee = _clamped_ptdf_a_diag(ptdf_a_diag[arc_idx])
        alpha = -delta_b / b_arc
        denom = 1.0 - alpha * H_ee

        # Step 6: Partial LODF column scaled by b_ℓ/b_e, in place on the fresh `H_col`. The
        # operand order is load-bearing: float multiply does not reassociate, and `s * (a * h)`
        # is what every stored reference row was produced with.
        partial_lodf = H_col
        partial_lodf .= (alpha / (denom * b_arc)) .* (core.arc_susceptances .* partial_lodf)

        # Full-outage self-element convention: -1.0.
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
on a double-circuit arc, set `delta_b = -b_circuit`.

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
    arc_idx = get_lookup(vlodf)[1][arc]
    return _getindex_partial(vlodf, arc_idx, delta_b)
end
