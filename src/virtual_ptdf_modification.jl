"""
    compute_woodbury_factors(vptdf, mod) -> WoodburyFactors

Precompute Woodbury correction factors for a network modification.
The returned `WoodburyFactors` can be reused across multiple monitored
arcs, making this the recommended path for optimization loops where
factors are computed once per modification and many rows are queried.

!!! note
    Concurrent callers serialize on the per-cache `solver_lock`.

$(TYPEDSIGNATURES)
"""
function compute_woodbury_factors(
    vptdf::VirtualPTDF,
    mod::NetworkModification,
)::WoodburyFactors
    return _compute_woodbury_factors(vptdf, mod.arc_modifications)
end

"""
    apply_woodbury_correction(vptdf, monitored_arc, wf) -> Vector{Float64}

Compute the post-modification PTDF row for a monitored arc using
precomputed Woodbury factors. Accepts either an integer arc index
or a `Tuple{Int, Int}` bus pair.

!!! note
    Concurrent callers serialize on the per-cache `solver_lock`.

$(TYPEDSIGNATURES)
"""
function apply_woodbury_correction(
    vptdf::VirtualPTDF,
    monitored_arc::Union{Int, Tuple{Int, Int}},
    wf::WoodburyFactors,
)::Vector{Float64}
    return _apply_woodbury_correction(vptdf, _resolve_arc_index(vptdf, monitored_arc), wf)
end

"""
    get_post_modification_ptdf_row(vptdf, monitored_arc, mod) -> Vector{Float64}

One-shot convenience function: compute the post-modification PTDF row
for a monitored arc under a network modification: the row
`compute_woodbury_factors` followed by `apply_woodbury_correction` gives.

No caching — each call recomputes. Use the two-step API
(`compute_woodbury_factors` + `apply_woodbury_correction`) when querying
multiple monitored arcs for the same modification.

$(TYPEDSIGNATURES)
"""
function get_post_modification_ptdf_row(
    vptdf::VirtualPTDF,
    monitored_arc::Union{Int, Tuple{Int, Int}},
    mod::NetworkModification,
)::Vector{Float64}
    return only(get_post_modification_ptdf_rows(vptdf, monitored_arc, [mod]))
end

"""
    get_post_modification_ptdf_rows(vptdf::VirtualPTDF, monitored_arc, mods) -> Vector{Vector{Float64}}

The post-modification PTDF rows of one monitored arc (an index or a bus pair) under each
modification in `mods`, from a single solve of the arc's base column:
`rows[i] == get_post_modification_ptdf_row(vptdf, monitored_arc, mods[i])`, bit for bit,
including `vptdf`'s distributed slack. No caching: each call recomputes every modification's
Woodbury factors.

!!! note
    Concurrent callers serialize on the per-cache `solver_lock`.
"""
function get_post_modification_ptdf_rows(
    vptdf::VirtualPTDF,
    monitored_arc::Union{Int, Tuple{Int, Int}},
    mods::AbstractVector{NetworkModification},
)::Vector{Vector{Float64}}
    core = get_core(vptdf)
    factors = [compute_woodbury_factors(vptdf, mod) for mod in mods]
    m = _resolve_arc_index(vptdf, monitored_arc)
    # The base network needs no correction, so it plans no factors.
    plans = [
        _base_topology_row_plan(
            core,
            m,
            mod.arc_modifications,
            mod == _BASE_NETWORK ? nothing : wf,
        ) for (mod, wf) in zip(mods, factors)
    ]
    rows = _rows_from_plans(core, m, plans)
    for (row, wf) in zip(rows, factors)
        _apply_dist_slack!(row, vptdf, wf)
    end
    return rows
end

"""
    getindex(vptdf::VirtualPTDF, monitored::Int, mod::NetworkModification) -> Vector{Float64}

Indexing overload for post-modification PTDF row queries.
Dispatches to [`get_post_modification_ptdf_row`](@ref).

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vptdf::VirtualPTDF,
    monitored::Union{Int, Tuple{Int, Int}},
    mod::NetworkModification,
)
    return get_post_modification_ptdf_row(vptdf, monitored, mod)
end

# --- PSY.UnplannedOutage convenience API ---

"""
    get_post_modification_ptdf_row(vptdf, monitored_arc, sys, outage) -> Vector{Float64}

Compute the post-contingency PTDF row for a monitored arc when the given
`PSY.UnplannedOutage` trips. Resolves the outage's associated branches through
the system and builds the modification automatically.

$(TYPEDSIGNATURES)
"""
function get_post_modification_ptdf_row(
    vptdf::VirtualPTDF,
    monitored_arc::Union{Int, Tuple{Int, Int}},
    sys::PSY.System,
    outage::PSY.UnplannedOutage,
)::Vector{Float64}
    mod = NetworkModification(vptdf, sys, outage)
    return get_post_modification_ptdf_row(vptdf, monitored_arc, mod)
end
