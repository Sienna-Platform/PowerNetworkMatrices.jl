"""
Public API for computing post-modification PTDF rows from a `VirtualPTDF`
and a `NetworkModification`, using the Woodbury matrix identity.
"""

"""
    compute_woodbury_factors(vptdf, mod) -> WoodburyFactors

Precompute Woodbury correction factors for a network modification.
The returned `WoodburyFactors` can be reused across multiple monitored
arcs, making this the recommended path for optimization loops where
factors are computed once per modification and many rows are queried.

!!! warning
    Not thread-safe. Mutates scratch vectors in `vptdf`. Do not call
    concurrently on the same `VirtualPTDF` instance.

$(TYPEDSIGNATURES)
"""
function compute_woodbury_factors(
    vptdf::VirtualPTDF,
    mod::NetworkModification,
)::WoodburyFactors
    return _compute_woodbury_factors(vptdf, mod.modifications)
end

"""
    apply_woodbury_correction(vptdf, monitored_arc, wf) -> Vector{Float64}

Compute the post-modification PTDF row for a monitored arc using
precomputed Woodbury factors. Accepts either an integer arc index
or a `Tuple{Int, Int}` bus pair.

!!! warning
    Not thread-safe. Mutates scratch vectors in `vptdf`. Do not call
    concurrently on the same `VirtualPTDF` instance.

$(TYPEDSIGNATURES)
"""
function apply_woodbury_correction(
    vptdf::VirtualPTDF,
    monitored_arc::Int,
    wf::WoodburyFactors,
)::Vector{Float64}
    return _apply_woodbury_correction(vptdf, monitored_arc, wf)
end

function apply_woodbury_correction(
    vptdf::VirtualPTDF,
    monitored_arc::Tuple{Int, Int},
    wf::WoodburyFactors,
)::Vector{Float64}
    m_idx = get_arc_lookup(vptdf)[monitored_arc]
    return _apply_woodbury_correction(vptdf, m_idx, wf)
end

"""
    get_post_modification_ptdf_row(vptdf, monitored_arc, mod) -> Vector{Float64}

One-shot convenience function: compute the post-modification PTDF row
for a monitored arc under a network modification. Internally calls
`compute_woodbury_factors` then `apply_woodbury_correction`.

No caching — each call recomputes. Use the two-step API
(`compute_woodbury_factors` + `apply_woodbury_correction`) when querying
multiple monitored arcs for the same modification.

$(TYPEDSIGNATURES)
"""
function get_post_modification_ptdf_row(
    vptdf::VirtualPTDF,
    monitored_arc::Int,
    mod::NetworkModification,
)::Vector{Float64}
    wf = compute_woodbury_factors(vptdf, mod)
    return apply_woodbury_correction(vptdf, monitored_arc, wf)
end

function get_post_modification_ptdf_row(
    vptdf::VirtualPTDF,
    monitored_arc::Tuple{Int, Int},
    mod::NetworkModification,
)::Vector{Float64}
    wf = compute_woodbury_factors(vptdf, mod)
    return apply_woodbury_correction(vptdf, monitored_arc, wf)
end

"""
    getindex(vptdf::VirtualPTDF, monitored::Int, mod::NetworkModification) -> Vector{Float64}

Indexing overload for post-modification PTDF row queries.
Dispatches to [`get_post_modification_ptdf_row`](@ref).

$(TYPEDSIGNATURES)
"""
function Base.getindex(
    vptdf::VirtualPTDF,
    monitored::Int,
    mod::NetworkModification,
)
    return get_post_modification_ptdf_row(vptdf, monitored, mod)
end

function Base.getindex(
    vptdf::VirtualPTDF,
    monitored::Tuple{Int, Int},
    mod::NetworkModification,
)
    return get_post_modification_ptdf_row(vptdf, monitored, mod)
end
