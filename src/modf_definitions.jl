"""
    ArcModification

A susceptance change on a single aggregated arc.
Full outage: `delta_b = -b_arc`. Single circuit on double-circuit: `delta_b = -b_circuit`.

# Fields
- `arc_index::Int`: Index of the modified arc in the network matrix.
- `delta_b::Float64`: Change in susceptance (negative for an outage or reduction).
"""
struct ArcModification
    arc_index::Int
    delta_b::Float64
end

"""
Merge ArcModifications that target the same arc index.
"""
function _merge_modifications(mods::Vector{ArcModification})
    by_arc = Dict{Int, Float64}()
    for m in mods
        by_arc[m.arc_index] = get(by_arc, m.arc_index, 0.0) + m.delta_b
    end
    return [ArcModification(idx, db) for (idx, db) in sort!(collect(by_arc); by = first)]
end

"""
    NetworkModification

Lightweight description of topology changes to a power network.
Wraps a vector of [`ArcModification`](@ref) plus a label.
No dependency on `PSY.System` after construction.

# Fields
- `label::String`: Human-readable identifier for the modification.
- `modifications::Vector{ArcModification}`: One entry per affected arc.
"""
struct NetworkModification
    label::String
    modifications::Vector{ArcModification}
    function NetworkModification(label::String, mods::Vector{ArcModification})
        return new(label, _merge_modifications(mods))
    end
end

function Base.hash(m::NetworkModification, h::UInt)
    h = hash(length(m.modifications), h)
    for mod in m.modifications
        h = hash(mod.arc_index, h)
        h = hash(mod.delta_b, h)
    end
    return h
end

Base.:(==)(a::NetworkModification, b::NetworkModification) =
    a.modifications == b.modifications

"""
    ContingencySpec

A resolved, self-contained contingency specification backed by a
[`NetworkModification`](@ref). The UUID links back to the source
`PSY.Outage` supplemental attribute for caching purposes.

# Fields
- `uuid::Base.UUID`: Unique identifier matching the source Outage supplemental attribute.
- `modification::NetworkModification`: The network topology change.
"""
struct ContingencySpec
    uuid::Base.UUID
    modification::NetworkModification
end

"""
    WoodburyFactors

Cached Woodbury intermediates shared across monitored arcs for one contingency.
Computed from van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹

# Fields
- `Z::Matrix{Float64}`: B⁻¹U matrix (n_bus × M), one column per modified arc
- `W_inv::Matrix{Float64}`: Pre-inverted W = (A⁻¹ + U⊤B⁻¹U)⁻¹ (M × M). For M ≤ 2, computed analytically; for M > 2, computed via LU factorization.
- `arc_indices::Vector{Int}`: Arc indices of modified arcs
- `delta_b::Vector{Float64}`: Susceptance changes per modified arc
- `is_islanding::Bool`: Whether this contingency islands the network
"""
struct WoodburyFactors
    Z::Matrix{Float64}
    W_inv::Matrix{Float64}
    arc_indices::Vector{Int}
    delta_b::Vector{Float64}
    is_islanding::Bool
end
