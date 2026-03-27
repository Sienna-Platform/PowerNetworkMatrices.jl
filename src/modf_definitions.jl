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
    ContingencySpec

A resolved, self-contained contingency specification.
After construction, contains only integer indices and floats —
no PSY types, no System reference.

# Fields
- `uuid::Base.UUID`: Unique identifier matching the source Outage supplemental attribute.
- `name::String`: Human-readable identifier, typically branch name(s) joined by '+'.
- `modifications::Vector{ArcModification}`: One entry per affected arc.
"""
struct ContingencySpec
    uuid::Base.UUID
    name::String
    modifications::Vector{ArcModification}
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
