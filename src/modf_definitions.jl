"""
A susceptance change on a single aggregated arc.
Full outage: `delta_b = -b_arc`. Single circuit on double-circuit: `delta_b = -b_circuit`.
"""
struct BranchModification
    arc_index::Int
    delta_b::Float64
end

"""
A resolved, self-contained contingency specification.
After construction, contains only integer indices and floats —
no PSY types, no System reference.
"""
struct ContingencySpec
    name::String
    modifications::Vector{BranchModification}
end

"""
    WoodburyFactors

Cached Woodbury intermediates shared across monitored arcs for one contingency.
Computed from van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹

# Fields
- `Z::Matrix{Float64}`: B⁻¹U matrix (n_bus × M), one column per modified arc
- `W_lu::LinearAlgebra.LU{Float64, Matrix{Float64}, Vector{Int64}}`: LU factorization of W = A⁻¹ + U⊤B⁻¹U (M × M)
- `branch_indices::Vector{Int}`: Arc indices of modified branches
- `delta_b::Vector{Float64}`: Susceptance changes per modified branch
- `is_islanding::Bool`: Whether this contingency islands the network
"""
struct WoodburyFactors
    Z::Matrix{Float64}
    W_lu::LinearAlgebra.LU{Float64, Matrix{Float64}, Vector{Int64}}
    branch_indices::Vector{Int}
    delta_b::Vector{Float64}
    is_islanding::Bool
end
