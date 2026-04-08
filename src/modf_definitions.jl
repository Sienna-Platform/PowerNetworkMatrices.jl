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
    ShuntModification

A diagonal admittance change on a single bus (shunt outage or modification).
Used to track shunt component outages (FixedAdmittance, SwitchedAdmittance, StandardLoad)
that affect the Ybus but not DC sensitivity factors.

# Fields
- `bus_index::Int`: Index of the bus in the network matrix.
- `delta_y::ComplexF32`: Change in shunt admittance (negative for an outage).
"""
struct ShuntModification
    bus_index::Int
    delta_y::ComplexF32
end

"""
Merge ArcModifications that target the same arc index.
"""
function _merge_modifications(mods::Vector{ArcModification})
    length(mods) <= 1 && return mods
    by_arc = Dict{Int, Float64}()
    for m in mods
        by_arc[m.arc_index] = get(by_arc, m.arc_index, 0.0) + m.delta_b
    end
    return [ArcModification(idx, db) for (idx, db) in sort!(collect(by_arc); by = first)]
end

"""
Merge ShuntModifications that target the same bus index, sorted by bus index.
"""
function _merge_shunt_modifications(mods::Vector{ShuntModification})
    length(mods) <= 1 && return mods
    # Accumulate in ComplexF64 to avoid precision loss, downcast on output.
    by_bus = Dict{Int, ComplexF64}()
    for m in mods
        by_bus[m.bus_index] = get(by_bus, m.bus_index, zero(ComplexF64)) + m.delta_y
    end
    # Sort by bus index so that hash/== are insertion-order independent,
    # preventing cache misses when identical modifications arrive in different order.
    return [
        ShuntModification(idx, YBUS_ELTYPE(dy)) for
        (idx, dy) in sort!(collect(by_bus); by = first)
    ]
end

"""
    NetworkModification

Canonical description of topology changes to a power network.
Wraps arc susceptance changes, shunt admittance changes, and islanding status.
No dependency on `PSY.System` after construction.

# Fields
- `label::String`: Human-readable identifier for the modification.
- `arc_modifications::Vector{ArcModification}`: One entry per affected arc.
- `shunt_modifications::Vector{ShuntModification}`: One entry per affected shunt bus.
- `is_islanding::Bool`: Whether this modification disconnects the network.
"""
struct NetworkModification
    label::String
    arc_modifications::Vector{ArcModification}
    shunt_modifications::Vector{ShuntModification}
    is_islanding::Bool
    function NetworkModification(
        label::String,
        mods::Vector{ArcModification},
        shunt_mods::Vector{ShuntModification},
        is_islanding::Bool,
    )
        return new(
            label,
            _merge_modifications(mods),
            _merge_shunt_modifications(shunt_mods),
            is_islanding,
        )
    end
    function NetworkModification(
        label::String,
        mods::Vector{ArcModification},
    )
        return new(label, _merge_modifications(mods), ShuntModification[], false)
    end
end

function Base.hash(m::NetworkModification, h::UInt)
    h = hash(length(m.arc_modifications), h)
    for mod in m.arc_modifications
        h = hash(mod.arc_index, h)
        h = hash(mod.delta_b, h)
    end
    for smod in m.shunt_modifications
        h = hash(smod.bus_index, h)
        h = hash(smod.delta_y, h)
    end
    h = hash(m.is_islanding, h)
    return h
end

Base.:(==)(a::NetworkModification, b::NetworkModification) =
    a.arc_modifications == b.arc_modifications &&
    a.shunt_modifications == b.shunt_modifications &&
    a.is_islanding == b.is_islanding

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
