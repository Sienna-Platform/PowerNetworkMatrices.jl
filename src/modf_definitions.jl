"""
    ArcModification

A susceptance change on a single aggregated arc, with optional Ybus Pi-model deltas.
Full outage: `delta_b = -b_arc`. Single circuit on double-circuit: `delta_b = -b_circuit`.

# Fields
- `arc_index::Int`: Index of the modified arc in the network matrix.
- `delta_b::Float64`: Change in susceptance (negative for an outage or reduction).
- `delta_y11::ComplexF32`: Change in Pi-model self-admittance at the from bus.
- `delta_y12::ComplexF32`: Change in Pi-model mutual admittance (from -> to).
- `delta_y21::ComplexF32`: Change in Pi-model mutual admittance (to -> from).
- `delta_y22::ComplexF32`: Change in Pi-model self-admittance at the to bus.
"""
struct ArcModification
    arc_index::Int
    delta_b::Float64
    delta_y11::ComplexF32
    delta_y12::ComplexF32
    delta_y21::ComplexF32
    delta_y22::ComplexF32
end

"""
Backward-compatible constructor that sets all Pi-model ΔY fields to zero.
"""
function ArcModification(arc_index::Int, delta_b::Float64)
    z = zero(YBUS_ELTYPE)
    return ArcModification(arc_index, delta_b, z, z, z, z)
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
function _merge_arc_modifications(mods::Vector{ArcModification})
    length(mods) <= 1 && return mods
    by_arc = Dict{Int, Tuple{Float64, ComplexF64, ComplexF64, ComplexF64, ComplexF64}}()
    for m in mods
        prev = get(
            by_arc,
            m.arc_index,
            (0.0, zero(ComplexF64), zero(ComplexF64), zero(ComplexF64), zero(ComplexF64)),
        )
        by_arc[m.arc_index] = (
            prev[1] + m.delta_b,
            prev[2] + m.delta_y11,
            prev[3] + m.delta_y12,
            prev[4] + m.delta_y21,
            prev[5] + m.delta_y22,
        )
    end
    return [
        ArcModification(
            idx,
            vals[1],
            YBUS_ELTYPE(vals[2]),
            YBUS_ELTYPE(vals[3]),
            YBUS_ELTYPE(vals[4]),
            YBUS_ELTYPE(vals[5]),
        ) for (idx, vals) in sort!(collect(by_arc); by = first)
    ]
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
- `arc_modifications::Tuple{Vararg{ArcModification}}`: One entry per affected arc (immutable).
- `shunt_modifications::Tuple{Vararg{ShuntModification}}`: One entry per affected shunt bus (immutable).
- `is_islanding::Bool`: Whether this modification disconnects the network.

Modification vectors are converted to tuples at construction time to guarantee
immutability. This is required because `NetworkModification` is used as a `Dict`
key (via custom `hash`/`==`) in VirtualMODF caches; mutable fields would
silently corrupt lookups if modified after insertion.
"""
struct NetworkModification
    label::String
    arc_modifications::Tuple{Vararg{ArcModification}}
    shunt_modifications::Tuple{Vararg{ShuntModification}}
    is_islanding::Bool
    function NetworkModification(
        label::String,
        mods::Vector{ArcModification},
        shunt_mods::Vector{ShuntModification},
        is_islanding::Bool,
    )
        return new(
            label,
            Tuple(_merge_arc_modifications(mods)),
            Tuple(_merge_shunt_modifications(shunt_mods)),
            is_islanding,
        )
    end
    function NetworkModification(
        label::String,
        mods::Vector{ArcModification},
    )
        return new(label, Tuple(_merge_arc_modifications(mods)), (), false)
    end
end

# `label` is intentionally excluded from hash and equality so that physically
# identical modifications compare equal regardless of naming. The woodbury_cache
# in VirtualMODF relies on this property for cache hits across naming paths.
function Base.hash(m::NetworkModification, h::UInt)
    h = hash(length(m.arc_modifications), h)
    for mod in m.arc_modifications
        h = hash(mod.arc_index, h)
        h = hash(mod.delta_b, h)
        h = hash(mod.delta_y11, h)
        h = hash(mod.delta_y12, h)
        h = hash(mod.delta_y21, h)
        h = hash(mod.delta_y22, h)
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
