"""
    ArcModification

A susceptance change on a single aggregated arc, with the matching change in the arc's DC
phase-shift injection and optional Ybus Pi-model deltas.
Full outage: `delta_b = -b_arc`. Single circuit on double-circuit: `delta_b = -b_circuit`.

# Fields
- `arc_index::Int`: Index of the modified arc in the network matrix.
- `delta_b::Float64`: Change in susceptance (negative for an outage or reduction).
- `delta_shift_injection::Float64`: Change in the arc's DC shift injection `b·α`, oriented
  to the arc, system base. Zero for every unshifted circuit.
- `delta_y11::ComplexF32`: Change in Pi-model self-admittance at the from bus.
- `delta_y12::ComplexF32`: Change in Pi-model mutual admittance (from -> to).
- `delta_y21::ComplexF32`: Change in Pi-model mutual admittance (to -> from).
- `delta_y22::ComplexF32`: Change in Pi-model self-admittance at the to bus.
- `opened::Int`: Number of the arc's members taken out of service: one per tripped branch,
  every member for a full arc outage, zero for a partial change. A `Ybus` modified in place
  treats an arc whose members are all out as removed.
"""
struct ArcModification
    arc_index::Int
    delta_b::Float64
    delta_shift_injection::Float64
    delta_y11::ComplexF32
    delta_y12::ComplexF32
    delta_y21::ComplexF32
    delta_y22::ComplexF32
    opened::Int
end

"""
Convenience constructor for an unshifted arc with no Pi-model deltas and no members opened.
It changes only the DC susceptance. [`apply_ybus_modification!`](@ref) leaves the Ybus
admittances unchanged for it and never removes the arc, even when `delta_b = -b_arc`.
"""
function ArcModification(arc_index::Int, delta_b::Float64)
    z = zero(YBUS_ELTYPE)
    return ArcModification(arc_index, delta_b, 0.0, z, z, z, z, 0)
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
    by_arc = Dict{
        Int,
        Tuple{Float64, Float64, ComplexF64, ComplexF64, ComplexF64, ComplexF64, Int},
    }()
    for m in mods
        prev = get(
            by_arc,
            m.arc_index,
            (
                0.0,
                0.0,
                zero(ComplexF64),
                zero(ComplexF64),
                zero(ComplexF64),
                zero(ComplexF64),
                0,
            ),
        )
        by_arc[m.arc_index] = (
            prev[1] + m.delta_b,
            prev[2] + m.delta_shift_injection,
            prev[3] + m.delta_y11,
            prev[4] + m.delta_y12,
            prev[5] + m.delta_y21,
            prev[6] + m.delta_y22,
            prev[7] + m.opened,
        )
    end
    return [
        ArcModification(
            idx,
            vals[1],
            vals[2],
            YBUS_ELTYPE(vals[3]),
            YBUS_ELTYPE(vals[4]),
            YBUS_ELTYPE(vals[5]),
            YBUS_ELTYPE(vals[6]),
            vals[7],
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
Base.hash(m::NetworkModification, h::UInt) =
    hash((m.arc_modifications, m.shunt_modifications, m.is_islanding), h)

Base.:(==)(a::NetworkModification, b::NetworkModification) =
    a.arc_modifications == b.arc_modifications &&
    a.shunt_modifications == b.shunt_modifications &&
    a.is_islanding == b.is_islanding

# The empty modification: VirtualMODF caches base-network rows under this key.
const _BASE_NETWORK = NetworkModification("base", ArcModification[])

"""
    ContingencySpec

A resolved, self-contained contingency specification backed by a
[`NetworkModification`](@ref). The id links back to the source
`PSY.UnplannedOutage` supplemental attribute for caching purposes.

# Fields
- `id::Int`: IS id matching the source unplanned outage supplemental attribute.
- `modification::NetworkModification`: The network topology change.
"""
struct ContingencySpec
    id::Int
    modification::NetworkModification
end

# Construction token: only callers that derive `tier2` as `post_contingency` − `topology`
# pass it, so a network never pairs a topology with a mismatched tier-2 modification.
struct _Tier2Derived end

"""
    PostContingencyNetwork

The network after a contingency on a scheduled topology K, keyed by K and the tier-2
modification Δ that takes K to K ∪ c. `hash` and `==` compare `topology` and `tier2` only,
so steps that share a topology share one network, and a network on the base topology equals
the registered contingency's.

# Fields
- `topology::NetworkModification`: The modification that takes the base network to K; the
  empty modification for the base topology.
- `post_contingency::NetworkModification`: The modification that takes the base network to
  K ∪ c, with the components out at K and the contingency's remaining components resolved
  together.
- `tier2::NetworkModification`: Δ = `post_contingency` − `topology`, arc by arc and bus by
  bus, with entries that are zero in every field dropped.
- `contingency_id::Int`: IS id of the contingency's `PSY.UnplannedOutage`.
- `key_hash::UInt`: `hash((topology, tier2))`, computed at construction.

Networks are built by [`get_post_contingency_network`](@ref) on a scheduled topology, or by
`PostContingencyNetwork(vmodf, contingency_id)` on the base topology, which derive `tier2`
from the other two modifications.
"""
struct PostContingencyNetwork
    topology::NetworkModification
    post_contingency::NetworkModification
    tier2::NetworkModification
    contingency_id::Int
    key_hash::UInt
    function PostContingencyNetwork(
        ::_Tier2Derived,
        topology::NetworkModification,
        post_contingency::NetworkModification,
        tier2::NetworkModification,
        contingency_id::Int,
    )
        return new(
            topology,
            post_contingency,
            tier2,
            contingency_id,
            hash((topology, tier2)),
        )
    end
end

# Networks are dictionary keys in every row and factor lookup, and hashing a modification
# walks its arcs, so the key hash is computed once.
Base.hash(n::PostContingencyNetwork, h::UInt) = hash(n.key_hash, h)

# Steps on one topology share its modification object, so identity settles most comparisons.
Base.:(==)(a::PostContingencyNetwork, b::PostContingencyNetwork) =
    a.key_hash == b.key_hash &&
    (a.topology === b.topology || a.topology == b.topology) &&
    (a.tier2 === b.tier2 || a.tier2 == b.tier2)

"""
    WoodburyFactors

Cached Woodbury intermediates shared across monitored arcs for one contingency.
Computed from van Dijk et al. Eq. 29:
    B_m⁻¹ = B_r⁻¹ - B_r⁻¹ U (A⁻¹ + U⊤ B_r⁻¹ U)⁻¹ U⊤ B_r⁻¹

# Fields
- `Z::Matrix{Float64}`: B⁻¹U matrix (n_bus × M), one column per arc whose susceptance
  changes; arcs with `delta_b == 0` leave `B` unchanged and are not included
- `W_inv::Matrix{Float64}`: Pre-inverted W = (A⁻¹ + U⊤B⁻¹U)⁻¹ (M × M). For M ≤ 2, computed analytically; for M > 2, computed via LU factorization.
- `arc_indices::Vector{Int}`: Arc indices of the arcs whose susceptance changes
- `delta_b::Vector{Float64}`: Susceptance change per arc in `arc_indices`
- `arc_out::Vector{Bool}`: Per modified arc, whether the modification opens every member of
  the arc. Set from member counts, because the summed `delta_b` of a lossy parallel group
  does not cancel the arc susceptance exactly.
- `is_islanding::Bool`: Whether this contingency islands the network
- `bus_island_labels::Vector{Int}`: Post-contingency connected-component label per
  bus position, used to force entries of buses disconnected from the monitored arc
  to exactly zero. Empty unless `is_islanding`.
"""
struct WoodburyFactors
    Z::Matrix{Float64}
    W_inv::Matrix{Float64}
    arc_indices::Vector{Int}
    delta_b::Vector{Float64}
    arc_out::Vector{Bool}
    is_islanding::Bool
    bus_island_labels::Vector{Int}
end
