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
- `contingency_id::Int`: IS id of the contingency's outage.
- `key_hash::UInt`: `hash((topology, tier2))`, computed at construction.

Networks are built with the token `_Tier2Derived()` by callers that derive `tier2` from the
other two modifications with `_tier2_modification`.
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

# Whether `net` is a contingency on the base network, whose rows and factors are those of its
# tier-2 modification from the base network.
_on_base_topology(net::PostContingencyNetwork) = net.topology == _BASE_NETWORK

# Susceptance of arc `m` on K ∪ c. The tier-2 modification alone would miss the change K
# already made to a partly out arc.
_post_contingency_susceptance(
    arc_sus::Vector{Float64},
    m::Int,
    net::PostContingencyNetwork,
) = _post_modification_susceptance(
    arc_sus,
    m,
    _arc_modification_vector(net.post_contingency),
)

# Whether arc `m` is out of service on K ∪ c.
_post_contingency_arc_out(
    arc_sus::Vector{Float64},
    core::FactorCore,
    m::Int,
    net::PostContingencyNetwork,
) = _post_modification_arc_out(
    arc_sus,
    core,
    m,
    _arc_modification_vector(net.post_contingency),
)

"""
    _tier2_modification(post::NetworkModification, topology::NetworkModification) -> NetworkModification

Δ = `post` − `topology`, arc by arc and bus by bus, with entries that are zero in every field
dropped. An arc's group susceptance after the contingency comes from `post`, where the
components out at K and the contingency's are resolved together, so adding Δ to `topology`
lands on `post` with no member-share formula. Δ keeps `post`'s label and islanding flag.
"""
function _tier2_modification(post::NetworkModification, topology::NetworkModification)
    arc_mods = _modification_difference(
        ArcModification,
        post.arc_modifications,
        topology.arc_modifications,
        m -> m.arc_index,
        _arc_difference,
        i -> ArcModification(i, 0.0),
    )
    shunt_mods = _modification_difference(
        ShuntModification,
        post.shunt_modifications,
        topology.shunt_modifications,
        m -> m.bus_index,
        _shunt_difference,
        i -> ShuntModification(i, zero(YBUS_ELTYPE)),
    )
    return NetworkModification(post.label, arc_mods, shunt_mods, post.is_islanding)
end

function _modification_difference(
    ::Type{T},
    post::Tuple{Vararg{T}},
    topology::Tuple{Vararg{T}},
    index_of::I,
    difference::D,
    zero_at::Z,
)::Vector{T} where {T, I, D, Z}
    by_index = Dict{Int, T}(index_of(m) => m for m in topology)
    diffs = T[]
    for m in post
        i = index_of(m)
        push!(diffs, difference(m, something(get(by_index, i, nothing), zero_at(i))))
        delete!(by_index, i)
    end
    for (i, m) in by_index
        push!(diffs, difference(zero_at(i), m))
    end
    filter!(!_is_zero_change, diffs)
    return sort!(diffs; by = index_of)
end

_arc_difference(a::ArcModification, b::ArcModification) = ArcModification(
    a.arc_index,
    a.delta_b - b.delta_b,
    a.delta_shift_injection - b.delta_shift_injection,
    a.delta_y11 - b.delta_y11,
    a.delta_y12 - b.delta_y12,
    a.delta_y21 - b.delta_y21,
    a.delta_y22 - b.delta_y22,
    a.opened - b.opened,
)

_shunt_difference(a::ShuntModification, b::ShuntModification) =
    ShuntModification(a.bus_index, a.delta_y - b.delta_y)

_is_zero_change(m::ArcModification) =
    iszero(m.delta_b) && iszero(m.delta_shift_injection) && iszero(m.delta_y11) &&
    iszero(m.delta_y12) && iszero(m.delta_y21) && iszero(m.delta_y22) &&
    iszero(m.opened)

_is_zero_change(m::ShuntModification) = iszero(m.delta_y)
