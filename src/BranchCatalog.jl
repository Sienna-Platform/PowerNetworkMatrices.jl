"""
One reduced arc: the entry occupying it, the name it is indexed under, and every physical
branch at its leaves.

`leaves` is the only derived thing worth storing. It is what every type-keyed lookup needs
and it costs a tree walk to recompute, whereas the *structure* is already reachable by
iterating `entry`, and the *provenance* is already `entry`'s type -- see
[`arc_provenance`](@ref).

`leaves` is never empty. `leaf_components` yields the branch itself for anything that is not
an aggregate, so even a Ward `GenericArcImpedance` -- an arc genuinely backed by no component
-- appears as its own leaf. Emptiness is therefore not a test for "synthetic"; ask
[`arc_provenance`](@ref).
"""
struct ArcEntry
    entry::PSY.ACTransmission
    name::String
    leaves::Vector{PSY.ACTransmission}
end

get_entry(e::ArcEntry) = e.entry
get_name(e::ArcEntry) = e.name
get_leaves(e::ArcEntry) = e.leaves

# The arc alone: concrete and isbits, 16 bytes. This carried a `Symbol` naming the
# `BranchMapsByType` field that held the entry, so consumers could navigate back to it; that
# is now `get_reduction_entry`, and the arc is the whole identity.
const ARC_ENTRY = Tuple{Int, Int}
const ARC_TABLE = Dict{ARC_ENTRY, ArcEntry}
const NAME_TO_ARC = Dict{DataType, DataStructures.SortedDict{String, ARC_ENTRY}}
const COMPONENT_TO_ENTRY = Dict{DataType, Dict{String, String}}
const COMPONENT_NAME_INDEX = Dict{String, Vector{Tuple{DataType, ARC_ENTRY}}}

# Shared empty-map sentinels returned on a miss; callers must never mutate these.
const EMPTY_NAME_TO_ARC_MAP = DataStructures.SortedDict{String, ARC_ENTRY}()
const EMPTY_COMPONENT_TO_ENTRY_MAP = Dict{String, String}()

"""
    BranchCatalog

An immutable, per-branch-type index over the branch maps a network reduction produced. Every
matrix carries one, reachable with `get_branch_catalog`.

`arcs` is the table; every other field is an index into it, keyed by arc. They are built in
one pass over the reduction maps.

# Fields
- `network_reduction_data::NetworkReductionData`: the reduction this indexes
- `arcs`: arc => [`ArcEntry`](@ref). The single source of truth.
- `maps_by_type::BranchMapsByType`: the six reduction maps re-bucketed by branch type
- `name_to_arc`: per type, entry name => arc
- `component_to_entry_name`: per type, component name => name of the entry representing it
- `component_name_index`: component name => every candidate claiming it. Names are unique
  only per type, so a name can have several; lookup reports the ambiguity rather than
  picking one.
"""
struct BranchCatalog
    network_reduction_data::NetworkReductionData
    arcs::ARC_TABLE
    maps_by_type::BranchMapsByType
    name_to_arc::NAME_TO_ARC
    component_to_entry_name::COMPONENT_TO_ENTRY
    component_name_index::COMPONENT_NAME_INDEX
end

get_network_reduction_data(c::BranchCatalog) = c.network_reduction_data
get_arc_table(c::BranchCatalog) = c.arcs
get_all_branch_maps_by_type(c::BranchCatalog) = c.maps_by_type
get_name_to_arc_maps(c::BranchCatalog) = c.name_to_arc
get_component_to_reduction_name_map(c::BranchCatalog) = c.component_to_entry_name
get_component_name_index(c::BranchCatalog) = c.component_name_index

"""
    get_reduction_entry(c::BranchCatalog, arc) -> PSY.ACTransmission
    get_reduction_entry(c::BranchCatalog, ::Type{T}, name) -> PSY.ACTransmission

The entry occupying `arc` -- a single branch, or the aggregate a reduction folded onto it.
"""
get_reduction_entry(c::BranchCatalog, arc::ARC_ENTRY) = get_entry(c.arcs[arc])

function get_reduction_entry(
    c::BranchCatalog,
    ::Type{T},
    name::AbstractString,
) where {T <: PSY.ACTransmission}
    return get_reduction_entry(c, get_name_to_arc_map(c, T)[name])
end

"""
Every physical branch at the leaves of the entry on `arc`, precomputed at build time.
"""
get_arc_leaves(c::BranchCatalog, arc::ARC_ENTRY) = get_leaves(c.arcs[arc])

"""
Entries for branch type `T`. An absent `T` yields an empty map: a type is legitimately
missing when every branch of it was absorbed by a reduction.
"""
get_name_to_arc_map(c::BranchCatalog, ::Type{T}) where {T <: PSY.ACTransmission} =
    get(c.name_to_arc, T, EMPTY_NAME_TO_ARC_MAP)

# 3W windings are filed under the parent transformer type, so the wrapper key translates.
get_name_to_arc_map(c::BranchCatalog, ::Type{ThreeWindingTransformerCircuit}) =
    get_name_to_arc_map(c, PSY.ThreeWindingTransformer)

get_component_to_reduction_name_map(
    c::BranchCatalog,
    ::Type{T},
) where {T <: PSY.ACTransmission} =
    get(c.component_to_entry_name, T, EMPTY_COMPONENT_TO_ENTRY_MAP)

get_component_to_reduction_name_map(
    c::BranchCatalog,
    ::Type{ThreeWindingTransformerCircuit},
) = get_component_to_reduction_name_map(c, PSY.ThreeWindingTransformer)

function Base.isempty(c::BranchCatalog)
    return isempty(c.maps_by_type) && isempty(c.name_to_arc) &&
           isempty(c.component_to_entry_name) && isempty(c.component_name_index)
end

##############################################################################
############################ Entry matching ##################################
##############################################################################

_keep_all(::Type, ::Any) = true

_is_unfiltered(predicate) = predicate === _keep_all

# PNM's `get_name`, not `PSY.get_name`: a group's leaves can include a
# `ThreeWindingTransformerCircuit`, whose fields are `(transformer, circuit, winding_number)`.
# `PSY.get_name` resolves to IS's generic fallback, which reads `.name` and throws a
# `FieldError` on it -- inside the logging call, and only once Debug logging is on.
function _warn_mixed_group(kind::String, branches)
    @warn "$kind contains mixed branch types, filters might be applied to more " *
          "components than intended. Use Logging.Debug for additional information."
    @debug "$kind branch types: $(typeof.(branches))"
    @debug "$kind branch names: $(get_name.(branches))"
    return
end

_entry_matches(device::T, predicate) where {T <: PSY.ACTransmission} =
    predicate(T, device)

"""
    _entry_name(arc, entry) -> String

The name an arc's entry is indexed under.

A direct arc is one branch and keeps that branch's own name: it is already unique per type,
already stable, and renaming it would move result keys for the ~90% of arcs no reduction
touched. A composite arc takes its name from the arc instead.

Deriving a composite's name here rather than on the aggregate is deliberate on two counts.
The catalog knows the arc the entry is *filed under*, while an aggregate's `arc_key` holds
original bus numbers that `reverse_bus_search_map` may since have remapped -- the two can
disagree. And `_composite_entries` admits at most one composite per unordered bus pair, so
arc => name is injective by construction here, where on the aggregate it was the longest
common prefix of member names: `La`/`Lb` and `Lc`/`Ld` both yielded `Lseries_chain`, and the
name moved whenever membership did.
"""
_entry_name(::Tuple{Int, Int}, entry::PSY.ACTransmission) = get_name(entry)
_entry_name(arc::Tuple{Int, Int}, ::AbstractBranchesParallel) =
    "parallel_$(arc[1])_$(arc[2])"
_entry_name(arc::Tuple{Int, Int}, ::BranchesSeries) = "series_$(arc[1])_$(arc[2])"

"""
How an arc of the reduced network came to exist. Read off the entry with
[`arc_provenance`](@ref); never stored, since the entry's type already determines it.

Radial has no member on purpose: it removes an arc rather than producing one, so there is
nothing left to describe.
"""
abstract type ArcProvenance end

"One physical branch holding its arc alone, untouched by any reduction."
struct DirectArc <: ArcProvenance end

"Two or more branches on one bus pair, folded into a single parallel group."
struct ParallelArc <: ArcProvenance end

"A degree-two chain, folded into one arc spanning the chain's endpoints."
struct SeriesArc <: ArcProvenance end

"""
A Ward equivalent: admittance from Gaussian elimination, backed by no component.

Unreachable today -- nothing routes such an arc into a catalog -- but `GenericArcImpedance`
subtypes `PSY.ACTransmission`, so without this arm one would answer `DirectArc` from the
blanket method and assert component backing it does not have.
"""
struct SyntheticArc <: ArcProvenance end

"""
    arc_provenance(entry) -> ArcProvenance
    arc_provenance(c::BranchCatalog, arc) -> ArcProvenance

How the arc `entry` occupies came to exist, read off the entry's own type.
"""
arc_provenance(::PSY.ACTransmission) = DirectArc()
arc_provenance(::AbstractBranchesParallel) = ParallelArc()
arc_provenance(::BranchesSeries) = SeriesArc()
arc_provenance(::PSY.GenericArcImpedance) = SyntheticArc()

arc_provenance(c::BranchCatalog, arc::ARC_ENTRY) =
    arc_provenance(get_reduction_entry(c, arc))

"""
    _branch_multiplier(provenance, entry, branch_name, arc) -> Float64

Factor scaling a per-arc matrix entry to the named branch's share of it, dispatched on how
the arc came to exist. Backs [`get_branch_multiplier`](@ref).
"""
_branch_multiplier(::DirectArc, ::PSY.ACTransmission, ::AbstractString, ::ARC_ENTRY) = 1.0

# Backed by no component, so nothing shares it.
_branch_multiplier(
    ::SyntheticArc,
    ::PSY.GenericArcImpedance,
    ::AbstractString,
    ::ARC_ENTRY,
) = 1.0

# A member carries its susceptance-fraction share of the group flow.
function _branch_multiplier(
    ::ParallelArc,
    group::AbstractBranchesParallel,
    branch_name::AbstractString,
    arc::ARC_ENTRY,
)
    for member in group
        get_name(member) == branch_name || continue
        return compute_parallel_multiplier(group, member)
    end
    return error(
        "Branch $branch_name is indexed on arc $(arc) but no member of the group there " *
        "carries that name.",
    )
end

# Unreachable today -- `_build_component_name_index` indexes no series entry -- but the arm
# names the limitation instead of failing as a missing key somewhere else.
_branch_multiplier(
    ::SeriesArc,
    ::BranchesSeries,
    branch_name::AbstractString,
    arc::ARC_ENTRY,
) = error(
    "Branch $branch_name is a segment of the series chain on arc $(arc). A chain's flow " *
    "does not decompose into per-segment shares of one matrix row, so it has no " *
    "multiplier; resolve the segment by component identity instead.",
)

##############################################################################
############################## Index building ################################
##############################################################################

_bucket_name_to_arc(name_to_arc::NAME_TO_ARC, T::DataType) =
    get!(() -> DataStructures.SortedDict{String, ARC_ENTRY}(), name_to_arc, T)

_bucket_entry_names(component_to_entry::COMPONENT_TO_ENTRY, T::DataType) =
    get!(() -> Dict{String, String}(), component_to_entry, T)

function _store!(bucket::Dict{K, V}, k, v) where {K, V}
    bucket[k] = v
    return
end

"""
Record `arc`'s row in the table and return the name it is indexed under.

The single place an entry's name and leaves are computed. Every index then reads them back
from here rather than recomputing.
"""
function _record_arc!(arcs::ARC_TABLE, arc::ARC_ENTRY, entry)
    row = get!(arcs, arc) do
        ArcEntry(entry, _entry_name(arc, entry), leaf_components(entry))
    end
    return get_name(row)
end

"""
Index a forward (arc-keyed) reduction map under the buckets `bucket_types(entry)` names,
creating each with `empty_bucket(entry)`.

The parallel map's buckets are widened to `AbstractBranchesParallel` so a
`MixedBranchesParallel` is reachable under every member type it contains.
"""
function _index_forward!(
    dest::Dict{DataType, Any},
    arcs::ARC_TABLE,
    name_to_arc::NAME_TO_ARC,
    source,
    predicate,
    bucket_types,
    empty_bucket,
)
    for (arc, entry) in source
        _entry_matches(entry, predicate) || continue
        name = _record_arc!(arcs, arc, entry)
        for T in bucket_types(entry)
            _store!(get!(() -> empty_bucket(entry), dest, T), arc, entry)
            _bucket_name_to_arc(name_to_arc, T)[name] = arc
        end
    end
    return
end

"""
Index a reverse (entry-keyed) reduction map, recording which entry represents each member so
a component absorbed into an aggregate can be redirected to the entry carrying its flow.

The bucket *key* is `_get_segment_type(member)`, a PSY component type, while the bucket's key
*type* is `typeof(member)`. For a 3W winding those differ: filed under the parent transformer
type, holding `ThreeWindingTransformerCircuit` keys.
"""
function _index_reverse!(
    dest::Dict{DataType, Any},
    arcs::ARC_TABLE,
    component_to_entry::COMPONENT_TO_ENTRY,
    source,
    predicate,
)
    for (member, arc) in source
        _entry_matches(member, predicate) || continue
        T = _get_segment_type(member)
        _store!(get!(() -> Dict{typeof(member), Tuple{Int, Int}}(), dest, T), member, arc)
        # The name comes from the table, not from a second call to `_entry_name`: one
        # computation, so forward and reverse cannot disagree about what the entry is called.
        _bucket_entry_names(component_to_entry, T)[get_name(member)] = get_name(arcs[arc])
    end
    return
end

"""
Index the series map. A chain is filed under every type appearing anywhere in it, so a caller
iterating one branch type finds every chain that type participates in.

One entry name per arc, not one per segment. Members reach their entry through
`component_to_entry`, which is where the per-component view belongs.
"""
function _index_series!(
    dest::Dict{DataType, Any},
    arcs::ARC_TABLE,
    name_to_arc::NAME_TO_ARC,
    component_to_entry::COMPONENT_TO_ENTRY,
    source,
    predicate,
)
    for (arc, chain) in source
        _entry_matches(chain, predicate) || continue
        name = _record_arc!(arcs, arc, chain)
        for T in _get_concrete_types(chain)
            _store!(
                get!(() -> Dict{Tuple{Int, Int}, BranchesSeries}(), dest, T),
                arc,
                chain,
            )
            _bucket_name_to_arc(name_to_arc, T)[name] = arc
        end
        # Each leaf redirects to the one entry carrying its flow, filed under the leaf's own
        # bucket rather than under every type in the chain.
        for leaf in leaf_components(chain)
            _bucket_entry_names(component_to_entry, _get_segment_type(leaf))[get_name(
                leaf,
            )] =
                name
        end
    end
    return
end

# Names come from `_index_series!`, which sees the segment structure this map flattens away.
function _index_reverse_series!(dest::Dict{DataType, Any}, source, predicate)
    for (member, arc) in source
        _entry_matches(member, predicate) || continue
        _store!(
            get!(
                () -> Dict{PSY.ACTransmission, Tuple{Int, Int}}(),
                dest,
                _get_segment_type(member),
            ),
            member,
            arc,
        )
    end
    return
end

_name_candidates(index::COMPONENT_NAME_INDEX, name::String) =
    get!(() -> Tuple{DataType, ARC_ENTRY}[], index, name)

"""
Component-name index for name-based matrix indexing (`get_branch_multiplier`), whose API takes
a bare name.

Built from the component-keyed maps rather than `name_to_arc`, which holds *entry* names: an
aggregate's entry name is the group's, not any component's. Series-chain members are absent;
name-based matrix indexing does not resolve them.
"""
function _build_component_name_index(nrd::NetworkReductionData, predicate)
    index = COMPONENT_NAME_INDEX()
    for (arc, entry) in nrd.direct_branch_map
        _entry_matches(entry, predicate) || continue
        push!(
            _name_candidates(index, get_name(entry)),
            (typeof(entry), arc),
        )
    end
    for (member, arc) in nrd.reverse_parallel_branch_map
        _entry_matches(member, predicate) || continue
        push!(
            _name_candidates(index, get_name(member)),
            (typeof(member), arc),
        )
    end
    return index
end

"""
Throws unless every arc the reduction *folded* is still reachable by component type. Opt-in:
pass `validate = true` to [`BranchCatalog`](@ref), or call this directly on a catalog under
test. Construction does not run it by default.

Radial reduction legitimately removes a branch: its arc leaves the network and leaves these
maps, so there is nothing left to index. Series and parallel reductions remove nothing --
they fold branches into a composite arc that still exists and still carries flow, so it must
stay reachable. This checks that the distinction held.

The two are indistinguishable to a consumer, which is why the check is worth writing: an arc
is reached only by asking for a PSY component type, and a type that indexes nothing returns
an empty map rather than an error. A folded arc that lost its entry is therefore invisible --
no flow variable, no rating constraint, no nodal-balance term, and no complaint.
"""
function _validate_catalog_closure(nrd::NetworkReductionData, name_to_arc::NAME_TO_ARC)
    indexed = Set{Tuple{Int, Int}}()
    for by_name in values(name_to_arc)
        for arc in values(by_name)
            push!(indexed, arc)
        end
    end
    # Only the component-backed maps. Ward's `added_arc_impedance_map` arcs come out of
    # Gaussian elimination and are backed by no component, so no component type can claim
    # them; they are outside this invariant by construction.
    for (map_name, source) in (
        (:direct_branch_map, nrd.direct_branch_map),
        (:parallel_branch_map, nrd.parallel_branch_map),
        (:series_branch_map, nrd.series_branch_map),
    )
        for (arc, entry) in source
            arc in indexed && continue
            error(
                "Arc $arc ($(get_name(entry)) in $map_name) is reachable from no branch " *
                "type in the catalog, so nothing that indexes by component type can find " *
                "it. Leaf types: $(_get_concrete_types(entry)).",
            )
        end
    end
    return
end

"""
    BranchCatalog(nrd::NetworkReductionData; validate = false)

The complete index over `nrd`. See `validate` on the filtered method below.
"""
BranchCatalog(nrd::NetworkReductionData; validate::Bool = false) =
    BranchCatalog(nrd, _keep_all; validate = validate)

"""
    BranchCatalog(nrd::NetworkReductionData, predicate; validate = false)

Index over `nrd` holding only entries `predicate` accepts, where `predicate(T, component)`
returns whether a component of branch type `T` should be indexed. An aggregate is judged by
`_entry_matches`, which applies the predicate to every physical branch at its leaves.

`validate` runs [`_validate_catalog_closure`](@ref) before returning. It is off by default so
that indexing stays a pure build step; turn it on in tests, or wherever a caller wants the
folded-arc invariant enforced rather than assumed. It is rejected for a filtered catalog,
where the invariant does not hold by design.
"""
function BranchCatalog(nrd::NetworkReductionData, predicate; validate::Bool = false)
    maps = BranchMapsByType()
    arcs = ARC_TABLE()
    name_to_arc = NAME_TO_ARC()
    component_to_entry = COMPONENT_TO_ENTRY()

    _index_forward!(
        maps.direct_branch_map, arcs, name_to_arc, nrd.direct_branch_map,
        predicate,
        entry -> (_get_segment_type(entry),),
        entry -> Dict{Tuple{Int, Int}, typeof(entry)}(),
    )
    _index_reverse!(
        maps.reverse_direct_branch_map, arcs, component_to_entry,
        nrd.reverse_direct_branch_map, predicate,
    )

    _index_forward!(
        maps.parallel_branch_map, arcs, name_to_arc, nrd.parallel_branch_map,
        predicate,
        _get_concrete_types,
        _ -> _empty_parallel_branch_map(),
    )
    _index_reverse!(
        maps.reverse_parallel_branch_map, arcs, component_to_entry,
        nrd.reverse_parallel_branch_map, predicate,
    )

    _index_series!(
        maps.series_branch_map, arcs, name_to_arc, component_to_entry,
        nrd.series_branch_map, predicate,
    )
    _index_reverse_series!(
        maps.reverse_series_branch_map, nrd.reverse_series_branch_map, predicate,
    )

    if validate
        # Refused rather than skipped: a filter drops arcs by design, so an unreachable arc
        # there is a filter decision, not a lost entry. Silently answering "valid" would
        # report a guarantee this cannot give.
        _is_unfiltered(predicate) || throw(
            ArgumentError(
                "validate = true applies only to an unfiltered catalog; a filtered one \
                 omits arcs by design.",
            ),
        )
        _validate_catalog_closure(nrd, name_to_arc)
    end

    return BranchCatalog(
        nrd,
        arcs,
        maps,
        name_to_arc,
        component_to_entry,
        _build_component_name_index(nrd, predicate),
    )
end
