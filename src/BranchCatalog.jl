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

# The arc is the whole identity; `get_reduction_entry` recovers the entry.
const ARC_ENTRY = Tuple{Int, Int}
const ARC_TABLE = Dict{ARC_ENTRY, ArcEntry}
const NAME_TO_ARC = Dict{DataType, DataStructures.SortedDict{String, ARC_ENTRY}}
const COMPONENT_TO_ENTRY = Dict{DataType, Dict{String, String}}
const COMPONENT_NAME_INDEX = Dict{String, Vector{Tuple{DataType, ARC_ENTRY}}}

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

The entry occupying `arc` -- a single branch, or the aggregate a reduction folded onto it.
"""
get_reduction_entry(c::BranchCatalog, arc::ARC_ENTRY) = get_entry(c.arcs[arc])

"""
Every physical branch at the leaves of the entry on `arc`, precomputed at build time.
"""
get_arc_leaves(c::BranchCatalog, arc::ARC_ENTRY) = get_leaves(c.arcs[arc])

"""
Entries for branch type `T`. An absent `T` yields an empty map: a type is legitimately
missing when every branch of it was absorbed by a reduction.

A miss returns a fresh map; a shared empty would be mutable across every catalog.
"""
get_name_to_arc_map(c::BranchCatalog, ::Type{T}) where {T <: PSY.ACTransmission} =
    get(() -> DataStructures.SortedDict{String, ARC_ENTRY}(), c.name_to_arc, T)

# 3W windings are filed under the parent transformer type, so the wrapper key translates.
get_name_to_arc_map(c::BranchCatalog, ::Type{ThreeWindingTransformerCircuit}) =
    get_name_to_arc_map(c, PSY.ThreeWindingTransformer)

get_component_to_reduction_name_map(
    c::BranchCatalog,
    ::Type{T},
) where {T <: PSY.ACTransmission} =
    get(() -> Dict{String, String}(), c.component_to_entry_name, T)

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
    "$(arc[1])_$(arc[2])_double_circuit"
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
    _branch_multiplier(provenance, entry, branch_name, arc, nr) -> Float64

Factor scaling a per-arc matrix entry to the named branch's share of it, dispatched on how
the arc came to exist. Backs [`get_branch_multiplier`](@ref).
"""
_branch_multiplier(
    ::DirectArc,
    ::PSY.ACTransmission,
    ::AbstractString,
    ::ARC_ENTRY,
    ::NetworkReductionData,
) = 1.0

# Backed by no component, so nothing shares it.
_branch_multiplier(
    ::SyntheticArc,
    ::PSY.GenericArcImpedance,
    ::AbstractString,
    ::ARC_ENTRY,
    ::NetworkReductionData,
) = 1.0

# A member carries its susceptance-fraction share of the group flow.
function _branch_multiplier(
    ::ParallelArc,
    group::AbstractBranchesParallel,
    branch_name::AbstractString,
    arc::ARC_ENTRY,
    nr::NetworkReductionData,
)
    for member in group
        get_name(member) == branch_name || continue
        return compute_parallel_multiplier(group, member, nr)
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
    ::NetworkReductionData,
) = error(
    "Branch $branch_name is a segment of the series chain on arc $(arc). A chain's flow " *
    "does not decompose into per-segment shares of one matrix row, so it has no " *
    "multiplier; resolve the segment by component identity instead.",
)

##############################################################################
############################## Index building ################################
##############################################################################

"""
Values staged per bucket key, keys in first-encounter order. Each bucket is then built once,
with concrete types, from the same insert sequence the per-entry writes used to apply.
"""
const _VectorsByType{P} = DataStructures.OrderedDict{DataType, Vector{P}}

_push_by_type!(staged::_VectorsByType{P}, T::DataType, value) where {P} =
    push!(get!(Vector{P}, staged, T), value)

function _fill_bucket!(bucket::Dict, pairs::Vector)
    sizehint!(bucket, length(pairs))
    for (k, v) in pairs
        bucket[k] = v
    end
    return bucket
end

# `empty_bucket(first(pairs))` fixes the bucket's type, as the first insert used to.
function _fill_buckets!(dest::Dict{DataType, Any}, staged::_VectorsByType, empty_bucket)
    for (T, pairs) in staged
        dest[T] = _fill_bucket!(empty_bucket(first(pairs)), pairs)
    end
    return
end

"""
One `SortedDict` per type, bulk-loaded from the staged `name => arc` writes. The stable sort keeps
writes to one name in order and the last one wins, as repeated `setindex!` did; a collision that
drops an arc is `_validate_catalog_closure`'s to report.
"""
function _sorted_name_buckets!(
    name_to_arc::NAME_TO_ARC,
    staged::_VectorsByType{Pair{String, ARC_ENTRY}},
)
    for (T, pairs) in staged
        sort!(pairs; by = first, alg = Base.Sort.DEFAULT_STABLE)
        n = 0
        for i in eachindex(pairs)
            if i < lastindex(pairs) && first(pairs[i + 1]) == first(pairs[i])
                continue
            end
            n += 1
            pairs[n] = pairs[i]
        end
        resize!(pairs, n)
        name_to_arc[T] = DataStructures.SortedDict{String, ARC_ENTRY}(Val(true), pairs)
    end
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
Stage a forward (arc-keyed) reduction map under the buckets `bucket_types(entry)` names.

The parallel map's buckets are widened to `AbstractBranchesParallel` so a
`MixedBranchesParallel` is reachable under every member type it contains.
"""
function _index_forward!(
    staged::_VectorsByType,
    names::_VectorsByType,
    arcs::ARC_TABLE,
    source,
    predicate,
    bucket_types,
)
    for (arc, entry) in source
        _index_forward_entry!(staged, names, arcs, arc, entry, predicate, bucket_types)
    end
    return
end

# Function barrier: one dynamic dispatch per entry, concrete below it.
function _index_forward_entry!(
    staged::_VectorsByType,
    names::_VectorsByType,
    arcs::ARC_TABLE,
    arc::ARC_ENTRY,
    entry::PSY.ACTransmission,
    predicate::F,
    bucket_types::G,
) where {F, G}
    _entry_matches(entry, predicate) || return
    name = _record_arc!(arcs, arc, entry)
    for T in bucket_types(entry)
        _push_by_type!(staged, T, arc => entry)
        _push_by_type!(names, T, name => arc)
    end
    return
end

"""
Stage a reverse (entry-keyed) reduction map, recording which entry represents each member so
a component absorbed into an aggregate can be redirected to the entry carrying its flow.

The bucket *key* is `_get_segment_type(member)`, a PSY component type, while the bucket's key
*type* is `typeof(member)`. For a 3W winding those differ: filed under the parent transformer
type, holding `ThreeWindingTransformerCircuit` keys.
"""
function _index_reverse!(
    staged::_VectorsByType,
    entry_names::_VectorsByType,
    arcs::ARC_TABLE,
    source,
    predicate,
)
    for (member, arc) in source
        _index_reverse_entry!(staged, entry_names, arcs, member, arc, predicate)
    end
    return
end

function _index_reverse_entry!(
    staged::_VectorsByType,
    entry_names::_VectorsByType,
    arcs::ARC_TABLE,
    member::PSY.ACTransmission,
    arc::ARC_ENTRY,
    predicate::F,
) where {F}
    _entry_matches(member, predicate) || return
    # `MixedBranchesParallel` matches on `all`, so a member can pass this predicate while its
    # group fails. Skip such a member: the forward pass gave its arc no row in `arcs`.
    haskey(arcs, arc) || return
    T = _get_segment_type(member)
    _push_by_type!(staged, T, member => arc)
    # The name comes from the table, not from a second call to `_entry_name`: one
    # computation, so forward and reverse cannot disagree about what the entry is called.
    _push_by_type!(entry_names, T, get_name(member) => get_name(arcs[arc]))
    return
end

"""
Stage the series map. A chain is filed under every type appearing anywhere in it, so a caller
iterating one branch type finds every chain that type participates in.

One entry name per arc, not one per segment. Members reach their entry through
`component_to_entry`, which is where the per-component view belongs.
"""
function _index_series!(
    staged::_VectorsByType,
    arcs::ARC_TABLE,
    names::_VectorsByType,
    entry_names::_VectorsByType,
    source,
    predicate,
)
    for (arc, chain) in source
        _entry_matches(chain, predicate) || continue
        _record_arc!(arcs, arc, chain)
        # One row per SEGMENT, not one per chain. The rows of `name_to_arc` are the rows
        # results are reported under, and a lossless chain carries the same flow in every
        # segment -- so each segment reports under its own name, and that name is a real
        # component the caller can look up. A parallel segment is the exception: the flows
        # of its members are never computed individually, so the group reports once under
        # its own name, exactly as a top-level parallel group does. The chain's own
        # `series_<from>_<to>` identity lives in `arcs`; it is not a reporting row.
        for segment in chain
            segment_name = get_name(segment)
            for T in _get_concrete_types(segment)
                _push_by_type!(staged, T, arc => chain)
                _push_by_type!(names, T, segment_name => arc)
            end
            # A leaf redirects to the row its flow is reported under -- its segment, which
            # for a plain branch is the leaf itself.
            for component in leaf_components(segment)
                _push_by_type!(
                    entry_names,
                    _get_segment_type(component),
                    get_name(component) => segment_name,
                )
            end
        end
    end
    return
end

# Names come from `_index_series!`, which sees the segment structure this map flattens away.
function _index_reverse_series!(staged::_VectorsByType, source, predicate)
    for (member, arc) in source
        _index_reverse_series_entry!(staged, member, arc, predicate)
    end
    return
end

function _index_reverse_series_entry!(
    staged::_VectorsByType,
    member::PSY.ACTransmission,
    arc::ARC_ENTRY,
    predicate::F,
) where {F}
    _entry_matches(member, predicate) || return
    _push_by_type!(staged, _get_segment_type(member), member => arc)
    return
end

_name_candidates(index::COMPONENT_NAME_INDEX, name::String) =
    get!(() -> Tuple{DataType, ARC_ENTRY}[], index, name)

"""
Component-name index for name-based matrix indexing (`get_branch_multiplier`), whose API takes
a bare name.

Built from the component-keyed maps rather than `name_to_arc`, which holds *entry* names: an
aggregate's entry name is the group's, not any component's.

A name is indexed only where `_branch_multiplier` can answer: the arc's entry or one of its
direct members. Chain members, standalone or grouped, are never indexed.
"""
function _build_component_name_index(
    nrd::NetworkReductionData,
    arcs::ARC_TABLE,
    predicate,
)
    index = COMPONENT_NAME_INDEX()
    for (arc, entry) in nrd.direct_branch_map
        _index_direct_name!(index, entry, arc, predicate)
    end
    for (member, arc) in nrd.reverse_parallel_branch_map
        _index_parallel_member_name!(index, arcs, member, arc, predicate)
    end
    return index
end

function _index_direct_name!(
    index::COMPONENT_NAME_INDEX,
    entry::PSY.ACTransmission,
    arc::ARC_ENTRY,
    predicate::F,
) where {F}
    _entry_matches(entry, predicate) || return
    push!(_name_candidates(index, get_name(entry)), (typeof(entry), arc))
    return
end

function _index_parallel_member_name!(
    index::COMPONENT_NAME_INDEX,
    arcs::ARC_TABLE,
    member::PSY.ACTransmission,
    arc::ARC_ENTRY,
    predicate::F,
) where {F}
    _entry_matches(member, predicate) || return
    # Skip a member whose arc has no row (see `_index_reverse!`).
    haskey(arcs, arc) || return
    _entry_carries(get_entry(arcs[arc]), member) || return
    push!(_name_candidates(index, get_name(member)), (typeof(member), arc))
    return
end

"""
Throws when an arc in `nrd`'s direct, parallel or series map has no row in `name_to_arc`. It
catches two silent losses: an aggregate with no leaf components (it registers under no type),
and an entry-name collision inside one type bucket (the later arc overwrites the earlier one).
Either leaves an arc that carries flow but that no component-type query can reach.
"""
function _validate_catalog_closure(nrd::NetworkReductionData, name_to_arc::NAME_TO_ARC)
    indexed = Set{Tuple{Int, Int}}()
    sizehint!(indexed, sum(length, values(name_to_arc); init = 0))
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
    BranchCatalog(nrd::NetworkReductionData)

The complete index over `nrd`.
"""
BranchCatalog(nrd::NetworkReductionData) = BranchCatalog(nrd, _keep_all)

"""
    BranchCatalog(nrd::NetworkReductionData, predicate)

Index over `nrd` holding only entries `predicate` accepts, where `predicate(T, component)`
returns whether a component of branch type `T` should be indexed. An aggregate is judged by
`_entry_matches`, which applies the predicate to every physical branch at its leaves.

Only an unfiltered catalog runs [`_validate_catalog_closure`](@ref); a filter drops arcs by
design.
"""
function BranchCatalog(nrd::NetworkReductionData, predicate)
    maps = BranchMapsByType()
    arcs = ARC_TABLE()
    names = _VectorsByType{Pair{String, ARC_ENTRY}}()
    entry_names = _VectorsByType{Pair{String, String}}()

    direct = _VectorsByType{Pair{ARC_ENTRY, PSY.ACTransmission}}()
    _index_forward!(direct, names, arcs, nrd.direct_branch_map, predicate,
        entry -> (_get_segment_type(entry),))
    _fill_buckets!(maps.direct_branch_map, direct,
        ((_, entry),) -> Dict{Tuple{Int, Int}, typeof(entry)}())

    reverse_direct = _VectorsByType{Pair{PSY.ACTransmission, ARC_ENTRY}}()
    _index_reverse!(reverse_direct, entry_names, arcs, nrd.reverse_direct_branch_map,
        predicate)
    _fill_buckets!(maps.reverse_direct_branch_map, reverse_direct,
        ((member, _),) -> Dict{typeof(member), Tuple{Int, Int}}())

    parallel = _VectorsByType{Pair{ARC_ENTRY, AbstractBranchesParallel}}()
    _index_forward!(parallel, names, arcs, nrd.parallel_branch_map, predicate,
        _get_concrete_types)
    # Value type is `AbstractBranchesParallel`: a per-type bucket holds either a
    # `BranchesParallel{T}` or a `MixedBranchesParallel` that includes a `T`.
    _fill_buckets!(maps.parallel_branch_map, parallel,
        _ -> Dict{Tuple{Int, Int}, AbstractBranchesParallel}())

    reverse_parallel = _VectorsByType{Pair{PSY.ACTransmission, ARC_ENTRY}}()
    _index_reverse!(reverse_parallel, entry_names, arcs, nrd.reverse_parallel_branch_map,
        predicate)
    _fill_buckets!(maps.reverse_parallel_branch_map, reverse_parallel,
        ((member, _),) -> Dict{typeof(member), Tuple{Int, Int}}())

    series = _VectorsByType{Pair{ARC_ENTRY, BranchesSeries}}()
    _index_series!(series, arcs, names, entry_names, nrd.series_branch_map, predicate)
    _fill_buckets!(maps.series_branch_map, series,
        _ -> Dict{Tuple{Int, Int}, BranchesSeries}())

    reverse_series = _VectorsByType{Pair{PSY.ACTransmission, ARC_ENTRY}}()
    _index_reverse_series!(reverse_series, nrd.reverse_series_branch_map, predicate)
    # Same key type as `reverse_series_branch_map` in `NetworkReductionData`.
    _fill_buckets!(maps.reverse_series_branch_map, reverse_series,
        _ -> Dict{PSY.ACTransmission, Tuple{Int, Int}}())

    name_to_arc = NAME_TO_ARC()
    _sorted_name_buckets!(name_to_arc, names)
    component_to_entry = COMPONENT_TO_ENTRY()
    for (T, pairs) in entry_names
        component_to_entry[T] = _fill_bucket!(Dict{String, String}(), pairs)
    end

    if _is_unfiltered(predicate)
        _validate_catalog_closure(nrd, name_to_arc)
    end

    return BranchCatalog(
        nrd,
        arcs,
        maps,
        name_to_arc,
        component_to_entry,
        _build_component_name_index(nrd, arcs, predicate),
    )
end
