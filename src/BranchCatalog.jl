const ARC_ENTRY = Tuple{Tuple{Int, Int}, Symbol}
const NAME_TO_ARC = Dict{DataType, DataStructures.SortedDict{String, ARC_ENTRY}}
const COMPONENT_TO_ENTRY = Dict{DataType, Dict{String, String}}
const COMPONENT_NAME_INDEX = Dict{String, Vector{Tuple{DataType, Tuple{Int, Int}, Symbol}}}

# Shared empty-map sentinels returned on a miss; callers must never mutate these.
const EMPTY_NAME_TO_ARC_MAP = DataStructures.SortedDict{String, ARC_ENTRY}()
const EMPTY_COMPONENT_TO_ENTRY_MAP = Dict{String, String}()

"""
    BranchCatalog

An immutable, per-branch-type index over the branch maps a network reduction produced. Every
matrix carries one, reachable with `get_branch_catalog`.

# Fields
- `network_reduction_data::NetworkReductionData`: the reduction this indexes
- `maps_by_type::BranchMapsByType`: the six reduction maps re-bucketed by branch type
- `name_to_arc`: per type, entry name => (arc, which map holds it)
- `component_to_entry_name`: per type, component name => name of the entry representing it
- `component_name_index`: component name => every candidate claiming it. Names are unique
  only per type, so a name can have several; lookup reports the ambiguity rather than
  picking one.
"""
struct BranchCatalog
    network_reduction_data::NetworkReductionData
    maps_by_type::BranchMapsByType
    name_to_arc::NAME_TO_ARC
    component_to_entry_name::COMPONENT_TO_ENTRY
    component_name_index::COMPONENT_NAME_INDEX
end

get_network_reduction_data(c::BranchCatalog) = c.network_reduction_data
get_all_branch_maps_by_type(c::BranchCatalog) = c.maps_by_type
get_name_to_arc_maps(c::BranchCatalog) = c.name_to_arc
get_component_to_reduction_name_map(c::BranchCatalog) = c.component_to_entry_name
get_component_name_index(c::BranchCatalog) = c.component_name_index

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

function _warn_mixed_group(kind::String, branches)
    @warn "$kind contains mixed branch types, filters might be applied to more " *
          "components than intended. Use Logging.Debug for additional information."
    @debug "$kind branch types: $(typeof.(branches))"
    @debug "$kind branch names: $(PSY.get_name.(branches))"
    return
end

_entry_matches(device::T, predicate) where {T <: PSY.ACTransmission} =
    predicate(T, device)

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
Index a forward (arc-keyed) reduction map under the buckets `bucket_types(entry)` names,
creating each with `empty_bucket(entry)`.

The parallel map's buckets are widened to `AbstractBranchesParallel` so a
`MixedBranchesParallel` is reachable under every member type it contains.
"""
function _index_forward!(
    dest::Dict{DataType, Any},
    name_to_arc::NAME_TO_ARC,
    source,
    kind::Symbol,
    predicate,
    bucket_types,
    empty_bucket,
)
    for (arc, entry) in source
        _entry_matches(entry, predicate) || continue
        for T in bucket_types(entry)
            _store!(get!(() -> empty_bucket(entry), dest, T), arc, entry)
            _bucket_name_to_arc(name_to_arc, T)[get_name(entry)] = (arc, kind)
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
    component_to_entry::COMPONENT_TO_ENTRY,
    source,
    forward,
    predicate,
)
    for (member, arc) in source
        _entry_matches(member, predicate) || continue
        T = _get_segment_type(member)
        _store!(get!(() -> Dict{typeof(member), Tuple{Int, Int}}(), dest, T), member, arc)
        _bucket_entry_names(component_to_entry, T)[get_name(member)] =
            get_name(forward[arc])
    end
    return
end

"""
Index the series map. A chain is filed under every type appearing anywhere in it, so a caller
iterating one branch type finds every chain that type participates in.
"""
function _index_series!(
    dest::Dict{DataType, Any},
    name_to_arc::NAME_TO_ARC,
    component_to_entry::COMPONENT_TO_ENTRY,
    source,
    predicate,
)
    for (arc, chain) in source
        _entry_matches(chain, predicate) || continue
        for segment in chain
            for T in _get_concrete_types(segment)
                _store!(
                    get!(() -> Dict{Tuple{Int, Int}, BranchesSeries}(), dest, T),
                    arc,
                    chain,
                )
                _bucket_name_to_arc(name_to_arc, T)[get_name(segment)] =
                    (arc, :series_branch_map)
                names = _bucket_entry_names(component_to_entry, T)
                for component in _get_segment_components(segment)
                    names[get_name(component)] = get_name(segment)
                end
            end
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
    get!(() -> Tuple{DataType, Tuple{Int, Int}, Symbol}[], index, name)

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
            (typeof(entry), arc, :direct_branch_map),
        )
    end
    for (member, arc) in nrd.reverse_parallel_branch_map
        _entry_matches(member, predicate) || continue
        push!(
            _name_candidates(index, get_name(member)),
            (typeof(member), arc, :parallel_branch_map),
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
        for (arc, _) in values(by_name)
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
    name_to_arc = NAME_TO_ARC()
    component_to_entry = COMPONENT_TO_ENTRY()

    _index_forward!(
        maps.direct_branch_map, name_to_arc, nrd.direct_branch_map,
        :direct_branch_map, predicate,
        entry -> (_get_segment_type(entry),),
        entry -> Dict{Tuple{Int, Int}, typeof(entry)}(),
    )
    _index_reverse!(
        maps.reverse_direct_branch_map, component_to_entry,
        nrd.reverse_direct_branch_map, nrd.direct_branch_map, predicate,
    )

    _index_forward!(
        maps.parallel_branch_map, name_to_arc, nrd.parallel_branch_map,
        :parallel_branch_map, predicate,
        _get_concrete_types,
        _ -> _empty_parallel_branch_map(),
    )
    _index_reverse!(
        maps.reverse_parallel_branch_map, component_to_entry,
        nrd.reverse_parallel_branch_map, nrd.parallel_branch_map, predicate,
    )

    _index_series!(
        maps.series_branch_map, name_to_arc, component_to_entry,
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
        maps,
        name_to_arc,
        component_to_entry,
        _build_component_name_index(nrd, predicate),
    )
end
