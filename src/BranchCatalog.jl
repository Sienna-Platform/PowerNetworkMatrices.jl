const ARC_ENTRY = Tuple{Tuple{Int, Int}, Symbol}
const NAME_TO_ARC = Dict{DataType, DataStructures.SortedDict{String, ARC_ENTRY}}
const COMPONENT_TO_ENTRY = Dict{DataType, Dict{String, String}}
const COMPONENT_NAME_INDEX = Dict{String, Vector{Tuple{DataType, Tuple{Int, Int}, Symbol}}}

"""
    BranchCatalog

An immutable, per-branch-type index over the branch maps a network reduction produced, built
eagerly when a matrix is constructed. Wraps the [`NetworkReductionData`](@ref) it indexes, so
a consumer holding a catalog can always reach the reduction it describes.

Every matrix carries one, reachable with `get_branch_catalog`.
`BranchCatalog(nrd, predicate)` derives a restricted catalog over the same reduction, for a
consumer that models only some branches.

# Fields
- `network_reduction_data::NetworkReductionData`: the reduction this indexes
- `maps_by_type::BranchMapsByType`: the six reduction maps re-bucketed by branch type
- `name_to_arc`: per type, entry name => (arc, which map holds it)
- `component_to_entry_name`: per type, component name => name of the entry representing it
- `component_name_index`: component name => every candidate claiming it, across all types.
  Names are unique only per type, so a name can have more than one candidate; name-based
  lookup reports the ambiguity rather than picking one.
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
Entries for branch type `T`. An absent `T` yields an empty map rather than throwing: a type
is legitimately missing when every branch of it was absorbed by a reduction.
"""
get_name_to_arc_map(c::BranchCatalog, ::Type{T}) where {T <: PSY.ACTransmission} =
    get(c.name_to_arc, T, DataStructures.SortedDict{String, ARC_ENTRY}())

# 3W winding entries are filed under the parent transformer type (see `_get_segment_type`),
# so a lookup keyed by the non-parametric wrapper translates to that key.
get_name_to_arc_map(c::BranchCatalog, ::Type{ThreeWindingTransformerCircuit}) =
    get_name_to_arc_map(c, PSY.ThreeWindingTransformer)

get_component_to_reduction_name_map(
    c::BranchCatalog,
    ::Type{T},
) where {T <: PSY.ACTransmission} =
    get(c.component_to_entry_name, T, Dict{String, String}())

get_component_to_reduction_name_map(
    c::BranchCatalog,
    ::Type{ThreeWindingTransformerCircuit},
) = get_component_to_reduction_name_map(c, PSY.ThreeWindingTransformer)

Base.isempty(c::BranchCatalog) =
    isempty(c.maps_by_type) && isempty(c.name_to_arc) &&
    isempty(c.component_to_entry_name) && isempty(c.component_name_index)

##############################################################################
############################ Entry matching ##################################
##############################################################################

# Every entry passes when no predicate is supplied.
_keep_all(::Type, ::Any) = true

# `_keep_all` is the unfiltered path; a group is only "filtered more than intended" when a
# real predicate is applied.
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

"""
Index a forward (arc-keyed) reduction map. `bucket_types(entry)` yields the type buckets the
entry is filed under, and `empty_bucket(entry)` the per-bucket dict to create on first use.

The direct map files one bucket keyed by `_get_segment_type` (the parent transformer type for
a 3W winding) holding the concrete entry type. The parallel map files one bucket per member
type, each widened to `AbstractBranchesParallel` so a `MixedBranchesParallel` is reachable
under every member type it contains.
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
            get!(() -> empty_bucket(entry), dest, T)[arc] = entry
            _bucket_name_to_arc(name_to_arc, T)[get_name(entry)] = (arc, kind)
        end
    end
    return
end

"""
Index a reverse (entry-keyed) reduction map. Each member also records which entry represents
it, so a component absorbed into an aggregate can be redirected to the entry carrying its
flow.

The bucket *key* is `_get_segment_type(member)` — a PSY component type — while the bucket's
key *type* is `typeof(member)`. For a 3W winding those differ: the bucket is filed under the
parent transformer type but holds `ThreeWindingTransformerCircuit` keys.
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
        get!(() -> Dict{typeof(member), Tuple{Int, Int}}(), dest, T)[member] = arc
        _bucket_entry_names(component_to_entry, T)[get_name(member)] =
            get_name(forward[arc])
    end
    return
end

"""
Index the series map. A chain is filed under every type appearing anywhere in it, because a
caller iterating one branch type must find every chain that type participates in. Each of the
chain's segments contributes its own name to `name_to_arc`, and every physical component
inside a segment redirects to that segment's name.
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
                get!(() -> Dict{Tuple{Int, Int}, BranchesSeries}(), dest, T)[arc] = chain
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

# The reverse series map records no names: a chain member's redirect is written by
# `_index_series!`, which sees the segment structure this map has flattened away.
function _index_reverse_series!(dest::Dict{DataType, Any}, source, predicate)
    for (member, arc) in source
        _entry_matches(member, predicate) || continue
        get!(
            () -> Dict{PSY.ACTransmission, Tuple{Int, Int}}(),
            dest,
            _get_segment_type(member),
        )[member] = arc
    end
    return
end

"""
Type-agnostic component-name index, for name-based matrix indexing (`get_branch_multiplier`),
whose API takes a bare name.

Built from the *component*-keyed maps, not from `name_to_arc`: `name_to_arc` holds *entry*
names, and an aggregate's entry name is the group's, not any component's. A branch owning its
arc one-to-one is indexed from `direct_branch_map`; a parallel-group member from
`reverse_parallel_branch_map`. Series-chain members are deliberately absent — name-based
matrix indexing has never resolved them.
"""
_name_candidates(index::COMPONENT_NAME_INDEX, name::String) =
    get!(() -> Tuple{DataType, Tuple{Int, Int}, Symbol}[], index, name)

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
    BranchCatalog(nrd::NetworkReductionData)

The complete index over `nrd` — every entry in every reduction map.
"""
BranchCatalog(nrd::NetworkReductionData) = BranchCatalog(nrd, _keep_all)

"""
    BranchCatalog(nrd::NetworkReductionData, predicate)

Index over `nrd` holding only entries `predicate` accepts. `predicate(T, component)` returns
whether a component of branch type `T` should be indexed; an aggregate is judged by
`_entry_matches`, which applies the predicate to its members.
"""
function BranchCatalog(nrd::NetworkReductionData, predicate)
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

    return BranchCatalog(
        nrd,
        maps,
        name_to_arc,
        component_to_entry,
        _build_component_name_index(nrd, predicate),
    )
end
