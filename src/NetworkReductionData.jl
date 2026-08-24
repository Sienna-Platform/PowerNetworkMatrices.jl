@kwdef mutable struct BranchMapsByType
    direct_branch_map::Dict{DataType, Any} = Dict{DataType, Any}()
    reverse_direct_branch_map::Dict{DataType, Any} = Dict{DataType, Any}()
    parallel_branch_map::Dict{DataType, Any} = Dict{DataType, Any}()
    reverse_parallel_branch_map::Dict{DataType, Any} = Dict{DataType, Any}()
    series_branch_map::Dict{DataType, Any} = Dict{DataType, Any}()
    reverse_series_branch_map::Dict{DataType, Any} = Dict{DataType, Any}()
end

const _BRANCH_MAPS_BY_TYPE_FIELDS = fieldnames(BranchMapsByType)

function Base.iterate(b::BranchMapsByType, state = 1)
    state > length(_BRANCH_MAPS_BY_TYPE_FIELDS) && return nothing
    f = _BRANCH_MAPS_BY_TYPE_FIELDS[state]
    return (String(f) => getfield(b, f), state + 1)
end

Base.length(::BranchMapsByType) = length(_BRANCH_MAPS_BY_TYPE_FIELDS)

function Base.getindex(b::BranchMapsByType, key::String)
    return getfield(b, Symbol(key))
end

# `BranchCatalog` tags each entry with the map it came from as a `Symbol`, so a lookup keyed
# by that tag indexes without round-tripping through a string.
function Base.getindex(b::BranchMapsByType, key::Symbol)
    return getfield(b, key)
end

function Base.isempty(b::BranchMapsByType)
    for f in _BRANCH_MAPS_BY_TYPE_FIELDS
        isempty(getfield(b, f)) || return false
    end
    return true
end

function Base.empty!(b::BranchMapsByType)
    for f in _BRANCH_MAPS_BY_TYPE_FIELDS
        empty!(getfield(b, f))
    end
    return
end

function Base.:(==)(a::BranchMapsByType, b::BranchMapsByType)
    for f in _BRANCH_MAPS_BY_TYPE_FIELDS
        getfield(a, f) == getfield(b, f) || return false
    end
    return true
end

# Typed accessors for BranchMapsByType — function barriers that recover concrete types.
function get_typed_direct_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ACTransmission}
    return b.direct_branch_map[T]::Dict{Tuple{Int, Int}, T}
end

# `ThreeWindingTransformerCircuit` is non-parametric (one concrete parent type), so the
# per-type bucket is keyed by the parent transformer type and holds the wrapper. `T`
# (a concrete `PSY.ThreeWindingTransformer` subtype) selects that bucket.
function get_typed_direct_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ThreeWindingTransformer}
    return b.direct_branch_map[T]::Dict{Tuple{Int, Int}, ThreeWindingTransformerCircuit}
end

function get_typed_reverse_direct_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ACTransmission}
    return b.reverse_direct_branch_map[T]::Dict{T, Tuple{Int, Int}}
end

function get_typed_reverse_direct_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ThreeWindingTransformer}
    return b.reverse_direct_branch_map[T]::Dict{
        ThreeWindingTransformerCircuit,
        Tuple{Int, Int},
    }
end

# Per-type bucket is widened to `AbstractBranchesParallel` so that a
# `MixedBranchesParallel` group is reachable under each of its underlying
# branch types (e.g. both `parallel_branch_map[Line]` and
# `parallel_branch_map[MonitoredLine]` see the same group). Pure
# `BranchesParallel{T}` groups remain assignment-compatible.
function get_typed_parallel_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ACTransmission}
    return b.parallel_branch_map[T]::Dict{Tuple{Int, Int}, AbstractBranchesParallel}
end

function get_typed_reverse_parallel_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ACTransmission}
    return b.reverse_parallel_branch_map[T]::Dict{T, Tuple{Int, Int}}
end

function get_typed_series_branch_map(
    b::BranchMapsByType,
    ::Type{T},
) where {T <: PSY.ACTransmission}
    return b.series_branch_map[T]::Dict{Tuple{Int, Int}, BranchesSeries}
end

"""
    NetworkReductionData

Mutable struct containing all data mappings and metadata for network reduction operations.
This structure tracks how buses and branches are mapped, combined, or eliminated during
network reduction algorithms.

# Fields
- `irreducible_buses::Set{Int}`: Buses that cannot be reduced
- `bus_reduction_map::Dict{Int, Set{Int}}`: Maps retained buses to sets of eliminated buses
- `reverse_bus_search_map::Dict{Int, Int}`: Maps eliminated buses to their parent buses
- `direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission}`: One-to-one branch mappings, including each `ThreeWindingTransformerCircuit` on its star-point arc
- `reverse_direct_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}}`: Reverse direct mappings
- `parallel_branch_map::Dict{Tuple{Int, Int}, AbstractBranchesParallel}`: Parallel branch combinations (homogeneous `BranchesParallel{T}` or `MixedBranchesParallel`)
- `reverse_parallel_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}}`: Reverse parallel mappings
- `series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries}`: Series branch combinations
- `reverse_series_branch_map::Dict{Any, Tuple{Int, Int}}`: Reverse series mappings
- `removed_buses::Set{Int}`: Set of buses eliminated from the network
- `removed_arcs::Set{Tuple{Int, Int}}`: Set of arcs eliminated from the network
- `merged_bus_pairs::Dict{Int, Int}`: Maps removed bus number to surviving bus number for zero-impedance branch bus merges; drives row/column summation in `_merge_ybus_buses!`
- `removed_arc_to_surviving_bus::Dict{Tuple{Int, Int}, Int}`: Maps removed arcs to the connected surviving bus number (occurs for radial reduction or Ward reduction)
- `boundary_bus_to_removed_arcs::Dict{Int, Set{Tuple{Int, Int}}}`: Maps boundary buses to the set of removed arcs connected to them
- `added_admittance_map::Dict{Int, PSY.FixedAdmittance}`: Admittances added to buses during reduction
- `added_arc_impedance_map::Dict{Tuple{Int, Int}, PSY.GenericArcImpedance}`: New arcs created during reduction
- `reductions::ReductionContainer`: Container tracking applied reduction algorithms
"""
@kwdef mutable struct NetworkReductionData <:
                      IS.InfrastructureMatrices.AbstractInfrastructureNetworkReductionData
    irreducible_buses::Set{Int} = Set{Int}() # Buses that are not reduced in the network reduction
    bus_reduction_map::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}() # Maps reduced bus to the set of buses it was reduced to
    reverse_bus_search_map::Dict{Int, Int} = Dict{Int, Int}()
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission} =
        Dict{Tuple{Int, Int}, PSY.ACTransmission}()
    reverse_direct_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}} =
        Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    parallel_branch_map::Dict{Tuple{Int, Int}, AbstractBranchesParallel} =
        Dict{Tuple{Int, Int}, AbstractBranchesParallel}()
    reverse_parallel_branch_map::Dict{<:PSY.ACTransmission, Tuple{Int, Int}} =
        Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries} =
        Dict{Tuple{Int, Int}, BranchesSeries}()
    reverse_series_branch_map::Dict{<:PSY.ACTransmission, Tuple{Int, Int}} =
        Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    removed_buses::Set{Int} = Set{Int}()
    removed_arcs::Set{Tuple{Int, Int}} = Set{Tuple{Int, Int}}()
    merged_bus_pairs::Dict{Int, Int} = Dict{Int, Int}()
    removed_arc_to_surviving_bus::Dict{Tuple{Int, Int}, Int} = Dict{Tuple{Int, Int}, Int}()
    boundary_bus_to_removed_arcs::Dict{Int, Set{Tuple{Int, Int}}} =
        Dict{Int, Set{Tuple{Int, Int}}}()
    added_admittance_map::Dict{Int, PSY.FixedAdmittance} = Dict{Int, PSY.FixedAdmittance}()
    added_arc_impedance_map::Dict{Tuple{Int, Int}, PSY.GenericArcImpedance} =
        Dict{Tuple{Int, Int}, PSY.GenericArcImpedance}()
    reductions::ReductionContainer = ReductionContainer()
end


function get_name(device::T) where {T <: PSY.ACTransmission}
    return PSY.get_name(device)
end


_get_segment_components(x::T) where {T <: PSY.ACBranch} = [x]
_get_segment_components(x::AbstractBranchesParallel) = x.branches
_get_segment_type(::T) where {T <: PSY.ACBranch} = T
_get_segment_type(::BranchesParallel{T}) where {T <: PSY.ACTransmission} = T
_get_segment_type(::MixedBranchesParallel) = MixedBranchesParallel
# The 3W reduction maps are keyed by the parent transformer type
# (`PSY.ThreeWindingTransformer`), so 3W entries are looked up by the transformer type.
_get_segment_type(w::ThreeWindingTransformerCircuit) = get_transformer_type(w)

_get_concrete_types(x::T) where {T <: PSY.ACBranch} = [T]
_get_concrete_types(::BranchesParallel{T}) where {T <: PSY.ACTransmission} = [T]
# Bucket keys are always PSY component types, so a group of 3W windings is filed under the
# parent transformer type -- matching how a winding in the direct map is filed, and what the
# wrapper-keyed accessors redirect to. A group can hold windings of transformers with
# different concrete types, so every distinct parent is returned.
_get_concrete_types(bp::BranchesParallel{ThreeWindingTransformerCircuit}) =
    unique(get_transformer_type.(bp.branches))
# A heterogeneous group must be discoverable under each of its members' branch types so that
# downstream per-type iteration (e.g. the catalog's per-type buckets) can find it.
# Fan-out is by `_get_segment_type`, which maps a winding to its parent transformer and every
# other branch to its own type, keeping every bucket key a PSY component type.
_get_concrete_types(bp::MixedBranchesParallel) = unique(_get_segment_type.(bp.branches))

# Construct an empty per-slot dict for `BranchMapsByType.parallel_branch_map`.
# Value type is `AbstractBranchesParallel` so that the same per-type bucket can
# hold either a homogeneous `BranchesParallel{T}` or a `MixedBranchesParallel`
# that includes a branch of type `T`.
_empty_parallel_branch_map() = Dict{Tuple{Int, Int}, AbstractBranchesParallel}()

get_irreducible_buses(rb::NetworkReductionData) = rb.irreducible_buses
"""
    get_bus_reduction_map(rb::NetworkReductionData)

Get the bus reduction map from NetworkReductionData.

# Arguments
- `rb::NetworkReductionData`: The network reduction data

# Returns
- `Dict{Int, Set{Int}}`: Dictionary mapping retained buses to sets of removed buses
"""
get_bus_reduction_map(rb::NetworkReductionData) = rb.bus_reduction_map
get_reverse_bus_search_map(rb::NetworkReductionData) = rb.reverse_bus_search_map
get_direct_branch_map(rb::NetworkReductionData) = rb.direct_branch_map
get_reverse_direct_branch_map(rb::NetworkReductionData) = rb.reverse_direct_branch_map
get_parallel_branch_map(rb::NetworkReductionData) = rb.parallel_branch_map
get_reverse_parallel_branch_map(rb::NetworkReductionData) = rb.reverse_parallel_branch_map
get_series_branch_map(rb::NetworkReductionData) = rb.series_branch_map
get_reverse_series_branch_map(rb::NetworkReductionData) = rb.reverse_series_branch_map
get_removed_buses(rb::NetworkReductionData) = rb.removed_buses
get_removed_arcs(rb::NetworkReductionData) = rb.removed_arcs
get_removed_arc_to_surviving_bus(rb::NetworkReductionData) = rb.removed_arc_to_surviving_bus
get_added_admittance_map(rb::NetworkReductionData) = rb.added_admittance_map
get_added_arc_impedance_map(rb::NetworkReductionData) = rb.added_arc_impedance_map

"""
    get_reductions(rb::NetworkReductionData)

Get the reduction container from NetworkReductionData.

# Arguments
- `rb::NetworkReductionData`: The network reduction data

# Returns
- `ReductionContainer`: Container with the applied network reductions
"""
get_reductions(rb::NetworkReductionData) = rb.reductions

"""
    get_applied_reductions(rb::NetworkReductionData)

The reductions `rb` was built with, in the order `Ybus` must re-apply them to reproduce it:
zero-impedance first, then radial, degree-two and Ward (Ward must be last). Rebuild with
`Ybus(sys; network_reductions = get_applied_reductions(rb))`.
"""
function get_applied_reductions(rb::NetworkReductionData)
    c = get_reductions(rb)
    applied = NetworkReduction[]
    has_zero_impedance_reduction(c) && push!(applied, c.zero_impedance_reduction)
    has_radial_reduction(c) && push!(applied, c.radial_reduction)
    has_degree_two_reduction(c) && push!(applied, c.degree_two_reduction)
    has_ward_reduction(c) && push!(applied, c.ward_reduction)
    return applied
end





has_radial_reduction(rb::NetworkReductionData) = has_radial_reduction(rb.reductions)
has_degree_two_reduction(rb::NetworkReductionData) = has_degree_two_reduction(rb.reductions)
has_ward_reduction(rb::NetworkReductionData) = has_ward_reduction(rb.reductions)

function Base.isempty(rb::NetworkReductionData)
    for field in fieldnames(NetworkReductionData)
        if !isempty(getfield(rb, field))
            return false
        end
    end
    return true
end

function Base.empty!(rb::NetworkReductionData)
    for field in fieldnames(NetworkReductionData)
        empty!(getfield(rb, field))
    end
    return
end

"""
   get_retained_branches_names(network_reduction_data::NetworkReductionData)

Gets the branch names that are retained after network reduction. This method only returns the
branch names from non-three winding transformer branches that have a one-to-one correspondence with
arcs after the reduction. This does not include parallel branches or branches that have been reduced as
part of a series chain of degree two nodes.

# Arguments
- `network_reduction_data::NetworkReductionData`

# Returns
- `Vector{String}`: Vector of the retained branch names.
"""
function get_retained_branches_names(network_reduction_data::NetworkReductionData)
    return [
        PSY.get_name(branch) for
        branch in keys(network_reduction_data.reverse_direct_branch_map) if
        !_is_three_winding_circuit(branch)
    ]
end

_is_three_winding_circuit(::PSY.ACTransmission) = false
_is_three_winding_circuit(::ThreeWindingTransformerCircuit) = true

"""
   get_ac_transmission_types(network_reduction_data::NetworkReductionData)

Gets the concrete types of all AC transmission branches included in an instance of NetworkReductionData

# Arguments
- `network_reduction_data::NetworkReductionData`

# Returns
- `Set{DataType}`: Vector of the retained branch types.
"""
# A `ThreeWindingTransformerCircuit` reports the parent transformer type; every other branch
# reports its own concrete type.
_ac_transmission_type(x::PSY.ACTransmission) = typeof(x)
_ac_transmission_type(w::ThreeWindingTransformerCircuit) = get_transformer_type(w)
function get_ac_transmission_types(network_reduction_data::NetworkReductionData)
    direct_types = Set{DataType}(
        _ac_transmission_type.(keys(network_reduction_data.reverse_direct_branch_map)),
    )
    parallel_types =
        Set{DataType}(typeof.(keys(network_reduction_data.reverse_parallel_branch_map)))
    series_types =
        Set{DataType}(typeof.(keys(network_reduction_data.reverse_series_branch_map)))
    return union(direct_types, parallel_types, series_types)
end

##############################################################################
########################### Auxiliary functions ##############################
##############################################################################

function isequal(rb1::NetworkReductionData, rb2::NetworkReductionData)
    for field in fieldnames(NetworkReductionData)
        if getfield(rb1, field) != getfield(rb2, field)
            return false
        end
    end
    return true
end

"""
Interface to obtain the parent bus number of a reduced bus when radial branches are eliminated

# Arguments
- `rb::NetworkReduction`: NetworkReduction object
- `bus_number::Int`: Bus number of the reduced bus
"""
function get_mapped_bus_number(rb::NetworkReductionData, bus_number::Int)
    return get(rb.reverse_bus_search_map, bus_number, bus_number)
end

"""
Interface to obtain the parent bus number of a reduced bus when radial branches are eliminated

# Arguments
- `rb::NetworkReduction`: NetworkReduction object
- `bus::ACBus`: Reduced bus
"""
function get_mapped_bus_number(rb::NetworkReductionData, bus::PSY.ACBus)
    return get_mapped_bus_number(rb, PSY.get_number(bus))
end

"""
Interface to obtain the arc axis based on the network reduction data
"""
function get_arc_axis(nr::NetworkReductionData)
    direct_arcs = collect(keys(nr.direct_branch_map))
    parallel_arcs = collect(keys(nr.parallel_branch_map))
    series_arcs = collect(keys(nr.series_branch_map))
    additional_arcs = collect(keys(nr.added_arc_impedance_map))
    arc_ax = unique(vcat(direct_arcs, parallel_arcs, series_arcs, additional_arcs))
    return arc_ax
end

function is_arc_in_series_map(nr::NetworkReductionData, arc::Tuple{Int64, Int64})
    return haskey(nr.series_branch_map, arc)
end

function get_mapped_series_branch(nr::NetworkReductionData, arc::Tuple{Int64, Int64})
    if is_arc_in_series_map(nr, arc)
        return nr.series_branch_map[arc]
    else
        error("Arc $arc not found in series branch map")
    end
    return
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, nrd::NetworkReductionData)
    println("Network Reduction Summary:")
    println("\tNumber of remapped buses: $(length(nrd.reverse_bus_search_map))")
    println("\tNumber of direct branch mappings: $(length(nrd.direct_branch_map))")
    println(
        "\tNumber of parallel arcs (number of branches): $(length(nrd.parallel_branch_map)) ($(length(nrd.reverse_parallel_branch_map)))",
    )
    println(
        "\tNumber of series arcs (number of branches): $(length(nrd.series_branch_map)) ($(length(nrd.reverse_series_branch_map)))",
    )
    println("\tNumber of removed buses: $(length(nrd.removed_buses))")
    println("\tNumber of removed arcs: $(length(nrd.removed_arcs))")
    println("\tNumber of added arcs: $(length(nrd.added_arc_impedance_map))")
    println("\tNumber of added admittances: $(length(nrd.added_admittance_map))")
end
