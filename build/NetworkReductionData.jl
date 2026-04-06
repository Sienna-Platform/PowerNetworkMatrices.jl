"""
    NetworkReductionData

Mutable struct containing all data mappings and metadata for network reduction operations.
This structure tracks how buses and branches are mapped, combined, or eliminated during
network reduction algorithms.

# Fields
- `irreducible_buses::Set{Int}`: Buses that cannot be reduced
- `bus_reduction_map::Dict{Int, Set{Int}}`: Maps retained buses to sets of eliminated buses
- `reverse_bus_search_map::Dict{Int, Int}`: Maps eliminated buses to their parent buses
- `direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission}`: One-to-one branch mappings
- `reverse_direct_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}}`: Reverse direct mappings
- `parallel_branch_map::Dict{Tuple{Int, Int}, BranchesParallel}`: Parallel branch combinations
- `reverse_parallel_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}}`: Reverse parallel mappings
- `series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries}`: Series branch combinations
- `reverse_series_branch_map::Dict{Any, Tuple{Int, Int}}`: Reverse series mappings
- `transformer3W_map::Dict{Tuple{Int, Int}, ThreeWindingTransformerWinding}`: Three-winding transformer mappings
- `reverse_transformer3W_map::Dict{ThreeWindingTransformerWinding, Tuple{Int, Int}}`: Reverse transformer mappings
- `removed_buses::Set{Int}`: Set of buses eliminated from the network
- `removed_arcs::Set{Tuple{Int, Int}}`: Set of arcs eliminated from the network
- `radial_arc_to_surviving_bus::Dict{Tuple{Int, Int}, Int}`: Maps final arcs in radial reduction chains to their surviving bus numbers
- `added_admittance_map::Dict{Int, Complex{Float32}}`: Admittances added to buses during reduction
- `added_branch_map::Dict{Tuple{Int, Int}, Complex{Float32}}`: New branches created during reduction
- `all_branch_maps_by_type::Dict{String, Any}`: Branch mappings organized by component type
- `reductions::ReductionContainer`: Container tracking applied reduction algorithms
- `name_to_arc_map::Dict{Type, DataStructures.SortedDict{String, Tuple{Tuple{Int, Int}, String}}}`: Lazily filled with the call to [`populate_branch_maps_by_type!`](@ref), maps string names to their corresponding arcs and the map where the arc can be found. Used in optimization models or power flow reporting after reductions are applied. It is possible to have repeated arcs for some names if case of serial or parallel combinations.
- `component_to_reduction_name_map::Dict{Type, Dict{String, String}}`: Lazily filled with the call to [`populate_branch_maps_by_type!`](@ref), maps component names to the names of the reduction entries used in name_to_arc_map. Used in optimization models for connecting component attributes (e.g. outages) to network reduction entries.
- `filters_applied::Dict{Type, Function}`: Filters applied when populating branch maps by type
- `direct_branch_name_map::Dict{String, Tuple{Int, Int}}`: Lazily filled, maps branch names to their corresponding arc tuples for direct branches
"""
@kwdef mutable struct NetworkReductionData
    irreducible_buses::Set{Int} = Set{Int}() # Buses that are not reduced in the network reduction
    bus_reduction_map::Dict{Int, Set{Int}} = Dict{Int, Set{Int}}() # Maps reduced bus to the set of buses it was reduced to
    reverse_bus_search_map::Dict{Int, Int} = Dict{Int, Int}()
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission} =
        Dict{Tuple{Int, Int}, PSY.ACTransmission}()
    reverse_direct_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}} =
        Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    parallel_branch_map::Dict{Tuple{Int, Int}, BranchesParallel} =
        Dict{Tuple{Int, Int}, BranchesParallel}()
    reverse_parallel_branch_map::Dict{<:PSY.ACTransmission, Tuple{Int, Int}} =
        Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries} =
        Dict{Tuple{Int, Int}, BranchesSeries}()
    reverse_series_branch_map::Dict{<:PSY.ACTransmission, Tuple{Int, Int}} =
        Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    transformer3W_map::Dict{
        Tuple{Int, Int},
        ThreeWindingTransformerWinding,
    } = Dict{Tuple{Int, Int}, ThreeWindingTransformerWinding}()
    reverse_transformer3W_map::Dict{
        ThreeWindingTransformerWinding,
        Tuple{Int, Int},
    } = Dict{ThreeWindingTransformerWinding, Tuple{Int, Int}}()
    removed_buses::Set{Int} = Set{Int}()
    removed_arcs::Set{Tuple{Int, Int}} = Set{Tuple{Int, Int}}()
    radial_arc_to_surviving_bus::Dict{Tuple{Int, Int}, Int} = Dict{Tuple{Int, Int}, Int}()
    added_admittance_map::Dict{Int, Complex{Float32}} = Dict{Int, Complex{Float32}}()
    added_branch_map::Dict{Tuple{Int, Int}, Complex{Float32}} =
        Dict{Tuple{Int, Int}, Complex{Float32}}()
    all_branch_maps_by_type::Dict{String, Any} = Dict{String, Any}()
    reductions::ReductionContainer = ReductionContainer()
    name_to_arc_map::Dict{
        Type,
        DataStructures.SortedDict{String, Tuple{Tuple{Int, Int}, String}},
    } =
        Dict{Type, DataStructures.SortedDict{String, Tuple{Tuple{Int, Int}, String}}}()
    component_to_reduction_name_map::Dict{
        Type,
        Dict{String, String}} = Dict{Type, Dict{String, String}}()
    filters_applied = Dict{Type, Function}() #Filters applied when populating branch maps by type
    direct_branch_name_map::Dict{String, Tuple{Int, Int}} =
        Dict{String, Tuple{Int, Int}}()
end

function add_to_map(device::T, filters::Dict) where {T <: PSY.ACTransmission}
    if !haskey(filters, T)
        return true
    end
    return filters[T](device)
end

function get_name(device::T) where {T <: PSY.ACTransmission}
    return PSY.get_name(device)
end

function populate_direct_branch_name_map!(nr::NetworkReductionData)
    for (arc_tuple, branch) in nr.direct_branch_map
        branch_name = get_name(branch)
        nr.direct_branch_name_map[branch_name] = arc_tuple
    end
end

"""
    populate_branch_maps_by_type!(nrd::NetworkReductionData, filters = Dict())

Populate the branch maps organized by component type within the NetworkReductionData structure.

This function processes various types of branch mappings (direct, parallel, series, and 3-winding transformers)
and organizes them by their component types. It applies optional filters to determine which branches should
be included in the type-organized maps.

# Arguments
- `nrd::NetworkReductionData`: The network reduction data structure to populate
- `filters`: Optional dictionary of filters to apply when determining which branches to include (default: empty Dict)

# Details
The function creates and populates the following map types organized by component type:
- `direct_branch_map`: Direct branch connections between buses
- `reverse_direct_branch_map`: Reverse lookup for direct branches
- `parallel_branch_map`: Parallel branch connections between the same bus pair
- `reverse_parallel_branch_map`: Reverse lookup for parallel branches
- `series_branch_map`: Series branch connections (chains of branches)
- `reverse_series_branch_map`: Reverse lookup for series branches
- `transformer3W_map`: Three-winding transformer connections
- `reverse_transformer3W_map`: Reverse lookup for three-winding transformers

The function also populates the `name_to_arc_map` to provide name-based lookups for branches
and stores the applied filters in `nrd.filters_applied`.

# Modifies
- `nrd.all_branch_maps_by_type`: Populated with type-organized branch maps
- `nrd.name_to_arc_map`: Updated with name-to-arc mappings
- `nrd.filters_applied`: Set to the provided filters

# Returns
- `nothing`: This function modifies the input structure in-place
"""
function populate_branch_maps_by_type!(nrd::NetworkReductionData, filters = Dict())
    all_branch_maps_by_type = Dict(
        "direct_branch_map" =>
            Dict{Type{<:PSY.ACTransmission}, Dict{Tuple{Int, Int}, PSY.ACTransmission}}(),
        "reverse_direct_branch_map" =>
            Dict{Type{<:PSY.ACTransmission}, Dict{PSY.ACTransmission, Tuple{Int, Int}}}(),
        "parallel_branch_map" =>
            Dict{
                Type{<:PSY.ACTransmission},
                Dict{Tuple{Int, Int}, BranchesParallel},
            }(),
        "reverse_parallel_branch_map" =>
            Dict{Type{<:PSY.ACTransmission}, Dict{PSY.ACTransmission, Tuple{Int, Int}}}(),
        "series_branch_map" =>
            Dict{Type{<:PSY.ACTransmission}, Dict{Tuple{Int, Int}, BranchesSeries}}(),
        "reverse_series_branch_map" =>
            Dict{Type{<:PSY.ACTransmission}, Dict{PSY.ACTransmission, Tuple{Int, Int}}}(),
        "transformer3W_map" => Dict{
            Type{<:PSY.ThreeWindingTransformer},
            Dict{
                Tuple{Int, Int},
                ThreeWindingTransformerWinding,
            },
        }(),
        "reverse_transformer3W_map" => Dict{
            Type{<:PSY.ThreeWindingTransformer},
            Dict{
                ThreeWindingTransformerWinding,
                Tuple{Int, Int},
            },
        }())

    for (k, v) in nrd.direct_branch_map
        if add_to_map(v, filters)
            map_by_type = get!(
                all_branch_maps_by_type["direct_branch_map"],
                _get_segment_type(v),
                Dict{Tuple{Int, Int}, _get_segment_type(v)}(),
            )
            map_by_type[k] = v
            name_to_arc_map = get!(
                nrd.name_to_arc_map,
                _get_segment_type(v),
                DataStructures.SortedDict{String, Tuple{Int, Int}}(),
            )
            name_to_arc_map[get_name(v)] = (k, "direct_branch_map")
        end
    end
    for (k, v) in nrd.reverse_direct_branch_map
        if add_to_map(k, filters)
            map_by_type = get!(
                all_branch_maps_by_type["reverse_direct_branch_map"],
                _get_segment_type(k),
                Dict{_get_segment_type(k), Tuple{Int, Int}}(),
            )
            map_by_type[k] = v
            component_name_map = get!(
                nrd.component_to_reduction_name_map,
                _get_segment_type(k),
                Dict{String, String}(),
            )
            component_name_map[get_name(k)] = get_name(nrd.direct_branch_map[v])
        end
    end
    for (k, v) in nrd.parallel_branch_map
        if add_to_map(v, filters)
            map_by_type = get!(
                all_branch_maps_by_type["parallel_branch_map"],
                _get_segment_type(v),
                Dict{Tuple{Int, Int}, BranchesParallel{_get_segment_type(v)}}(),
            )
            map_by_type[k] = v
            name_to_arc_map = get!(
                nrd.name_to_arc_map,
                _get_segment_type(v),
                DataStructures.SortedDict{String, Tuple{Int, Int}}(),
            )
            name_to_arc_map[get_name(v)] = (k, "parallel_branch_map")
        end
    end
    for (k, v) in nrd.reverse_parallel_branch_map
        if add_to_map(k, filters)
            map_by_type = get!(
                all_branch_maps_by_type["reverse_parallel_branch_map"],
                _get_segment_type(k),
                Dict{BranchesParallel{_get_segment_type(k)}, Tuple{Int, Int}}(),
            )
            map_by_type[k] = v
            component_name_map = get!(
                nrd.component_to_reduction_name_map,
                _get_segment_type(k),
                Dict{String, String}(),
            )
            component_name_map[get_name(k)] = get_name(nrd.parallel_branch_map[v])
        end
    end
    for (k, v) in nrd.series_branch_map
        #Repeated entry for each type in series chain
        if add_to_map(v, filters)
            for segment in v
                map_by_type = get!(
                    all_branch_maps_by_type["series_branch_map"],
                    _get_segment_type(segment),
                    Dict{Tuple{Int, Int}, BranchesSeries}(),
                )
                map_by_type[k] = v

                name_to_arc_map = get!(
                    nrd.name_to_arc_map,
                    _get_segment_type(segment),
                    DataStructures.SortedDict{String, Tuple{Int, Int}}(),
                )
                name_to_arc_map[get_name(segment)] = (k, "series_branch_map")
                component_name_map = get!(
                    nrd.component_to_reduction_name_map,
                    _get_segment_type(segment),
                    Dict{String, String}(),
                )
                for x in _get_segment_components(segment)
                    component_name_map[get_name(x)] = get_name(segment)
                end
            end
        end
    end
    for (k, v) in nrd.reverse_series_branch_map
        if add_to_map(k, filters)
            map_by_type = get!(
                all_branch_maps_by_type["reverse_series_branch_map"],
                _get_segment_type(k),
                # Dict can be indexed by individual branches or BranchesParallel
                Dict{PSY.ACTransmission, Tuple{Int, Int}}(),
            )
            map_by_type[k] = v
        end
    end
    for (k, v) in nrd.transformer3W_map
        if add_to_map(v, filters)
            map_by_type = get!(
                all_branch_maps_by_type["transformer3W_map"],
                _get_segment_type(v),
                Dict{Tuple{Int, Int}, Vector{eltype(v)}}(),
            )
            map_by_type[k] = v

            name_to_arc_map = get!(
                nrd.name_to_arc_map,
                _get_segment_type(v),
                DataStructures.SortedDict{String, Tuple{Int, Int}}(),
            )
            name_to_arc_map[get_name(v)] = (k, "transformer3W_map")
        end
    end
    for (k, v) in nrd.reverse_transformer3W_map
        if add_to_map(k, filters)
            map_by_type = get!(
                all_branch_maps_by_type["reverse_transformer3W_map"],
                _get_segment_type(k),
                Dict{Tuple{Int, Int}, Vector{eltype(v)}}(),
            )
            map_by_type[k] = v
            component_name_map = get!(
                nrd.component_to_reduction_name_map,
                _get_segment_type(k),
                Dict{String, String}(),
            )
            component_name_map[get_name(k)] = get_name(nrd.transformer3W_map[v])
        end
    end
    populate_direct_branch_name_map!(nrd)
    nrd.all_branch_maps_by_type = all_branch_maps_by_type
    nrd.filters_applied = filters
    return
end

_get_segment_components(x::T) where {T <: PSY.ACBranch} = [x]
_get_segment_components(x::BranchesParallel{T}) where {T <: PSY.ACTransmission} = x.branches
_get_segment_type(::T) where {T <: PSY.ACBranch} = T
_get_segment_type(::BranchesParallel{T}) where {T <: PSY.ACTransmission} = T
_get_segment_type(
    ::ThreeWindingTransformerWinding{T},
) where {T <: PSY.ThreeWindingTransformer} = T

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
get_transformer3W_map(rb::NetworkReductionData) = rb.transformer3W_map
get_reverse_transformer3W_map(rb::NetworkReductionData) = rb.reverse_transformer3W_map
get_removed_buses(rb::NetworkReductionData) = rb.removed_buses
get_removed_arcs(rb::NetworkReductionData) = rb.removed_arcs
get_radial_arc_to_surviving_bus(rb::NetworkReductionData) = rb.radial_arc_to_surviving_bus
get_added_admittance_map(rb::NetworkReductionData) = rb.added_admittance_map
get_added_branch_map(rb::NetworkReductionData) = rb.added_branch_map
get_all_branch_maps_by_type(rb::NetworkReductionData) = rb.all_branch_maps_by_type

"""
    get_reductions(rb::NetworkReductionData)

Get the reduction container from NetworkReductionData.

# Arguments
- `rb::NetworkReductionData`: The network reduction data

# Returns
- `ReductionContainer`: Container with the applied network reductions
"""
get_reductions(rb::NetworkReductionData) = rb.reductions

get_component_to_reduction_name_map(rb::NetworkReductionData) =
    rb.component_to_reduction_name_map

get_component_to_reduction_name_map(
    rb::NetworkReductionData,
    ::Type{T},
) where {T <: PSY.ACTransmission} =
    rb.component_to_reduction_name_map[T]
get_component_to_reduction_name_map(
    rb::NetworkReductionData,
    ::Type{ThreeWindingTransformerWinding{T}},
) where {T <: PSY.ThreeWindingTransformer} = rb.component_to_reduction_name_map[T]

get_name_to_arc_maps(rb::NetworkReductionData) = rb.name_to_arc_map

get_name_to_arc_map(rb::NetworkReductionData, ::Type{T}) where {T <: PSY.ACTransmission} =
    rb.name_to_arc_map[T]
get_name_to_arc_map(
    rb::NetworkReductionData,
    ::Type{ThreeWindingTransformerWinding{T}},
) where {T <: PSY.ThreeWindingTransformer} = rb.name_to_arc_map[T]

has_radial_reduction(rb::NetworkReductionData) = has_radial_reduction(rb.reductions)
has_degree_two_reduction(rb::NetworkReductionData) = has_degree_two_reduction(rb.reductions)
has_ward_reduction(rb::NetworkReductionData) = has_ward_reduction(rb.reductions)
has_filtered_branches(rb::NetworkReductionData) = !isempty(rb.filters_applied)

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
        branch in keys(network_reduction_data.reverse_direct_branch_map)
    ]
end

"""
   get_ac_transmission_types(network_reduction_data::NetworkReductionData)

Gets the concrete types of all AC transmission branches included in an instance of NetworkReductionData

# Arguments
- `network_reduction_data::NetworkReductionData`

# Returns
- `Set{DataType}`: Vector of the retained branch types.
"""
function get_ac_transmission_types(network_reduction_data::NetworkReductionData)
    direct_types =
        Set(typeof.(keys(network_reduction_data.reverse_direct_branch_map)))
    parallel_types =
        Set{DataType}(typeof.(keys(network_reduction_data.reverse_parallel_branch_map)))
    series_types =
        Set{DataType}(typeof.(keys(network_reduction_data.reverse_series_branch_map)))
    transformer_3W_types =
        Set{DataType}(
            get_transformer_type.(keys(network_reduction_data.reverse_transformer3W_map)),
        )
    return union(direct_types, parallel_types, series_types, transformer_3W_types)
end

##############################################################################
########################### Auxiliary functions ##############################
##############################################################################

function isequal(
    rb1::NetworkReductionData,
    rb2::NetworkReductionData,
)
    for field in fieldnames(NetworkReductionData)
        # direct_branch_name_map is populated when indexing into matrices with branch names
        # this should not prevent using matrices for downstream computations (e.g. LODF(A, BA, ABA))
        field == :direct_branch_name_map && continue
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
    transformer_arcs = collect(keys(nr.transformer3W_map))
    additional_arcs = collect(keys(nr.added_branch_map))
    arc_ax = unique(
        vcat(direct_arcs, parallel_arcs, series_arcs, transformer_arcs, additional_arcs),
    )
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
    println("\tNumber of 3WT winding arcs:$(length(nrd.transformer3W_map))")
    println("\tNumber of removed buses: $(length(nrd.removed_buses))")
    println("\tNumber of removed arcs: $(length(nrd.removed_arcs))")
    println("\tNumber of added branches: $(length(nrd.added_branch_map))")
    println("\tNumber of added admittances: $(length(nrd.added_admittance_map))")
end
