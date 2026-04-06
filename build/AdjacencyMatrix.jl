
"""
    AdjacencyMatrix{Ax, L} <: PowerNetworkMatrix{Int8}

An N × N adjacency matrix representing the connectivity structure of a power system with N buses.
This matrix describes the directed connectivity between buses, where non-zero entries indicate
electrical connections through transmission lines, transformers, or other network elements.

The matrix is indexed using bus numbers, which do not need to be sequential. Each element
`A[i,j]` is non-zero if there is a direct electrical connection between bus `i` and bus `j`.
Diagonal elements are typically zero since self-loops are not meaningful in power network topology.

# Fields
- `data::SparseArrays.SparseMatrixCSC{Int8, Int}`: The sparse adjacency matrix storing
  connectivity information as Int8 values (zero for no connection, non-zero for connection)
- `axes::Ax`: Tuple containing the axis labels for both dimensions. The first element contains
  bus identifiers for rows, the second contains bus identifiers for columns (typically identical)
- `lookup::L`: Tuple of dictionaries providing bidirectional mapping between bus numbers
  and their corresponding matrix indices
- `subnetwork_axes::Dict{Int, Ax}`: Dictionary mapping subnetwork identifiers to their
  corresponding axis information, used for handling electrical islands
- `network_reduction_data::NetworkReductionData`: Container for network reduction algorithms
  and their associated data, enabling efficient matrix operations on reduced networks

# Examples
```julia
# Create from a PowerSystems.System
sys = System("case5.m")
adj = AdjacencyMatrix(sys)

# Create from a Ybus matrix
ybus = Ybus(sys)
adj = AdjacencyMatrix(ybus)

# Check connectivity
is_connected = validate_connectivity(adj)
subnetworks = find_subnetworks(adj)
```

See also: [`Ybus`](@ref), [`IncidenceMatrix`](@ref), [`PowerNetworkMatrix`](@ref)
"""
struct AdjacencyMatrix{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{Int8}
    data::SparseArrays.SparseMatrixCSC{Int8, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    network_reduction_data::NetworkReductionData
end

# functions to get stored data
get_axes(M::AdjacencyMatrix) = M.axes
get_lookup(M::AdjacencyMatrix) = M.lookup
get_ref_bus(M::AdjacencyMatrix) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::AdjacencyMatrix) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::AdjacencyMatrix) = M.network_reduction_data
get_bus_axis(M::AdjacencyMatrix) = M.axes[1]
get_bus_lookup(M::AdjacencyMatrix) = M.lookup[1]

function _arc_conecting_two_areas(arc::PSY.Arc)
    from_bus = PSY.get_from(arc)
    to_bus = PSY.get_to(arc)
    area_from = IS.get_uuid(PSY.get_area(from_bus))
    area_to = IS.get_uuid(PSY.get_area(to_bus))
    return area_from != area_to
end

function _add_arc_buses_to_irreducible!(
    irreducible_buses::Vector{Int},
    arc::PSY.Arc,
)
    from_bus = PSY.get_from(arc)
    to_bus = PSY.get_to(arc)
    if PSY.get_available(from_bus)
        push!(irreducible_buses, PSY.get_number(from_bus))
    end
    if PSY.get_available(to_bus)
        push!(irreducible_buses, PSY.get_number(to_bus))
    end
    return
end

function _arc_conecting_two_areas(br::PSY.ACTransmission)
    arc = PSY.get_arc(br)
    return _arc_conecting_two_areas(arc)
end

function _add_arc_buses_to_irreducible!(
    irreducible_buses::Vector{Int},
    br::PSY.ACTransmission,
)
    arc = PSY.get_arc(br)
    _add_arc_buses_to_irreducible!(irreducible_buses, arc)
    return
end

function _arc_conecting_two_areas(br::PSY.ThreeWindingTransformer)
    # For the 3WT all the 4 buses are kept if any of the 3 arcs connect two areas
    arcs = [
        PSY.get_primary_star_arc(br),
        PSY.get_secondary_star_arc(br),
        PSY.get_tertiary_star_arc(br),
    ]
    for arc in arcs
        if _arc_conecting_two_areas(arc)
            return true
        end
    end
    return false
end

function _add_arc_buses_to_irreducible!(
    irreducible_buses::Vector{Int},
    br::PSY.ThreeWindingTransformer,
)
    arcs = [
        PSY.get_primary_star_arc(br),
        PSY.get_secondary_star_arc(br),
        PSY.get_tertiary_star_arc(br),
    ]
    for arc in arcs
        _add_arc_buses_to_irreducible!(irreducible_buses, arc)
    end
    return
end

_is_not_nodal_branch(::PSY.ACTransmission) = false
_is_not_nodal_branch(::PSY.AreaInterchange) = true

function get_reduction(
    A::AdjacencyMatrix,
    sys::PSY.System,
    reduction::DegreeTwoReduction,
)
    irreducible_buses = get_irreducible_buses(reduction)
    validate_buses(A, irreducible_buses)
    network_reduction_data = get_network_reduction_data(A)
    direct_branch_map = network_reduction_data.direct_branch_map
    parallel_branch_map = network_reduction_data.parallel_branch_map
    transformer3W_map = network_reduction_data.transformer3W_map

    for c in PSY.get_components(PSY.StaticInjection, sys)
        bus = PSY.get_bus(c)
        if PSY.get_available(bus)
            push!(irreducible_buses, PSY.get_number(bus))
        end
    end

    for tw_hvdc in PSY.get_components(PSY.TwoTerminalHVDC, sys)
        arc = PSY.get_arc(tw_hvdc)
        if PSY.get_available(PSY.get_from(arc))
            push!(irreducible_buses, PSY.get_number(PSY.get_from(arc)))
        end
        if PSY.get_available(PSY.get_to(arc))
            push!(irreducible_buses, PSY.get_number(PSY.get_to(arc)))
        end
    end

    # Keep buses connected by area interchange lines
    if PSY.has_components(sys, PSY.AreaInterchange)
        for br in PSY.get_components(PSY.ACTransmission, sys)
            if _arc_conecting_two_areas(br) && PSY.get_available(br)
                _add_arc_buses_to_irreducible!(irreducible_buses, br)
            end
        end
    end

    if PSY.has_components(sys, PSY.TransmissionInterface)
        for interface in PSY.get_components(PSY.TransmissionInterface, sys)
            for br in PSY.get_contributing_devices(sys, interface)
                if _is_not_nodal_branch(br) || !PSY.get_available(br)
                    continue
                end
                _add_arc_buses_to_irreducible!(irreducible_buses, br)
            end
        end
    end

    exempt_bus_positions = Set(get_irreducible_indices(A, irreducible_buses))
    series_branch_map, reverse_series_branch_map, removed_buses, removed_arcs =
        get_degree2_reduction(
            A.data,
            A.lookup[2],
            exempt_bus_positions,
            direct_branch_map,
            parallel_branch_map,
            transformer3W_map,
        )
    return NetworkReductionData(;
        irreducible_buses = Set(irreducible_buses),
        series_branch_map = series_branch_map,
        reverse_series_branch_map = reverse_series_branch_map,
        removed_buses = removed_buses,
        removed_arcs = removed_arcs,
        reductions = ReductionContainer(; degree_two_reduction = reduction),
    )
end

"""
    AdjacencyMatrix(sys::PSY.System; kwargs...)

Construct an AdjacencyMatrix from a PowerSystems.System.

# Arguments
- `sys::PSY.System`: The power system from which to construct the adjacency matrix

# Keyword arguments
- `network_reductions::Vector{NetworkReduction}=[]`: Network reduction algorithms to apply
- `include_constant_impedance_loads::Bool=true`: Whether to include constant impedance loads as shunt admittances
- `subnetwork_algorithm=iterative_union_find`: Algorithm for finding electrical islands

# Returns
- `AdjacencyMatrix`: An N x N adjacency matrix indexed with bus numbers showing connectivity
"""
function AdjacencyMatrix(sys::PSY.System; kwargs...)
    ybus = Ybus(sys; kwargs...)
    return AdjacencyMatrix(ybus)
end

"""
    AdjacencyMatrix(ybus::Ybus)

Construct an AdjacencyMatrix from a Ybus matrix.

# Arguments
- `ybus::Ybus`: The Ybus matrix from which to construct the adjacency matrix

# Returns
- `AdjacencyMatrix`: The constructed adjacency matrix showing bus connectivity
"""
function AdjacencyMatrix(ybus::Ybus)
    adj_matrix = deepcopy(ybus.adjacency_data)
    for i in 1:size(adj_matrix, 1)
        adj_matrix[i, i] = 0
    end
    SparseArrays.dropzeros!(adj_matrix)
    return AdjacencyMatrix(
        adj_matrix,
        deepcopy(ybus.axes),
        deepcopy(ybus.lookup),
        deepcopy(ybus.subnetwork_axes),
        ybus.network_reduction_data,
    )
end

"""
Validates connectivity by checking that the number of subnetworks is 1 (fully connected network).
"""
function validate_connectivity(M::AdjacencyMatrix)
    sub_nets = find_subnetworks(M)
    return length(sub_nets) == 1
end

"""
Evaluates subnetworks by looking for the subsets of buses connected each other,
but not connected with buses of other subsets.
"""
function find_subnetworks(M::AdjacencyMatrix)
    bus_numbers = get_bus_axis(M)
    return find_subnetworks(M.data, bus_numbers)
end
