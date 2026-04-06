"""
Structure containing the network incidence matrix and related topology data.

The incidence matrix A represents the bus-branch connectivity of the power network, where
each row corresponds to a branch and each column corresponds to a bus. Elements are:
- +1 for the "from" bus of a branch
- -1 for the "to" bus of a branch  
- 0 for buses not connected to the branch

# Fields
- `data::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        The incidence matrix data with dimensions (n_branches × n_buses). Values are {-1, 0, +1}
        representing the directed connectivity between branches and buses
- `axes::Ax`:
        Tuple containing (arc_identifiers, bus_numbers) where arcs are branch endpoint pairs
        and buses are the network bus numbers
- `lookup::L <: NTuple{2, Dict}`:
        Tuple of dictionaries providing fast lookup from arc/bus identifiers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`:
        Mapping from reference bus numbers to their corresponding subnetwork axes
- `network_reduction_data::NetworkReductionData`:
        Container for network reduction information applied during matrix construction

# Mathematical Properties
- **Matrix Dimensions**: (n_branches × n_buses)
- **Element Values**: {-1, 0, +1} representing directed branch-bus connectivity
- **Row Sum**: Each row sums to zero (conservation at branch level)
- **Rank**: Rank is (n_buses - n_islands) for connected networks
- **Sparsity**: Very sparse with exactly 2 non-zero elements per branch row

# Applications
- **Power Flow**: Forms the basis for DC power flow equations: P = A^T * f
- **Sensitivity Analysis**: Used in PTDF and LODF calculations
- **Network Analysis**: Identifies connected components and network structure
- **Topology Processing**: Enables network reduction and equivalencing algorithms

# Notes
- Each branch contributes exactly one row with two non-zero entries (+1, -1)
- Reference buses are preserved in the matrix but identified separately
- Supports various network reduction techniques for computational efficiency
- Essential building block for BA_Matrix and ABA_Matrix constructions
"""
struct IncidenceMatrix{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{Int8}
    data::SparseArrays.SparseMatrixCSC{Int8, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    network_reduction_data::NetworkReductionData
end

# functions to get stored data
get_axes(M::IncidenceMatrix) = M.axes
get_lookup(M::IncidenceMatrix) = M.lookup
get_ref_bus(M::IncidenceMatrix) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::IncidenceMatrix) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::IncidenceMatrix) = M.network_reduction_data
get_arc_axis(M::IncidenceMatrix) = M.axes[1]
get_arc_lookup(M::IncidenceMatrix) = M.lookup[1]
get_bus_axis(M::IncidenceMatrix) = M.axes[2]
get_bus_lookup(M::IncidenceMatrix) = M.lookup[2]

function get_reduction(
    A::IncidenceMatrix,
    ::PSY.System,
    reduction::RadialReduction,
)
    irreducible_buses = get_irreducible_buses(reduction)
    irreducible_positions = Set([A.lookup[2][x] for x in irreducible_buses])
    exempt_positions = union(get_ref_bus_position(A), irreducible_positions)
    bus_reduction_map, reverse_bus_search_map, radial_arcs, final_arc_map =
        calculate_radial_arcs(A.data, A.lookup[1], A.lookup[2], Set(exempt_positions))

    return NetworkReductionData(;
        irreducible_buses = Set(irreducible_buses),
        bus_reduction_map = bus_reduction_map,
        reverse_bus_search_map = reverse_bus_search_map,
        removed_arcs = radial_arcs,
        radial_arc_to_surviving_bus = final_arc_map,
        reductions = ReductionContainer(; radial_reduction = reduction),
    )
end

"""
    IncidenceMatrix(sys::PSY.System; network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct an IncidenceMatrix from a PowerSystems.System by extracting the network topology
and creating the bus-branch connectivity matrix fundamental to power system analysis.

# Arguments
- `sys::PSY.System`: The power system from which to construct the incidence matrix

# Keyword Arguments
- `network_reductions::Vector{NetworkReduction} = NetworkReduction[]`: 
        Vector of network reduction algorithms to apply before matrix construction
- `include_constant_impedance_loads::Bool=true`: 
        Whether to include constant impedance loads as shunt admittances in the network model
- `subnetwork_algorithm=iterative_union_find`: 
        Algorithm used for identifying electrical islands and connected components
- Additional keyword arguments are passed to the underlying `Ybus` constructor

# Returns
- `IncidenceMatrix`: The constructed incidence matrix structure containing:
  - Bus-branch connectivity matrix with {-1, 0, +1} elements
  - Network topology information and reference bus identification
  - Support for network reductions and connected component analysis

# Mathematical Construction
1. **Network Extraction**: Identifies all branches and buses from the power system
2. **Connectivity Mapping**: Creates directed branch-bus relationships  
3. **Matrix Assembly**: Constructs sparse matrix with +1/-1 entries for branch endpoints
4. **Topology Analysis**: Identifies reference buses and connected components
5. **Network Reductions**: Applies specified reduction algorithms if provided

# Applications
- **Foundation Matrix**: Essential for constructing BA_Matrix and ABA_Matrix
- **DC Power Flow**: Enables linearized power flow analysis through P = A^T * f
- **Sensitivity Analysis**: Required for PTDF, LODF, and other sensitivity calculations
- **Network Analysis**: Supports topology processing and network equivalencing

# Notes
- Each branch creates exactly one matrix row with two non-zero entries
- Network reductions can significantly improve computational efficiency
- Reference buses are automatically identified for later matrix operations
- The matrix preserves full network topology for comprehensive power system analysis
"""
function IncidenceMatrix(sys::PSY.System;
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    return IncidenceMatrix(
        Ybus(
            sys;
            network_reductions = network_reductions,
            kwargs...,
        ),
    )
end

"""
    IncidenceMatrix(ybus::Ybus)

Construct an IncidenceMatrix from an existing Ybus matrix by extracting the network topology
and creating the bus-branch connectivity matrix. This constructor leverages the network 
structure already captured in the Ybus matrix.

# Arguments
- `ybus::Ybus`: The Ybus matrix containing network topology and admittance data

# Returns
- `IncidenceMatrix`: The constructed incidence matrix structure containing:
  - Bus-branch connectivity matrix with {-1, 0, +1} elements representing directed connections
  - Network topology information extracted from the Ybus structure
  - Reference bus identification and subnetwork axes from the source matrix
  - Network reduction data inherited from the Ybus matrix

# Construction Process
1. **Topology Extraction**: Retrieves bus and branch information from the Ybus matrix
2. **Arc Processing**: Creates directed arc representations from branch connectivity
3. **Matrix Assembly**: Constructs sparse incidence matrix with +1 (from bus) and -1 (to bus) entries
4. **Isolated Bus Handling**: Includes isolated buses with zero entries for completeness
5. **Metadata Transfer**: Preserves reference bus positions and network reduction information

# Mathematical Properties
- **Matrix Form**: A[i,j] = +1 if branch i originates at bus j, -1 if it terminates at bus j, 0 otherwise
- **Dimensions**: (n_branches × n_buses) including all network branches and buses
- **Sparsity**: Exactly 2 non-zero entries per branch row (except for isolated buses)
- **Consistency**: Maintains the same network topology and reduction state as the source Ybus

# Notes
- This constructor is more efficient when a Ybus matrix is already available
- Preserves all network reduction information from the source matrix
- Isolated buses are handled explicitly to maintain network completeness
- Essential for creating downstream matrices like BA_Matrix and ABA_Matrix from existing Ybus
"""
function IncidenceMatrix(ybus::Ybus)
    nr = ybus.network_reduction_data
    bus_ax = get_bus_axis(ybus)
    bus_lookup = ybus.lookup[1]
    arc_ax = get_arc_axis(nr)
    n_isolated_buses = length(get_isolated_buses(ybus))
    n_entries = length(arc_ax) * 2 + n_isolated_buses
    A_I = Vector{Int}(undef, n_entries)
    A_J = Vector{Int}(undef, n_entries)
    A_V = Vector{Int8}(undef, n_entries)
    for (ix, arc) in enumerate(arc_ax)
        A_I[2 * ix - 1] = ix
        A_J[2 * ix - 1] = get_bus_index(arc[1], bus_lookup, nr)
        A_V[2 * ix - 1] = 1
        A_I[2 * ix] = ix
        A_J[2 * ix] = get_bus_index(arc[2], bus_lookup, nr)
        A_V[2 * ix] = -1
    end
    A_I[(end - n_isolated_buses + 1):end] = ones(n_isolated_buses)
    A_J[(end - n_isolated_buses + 1):end] =
        [get_bus_index(x, bus_lookup, nr) for x in get_isolated_buses(ybus)]
    A_V[(end - n_isolated_buses + 1):end] = zeros(n_isolated_buses)
    data = SparseArrays.sparse(A_I, A_J, A_V)
    SparseArrays.dropzeros!(data)
    axes = (arc_ax, bus_ax)
    lookup = (make_ax_ref(arc_ax), make_ax_ref(bus_ax))
    subnetwork_axes = make_arc_bus_subnetwork_axes(ybus)

    return IncidenceMatrix(
        data,
        axes,
        lookup,
        subnetwork_axes,
        ybus.network_reduction_data,
    )
end

"""
Make subnetwork axes for LODF and VirtualLODF
"""
function make_arc_arc_subnetwork_axes(A::IncidenceMatrix)
    subnetwork_axes = Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Tuple{Int, Int}}}}()
    for key in keys(A.subnetwork_axes)
        subnetwork_axes[key] = (A.subnetwork_axes[key][1], A.subnetwork_axes[key][1])
    end
    return subnetwork_axes
end
