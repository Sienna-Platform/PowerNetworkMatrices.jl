"""
    RadialReduction <: NetworkReduction

Network reduction algorithm that eliminates radial (dangling) buses and their associated branches
from the power network. Radial buses are leaf nodes with only one connection that do not affect
the electrical behavior of the rest of the network.

# Fields
- `irreducible_buses::Vector{Int}`: List of bus numbers that should not be eliminated even if they are radial

# Examples
```julia
# Create radial reduction with no protected buses
reduction = RadialReduction()

# Create radial reduction protecting specific buses
reduction = RadialReduction(irreducible_buses=[101, 205])

# Apply to system
ybus = Ybus(system; network_reductions=[reduction])
```
"""
@kwdef struct RadialReduction <: NetworkReduction
    irreducible_buses::Vector{Int} = Vector{Int}()
end
get_irreducible_buses(nr::RadialReduction) = nr.irreducible_buses

function _find_upstream_bus(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    j::Int,
    reverse_arc_map::Dict{Int, Tuple{Int, Int}},
    radial_arcs::Set{Tuple{Int, Int}},
    reduced_buses::Set{Int},
    reverse_bus_map::Dict{Int, Int},
)
    row_ix = A.rowval[A.colptr[j]]
    parent = setdiff!(A[row_ix, :].nzind, j)[1]
    if reverse_bus_map[parent] ∈ reduced_buses
        row_ix = A.rowval[A.colptr[j] + 1]
        parent = setdiff!(A[row_ix, :].nzind, j)[1]
    end
    arc = reverse_arc_map[row_ix]
    push!(radial_arcs, arc)
    return parent, arc
end

function _new_parent(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    parent::Int,
    bus_reduction_map_index::Dict{Int, Set{Int}},
    reverse_arc_map::Dict{Int, Tuple{Int, Int}},
    radial_arcs::Set{Tuple{Int, Int}},
    reverse_bus_map::Dict{Int, Int},
    final_arc_map::Dict{Tuple{Int, Int}, Int},
)
    if length(SparseArrays.nzrange(A, parent)) == 2
        parent_bus_number = reverse_bus_map[parent]
        new_parent_val, arc = _find_upstream_bus(
            A,
            parent,
            reverse_arc_map,
            radial_arcs,
            bus_reduction_map_index[parent_bus_number],
            reverse_bus_map,
        )
        new_parent_bus_number = reverse_bus_map[new_parent_val]
        # This check is meant to capture cases of a full radial network which can happen in
        # system with small islands that represent larger interconnected areas.
        if length(SparseArrays.nzrange(A, new_parent_val)) < 2
            @warn "Bus $parent_bus_number Parent $new_parent_bus_number is a leaf node, indicating there is an island."
            push!(bus_reduction_map_index[parent_bus_number], new_parent_bus_number)
            final_arc_map[arc] = new_parent_bus_number
            return
        end
        # Check if chain terminates (new_parent survives with >2 connections)
        if length(SparseArrays.nzrange(A, new_parent_val)) > 2
            final_arc_map[arc] = new_parent_bus_number
        end
        new_set = push!(pop!(bus_reduction_map_index, parent_bus_number), parent_bus_number)
        union!(bus_reduction_map_index[new_parent_bus_number], new_set)
        _new_parent(
            A,
            new_parent_val,
            bus_reduction_map_index,
            reverse_arc_map,
            radial_arcs,
            reverse_bus_map,
            final_arc_map,
        )
    end
    return
end

function _reverse_search(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    j::Int,
    bus_reduction_map_index::Dict{Int, Set{Int}},
    reverse_arc_map::Dict{Int, Tuple{Int, Int}},
    radial_arcs::Set{Tuple{Int, Int}},
    reverse_bus_map::Dict{Int, Int},
    ref_bus_positions::Set{Int},
    final_arc_map::Dict{Tuple{Int, Int}, Int},
)
    if j ∈ ref_bus_positions
        return
    end
    j_bus_number = reverse_bus_map[j]
    pop!(bus_reduction_map_index, j_bus_number)
    reduction_set = Set{Int}(j_bus_number)
    parent, arc = _find_upstream_bus(
        A,
        j,
        reverse_arc_map,
        radial_arcs,
        reduction_set,
        reverse_bus_map,
    )
    parent_bus_number = reverse_bus_map[parent]
    union!(bus_reduction_map_index[parent_bus_number], reduction_set)
    if parent ∈ ref_bus_positions
        final_arc_map[arc] = parent_bus_number
        return
    end
    # Check if chain terminates (parent survives with >2 connections)
    if length(SparseArrays.nzrange(A, parent)) > 2
        final_arc_map[arc] = parent_bus_number
    end
    _new_parent(
        A,
        parent,
        bus_reduction_map_index,
        reverse_arc_map,
        radial_arcs,
        reverse_bus_map,
        final_arc_map,
    )
    return
end

function _make_reverse_bus_search_map(bus_reduction_map::Dict{Int, Set{Int}}, n_buses::Int)
    map = Dict{Int, Int}()
    sizehint!(map, n_buses)
    for (parent, children) in bus_reduction_map
        for bus in children
            map[bus] = parent
        end
    end
    return map
end

"""
    calculate_radial_arcs(A::SparseArrays.SparseMatrixCSC{Int8, Int}, arc_map::Dict{Tuple{Int, Int}, Int}, bus_map::Dict{Int, Int}, ref_bus_positions::Set{Int})

Identify and calculate radial branches and buses that can be eliminated from the network model
by analyzing the topological structure of the incidence matrix. Radial elements are leaf nodes
with only one connection that do not affect the electrical behavior of the core network.

# Arguments
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        The incidence matrix data representing bus-branch connectivity structure
- `arc_map::Dict{Tuple{Int, Int}, Int}`:
        Dictionary mapping branch endpoint pairs (from_bus, to_bus) to matrix row indices
- `bus_map::Dict{Int, Int}`:
        Dictionary mapping bus numbers to matrix column indices
- `ref_bus_positions::Set{Int}`:
        Set of matrix column indices corresponding to reference (slack) buses that cannot be eliminated

# Returns
- `bus_reduction_map::Dict{Int, Set{Int}}`:
        Dictionary mapping parent bus numbers to sets of child buses that can be reduced to the parent
- `reverse_bus_search_map::Dict{Int, Int}`:
        Dictionary mapping each bus number to its ultimate parent bus after all reductions
- `radial_arcs::Set{Tuple{Int, Int}}`:
        Set of branch endpoint pairs representing radial branches that can be eliminated
- `final_arc_map::Dict{Tuple{Int, Int}, Int}`:
        Dictionary mapping each final arc to the surviving bus number it connects to
        (the arc whose admittance must be subtracted from the surviving bus's diagonal)

# Algorithm Overview
1. **Leaf Detection**: Identifies buses with exactly one connection (radial buses)
2. **Reference Protection**: Preserves reference buses from elimination regardless of connectivity
3. **Upstream Tracing**: Traces from radial buses toward the core network to find parent buses
4. **Cascading Reduction**: Recursively eliminates buses that become radial after initial reductions
5. **Parallel Processing**: Uses multithreading for efficient analysis of large networks

# Network Topology Preservation
- **Electrical Equivalence**: Ensures reduced network maintains same electrical behavior
- **Connectivity Integrity**: Preserves essential network connectivity and reference structure
- **Reduction Validity**: Only eliminates elements that truly don't affect network analysis
- **Reversibility**: Maintains mapping information for potential reconstruction if needed

# Use Cases
- **Network Simplification**: Reduces computational burden by eliminating unnecessary elements
- **Matrix Conditioning**: Improves numerical properties of network matrices
- **Analysis Acceleration**: Speeds up power flow and other network computations
- **Memory Optimization**: Reduces storage requirements for large network models

# Implementation Notes
- Uses sparse matrix operations for efficiency with large, sparse networks
- Handles edge cases like fully radial networks and isolated islands
- Maintains thread safety for concurrent processing of network analysis
- Provides comprehensive mapping for traceability and debugging
"""
function calculate_radial_arcs(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    arc_map::Dict{Tuple{Int, Int}, Int},
    bus_map::Dict{Int, Int},
    ref_bus_positions::Set{Int},
)
    lk = ReentrantLock()
    buscount = length(bus_map)
    radial_arcs = Set{Tuple{Int, Int}}()
    final_arc_map = Dict{Tuple{Int, Int}, Int}()
    reverse_arc_map = Dict(reverse(kv) for kv in arc_map)
    reverse_bus_map = Dict(reverse(kv) for kv in bus_map)
    bus_reduction_map_index = Dict{Int, Set{Int}}(k => Set{Int}() for k in keys(bus_map))
    Threads.@threads for j in 1:buscount
        if length(SparseArrays.nzrange(A, j)) == 1
            lock(lk) do
                _reverse_search(
                    A,
                    j,
                    bus_reduction_map_index,
                    reverse_arc_map,
                    radial_arcs,
                    reverse_bus_map,
                    ref_bus_positions,
                    final_arc_map,
                )
            end
        end
    end
    reverse_bus_search_map = _make_reverse_bus_search_map(bus_reduction_map_index, buscount)
    return bus_reduction_map_index,
    reverse_bus_search_map,
    radial_arcs,
    final_arc_map
end
