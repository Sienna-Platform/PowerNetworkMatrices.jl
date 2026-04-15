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

"""
Pre-compute a mapping from each row (branch) in a CSC sparse matrix to its two column
(bus) endpoints. This avoids the expensive `A[row, :].nzind` operation on CSC matrices
which requires a full scan of all columns (O(nnz) per call).

Returns a Vector where index `row` gives `(col1, col2)` — the two bus columns connected
by that branch.
"""
function _build_row_to_cols(A::SparseArrays.SparseMatrixCSC{Int8, Int}, buscount::Int)
    n_rows = size(A, 1)
    row_first_col = zeros(Int, n_rows)
    row_to_cols = Vector{Tuple{Int, Int}}(undef, n_rows)
    for col in 1:buscount
        for k in A.colptr[col]:(A.colptr[col + 1] - 1)
            row = A.rowval[k]
            if row_first_col[row] == 0
                row_first_col[row] = col
            else
                row_to_cols[row] = (row_first_col[row], col)
            end
        end
    end
    return row_to_cols
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
1. **Adjacency Pre-computation**: Builds adjacency lists from the incidence matrix to avoid
   expensive sparse row access operations on the CSC matrix
2. **Leaf Detection**: Identifies buses with exactly one connection (radial buses)
3. **Reference Protection**: Preserves reference buses from elimination regardless of connectivity
4. **Iterative Chain Walking**: Uses a queue-based approach to trace from radial buses toward
   the core network, walking through chains of degree-2 buses
5. **Cascading Reduction**: Eliminates buses that become radial after initial reductions

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
- Pre-computes row-to-column mapping for O(1) branch endpoint lookup instead of O(nnz) sparse
  row access
- Uses iterative queue-based processing instead of recursive DFS for better performance
- Handles edge cases like fully radial networks and isolated islands
- Provides comprehensive mapping for traceability and debugging
"""
function calculate_radial_arcs(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    arc_map::Dict{Tuple{Int, Int}, Int},
    bus_map::Dict{Int, Int},
    ref_bus_positions::Set{Int},
)
    buscount = length(bus_map)
    n_arcs = size(A, 1)
    radial_arcs = Set{Tuple{Int, Int}}()
    final_arc_map = Dict{Tuple{Int, Int}, Int}()
    reverse_arc_map = Dict(reverse(kv) for kv in arc_map)
    reverse_bus_map = Dict(reverse(kv) for kv in bus_map)
    bus_reduction_map_index = Dict{Int, Set{Int}}(k => Set{Int}() for k in keys(bus_map))

    # Pre-compute row → (col1, col2) mapping. This replaces the expensive A[row, :].nzind
    # operations on CSC matrices (O(nnz) per call) with O(1) lookups.
    row_to_cols = _build_row_to_cols(A, buscount)

    # Build adjacency lists: for each bus column, store (neighbor_col, row_index) pairs.
    adj = Vector{Vector{Tuple{Int, Int}}}(undef, buscount)
    for j in 1:buscount
        adj[j] = Vector{Tuple{Int, Int}}()
    end
    for row in 1:n_arcs
        c1, c2 = row_to_cols[row]
        push!(adj[c1], (c2, row))
        push!(adj[c2], (c1, row))
    end

    # Compute original degree of each bus (used for chain-walking decisions).
    orig_degree = Vector{Int}(undef, buscount)
    for j in 1:buscount
        orig_degree[j] = length(adj[j])
    end

    # Initialize queue with all original leaf buses (degree 1, not exempt).
    queue = Vector{Int}()
    for j in 1:buscount
        if orig_degree[j] == 1 && j ∉ ref_bus_positions
            push!(queue, j)
        end
    end

    # Track which bus columns have been removed during reduction.
    removed = falses(buscount)

    # Iterative radial chain peeling: process leaves and walk up through degree-2 chains.
    while !isempty(queue)
        j = popfirst!(queue)
        if removed[j]
            continue
        end

        # Find the non-removed parent (neighbor) of j.
        parent = 0
        row_ix = 0
        for (neighbor, rix) in adj[j]
            if !removed[neighbor]
                parent = neighbor
                row_ix = rix
                break
            end
        end

        if parent == 0
            continue
        end

        # Mark j as removed and record the radial arc.
        removed[j] = true
        j_bus_number = reverse_bus_map[j]
        arc = reverse_arc_map[row_ix]
        push!(radial_arcs, arc)

        # Update reduction maps: merge j's reduction set into parent's.
        reduction_set = pop!(bus_reduction_map_index, j_bus_number)
        push!(reduction_set, j_bus_number)
        parent_bus_number = reverse_bus_map[parent]
        union!(bus_reduction_map_index[parent_bus_number], reduction_set)

        if parent ∈ ref_bus_positions
            # Parent is a reference/exempt bus — chain terminates here.
            final_arc_map[arc] = parent_bus_number
        elseif orig_degree[parent] > 2
            # Parent has >2 original connections — it survives.
            final_arc_map[arc] = parent_bus_number
        elseif orig_degree[parent] == 2
            # Parent is an original degree-2 chain node — continue walking.
            push!(queue, parent)
        else
            # This check captures cases of a full radial network which can happen in
            # systems with small islands that represent larger interconnected areas.
            @warn "Bus $j_bus_number Parent $parent_bus_number is a leaf node, indicating there is an island."
            final_arc_map[arc] = parent_bus_number
        end
    end

    reverse_bus_search_map = _make_reverse_bus_search_map(bus_reduction_map_index, buscount)
    return bus_reduction_map_index,
    reverse_bus_search_map,
    radial_arcs,
    final_arc_map
end
