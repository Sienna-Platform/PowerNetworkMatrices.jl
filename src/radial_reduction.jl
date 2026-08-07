"""
    RadialReduction <: NetworkReduction

Eliminates leaf (degree-1) buses and their branches. Protect specific buses via
`Ybus(sys; irreducible_buses=...)`.
"""
@kwdef struct RadialReduction <: NetworkReduction end

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
    # `(0, 0)` sentinel for rows lacking a second bus column (e.g. a self-loop arc
    # whose incidence entries cancel), which would otherwise be read undefined.
    row_to_cols = fill((0, 0), n_rows)
    Arowval = SparseArrays.rowvals(A)
    for col in 1:buscount
        for k in SparseArrays.nzrange(A, col)
            row = Arowval[k]
            if iszero(row_first_col[row])
                row_first_col[row] = col
            else
                row_to_cols[row] = (row_first_col[row], col)
            end
        end
    end
    return row_to_cols
end

"""
Follow `parent_of` from an eliminated bus column to the surviving column that absorbs it,
compressing the traversed chain in place so later lookups along it are O(1).
"""
function _surviving_root!(parent_of::Vector{Int}, removed::BitVector, j::Int)
    root = parent_of[j]
    while removed[root]
        root = parent_of[root]
    end
    cur = j
    while removed[cur]
        parent_of[cur], cur = root, parent_of[cur]
    end
    return root
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
        Dictionary mapping each removed arc that still touches a surviving bus to that bus
        number (the arc whose admittance must be subtracted from the surviving bus's
        diagonal). Arcs whose endpoints are both eliminated are absent.

# Algorithm Overview
1. **Adjacency Pre-computation**: Builds adjacency lists from the incidence matrix to avoid
   expensive sparse row access operations on the CSC matrix
2. **Leaf Detection**: Identifies buses with exactly one connection (radial buses)
3. **Reference Protection**: Preserves reference buses from elimination regardless of connectivity
4. **Cascading Reduction**: Peels leaves from a queue, decrementing each parent's degree as
   its neighbors are eliminated and enqueuing any parent that becomes a leaf in turn. The
   surviving buses are the graph's 2-core, together with the exempt buses and one bus per
   fully radial island.

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
        # Sentinel rows (a self-loop arc, no second bus column) add no graph edge.
        (iszero(c1) || iszero(c2)) && continue
        push!(adj[c1], (c2, row))
        push!(adj[c2], (c1, row))
    end

    # Degree counting only buses not yet eliminated, so a bus that becomes a leaf part-way
    # through the peel is eliminated on the same pass.
    live_degree = Vector{Int}(undef, buscount)
    for j in 1:buscount
        live_degree[j] = length(adj[j])
    end

    queue = Vector{Int}()
    for j in 1:buscount
        if live_degree[j] == 1 && j ∉ ref_bus_positions
            push!(queue, j)
        end
    end

    # `parent_of` records the neighbor that absorbed each removed bus. That neighbor can be
    # eliminated later, so chains resolve to surviving buses only after the peel.
    removed = falses(buscount)
    parent_of = zeros(Int, buscount)

    while !isempty(queue)
        j = popfirst!(queue)
        if removed[j]
            continue
        end

        parent = 0
        row_ix = 0
        for (neighbor, rix) in adj[j]
            if !removed[neighbor]
                parent = neighbor
                row_ix = rix
                break
            end
        end

        if iszero(parent)
            # Last bus standing in a fully radial component; it survives to represent it.
            @warn "Bus $(reverse_bus_map[j]) has no surviving neighbor, indicating a fully radial island."
            continue
        end

        removed[j] = true
        parent_of[j] = parent
        push!(radial_arcs, reverse_arc_map[row_ix])

        # Removing j may leave the parent radial in turn.
        live_degree[parent] -= 1
        if live_degree[parent] == 1 && parent ∉ ref_bus_positions
            push!(queue, parent)
        end
    end

    # Attribute every eliminated bus to the surviving bus at the end of its parent chain.
    # Resolving the chains in one pass keeps the cost near-linear in the bus count.
    for j in 1:buscount
        removed[j] && delete!(bus_reduction_map_index, reverse_bus_map[j])
    end
    for j in 1:buscount
        removed[j] || continue
        root = _surviving_root!(parent_of, removed, j)
        push!(bus_reduction_map_index[reverse_bus_map[root]], reverse_bus_map[j])
    end

    # A removed arc needs a diagonal correction only on an endpoint that survives; an arc
    # that loses both endpoints leaves no trace in the reduced Ybus.
    for arc in radial_arcs
        c1, c2 = row_to_cols[arc_map[arc]]
        if removed[c1] && !removed[c2]
            final_arc_map[arc] = reverse_bus_map[c2]
        elseif removed[c2] && !removed[c1]
            final_arc_map[arc] = reverse_bus_map[c1]
        end
    end

    reverse_bus_search_map = _make_reverse_bus_search_map(bus_reduction_map_index, buscount)
    return bus_reduction_map_index,
    reverse_bus_search_map,
    radial_arcs,
    final_arc_map
end
