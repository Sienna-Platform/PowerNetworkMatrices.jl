"""
    DegreeTwoReduction <: NetworkReduction

Network reduction algorithm that eliminates buses with exactly two connections by combining
the incident branches into a single equivalent branch. This reduction preserves the electrical
characteristics of the network while simplifying its topology.

# Fields
- `irreducible_buses::Vector{Int}`: List of bus numbers that should not be eliminated even if they have degree two
- `reduce_reactive_power_injectors::Bool`: Whether to reduce buses with reactive power injections (default: true)

# Examples
```julia
# Create degree-two reduction with default settings
reduction = DegreeTwoReduction()

# Create degree-two reduction protecting specific buses
reduction = DegreeTwoReduction(irreducible_buses=[101, 205])

# Create reduction that preserves buses with reactive power injections
reduction = DegreeTwoReduction(reduce_reactive_power_injectors=false)

# Apply to system
ybus = Ybus(system; network_reductions=[reduction])
```
"""
@kwdef struct DegreeTwoReduction <: NetworkReduction
    irreducible_buses::Vector{Int} = Vector{Int}()
    reduce_reactive_power_injectors::Bool = true
end
get_irreducible_buses(nr::DegreeTwoReduction) = nr.irreducible_buses
get_reduce_reactive_power_injectors(nr::DegreeTwoReduction) =
    nr.reduce_reactive_power_injectors

function get_degree2_reduction(
    data::SparseArrays.SparseMatrixCSC{Int8, Int},
    bus_lookup::Dict{Int, Int},
    exempt_bus_positions::Set{Int},
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission},
    parallel_branch_map::Dict{Tuple{Int, Int}, BranchesParallel},
    transformer3W_map::Dict{Tuple{Int, Int}, ThreeWindingTransformerWinding},
)
    reverse_bus_lookup = Dict(v => k for (k, v) in bus_lookup)
    arc_maps = find_degree2_chains(data, exempt_bus_positions)
    series_branch_map = Dict{Tuple{Int, Int}, BranchesSeries}()

    removed_buses = Set{Int}()
    removed_arcs = Set{Tuple{Int, Int}}()
    for (composite_arc_ix, segment_ix) in arc_maps
        composite_arc = (
            reverse_bus_lookup[composite_arc_ix[1]],
            reverse_bus_lookup[composite_arc_ix[2]],
        )
        segment_numbers = [reverse_bus_lookup[x] for x in segment_ix]
        @assert composite_arc[1] == segment_numbers[1]
        @assert composite_arc[2] == segment_numbers[end]
        segments = BranchesSeries()
        for ix in 1:(length(segment_numbers) - 1)
            segment_arc = (segment_numbers[ix], segment_numbers[ix + 1])
            segment_arc, entry, orientation = _get_branch_map_entry(
                direct_branch_map,
                parallel_branch_map,
                transformer3W_map,
                segment_arc,
            )
            add_branch!(segments, entry, orientation)
            push!(removed_arcs, segment_arc)
            ix != 1 && push!(removed_buses, segment_numbers[ix])
        end
        series_branch_map[composite_arc] = segments
    end
    reverse_series_branch_map = _make_reverse_series_branch_map(series_branch_map)
    return series_branch_map, reverse_series_branch_map, removed_buses, removed_arcs
end

function _add_to_reverse_series_branch_map!(
    reverse_series_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}},
    composite_arc::Tuple{Int, Int},
    segment::BranchesParallel,
)
    for branch in segment.branches
        reverse_series_branch_map[branch] = composite_arc
    end
    return
end

function _add_to_reverse_series_branch_map!(
    reverse_series_branch_map::Dict{PSY.ACTransmission, Tuple{Int, Int}},
    composite_arc::Tuple{Int, Int},
    segment::PSY.ACTransmission,
)
    reverse_series_branch_map[segment] = composite_arc
    return
end

function _make_reverse_series_branch_map(
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries},
)
    reverse_series_branch_map = Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    for (composite_arc, vector_segments) in series_branch_map
        for segment_collection in values(vector_segments.branches)
            for segment in segment_collection
                _add_to_reverse_series_branch_map!(
                    reverse_series_branch_map,
                    composite_arc,
                    segment,
                )
            end
        end
    end
    return reverse_series_branch_map
end

function _get_branch_map_entry(
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission},
    parallel_branch_map::Dict{Tuple{Int, Int}, BranchesParallel},
    transformer3W_map::Dict{Tuple{Int, Int}, ThreeWindingTransformerWinding},
    arc::Tuple{Int, Int},
)
    reverse_arc = (arc[2], arc[1])
    if haskey(direct_branch_map, arc)
        return arc, direct_branch_map[arc], :FromTo
    elseif haskey(direct_branch_map, reverse_arc)
        return reverse_arc, direct_branch_map[reverse_arc], :ToFrom
    elseif haskey(parallel_branch_map, arc)
        return arc, parallel_branch_map[arc], :FromTo
    elseif haskey(parallel_branch_map, reverse_arc)
        return reverse_arc, parallel_branch_map[reverse_arc], :ToFrom
    elseif haskey(transformer3W_map, arc)
        return arc, transformer3W_map[arc], :FromTo
    elseif haskey(transformer3W_map, reverse_arc)
        return reverse_arc, transformer3W_map[reverse_arc], :ToFrom
    else
        error("Arc $arc not found in the existing network reduction mappings.")
    end
end

"""
    _should_visit_node(node::Int, reduced_indices::BitVector, irreducible_indices::BitVector)

Determines whether a node should be visited during network traversal.

# Arguments
- `node::Int`: The index of the node to check.
- `reduced_indices::BitVector`: Bitmask of indices that have already been reduced.
- `irreducible_indices::BitVector`: Bitmask of indices that cannot be reduced.

# Returns
- `Bool`: `true` if the node should be visited, `false` otherwise.
"""
function _should_visit_node(
    node::Int,
    reduced_indices::BitVector,
    irreducible_indices::BitVector,
)
    if irreducible_indices[node]
        return false
    end
    if reduced_indices[node]
        return false
    end
    return true
end

"""
    _is_final_node(node::Int, adj_matrix::SparseArrays.SparseMatrixCSC, reduced_indices::BitVector, irreducible_indices::BitVector)

Determines if a node is a final node in a path traversal.

# Arguments
- `node::Int`: The index of the node to check.
- `adj_matrix::SparseArrays.SparseMatrixCSC`: The adjacency matrix of the network.
- `reduced_indices::BitVector`: Bitmask of indices that have already been reduced.
- `irreducible_indices::BitVector`: Bitmask of indices that should not be reduced.

# Returns
- `Bool`: `true` if the node is a final node, `false` otherwise.
"""
function _is_final_node(
    node::Int,
    adj_matrix::SparseArrays.SparseMatrixCSC,
    reduced_indices::BitVector,
    irreducible_indices::BitVector,
)
    if !_is_2degree_node(adj_matrix, node)
        return true
    end
    if reduced_indices[node]
        return true
    end
    if irreducible_indices[node]
        return true
    end
    return false
end

"""
    _is_2degree_node(adj_matrix::SparseArrays.SparseMatrixCSC, node::Int)

Checks if a node has exactly two connections in the network.

# Arguments
- `adj_matrix::SparseArrays.SparseMatrixCSC`: The adjacency matrix of the network.
- `node::Int`: The index of the node to check.

# Returns
- `Bool`: `true` if the node has exactly two neighbors, `false` otherwise.
"""
function _is_2degree_node(adj_matrix::SparseArrays.SparseMatrixCSC, node::Int)
    neighbor_count = SparseArrays.nzrange(adj_matrix, node)
    return length(neighbor_count) == 2
end

"""
    _get_neighbors(adj_matrix::SparseArrays.SparseMatrixCSC, node::Int)

Get all neighbors of a given node from the adjacency matrix.
For undirected graphs, checks both directions.
"""
function _get_neighbors(adj_matrix::SparseArrays.SparseMatrixCSC, node::Int)
    nzrange = SparseArrays.nzrange(adj_matrix, node)
    @assert length(nzrange) == 2
    return rowvals(adj_matrix)[nzrange]
end

"""
    _get_complete_chain(adj_matrix::SparseArrays.SparseMatrixCSC, start_node::Int, reduced_indices::Set{Int}, irreducible_indices::Set{Int})

Build a complete chain of degree-2 nodes starting from a given node.
"""
function _get_complete_chain(
    adj_matrix::SparseArrays.SparseMatrixCSC,
    start_node::Int,
    reduced_indices::BitVector,
    irreducible_indices::BitVector,
)
    neighbors = _get_neighbors(adj_matrix, start_node)
    current_chain = [start_node]
    reduced_indices[start_node] = true
    _get_partial_chain_recursive!(
        current_chain,
        adj_matrix,
        neighbors[1],
        start_node,
        reduced_indices,
        irreducible_indices,
    )
    reverse!(current_chain)
    _get_partial_chain_recursive!(
        current_chain,
        adj_matrix,
        neighbors[2],
        start_node,
        reduced_indices,
        irreducible_indices,
    )
    return current_chain
end

"""
    _get_partial_chain(adj_matrix::SparseArrays.SparseMatrixCSC,
                      current_node::Int,
                      prev_node::Int,
                      reduced_indices::Set{Int},
                      irreducible_indices::Set{Int})

Recursively build a chain in one direction from current_node, avoiding prev_node.
"""
function _get_partial_chain_recursive!(
    current_chain::Vector{Int},
    adj_matrix::SparseArrays.SparseMatrixCSC,
    current_node::Int,
    prev_node::Int,
    reduced_indices::BitVector,
    irreducible_indices::BitVector)
    # If current node is reduced stop
    if reduced_indices[current_node]
        return Int[]
    end

    push!(current_chain, current_node)

    if _is_final_node(current_node, adj_matrix, reduced_indices, irreducible_indices)
        return
    end

    reduced_indices[current_node] = true
    # Get neighbors
    neighbors = _get_neighbors(adj_matrix, current_node)

    # Determine the next node to visit. It must not be the `previous_node`.
    # This prevents the traversal from going back and forth between two nodes.
    next_node = (neighbors[1] == prev_node) ? neighbors[2] : neighbors[1]
    _get_partial_chain_recursive!(
        current_chain,
        adj_matrix,
        next_node,
        current_node,
        reduced_indices,
        irreducible_indices,
    )
    return
end

"""
    _get_degree2_nodes(adj_matrix::SparseArrays.SparseMatrixCSC, irreducible_indices::Set{Int})

Return all degree-2 nodes in the adjacency matrix, excluding irreducible indices.
"""
function _get_degree2_nodes(
    adj_matrix::SparseArrays.SparseMatrixCSC,
    irreducible_indices::BitVector,
)
    node_count = size(adj_matrix, 1)
    nodes = sizehint!(Vector{Int}(), node_count)
    for i in 1:node_count
        if irreducible_indices[i]
            continue
        end
        if _is_2degree_node(adj_matrix, i)
            push!(nodes, i)
        end
    end
    return nodes
end

"""
    find_degree2_chains(adj_matrix::SparseArrays.SparseMatrixCSC, irreducible_indices::Set{Int})

Find all chains of degree-2 nodes in a graph represented by a CSC adjacency matrix.
A chain is a sequence of connected degree-2 nodes.

Returns a dictionary mapping each starting node to its chain of node indices.
"""
function find_degree2_chains(
    adj_matrix::SparseArrays.SparseMatrixCSC,
    irreducible_indices::Set{Int},
)
    node_count = size(adj_matrix, 1)
    # Convert the exempt set into a BitVector for O(1) membership checks keyed by column
    # index. `reduced_indices` tracks nodes already consumed by a chain.
    irreducible_mask = falses(node_count)
    for i in irreducible_indices
        irreducible_mask[i] = true
    end
    reduced_indices = falses(node_count)
    arc_map = Dict{Tuple{Int, Int}, Vector{Int}}()
    degree2_nodes = _get_degree2_nodes(adj_matrix, irreducible_mask)
    for node in degree2_nodes
        if reduced_indices[node]
            continue
        end
        chain_path =
            _get_complete_chain(adj_matrix, node, reduced_indices, irreducible_mask)
        valid_chain_path = _find_longest_valid_chain(adj_matrix, chain_path)
        if !isempty(valid_chain_path)
            arc_map[valid_chain_path[1], valid_chain_path[end]] = valid_chain_path
        end
    end
    return arc_map
end

function _find_longest_valid_chain(
    adj_matrix::SparseArrays.SparseMatrixCSC,
    chain_path::Vector{Int},
)
    if _is_valid_chain(adj_matrix, chain_path)
        return chain_path
    end
    @info "Nodes $(chain_path[1]) and $(chain_path[end]) already have a parallel path or is circular, searching for valid subchains."
    # Enumerate subchain index ranges (i, j) in descending length order and return the
    # first whose endpoints form a valid chain. Avoids the prior O(n^2) materialization
    # and sort of every contiguous subchain.
    n = length(chain_path)
    for len in n:-1:3
        for i in 1:(n - len + 1)
            j = i + len - 1
            endpoint_i = chain_path[i]
            endpoint_j = chain_path[j]
            if endpoint_i != endpoint_j && adj_matrix[endpoint_i, endpoint_j] == 0
                subchain = chain_path[i:j]
                @info "found a valid subchain $subchain"
                return subchain
            end
        end
    end
    @debug "No valid subchains found; skipping chain creation"
    return Vector{Int}()
end

function _is_valid_chain(adj_matrix::SparseArrays.SparseMatrixCSC, chain_path::Vector{Int})
    if adj_matrix[chain_path[1], chain_path[end]] == 0 && chain_path[1] != chain_path[end]
        return true
    else
        return false
    end
end
