"""
    DegreeTwoReduction <: NetworkReduction

Folds degree-2 buses into equivalent series branches. Additionally protects
system-derived buses (static-injection hosts, HVDC terminals, area-interchange
and `TransmissionInterface` endpoints) on top of any user set passed via
`Ybus(sys; irreducible_buses=...)`.

# Fields
- `reduce_reactive_power_injectors::Bool = true`: when `true`, buses whose only
  injectors support reactive power (e.g. a `SynchronousCondenser`, or a purely
  susceptive `FixedAdmittance`) are treated as reduction candidates. When `false`,
  such reactive-only injector hosts are kept. Buses hosting an active-power
  injector are always kept. Capability is read from the PowerSystems
  `supports_active_power` / `supports_reactive_power` traits.
"""
@kwdef struct DegreeTwoReduction <: NetworkReduction
    reduce_reactive_power_injectors::Bool = true
end
get_reduce_reactive_power_injectors(nr::DegreeTwoReduction) =
    nr.reduce_reactive_power_injectors

function get_degree2_reduction(
    data::SparseArrays.SparseMatrixCSC{Int8, Int},
    bus_lookup::Dict{Int, Int},
    exempt_bus_positions::Set{Int},
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission},
    parallel_branch_map::Dict{Tuple{Int, Int}, AbstractBranchesParallel},
)
    reverse_bus_lookup = Dict(v => k for (k, v) in bus_lookup)
    chains = find_degree2_chains(data, exempt_bus_positions)
    series_branch_map = Dict{Tuple{Int, Int}, BranchesSeries}()
    parallel_additions = Dict{Tuple{Int, Int}, AbstractBranchesParallel}()

    removed_buses = Set{Int}()
    removed_arcs = Set{Tuple{Int, Int}}()
    # Chains sharing an endpoint pair are electrically in parallel. Group them by the unordered
    # pair so traversal direction cannot split a sibling pair across two arcs: the pair is an
    # ordered tuple and which way a chain is traversed depends on interior bus numbering, so
    # siblings can present as `(A, B)` and `(B, A)` and both reach the branch maps.
    by_endpoints = Dict{Tuple{Int, Int}, Vector{BranchesSeries}}()
    for segment_ix in chains
        composite_arc = (
            reverse_bus_lookup[segment_ix[1]],
            reverse_bus_lookup[segment_ix[end]],
        )
        segments = _build_chain_segments!(
            removed_arcs,
            removed_buses,
            composite_arc,
            segment_ix,
            reverse_bus_lookup,
            direct_branch_map,
            parallel_branch_map,
        )
        key = minmax(composite_arc[1], composite_arc[2])
        push!(get!(by_endpoints, key, Vector{BranchesSeries}()), segments)
    end

    for siblings in values(by_endpoints)
        # The seed chain's orientation is the group's arc frame, matching BranchesParallel.
        first_chain = first(siblings)
        composite_arc = get_arc_tuple(first_chain)
        if length(siblings) == 1
            series_branch_map[composite_arc] = first_chain
        else
            parallel_additions[composite_arc] = BranchesParallel(siblings)
        end
    end

    reverse_series_branch_map = _make_reverse_series_branch_map(series_branch_map)
    return series_branch_map,
    parallel_additions,
    reverse_series_branch_map,
    removed_buses,
    removed_arcs
end

# Assemble one chain's `BranchesSeries` and record the arcs and interior buses it consumes.
function _build_chain_segments!(
    removed_arcs::Set{Tuple{Int, Int}},
    removed_buses::Set{Int},
    composite_arc::Tuple{Int, Int},
    segment_ix::Vector{Int},
    reverse_bus_lookup::Dict{Int, Int},
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission},
    parallel_branch_map::Dict{Tuple{Int, Int}, AbstractBranchesParallel},
)
    segment_numbers = [reverse_bus_lookup[x] for x in segment_ix]
    @assert composite_arc[1] == segment_numbers[1]
    @assert composite_arc[2] == segment_numbers[end]
    segments = BranchesSeries(composite_arc)
    for ix in 1:(length(segment_numbers) - 1)
        segment_arc = (segment_numbers[ix], segment_numbers[ix + 1])
        entries =
            _get_branch_map_entries(direct_branch_map, parallel_branch_map, segment_arc)
        # The pair's principal key is the group frame, so a member keyed the other way is
        # transposed by `_subset_two_port` when the group's two-port is assembled.
        principal_arc, principal_entry, orientation = first(entries)
        entry = if length(entries) == 1
            principal_entry
        else
            _build_chain_merge_group(
                PSY.ACTransmission[e[2] for e in entries],
                principal_arc,
            )
        end
        add_branch!(segments, entry, orientation)
        for (key, _, _) in entries
            push!(removed_arcs, key)
        end
        ix != 1 && push!(removed_buses, segment_numbers[ix])
    end
    return segments
end

# The container for several arcs resolved onto one bus pair. A homogeneous set keeps its concrete
# type; anything mixed needs `MixedBranchesParallel`. Built here rather than through
# `_make_parallel_branch_pair` / `_push_parallel_branch!` because their mixed-type arms emit a
# `@warn` about suspect input data, which an anti-parallel pair on a chain segment is not.
function _build_chain_merge_group(
    members::Vector{PSY.ACTransmission},
    arc_key::Tuple{Int, Int},
)
    member_types = unique(typeof.(members))
    if length(member_types) == 1
        T = only(member_types)
        return BranchesParallel{T}(
            T[m for m in members],
            arc_key,
            EMPTY_TWO_PORT,
            false,
        )
    end
    return MixedBranchesParallel(members, arc_key, EMPTY_TWO_PORT, false)
end

# A composite arc's members can nest: a chain segment can be a parallel group. Recursion
# registers the physical branches at the leaves, which is what `_resolve_branch_arc` answers
# with.
function _register_composite_members!(
    reverse_map::Dict{PSY.ACTransmission, Tuple{Int, Int}},
    composite_arc::Tuple{Int, Int},
    segment::AbstractReductionAggregate,
)
    for member in segment
        _register_composite_members!(reverse_map, composite_arc, member)
    end
    return
end

function _register_composite_members!(
    reverse_map::Dict{PSY.ACTransmission, Tuple{Int, Int}},
    composite_arc::Tuple{Int, Int},
    segment::PSY.ACTransmission,
)
    reverse_map[segment] = composite_arc
    return
end

function _make_reverse_series_branch_map(
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries},
)
    reverse_map = Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    for (composite_arc, entry) in series_branch_map
        _register_composite_members!(reverse_map, composite_arc, entry)
    end
    return reverse_map
end

# The key an arc between the same bus pair is already stored under, in either orientation and in
# any of the three forward maps, or `nothing` when the pair carries no arc. Anti-parallel branches
# are separate keys, so both orientations have to be probed.
function _existing_arc_key(
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission},
    parallel_branch_map::Dict{Tuple{Int, Int}, AbstractBranchesParallel},
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries},
    arc::Tuple{Int, Int},
)
    for candidate in (arc, (arc[2], arc[1]))
        if haskey(direct_branch_map, candidate) ||
           haskey(parallel_branch_map, candidate) ||
           haskey(series_branch_map, candidate)
            return candidate
        end
    end
    return nothing
end

# Every forward-map entry on `arc`'s unordered bus pair, as `(key, entry, orientation)` triples
# where `orientation` relates the stored key to `arc`'s direction.
#
# A bus pair can carry more than one key: anti-parallel branches are keyed on the raw (from, to)
# tuple, so `(a, b)` and `(b, a)` are distinct entries, while the adjacency matrix holds one entry
# per pair. A caller that resolves a degree-two chain segment to "the" entry on its pair therefore
# folds only one twin into the equivalent and leaves the other keyed on a bus the reduction is
# about to eliminate.
#
# `ThreeWindingTransformerCircuit`s are one-to-one arcs held in `direct_branch_map`. Direct entries
# come first, then parallel, each in forward-then-reverse order, so the first triple is the pair's
# principal entry and fixes the group frame when there are several.
function _get_branch_map_entries(
    direct_branch_map::Dict{Tuple{Int, Int}, PSY.ACTransmission},
    parallel_branch_map::Dict{Tuple{Int, Int}, AbstractBranchesParallel},
    arc::Tuple{Int, Int},
)
    reverse_arc = (arc[2], arc[1])
    entries = Tuple{Tuple{Int, Int}, PSY.ACTransmission, Symbol}[]
    for map in (direct_branch_map, parallel_branch_map)
        haskey(map, arc) && push!(entries, (arc, map[arc], :FromTo))
        haskey(map, reverse_arc) && push!(entries, (reverse_arc, map[reverse_arc], :ToFrom))
    end
    isempty(entries) &&
        error("Arc $arc not found in the existing network reduction mappings.")
    return entries
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
    irreducible_indices::BitVector,
)
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

Returns a vector of chains, each a vector of node indices whose first and last entries are the
chain's terminal (non-degree-2 or irreducible) nodes. Several chains may share an endpoint pair;
grouping them onto one arc is the caller's job.
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
    chains = Vector{Vector{Int}}()
    degree2_nodes = _get_degree2_nodes(adj_matrix, irreducible_mask)
    for node in degree2_nodes
        if reduced_indices[node]
            continue
        end
        chain_path =
            _get_complete_chain(adj_matrix, node, reduced_indices, irreducible_mask)
        valid_chain_path = _find_longest_valid_chain(adj_matrix, chain_path)
        if !isempty(valid_chain_path)
            push!(chains, valid_chain_path)
        end
    end
    return chains
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
