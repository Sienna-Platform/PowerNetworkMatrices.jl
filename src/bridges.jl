# Bridges of an undirected multigraph by iterative Tarjan (no recursion, so deep radial chains
# cannot overflow the stack). Parent edges are skipped by edge id, not by parent vertex, so a
# second edge between the same pair counts as a back edge and parallel edges are never bridges.
# Self-loops are ignored.
#
# `preorder` lists the vertices in DFS discovery order; `tin[v]` is `v`'s position in it and
# `last[v]` the position of the last vertex of `v`'s DFS subtree, so a subtree is the contiguous
# slice `preorder[tin[v]:last[v]]`. Removing bridge `e` separates exactly the DFS subtree of its
# child endpoint `far_end[e]` (zero for an edge that is not a bridge).
struct BridgeTree
    is_bridge::BitVector
    far_end::Vector{Int}
    preorder::Vector{Int}
    tin::Vector{Int}
    last::Vector{Int}
end

is_bridge(b::BridgeTree, e::Int) = b.is_bridge[e]

# The vertices cut off from the rest of their component when bridge `e` is removed.
function bridge_far_side(b::BridgeTree, e::Int)
    is_bridge(b, e) || error("Edge $e is not a bridge.")
    c = b.far_end[e]
    return view(b.preorder, b.tin[c]:b.last[c])
end

function _other_end(edge::Tuple{Int, Int}, u::Int)
    if edge[1] == u
        return edge[2]
    end
    return edge[1]
end

# `edges` holds vertex pairs in `1:n_vertices`; the result is indexed by edge position.
function find_bridges(n_vertices::Int, edges::AbstractVector{Tuple{Int, Int}})
    ptr = zeros(Int, n_vertices + 1)
    ptr[1] = 1
    for (u, v) in edges
        u == v && continue
        ptr[u + 1] += 1
        ptr[v + 1] += 1
    end
    cumsum!(ptr, ptr)
    incident = Vector{Int}(undef, ptr[end] - 1)
    cursor = ptr[1:n_vertices]
    for (e, (u, v)) in enumerate(edges)
        u == v && continue
        incident[cursor[u]] = e
        cursor[u] += 1
        incident[cursor[v]] = e
        cursor[v] += 1
    end

    tin = zeros(Int, n_vertices)
    low = zeros(Int, n_vertices)
    last = zeros(Int, n_vertices)
    preorder = zeros(Int, n_vertices)
    parent_edge = zeros(Int, n_vertices)
    is_bridge = falses(length(edges))
    far_end = zeros(Int, length(edges))
    stack = Int[]
    timer = 0
    for root in 1:n_vertices
        iszero(tin[root]) || continue
        timer += 1
        tin[root] = timer
        low[root] = timer
        preorder[timer] = root
        cursor[root] = ptr[root]
        push!(stack, root)
        while !isempty(stack)
            u = stack[end]
            if cursor[u] < ptr[u + 1]
                e = incident[cursor[u]]
                cursor[u] += 1
                e == parent_edge[u] && continue
                w = _other_end(edges[e], u)
                if iszero(tin[w])
                    timer += 1
                    tin[w] = timer
                    low[w] = timer
                    preorder[timer] = w
                    cursor[w] = ptr[w]
                    parent_edge[w] = e
                    push!(stack, w)
                else
                    low[u] = min(low[u], tin[w])
                end
                continue
            end
            pop!(stack)
            last[u] = timer
            e = parent_edge[u]
            iszero(e) && continue
            p = _other_end(edges[e], u)
            low[p] = min(low[p], low[u])
            if low[u] > tin[p]
                is_bridge[e] = true
                far_end[e] = u
            end
        end
    end
    return BridgeTree(is_bridge, far_end, preorder, tin, last)
end
