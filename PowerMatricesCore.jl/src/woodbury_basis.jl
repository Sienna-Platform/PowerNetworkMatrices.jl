"""
    _kron_component(terminals, dead, edges) -> Union{Nothing, Matrix{Float64}}

Kron reduction onto `terminals` of the Laplacian of `edges` (`(i, j, susceptance)` in bus
positions) over the nodes `terminals ∪ dead`. Returns `R[2:k, 2:k]` of the `k × k` reduced
Laplacian `R`, or `nothing` when the dead block has a pivot below
`MODF_ISLANDING_TOLERANCE` relative to its largest absolute diagonal entry.
"""
function _kron_component(
    terminals::Vector{Int},
    dead::Vector{Int},
    edges::Vector{Tuple{Int, Int, Float64}},
)::Union{Nothing, Matrix{Float64}}
    k = length(terminals)
    n = k + length(dead)
    local_index = Dict{Int, Int}()
    for (i, bus) in enumerate(terminals)
        local_index[bus] = i
    end
    for (i, bus) in enumerate(dead)
        local_index[bus] = k + i
    end
    lap = zeros(n, n)
    for (i, j, w) in edges
        a = local_index[i]
        b = local_index[j]
        a == b && continue
        lap[a, a] += w
        lap[b, b] += w
        lap[a, b] -= w
        lap[b, a] -= w
    end
    R = lap[1:k, 1:k]
    if n > k
        dead_block = lap[(k + 1):n, (k + 1):n]
        largest = maximum(abs, LinearAlgebra.diag(dead_block))
        F = LinearAlgebra.lu(dead_block; check = false)
        threshold = MODF_ISLANDING_TOLERANCE * largest
        if iszero(largest) || any(i -> abs(F.factors[i, i]) < threshold, 1:(n - k))
            return nothing
        end
        coupling = lap[(k + 1):n, 1:k]
        R .-= transpose(coupling) * (F \ coupling)
    end
    return R[2:k, 2:k]
end

# Whether each bus is in an island holding a reference bus; an empty `labels` means every bus.
function _live_buses(
    labels::Vector{Int},
    ref_positions::Vector{Int},
    n_bus::Int,
)::Vector{Bool}
    isempty(labels) && return fill(true, n_bus)
    reference = Set{Int}(labels[r] for r in ref_positions)
    return Bool[labels[i] in reference for i in 1:n_bus]
end

# From and to bus positions of an arc whose `BA` column holds both ends: the from bus holds the
# entry whose sign matches the arc's susceptance sign `sign`.
function _ba_arc_ends(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    sign::Float64,
    e::Int,
)::Tuple{Int, Int}
    rng = nzrange(BA, e)
    rv = rowvals(BA)
    i, j = rv[first(rng)], rv[last(rng)]
    return SparseArrays.nonzeros(BA)[first(rng)] * sign > 0 ? (i, j) : (j, i)
end

# From and to bus positions of arc `e`. An arc without susceptance has no BA entries and is
# read from the incidence matrix.
function _arc_ends(core::FactorCore, e::Int)::Tuple{Int, Int}
    length(nzrange(core.BA, e)) == 2 &&
        return _ba_arc_ends(core.BA, core.arc_susceptance_signs[e], e)
    row = core.A[e, :]
    buses = SparseArrays.nonzeroinds(row)
    vals = SparseArrays.nonzeros(row)
    from = buses[findfirst(==(Int8(1)), vals)]
    to = buses[findfirst(==(Int8(-1)), vals)]
    return (from, to)
end

function _arc_form_basis(
    core::FactorCore,
    change::Vector{ArcModification},
)::WoodburyBasis
    signs = core.arc_susceptance_signs
    p = Int[]
    q = Int[]
    scale = Float64[]
    signed_delta_b = Float64[]
    for m in change
        iszero(m.delta_b) && continue
        from, to = _arc_ends(core, m.arc_index)
        push!(p, from)
        push!(q, to)
        push!(scale, signs[m.arc_index])
        push!(signed_delta_b, signs[m.arc_index] * m.delta_b)
    end
    return WoodburyBasis(p, q, scale, Matrix(LinearAlgebra.Diagonal(signed_delta_b)), true)
end

# The modifications of `change` with an end live on the starting network, whose buses are
# live where `live_start` holds.
function _start_live_change(
    core::FactorCore,
    live_start::Vector{Bool},
    change::Vector{ArcModification},
)::Vector{ArcModification}
    return ArcModification[
        m for m in change if any(b -> live_start[b], _arc_ends(core, m.arc_index))
    ]
end

"""
    _woodbury_basis(core, start, start_labels, change, post_labels) -> WoodburyBasis

Minimal-rank Woodbury basis of `change` applied to the starting network, which `start` takes
the base network to. `start_labels` and `post_labels` are the island labels of the starting
and post-change networks (`Int[]` when every bus is live); a bus is live when its island holds
a reference bus.

The changed subgraph holds each changed arc between buses live after the change, weighted by
the signed susceptance it loses, and each in-service starting arc with an end that the change
kills and no end dead on the starting network, weighted by its signed starting susceptance.
Each connected component is Kron-reduced onto its live terminals. The basis is in arc form
when every component is a single changed arc between live buses, or when any component's dead
block is singular.

On a starting network other than the base network, changed arcs with both ends dead on it
leave the basis, and a change with no changed-subgraph edges has the empty node-pair basis.
Throws an `ArgumentError` when a bus dead on the starting network is live after the change.
"""
function _woodbury_basis(
    core::FactorCore,
    start::Vector{ArcModification},
    start_labels::Vector{Int},
    change::Vector{ArcModification},
    post_labels::Vector{Int},
)::WoodburyBasis
    BA = core.BA
    signs = core.arc_susceptance_signs
    n_bus = size(BA, 1)
    ref_positions = _ref_bus_positions(core)
    live_start = _live_buses(start_labels, ref_positions, n_bus)
    live_post = _live_buses(post_labels, ref_positions, n_bus)
    revived = findfirst(i -> live_post[i] && !live_start[i], 1:n_bus)
    isnothing(revived) || throw(
        ArgumentError(
            "The change makes bus $(get_bus_axis(core)[revived]) live, but it is dead on " *
            "the starting network; a change cannot reconnect a dead island.",
        ),
    )
    base_start = isempty(start)
    arc_change = base_start ? change : _start_live_change(core, live_start, change)

    edges = Tuple{Int, Int, Float64}[]
    for m in change
        iszero(m.delta_b) && continue
        from, to = _arc_ends(core, m.arc_index)
        (live_post[from] && live_post[to]) || continue
        push!(edges, (from, to, -signs[m.arc_index] * m.delta_b))
    end
    n_changed_edges = length(edges)

    killed = [i for i in 1:n_bus if live_start[i] && !live_post[i]]
    if !isempty(killed)
        bus_arcs = SparseArrays.sparse(transpose(BA))
        bus_arcs_rv = rowvals(bus_arcs)
        visited = Set{Int}()
        for x in killed, nz in nzrange(bus_arcs, x)
            e = bus_arcs_rv[nz]
            e in visited && continue
            push!(visited, e)
            from, to = _arc_ends(core, e)
            (live_start[from] && live_start[to]) || continue
            b = signs[e] * _post_modification_susceptance(core.arc_susceptances, e, start)
            abs(b) < MODF_ISLANDING_TOLERANCE && continue
            push!(edges, (from, to, b))
        end
    end

    uf = collect(1:n_bus)
    for (i, j, _) in edges
        union_sets!(uf, i, j)
    end
    component_of_root = Dict{Int, Int}()
    component_edges = Vector{Vector{Tuple{Int, Int, Float64}}}()
    for edge in edges
        root = get_representative(uf, edge[1])
        c = get!(component_of_root, root) do
            push!(component_edges, Tuple{Int, Int, Float64}[])
            length(component_edges)
        end
        push!(component_edges[c], edge)
    end

    if !base_start && isempty(edges) && !isempty(change)
        return WoodburyBasis(Int[], Int[], Float64[], zeros(0, 0), false)
    end
    if n_changed_edges == length(edges) && all(es -> length(es) == 1, component_edges)
        return _arc_form_basis(core, arc_change)
    end

    p = Int[]
    q = Int[]
    blocks = Matrix{Float64}[]
    for es in component_edges
        nodes = Set{Int}()
        for (i, j, _) in es
            push!(nodes, i, j)
        end
        terminals = sort!([b for b in nodes if live_post[b]])
        length(terminals) < 2 && continue
        dead = sort!([b for b in nodes if !live_post[b]])
        R = _kron_component(terminals, dead, es)
        isnothing(R) && return _arc_form_basis(core, arc_change)
        for t in @view terminals[2:end]
            push!(p, t)
            push!(q, terminals[1])
        end
        push!(blocks, -R)
    end

    rank = length(p)
    D = zeros(rank, rank)
    offset = 0
    for block in blocks
        r = size(block, 1)
        D[(offset + 1):(offset + r), (offset + 1):(offset + r)] .= block
        offset += r
    end
    return WoodburyBasis(p, q, ones(rank), D, false)
end
