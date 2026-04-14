function _goderya(ybus::SparseArrays.SparseMatrixCSC)
    node_count = size(ybus)[1]
    max_I = node_count^2
    I, J, val = SparseArrays.findnz(ybus)
    T = SparseArrays.sparse(I, J, ones(Int, length(val)))
    T_ = T * T
    for n in 1:(node_count - 1)
        I, _, _ = SparseArrays.findnz(T_)
        if length(I) == max_I
            @info "The System has no islands"
            break
        elseif length(I) < max_I
            temp = T_ * T
            I_temp, _, _ = SparseArrays.findnz(temp)
            if all(I_temp == I)
                @warn "The system contains islands" maxlog = 1
            end
            T_ = temp
        else
            @assert false
        end
        #@assert n < node_count - 1
    end
    return I
end

"""
    validate_connectivity(M, nodes, bus_lookup; connectivity_method) -> Bool

Check whether the network described by matrix `M` is fully connected, using the
specified connectivity algorithm.

# Arguments
- `M`: Matrix representation of the network (e.g., admittance or adjacency matrix)
- `nodes::Vector{PSY.ACBus}`: AC buses in the network
- `bus_lookup::Dict{Int64, Int64}`: Mapping from bus numbers to matrix indices
- `connectivity_method::Function`: Algorithm to use (default: `goderya_connectivity`)

# Returns
- `Bool`: `true` if the network is fully connected, `false` otherwise
"""
function validate_connectivity(
    M,
    nodes::Vector{PSY.ACBus},
    bus_lookup::Dict{Int64, Int64};
    connectivity_method::Function = goderya_connectivity,
)
    connected = connectivity_method(M, nodes, bus_lookup)
    return connected
end

function goderya_connectivity(M, nodes::Vector{PSY.ACBus}, bus_lookup::Dict{Int64, Int64})
    @info "Validating connectivity with Goderya algorithm"
    length(nodes) > 15_000 &&
        @warn "The Goderya algorithm is memory intensive on large networks and may not scale well, try `connectivity_method = dfs_connectivity"

    I = _goderya(M)

    node_count = length(nodes)
    connections = Dict([i => count(x -> x == i, I) for i in Set(I)])

    if length(Set(I)) == node_count
        connected = true
        if any(values(connections) .!= node_count)
            cc = Set(values(connections))
            @warn "Network has at least $(length(cc)) connected components with $cc nodes"
            connected = false
        end
    else
        disconnected_nodes = PSY.get_name.(nodes[setdiff(values(bus_lookup), I)])
        @warn "Principal connected component does not contain:" disconnected_nodes
        connected = false
    end
    return connected
end

"""
Finds the set of bus numbers that belong to each connected component in the System
"""
# this function extends the PowerModels.jl implementation to accept a System
function find_connected_components(sys::PSY.System)
    a = Adjacency(sys)
    return find_connected_components(a.data, a.lookup[1])
end

# this function extends the PowerModels.jl implementation to accept an adjacency matrix and bus lookup
function find_connected_components(M, bus_lookup::Dict{Int64, Int64})
    pm_buses = Dict([i => Dict("bus_type" => 1, "bus_i" => b) for (i, b) in bus_lookup])

    arcs = findall((LinearAlgebra.UpperTriangular(M) - LinearAlgebra.I) .!= 0)
    pm_branches = Dict([
        i => Dict("f_bus" => a[1], "t_bus" => a[2], "br_status" => 1) for
        (i, a) in enumerate(arcs)
    ],)

    data = Dict("bus" => pm_buses, "branch" => pm_branches)
    cc = PSY.calc_connected_components(data)
    bus_decode = Dict(value => key for (key, value) in bus_lookup)
    connected_components = Vector{Set{Int64}}()
    for c in cc
        push!(connected_components, Set([bus_decode[b] for b in c]))
    end
    return Set(connected_components)
end

function dfs_connectivity(M, ::Vector{PSY.ACBus}, bus_lookup::Dict{Int64, Int64})
    @info "Validating connectivity with depth first search (network traversal)"
    cc = find_connected_components(M, bus_lookup)
    if length(cc) != 1
        @warn "Network has at least $(length(cc)) connected components with $(length.(cc)) nodes"
        connected = false
    else
        connected = true
    end
    return connected
end

"""Find part of the union-find disjoint set data structure. Vector because nodes are 1:n."""
function get_representative(uf::Vector{Int}, x::Int)
    while uf[x] != x
        uf[x] = uf[uf[x]] # path compression
        x = uf[x]
    end
    return x
end

"""Union part of the union-find disjoint set data structure. Vector because nodes are 1:n."""
function union_sets!(uf::Vector{Int}, x::Int, y::Int)
    x == y && return
    rootX = get_representative(uf, x)
    rootY = get_representative(uf, y)
    if rootX != rootY
        uf[rootY] = rootX
    end
end

"""
    iterative_union_find(M::SparseArrays.SparseMatrixCSC, bus_numbers::Vector{Int})

Find connected subnetworks using iterative union-find algorithm.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`: Sparse matrix representing network connectivity
- `bus_numbers::Vector{Int}`: Vector containing the bus numbers of the system

# Returns
- `Dict{Int, Set{Int}}`: Dictionary mapping representative bus numbers to sets of connected buses
"""
function iterative_union_find(M::SparseArrays.SparseMatrixCSC, bus_numbers::Vector{Int})
    @info "Finding subnetworks via iterative union find"
    rows = SparseArrays.rowvals(M)
    # find connected components working with indices, so can use a vector instead of a set.
    # uf for union-find data structure: initially, each bus as its own set of a single element.
    uf = collect(1:size(bus_numbers, 1))
    for ix in 1:size(bus_numbers, 1)
        neighbors = SparseArrays.nzrange(M, ix)
        if length(neighbors) <= 1
            @warn "Bus $(bus_numbers[ix]) is islanded"
            continue
        end
        for j in neighbors
            row_ix = rows[j]
            union_sets!(uf, ix, row_ix)
        end
    end
    for i in 1:length(uf)
        uf[i] = get_representative(uf, i)
    end
    # now we have the representatives, so assemble the subnetworks.
    num_subnetworks = length(unique(uf))
    avg_size = div(size(bus_numbers, 1), num_subnetworks)
    subnetworks = Dict{Int, Set{Int}}()
    for (representative, bus_num) in zip(uf, bus_numbers)
        if !haskey(subnetworks, bus_numbers[representative])
            subnetworks[bus_numbers[representative]] = Set{Int}()
            sizehint!(subnetworks[bus_numbers[representative]], avg_size)
        end
        push!(subnetworks[bus_numbers[representative]], bus_num)
    end
    return subnetworks
end

"""
    depth_first_search(M::SparseArrays.SparseMatrixCSC, bus_numbers::Vector{Int})

Find connected subnetworks using depth-first search algorithm.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`: Sparse matrix representing network connectivity
- `bus_numbers::Vector{Int}`: Vector containing the bus numbers of the system

# Returns
- `Dict{Int, Set{Int}}`: Dictionary mapping representative bus numbers to sets of connected buses
"""
function depth_first_search(M::SparseArrays.SparseMatrixCSC, bus_numbers::Vector{Int})
    @info "Finding subnetworks via depth first search"
    rows = SparseArrays.rowvals(M)
    touched = Set{Int}()
    subnetworks = Dict{Int, Set{Int}}()
    for (ix, bus_number) in enumerate(bus_numbers)
        neighbors = SparseArrays.nzrange(M, ix)
        if length(neighbors) <= 1
            @warn "Bus $bus_number is islanded"
            subnetworks[bus_number] = Set{Int}(bus_number)
            continue
        end
        for j in SparseArrays.nzrange(M, ix)
            row_ix = rows[j]
            if bus_number ∉ touched
                push!(touched, bus_number)
                subnetworks[bus_number] = Set{Int}(bus_number)
                _dfs(row_ix, M, bus_numbers, subnetworks[bus_number], touched)
            end
        end
    end
    return subnetworks
end

function _dfs(
    index::Int,
    M::SparseArrays.SparseMatrixCSC,
    bus_numbers::Vector{Int},
    bus_group::Set{Int},
    touched::Set{Int},
)
    rows = SparseArrays.rowvals(M)
    for j in SparseArrays.nzrange(M, index)
        row_ix = rows[j]
        if bus_numbers[row_ix] ∉ touched
            push!(touched, bus_numbers[row_ix])
            push!(bus_group, bus_numbers[row_ix])
            _dfs(row_ix, M, bus_numbers, bus_group, touched)
        end
    end
    return
end

"""
Finds the subnetworks present in the considered System. This is evaluated by taking
a the ABA or Adjacency Matrix.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`:
        input sparse matrix.
- `bus_numbers::Vector{Int}`:
        vector containing the indices of the system's buses.
- `subnetwork_algorithm::Function`:
        algorithm for computing subnetworks. Valid options are iterative_union_find (default) and depth_first_search
"""
function find_subnetworks(
    M::SparseArrays.SparseMatrixCSC,
    bus_numbers::Vector{Int};
    subnetwork_algorithm::Function = iterative_union_find,
)
    return subnetwork_algorithm(M, bus_numbers)
end
