"""
Finds the set of bus numbers that belong to each connected component in the System
"""
# this function extends the PowerModels.jl implementation to accept a System
function find_connected_components(sys::PSY.System)
    a = Adjacency(sys)
    return find_connected_components(a.data, a.lookup[1])
end

# Group bus numbers into connected components of the graph whose edges are the
# off-diagonal entries of `M` holding a nonzero value.
function find_connected_components(
    M::SparseArrays.SparseMatrixCSC,
    bus_lookup::Dict{Int64, Int64},
)
    bus_numbers = Vector{Int}(undef, length(bus_lookup))
    for (bus_number, index) in bus_lookup
        bus_numbers[index] = bus_number
    end
    return Set(values(_union_find_components(M, bus_numbers)))
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

# An edge is a stored entry holding a nonzero value; every routine in this file reads it
# that way. In-place outage edits write exact zeros instead of deleting entries, and a
# reduced Ybus can store a cancelled (zero) admittance: neither couples its buses. Topology
# that must survive a cancellation lives in `adjacency_data` (±1, see
# `_repair_merged_adjacencies!`). Zeros are assumed symmetric.
_live_entry_count(vals::AbstractVector, r::AbstractUnitRange{Int}) =
    count(j -> !iszero(vals[j]), r)

# Components keyed by the bus number of their union-find root.
function _union_find_components(M::SparseArrays.SparseMatrixCSC, bus_numbers::Vector{Int})
    rows = SparseArrays.rowvals(M)
    vals = SparseArrays.nonzeros(M)
    uf = collect(1:length(bus_numbers))
    for ix in eachindex(bus_numbers)
        for j in SparseArrays.nzrange(M, ix)
            iszero(vals[j]) || union_sets!(uf, ix, rows[j])
        end
    end
    subnetworks = Dict{Int, Set{Int}}()
    for (ix, bus_number) in enumerate(bus_numbers)
        root_bus = bus_numbers[get_representative(uf, ix)]
        push!(get!(() -> Set{Int}(), subnetworks, root_bus), bus_number)
    end
    return subnetworks
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
    vals = SparseArrays.nonzeros(M)
    for (ix, bus_number) in enumerate(bus_numbers)
        if _live_entry_count(vals, SparseArrays.nzrange(M, ix)) <= 1
            @warn "Bus $bus_number is islanded"
        end
    end
    return _union_find_components(M, bus_numbers)
end

"""
Finds the subnetworks present in the considered System. This is evaluated by taking
a the ABA or Adjacency Matrix.

# Arguments
- `M::SparseArrays.SparseMatrixCSC`:
        input sparse matrix.
- `bus_numbers::Vector{Int}`:
        vector containing the indices of the system's buses.
"""
function find_subnetworks(M::SparseArrays.SparseMatrixCSC, bus_numbers::Vector{Int})
    return iterative_union_find(M, bus_numbers)
end
