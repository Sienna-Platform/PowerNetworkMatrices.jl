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
