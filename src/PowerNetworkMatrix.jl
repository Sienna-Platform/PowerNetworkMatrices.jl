"""
The container code for PowerNetworkMatrix is based in JuMP's Container in order to
remove the limitations of AxisArrays and the doubts about long term maintenance
https://github.com/JuliaOpt/JuMP.jl/blob/master/src/Containers/DenseAxisArray.jl
JuMP'sCopyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0.
"""

""" Type PowerNetworkMatrix gathers all the different types of Matrices considered in this package """
abstract type PowerNetworkMatrix{T} <:
              IS.InfrastructureMatrices.AbstractInfrastructureNetworkMatrix{T} end

"""
Evaluates the map linking the system's buses and branches.

# Arguments
- `buses::AbstractVector{PSY.ACBus}`:
        system's buses
"""
function make_ax_ref(buses::AbstractVector{PSY.ACBus})
    return make_ax_ref(PSY.get_number.(buses))
end

"""
Checks if repetitions are present in the dictionary mapping buses and branches.

# Arguments
- `ax::AbstractVector`:
        generic abstract vector
"""
function make_ax_ref(ax::AbstractVector)
    ref = Dict{eltype(ax), Int}()
    for (ix, el) in enumerate(ax)
        if haskey(ref, el)
            @error("Repeated index element $el. Index sets must have unique elements.")
        end
        ref[el] = ix
    end
    return ref
end
stores_transpose(::PowerNetworkMatrix) = false

# AbstractArray interface: overloading methods
Base.isempty(A::PowerNetworkMatrix) = isempty(A.data)
Base.size(A::PowerNetworkMatrix) = size(A.data)
Base.LinearIndices(A::PowerNetworkMatrix) =
    error("PowerSystems.$(typeof(A)) does not support this operation.")
Base.axes(A::PowerNetworkMatrix) = A.axes
Base.CartesianIndices(A::PowerNetworkMatrix) =
    error("PowerSystems.$(typeof(A)) does not support this operation.")

# Indexing ###################################################################

Base.eachindex(A::PowerNetworkMatrix) = CartesianIndices(size(A.data))

"""
Gets the matrix index corresponding to a given key (arc tuple, bus number, etc.)
"""
function lookup_index(i, lookup::Dict)
    return isa(i, Colon) ? Colon() : lookup[i]
end

"""
Gets the matrix index for a `PSY.Arc`, converting it to an arc tuple first.

# Arguments
- `i::PSY.Arc`:
        Power System Arc object
- `lookup::Dict`:
        Dictionary mapping arc tuples or bus numbers to matrix indices
"""
function lookup_index(i::PSY.Arc, lookup::Dict)
    return isa(i, Colon) ? Colon() : lookup[Base.to_index(i)]
end

"""
Gets the matrix index for a `PSY.ACBus`, converting it to a bus number first.

# Arguments
- `i::PSY.ACBus`:
        Power System AC bus
- `lookup::Dict`:
        Dictionary mapping arc tuples or bus numbers to matrix indices
"""
function lookup_index(i::PSY.ACBus, lookup::Dict)
    return isa(i, Colon) ? Colon() : lookup[Base.to_index(i)]
end

# Lisp-y tuple recursion trick to handle indexing in a nice type-
# stable way. The idea here is that `_to_index_tuple(idx, lookup)`
# performs a lookup on the first element of `idx` and `lookup`,
# then recurses using the remaining elements of both tuples.
# The compiler knows the lengths and types of each tuple, so
# all of the types are inferable.
function _to_index_tuple(idx::Tuple, lookup::Tuple)
    return tuple(
        lookup_index(first(idx), first(lookup)),
        _to_index_tuple(Base.tail(idx), Base.tail(lookup))...,
    )
end

# Handle the base case when we have more indices than lookups:
function _to_index_tuple(idx::NTuple{N}, ::NTuple{0}) where {N}
    return ntuple(k -> begin
            i = idx[k]
            (i == 1) ? 1 : error("invalid index $i")
        end, Val(N))
end

# Handle the base case when we have fewer indices than lookups:
_to_index_tuple(idx::NTuple{0}, lookup::Tuple) = ()

# Resolve ambiguity with the above two base cases
_to_index_tuple(idx::NTuple{0}, lookup::NTuple{0}) = ()

"""
Given the indices, gets the values of the power network matrix considered
"""
to_index(A::PowerNetworkMatrix, idx...) = _to_index_tuple(idx, A.lookup)

# Doing `Colon() in idx` is relatively slow because it involves
# a non-unrolled loop through the `idx` tuple which may be of
# varying element type. Another lisp-y recursion trick fixes that
has_colon(idx::Tuple{}) = false
has_colon(idx::Tuple) = isa(first(idx), Colon) || has_colon(Base.tail(idx))

# TODO: better error (or just handle correctly) when user tries to index with a range like a:b
# overloading other methods to consider PowerNetworkMatrix
function Base.getindex(A::PowerNetworkMatrix, row, column)
    i, j = to_index(A, row, column)
    return A.data[i, j]
end
Base.getindex(A::PowerNetworkMatrix, idx::CartesianIndex) = A.data[idx]
Base.setindex!(A::PowerNetworkMatrix, v, idx...) = A.data[to_index(A, idx...)...] = v
Base.setindex!(A::PowerNetworkMatrix, v, idx::CartesianIndex) = A.data[idx] = v

Base.IndexStyle(::Type{PowerNetworkMatrix}) = IndexAnyCartesian()

# Keys #######################################################################

"""
Structure to store the keys of a power network matrix

# Arguments
- `I<:Tuple`:
        tuple containing the indices of the matrix
"""
struct PowerNetworkMatrixKey{T <: Tuple}
    I::T
end

Base.getindex(k::PowerNetworkMatrixKey, args...) = getindex(k.I, args...)

"""
Structure to store the keys of a power network matrix

# Arguments
- `product_iter::Base.Iterators.ProductIterator{T} where T <: Tuple`:
        iterator of the indices of the network power matrix
"""
struct PowerNetworkMatrixKeys{T <: Tuple}
    product_iter::Base.Iterators.ProductIterator{T}
end

# overloading methods
Base.length(iter::PowerNetworkMatrixKeys) = length(iter.product_iter)
function Base.eltype(iter::PowerNetworkMatrixKeys)
    return PowerNetworkMatrixKey{eltype(iter.product_iter)}
end
function Base.iterate(iter::PowerNetworkMatrixKeys)
    next = iterate(iter.product_iter)
    isnothing(next) && return nothing
    return (PowerNetworkMatrixKey(next[1]), next[2])
end
function Base.iterate(iter::PowerNetworkMatrixKeys, state)
    next = iterate(iter.product_iter, state)
    isnothing(next) && return nothing
    return (PowerNetworkMatrixKey(next[1]), next[2])
end
function Base.keys(a::PowerNetworkMatrix)
    return PowerNetworkMatrixKeys(Base.Iterators.product(a.axes...))
end
Base.getindex(a::PowerNetworkMatrix, k::PowerNetworkMatrixKey) = a[k.I...]

########
# Show #
########

# Adapted printing from JuMP's implementation of the Julia's show.jl
# used in PowerNetworkMatrixs

# Copyright (c) 2009-2016: Jeff Bezanson, Stefan Karpinski, Viral B. Shah,
# and other contributors:
#
# https://github.com/JuliaLang/julia/contributors
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

function Base.summary(io::IO, A::PowerNetworkMatrix)
    _summary(io, A)
    if stores_transpose(A)
        axes = (A.axes[2], A.axes[1])
    else
        axes = A.axes
    end
    for (k, ax) in enumerate(axes)
        print(io, "    Dimension $k, ")
        show(IOContext(io, :limit => true), ax)
        println(io)
    end
    stores_transpose(A) &&
        println(io, "Note!! The data shown below corresponds to the transposed matrix.")
    print(io, "And data with size ", size(A))
    return
end

_summary(io::IO, ::T) where {T <: PowerNetworkMatrix} = println(io, "$T")

function Base.summary(
    io::IOContext{Base.GenericIOBuffer{Array{UInt8, 1}}},
    ::PowerNetworkMatrix,
)
    println(io, "PowerNetworkMatrix")
    return
end

function Base.summary(A::PowerNetworkMatrix)
    io = IOBuffer()
    Base.summary(io, A)
    String(take!(io))
    return
end

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::PowerNetworkMatrix) = Base.print_matrix(io, X.data)
end

# n-dimensional arrays
function Base.show_nd(
    io::IO,
    a::PowerNetworkMatrix,
    print_matrix::Function,
    label_slices::Bool,
)
    limit::Bool = get(io, :limit, false)
    if isempty(a)
        return
    end
    tailinds = Base.tail(Base.tail(axes(a.data)))
    nd = ndims(a) - 2
    for I in CartesianIndices(tailinds)
        idxs = I.I
        if limit
            for i in 1:nd
                ii = idxs[i]
                ind = tailinds[i]
                if length(ind) > 10
                    if ii == ind[4] && all(d -> idxs[d] == first(tailinds[d]), 1:(i - 1))
                        for j in (i + 1):nd
                            szj = size(a.data, j + 2)
                            indj = tailinds[j]
                            if szj > 10 && first(indj) + 2 < idxs[j] <= last(indj) - 3
                                @goto skip
                            end
                        end
                        #println(io, idxs)
                        print(io, "...\n\n")
                        @goto skip
                    end
                    if ind[3] < ii <= ind[end - 3]
                        @goto skip
                    end
                end
            end
        end
        if label_slices
            print(io, "[:, :, ")
            for i in 1:(nd - 1)
                show(io, a.axes[i + 2][idxs[i]])
                print(io, ", ")
            end
            show(io, a.axes[end][idxs[end]])
            println(io, "] =")
        end
        slice = view(a.data, axes(a.data, 1), axes(a.data, 2), idxs...)
        Base.print_matrix(io, slice)
        print(io, idxs == map(last, tailinds) ? "" : "\n\n")
        @label skip
    end
end

function Base.show(io::IO, array::PowerNetworkMatrix)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

Base.to_index(b::PSY.ACBus) = PSY.get_number(b)
Base.to_index(b::T) where {T <: PSY.ACBranch} = get_arc_tuple(b)
Base.to_index(b::PSY.Arc) = get_arc_tuple(b)
"""returns the raw array data of the `PowerNetworkMatrix`"""
get_data(mat::PowerNetworkMatrix) = mat.data

"""
    get_system_uuid(mat::PowerNetworkMatrix) -> Union{Base.UUID, Nothing}

Return the UUID of the system used to construct the matrix, or `nothing`
if the matrix type does not track system origin.
"""
get_system_uuid(::PowerNetworkMatrix) = nothing

"""Get the [`NetworkReduction`](@ref) data applied to this matrix, via its branch catalog."""
get_network_reduction_data(M::PowerNetworkMatrix) =
    get_network_reduction_data(get_branch_catalog(M))

"""
    _validate_system_uuid(mat::PowerNetworkMatrix, sys::PSY.System)

Validate that the matrix was constructed from the same system. Throws an
`ErrorException` if the matrix stores a system UUID that does not match
the UUID of `sys`. No-op when the matrix does not track system origin.
"""
function _validate_system_uuid(mat::PowerNetworkMatrix, sys::PSY.System)
    mat_uuid = get_system_uuid(mat)
    if !isnothing(mat_uuid) && mat_uuid != IS.get_uuid(sys)
        error(
            "System UUID mismatch: the matrix was constructed from a system with " *
            "UUID $mat_uuid, but the provided system has UUID $(IS.get_uuid(sys)). " *
            "Ensure the matrix and system originate from the same source.",
        )
    end
    return
end

"""
    returns the lookup tuple of the `PowerNetworkMatrix`. The entries correspond
    to the dimensions of the underlying `axes` tuple, and each lookup dictionary maps
    arc tuples `(from_bus, to_bus)` or bus numbers to integer indices into the stored
    data.
"""
get_lookup(mat::PowerNetworkMatrix) = mat.lookup

# A subnetwork's representative can itself be merged away by a later reduction (e.g.
# ZeroImpedanceBranchReduction folding a swing into another bus); resolve it through the
# reduction's reverse map to the surviving bus it now shares a position with.
function get_ref_bus_position(M::PowerNetworkMatrix)
    bus_lookup = get_bus_lookup(M)
    nr = get_network_reduction_data(M)
    return [get_bus_index(x, bus_lookup, nr) for x in keys(M.subnetwork_axes)]
end

"""
    get_branch_multiplier(A::PowerNetworkMatrix, branch_name::String) -> (Float64, Tuple{Int, Int})

Resolve a branch name to the retained arc that carries it, plus the factor by which matrix
entries retrieved for that arc must be scaled to represent the named branch. A branch that
owns its arc one-to-one (direct map) scales by `1.0`; a member of a parallel group scales by
its susceptance-fraction share of the group flow (`compute_parallel_multiplier`). Name-based
indexing into PTDF/LODF/VirtualPTDF uses this so per-branch values can be read from matrices
whose rows are per-arc.

Throws when the name matches no retained branch, or matches more than one parallel-group
member (name-based lookup is ambiguous).
"""
function get_branch_multiplier(A::T, branch_name::String) where {T <: PowerNetworkMatrix}
    catalog = get_branch_catalog(A)
    candidates = get(get_component_name_index(catalog), branch_name, nothing)
    if isnothing(candidates)
        error("Branch $branch_name not found in the network reduction data.")
    end
    if length(candidates) > 1
        listed = join(
            ("$(component_type) on arc $(arc)" for (component_type, arc, _) in candidates),
            ", ",
        )
        error(
            "Branch name $(branch_name) is claimed by $(length(candidates)) components " *
            "($(listed)); a bare name cannot resolve between them because component " *
            "names are unique only per type.",
        )
    end
    (_, arc_tuple, kind) = only(candidates)
    kind === :direct_branch_map && return 1.0, arc_tuple
    # A parallel-group member carries its susceptance-fraction share of the group flow.
    group = get_parallel_branch_map(get_network_reduction_data(catalog))[arc_tuple]
    for member in group
        get_name(member) == branch_name || continue
        return compute_parallel_multiplier(group, member), arc_tuple
    end
    error(
        "Branch $branch_name is indexed on arc $(arc_tuple) but no member of the group " *
        "there carries that name.",
    )
end
