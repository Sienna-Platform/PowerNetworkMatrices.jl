"""
The container code for PowerNetworkMatrix is based in JuMP's Container in order to
remove the limitations of AxisArrays and the doubts about long term maintenance
https://github.com/JuliaOpt/JuMP.jl/blob/master/src/Containers/DenseAxisArray.jl
JuMP'sCopyright 2017, Iain Dunning, Joey Huchette, Miles Lubin, and contributors
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0.
"""

""" Type PowerNetworkMatrix gathers all the different types of Matrices considered in this package """
abstract type PowerNetworkMatrix{T} <: AbstractArray{T, 2} end

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
Gets bus indices to a certain branch index
"""
function lookup_index(i, lookup::Dict)
    return isa(i, Colon) ? Colon() : lookup[i]
end

"""
Gets bus indices to a certain branch name

# Arguments
- `i::PSY.ACBranch`:
        Power System AC branch
- `lookup::Dict`:
        Dictionary mapping branch and buses
"""
function lookup_index(i::PSY.Arc, lookup::Dict)
    return isa(i, Colon) ? Colon() : lookup[Base.to_index(i)]
end

"""
Gets bus indices to a certain branch name

# Arguments
- `i::PSY.ACBranch`:
        Power System AC branch
- `lookup::Dict`:
        Dictionary mapping branches and buses
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
    return next === nothing ? nothing : (PowerNetworkMatrixKey(next[1]), next[2])
end
function Base.iterate(iter::PowerNetworkMatrixKeys, state)
    next = iterate(iter.product_iter, state)
    return next === nothing ? nothing : (PowerNetworkMatrixKey(next[1]), next[2])
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
    returns the lookup tuple of the `PowerNetworkMatrix`. The first entry corresponds
    to the first dimension and the second entry corresponds to the second dimension. For
    instance in Ybus the first dimension is buses and second dimension is buses too, and in
    PTDF the first dimension is branches and the second dimension is buses
"""
get_lookup(mat::PowerNetworkMatrix) = mat.lookup

function get_branch_multiplier(A::T, branch_name::String) where {T <: PowerNetworkMatrix}
    nr = A.network_reduction_data
    if isempty(nr.direct_branch_name_map)
        populate_direct_branch_name_map!(nr)
    end
    if haskey(nr.direct_branch_name_map, branch_name)
        arc_tuple = nr.direct_branch_name_map[branch_name]
        return 1.0, arc_tuple
    end

    if !isempty(nr.reverse_parallel_branch_map)
        for (k, v) in nr.reverse_parallel_branch_map
            if branch_name == PSY.get_name(k)
                parallel_branch_set = nr.parallel_branch_map[v]
                multiplier = compute_parallel_multiplier(parallel_branch_set, branch_name)
                return multiplier, v
            end
        end
    end

    if !isempty(nr.reverse_transformer3W_map)
        if branch_name in PSY.get_name.(keys(nr.reverse_transformer3W_map))
            throw(
                IS.ConflictingInputsError(
                    "Branch $branch_name is a three-winding transformer, it can't be used to index directly in to a $T.",
                ),
            )
        end
    end

    error("Branch $branch_name not found in the network reduction data.")
    return
end
