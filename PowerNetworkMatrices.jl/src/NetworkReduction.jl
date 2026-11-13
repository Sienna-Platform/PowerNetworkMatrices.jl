"""
    NetworkReduction

Abstract base type for all network reduction algorithms used in power network analysis.
Network reductions are mathematical transformations that eliminate buses and branches 
while preserving the electrical behavior of the remaining network elements.

Concrete implementations include:
- [`RadialReduction`](@ref): Eliminates radial (dangling) buses and branches
- [`DegreeTwoReduction`](@ref): Eliminates buses with exactly two connections
- [`WardReduction`](@ref): Reduces external buses while preserving study bus behavior
"""
abstract type NetworkReduction end

function _fieldwise_equal(a::T, b::T) where {T}
    return all(f -> getfield(a, f) == getfield(b, f), fieldnames(T))
end

Base.:(==)(x::T, y::T) where {T <: NetworkReduction} = _fieldwise_equal(x, y)
