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

function Base.:(==)(x::T1, y::T1) where {T1 <: NetworkReduction}
    for field in fieldnames(T1)
        if getfield(x, field) != getfield(y, field)
            return false
        end
    end
    return true
end
