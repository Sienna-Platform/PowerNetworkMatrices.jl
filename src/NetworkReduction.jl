"""
    NetworkReduction

Abstract base type for all network reduction algorithms used in power network analysis.
Network reductions are mathematical transformations that eliminate buses and branches
while preserving the electrical behavior of the remaining network elements.

Concrete implementations include:
- [`RadialReduction`](@ref): Eliminates radial (dangling) buses and branches
- [`DegreeTwoReduction`](@ref): Eliminates buses with exactly two connections
- [`WardReduction`](@ref): Reduces external buses while preserving study bus behavior

# Applying reductions

Reductions are applied only through the `network_reductions` keyword, a
`Vector{NetworkReduction}` accepted by every matrix constructor that builds from a
`System` — [`PTDF`](@ref), [`Ybus`](@ref), [`BA_Matrix`](@ref), [`ABA_Matrix`](@ref),
[`LODF`](@ref), [`VirtualPTDF`](@ref), [`VirtualLODF`](@ref), and [`VirtualMODF`](@ref).
The specs are applied in vector order; the default (an empty vector) applies no
reduction.

```julia
ptdf = PTDF(sys; network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()])
```

Write the vector with the `NetworkReduction[...]` element-type prefix: a bare
`[RadialReduction()]` infers the narrower `Vector{RadialReduction}`, which the keyword
(typed `Vector{NetworkReduction}`) will not accept. The prefix is unnecessary only when
the vector already holds two or more different spec types.

# Ordering and validation rules

The applied specs are validated at construction (a violation throws or warns):

- each reduction **type** may appear at most once;
- [`WardReduction`](@ref) must be **last** when present;
- `ZeroImpedanceBranchReduction` may not be listed — it is auto-applied during
  [`Ybus`](@ref) construction;
- a [`DegreeTwoReduction`](@ref) placed before a [`RadialReduction`](@ref) warns, since
  running radial first usually exposes more degree-two buses for the second pass.

# Reading back what changed

The applied reductions are recorded on the matrix; retrieve the record with
[`get_network_reduction_data`](@ref) and inspect it through the
[`NetworkReductionData`](@ref) accessors (e.g. `get_removed_buses`, `get_removed_arcs`,
and the `keys(get_bus_reduction_map(nrd))` bus-survival check).
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
