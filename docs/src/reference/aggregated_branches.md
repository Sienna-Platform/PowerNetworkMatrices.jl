# Aggregated-branch types

When a [network reduction](network_reduction.md) folds several physical branches
into one equivalent arc, PowerNetworkMatrices represents the group with an
aggregated-branch object. This page documents those types and the accessors that
compute their equivalent electrical parameters and ratings. It is a dry
reference; for the reasoning behind the rating policies see
[Equivalent representation of reduced branches](../explanation/equivalent_branches.md).

All the types below subtype [`ACTransmission`](@extref PowerSystems.ACTransmission),
so they can stand in for a real branch wherever the reduction machinery expects
one. None of them are exported except the three rating-strategy functions
[`get_sum_of_max_rating`](@ref), [`get_single_element_contingency_rating`](@ref),
and [`get_impedance_averaged_rating`](@ref); the remaining accessors and the
aggregate types themselves are internal.

## `BranchesParallel`, `MixedBranchesParallel`, `AbstractBranchesParallel`

Defined in `src/BranchesParallel.jl`. A parallel group represents two or more
branches spanning the **same** bus pair (a double / multi circuit).

```julia
abstract type AbstractBranchesParallel <: PSY.ACTransmission end

mutable struct BranchesParallel{T <: PSY.ACTransmission} <: AbstractBranchesParallel
    branches::Vector{T}
    arc_key::Tuple{Int, Int}
    equivalent_ybus::Union{Matrix{YBUS_ELTYPE}, Nothing}
end

mutable struct MixedBranchesParallel <: AbstractBranchesParallel
    branches::Vector{PSY.ACTransmission}
    arc_key::Tuple{Int, Int}
    equivalent_ybus::Union{Matrix{YBUS_ELTYPE}, Nothing}
end
```

  - `BranchesParallel{T}` is the homogeneous group: every member has the same
    concrete branch type `T`. The inner constructor errors if `T` is not concrete
    — use `MixedBranchesParallel` for heterogeneous groups.
  - `MixedBranchesParallel` holds members of differing concrete types under the
    abstract element type `PSY.ACTransmission`.
  - `arc_key` is the group's canonical arc in original bus numbers (the seed
    member's orientation); it is remapped through the
    [`NetworkReductionData`](network_reduction.md) on read, so orientation does
    not depend on the order of `branches`.
  - `equivalent_ybus` caches the 2×2 equivalent admittance block; it is `nothing`
    until populated.

Convenience constructors infer `arc_key` from the first member:

```julia
BranchesParallel(branches::Vector{T})            # T concrete -> BranchesParallel{T}
MixedBranchesParallel(branches::Vector{<:PSY.ACTransmission})
```

Both support `add_branch!(group, branch)`, iteration over members
(`for br in group`), `length`, and `PSY.get_available`. `get_name` synthesizes a
group name from the longest common name prefix suffixed with `"double_circuit"`.
The equivalent series susceptance of a parallel group is the sum of the members'
series susceptances.

## `BranchesSeries`

Defined in `src/BranchesSeries.jl`. Represents a chain of branches connected in
series through eliminated degree-2 buses, as produced by
[`DegreeTwoReduction`](network_reduction.md).

```julia
mutable struct BranchesSeries <: PSY.ACTransmission
    branches::Dict{DataType, Vector{<:PSY.ACTransmission}}
    needs_insertion_order::Bool
    insertion_order::Vector{Tuple{DataType, Int}}
    segment_orientations::Vector{Symbol}
    equivalent_ybus::Union{Matrix{YBUS_ELTYPE}, Nothing}
end
```

Members are bucketed by concrete type; `insertion_order` preserves the physical
ordering along the chain when the chain mixes types. A member may itself be a
parallel group, so a series chain can nest parallel blocks.
`BranchesSeries()` builds an empty chain; `add_branch!(bs, branch, orientation)`
appends a segment with its `:FromTo` / `:ToFrom` orientation. The type supports
iteration in chain order, `length`, and `PSY.get_available`. The equivalent
series susceptance is the reciprocal of the sum of member reciprocal
susceptances.

## `ThreeWindingTransformerWinding` (internal)

Defined in `src/ThreeWindingTransformerWinding.jl`. This type is **internal** —
it is not exported and must not be constructed directly by users or added to a
system. It represents a single winding of a `PSY.ThreeWindingTransformer`,
decomposed for matrix construction (a three-winding transformer is modeled as a
wye to a zero-injection star bus).

```julia
struct ThreeWindingTransformerWinding{T <: PSY.ThreeWindingTransformer} <:
       PSY.ACTransmission
    transformer::T
    winding_number::Int   # 1 (primary), 2 (secondary), or 3 (tertiary)
end
```

Structural accessors:

```julia
get_transformer(tw)        # -> the parent PSY.ThreeWindingTransformer
get_winding_number(tw)     # -> Int in {1, 2, 3}
get_transformer_type(tw)   # -> the concrete transformer type T
get_name(tw)               # -> "<transformer name>_winding_<n>"
```

Its equivalent-parameter accessors (below) read the winding-specific `r`, `x`,
`b`, rating, tap, and availability directly from the parent transformer.

## Equivalent-parameter accessors

For a `ThreeWindingTransformerWinding`, the per-winding electrical parameters are
read directly:

```julia
get_equivalent_r(tw)          # winding series resistance (p.u.)
get_equivalent_x(tw)          # winding series reactance (p.u.)
get_equivalent_b(tw)          # NamedTuple (from = ..., to = ...); shunt only on winding 1
get_equivalent_tap(tw)        # turns ratio; PhaseShiftingTransformer3W windings only
get_equivalent_α(tw)          # phase-shift angle; PhaseShiftingTransformer3W windings only
```

For `BranchesParallel` / `MixedBranchesParallel` and `BranchesSeries`, the lumped
`r` / `x` / `b` / `tap` / `shift` are not stored on the group; they are derived
from the group's equivalent Ybus block by
`get_equivalent_physical_branch_parameters` (see below), which returns an
`EquivalentBranch` (documented below).

### `EquivalentBranch`

Defined in `src/EquivalentBranch.jl`. A plain record of the lumped physical
parameters of a reduced branch group.

```julia
mutable struct EquivalentBranch
    equivalent_r::Float64        # series resistance (p.u.)
    equivalent_x::Float64        # series reactance (p.u.)
    equivalent_g_from::Float64   # shunt conductance at the "from" bus (p.u.)
    equivalent_b_from::Float64   # shunt susceptance at the "from" bus (p.u.)
    equivalent_g_to::Float64     # shunt conductance at the "to" bus (p.u.)
    equivalent_b_to::Float64     # shunt susceptance at the "to" bus (p.u.)
    equivalent_tap::Float64      # transformer tap ratio
    equivalent_shift::Float64    # phase-shift angle (radians)
end
```

Field getters, one per field:

```julia
get_equivalent_r(eb)
get_equivalent_x(eb)
get_equivalent_g_from(eb)
get_equivalent_b_from(eb)
get_equivalent_g_to(eb)
get_equivalent_b_to(eb)
get_equivalent_tap(eb)
get_equivalent_shift(eb)
```

### `get_equivalent_physical_branch_parameters`

```julia
get_equivalent_physical_branch_parameters(
    segment::Union{AbstractBranchesParallel, BranchesSeries},
    nr::NetworkReductionData,
) -> EquivalentBranch
```

Defined in `src/common.jl`. Returns the lumped `EquivalentBranch` for a parallel
or series group. It lazily populates the group's cached `equivalent_ybus` (via
`populate_equivalent_ybus!`) on first call, then decomposes that 2×2 admittance
block back into `r`, `x`, shunt `g`/`b` on each end, `tap`, and `shift`. It lives
in `common.jl` (rather than beside the group types) because it needs the
`NetworkReductionData` type, which is included later in the module.

## Rating and availability accessors

### Series and per-branch ratings

`get_equivalent_rating` and `get_equivalent_emergency_rating` are defined for
every branch-like type:

  - A plain `PSY.ACTransmission` branch returns `PSY.get_rating` /
    `PSY.get_rating_b` (falling back to the normal rating when `rating_b` is
    unset). A `PSY.GenericArcImpedance` returns its `max_flow` as a proxy.
  - A `BranchesSeries` is limited by its weakest link: the rating is the
    **minimum** member rating (a nested parallel member contributes its N-1
    single-element-contingency rating). The emergency rating is likewise the
    minimum member emergency rating.
  - A `ThreeWindingTransformerWinding` returns the winding-specific rating, or the
    parent transformer's rating when the winding rating is zero.

```julia
get_equivalent_rating(branch)             # -> Float64
get_equivalent_emergency_rating(branch)   # -> Float64
get_equivalent_available(branch)          # -> Bool (all members must be available)
get_equivalent_α(branch)                  # phase-shift aggregate
```

`get_equivalent_available` (aliased to `PSY.get_available`) requires **all**
members to be available for the group to be available.

### Parallel-group rating strategies

A parallel group has three distinct rating aggregations, each answering a
different operational question. These three functions are exported.

```julia
get_sum_of_max_rating(bp::AbstractBranchesParallel)
```

Sum of the individual branch ratings — each circuit treated as independently
loadable to its own thermal limit. Least conservative; assumes unconstrained flow
steering across the group.

```julia
get_single_element_contingency_rating(bp::AbstractBranchesParallel)
```

N-1 rating: surviving capacity after the largest-rated circuit trips,
``\sum_i S_i - \max_i S_i``. Zero for a group of one branch. This is the value a
nested parallel block contributes to a series chain's weakest-link rating.

```julia
get_impedance_averaged_rating(bp::AbstractBranchesParallel)
```

Susceptance-weighted average of the individual ratings,
``\sum_i f_i \, S_i`` with ``f_i = b_i / \sum_k b_k`` — reflecting how DC flow
physically splits across the group. Throws `ArgumentError` if the total series
susceptance is zero or non-finite.

The parallel-group emergency rating (`get_equivalent_emergency_rating`) is the
sum of the members' emergency ratings, since emergency conditions permit using
the full aggregate capacity.

## See also

  - [Network reduction reference](network_reduction.md) — the reduction specs and
    the `NetworkReductionData` branch maps that hold these aggregated types.
  - [Equivalent representation of reduced branches](../explanation/equivalent_branches.md)
    — why the parallel rating policies differ.
  - [Full public API reference](public.md).
