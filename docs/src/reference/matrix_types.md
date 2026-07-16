# Matrix type reference

This page describes each matrix type PowerNetworkMatrices exports: its
constructor signatures, keyword arguments, and what its entries represent. It is
a reference companion to the [matrix overview and indexing hub](network_matrices_overview.md),
which describes how `A[row, col]` resolves across all of these types. For the
authoritative, auto-generated docstrings of every symbol, see the
[full public API](public.md).

Unless noted otherwise, every type below is a subtype of the common supertype
`PowerNetworkMatrix` and shares the same indexing and accessor interface.

## The `PowerNetworkMatrix` supertype

All matrix types are subtypes of the abstract array type

```julia
abstract type PowerNetworkMatrix{T} <: AbstractArray{T, 2} end
```

Because it is an `AbstractArray{T,2}`, every concrete matrix supports `size`,
`axes`, and `getindex`. The package overloads indexing so that rows and columns
are addressed by *power-system identifiers* (bus numbers, arc tuples, branch
names, `PSY` components) rather than raw integer positions — see the
[overview hub](network_matrices_overview.md) for the full set of accepted key
types. Concrete types carry a `data` field (dense `Matrix` or
`SparseArrays.SparseMatrixCSC`), an `axes` tuple of the row/column identifier
vectors, a `lookup` tuple of `Dict`s mapping identifiers to integer positions,
and a `network_reduction_data::NetworkReductionData` describing any reduction
applied at construction.

Several types store their `data` in **transposed** form for computational
efficiency (`stores_transpose` returns `true` for `PTDF`, `LODF`, and
`BA_Matrix`). Indexing hides this: `ptdf[arc, bus]` returns the logical entry
regardless of internal orientation. Use the type-specific data accessors
(`get_ptdf_data`, `get_lodf_data`) to obtain the data in the standard
(non-transposed) orientation.

## Distribution-factor matrices

`PTDF` and `LODF` are dense distribution-factor matrices. They share
construction options (`linear_solver`, `tol`) and arc-tuple indexing, and differ
only in what each entry represents.

### `PTDF`

The Power Transfer Distribution Factor matrix. Entry `PTDF[arc, bus]` is the
sensitivity of the flow on `arc` to a unit power injection at `bus`, under the DC
power-flow approximation. Rows are arcs, columns are buses. The data is stored
transposed (bus × arc); `get_ptdf_data(ptdf)` returns it as arcs × buses.

```julia
PTDF(sys::PSY.System;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    kwargs...,
)

PTDF(ybus::Ybus;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)

PTDF(A::IncidenceMatrix, BA::BA_Matrix;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)

PTDF(filename::AbstractString)   # deserialize a previously saved PTDF from HDF5
```

Keyword arguments:

  - `dist_slack::Dict{Int, Float64}` — distributed-slack weights, mapping bus
    numbers to participation factors. The empty default uses a single reference
    bus. Factors are normalized to sum to 1.0. (Note the `Dict` type is specific to
    `PTDF`/`VirtualPTDF`; the virtual LODF/MODF use a `Vector{Float64}` instead.)
  - `linear_solver` — backend used for the matrix inversion/solve: `"KLU"`
    (default off-Apple), `"Dense"`, `"MKLPardiso"` (weak-dependency extension), or
    `"AppleAccelerateLU"` (default on macOS ≥ 15.5).
  - `tol::Union{Float64, AutoTolerance}` — sparsification tolerance, default
    `DEFAULT_AUTO_TOLERANCE`. A `Float64` applies a fixed absolute cutoff and
    stores the result as a `SparseMatrixCSC`; an `AutoTolerance` is a no-op on the
    dense `PTDF` path (it preserves the `Matrix{Float64}` storage). See the
    [tolerance and solver reference](tolerance_and_solvers.md).
  - `kwargs...` (system constructor only) — forwarded to `Ybus`; this is how
    network reductions reach a `PTDF`: pass
    `network_reductions = [RadialReduction(), ...]`. There is no `reduce_*` kwarg
    on the PTDF constructor.

The `PTDF(A, BA)` and `PTDF(ybus)` constructors reuse prerequisite matrices and
are cheaper when those are already built. `PTDF(filename)` reads a serialized
PTDF; only `PTDF` supports HDF5 serialization, and reduction data is not
persisted (see the [serialization reference](serialization.md)).

### `LODF`

The Line Outage Distribution Factor matrix. Entry `LODF[monitored, outaged]` is
the fraction of the pre-outage flow on `outaged` that redistributes onto
`monitored` when `outaged` is removed. **Both** dimensions are arcs. Diagonal
entries are structurally `-1.0` (the outaged arc loses all its own flow).

```julia
LODF(sys::PSY.System;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)

LODF(A::IncidenceMatrix, PTDFm::PTDF;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)

LODF(A::IncidenceMatrix, ABA::ABA_Matrix, BA::BA_Matrix;
    linear_solver::String = "KLU",
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
)
```

Keyword arguments:

  - `linear_solver::String` — as for `PTDF`. The three-matrix constructor is
    KLU-only because `ABA.K` is always a KLU factorization; the keyword is kept
    for API consistency and errors on any other value.
  - `tol::Union{Float64, AutoTolerance}` — sparsification tolerance for the LODF,
    default `DEFAULT_AUTO_TOLERANCE`. When sparsifying, the structural `-1.0`
    diagonal is preserved.
  - `network_reductions` (system constructor) — reductions to apply, forwarded to
    `Ybus`.

`LODF(sys)` internally builds the incidence and PTDF matrices and calls
`LODF(A, ptdf)`. The `LODF(A, PTDFm)` form warns and densifies if the supplied
PTDF was itself sparsified, since sparsification of the PTDF degrades LODF
accuracy. Use `get_lodf_data(lodf)` to read the data in standard orientation.

## Virtual (on-demand) matrices

Virtual matrices trade compute for memory: rather than materializing a full
dense matrix, they store the factorized system data needed to compute any single
row on demand and cache each row (in an LRU `RowCache`) the first time it is
requested. They expose the same identifier-based indexing as their dense
counterparts. Prefer these for large systems where only a subset of rows is ever
needed. Virtual matrices are **not** serializable.

### `VirtualPTDF`

Lazy counterpart of `PTDF`; entries mean the same thing (arc-flow sensitivity to
bus injection). Rows are computed on first access and cached.

```julia
VirtualPTDF(sys::PSY.System;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)

VirtualPTDF(ybus::Ybus;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    system_uuid::Union{Base.UUID, Nothing} = nothing,
)
```

Keyword arguments:

  - `dist_slack::Dict{Int, Float64}` — distributed-slack weights, applied per row
    as it is computed. Same `Dict` type as `PTDF`.
  - `linear_solver::String` — `"KLU"` or `"AppleAccelerateLU"` (the two factorized
    backends supported by the virtual matrices).
  - `tol::Union{Float64, AutoTolerance}` — per-row sparsification. A `Float64` is a
    fixed absolute cutoff; an `AutoTolerance` (default) is a relative per-row
    cutoff that keeps rows sparse on large systems.
  - `max_cache_size::Int` — LRU row-cache budget in MiB (`MAX_CACHE_SIZE_MiB`).
  - `persistent_arcs::Vector{Tuple{Int, Int}}` — arc rows to precompute at
    construction and pin in the cache (never evicted).
  - `network_reductions` (system constructor) — reductions to apply.

### `VirtualLODF`

Lazy counterpart of `LODF`; both dimensions are arcs. In addition to the PTDF
system data it stores the diagonal of `PTDF·A`, which its per-row computation
requires.

```julia
VirtualLODF(sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
```

Keyword arguments are as for `VirtualPTDF`, with one important difference:

  - `dist_slack::Vector{Float64}` — for `VirtualLODF` (and `VirtualMODF`) the
    distributed slack is a **`Vector{Float64}`**, not a `Dict{Int, Float64}`. The
    empty default uses a single reference bus.

### `VirtualMODF`

The on-demand Modification (post-contingency) Distribution Factor matrix. It
answers "what is the PTDF row of a monitored arc *after* a network
modification/contingency is applied", using Woodbury updates over the base
factorization. **There is no dense `MODF` type — only `VirtualMODF` exists.**

```julia
VirtualMODF(sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    irreducible_buses = Set{Int}(),
    automatically_register_outages::Bool = true,
    kwargs...,
)
```

Keyword arguments:

  - `dist_slack::Vector{Float64}` — distributed slack, a vector (as for
    `VirtualLODF`).
  - `linear_solver`, `tol`, `max_cache_size`, `network_reductions` — as for the
    other virtual matrices; `max_cache_size` is a per-contingency budget.
  - `irreducible_buses` — bus numbers to protect from reduction. Monitored and
    outaged branch endpoints are protected automatically.
  - `automatically_register_outages::Bool` — when `true` (default), every
    `PSY.Outage` supplemental attribute on the source system is registered as a
    contingency at construction. There is no public `register_contingency`
    function; query the registered set with
    `get_registered_contingencies(vmodf) -> Dict{Base.UUID, ContingencySpec}`.

See the [contingency and modification reference](contingencies.md) for the
`ContingencySpec` / `NetworkModification` types used to query this matrix.

## Admittance and network-structure matrices

### `Ybus`

The complex nodal admittance matrix (`YBUS_ELTYPE` is `ComplexF64`). `Ybus[i, j]`
is the mutual admittance between buses `i` and `j` (off-diagonal) or the
self-admittance of bus `i` (diagonal). Ybus is the foundation matrix — the DC
matrices (`IncidenceMatrix`, `BA_Matrix`, `ABA_Matrix`, `PTDF`, `LODF`) and the
virtual matrices are all built from it. Because it stays complex, it is
factorized with KLU rather than the real-only backends.

```julia
Ybus(sys::PSY.System;
    make_arc_admittance_matrices::Bool = false,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    irreducible_buses = Set{Int}(),
    zero_impedance_reduction::ZeroImpedanceBranchReduction = ZeroImpedanceBranchReduction(),
    include_constant_impedance_loads = true,
    subnetwork_algorithm = iterative_union_find,
)
```

Keyword arguments:

  - `make_arc_admittance_matrices::Bool` — when `true`, also build the from-to and
    to-from `ArcAdmittanceMatrix` objects (stored in the `arc_admittance_from_to`
    and `arc_admittance_to_from` fields) for power-flow use.
  - `network_reductions::Vector{NetworkReduction}` — reductions to apply; e.g.
    `[RadialReduction(), DegreeTwoReduction()]`. See the
    [network reduction reference](network_reduction.md).
  - `irreducible_buses` — bus numbers to protect from reduction.
  - `zero_impedance_reduction` — spec for the always-applied zero-impedance-branch
    reduction.
  - `include_constant_impedance_loads` — whether to fold constant-impedance loads
    into shunt admittances.
  - `subnetwork_algorithm` — electrical-island detection algorithm
    (`iterative_union_find` by default).

Asymmetry in `Ybus` is legitimate for phase-shifting transformers and must not
be "corrected".

### `ArcAdmittanceMatrix`

Per-arc admittance in one direction (`:FromTo` or `:ToFrom`), used by power-flow
calculations. Rows are arc tuples, columns are bus numbers.

`ArcAdmittanceMatrix` is **not constructed standalone by typical users** — it is
built as part of `Ybus` when `make_arc_admittance_matrices = true`:

```julia
ybus = Ybus(sys; make_arc_admittance_matrices = true)
ybus.arc_admittance_from_to   # ::ArcAdmittanceMatrix (direction :FromTo)
ybus.arc_admittance_to_from   # ::ArcAdmittanceMatrix (direction :ToFrom)
```

Query the direction with `get_direction(m)`.

### `IncidenceMatrix`

The bus-branch incidence matrix `A`. Rows are arcs, columns are buses;
`A[arc, bus]` is `+1` if `arc` originates at `bus`, `-1` if it terminates there,
and `0` otherwise (exactly two nonzeros per arc row). The reference-bus column is
handled through `ref_bus_positions`. It is the structural building block for
`BA_Matrix`, `ABA_Matrix`, `PTDF`, and `LODF`.

```julia
IncidenceMatrix(sys::PSY.System;
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)

IncidenceMatrix(ybus::Ybus)
```

### `BA_Matrix`

The branch-susceptance-weighted incidence matrix, `B · A`, where `B` is the
diagonal of branch series susceptances. Axes match `IncidenceMatrix` (arcs ×
buses); the data is stored transposed for efficiency. Building it from a
`Ybus` reuses the already-computed topology.

```julia
BA_Matrix(sys::PSY.System;
    network_reductions::Vector{NetworkReduction} = Vector{NetworkReduction}(),
    kwargs...,
)

BA_Matrix(ybus::Ybus)
```

The number of columns is one fewer than the bus count because the reference-bus
column is dropped; restore it with `get_ref_bus_position`.

### `ABA_Matrix`

The reduced bus-susceptance matrix `Aᵀ · B · A` with reference buses removed for
invertibility — the DC-power-flow system matrix. Its `K` field optionally holds
a KLU LU factorization for fast repeated solves.

```julia
ABA_Matrix(sys::PSY.System;
    factorize::Bool = false,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)

ABA_Matrix(ybus::Ybus; factorize::Bool = false)
```

  - `factorize::Bool` — when `true`, compute and store the KLU factorization in
    `K` at construction. When `false`, `K` is `nothing`.

If an `ABA_Matrix` was built unfactorized, factorize it after the fact and query
its state:

```julia
aba = ABA_Matrix(sys)          # K === nothing
aba = factorize(aba)           # returns a factorized ABA_Matrix
is_factorized(aba)             # true
```

See the [aggregated-branch reference](aggregated_branches.md) and reduction
pages for how reduced networks alter these matrices.

### `AdjacencyMatrix`

A symmetric bus-by-bus connectivity matrix (`Int8` entries) indexed by bus
number: nonzero where two buses share a branch, zero on the diagonal. Used for
connectivity checks and island detection (`validate_connectivity`,
`find_subnetworks`).

```julia
AdjacencyMatrix(sys::PSY.System; kwargs...)

AdjacencyMatrix(ybus::Ybus)
```

`kwargs...` are forwarded to `Ybus` (`network_reductions`,
`include_constant_impedance_loads`, `subnetwork_algorithm`).

## Concrete type aliases

`PowerflowMatrixTypes.jl` defines concrete aliases for the fully-parameterized
matrix types that appear in hot paths and downstream dispatch. Use them in
method signatures when you need to fix a concrete storage layout:

| Alias                        | Underlying type                                   | Meaning                                             |
|:---------------------------- |:------------------------------------------------- |:--------------------------------------------------- |
| `DC_PTDF_Matrix`             | `PTDF{…, Matrix{Float64}}`                        | Dense PTDF with bus/arc axes.                       |
| `DC_vPTDF_Matrix`            | `VirtualPTDF{…, K} where {K}`                     | Virtual PTDF (factorization type `K` left free).    |
| `DC_BA_Matrix`               | `BA_Matrix{…}`                                    | BA matrix with bus/arc axes.                        |
| `DC_ABA_Matrix_Factorized`   | `ABA_Matrix{…, KLULinSolveCache{Float64, Int64}}` | ABA matrix carrying a KLU factorization.            |
| `DC_ABA_Matrix_Unfactorized` | `ABA_Matrix{…, Nothing}`                          | ABA matrix with no factorization (`K === nothing`). |
| `AC_Ybus_Matrix`             | `Ybus{…}`                                         | Ybus with integer bus axes.                         |

All aliases are exported. `YBUS_ELTYPE` (the `Ybus`/`ArcAdmittanceMatrix` element
type) is exported as well.

## See also

  - [Matrix overview and indexing hub](network_matrices_overview.md) — how
    `A[row, col]` resolves and the per-type axis summary.
  - [Full public API](public.md) — authoritative docstrings for every symbol.
  - [Network reduction reference](network_reduction.md) — the
    `NetworkReduction` spec types passed via `network_reductions`.
  - [Contingency and modification reference](contingencies.md) — types used to
    query `VirtualMODF`.
  - [Tolerance and solver settings](tolerance_and_solvers.md) — `AutoTolerance`,
    `DEFAULT_AUTO_TOLERANCE`, and `linear_solver` backends.
  - [Serialization reference](serialization.md) — PTDF-only HDF5 persistence.
