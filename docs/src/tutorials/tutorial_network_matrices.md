# Network Matrices

This tutorial walks through creating and interacting with the core network
matrices provided by `PowerNetworkMatrices.jl`, including the
[`IncidenceMatrix`](@ref), [`BA_Matrix`](@ref), [`ABA_Matrix`](@ref),
[`PTDF`](@ref), and [`LODF`](@ref).

## Setup

Load the packages and build a test system. We suppress informational log
messages so the tutorial output stays focused on the results:

```@repl tutorial_network_matrices
using PowerNetworkMatrices
using PowerSystems
using PowerSystemCaseBuilder
using Logging

configure_logging(console_level = Logging.Error)

sys = build_system(PSITestSystems, "c_sys5");
```

## Incidence Matrix

The [`IncidenceMatrix`](@ref) represents the bus–arc connectivity of the
network. Each row corresponds to an arc and each column to a bus (excluding the
reference bus):

```@repl tutorial_network_matrices
incidence_matrix = IncidenceMatrix(sys)
```

Use the getter functions to inspect the structure:

```@repl tutorial_network_matrices
get_axes(incidence_matrix)
get_data(incidence_matrix)
```

Note that the column for the reference bus is omitted. The position of that
column is available via `get_ref_bus_position`:

```@repl tutorial_network_matrices
get_ref_bus_position(incidence_matrix)
```

## BA Matrix

The [`BA_Matrix`](@ref) weights the incidence matrix by branch susceptances.
It is an intermediate building block for the [`ABA_Matrix`](@ref) and
downstream sensitivity matrices:

```@repl tutorial_network_matrices
ba_matrix = BA_Matrix(sys)
get_data(ba_matrix)
```

As with the [`IncidenceMatrix`](@ref), the reference-bus column is excluded.
To reconstruct the full matrix, insert a column of zeros at the reference bus
position:

```@repl tutorial_network_matrices
ref_pos = first(get_ref_bus_position(ba_matrix))
full_ba = hcat(
    ba_matrix.data[:, 1:(ref_pos - 1)],
    zeros(size(ba_matrix, 1), 1),
    ba_matrix.data[:, ref_pos:end],
)
```

## ABA Matrix

The [`ABA_Matrix`](@ref) is the product of the incidence matrix, branch
susceptance diagonal, and the transpose of the incidence matrix. It optionally
stores the [KLU](https://github.com/JuliaSparse/KLU.jl) LU factorization for
fast solves:

```@repl tutorial_network_matrices
aba_matrix = ABA_Matrix(sys);
is_factorized(aba_matrix)
```

The factorization is not computed by default. Pass `factorize = true` to
include it, or call [`factorize`](@ref) after the fact:

```@repl tutorial_network_matrices
aba_matrix = factorize(aba_matrix);
is_factorized(aba_matrix)
```

## PTDF Matrix

The [`PTDF`](@ref) (Power Transfer Distribution Factor) matrix relates bus
power injections to branch flows. Build it from the system directly:

```@repl tutorial_network_matrices
ptdf = PTDF(sys);
get_ptdf_data(ptdf)
```

The [`PTDF`](@ref) constructor also accepts pre-computed intermediate matrices
when they are already available, avoiding redundant work:

```@repl tutorial_network_matrices
ptdf_from_parts = PTDF(IncidenceMatrix(sys), BA_Matrix(sys));
get_ptdf_data(ptdf_from_parts)
```

### Choosing a linear solver

By default the sparse [KLU](https://github.com/JuliaSparse/KLU.jl) solver is
used. A dense solver is also available. Compare the two with `@time` to see
the difference in allocations:

```@repl tutorial_network_matrices
@time PTDF(sys; linear_solver = "KLU");
@time PTDF(sys; linear_solver = "Dense");
```

See [How to Choose a Linear Solver](@ref) for guidance on selecting a solver.

### Distributed slack bus

A distributed slack bus spreads the slack-bus duty across multiple buses. Pass
a weight dictionary to the `dist_slack` keyword:

```@repl tutorial_network_matrices
buscount = length(get_available_components(ACBus, sys));
dist_slack_dict = Dict(i => 1.0 / buscount for i in 1:buscount);

ptdf_distr = PTDF(sys; dist_slack = dist_slack_dict);
get_ptdf_data(ptdf_distr)
```

### Sparse PTDF

Elements below a tolerance can be dropped, yielding a sparse storage format.
Pass the `tol` keyword — here we use an exaggerated value for illustration:

```@repl tutorial_network_matrices
ptdf_sparse = PTDF(sys; tol = 0.2);
get_ptdf_data(ptdf_sparse)
```

In practice much smaller values are typical (e.g., `1e-5`).

## LODF Matrix

The [`LODF`](@ref) (Line Outage Distribution Factor) matrix describes how
branch flows redistribute when a line is outaged. Build it from the system:

```@repl tutorial_network_matrices
lodf = LODF(sys);
get_lodf_data(lodf)
```

The [`LODF`](@ref) can also be built from pre-computed matrices:

```@repl tutorial_network_matrices
a = IncidenceMatrix(sys);
ptdf_for_lodf = PTDF(sys);
lodf_from_parts = LODF(a, ptdf_for_lodf);
get_lodf_data(lodf_from_parts)
```

### Sparse LODF

As with the [`PTDF`](@ref), a tolerance can be applied to drop small entries:

```@repl tutorial_network_matrices
lodf_sparse = LODF(sys; tol = 0.4);
get_lodf_data(lodf_sparse)
```

## Next Steps

  - Learn about lazy, on-demand evaluation in the [Virtual Matrices](@ref) tutorial
  - See how to simplify large networks in the [Network Reduction](@ref) tutorial
  - Consult the [Public API Reference](@ref) for full function signatures
