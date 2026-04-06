# Virtual Matrices

This tutorial introduces the [`VirtualPTDF`](@ref) and [`VirtualLODF`](@ref)
structures, which compute and cache rows of the full [`PTDF`](@ref) and
[`LODF`](@ref) matrices on demand. When only a subset of rows is needed, or
when the full matrix is too large to store, virtual matrices avoid unnecessary
computation and reduce memory usage.

## Setup

```@repl tutorial_virtual_matrices
using PowerNetworkMatrices
using PowerSystems
using PowerSystemCaseBuilder
using Logging

configure_logging(console_level = Logging.Error)

sys = build_system(PSISystems, "RTS_GMLC_DA_sys");
```

## VirtualPTDF

### Initialization

Create a [`VirtualPTDF`](@ref) the same way you would create a [`PTDF`](@ref) —
the difference is that no rows are computed yet:

```@repl tutorial_virtual_matrices
v_ptdf = VirtualPTDF(sys);
```

The structure stores the [`BA_Matrix`](@ref) and the LU factorization of the
[`ABA_Matrix`](@ref) internally, so individual rows can be solved on the fly.

### Accessing elements

Index by `(from_bus, to_bus)` arc tuple and bus number. The first access
computes and caches the entire row for that arc:

```@repl tutorial_virtual_matrices
v_ptdf[(318, 321), 105]
```

Subsequent accesses to the same row are fast cache lookups.

### Comparing with the full PTDF

To see how the virtual version relates to the full matrix, compare on a smaller
system:

```@repl tutorial_virtual_matrices
sys_small = build_system(PSITestSystems, "c_sys5");

v_ptdf_small = VirtualPTDF(sys_small);
ptdf_full = PTDF(sys_small);

# Evaluate a complete row via the virtual structure
row_virtual = [v_ptdf_small["1", j] for j in v_ptdf_small.axes[2]]

# The same row from the full matrix
get_ptdf_data(ptdf_full)[1, :]
```

The values match — the virtual version simply defers computation until a row is
requested.

### Demonstrating the caching benefit

On a larger system the first access triggers a solve, while subsequent accesses
are near-instant:

```@repl tutorial_virtual_matrices
sys_2k = build_system(PSYTestSystems, "tamu_ACTIVSg2000_sys");
v_ptdf_2k = VirtualPTDF(sys_2k);

@time v_ptdf_2k[(5270, 5474), 8155]  # computes and caches the row

@time v_ptdf_2k[(5270, 5474), 8155]  # cache lookup
```

Notice the large difference in time between the first and second call — this is
the key advantage of the virtual approach.

### Distributed slack bus

A distributed slack bus is supported via the `dist_slack` keyword, just like
[`PTDF`](@ref):

```@repl tutorial_virtual_matrices
buscount = length(get_available_components(ACBus, sys_small));
dist_slack_dict = Dict(i => 1.0 / buscount for i in 1:buscount);

v_ptdf_distr = VirtualPTDF(sys_small; dist_slack = dist_slack_dict);
v_ptdf_orig = VirtualPTDF(sys_small);

row_distr = [v_ptdf_distr["1", j] for j in v_ptdf_distr.axes[2]]
row_original = [v_ptdf_orig["1", j] for j in v_ptdf_orig.axes[2]]
```

### Sparse VirtualPTDF

The `tol` keyword drops elements below a threshold in each cached row:

```@repl tutorial_virtual_matrices
v_ptdf_sparse = VirtualPTDF(sys_small; tol = 0.2);
v_ptdf_dense = VirtualPTDF(sys_small);

sparse_row = [v_ptdf_sparse["1", j] for j in v_ptdf_sparse.axes[2]]
dense_row = [v_ptdf_dense["1", j] for j in v_ptdf_dense.axes[2]]
```

## VirtualLODF

### Initialization

[`VirtualLODF`](@ref) works the same way as [`VirtualPTDF`](@ref) but for the
[`LODF`](@ref) matrix. Internally it stores the diagonal of the [`PTDF`](@ref)
and the factorized [`ABA_Matrix`](@ref) to solve rows on demand:

```@repl tutorial_virtual_matrices
v_lodf = VirtualLODF(sys);
```

### Accessing elements

Index by outaged-arc and monitored-arc tuples:

```@repl tutorial_virtual_matrices
v_lodf[(221, 222), (202, 206)]
```

The value represents the fraction of flow from arc `(202, 206)` that is
diverted to arc `(221, 222)` when `(202, 206)` is outaged.

### Caching demonstration

```@repl tutorial_virtual_matrices
v_lodf_2k = VirtualLODF(sys_2k);

@time v_lodf_2k[(5270, 5474), (2118, 2113)]

@time v_lodf_2k[(5270, 5474), (2118, 2113)]
```

Again, the second call is much faster because the row is already cached.

### Sparse VirtualLODF

```@repl tutorial_virtual_matrices
v_lodf_sparse = VirtualLODF(sys_small; tol = 0.4);
v_lodf_dense = VirtualLODF(sys_small);

sparse_row = [v_lodf_sparse[(1, 2), j] for j in v_lodf_sparse.axes[2]]
dense_row = [v_lodf_dense[(1, 2), j] for j in v_lodf_dense.axes[2]]
```

## When to use Virtual vs. Full matrices

| Scenario | Recommended |
|:--- |:--- |
| Need the entire matrix | [`PTDF`](@ref) / [`LODF`](@ref) |
| Need only a few rows | [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref) |
| System is very large (> 10 000 arcs) | Virtual versions to limit memory |
| Repeated access to all rows | Full versions (no per-row solve overhead) |

## Next Steps

  - See the [Network Matrices](@ref) tutorial for building full matrices
  - See the [Network Reduction](@ref) tutorial for simplifying large systems
  - Consult the [Public API Reference](@ref) for full function signatures
