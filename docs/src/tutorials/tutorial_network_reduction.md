# Network Reduction

This tutorial demonstrates the [`RadialReduction`](@ref) and
[`DegreeTwoReduction`](@ref) algorithms for simplifying power network topology
while preserving electrical characteristics. Reductions shrink the network
matrices and speed up downstream computations.

For the mathematical foundations behind these techniques, see
[Network Reduction Theory](@ref).

## Setup

```@repl tutorial_network_reduction
using PowerNetworkMatrices
using PowerSystems
using PowerSystemCaseBuilder
using Logging

configure_logging(console_level = Logging.Error)

sys = build_system(PSITestSystems, "c_sys14");
```

## Radial Reduction

Radial (dangling) buses have only one connection to the rest of the network.
[`RadialReduction`](@ref) eliminates them, reducing matrix dimensions without
affecting the core network behavior.

### Applying RadialReduction

Pass a vector of [`NetworkReduction`](@ref) objects when constructing any
network matrix. Here we use [`Ybus`](@ref):

```@repl tutorial_network_reduction
ybus_full = Ybus(sys);
ybus_radial = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()]);

size(ybus_full)
size(ybus_radial)
```

The reduced matrix has fewer rows and columns, corresponding to the eliminated
radial buses and their branches.

### Inspecting reduction results

Use [`get_network_reduction_data`](@ref) and its accessors to see what was
removed:

```@repl tutorial_network_reduction
rd = get_network_reduction_data(ybus_radial);

get_bus_reduction_map(rd)
get_removed_arcs(rd)
```

[`get_bus_reduction_map`](@ref) shows which buses were folded into which parent
buses.

### Protecting buses from reduction

Some radial buses may need to be preserved (e.g., monitoring points). Use the
`irreducible_buses` keyword:

```@repl tutorial_network_reduction
reduction = RadialReduction(; irreducible_buses = [8, 14]);
ybus_protected = Ybus(sys; network_reductions = NetworkReduction[reduction]);
rd_protected = get_network_reduction_data(ybus_protected);

get_bus_reduction_map(rd_protected)
```

Compare with the unprotected case above — buses 8 and 14 are now retained.

## Degree Two Reduction

Degree-two buses connect exactly two other buses and act as pass-through
points. [`DegreeTwoReduction`](@ref) replaces the two incident branches with a
single equivalent branch.

### Applying DegreeTwoReduction

Use a system that has degree-two buses for a clearer demonstration:

```@repl tutorial_network_reduction
sys_d2 = build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system");

ybus_d2_full = Ybus(sys_d2);
ybus_d2 = Ybus(sys_d2; network_reductions = NetworkReduction[DegreeTwoReduction()]);

size(ybus_d2_full)
size(ybus_d2)
```

### Inspecting degree-two reduction results

```@repl tutorial_network_reduction
rd_d2 = get_network_reduction_data(ybus_d2);

get_removed_buses(rd_d2)
get_removed_arcs(rd_d2)
get_series_branch_map(rd_d2)
```

[`get_series_branch_map`](@ref) shows how pairs of branches were merged into
equivalent branches.

### Protecting buses and reactive injectors

As with [`RadialReduction`](@ref), specific buses can be protected:

```@repl tutorial_network_reduction
reduction = DegreeTwoReduction(; irreducible_buses = [115]);
ybus_prot = Ybus(sys_d2; network_reductions = NetworkReduction[reduction]);
get_removed_buses(get_network_reduction_data(ybus_prot))
```

By default, buses with reactive power injections are also reduced. To preserve
them, set `reduce_reactive_power_injectors = false`:

```@repl tutorial_network_reduction
reduction = DegreeTwoReduction(; reduce_reactive_power_injectors = false);
ybus_no_reactive = Ybus(sys_d2; network_reductions = NetworkReduction[reduction]);
```

## Combining Reductions

Multiple reductions can be applied in sequence. Applying
[`RadialReduction`](@ref) first is recommended because eliminating radial buses
may expose new degree-two buses:

```@repl tutorial_network_reduction
reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()];
ybus_combined = Ybus(sys; network_reductions = reductions);

size(ybus_full)
size(ybus_combined)
```

## Reductions with Sensitivity Matrices

All network matrix types support the `network_reductions` keyword. Here is an
example with [`PTDF`](@ref) and [`LODF`](@ref):

```@repl tutorial_network_reduction
ptdf_reduced = PTDF(sys; network_reductions = NetworkReduction[RadialReduction()]);
lodf_reduced = LODF(sys; network_reductions = NetworkReduction[RadialReduction()]);

get_network_reduction_data(ptdf_reduced)
```

Reductions can also be combined with other options such as `dist_slack` and
`tol`:

```@repl tutorial_network_reduction
buscount = length(get_available_components(ACBus, sys));
dist_slack_dict = Dict(i => 1.0 / buscount for i in 1:buscount);

ptdf_opts = PTDF(sys;
    linear_solver = "KLU",
    dist_slack = dist_slack_dict,
    tol = 1e-5,
    network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
);
```

## Important Notes

  - **Reference bus protection**: The reference (slack) bus is never eliminated,
    regardless of its connectivity.
  - **Order matters**: Reductions are applied in the order given. Radial first
    is usually best.
  - **Mapping information**: The reduction data includes maps
    ([`get_bus_reduction_map`](@ref), [`get_reverse_bus_search_map`](@ref)) for
    interpreting results in terms of the original network.
  - **Electrical equivalence**: Reduced networks maintain the same electrical
    behavior at all retained buses.

## Next Steps

  - See the [Network Matrices](@ref) tutorial for building matrices without reductions
  - Read [Network Reduction Theory](@ref) for the mathematical background
  - Consult the [Public API](@ref) reference for full function signatures
