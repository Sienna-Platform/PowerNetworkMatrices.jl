# RadialReduction

In this tutorial the `RadialReduction` network reduction algorithm is presented. This reduction eliminates radial (dangling) buses and their associated branches from the power network while preserving the electrical behavior of the core network.

Before diving into this tutorial we encourage the user to load `PowerNetworkMatrices`, hit the `?` key in the REPL terminal and look for the documentation of `RadialReduction`.

## Understanding Radial Branches

Radial buses are leaf nodes in the network topology with only one connection. These buses do not affect the electrical behavior of the rest of the network and for certain applications can be safely eliminated to simplify network matrices and improve computational efficiency.

## Basic Usage of RadialReduction

The `RadialReduction` can be applied when constructing various network matrices. The most common use case is with the `Ybus` matrix:

```@repl tutorial_RadialReduction
using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# Load a test system
sys = PSB.build_system(PSB.PSITestSystems, "c_sys14");

# Create Ybus with radial reduction
ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
```

## Accessing Reduction Information

After applying the reduction, you can access information about which buses and branches were eliminated:

```@repl tutorial_RadialReduction
# Get the network reduction data
reduction_data = get_network_reduction_data(ybus);

# View the bus reduction mapping
# This shows which buses were reduced to which parent buses
get_bus_reduction_map(reduction_data)

# View the reverse bus search mapping
# This maps each reduced bus to its ultimate parent
PNM.get_reverse_bus_search_map(reduction_data)

# View the removed arcs (branches)
PNM.get_removed_arcs(reduction_data)
```

## Protecting Specific Buses from Reduction

In some cases, you may want to preserve certain buses even if they are radial. This can be done using the `irreducible_buses` parameter:

```@repl tutorial_RadialReduction
# Create radial reduction that protects buses 8 and 14
reduction = RadialReduction(; irreducible_buses = [8, 14]);

# Apply to system (buses must exist in the system)
ybus_protected = Ybus(sys; network_reductions = NetworkReduction[reduction]);

# Bus 8 was radial, but preserved from reduction
reduction_data = get_network_reduction_data(ybus_protected);
get_bus_reduction_map(reduction_data)
```

## Combining with Other Network Matrices

The `RadialReduction` can be applied to other network matrix types as well:

```@repl tutorial_RadialReduction
# Apply to PTDF matrix
ptdf = PTDF(sys; network_reductions = NetworkReduction[RadialReduction()]);

# Apply to LODF matrix
lodf = LODF(sys; network_reductions = NetworkReduction[RadialReduction()]);

# Apply to Incidence Matrix
incidence = IncidenceMatrix(sys; network_reductions = NetworkReduction[RadialReduction()]);
```

## Benefits of Radial Reduction

Using `RadialReduction` provides several advantages:

 1. **Smaller Matrices**: Eliminates unnecessary rows and columns from network matrices
 2. **Faster Computations**: Reduced matrix dimensions lead to faster linear algebra operations
 3. **Better Conditioning**: Removing radial elements can improve numerical properties
 4. **Memory Efficiency**: Reduces storage requirements for large network models

## Example: Comparing Matrix Sizes

```@repl tutorial_RadialReduction
# Create Ybus without reduction
ybus_full = Ybus(sys);

# Create Ybus with radial reduction
ybus_reduced = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()]);

# Compare sizes
size(ybus_full)
size(ybus_reduced)
```

## Combining Multiple Reductions

`RadialReduction` can be combined with other network reduction algorithms like `DegreeTwoReduction`:

```@repl tutorial_RadialReduction
# Apply both radial and degree-two reductions
reductions = [RadialReduction(), DegreeTwoReduction()];
ybus_multi = Ybus(sys; network_reductions = reductions);
```

## Important Notes

  - **Reference Bus Protection**: Reference (slack) buses are automatically protected from elimination, regardless of their connectivity
  - **Order Matters**: When combining multiple reductions, they are applied in the order specified in the vector
  - **Reversibility**: The reduction maintains mapping information (`bus_reduction_map` and `reverse_bus_search_map`) that can be used for result interpretation
  - **Electrical Equivalence**: The reduced network maintains the same electrical behavior as the original network for all non-eliminated elements
