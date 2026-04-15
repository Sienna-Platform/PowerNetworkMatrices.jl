# DegreeTwoReduction

In this tutorial the `DegreeTwoReduction` network reduction algorithm is presented. This reduction eliminates buses with exactly two connections by combining the incident branches into a single equivalent branch while preserving the electrical characteristics of the network.

Before diving into this tutorial we encourage the user to load `PowerNetworkMatrices`, hit the `?` key in the REPL terminal and look for the documentation of `DegreeTwoReduction`.

## Understanding Degree-Two Buses

Degree-two buses are nodes in the network topology that have exactly two connections. These intermediate buses can be eliminated by replacing the two incident branches with a single equivalent branch, simplifying the network while maintaining its electrical behavior. The reduction is performed recursively, identifying and eliminating chains of degree-two nodes to maximize network simplification.

## Basic Usage of DegreeTwoReduction

The `DegreeTwoReduction` can be applied when constructing various network matrices. The most common use case is with the `Ybus` matrix:

```@repl tutorial_DegreeTwoReduction
using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# Load a test system
sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")

# Create Ybus with degree-two reduction
ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);
```

## Accessing Reduction Information

After applying the reduction, you can access information about which buses were eliminated and how branches were combined:

```@repl tutorial_DegreeTwoReduction
# Get the network reduction data
reduction_data = get_network_reduction_data(ybus);

# View the series branch mapping
# This shows how multiple branches were combined into composite branches
PNM.get_series_branch_map(reduction_data)

# View the removed buses
PNM.get_removed_buses(reduction_data)

# View the removed arcs (series branches that were combined)
PNM.get_removed_arcs(reduction_data)
```

## Configuration Options

The `DegreeTwoReduction` provides several configuration options:

### Protecting Specific Buses

You can protect certain buses from reduction even if they have degree two:

```@repl tutorial_DegreeTwoReduction
# Create degree-two reduction that protects specific buses
reduction = DegreeTwoReduction(; irreducible_buses = [115]);

# Apply to system (if these buses exist in the system)
ybus_protected = Ybus(sys; network_reductions = NetworkReduction[reduction]);
reduction_data_protected = get_network_reduction_data(ybus_protected);
# Compare with unprotected case: 
PNM.get_removed_buses(reduction_data)
PNM.get_removed_buses(reduction_data_protected)
```

### Handling Reactive Power Injectors

By default, `DegreeTwoReduction` reduces buses with reactive power injections. You can change this behavior:

```@repl tutorial_DegreeTwoReduction
# Create reduction that preserves buses with reactive power injections
reduction = DegreeTwoReduction(; reduce_reactive_power_injectors = false);

# Apply to system
ybus_preserve_reactive = Ybus(sys; network_reductions = NetworkReduction[reduction]);
```

## Combining with Other Network Matrices

The `DegreeTwoReduction` can be applied to other network matrix types as well:

```@repl tutorial_DegreeTwoReduction
# Apply to PTDF matrix
ptdf = PTDF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);

# Apply to LODF matrix
lodf = LODF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);

# Apply to BA Matrix
ba_matrix = BA_Matrix(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);

# Apply to ABA Matrix
aba_matrix = ABA_Matrix(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);
```

## Benefits of Degree-Two Reduction

Using `DegreeTwoReduction` provides several advantages:

 1. **Smaller Matrices**: Eliminates intermediate buses from network matrices
 2. **Faster Computations**: Reduced matrix dimensions lead to faster operations
 3. **Simplified Topology**: Creates a more direct representation of the network
 4. **Preserved Accuracy**: Maintains exact electrical equivalence for the reduced network

## Example: Comparing Matrix Sizes

```@repl tutorial_DegreeTwoReduction
# Create Ybus without reduction
ybus_full = Ybus(sys);

# Create Ybus with degree-two reduction
ybus_reduced = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);

# Compare sizes
size(ybus_full)
size(ybus_reduced)
```

## Understanding Series Branch Chains

When degree-two buses are eliminated, the reduction algorithm identifies chains of series-connected branches. For example:

```
Bus A --- Branch 1 --- Bus B --- Branch 2 --- Bus C
```

If Bus B has degree two, it can be eliminated, and Branches 1 and 2 are combined into a single equivalent branch:

```
Bus A --- Equivalent Branch --- Bus C
```

The equivalent branch's electrical parameters (impedance, admittance) are calculated to preserve the overall electrical behavior.

## Combining Multiple Reductions

`DegreeTwoReduction` can be combined with other network reduction algorithms like `RadialReduction`:

```@repl tutorial_DegreeTwoReduction
# Apply both radial and degree-two reductions
reductions = [RadialReduction(), DegreeTwoReduction()];
ybus_multi = Ybus(sys; network_reductions = reductions);

# Get combined reduction data
multi_reduction_data = get_network_reduction_data(ybus_multi);
```

## Order of Reductions

When combining multiple reductions, the order can affect the final result:

```@repl tutorial_DegreeTwoReduction
# First apply radial, then degree-two
reductions1 = [RadialReduction(), DegreeTwoReduction()];
ybus1 = Ybus(sys; network_reductions = reductions1)

# First apply degree-two, then radial  
reductions2 = [DegreeTwoReduction(), RadialReduction()];
ybus2 = Ybus(sys; network_reductions = reductions2)

# Compare results
size(ybus1)
size(ybus2)
```

In this case, the result is the same, however this is not guaranteed.
In general, applying `RadialReduction` first is recommended, as it can create new degree-two buses that can then be eliminated by `DegreeTwoReduction`.

## Important Notes

  - **Topology Preservation**: The reduction maintains essential network connectivity
  - **Reference Bus Protection**: Reference (slack) buses are automatically protected from elimination
  - **Parallel Paths**: The algorithm handles parallel branches correctly
  - **Three-Winding Transformers**: Special handling for three-winding transformer connections
  - **Reversibility**: The reduction maintains detailed mapping information for result interpretation
  - **Electrical Equivalence**: Equivalent branches are computed to maintain exact electrical behavior
