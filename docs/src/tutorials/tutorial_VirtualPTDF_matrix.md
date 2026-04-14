# VirtualPTDF

Contrary to the traditional `PTDF` matrix, the `VirtualPTDF` is a structure containing rows of the original matrix, related to specific system arcs.
The different rows of the `PTDF` matrix are cached in the `VirtualPTDF` structure as they are evaluated. This allows to keep just the portion of the original matrix which is of interest to the user, avoiding the unnecessary computation of the whole matrix.

Refer to the different arguments of the `VirtualPTDF` methods by looking at the "Public API Reference" page.

## How the `VirtualPTDF` works

The `VirtualPTDF` is a structure containing everything needed to compute any row of the PTDF matrix and store it. To do so, the `VirtualPTDF` must store the BA matrix (coming from the `BA_Matrix` struct) and the inverse of the ABA matrix (coming from `ABA_MAtrix` struct). In particular, `KLU` is used to get the LU factorization matrices of the ABA matrix and these ones are stored, avoid the inversion.

Once the `VirtualPTDF` is initialized, each row of the PTDF matrix can be evaluated separately. The algorithmic procedure is the following:

 1. Define the `VirtualPTDF` structure
 2. Call any element of the matrix to define and store the relative row as well as showing the selected element

Regarding point 2, if the row has been stored previously then the desired element is just loaded from the cache and shown.

The flowchart below shows how the `VirtualPTDF` is structured and how it works. Examples will be presented in the following sections.

```@raw html
<img src="../../assets/VirtualPTDF_scheme.png"/>
```

## Initialize `VirtualPTDF` and compute/access row/element

As for the `PTDF` matrix, at first the `System` data must be loaded. The "RTS-GMLC" systems is considered as example:

```@repl tutorial_VirtualPTDF_matrix
using PowerNetworkMatrices
using PowerSystems
using PowerSystemCaseBuilder

import PowerSystems as PSY
import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys");
```

At this point the `VirtualPTDF` is initialized with the following simple command:

```@repl tutorial_VirtualPTDF_matrix
v_ptdf = VirtualPTDF(sys);
```

Now, an element of the matrix can be computed by calling the arc tuple and bus number:

```@repl tutorial_VirtualPTDF_matrix
el_C31_105 = v_ptdf[(318, 321), 105]
```

Alternatively, the value can be indexed by row and column numbers directly. In this case the row and column numbers are mapped by the dictonaries contained in the `lookup` field.

```@repl tutorial_VirtualPTDF_matrix
row_number = v_ptdf.lookup[1][(318, 321)]
col_number = v_ptdf.lookup[2][105]
el_C31_105_bis = v_ptdf[row_number, col_number]
```

**NOTE**: this example was made for the sake of completeness and considering the actual arc tuple and bus number is recommended.

As previously mentioned, in order to evaluate a single element of the `VirtualPTDF`, the entire row related to the selected arc must be considered. For this reason it is cached in the `VirtualPTDF` structure for later calls.
This is evident by looking at the following example:

```@repl tutorial_VirtualPTDF_matrix
sys_2k = PSB.build_system(PSB.PSYTestSystems, "tamu_ACTIVSg2000_sys");

v_ptdf_2k = VirtualPTDF(sys_2k);

# evaluate PTDF row related to arc (5270, 5474)
@time v_ptdf_2k[(5270, 5474), 8155]

# call same element after the row has been stored
@time v_ptdf_2k[(5270, 5474), 8155]
```

## `VirtualPTDF` with distributed slack bus

As for the `PTDF` matrix, here too each row can be evaluated considering distributed slack buses.
A vector of type `Vector{Float64}` is defined, specifying the weights for each bus of the system.

```@repl tutorial_VirtualPTDF_matrix
# smaller system for the next examples
sys_2 = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# consider equal distribution accross each bus for this example
buscount = length(PSY.get_available_components(PSY.ACBus, sys_2));
dist_slack = 1 / buscount * ones(buscount);
dis_slack_dict = Dict(i => dist_slack[i] / sum(dist_slack) for i in 1:buscount)
```

Now initialize the `VirtualPTDF` by defining the `dist_slack` field with the vector of weights previously computed:

```@repl tutorial_VirtualPTDF_matrix
v_ptdf_distr = VirtualPTDF(sys_2; dist_slack = dis_slack_dict);
v_ptdf_orig = VirtualPTDF(sys_2);
```

Now check the difference with the same row related to the branch `"1"` evaluated without considering distributed slack bus.

```@repl tutorial_VirtualPTDF_matrix
row_distr = [v_ptdf_distr["1", j] for j in v_ptdf_distr.axes[2]]
row_original = [v_ptdf_orig["1", j] for j in v_ptdf_orig.axes[2]]
```

## "Sparse" `VirtualPTDF`

Sparsification of each row can be achieved in the same fashion as for the `PTDF` matrix, by removing those elements whose absolute values is below a certain threshold.

As for the example show for the `PTDF` matrix, here to a very high values of 0.2 is considered for the `tol` field. Again, this value is considered just for the sake of this example.

```@repl tutorial_VirtualPTDF_matrix
v_ptdf_dense = VirtualPTDF(sys_2);
v_ptdf_sparse = VirtualPTDF(sys_2; tol = 0.2);
```

Let's now evaluate the same row as before and compare the results:

```@repl tutorial_VirtualPTDF_matrix
original_row = [v_ptdf_dense["1", j] for j in v_ptdf_dense.axes[2]]
sparse_row = [v_ptdf_sparse["1", j] for j in v_ptdf_sparse.axes[2]]
```

## Network Reductions

The `VirtualPTDF` can be computed with network reductions applied to simplify the system topology. Network reductions eliminate certain buses and branches while preserving the electrical characteristics of the network. This can significantly reduce computation time and memory usage for large systems.

Two types of network reductions are supported:

  - `RadialReduction`: Eliminates radial (leaf) buses that have only one connection
  - `DegreeTwoReduction`: Eliminates degree-two buses (buses with exactly two connections) by combining their incident branches

For detailed information about these reductions, see the [RadialReduction](@ref) and [DegreeTwoReduction](@ref) tutorials.

### Using Network Reductions with VirtualPTDF

To apply network reductions, pass a vector of `NetworkReduction` objects to the `network_reductions` keyword argument:

```@repl tutorial_VirtualPTDF_matrix
# Apply radial reduction
v_ptdf_radial = VirtualPTDF(sys_2; network_reductions = NetworkReduction[RadialReduction()]);

# Apply degree-two reduction
v_ptdf_degree_two =
    VirtualPTDF(sys_2; network_reductions = NetworkReduction[DegreeTwoReduction()]);

# Combine multiple reductions (order matters - RadialReduction first is recommended)
v_ptdf_combined =
    VirtualPTDF(
        sys_2;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    );
```

### Protecting Specific Buses from Reduction

Both reduction types allow you to specify buses that should not be eliminated using the `irreducible_buses` parameter:

```@repl tutorial_VirtualPTDF_matrix
# Protect specific buses from radial reduction
reduction = RadialReduction(; irreducible_buses = [1, 2])
v_ptdf_protected = VirtualPTDF(sys_2; network_reductions = NetworkReduction[reduction]);
```

### DegreeTwoReduction Options

The `DegreeTwoReduction` has an additional option to control whether buses with reactive power injections are reduced:

```@repl tutorial_VirtualPTDF_matrix
# Preserve buses with reactive power injections
reduction = DegreeTwoReduction(; reduce_reactive_power_injectors = false)
v_ptdf_preserve_reactive =
    VirtualPTDF(sys_2; network_reductions = NetworkReduction[reduction]);
```

### Accessing Reduction Information

After initializing the VirtualPTDF with reductions, you can access information about what was reduced:

```@repl tutorial_VirtualPTDF_matrix
v_ptdf_reduced =
    VirtualPTDF(
        sys_2;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    );

# Get the reduction data
reduction_data = get_network_reduction_data(v_ptdf_reduced)
```

### Combining Reductions with Other Options

Network reductions can be combined with other VirtualPTDF options like distributed slack and sparsification:

```@repl tutorial_VirtualPTDF_matrix
v_ptdf_full_options = VirtualPTDF(sys_2;
    dist_slack = dis_slack_dict,
    tol = 1e-5,
    network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
);
```

**NOTE**: The `VirtualPTDF` only supports `KLU` and `AppleAccelerate` linear solvers when using network reductions. The reference (slack) bus is automatically protected from elimination during reductions.
