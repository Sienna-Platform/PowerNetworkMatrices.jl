# PTDF matrix

In this tutorial the methods for computing the Power Transfer Distribution Factors (`PTDF`) are presented.
Before diving into this tutorial we encourage the user to load `PowerNetworkMatrices`, hit the `?` key in the REPL terminal and look for the documention of the different `PTDF` methods available.

## Evaluation of the `PTDF` matrix

The `PTDF` matrix can be evaluated according to three different approaches:

  - `Dense`: considers functions for dense matrix multiplication and inversion
  - `KLU`: considers functions for sparse matrix multiplication and inversion (**default**)
  - `MKLPardiso`: uses Intel's MKL Pardiso solver for sparse matrix operations (only available on Intel-based systems running Linux or Windows)

The evaluation of the `PTDF` matrix can be easily performed starting from importing the system's data and then by simply calling the `PTDF` method.

```@repl tutorial_PTDF_matrix
using PowerSystems
using PowerNetworkMatrices
using PowerSystemCaseBuilder

const PSY = PowerSystems
const PNM = PowerNetworkMatrices
const PSB = PowerSystemCaseBuilder

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

ptdf_1 = PTDF(sys);

get_ptdf_data(ptdf_1)
```

Note that while the `PTDF` stores the transpose of the matrix data, the function `get_ptdf_data` returns the data in the standard orientation.

Advanced users might be interested in computing the `PTDF` matrix starting from either the data contained in the `IncidenceMatrix` and `BA_matrix` structures.

```@repl tutorial_PTDF_matrix
# evaluate the BA_matrix and Incidence_Matrix
ba_matrix = BA_Matrix(sys);
a_matrix = IncidenceMatrix(sys);

# get the PTDF matrix starting from the values of the
# previously computed matrices
ptdf_2 = PTDF(a_matrix, ba_matrix);
get_ptdf_data(ptdf_2)
```

## Available methods for the computation of the `PTDF` matrix

As previously mentioned, the `PTDF` matrix can be evaluated considering different approaches. The method can be selected by specifying the field `linear_solver` in the `PTDF` function.

```@repl tutorial_PTDF_matrix
ptdf_dense = PTDF(sys; linear_solver = "Dense");
get_ptdf_data(ptdf_dense)

ptdf_klu = PTDF(sys; linear_solver = "KLU");
get_ptdf_data(ptdf_klu)

# For MKLPardiso (if available)
# ptdf_mkl = PTDF(sys; linear_solver = "MKLPardiso");
# get_ptdf_data(ptdf_mkl)
```

By default the "KLU" method is selected, which appeared to require significant less time and memory with respect to "Dense".
Please note that regardless of which method (`KLU`, `Dense`, or `MKLPardiso`) is used, the resulting `PTDF` matrix is stored as a dense one.

**Note on MKLPardiso**: The `MKLPardiso` solver option is only available on Intel-based systems running Linux or Windows. On other platforms (e.g., ARM-based systems or macOS), use `KLU` or `Dense` instead.

## Evaluating the `PTDF` matrix considering distributed slack bus

Whenever needed, the `PTDF` matrix can be computed with a distributed slack bus. To do so, a vector of type `Dict{Int64, Float64}` needs to be defined, specifying the weights for each bus of the system. These weights identify how the load on the slack bus is redistributed accross the system.

```@repl tutorial_PTDF_matrix
# consider equal distribution accross each bus for this example
buscount = length(PSY.get_available_components(PSY.ACBus, sys));
dist_slack = 1 / buscount * ones(buscount);
dist_slack_dict = Dict(i => dist_slack[i] / sum(dist_slack) for i in 1:buscount);
```

Once the dictionary of the weights is defined, the `PTDF` matrix can be computed by defining the input argument `dist_slack` (empty array `Dict{Int64, Float64}()` by default):

```@repl tutorial_PTDF_matrix
ptdf_distr = PTDF(sys; dist_slack = dist_slack_dict);
```

The difference between a the matrix computed with and without the `dist_slack` field defined can be seen as follows:

```@repl tutorial_PTDF_matrix
# with no distributed slack bus
get_ptdf_data(ptdf_klu)
# with distributed slack bus
get_ptdf_data(ptdf_distr)
```

## "Sparse" `PTDF` matrix

The `PTDF` matrix can be computed in a "sparse" fashion by defining the input argument `tol`. If this argument is defined, then elements of the `PTDF` matrix whose absolute values are below the set threshold are dropped. In addition, the matrix will be stored as a sparse one of type `SparseArrays.SparseMatrixCSC{Float64, Int}` instead of `Matrix{Float64}`.

By considering an "extreme" value of 0.2 as `tol`, the `PTDF` matrix can be computed as follows:

```@repl tutorial_PTDF_matrix
ptdf_sparse = PTDF(sys; tol = 0.2);
get_ptdf_data(ptdf_sparse)
```

**NOTE:** In practice, much smaller values are typically used for `tol`(e.g., 1e-5).

## Network Reductions

The `PTDF` matrix can be computed with network reductions applied to simplify the system topology. Network reductions eliminate certain buses and branches while preserving the electrical characteristics of the network. This can significantly reduce computation time and memory usage for large systems.

Two types of network reductions are supported:

  - `RadialReduction`: Eliminates radial (leaf) buses that have only one connection
  - `DegreeTwoReduction`: Eliminates degree-two buses (buses with exactly two connections) by combining their incident branches

For detailed information about these reductions, see the [RadialReduction](@ref) and [DegreeTwoReduction](@ref) tutorials.

### Using Network Reductions with PTDF

To apply network reductions, pass a vector of `NetworkReduction` objects to the `network_reductions` keyword argument:

```@repl tutorial_PTDF_matrix
# Apply radial reduction
ptdf_radial = PTDF(sys; network_reductions = NetworkReduction[RadialReduction()]);

# Apply degree-two reduction
ptdf_degree_two = PTDF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()]);

# Combine multiple reductions (order matters - RadialReduction first is recommended)
ptdf_combined = PTDF(
    sys;
    network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
);
```

### Protecting Specific Buses from Reduction

Both reduction types allow you to specify buses that should not be eliminated using the `irreducible_buses` parameter:

```@repl tutorial_PTDF_matrix
# Protect specific buses from radial reduction
reduction = RadialReduction(; irreducible_buses = [1, 2])
ptdf_protected = PTDF(sys; network_reductions = NetworkReduction[reduction]);
```

### DegreeTwoReduction Options

The `DegreeTwoReduction` has an additional option to control whether buses with reactive power injections are reduced:

```@repl tutorial_PTDF_matrix
# Preserve buses with reactive power injections
reduction = DegreeTwoReduction(; reduce_reactive_power_injectors = false)
ptdf_preserve_reactive = PTDF(sys; network_reductions = NetworkReduction[reduction]);
```

### Accessing Reduction Information

After computing the PTDF matrix with reductions, you can access information about what was reduced:

```@repl tutorial_PTDF_matrix
ptdf_reduced = PTDF(sys; network_reductions = [RadialReduction(), DegreeTwoReduction()]);

# Get the reduction data
reduction_data = get_network_reduction_data(ptdf_reduced)
```

### Combining Reductions with Other Options

Network reductions can be combined with other PTDF options like distributed slack and sparsification:

```@repl tutorial_PTDF_matrix
ptdf_full_options = PTDF(sys;
    linear_solver = "KLU",
    dist_slack = dist_slack_dict,
    tol = 1e-5,
    network_reductions = [RadialReduction(), DegreeTwoReduction()],
);
```

**NOTE**: The reference (slack) bus is automatically protected from elimination during reductions.
