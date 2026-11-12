# Quick Start Guide

!!! note
    
    `PowerSystemCaseBuilder.jl` is a helper library that makes it easier to reproduce examples in the documentation and tutorials. Normally you would pass your local files to create the system data instead of calling the function `build_system`.
    For more details visit [PowerSystemCaseBuilder Documentation](https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable)

For more details about loading data and adding more dynamic components check the
[Creating a System with Dynamic devices](https://sienna-platform.github.io/PowerSystems.jl/stable/tutorials/add_dynamic_data/)
section of the documentation in `PowerSystems.jl`.

## Loading data

Data can be loaded from a pss/e raw file and a pss/e dyr file.

```@repl quick_start_guide
using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
```

## Computation of the PTDF matrix

Once system data is loaded, network matrices can be evaluated. The following
example shows how the PTDF matrix is computed.

The function `PTDF` is called for the evaluation of the matrix and other data. These
are stored in a structure of type `PTDF`.

```@repl quick_start_guide
# evaluate the PTDF structure containing the matrix and other data.
ptdf_matrix = PNM.PTDF(sys);

# show the PTDF matrix.
PNM.get_data(ptdf_matrix)
```

As it can be seen, the PTDF matrix is stored internally in transposed form for computational efficiency.
The function `get_ptdf_data` returns the data in the standard orientation (arcs × buses).

The matrix axes are indexed by arc tuples `(from_bus_number, to_bus_number)` and bus numbers.
You can inspect the axes and lookup dictionaries as follows:

```@repl quick_start_guide
# axes and lookup dictionaries describe the arc tuples and bus numbers for each dimension
PNM.get_axes(ptdf_matrix)
PNM.get_lookup(ptdf_matrix)
```

Elements can be accessed using arc tuples and bus numbers directly. The example below picks
the first arc and first bus from the matrix axes so it works for any system:

```@repl quick_start_guide
some_arc = PNM.get_axes(ptdf_matrix)[2][1]
some_bus = PNM.get_axes(ptdf_matrix)[1][1]
ptdf_matrix[some_arc, some_bus]
```
