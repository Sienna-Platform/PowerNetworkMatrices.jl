# Quick Start Guide

!!! note
    
    `PowerSystemCaseBuilder.jl` is a helper library that makes it easier to reproduce examples in the documentation and tutorials. Normally you would pass your local files to create the system data instead of calling the function `build_system`.
    For more details visit [PowerSystemCaseBuilder Documentation](https://nrel-sienna.github.io/PowerSystemCaseBuilder.jl/stable)

For more details about loading data and adding more dynamic components check the
[Creating a System with Dynamic devices](https://nrel-sienna.github.io/PowerSystems.jl/stable/tutorials/add_dynamic_data/)
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

As it can be seen, PTDF matrix is stored such that the number of rows is equal
to the number of buses, number of columns equal to the number of branches.
