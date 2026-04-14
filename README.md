# PowerNetworkMatrices.jl

[![Main - CI](https://github.com/NREL-Sienna/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml/badge.svg)](https://github.com/NREL-Sienna/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml)
[![codecov](https://codecov.io/gh/NREL-Sienna/PowerNetworkMatrices.jl/branch/main/graph/badge.svg?token=2VvekKsf11)](https://codecov.io/gh/NREL-Sienna/PowerNetworkMatrices.jl)
[![Documentation Build](https://github.com/NREL-Sienna/PowerNetworkMatrices.jl/workflows/Documentation/badge.svg?)](https://nrel-sienna.github.io/PowerNetworkMatrices.jl/stable)
[<img src="https://img.shields.io/badge/slack-@Sienna/PNM-sienna.svg?logo=slack">](https://join.slack.com/t/nrel-sienna/shared_invite/zt-glam9vdu-o8A9TwZTZqqNTKHa7q3BpQ)
[![PowerNetworkMatrices.jl Downloads](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FPowerNetworkMatrices&query=total_requests&label=Downloads)](http://juliapkgstats.com/pkg/PowerNetworkMatrices)

`PowerNetworkMatrices.jl` is able to build classic power systems modeling network matrices such as
[Ybus](https://en.wikipedia.org/wiki/Nodal_admittance_matrix), [PTDF](https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Power_Transfer_Distribution_Factors.htm) and [LODF](https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Line_Outage_Distribution_Factors_LODFs.htm#:%7E:text=Line%20Outage%20Distribution%20Factors%20(LODFs)%20are%20a%20sensitivity%20measure%20of,other%20lines%20in%20the%20system.).

## Version Advisory

  - PowerNetworkMatrices.jl will work with Julia v1.6+. In Mac the requirement is to have Julia 1.9.2+ due to issues with MKL and dynamic library loading.
  - PowerNetworkMatrices.jl exports Matrix methods that were available in PowerSystems.jl version 1.0 and have been implemented as a separate package.

## License

PowerNetworkMatrices is released under a BSD [license](https://github.com/NREL/PowerNetworkMatrices.jl/blob/master/LICENSE).
PowerNetworkMatrices has been developed as part of the Scalable Integrated Infrastructure Planning (SIIP)
initiative at the U.S. Department of Energy's National Laboratory of the Rockies (formerly known as NREL) ([NLR](https://www.nrel.gov/)).
