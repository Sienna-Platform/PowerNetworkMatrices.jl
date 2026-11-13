# PowerNetworkMatrices

This repository holds two Julia packages.

| **Package**                                          | **Documentation**                                                                                                                                                                                                                                                                                                                                                                         | **Build Status**                                                                                                                                                                                                      | **Coverage**                                                                                                                                                                                                 |
|:---------------------------------------------------- |:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| [`PowerNetworkMatrices.jl`](PowerNetworkMatrices.jl) | [![](https://img.shields.io/badge/Central_Sienna_docs-blue.svg)](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/index/) [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/) [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://sienna-platform.github.io/PowerNetworkMatrices.jl/dev/) | [![](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml/badge.svg?branch=main)](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml) | [![](https://codecov.io/gh/Sienna-Platform/PowerNetworkMatrices.jl/branch/main/graph/badge.svg?token=2VvekKsf11&flag=pnm)](https://codecov.io/gh/Sienna-Platform/PowerNetworkMatrices.jl?flags%5B0%5D=pnm)   |
| [`PowerMatricesCore.jl`](PowerMatricesCore.jl)       | [README](PowerMatricesCore.jl/README.md)                                                                                                                                                                                                                                                                                                                                                  | [![](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml/badge.svg?branch=main)](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml) | [![](https://codecov.io/gh/Sienna-Platform/PowerNetworkMatrices.jl/branch/main/graph/badge.svg?token=2VvekKsf11&flag=core)](https://codecov.io/gh/Sienna-Platform/PowerNetworkMatrices.jl?flags%5B0%5D=core) |

[![](https://img.shields.io/badge/slack-@Sienna/PNM-sienna.svg?logo=slack)](https://join.slack.com/t/core-sienna/shared_invite/zt-glam9vdu-o8A9TwZTZqqNTKHa7q3BpQ) [![](https://img.shields.io/badge/dynamic/json?url=http%3A%2F%2Fjuliapkgstats.com%2Fapi%2Fv1%2Ftotal_downloads%2FPowerNetworkMatrices&query=total_requests&label=Downloads)](http://juliapkgstats.com/pkg/PowerNetworkMatrices)

## Packages

  - [`PowerNetworkMatrices.jl`](PowerNetworkMatrices.jl) builds classic network matrices from a
    `PowerSystems.System`:
    [Ybus](https://en.wikipedia.org/wiki/Nodal_admittance_matrix),
    [PTDF](https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Power_Transfer_Distribution_Factors.htm),
    [LODF](https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Line_Outage_Distribution_Factors_LODFs.htm)
    and [MODF](https://arxiv.org/pdf/2501.17529v1), their lazy `Virtual` variants, and the
    network reductions. See its [README](PowerNetworkMatrices.jl/README.md).
  - [`PowerMatricesCore.jl`](PowerMatricesCore.jl) holds the linear algebra that does not need
    `PowerSystems.jl`: the sparse solver backends, the factorization of the DC network, the PTDF
    and Woodbury row kernels, and the factor-file reads and writes. See its
    [README](PowerMatricesCore.jl/README.md).

`PowerNetworkMatrices.jl` depends on `PowerMatricesCore.jl`. `PowerMatricesCore.jl` has no
dependency on `PowerSystems.jl`. Both packages work with Julia v1.10+.

## Installation

```julia
using Pkg
Pkg.add("PowerNetworkMatrices")  # also installs PowerMatricesCore
Pkg.add("PowerMatricesCore")     # the linear algebra only, without PowerSystems
```

## Development

Run each command from the folder shown.

Tests for `PowerNetworkMatrices.jl`, from `PowerNetworkMatrices.jl/`:

```sh
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
```

Tests for `PowerMatricesCore.jl`, from `PowerMatricesCore.jl/`:

```sh
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
```

Formatter for both packages, from the repository root:

```sh
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution guidelines.

## License

Both packages are released under a BSD [license](LICENSE).
PowerNetworkMatrices has been developed as part of the Scalable Integrated Infrastructure Planning (SIIP)
initiative at the U.S. Department of Energy's National Laboratory of the Rockies (formerly known as NREL) ([NLR](https://www.nlr.gov/)).
