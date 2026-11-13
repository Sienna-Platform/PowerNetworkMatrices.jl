# PowerNetworkMatrices

[![Main - CI](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml/badge.svg)](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl/actions/workflows/main-tests.yml)
[![codecov](https://codecov.io/gh/Sienna-Platform/PowerNetworkMatrices.jl/branch/main/graph/badge.svg?token=2VvekKsf11)](https://codecov.io/gh/Sienna-Platform/PowerNetworkMatrices.jl)

This repository holds two Julia packages.

## Packages

  - [`PowerNetworkMatrices.jl`](PowerNetworkMatrices.jl) builds network matrices from a
    `PowerSystems.System`: Ybus, PTDF, LODF and MODF, their lazy `Virtual` variants, and the
    network reductions. See its [README](PowerNetworkMatrices.jl/README.md) and its
    [manual](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable).
  - [`PowerMatricesCore.jl`](PowerMatricesCore.jl) holds the linear algebra that does not need
    `PowerSystems.jl`: the sparse solver backends, the factorization of the DC network, the PTDF
    and Woodbury row kernels, and the factor-file reads and writes. See its
    [README](PowerMatricesCore.jl/README.md). Its manual is not published yet.
    Build it from `PowerMatricesCore.jl` with
    `julia --project=docs docs/make.jl`.

`PowerNetworkMatrices.jl` depends on `PowerMatricesCore.jl`. `PowerMatricesCore.jl` has no
dependency on `PowerSystems.jl`.

## Installation

Install the released `PowerNetworkMatrices.jl` from the registry:

```julia
using Pkg
Pkg.add("PowerNetworkMatrices")
```

`PowerMatricesCore.jl` is not registered yet. Add both packages in one `Pkg.add` call.
Until this layout reaches `main`, pass `rev` in each `PackageSpec` with the name of a branch that has this layout.

```julia
using Pkg
Pkg.add([
    PackageSpec(;
        url = "https://github.com/Sienna-Platform/PowerNetworkMatrices.jl",
        subdir = "PowerMatricesCore.jl",
    ),
    PackageSpec(;
        url = "https://github.com/Sienna-Platform/PowerNetworkMatrices.jl",
        subdir = "PowerNetworkMatrices.jl",
    ),
])
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
