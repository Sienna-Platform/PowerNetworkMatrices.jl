# PowerMatricesCore.jl

`PowerMatricesCore.jl` holds the linear algebra that
[`PowerNetworkMatrices.jl`](https://github.com/Sienna-Platform/PowerNetworkMatrices.jl)
builds on: the sparse factorization backends (KLU, Apple Accelerate and MKL Pardiso), the
DC factorization of the network, the PTDF and Woodbury row kernels, and the reads and
writes of the factor-file arrays. It has no dependency on `PowerSystems.jl`.

Most users load `PowerNetworkMatrices.jl`, which builds the network matrices from a
`PowerSystems.System` and re-exports the public names of this package.

## Installation

```julia
using Pkg
Pkg.add("PowerMatricesCore")
```

## License

PowerMatricesCore is released under a BSD [license](LICENSE). PowerMatricesCore has been
developed as part of the Sienna platform.
