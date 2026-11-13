# PowerMatricesCore.jl

```@meta
CurrentModule = PowerMatricesCore
```

## Overview

`PowerMatricesCore.jl` is a [`Julia`](http://www.julialang.org) package that holds the
linear algebra under
[`PowerNetworkMatrices.jl`](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/):

  - the sparse factorization backends: KLU, Apple Accelerate and MKL Pardiso;
  - the DC factorization of the network and the PTDF row kernel;
  - the network modification types and the Woodbury row update for outages;
  - the sparsification cutoffs and the row cache;
  - the reads and writes of every group of the factor file, including `reduction`;
  - the split of arc flows into branch flows.

The package has no dependency on `PowerSystems.jl`. `PowerNetworkMatrices.jl` builds the
network matrices from a `PowerSystems.System`, calls this package for the numerics, and
re-exports the names its users need. Most users load `PowerNetworkMatrices.jl` and do not need to
load this package directly.

## Documentation Structure

PowerMatricesCore.jl follows the [Diataxis documentation framework](https://diataxis.fr/).
The tutorials, how-to guides and explanations for the network matrices are in the
[`PowerNetworkMatrices.jl` documentation](https://sienna-platform.github.io/PowerNetworkMatrices.jl/stable/).
This manual holds the reference:

### Reference

**Information-oriented technical descriptions**

  - [Public API](@ref "Public API Reference"): every documented name of the package.
  - [Internals](@ref): the solver wrappers `KLUWrapper` and `AccelerateWrapper`.
