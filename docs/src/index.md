# PowerNetworkMatrices.jl

```@meta
CurrentModule = PowerNetworkMatrices
```

## Overview

`PowerNetworkMatrices.jl` is a [`Julia`](http://www.julialang.org) package for
building the network matrices used in DC/AC power flow, sensitivity, and
contingency analysis. Given a `PowerSystems.jl` [`System`](@extref PowerSystems.System), it produces the
linear-algebra layer of the [Sienna](https://sienna-platform.github.io/Sienna/))
power-systems platform: it reads the network data and returns matrix objects, it
does not own the data model.

`PowerNetworkMatrices.jl` is an active project under development, and we welcome
your feedback, suggestions, and bug reports.

## About Sienna

`PowerNetworkMatrices.jl` is part of the National Laboratory of the Rockies (formerly known as NREL)'s
[Sienna ecosystem](https://sienna-platform.github.io/Sienna/), an open source framework for
scheduling problems and dynamic simulations for power systems. The Sienna ecosystem can be
[found on GitHub](https://github.com/Sienna-Platform). It contains three applications:

  - [Sienna\Data](https://sienna-platform.github.io/Sienna/pages/applications/sienna_data.html) enables
    efficient data input, analysis, and transformation
  - [Sienna\Ops](https://sienna-platform.github.io/Sienna/pages/applications/sienna_ops.html) enables
    system scheduling simulations by formulating and solving optimization problems
  - [Sienna\Dyn](https://sienna-platform.github.io/Sienna/pages/applications/sienna_dyn.html) enables
    system transient analysis including small signal stability and full system dynamic
    simulations

Each application uses multiple packages in the [`Julia`](http://www.julialang.org)
programming language. `PowerNetworkMatrices.jl` is part of
[Sienna\Net](https://sienna-platform.github.io/Sienna/pages/applications/sienna_network.html):
it computes network matrices (for example Ybus, PTDF, and LODF) from `PowerSystems.jl` data.

## How to use this documentation

The documentation follows the [Diátaxis](https://diataxis.fr/) framework, split
into four kinds of material:

  - **Tutorials** - Detailed walk-throughs to help you *learn* how to use
    `PowerNetworkMatrices.jl`
  - **How to...** - Directions to help *guide* your work for a particular task
  - **Explanation** - Additional details and background information to help you *understand*
    `PowerNetworkMatrices.jl`, its structure, and how it works behind the scenes
  - **Reference** - Technical references and API for a quick *look-up* during your work

## Installation and Quick Links

  - [Sienna installation page](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/how-to/install/):
    Instructions to install `PowerNetworkMatrices.jl` and other Sienna packages
  - [Central Sienna documentation](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/index.html):
    Cross-linked documentation website for the core user-facing Sienna packages

`PowerNetworkMatrices.jl` has been developed as part of the Scalable Integrated
Infrastructure Planning (SIIP) initiative at the U.S. Department of Energy's
National Laboratory of the Rockies (formerly known as NREL) ([NLR](https://www.nlr.gov/)).
