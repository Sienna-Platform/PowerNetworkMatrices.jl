# PowerNetworkMatrices.jl

```@meta
CurrentModule = PowerNetworkMatrices
```

## Overview

`PowerNetworkMatrices.jl` is a [`Julia`](http://www.julialang.org) package for
the evaluation of network matrices given the system's data. The package computes
matrices such as the [`PTDF`](@ref), [`LODF`](@ref), [`Ybus`](@ref), and
[`IncidenceMatrix`](@ref), along with virtual (lazy) counterparts
[`VirtualPTDF`](@ref) and [`VirtualLODF`](@ref), and supports
[network reduction](@ref "Network Reduction Theory") techniques.

`PowerNetworkMatrices.jl` is an active project under development, and we welcome your feedback,
suggestions, and bug reports.

## About Sienna

`PowerNetworkMatrices.jl` is part of the National Renewable Energy Laboratory's
[Sienna ecosystem](https://nrel-sienna.github.io/Sienna/), an open source framework for
power system modeling, simulation, and optimization. The Sienna ecosystem can be
[found on Github](https://github.com/NREL-Sienna/Sienna). It contains three applications:

  - [Sienna\Data](https://nrel-sienna.github.io/Sienna/pages/applications/sienna_data.html) enables
    efficient data input, analysis, and transformation
  - [Sienna\Ops](https://nrel-sienna.github.io/Sienna/pages/applications/sienna_ops.html) enables
    enables system scheduling simulations by formulating and solving optimization problems
  - [Sienna\Dyn](https://nrel-sienna.github.io/Sienna/pages/applications/sienna_dyn.html) enables
    system transient analysis including small signal stability and full system dynamic
    simulations

Each application uses multiple packages in the [`Julia`](http://www.julialang.org)
programming language.

## Installation and Quick Links

  - [Sienna installation page](https://nrel-sienna.github.io/Sienna/SiennaDocs/docs/build/how-to/install/):
    Instructions to install `PowerNetworkMatrices.jl` and other Sienna packages
  - [Sienna Documentation Hub](https://nrel-sienna.github.io/Sienna/SiennaDocs/docs/build/index.html):
    Links to other Sienna packages' documentation

## How To Use This Documentation

There are four main sections containing different information:

  - **Tutorials** - Detailed walk-throughs to help you *learn* how to use
    `PowerNetworkMatrices.jl`
  - **How to...** - Directions to help *guide* your work for a particular task
  - **Explanation** - Additional details and background information to help you *understand*
    `PowerNetworkMatrices.jl`, its structure, and how it works behind the scenes
  - **Reference** - Technical references and API for a quick *look-up* during your work

`PowerNetworkMatrices.jl` strives to follow the [Diataxis](https://diataxis.fr/) documentation
framework.
