# PowerNetworkMatrices.jl

```@meta
CurrentModule = PowerNetworkMatrices
```

## Overview

`PowerNetworkMatrices.jl` is a [`Julia`](http://www.julialang.org) package for
building the network matrices used in DC/AC power flow, sensitivity, and
contingency analysis. Given a `PowerSystems.jl` [`System`](@extref PowerSystems.System), it produces the
linear-algebra layer of the [Sienna](https://www.nrel.gov/analysis/sienna.html)
power-systems platform: it reads the network data and returns matrix objects, it
does not own the data model.

`PowerNetworkMatrices.jl` is an active project under development, and we welcome
your feedback, suggestions, and bug reports.

## Installation

```text
] add PowerNetworkMatrices
```

## Where to start

The documentation follows the [Diátaxis](https://diataxis.fr/) framework, split
into four kinds of material:

  - **[Tutorials](tutorials/generated_getting_started.md)** — learning-oriented
    journeys, each answering one question end to end. Start with
    [Getting Started](tutorials/generated_getting_started.md), which screens a line
    outage on a small network with a PTDF, an LODF, and a reduction; then
    [Analysis at Scale](tutorials/generated_analysis_at_scale.md) does the same at
    scale with virtual matrices and cache control.
  - **[How-To Guides](how_to_guides/generated_build_multiple_matrices.md)** —
    task recipes for a specific goal (build matrices efficiently, choose a solver,
    reproduce industry DFAX values, define contingencies, diagnose connectivity).
  - **[Reference](reference/network_matrices_overview.md)** — exhaustive
    descriptions of every matrix type, accessor, and setting. Begin at the
    [Matrix Overview and Indexing](reference/network_matrices_overview.md) hub.
  - **[Explanation](explanation/dc_power_flow_approximation.md)** — the concepts
    and trade-offs: the DC approximation, network-reduction theory, computational
    considerations, concurrency, and slack conventions.

## About

`PowerNetworkMatrices.jl` has been developed as part of the Scalable Integrated
Infrastructure Planning (SIIP) initiative at the U.S. Department of Energy's
National Renewable Energy Laboratory ([NREL](https://www.nrel.gov/)).
