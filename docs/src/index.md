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

## The matrix taxonomy

| Matrix                                                                | What it is                                  |
|:--------------------------------------------------------------------- |:------------------------------------------- |
| [`Ybus`](@ref)                                                        | Complex nodal admittance matrix             |
| [`IncidenceMatrix`](@ref)                                             | Branch–bus connection topology              |
| [`AdjacencyMatrix`](@ref)                                             | Bus–bus connection topology                 |
| [`BA_Matrix`](@ref) / [`ABA_Matrix`](@ref)                            | DC susceptance forms                        |
| [`PTDF`](@ref)                                                        | Power transfer distribution factors         |
| [`LODF`](@ref)                                                        | Line outage distribution factors            |
| [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref) / [`VirtualMODF`](@ref) | Lazy, row-on-demand forms for large systems |

Matrices index directly by bus numbers and branch/arc identifiers (e.g.
`ptdf["branch_name", bus_number]`) and transparently handle electrical islands
and network reductions.

## Where to start

The documentation follows the [Diátaxis](https://diataxis.fr/) framework, split
into four kinds of material:

  - **[Tutorials](tutorials/generated_getting_started.md)** — learning-oriented
    journeys. Start with [Getting Started](tutorials/generated_getting_started.md):
    a guided tour that builds a PTDF, reads a sensitivity, swaps in a virtual
    matrix, and reduces a network.
  - **[How-To Guides](how_to_guides/generated_build_multiple_matrices.md)** —
    task recipes for a specific goal (build matrices efficiently, choose a solver,
    apply reductions, set a tolerance, define contingencies, diagnose connectivity,
    persist a PTDF, and more).
  - **[Reference](reference/network_matrices_overview.md)** — exhaustive
    descriptions of every matrix type, accessor, and setting. Begin at the
    [Matrix Overview and Indexing](reference/network_matrices_overview.md) hub.
  - **[Explanation](explanation/dc_power_flow_approximation.md)** — the concepts
    and trade-offs: the DC approximation, network-reduction theory, virtual vs.
    materialized matrices, concurrency, and slack conventions.

## Installation

```text
] add PowerNetworkMatrices
```

## About

`PowerNetworkMatrices.jl` has been developed as part of the Scalable Integrated
Infrastructure Planning (SIIP) initiative at the U.S. Department of Energy's
National Renewable Energy Laboratory ([NREL](https://www.nrel.gov/)).
