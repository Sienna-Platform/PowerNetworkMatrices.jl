# PowerNetworkMatrices.jl

```@meta
CurrentModule = PowerNetworkMatrices
```

## Overview

`PowerNetworkMatrices.jl` is a [`Julia`](http://www.julialang.org) package for
the evaluation of network matrices given the system's data. The package allows to compute
the matrices according to different methods, providing a flexible and powerful tool.

`PowerNetworkMatrices.jl` is an active project under development, and we welcome your feedback,
suggestions, and bug reports.

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

PowerNetworkMatrices.jl strives to follow the [Diátaxis documentation framework](https://diataxis.fr/),
which organizes documentation according to the different needs of users. The documentation is
structured into four main sections:

### Tutorials

**Learning-oriented guides to help you get started**

Tutorials are hands-on lessons that take you through practical examples step-by-step.
They are designed to help you learn by doing, building understanding through practical experience.

  - Start here if you're new to PowerNetworkMatrices.jl
  - Follow along with executable examples
  - Build foundational knowledge

### How-To Guides

**Task-oriented guides for accomplishing specific goals**

How-to guides provide direct instructions for solving specific problems or completing
particular tasks. They assume you have basic knowledge and want to accomplish something specific.

  - Use when you know what you want to do
  - Get straight to the solution
  - Focus on practical application

### Explanation

**Understanding-oriented discussion of key topics**

Explanations provide background, context, and deeper understanding of concepts, design
decisions, and the theory behind the implementation.

  - Understand the "why" behind the features
  - Learn about the mathematical foundations
  - Explore conceptual relationships

### Reference

**Information-oriented technical descriptions**

Reference documentation provides detailed, technical information about the API, functions,
and data structures. It's organized for quick lookup of specific details.

  - Look up function signatures and parameters
  - Find available methods and options
  - Access complete API documentation

## Installation and Quick Links

  - [Sienna installation page](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/how-to/install/):
    Instructions to install `PowerNetworkMatrices.jl` and other Sienna packages
  - [Central Sienna documentation](https://sienna-platform.github.io/Sienna/SiennaDocs/docs/build/index.html):
    Cross-linked documentation website for the core user-facing Sienna packages

* * *

PowerNetworkMatrices has been developed as part of the Scalable Integrated Infrastructure Planning (SIIP) initiative at the U.S. Department of Energy's National Laboratory of the Rockies (formerly known as NREL) ([NLR](https://www.nlr.gov/)).
