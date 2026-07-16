# Documenter.jl cheatsheet (Sienna packages)

## Build

```bash
julia --project=docs docs/make.jl
```

## `make.jl` pages entry

```julia
using DataStructures: OrderedDict

pages = OrderedDict(
    "Home" => "index.md",
    "Tutorials" => ["tutorials/getting_started.md"],
    "How to..." => ["how_to_guides/configure.md"],
    "Explanation" => ["explanation/architecture.md"],
    "Reference" => ["reference/public.md"],
)

makedocs(;
    modules = [MyPackage],
    sitename = "MyPackage.jl",
    pages = collect(pages),
)
```

## In-line math

  - DO: `My math: ``\sqrt[n]{1 + x + x^2 + \ldots}`` `
  - DON'T: `My math: $\sqrt[n]{1 + x + x^2 + \ldots}$`
    Most of our math is using the old fragile single `$...$` format. Don't rely on that.

## Common markdown directives

````markdown
```@docs
foo
bar
```

```@index
Pages = ["reference/public.md"]
```

```@meta
CurrentModule = MyPackage
```

!!! note
    Admonition body indented four spaces.
````

## Cross-references

| Goal                 | Syntax                                                 |
|:-------------------- |:------------------------------------------------------ |
| Link to docstring    | `[`foo`](@ref)` or `@ref foo`                          |
| Link to another page | `[text](@ref page_slug)` if slug defined               |
| External package     | `@extref OtherPackage.symbol` via DocumenterInterLinks |

## DocumenterInterLinks (in `make.jl`)

```julia
using DocumenterInterLinks

links = InterLinks(
    "PowerSystems" => "https://https://sienna-platform.github.io/PowerSystems.jl/stable/",
)
```

## External cross-reference format

  - Do, so user can see correct package: ```[`PowerSimulations.SimulationResults`](@extref)```
  - Don't: ```[`SimulationResults`](@extref PowerSimulations.SimulationResults)```

## Live docs (doctests)

Use `@example` blocks when showing REPL output so documentation build will error if example errors and bugs can be caught; keep examples runnable in CI when possible.
