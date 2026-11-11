# Public API Reference

```@autodocs
Modules = [PowerNetworkMatrices]
Public = true
```

## Names from PowerMatricesCore

`PowerNetworkMatrices` re-exports these names from `PowerMatricesCore`.

```@autodocs
Modules = [PowerMatricesCore]
Filter = t -> t in (
    PowerMatricesCore.LinearSolverCache,
    PowerMatricesCore.AutoTolerance,
    PowerMatricesCore.discover_data_precision,
    PowerMatricesCore.ArcModification,
    PowerMatricesCore.ShuntModification,
    PowerMatricesCore.ContingencySpec,
)
```
