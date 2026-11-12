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
    PowerMatricesCore.NetworkModification,
    PowerMatricesCore.BranchOutageTable,
    PowerMatricesCore.get_outage_rows,
    PowerMatricesCore.get_outage_ids,
    PowerMatricesCore.get_outage_opens_arc,
    PowerMatricesCore.get_outage_member,
    PowerMatricesCore.BranchArcTable,
    PowerMatricesCore.BranchReasonTable,
    PowerMatricesCore.no_arc_reason,
    PowerMatricesCore.no_arc_surviving_bus,
    PowerMatricesCore.get_branch_kind,
    PowerMatricesCore.get_branch_id,
    PowerMatricesCore.get_branch_parent,
    PowerMatricesCore.get_branch_sign,
    PowerMatricesCore.get_branch_susceptance,
    PowerMatricesCore.get_branch_alpha,
    PowerMatricesCore.get_branch_arc_count,
    PowerMatricesCore.branch_flows,
    PowerMatricesCore.worker_core,
    PowerMatricesCore.worker_network_data,
)
```
