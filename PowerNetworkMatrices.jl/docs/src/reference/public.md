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
    PowerMatricesCore.BranchReductionTable,
    PowerMatricesCore.branch_reduction,
    PowerMatricesCore.branch_surviving_bus,
    PowerMatricesCore.get_branch_kind,
    PowerMatricesCore.get_branch_id,
    PowerMatricesCore.get_branch_parent,
    PowerMatricesCore.get_branch_sign,
    PowerMatricesCore.get_branch_susceptance,
    PowerMatricesCore.get_branch_alpha,
    PowerMatricesCore.get_branch_arc_count,
    PowerMatricesCore.branch_flows,
    PowerMatricesCore.compute_woodbury_factors,
    PowerMatricesCore.apply_woodbury_correction,
    PowerMatricesCore.YBUS_ELTYPE,
    PowerMatricesCore.BRANCH_REDUCTION_HAS_ARC,
    PowerMatricesCore.BRANCH_REDUCTION_RADIAL,
    PowerMatricesCore.BRANCH_REDUCTION_ZERO_IMPEDANCE,
    PowerMatricesCore.BRANCH_REDUCTION_WARD,
    PowerMatricesCore.BRANCH_REDUCTION_NOT_IN_NETWORK,
)
```

## Core names that PowerNetworkMatrices does not export

These names are documented here but not exported. Use them qualified, for example
`PowerMatricesCore.Winding`.

```@autodocs
Modules = [PowerMatricesCore]
Filter = t -> t in (
    PowerMatricesCore.Winding,
    PowerMatricesCore.worker_core,
    PowerMatricesCore.worker_network_data,
)
```
