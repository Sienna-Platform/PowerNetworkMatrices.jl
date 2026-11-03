# Save and load the virtual matrices

Building a `VirtualPTDF` or `VirtualMODF` needs a `PowerSystems.System`, a `Ybus`, and a
factorization. Use a factor file to skip the `System`: write the factor data once, then
rebuild the matrix in another process, for example a service that only computes rows.

## Write and read

```julia
using PowerNetworkMatrices

vptdf = VirtualPTDF(sys; network_reductions = [RadialReduction(), DegreeTwoReduction()])
to_hdf5(vptdf, "factors.h5")

restored = from_hdf5(VirtualPTDF, "factors.h5")
row = restored[(1, 2), 3]
```

`to_hdf5(mat, filename; force = false)` accepts a `VirtualPTDF` or a `VirtualMODF` and throws if
the file exists and `force` is `false`. `from_hdf5(VirtualPTDF, filename)` and
`from_hdf5(VirtualMODF, filename)` accept `linear_solver` and `max_cache_size`. The reader
factorizes the ABA matrix again, so the solver can differ from the one that wrote the file.
Cached rows are not stored.

## File layout (format version 1)

Root attributes:

| Name                  | Type    | Meaning                                                                                                     |
|:--------------------- |:------- |:----------------------------------------------------------------------------------------------------------- |
| `format_version`      | Int     | Layout version. The reader accepts 1 only.                                                                  |
| `system_uuid`         | String  | UUID of the source system, empty if unknown.                                                                |
| `tol_kind`            | String  | `"absolute"` for a `Float64` `tol`, `"auto"` for an `AutoTolerance`.                                        |
| `tol_value`           | Float64 | The absolute cutoff. Present when `tol_kind` is `"absolute"`.                                               |
| `auto_data_precision` | Float64 | `AutoTolerance` field. Present when `tol_kind` is `"auto"` and the field is a number. Absent means `:auto`. |
| `auto_safety`         | Float64 | `AutoTolerance` field. Present when `tol_kind` is `"auto"`.                                                 |
| `auto_quantile`       | Float64 | `AutoTolerance` field. Present when `tol_kind` is `"auto"`.                                                 |

The reader passes the stored `tol` to the constructor, which resolves the per-row cutoff
from the stored arrays exactly as a build from a `System` does.

The reader treats every entry as untrusted input. It checks the type of each entry and the
length of each table column, and it throws an error that names the entry on a mismatch.
Branch type names resolve through a fixed table of `PowerSystems` branch types.

Groups and datasets:

| Name                                 | Type            | Meaning                                                                                                                                                            |
|:------------------------------------ |:--------------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `bus_axis`                           | Vector{Int}     | Bus numbers of the matrix columns.                                                                                                                                 |
| `arc_from`, `arc_to`                 | Vector{Int}     | Endpoints of each arc.                                                                                                                                             |
| `arc_susceptances`                   | Vector{Float64} | Signed DC susceptance of each arc, positive from `arc_from` to `arc_to`. The reader builds the BA and incidence matrices from it.                                  |
| `ref_bus_positions`                  | Vector{Int}     | Sorted positions of the reference buses in `bus_axis`.                                                                                                             |
| `arc_live_members`                   | Vector{Int}     | Members an outage can open on each arc.                                                                                                                            |
| `shift_injection`                    | Vector{Float64} | DC phase shift injection of each arc, for readers that compute flow offsets. The Julia reader does not use it.                                                     |
| `dist_slack`                         | Vector{Float64} | Distributed slack weights of a `VirtualPTDF`. Empty if unset, and for a `VirtualMODF`.                                                                             |
| `reduced_bus_from`, `reduced_bus_to` | Vector{Int}     | The reverse bus search map as two columns.                                                                                                                         |
| `names`                              | group           | One row per branch entry: `type`, `name`, `from`, `to`.                                                                                                            |
| `subnetworks`                        | group           | One subgroup per reference bus, with `ref`, `arc_from`, `arc_to` and `bus`.                                                                                        |
| `branch_outages`                     | group           | Optional. One row per branch: `type`, `name`, `tag`, `is_islanding`, `start`, `len`. The arrays `arc_index`, `delta_b`, `delta_shift`, `opened` hold the arc rows. |
| `outages`                            | group           | Optional, `VirtualMODF` only. One row per registered contingency: `id`, `label`, `is_islanding`, `start`, `len`, and the same four arrays.                         |

## Outages by name

Pass the `System` to write a table of branch outages:

```julia
to_hdf5(vmodf, "factors.h5", sys)

restored = from_hdf5(VirtualMODF, "factors.h5")
table = from_hdf5(BranchOutageTable, "factors.h5")
mod = NetworkModification(table, "Line", "line-1")
two = NetworkModification(table, [("Line", "line-1"), ("Line", "line-2")])
row = restored[(1, 2), mod]
```

The table has one row for each branch whose `NetworkModification(mat, branch)` succeeds. The
tag of a row is `direct`, `parallel`, `series` or `3wt`. A branch that a network reduction
absorbed, or that `NetworkModification` rejects, has no row. The writer logs one warning with
the number of such branches and the first reason. Two members of one series chain give one
outage of the whole arc.

For a `VirtualMODF`, the file also holds the registered contingencies. `from_hdf5(VirtualMODF, filename)` restores them. Restored modifications have no shunt modifications and no Ybus deltas,
so they serve the DC row kernels only.

## Limits of a restored matrix

A restored matrix has no components and no reduction maps. Rows by arc and bus number work.
These calls fail with their usual error messages:

  - `NetworkModification(mat, arc)`
  - `NetworkModification(mat, branch)`
  - `mat[name, bus]`, which looks up a branch by name

Build a `NetworkModification` on the matrix that wrote the file, then apply it to the restored
`VirtualMODF`. The modification refers to arcs by index, and the index is the same in both.
