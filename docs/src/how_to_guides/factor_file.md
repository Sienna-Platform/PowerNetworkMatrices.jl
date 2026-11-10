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
Cached rows and cache settings are not stored. `from_hdf5(VirtualPTDF, filename)` also
accepts `persistent_arcs`, as the `VirtualPTDF` constructor does.

## File layout (format version 1)

Root attributes:

| Name                  | Type     | Meaning                                                                                                     |
|:--------------------- |:-------- |:----------------------------------------------------------------------------------------------------------- |
| `format_version`      | Int      | Layout version. The reader accepts 1 only.                                                                  |
| `system_uuid`         | Int64[2] | High and low half of the source system UUID. Absent if unknown.                                             |
| `tol_kind`            | String   | `"absolute"` for a `Float64` `tol`, `"auto"` for an `AutoTolerance`.                                        |
| `tol_value`           | Float64  | The absolute cutoff. Present when `tol_kind` is `"absolute"`.                                               |
| `auto_data_precision` | Float64  | `AutoTolerance` field. Present when `tol_kind` is `"auto"` and the field is a number. Absent means `:auto`. |
| `auto_safety`         | Float64  | `AutoTolerance` field. Present when `tol_kind` is `"auto"`.                                                 |
| `auto_quantile`       | Float64  | `AutoTolerance` field. Present when `tol_kind` is `"auto"`.                                                 |

The reader passes the stored `tol` to the constructor, which resolves the per-row cutoff
from the stored arrays exactly as a build from a `System` does.

The reader treats every entry as untrusted input. It checks the type of each entry and the
length of each table column, and it throws an error that names the entry on a mismatch.
Branch type names resolve through a fixed table of `PowerSystems` branch types.

The reader also applies these checks before it allocates memory:

  - It accepts hard links only. It rejects soft and external links.
  - It rejects filtered or compressed datasets and virtual layouts.
  - It reads the stored class and element size of each entry before it builds any Julia
    type. A number entry must have the class and size that the writer produces.
  - It limits the bytes that all entries declare to the size of the file. The reader charges
    the declared size of each entry against this budget before it allocates, so a small
    file cannot make the reader allocate a large array.
  - It requires fixed-length strings, each at most 65536 bytes wide, and rejects
    variable-length strings. The writer stores every string column and string attribute as
    a fixed-length UTF-8 string as wide as the longest value, and it rejects a longer value
    or a value that contains a null byte.
  - It checks the network structure: unique bus and arc axes, no self-loops, one reference
    bus for each connected subnetwork, and a bound on the susceptance values.
  - It requires a finite, non-negative absolute `tol_value`, a finite positive sum of
    `dist_slack` weights, valid row ranges, and unique branch keys and outage ids.

Groups and datasets:

| Name                                 | Type            | Meaning                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
|:------------------------------------ |:--------------- |:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bus_axis`                           | Vector{Int}     | Bus numbers of the matrix columns.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `arc_from`, `arc_to`                 | Vector{Int}     | Endpoints of each arc.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `arc_susceptances`                   | Vector{Float64} | Signed DC susceptance of each arc, positive from `arc_from` to `arc_to`. The reader builds the BA and incidence matrices from it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `ref_bus_positions`                  | Vector{Int}     | Sorted positions of the reference buses in `bus_axis`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `arc_live_members`                   | Vector{Int}     | Members an outage can open on each arc.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `shift_injection`                    | Vector{Float64} | DC phase shift injection of each arc, for readers that compute flow offsets. The Julia reader does not use it.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `dist_slack`                         | Vector{Float64} | Distributed slack weights of a `VirtualPTDF`. Empty if unset, and for a `VirtualMODF`. Every weight is non-negative; the writer and the reader enforce it.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `reduced_bus_from`, `reduced_bus_to` | Vector{Int}     | The reverse bus search map as two columns.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `names`                              | group           | One row per branch entry: `type`, `name`, `from`, `to`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `branch_map`                         | group           | The reduction tree of each arc. Six columns of equal length: `kind`, `id`, `parent`, `sign`, `b`, `alpha`. Written without a `System`. See the section on branch flows.                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `reduction`                          | group           | The records of the network reductions that need no component: `applied`, `removed_arc_from`, `removed_arc_to`, `removed_arc_code`, `composite_from`, `composite_to`, `survivor_from`, `survivor_to`, `survivor_bus`, `boundary_bus`, `boundary_from`, `boundary_to`, `removed_buses`, `merged_from`, `merged_to`, `irreducible_buses`. All are integer vectors. `applied` lists the reductions in order of application (1 zero-impedance, 2 radial, 3 degree-two, 4 Ward). It holds each radial, degree-two and Ward reduction that ran, and the zero-impedance reduction only when it merged a bus. |
| `subnetworks`                        | group           | One subgroup per reference bus, with `ref`, `arc_from`, `arc_to` and `bus`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `branch_outages`                     | group           | Optional. One row per branch: `type`, `name`, `id`, `is_islanding`, `start`, `len`. `id` is `IS.get_id` of the branch, the integer id of the component in the system document, and it is unique. The arrays `arc_index`, `delta_b`, `delta_shift`, `opened`, `opens_arc`, `member` hold the arc rows.                                                                                                                                                                                                                                                                                                |
| `outages`                            | group           | Optional, `VirtualMODF` only. One row per registered contingency: `id`, `label`, `is_islanding`, `start`, `len`, and the arrays `arc_index`, `delta_b`, `delta_shift`, `opened`.                                                                                                                                                                                                                                                                                                                                                                                                                     |

## Outages by name

Pass the `System` to write a table of branch outages:

```julia
to_hdf5(vmodf, "outages.h5", sys)

restored = from_hdf5(VirtualMODF, "outages.h5")
table = from_hdf5(BranchOutageTable, "outages.h5")
mod = NetworkModification(table, "Line", "line-1")
two = NetworkModification(table, [("Line", "line-1"), ("Line", "line-2")])
by_id = NetworkModification(table, 1234)  # IS.get_id of the branch
row = restored[(1, 2), mod]
```

The table has one row for each branch whose `NetworkModification(mat, branch)` succeeds.
Each arc row has `opens_arc`, which is `true` when the outage of that
one branch removes all admittance of the arc, and `member`, the position of the tripped
member in the parallel group of the arc (zero when the arc is not a parallel group). A branch that a network reduction absorbed, or
that is not in service, or that `NetworkModification` rejects, has no row. A branch whose arcs all
have zero susceptance has no row, because its outage has no DC effect. A branch with some
zero-susceptance arcs (a three-winding transformer) has rows for the other arcs only. The writer
logs one warning for each of these cases, with the number of such branches. The warning for
rejected branches also lists the first branch and each distinct message with its count, at most
10 lines.

`NetworkModification(table, keys)` groups the arc rows of all keys by arc:

  - If a row opens the arc, the arc gets the first such row only. Two members of one series
    chain, or a three-winding transformer circuit and a line on one chain, give one outage of
    the whole arc.
  - Otherwise the arc keeps the first row of each tripped member, so partial outages of
    different parallel members add. Two lines of one chain that sits in a parallel group
    remove that chain once.

For a `VirtualMODF`, the file also holds the registered contingencies, with or without `sys`.
`from_hdf5(VirtualMODF, filename)` restores them. Restored modifications have no shunt
modifications and no Ybus deltas, so they serve the DC row kernels only.

`to_hdf5` writes a temporary file in the target directory and moves it into place when the
write succeeds. A failed write with `force = true` leaves the old file unchanged. With `force = true`, a symbolic link at `filename` is replaced by the new file; the writer does not write through the link. If the target file system does not support hard links, a write with `force = false` fails with an error that says so. Use `force = true` on such a file system, because it writes with a move.

## Split arc flows into branch flows

An application can hold the factor file and the system only as a `SystemDocument`, without a
`PowerSystems.System`. It can still map each branch of the document to the arcs of the
matrix and split an arc flow into branch flows. The `branch_map` group and the
`branch_outages/id` column make this possible.

```julia
table = from_hdf5(BranchArcTable, "factors.h5")
flows = branch_flows(table, arc_flows)
```

### The table

`from_hdf5(BranchArcTable, filename)` reads the `branch_map` group. Each row is one node of
the reduction tree of an arc, in depth-first preorder, arc by arc. The columns are:

| Column   | Meaning                                                                                                                                                                                                                                          |
|:-------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `kind`   | 1 `Line`, 2 `TwoWindingTransformer`, 3 `DiscreteControlledACBranch`, 4 `GenericArcImpedance`, 5 `SeriesCompensator`, 6 to 8 winding 1 to 3 of a `ThreeWindingTransformer`, 9 parallel group, 10 series chain, 11 Ward equivalent (no component). |
| `id`     | `IS.get_id` of the branch, which equals the `id` in the document. For kinds 6 to 8 it is the id of the transformer. It is 0 for kinds 9 to 11.                                                                                                   |
| `parent` | Above 0: the row of the parent aggregate, always below the own row. Below 0: the row is a top node of arc `-parent`, an index into `arc_from` and `arc_to`.                                                                                      |
| `sign`   | +1 or -1: the orientation of the row against its parent. For a top node the parent frame is (`arc_from`, `arc_to`).                                                                                                                              |
| `b`      | Susceptance of a leaf or Ward row, in system base. It is 0.0 for kinds 9 and 10.                                                                                                                                                                 |
| `alpha`  | Phase shift of a leaf row in radians. It is 0.0 for kinds 9 to 11.                                                                                                                                                                               |

A leaf row has the key (`id`, winding), with winding 0 for a branch and 1, 2 or 3 for the
primary, secondary and tertiary winding. Join the leaf rows to the document with this key. The
position of a winding is its order in `PSY.get_circuits`. An arc can have more than one top
node: a branch entry plus a Ward equivalent on the same arc is an implicit parallel group.

### Sign convention

A flow is in system base per unit. The flow of a leaf row is positive from the first to the
second bus of the `Arc` of that leaf (for a winding, from the terminal to the star bus). The
`sign` column converts between the frame of a row and the frame of its parent. A series
segment in the direction `:FromTo` has the sign +1. The arc flow follows the existing contract
of the matrix: `f[k] = Row_k . q - s[k]`, where `q` is the injection `p` with `q[i] += s[k]`
and `q[j] -= s[k]`, and `s` is the `shift_injection` dataset.

### The two passes

`branch_flows(table, arc_flows)` returns one flow for each row of the table. It runs two
passes.

 1. The bottom-up pass goes from the last row to the first. A parallel group gets the sum of
    the susceptances of its live members, and its phase shift is the susceptance-weighted sum of
    the member shifts divided by that sum. A series chain gets the reciprocal of the sum of the
    reciprocal susceptances of its segments, and the sum of their shifts.
 2. The top-down pass goes from the first row to the last. A member of a parallel group, or a
    top node of an arc, takes its share of the flow of the parent, `b / sum(b)`, corrected by
    the phase shifts. A segment of a series chain carries the flow of the chain.

### After an outage

`branch_outages` has one row for each stored branch, with the column `id`. `get_outage_ids(table)` of
the `BranchOutageTable` maps the `id` to the key `(type, name)`, and `get_outage_rows(table)` maps that
key to the arc rows. `get_outage_opens_arc(table)` and `get_outage_member(table)` return the
columns `opens_arc` and `member`. For each arc row of the branch: if `opens_arc` is
`false`, the `member`-th child of the parallel top node of that arc is out of service. If it
is `true`, every top node of that arc except a Ward equivalent is out of service. The outage of
a branch on an arc that also holds a Ward equivalent keeps the susceptance of that equivalent
on the arc, and the outage alone does not island the network. Compute the
post-outage arc flows from the post-outage PTDF row, then call
`branch_flows(table, post_outage_arc_flows, outaged_rows)` with the rows that are out of
service. A parallel group without a live member carries no flow. A series chain with an
outaged segment carries no flow, and neither do its live segments. A live parallel group inside
such a chain still splits a circulating flow between its members when a member has a phase
shift.

### Why a branch has no arc

A branch that the radial, zero-impedance or Ward reduction removed has no leaf row. `no_arc_reason(mat, from, to)`
takes the bus numbers of the branch, as the document has them, and returns a code. It throws an
error if `from` equals `to`, or if a bus is not a bus of the matrix, a reduced bus or a removed bus. It reads
only the records of the `reduction` group, so a restored matrix gives the same answer as the
matrix that wrote the file.

| Code | Constant                | Meaning                                                                                                       |
|:---- |:----------------------- |:------------------------------------------------------------------------------------------------------------- |
| 0    | `NO_ARC_HAS_ARC`        | The branch has an arc, alone or inside a parallel group or a series chain.                                    |
| 1    | `NO_ARC_RADIAL`         | The radial reduction absorbed the branch. `no_arc_surviving_bus(mat, from, to)` returns the bus that kept it. |
| 2    | `NO_ARC_ZERO_IMPEDANCE` | The zero-impedance reduction merged both ends into one bus.                                                   |
| 3    | `NO_ARC_WARD`           | The Ward reduction eliminated the branch.                                                                     |
| 4    | `NO_ARC_NOT_IN_NETWORK` | No record names the branch. It is out of service.                                                             |

The pair of buses identifies the branch. Two branches between the same pair of buses have the
same code, so a branch out of service in parallel with a branch in service has the code of the
latter. A branch with impedance between two buses that the zero-impedance reduction merged has
code 2, as the branch with zero impedance that caused the merge.

### Limits

  - Do not use `arc_susceptances` as the denominator of a split. It differs from the sum of
    the member susceptances. The largest difference occurs on an arc with a Ward equivalent
    beside a lossy branch (measured 5.0e-2 relative on the Ward test fixture). The split uses
    the `b` of the rows.
  - `b` is the uncorrected susceptance of the component. It is not the entry of the Ybus.
  - A purely resistive parallel member (`r > 0`, `x = 0`) gets `b = 1 / min_x_eps`, with
    `min_x_eps` equal to `1e-6` unless the zero-impedance reduction sets another value. It takes
    almost all the flow of its group.
  - The split assumes zero injection at the interior buses of a degree-two chain.
    The reduction removed these buses, so a load there changes the flow along the chain.

## Limits of a restored matrix

A restored matrix has no components. It holds the component-free records of the network
reductions, which `no_arc_reason` reads. Rows by arc and bus number work.
These calls fail with their usual error messages:

  - `NetworkModification(mat, arc)`
  - `mat[name, bus]`, which looks up a branch by name

These calls throw an error that names the cause:

  - `NetworkModification(mat, branch)` and `NetworkModification(mat, sys, outage)`. Use
    `NetworkModification(table, type, name)` with a `BranchOutageTable` instead.
  - `get_branch_susceptances_by_arc(core)`
  - `to_hdf5(mat, filename)`

Build a `NetworkModification` on the matrix that wrote the file, then apply it to the restored
`VirtualMODF`. The modification refers to arcs by index, and the index is the same in both.
