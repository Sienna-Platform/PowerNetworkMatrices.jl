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
| `shift_injection`                    | Vector{Float64} | DC phase shift injection of each arc, for readers that compute flow offsets. `from_hdf5` does not use it; `PowerMatricesCore.read_shift_injection` reads it.                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `dist_slack`                         | Vector{Float64} | Distributed slack weights of a `VirtualPTDF`. Empty if unset, and for a `VirtualMODF`. Every weight is non-negative; the writer and the reader enforce it.                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `reduced_bus_from`, `reduced_bus_to` | Vector{Int}     | The reverse bus search map as two columns.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `names`                              | group           | One row per branch entry: `type`, `name`, `from`, `to`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `branch_map`                         | group           | The reduction tree of each arc. Six columns of equal length: `kind`, `id`, `parent`, `sign`, `b`, `alpha`. Written without a `System`. See the section on branch flows.                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `reduction`                          | group           | The records of the network reductions that need no component: `applied`, `removed_arc_from`, `removed_arc_to`, `removed_arc_code`, `composite_from`, `composite_to`, `survivor_from`, `survivor_to`, `survivor_bus`, `boundary_bus`, `boundary_from`, `boundary_to`, `removed_buses`, `merged_from`, `merged_to`, `irreducible_buses`. All are integer vectors. `applied` lists the reductions in order of application (1 zero-impedance, 2 radial, 3 degree-two, 4 Ward). It holds each radial, degree-two and Ward reduction that ran, and the zero-impedance reduction only when it merged a bus. |
| `subnetworks`                        | group           | One subgroup per reference bus, with `ref`, `arc_from`, `arc_to` and `bus`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `branch_outages`                     | group           | Optional. One row per branch: `type`, `name`, `id`, `is_islanding`, `start`, `len`. `id` is `IS.get_id` of the branch, the integer id of the component in the system document, and it is unique. The arrays `arc_index`, `delta_b`, `delta_shift`, `opened`, `opens_arc`, `member` hold the arc rows.                                                                                                                                                                                                                                                                                                |
| `branch_reduction`                   | group           | Written when a `System` is passed. One row per branch or three-winding circuit of the system: `id`, `winding`, `code`, `surviving_bus`. See the section on why a branch has no arc.                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `outages`                            | group           | Optional, `VirtualMODF` only. One row per registered contingency: `id`, `label`, `is_islanding`, `start`, `len`, and the arrays `arc_index`, `delta_b`, `delta_shift`, `opened`.                                                                                                                                                                                                                                                                                                                                                                                                                     |

Readers of each group, by package:

| Group or dataset                                                                                                                                               | `PowerNetworkMatrices`                             | `PowerMatricesCore`                                             |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------- |:-------------------------------------------------- |:--------------------------------------------------------------- |
| `bus_axis`, `arc_from`, `arc_to`, `arc_susceptances`, `ref_bus_positions`, `arc_live_members`, `reduced_bus_from`, `reduced_bus_to`, `subnetworks`, attributes | `from_hdf5(VirtualPTDF)`, `from_hdf5(VirtualMODF)` | `read_factor_arrays`, `read_factor_core`, `read_branch_factors` |
| `dist_slack`                                                                                                                                                   | `from_hdf5(VirtualPTDF)`                           | `read_dist_slack`, `read_branch_factors`                        |
| `shift_injection`                                                                                                                                              | none                                               | `read_shift_injection`                                          |
| `names`                                                                                                                                                        | `from_hdf5(VirtualPTDF)`, `from_hdf5(VirtualMODF)` | `read_branch_names`                                             |
| `reduction`                                                                                                                                                    | `from_hdf5(VirtualPTDF)`, `from_hdf5(VirtualMODF)` | `read_reduction_records`                                        |
| `branch_map`                                                                                                                                                   | `from_hdf5(BranchArcTable)`                        | `read_branch_map`, `read_branch_factors`                        |
| `branch_outages`                                                                                                                                               | `from_hdf5(BranchOutageTable)`                     | `read_branch_outages`, `read_branch_factors`                    |
| `branch_reduction`                                                                                                                                             | `from_hdf5(BranchReductionTable)`                  | `read_branch_reductions`                                        |
| `outages`                                                                                                                                                      | `from_hdf5(VirtualMODF)`                           | `read_registered_outages`                                       |

The partial readers also read other datasets. `read_branch_map` and `read_branch_names` read `format_version`, `arc_from` and `arc_to`. `read_branch_outages` reads `format_version`, `arc_susceptances` and `arc_from`. `read_registered_outages` reads `format_version`. It reads `arc_susceptances` and `arc_from` only when the group `outages` exists. `read_branch_reductions` reads `format_version`. The name `format_version` is a file attribute, not a dataset.

## Outages by name

Pass the `System` to write a table of branch outages. Pass `outages = false` to skip the
table; the file still holds the `branch_reduction` group.

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
that is not in service, or that `NetworkModification` rejects, has no row. The writer also rejects a branch whose trip
leaves a series chain partly in service, so that branch has no row. A branch whose arcs all
have zero susceptance has no row, because its outage has no DC effect. A branch with some
zero-susceptance arcs (a three-winding transformer) has rows for the other arcs only. The writer
logs one warning for each of these cases, with the number of such branches. The warning for
rejected branches (either cause) also lists the first branch and each distinct message with its count, at most
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

`PowerMatricesCore` reads both tables without `PowerSystems`.
`PowerMatricesCore.read_branch_outages(filename)` returns the same `BranchOutageTable`, and
`PowerMatricesCore.read_registered_outages(filename)` returns the registered contingencies by
id. Neither call factorizes the network.

`to_hdf5` writes a temporary file in the target directory and moves it into place when the
write succeeds. A failed write with `force = true` leaves the old file unchanged. With `force = true`, a symbolic link at `filename` is replaced by the new file; the writer does not write through the link. If the target file system does not support hard links, a write with `force = false` fails with an error that says so. Use `force = true` on such a file system, because it writes with a move.

`PowerMatricesCore` writes every group of the file, through `write_branch_names!`,
`write_reduction_records!`, `write_branch_outages!` and `write_registered_outages!`. `to_hdf5`
builds only the tables that need a `System`, from the matrix and the `System`, and passes them to
these writers.

## Split arc flows into branch flows

An application can hold the factor file and the system only as a `SystemDocument`, without a
`PowerSystems.System`. It can still map each branch of the document to the arcs of the
matrix and split an arc flow into branch flows. The `branch_map` group and the
`branch_outages/id` column make this possible.

```julia
table = from_hdf5(BranchArcTable, "factors.h5")
flows = branch_flows(table, arc_flows)
```

`PowerMatricesCore.read_branch_map(filename)` and `PowerMatricesCore.branch_flows` do the same
without `PowerSystems`.

### The table

`from_hdf5(BranchArcTable, filename)` reads the `branch_map` group. Each row is one node of
the reduction tree of an arc, in depth-first preorder, arc by arc. The columns are:

| Column   | Meaning                                                                                                                                                                                                                                          |
|:-------- |:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `kind`   | 1 `Line`, 2 `TwoWindingTransformer`, 3 `DiscreteControlledACBranch`, 4 `GenericArcImpedance`, 5 `SeriesCompensator`, 6 to 8 winding 1 to 3 of a `ThreeWindingTransformer`, 9 parallel group, 10 series chain, 11 Ward equivalent (no component). |
| `id`     | `IS.get_id` of the branch, which equals the `id` in the document. For kinds 6 to 8 it is the id of the transformer. It is 0 for kinds 9 to 11.                                                                                                   |
| `parent` | Above 0: the row of the parent aggregate, always below the own row. Below 0: the row is a top node of arc `-parent`, an index into `arc_from` and `arc_to`.                                                                                      |
| `sign`   | +1 or -1: the orientation of the row against its parent. For a top node the parent frame is (`arc_from`, `arc_to`).                                                                                                                              |
| `b`      | Susceptance of a leaf or Ward row as `BA_Matrix` uses it, in system base. 0.0 means no DC coupling, for example a purely resistive branch. It is 0.0 for kinds 9 and 10.                                                                         |
| `alpha`  | Phase shift of a leaf row in radians. It is 0.0 for kinds 9 to 11.                                                                                                                                                                               |

A leaf row has the key (`id`, winding). The winding is 0 for a branch with two terminals. It is 1, 2
or 3 for the primary, secondary and tertiary winding of a three-winding transformer. In the queries
of `PowerMatricesCore`, a two-terminal branch takes only its `id`, and a winding takes the `id` and
a `Winding` value (`PRIMARY`, `SECONDARY` or `TERTIARY`). Join the leaf rows to the document with this key. The
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

A branch that the radial, zero-impedance or Ward reduction removed has no leaf row, and
neither has a branch out of service. Pass the `System` to `to_hdf5` to store the reduction code for
each branch, then query it with the key of a leaf row:

```julia
reductions = from_hdf5(BranchReductionTable, "factors.h5")
code = branch_reduction(reductions, id)                     # a branch with two terminals
code = branch_reduction(reductions, id, PowerNetworkMatrices.PRIMARY) # a three-winding circuit
```

`BranchReductionTable(mat, sys)` builds the same table without a file. `branch_reduction` throws an
error for an id that the table does not hold. `from_hdf5(BranchReductionTable, filename)` throws an
error for a file written without a `System`. `PowerMatricesCore.read_branch_reductions(filename)`
reads the same table without PowerSystems.

| Code | Constant                          | Meaning                                                                                                        |
|:---- |:--------------------------------- |:-------------------------------------------------------------------------------------------------------------- |
| 0    | `BRANCH_REDUCTION_HAS_ARC`        | The branch has an arc, alone or inside a parallel group or a series chain.                                     |
| 1    | `BRANCH_REDUCTION_RADIAL`         | The radial reduction absorbed the branch. `branch_surviving_bus(reductions, id)` returns the bus that kept it. |
| 2    | `BRANCH_REDUCTION_ZERO_IMPEDANCE` | The zero-impedance reduction merged both ends into one bus.                                                    |
| 3    | `BRANCH_REDUCTION_WARD`           | The Ward reduction eliminated the branch.                                                                      |
| 4    | `BRANCH_REDUCTION_NOT_IN_NETWORK` | The branch is out of service, or it touches a bus that the matrix excluded.                                    |

These codes are not the `REDUCTION_CODE_*` constants of `NetworkReductionData`. For a circuit of a
three-winding transformer, pass the `Winding` as the third argument of `branch_surviving_bus`.

A branch with impedance between two buses that the zero-impedance reduction merged has
code 2, as the branch with zero impedance that caused the merge.

### Limits

  - Do not use `arc_susceptances` as the denominator of a split. It differs from the sum of
    the member susceptances. The largest difference occurs on an arc with a Ward equivalent
    beside a lossy branch (measured 5.0e-2 relative on the Ward test fixture). The split uses
    the `b` of the rows.
  - Inside a series chain, `b` is the component value, because `BA_Matrix` sums the components
    of a chain. A purely resistive segment there gets `b = 1 / min_x_eps`, with `min_x_eps`
    equal to `1e-6` unless the zero-impedance reduction sets another value.
  - The split assumes zero injection at the interior buses of a degree-two chain.
    The reduction removed these buses, so a load there changes the flow along the chain.

## Query factors by branch id without PowerSystems

```julia
x = PMC.read_branch_factors("factors.h5")
core = PMC.get_core(x)                            # the FactorCore
buses = PMC.get_bus_axis(x)                       # bus numbers of the columns
PMC.has_bus_column(x, 101)                        # does the bus 101 have a column?
col = PMC.bus_column(x, 101)                      # column of the bus 101
PMC.has_branch(x, 7)                              # is the branch 7 in the matrix?
row = PMC.branch_ptdf_row(x, 7)                   # factors of the branch 7, per bus
ctg = PMC.branch_contingency(x, [12, 13])         # outage of the branches 12 and 13
PMC.is_islanding(ctg)                             # does the outage split the network?
row = PMC.branch_ptdf_row(x, 7, ctg)              # factors of the branch 7 after the outage
tr1 = PMC.branch_ptdf_row(x, 9, PMC.PRIMARY)      # primary circuit of a transformer
PMC.has_branch_outages(x)                         # does the file hold an outage table?
PMC.has_branch_outage(x, 12)                      # can the branch 12 be in a contingency?
PMC.has_dist_slack(x)                             # does the file hold slack weights?
```

The `id` is the id of the branch in the document. A branch with two terminals takes only its `id`.
A circuit of a three-winding transformer takes the `id` of the transformer and a `Winding`:
`PMC.PRIMARY`, `PMC.SECONDARY` or `PMC.TERTIARY`. The same key applies to `has_branch` and to the
query with a contingency. A branch that a
network reduction removed has no row, and the query throws an error.

A contingency by branch id needs the branch outage table. Write the file with the system:
`to_hdf5(matrix, filename, sys)`. The function `branch_contingency` finds the outaged rows of
the branch map itself, so you do not apply the procedure of the section "After an outage".
A contingency is valid only with the
`BranchFactors` that built it. Use `has_bus_column` for `bus_column`, and
`has_branch` for `branch_ptdf_row`. Use `has_branch_outages` and `has_branch_outage` for
`branch_contingency`. Every query has a predicate, so a caller needs no `try`. The vector of ids must not be empty and must not repeat an id. A row with a
contingency throws an error if `is_islanding` and `has_dist_slack` are both `true`.

`read_branch_factors` throws an error if the file holds slack weights and the network has more
than one reference bus. It also throws an error if a series chain of the branch map has a zero
impedance sum.

A bus that a Ward reduction removed has the column of one boundary bus, so its factor is an
approximation. `has_bus_column` is `false` for a bus that is not in the file, for a bus
inside a degree-two chain, and for a bus that a reduction dropped for another reason. `BranchFactors`
does not tell these cases apart. The group `reduction` lists `removed_buses`, but that list also holds
available isolated buses and the buses of islands that a Ward reduction drops. Membership in it does
not identify a degree-two bus. `PowerMatricesCore.read_reduction_records` reads that group as
plain columns, and only `PowerNetworkMatrices` interprets them.

## Limits of a restored matrix

A restored matrix has no components. It holds the component-free records of the network
reductions. Rows by arc and bus number work.
These calls fail with their usual error messages:

  - `NetworkModification(mat, arc)`
  - `mat[name, bus]`, which looks up a branch by name

`PowerMatricesCore.read_branch_names(filename)` returns the `names` group as a map from
`(type, name)` to the arc, so a caller can find the arc of a name and query the row by arc.

These calls throw an error that names the cause:

  - `NetworkModification(mat, branch)` and `NetworkModification(mat, sys, outage)`. Use
    `NetworkModification(table, type, name)` with a `BranchOutageTable` instead.
  - `get_branch_susceptances_by_arc(core)`
  - `to_hdf5(mat, filename)`

Build a `NetworkModification` on the matrix that wrote the file, then apply it to the restored
`VirtualMODF`. The modification refers to arcs by index, and the index is the same in both.
