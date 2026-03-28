# MODF vs LODF Reduction Combination Tests

## Purpose

Validate that VirtualMODF produces correct post-contingency PTDF rows for every
combination of network reduction and branch type. The ground truth is the
well-known N-1 identity:

```
post_PTDF[m, :] = pre_PTDF[m, :] + LODF[m, e] * pre_PTDF[e, :]
```

where LODF is computed on the same reduced network. For partial contingencies
(parallel single-circuit trips, series segment trips), `get_partial_lodf_row`
from VirtualLODF replaces the standard LODF column.

## Test File

`test/test_modf_lodf_reductions.jl`

## Test System

`psse_14_network_reduction_test_system` from PowerSystemCaseBuilder (PSSEParsingTestSystems).

This 14-bus system features:
- 14 direct branches (single-circuit arcs)
- 3 parallel branch pairs (double-circuit arcs)
- Degree-two chains (buses 117, 115, 118 form series reductions)
- Radial buses (116, 107, 108)
- 6 three-winding transformer mappings

## Reduction Configurations

| # | Config | `network_reductions` kwarg |
|---|--------|---------------------------|
| 1 | No reductions | `NetworkReduction[]` |
| 2 | Radial only | `[RadialReduction()]` |
| 3 | Degree-two only | `[DegreeTwoReduction()]` |
| 4 | Radial + degree-two | `[RadialReduction(), DegreeTwoReduction()]` |

WardReduction is excluded (changes external network topology, not suitable for
direct LODF comparison).

## Branch Types Under Test

For each reduction config, one representative branch is selected per branch type
(first entry from the corresponding map in `NetworkReductionData`):

| Branch Type | Source Map | `delta_b` | LODF Reference |
|---|---|---|---|
| Direct | `direct_branch_map` | `-b_arc` (full outage) | Standard LODF column `vlodf[m, e]` |
| Parallel circuit | `parallel_branch_map` | `-b_circuit` (single circuit) | `get_partial_lodf_row(vlodf, e, -b_circuit)` |
| Series segment | `series_branch_map` | `_compute_series_outage_delta_b(chain, segment)` | `get_partial_lodf_row(vlodf, e, delta_b)` |

Series branches only exist in configs 3 and 4 (degree-two reduction creates them).

### Coverage Matrix

| Config | Direct | Parallel | Series segment |
|--------|--------|----------|----------------|
| No reductions | Yes | Yes | N/A (empty map) |
| Radial only | Yes | Yes | N/A (empty map) |
| Degree-two only | Yes | Yes | Yes |
| Radial + degree-two | Yes | Yes | Yes |

## Helper Function

A shared helper `verify_modf_lodf_identity` encapsulates the comparison logic:

```julia
function verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b; atol=1e-6)
```

Steps:
1. Create a `ContingencySpec` with a single `ArcModification(arc_idx, delta_b)`
2. Register it in `vmodf.contingency_cache`
3. Determine LODF reference: if `delta_b == -b_arc`, use standard LODF column;
   otherwise use `get_partial_lodf_row`
4. For every monitored arc `m`: assert
   `vmodf[m, ctg] == ptdf[m, :] + lodf_col[m] * ptdf[arc_idx, :]`
5. Clean up Woodbury cache

## Tolerance

`atol = 1e-6` (consistent with existing `test_virtual_modf.jl`).

## Testset Structure

```
"MODF vs LODF: no reductions"
    -> direct branch outage
    -> parallel single-circuit outage

"MODF vs LODF: radial reduction"
    -> direct branch outage
    -> parallel single-circuit outage

"MODF vs LODF: degree-two reduction"
    -> direct branch outage
    -> parallel single-circuit outage
    -> series segment outage

"MODF vs LODF: radial + degree-two reduction"
    -> direct branch outage
    -> parallel single-circuit outage
    -> series segment outage
```

## Non-Goals

- N-2 or multi-branch contingencies (future work)
- WardReduction (topology change makes direct comparison non-trivial)
- Exhaustive iteration over all branches (one representative per type suffices)
- Performance benchmarking
