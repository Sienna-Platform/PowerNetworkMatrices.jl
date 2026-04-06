# Ybus Modification for Contingency Analysis

This tutorial describes the `YbusModification` feature for efficient contingency analysis
on the nodal admittance matrix (Ybus). Instead of rebuilding the full Ybus from scratch for
each contingency, `YbusModification` computes a sparse delta matrix ``\Delta Y`` that can be
added to the base Ybus in a single sparse matrix addition.

## Mathematical Background

### The Nodal Admittance Matrix

The Ybus matrix is an ``N \times N`` complex sparse matrix where ``N`` is the number of buses.
Each branch connecting bus ``i`` to bus ``j`` is represented by a Pi-model with four admittance
entries:

```math
\begin{bmatrix} I_i \\ I_j \end{bmatrix} =
\begin{bmatrix} Y_{11} & Y_{12} \\ Y_{21} & Y_{22} \end{bmatrix}
\begin{bmatrix} V_i \\ V_j \end{bmatrix}
```

For a standard transmission line with series impedance ``z = r + jx`` and shunt admittances
``y^{\text{sh}}_i``, ``y^{\text{sh}}_j`` at each end:

```math
Y_l = \frac{1}{r + jx}
```

```math
Y_{11} = Y_l + y^{\text{sh}}_i, \quad Y_{12} = Y_{21} = -Y_l, \quad Y_{22} = Y_l + y^{\text{sh}}_j
```

The full Ybus is assembled by summing these ``2 \times 2`` contributions over all branches.

### The Delta Matrix

When a contingency removes a branch between buses ``i`` and ``j``, the Ybus changes by exactly
the negation of that branch's Pi-model contribution:

```math
\Delta Y_{ii} = -Y_{11}, \quad \Delta Y_{ij} = -Y_{12}, \quad \Delta Y_{ji} = -Y_{21}, \quad \Delta Y_{jj} = -Y_{22}
```

The post-contingency Ybus is then:

```math
Y_{\text{bus}}^{\text{post}} = Y_{\text{bus}}^{\text{pre}} + \Delta Y
```

This is exact (not an approximation) because Ybus construction is a linear summation of branch
contributions. The key advantage is that ``\Delta Y`` is extremely sparse: a single branch outage
produces at most 4 nonzero entries in an ``N \times N`` matrix.

### Component Types in the Delta

The `YbusModification` handles several component categories, each with different
``\Delta Y`` computation:

**Direct branches.** A single branch on an arc. The delta is the negated Pi-model:
``\Delta Y = -Y^{\pi}_{\text{branch}}``.

**Parallel branches.** Multiple branches share the same arc ``(i, j)``. Tripping one branch
from a parallel group negates only that branch's individual Pi-model contribution. The other
branches on the arc remain in service and their contributions are unchanged.

**Series chains.** When network reductions (e.g., `DegreeTwoReduction`) eliminate intermediate
buses, branches that were in series are merged into a single equivalent arc between the
retained endpoint buses. The equivalent admittance is obtained by Kron reduction of the
internal nodes:

```math
Y_{\text{reduced}} = Y_{kk} - Y_{ke} \, Y_{ee}^{-1} \, Y_{ek}
```

where ``k`` indexes the two retained endpoint buses and ``e`` indexes the eliminated internal
buses. When a contingency trips a branch inside a series chain, the chain's equivalent must
be recomputed with the tripped branch removed, and the delta is:

```math
\Delta Y = Y_{\text{reduced}}^{\text{modified}} - Y_{\text{reduced}}^{\text{original}}
```

If tripping the branch breaks the chain entirely (a segment has no surviving branches), then
``Y_{\text{reduced}}^{\text{modified}} = 0`` and the delta is simply ``-Y_{\text{reduced}}^{\text{original}}``.

**Shunt elements.** `FixedAdmittance`, `StandardLoad` (impedance component), and
`SwitchedAdmittance` contribute only to the diagonal of Ybus. Their delta is a single
diagonal entry: ``\Delta Y_{ii} = -Y_{\text{shunt}}``.

**Impedance changes.** Rather than a full outage, a branch's series impedance can be
changed by a delta ``\Delta z``. The series admittance changes from
``Y_l = 1/z`` to ``Y_{l,\text{new}} = 1/(z + \Delta z)``, and the Pi-model entries
change proportionally. Tap ratio factors for transformers are preserved. For branches
inside series chains, the chain equivalent is recomputed with the modified impedance
and the delta is the difference of the reduced equivalents.

### Island Detection

A contingency may disconnect the network into electrically isolated subnetworks (islands).
`YbusModification` detects this automatically during construction.

The algorithm determines which arcs are **fully severed** by the contingency:
- A direct branch outage always severs its arc.
- A parallel branch outage severs the arc only if **all** branches in the parallel group are tripped.
- A series chain outage severs the equivalent arc if any segment of the chain is fully tripped (breaking the chain).
- Shunt outages and impedance changes never sever arcs.

Once the set of severed arcs is known, the algorithm runs a union-find connected components
check on the adjacency structure, skipping severed edges inline. If the post-contingency
network has more connected components than the base network, the contingency is flagged as
islanding (``\texttt{is\_islanding} = \text{true}``).

The computational cost is ``O(N + E)`` per contingency, where ``N`` is the number of buses
and ``E`` is the number of edges in the adjacency matrix. For sparse power networks where
``E \sim O(N)``, this reduces to effectively ``O(N)``, making it suitable for screening
large contingency lists.

## Usage

### Setup

``` @repl tutorial_YbusModification
using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB
import PowerSystems as PSY

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");
```

### Computing the Base Ybus

``` @repl tutorial_YbusModification
ybus = Ybus(sys);
```

### Single Branch Outage

Create a `YbusModification` for a single branch outage by passing the base Ybus and a vector
of components to trip:

``` @repl tutorial_YbusModification
line = PSY.get_component(PSY.Line, sys, "1")
mod = YbusModification(ybus, PSY.Component[line])
```

The modification is a sparse delta with at most 4 nonzero entries:

``` @repl tutorial_YbusModification
mod.data
```

Apply the modification to obtain the post-contingency Ybus data:

``` @repl tutorial_YbusModification
modified_data = apply_ybus_modification(ybus, mod);
```

### Multiple Branch Outage (N-k)

Multiple branches can be tripped simultaneously:

``` @repl tutorial_YbusModification
line1 = PSY.get_component(PSY.Line, sys, "1")
line2 = PSY.get_component(PSY.Line, sys, "2")
mod_nk = YbusModification(ybus, PSY.Component[line1, line2])
modified_data_nk = apply_ybus_modification(ybus, mod_nk);
```

### Impedance Change

To model a change in branch impedance rather than a full outage, pass the branch from the
system and a delta impedance value. The branch must belong to the system used to build the
Ybus:

``` @repl tutorial_YbusModification
line = PSY.get_component(PSY.Line, sys, "1")

# Increase reactance by 50%
delta_z = PSY.get_x(line) * 0.5im
mod_impedance = YbusModification(ybus, line, delta_z)
```

### Contingency via Supplemental Attributes

If contingencies are defined as `PowerSystems.Contingency` supplemental attributes, they can
be resolved directly:

``` @repl tutorial_YbusModification
line = PSY.get_component(PSY.Line, sys, "1")
outage = PSY.GeometricDistributionForcedOutage(;
    mean_time_to_recovery = 0.0,
    outage_transition_probability = 0.0,
)
PSY.add_supplemental_attribute!(sys, line, outage)

mod_ctg = YbusModification(ybus, sys, outage)
```

### Combining Modifications

Two modifications on the same Ybus can be combined with `+`:

``` @repl tutorial_YbusModification
mod_a = YbusModification(ybus, PSY.Component[PSY.get_component(PSY.Line, sys, "1")])
mod_b = YbusModification(ybus, PSY.Component[PSY.get_component(PSY.Line, sys, "2")])
mod_combined = mod_a + mod_b
```

### Checking for Island Formation

The `is_islanding` field indicates whether the contingency disconnects the network:

``` @repl tutorial_YbusModification
# Single line outage on a meshed network — typically non-islanding
mod_single = YbusModification(ybus, PSY.Component[PSY.get_component(PSY.Line, sys, "1")])
mod_single.is_islanding
```

### Series Chain Outages with Network Reductions

When network reductions are applied, branches that form series chains are merged into
equivalent arcs. The `YbusModification` handles these transparently:

``` @repl tutorial_YbusModification
sys_14 = PSB.build_system(
    PSB.PSSEParsingTestSystems,
    "psse_14_network_reduction_test_system",
);
ybus_reduced = Ybus(sys_14; network_reductions = NetworkReduction[DegreeTwoReduction()]);

nr = PNM.get_network_reduction_data(ybus_reduced);
```

If a branch in a series chain is tripped, the modification recomputes the chain's equivalent
with the remaining branches and produces the correct delta.

## Design Notes

- **Storage type.** The Ybus uses `ComplexF32` to reduce memory by 50% compared to
  `ComplexF64`. The `YbusModification` delta uses the same element type for consistent
  sparse addition.
- **Performance.** Constructing a `YbusModification` is dominated by the branch Pi-model
  lookups and sparse COO assembly, both ``O(k)`` where ``k`` is the number of tripped
  components. Applying the modification is a single `SparseMatrixCSC` addition, ``O(\text{nnz})``.
  Island detection adds ``O(N + E)`` via union-find over the adjacency structure.
- **Combining with `+`.** The `+` operator uses a conservative OR for `is_islanding`:
  if either operand is islanding, the result is flagged as islanding. For exact island
  detection on combined modifications, construct a single `YbusModification` with all
  components in one call.
