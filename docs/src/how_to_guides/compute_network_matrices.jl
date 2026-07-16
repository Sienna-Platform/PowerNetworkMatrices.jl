# # How to Compute Network Matrices

# This guide shows you how to build the network matrices for your power system.
# They all share one construction pattern and one indexing scheme, so once you
# can build and read a `PTDF` you can do the same for every other type.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` installed
#   - A power system model loaded (see [Getting Started](@ref))
#
# The examples below load a small test system with `PowerSystemCaseBuilder`; in
# your own work replace this with your own `System`.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Build any matrix the same way

# Every matrix type is a constructor that takes the `System` and returns the
# matrix object. The call is identical across types — only the name changes:

ptdf_matrix = PNM.PTDF(sys)

# The same `Type(sys; kwargs...)` form builds every matrix in the package:
#
# | Constructor                                                          | Matrix                              |
# |:-------------------------------------------------------------------- |:----------------------------------- |
# | [`PTDF`](@ref)                                                       | Power transfer distribution factors |
# | [`LODF`](@ref)                                                       | Line outage distribution factors    |
# | [`Ybus`](@ref)                                                       | Complex nodal admittance            |
# | [`BA_Matrix`](@ref) / [`ABA_Matrix`](@ref)                           | DC susceptance forms                |
# | [`IncidenceMatrix`](@ref) / [`AdjacencyMatrix`](@ref)                | Network topology                    |
# | [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref) / [`VirtualMODF`](@ref) | Lazy, row-on-demand forms          |
#
# So `PNM.LODF(sys)`, `PNM.Ybus(sys)`, `PNM.ABA_Matrix(sys)`, and the rest are all
# built exactly like the `PNM.PTDF(sys)` call above. The shared build-time
# keywords — `network_reductions`, `tol`, `linear_solver`, `dist_slack` — work on
# every constructor that accepts them; each has its own how-to.

# Pull the underlying array out with the matching `get_*_data` accessor when you
# need the raw `SparseMatrixCSC` / `Matrix` (`get_ptdf_data`, `get_lodf_data`, …):

matrix_data = PNM.get_ptdf_data(ptdf_matrix);

# ## Index any matrix the same way

# Matrices are indexed by **physical network elements** — bus numbers and **arc
# tuples** `(from_bus, to_bus)` — mapped to internal positions for you. What
# differs between types is only *which* identifier each dimension takes:

# | Matrix            | Dimension 1 (rows) | Dimension 2 (columns) |
# |:----------------- |:------------------ |:--------------------- |
# | `IncidenceMatrix` | Arc tuples         | Bus numbers           |
# | `PTDF`            | Arc tuples         | Bus numbers           |
# | `LODF`            | Arc tuples         | Arc tuples            |
# | `Ybus`            | Bus numbers        | Bus numbers           |
# | `VirtualPTDF`     | Arc tuples         | Bus numbers           |
# | `VirtualLODF`     | Arc tuples         | Arc tuples            |
# | `VirtualMODF`     | Arc tuples         | Bus numbers           |

# A `PTDF` entry is a `(monitored arc, injection bus)` sensitivity:

ptdf_matrix[(1, 2), 3]

# A `LODF` entry is a `(monitored arc, outaged arc)` sensitivity — here *both*
# dimensions are arc tuples, but the indexing call looks the same:

lodf_matrix = PNM.LODF(sys)
lodf_matrix[(1, 4), (2, 3)]

# The `axes` (identifiers per dimension) and `lookup` (identifier → integer index)
# behind this mapping are available on any matrix:

PNM.get_axes(ptdf_matrix)

#

PNM.get_lookup(ptdf_matrix)

# !!! note
#     For backward compatibility, branch **name** strings can also index `PTDF`
#     and `LODF` (mapped to arc tuples internally via `get_branch_multiplier`).
#     Arc tuples are the recommended, unambiguous form.

# ## Virtual matrices are drop-in

# The lazy [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref) / [`VirtualMODF`](@ref)
# forms build and index **exactly** like their materialized counterparts — the
# only difference is that they compute rows on demand and cache them instead of
# storing the whole matrix. Swap the type name; everything else stays the same:

vptdf_matrix = PNM.VirtualPTDF(sys)
vptdf_matrix[(1, 2), 3]

# `VirtualMODF` additionally serves post-contingency / post-modification rows; it
# needs `PSY.Outage` supplemental attributes on the system to auto-register
# contingencies (see [How to Define and Apply Contingencies](@ref)):

# ```julia
# vmodf_matrix = PNM.VirtualMODF(sys)
# contingency = first(values(PNM.get_registered_contingencies(vmodf_matrix)))
# vmodf_matrix[(1, 2), contingency]
# ```
#
# For when to prefer a virtual matrix over a materialized one, see
# [Virtual vs. Materialized Matrices](../explanation/virtual_vs_materialized.md).

# ## Building from pre-computed matrices

# Some constructors accept already-built matrices instead of a `System`, to reuse
# shared intermediates. For example, `PTDF` can be built from an
# [`IncidenceMatrix`](@ref) and a [`BA_Matrix`](@ref):

a_matrix = PNM.IncidenceMatrix(sys)
ba_matrix = PNM.BA_Matrix(sys)
ptdf_matrix = PNM.PTDF(a_matrix, ba_matrix)

# ## Next Steps
#
#   - [How to Choose a Linear Solver](@ref) for optimal performance
#   - [Matrix Overview and Indexing](../reference/network_matrices_overview.md) —
#     the reference for every type's axes and indexing
#   - Understand the theory behind network matrices in the Explanation section
