# # Network Matrices

# `PowerNetworkMatrices.jl` is able to build classic power systems modeling network matrices such as
# [Ybus](https://en.wikipedia.org/wiki/Nodal_admittance_matrix), [PTDF](https://www.powerworld.com/WebHelp/Content/MainDocumentation_HTML/Power_Transfer_Distribution_Factors.htm) and LODF.

# Check section [Network Matrices](@ref net_mat) for more details.

# ## Overview
#
# Network matrices are implemented in `PowerNetworkMatrices.jl` as arrays that support
# indexing by arc tuples `(from_bus_number, to_bus_number)` and bus numbers.
# The Ybus is stored as a SparseMatrix and the PTDF and LODF are stored as dense matrices.
# **Note**: Ybus is converted to a dense matrix for printing in the REPL.
# The network matrices code implements the dfs algorithm to find islands.

using PowerSystems
import PowerSystems as PSY
DATA_DIR = "../../../data" #hide
system_data = System(joinpath(DATA_DIR, "matpower/case14.m"))

# ## Ybus
#
# The Ybus can be calculated as follows:
ybus = Ybus(system_data)

# The matrix can be indexed using directly the bus numbers. In this example buses are numbered
# 1-14. However, in large systems buses don't usually follow sequential numbering. You can access
# the entries of the Ybus with direct indexing or using the buses.

ybus_entry = ybus[3, 3]
#
bus3 = get_component(Bus, system_data, "Bus 3     HV")
ybus_entry = ybus[bus3, bus3]

# We recognize that many models require matrix operations. For those cases, you can access the
# sparse data as follows:

sparse_array = get_data(ybus)

# ## PTDF
#
# The PTDF matrix can be calculated as follows:
ptdf = PTDF(system_data)

# The PTDF matrix is indexed by **arc tuples** `(from_bus, to_bus)` for branches and
# **bus numbers** for buses. An arc tuple represents a directed connection between two buses.
# For example, an arc `(1, 2)` represents a branch from bus 1 to bus 2.
#
# Elements can be accessed using arc tuples and bus numbers. The `axes` field lists the
# identifiers for each dimension and the `lookup` dictionaries map those identifiers to
# integer matrix indices:

get_axes(ptdf)

get_lookup(ptdf)

# Access a PTDF entry using an arc tuple and bus number taken from the axes:

first_arc = get_axes(ptdf)[2][1]
first_bus = get_axes(ptdf)[1][1]

ptdf_entry = ptdf[first_arc, first_bus]

# PTDF also takes a vector of distributed slacks, for now this feature requires passing a
# vector of weights with the same number of elements as buses in the system. For more details
# check the API entry for [`PTDF`](@ref).

# ## LODF
#
# The LODF matrix is indexed by arc tuples for both dimensions. Each element `lodf[arc_i, arc_j]`
# represents the change in flow on `arc_i` when `arc_j` is taken out of service.

lodf = LODF(system_data)

# Both lookup dictionaries map arc tuples to matrix indices:

get_lookup(lodf)
