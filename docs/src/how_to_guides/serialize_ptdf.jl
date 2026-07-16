# # How to Persist a PTDF to Disk

# This guide shows you how to write a [`PTDF`](@ref) to an HDF5 file and read it
# back. Serializing a large PTDF once and reloading it avoids recomputing the
# matrix on every run.

# !!! warning "Serialization is PTDF-only and lossy for reductions"
#
#     Only the dense [`PTDF`](@ref) type can be serialized. There is **no**
#     serialization for `LODF`, `VirtualPTDF`, `VirtualLODF`, `VirtualMODF`,
#     `Ybus`, `BA_Matrix`, or `ABA_Matrix`, and **no** Arrow/JLD/generic
#     `save`/`load` or `Base.serialize` support.
#
#     Network reduction data is **not persisted**. A PTDF read back from disk is
#     rehydrated with an *empty* `NetworkReductionData`. If your matrix was built
#     with `network_reductions`, that context is lost on reload — keep the
#     construction code if you need the reduction metadata.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` installed
#   - A power system model loaded (see [Getting Started](@ref))

using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

ptdf = PNM.PTDF(sys)

# ## Write the PTDF to disk

# Use [`to_hdf5`](@ref) to serialize. Compression is on by default:

filename = joinpath(mktempdir(), "ptdf.h5")
PNM.to_hdf5(ptdf, filename)

# `to_hdf5` refuses to overwrite an existing file unless you pass `force = true`.
# Tune compression with `compress` and `compression_level`:

PNM.to_hdf5(ptdf, filename; compress = true, compression_level = 5, force = true)

# ## Read the PTDF back

# Use [`from_hdf5`](@ref), passing the `PTDF` type as the first argument:

ptdf_loaded = PNM.from_hdf5(PNM.PTDF, filename)

# The single-argument `PTDF(filename)` constructor is a convenience shorthand for
# the same call:

ptdf_loaded = PNM.PTDF(filename)

# Indexing behaves identically to the freshly computed matrix:

ptdf_loaded[(1, 2), 3]

# ## Notes
#
#   - The `tol` used at construction is preserved across a round-trip; the
#     reduction data is not (see the warning above).
#   - See the [serialization reference](../reference/serialization.md) for the
#     full signatures.
