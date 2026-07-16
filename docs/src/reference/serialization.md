# Serialization

This page is the reference for persisting a matrix to disk and reading it back.
For a task recipe see the
[persist-a-PTDF how-to guide](../how_to_guides/generated_serialize_ptdf.md). Both
functions below are documented in full on the [public API page](public.md).

!!! warning "Serialization is PTDF-only and lossy for reductions"
    
    Only the dense [`PTDF`](@ref) type can be serialized. There is **no**
    serialization for `LODF`, `Ybus`, `BA_Matrix`, `ABA_Matrix`, or any of the
    virtual matrices (`VirtualPTDF` / `VirtualLODF` / `VirtualMODF`). There is no
    Arrow, JLD, or generic `save`/`load`, and no `Base.serialize` overload —
    HDF5 is the only supported format. Furthermore, **network reduction data is
    not persisted**: a deserialized `PTDF` is rehydrated with an *empty*
    [`NetworkReductionData`](@ref). If the original matrix was built with
    reductions, that mapping is lost on round-trip. Serialize the fully
    materialized `PTDF` you intend to reuse and treat the file as the matrix, not
    as a recipe for rebuilding one.

## `to_hdf5`

```julia
to_hdf5(ptdf::PTDF, filename::AbstractString;
    compress = true, compression_level = 3, force = false)
```

Serialize a [`PTDF`](@ref) to an HDF5 file.

Arguments and keywords:

  - `ptdf::PTDF`: the matrix to write.
  - `filename::AbstractString`: the file to create.
  - `compress::Bool = true`: whether to enable compression.
  - `compression_level::Int = 3`: deflate level used when compression is enabled.
  - `force::Bool = false`: whether to overwrite `filename` if it already exists. If
    the file exists and `force` is `false`, an error is raised.

The file stores the matrix data (as either a dense `Matrix` or a
`SparseMatrixCSC`, tagged by a `data_type` attribute), the stored `tol` value, the
two axes, the bus and arc lookup dictionaries, and the per-subnetwork axes. It
does **not** store the network reduction data.

## `from_hdf5`

```julia
from_hdf5(::Type{PTDF}, filename::AbstractString) -> PTDF
```

Deserialize a [`PTDF`](@ref) from an HDF5 file previously written by `to_hdf5`.
The reconstructed matrix carries its data, axes, lookups, subnetwork axes, and
`tol`, but its [`NetworkReductionData`](@ref) is empty (`NetworkReductionData()`) —
see the warning above.

### `PTDF(filename)` convenience constructor

```julia
PTDF(filename::AbstractString) -> PTDF
```

A convenience constructor equivalent to `from_hdf5(PTDF, filename)`.

## Round-trip example

```julia
using PowerNetworkMatrices

ptdf = PTDF(sys)                       # build from a PSY.System
to_hdf5(ptdf, "ptdf.h5"; force = true) # write
ptdf2 = from_hdf5(PTDF, "ptdf.h5")     # read back
ptdf3 = PTDF("ptdf.h5")                # equivalent convenience form
```

`ptdf2` and `ptdf3` reproduce the data, axes, lookups, and `tol` of `ptdf`, but
their network reduction data is empty regardless of how `ptdf` was built.

## See also

  - [Persist a PTDF to disk](../how_to_guides/generated_serialize_ptdf.md) — the task-oriented walkthrough.
  - [Matrix type reference](matrix_types.md) — the [`PTDF`](@ref) type.
  - [Network reduction reference](network_reduction.md) — [`NetworkReductionData`](@ref) and why it is not persisted.
  - [Public API](public.md) — full docstrings for [`to_hdf5`](@ref) and [`from_hdf5`](@ref).
