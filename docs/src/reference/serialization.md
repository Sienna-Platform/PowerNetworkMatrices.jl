# Serialization

This page is the reference for persisting a matrix to disk and reading it back.
Both functions below are documented in full, with signatures, on the
[public API page](public.md).

!!! warning "Serialization is PTDF-only and lossy for reductions"
    
    Only the dense [`PTDF`](@ref) type can be serialized. There is **no**
    serialization for [`LODF`](@ref), [`Ybus`](@ref), [`BA_Matrix`](@ref),
    [`ABA_Matrix`](@ref), or any virtual matrix, and no Arrow/JLD/`Base.serialize`
    path — HDF5 is the only supported format. **Network reduction data is not
    persisted**: a deserialized [`PTDF`](@ref) is rehydrated with an *empty*
    [`NetworkReductionData`](@ref), so if the original was built with reductions
    that mapping is lost on round-trip. Treat the file as the matrix, not as a
    recipe for rebuilding one.

## Functions

  - **[`to_hdf5`](@ref)** — write a [`PTDF`](@ref) to an HDF5 file (`compress`,
    `compression_level`, `force` keywords). The file stores the matrix data (dense
    `Matrix` or [`SparseMatrixCSC`](@extref Julia SparseArrays.SparseMatrixCSC),
    tagged by a `data_type` attribute), the stored `tol`, the two axes, the bus/arc
    lookups, and the per-subnetwork axes — but **not** the reduction data.
  - **[`from_hdf5`](@ref)** — read a [`PTDF`](@ref) back. Equivalent convenience
    constructor: `PTDF(filename)`.

## Round-trip example

```julia
using PowerNetworkMatrices

ptdf = PTDF(sys)                       # build from a PSY.System
to_hdf5(ptdf, "ptdf.h5"; force = true) # write
ptdf2 = from_hdf5(PTDF, "ptdf.h5")     # read back
ptdf3 = PTDF("ptdf.h5")                # equivalent convenience form
```

`ptdf2` and `ptdf3` reproduce the data, axes, lookups, and `tol` of `ptdf`, but
their reduction data is empty regardless of how `ptdf` was built.

## See also

  - [Matrix type reference](matrix_types.md) — the [`PTDF`](@ref) type.
  - [`NetworkReductionData`](@ref) — the reduction record, and why it is not persisted.
  - [Public API](public.md) — full docstrings for [`to_hdf5`](@ref) and [`from_hdf5`](@ref).
