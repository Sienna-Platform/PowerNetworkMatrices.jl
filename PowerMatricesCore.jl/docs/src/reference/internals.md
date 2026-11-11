# Internals

The symbols documented on this page are **internal** to `PowerNetworkMatrices` and are
not part of the public API. They are documented here so that the published manual covers
every docstring shipped with the package, but they may change at any time without notice
and should not be relied on by downstream packages.

## `KLUWrapper`

`PowerNetworkMatrices.KLUWrapper` is a thin, allocation-aware wrapper over `libklu`
(provided by `SuiteSparse_jll`) used internally for sparse linear solves. None of these
symbols are exported from `PowerNetworkMatrices`.

```@autodocs
Modules = [PowerNetworkMatrices.KLUWrapper]
```

## `AccelerateWrapper`

`PowerNetworkMatrices.AccelerateWrapper` is a thin, allocation-aware wrapper over Apple's
`libSparse.dylib` (provided by the system Accelerate framework) used internally for sparse
linear solves on macOS. Non-Apple builds load stub fallbacks that throw on use. None of
these symbols are exported from `PowerNetworkMatrices`.

```@autodocs
Modules = [PowerNetworkMatrices.AccelerateWrapper]
```
