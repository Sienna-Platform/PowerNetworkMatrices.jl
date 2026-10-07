
# The MKL/Pardiso path still uses the package-extension mechanism (Pardiso.jl
# is the only consumer-facing way to access the MKL Pardiso solver). The
# Apple Accelerate path no longer does — `AccelerateWrapper` is built in via
# a `@static if Sys.isapple()` gate.

function _has_mkl_pardiso_ext()
    ext = Base.get_extension(@__MODULE__, :MKLPardisoExt)
    return !isnothing(ext)
end

_mkl_pardiso_install_error() =
    """The MKL/Pardiso extension is not available.
    Install the Pardiso package:
    julia> using Pkg; Pkg.add(\"Pardiso\")"""

# Minimum macOS for the AppleAccelerate (libSparse LU) backend. The
# `SparseFactorizationLU` code is API_AVAILABLE(macos(15.5)); older
# libSparse rejects factorization type 80.
const _AA_MIN_MACOS = v"15.5"

# Query the running macOS product version via the `kern.osproductversion`
# sysctl (libc, no subprocess). Returns a VersionNumber, or v"0" if the
# sysctl is unavailable (treated as "too old").
function _macos_product_version()
    Sys.isapple() || return v"0"
    buf = Vector{UInt8}(undef, 64)
    len = Ref{Csize_t}(length(buf))
    rc = ccall(
        :sysctlbyname, Cint,
        (Cstring, Ptr{UInt8}, Ptr{Csize_t}, Ptr{Cvoid}, Csize_t),
        "kern.osproductversion", buf, len, C_NULL, 0,
    )
    rc == 0 || return v"0"
    s = String(buf[1:(len[] - 1)])  # NUL-terminated; drop the NUL
    try
        return VersionNumber(s)
    catch
        return v"0"
    end
end

_macos_at_least(v::VersionNumber) = _macos_product_version() >= v

_has_apple_accelerate_backend() = Sys.isapple() && _macos_at_least(_AA_MIN_MACOS)

function _apple_accelerate_unavailable_error()
    if Sys.isapple()
        return """The Apple Accelerate sparse backend requires macOS $(_AA_MIN_MACOS.major).$(_AA_MIN_MACOS.minor) or newer \
        (libSparse LU / SparseFactorizationLU is API_AVAILABLE(macos(15.5))); \
        detected macOS $(_macos_product_version()). Use the KLU solver (the default fallback)."""
    end
    return """The Apple Accelerate sparse backend is macOS-only (Sys.isapple() returned false).
    Use the KLU solver (the default) on non-Apple platforms."""
end

"""
    _default_linear_solver() -> String

Default sparse linear solver name. Returns "AppleAccelerateLU" on macOS
$(_AA_MIN_MACOS.major).$(_AA_MIN_MACOS.minor)+ (Apple's built-in libSparse LU via `AccelerateWrapper`)
and "KLU" elsewhere (non-Apple, or macOS older than $(_AA_MIN_MACOS.major).$(_AA_MIN_MACOS.minor)).
Used as the default for the `linear_solver` keyword on PTDF / LODF /
VirtualPTDF / VirtualLODF / VirtualMODF constructors.
"""
function _default_linear_solver()
    if _has_apple_accelerate_backend()
        return "AppleAccelerateLU"
    end
    return "KLU"
end
