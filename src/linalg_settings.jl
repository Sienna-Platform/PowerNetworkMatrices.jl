
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
    if Sys.isapple() && _macos_at_least(_AA_MIN_MACOS)
        return "AppleAccelerateLU"
    end
    return "KLU"
end

"Set a preference of the backend library for sparse linear algebra operations."
function set_linalg_backend_preference(linalglib::Union{String, Nothing})
    if !isnothing(linalglib) &&
       !(linalglib in ["MKLPardiso", "AppleAccelerateLU", "AppleAccelerate"])
        throw(
            ArgumentError(
                "Unsupported sparse linear algebra backend requested: $(linalglib)",
            ),
        )
    end
    Preferences.@set_preferences!("linalg_backend" => linalglib)
    @info("""Sparse linear algebra backend library: preference set to $(linalglib);
          you may need to restart your Julia session for this change to take effect.""")
end

set_linalg_backend_preference() = set_linalg_backend_preference(nothing)
set_linalg_backend_preference(linalglib::Symbol) =
    set_linalg_backend_preference(String(linalglib))

get_linalg_backend_preference() = Preferences.@load_preference("linalg_backend")

"Set a preference whether to run `check_linalg_backend` at the package loading time."
set_linalg_backend_check(check::Bool) =
    Preferences.@set_preferences!("linalg_backend_check" => check)

get_linalg_backend_check() = Preferences.@load_preference("linalg_backend_check")

function check_lbt_library()
    lb_msg(lib) = """The $(lib) library is being used for Julia's BLAS and LAPACK routines,
                 including dense linear algebra operations such as `BLAS.gemm`."""

    blas_config = lowercase(string(LinearAlgebra.BLAS.get_config()))
    if contains(blas_config, "mkl")
        @info lb_msg("Intel MKL")
    elseif contains(blas_config, "accelerate")
        @info lb_msg("AppleAccelerate")
    else
        @info lb_msg("default or user-provided")
    end
end

"""
Check for the recommended linear algebra sparse solver option for the
user's operating system and whether it matches their preferred one.
"""
function check_linalg_backend()
    check_lbt_library()

    user_linalg_backend = get_linalg_backend_preference()
    if !isnothing(user_linalg_backend)
        @info """The sparse linear algebra solver preference has been set to $(user_linalg_backend).
                To change this for your active project, call the function
                PowerNetworkMatrices.set_linalg_backend_preference()
                with one of "MKLPardiso", "AppleAccelerateLU", or `nothing` to turn off.
              """
    end

    no_msg(lib) = """For faster sparse linear solving operations, consider using $(lib):
                  pkg> add $(lib) # if not in your active project
                  using $(lib)    # after loading PowerNetworkMatrices and before any matrix operations
                  Sparse factorization uses KLU by default otherwise."""
    go_msg(lib) = "The linear algebra backend $(lib) is loaded for sparse solving."
    yo_msg(lib) = """The $(lib) extension for PowerNetworkMatrices is not loaded
                     even though the corresponding linear algebra library was requested."""

    if !Sys.isapple()
        if _has_mkl_pardiso_ext()
            @info go_msg("MKLPardiso")
        else
            if user_linalg_backend == "MKLPardiso"
                @warn yo_msg("MKLPardiso")
            end
            @info no_msg("Pardiso")
            @info "See https://github.com/JuliaSparse/Pardiso.jl for more details."
        end
        if (
            user_linalg_backend !== nothing &&
            startswith(user_linalg_backend, "AppleAccelerate")
        )
            @warn "AppleAccelerate is not supported on non-Apple systems."
        end
    end

    if Sys.isapple()
        if user_linalg_backend == "MKLPardiso"
            if Sys.ARCH == :x86_64
                @info """Note that MKLPardiso can be used on Apple macOS x86_64, however the
                    user will need to pin MKL_jll to version 2023 in their project environment:
                        using Pkg
                        if Sys.isapple() && (Sys.ARCH == :x86_64)
                            Pkg.add(name="MKL_jll"; version = "2023")
                        end
                        Pkg.add("Pardiso")
                    Once this is done for a project, run:
                        using Pardiso
                    after loading PowerNetworkMatrices.
                        """
            elseif Sys.ARCH == :aarch64
                @warn "MKLPardiso is not supported on Apple ARM (M series) chips."
            end
        end

        if _has_apple_accelerate_backend()
            @info go_msg("AppleAccelerate")
        else
            if (
                user_linalg_backend !== nothing &&
                startswith(user_linalg_backend, "AppleAccelerate")
            )
                @warn """AppleAccelerate was requested but is unavailable: it requires \
                macOS $(_AA_MIN_MACOS.major).$(_AA_MIN_MACOS.minor)+ (detected macOS $(_macos_product_version())). \
                Falling back to KLU."""
            end
        end
    end
end
