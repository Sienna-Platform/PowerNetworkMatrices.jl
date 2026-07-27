"""
    AutoTolerance(; data_precision = :auto, safety = 1.0, quantile = 0.5)

Request automatic, condition-aware sparsification of a PTDF/LODF matrix. The
matrix is sparsified with a *relative* per-row cutoff: an entry is dropped when
```math
|\\mathrm{entry}| < \\alpha \\cdot \\max|\\mathrm{row}|,
\\qquad \\alpha = \\mathrm{clamp}(\\mathrm{safety} \\cdot \\delta, \\, 10^{-6}, \\, 10^{-2}),
```
where ``\\delta`` is the relative precision of the branch data. Because the cutoff is
relative to each row's own peak, columns of large, ill-conditioned systems stay
sparse regardless of the conditioning of `ABA`; the 1-norm condition estimate of
`ABA` is still computed and logged as a diagnostic, but never multiplies the
cutoff.

- `data_precision`: relative precision ``\\delta`` of the branch parameters. `:auto`
  (default) discovers it from the branch susceptances (see
  [`discover_data_precision`](@ref)); a `Float64` sets it explicitly (e.g. `1e-3`
  for reactances good to 0.1%).
- `safety`: aggressiveness multiplier on `δ`; `> 1` sparsifies more, `< 1` less.
- `quantile`: only used when `data_precision = :auto`; which quantile of the
  per-branch significant-figure counts to adopt.

# Where the cutoff applies

An `AutoTolerance` is a **no-op below `AUTO_TOLERANCE_BUS_LIMIT`** (2000 buses): small
systems and the test cases are returned exactly, and its relative drop is reserved for
the large virtual matrices ([`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref) /
[`VirtualMODF`](@ref)). On the **dense** [`PTDF`](@ref) / [`LODF`](@ref) path it is also
a no-op, preserving the `Matrix{Float64}` element type.

A plain `Float64` `tol` is accepted by every constructor and applies a fixed *absolute*
cutoff (``|\\mathrm{entry}| < \\mathrm{tol}``) at any system size, dense or virtual — the backward-compatible,
exact-tolerance path.

# Examples

```julia
PTDF(sys; tol = 1e-5)                                  # fixed absolute cutoff, any size
PTDF(sys; tol = AutoTolerance(; safety = 5.0))         # sparsify large virtual matrices harder
PTDF(sys; tol = AutoTolerance(; data_precision = 1e-3))  # pin precision instead of discovering it
```
"""
struct AutoTolerance{D <: Union{Float64, Symbol}}
    data_precision::D
    safety::Float64
    quantile::Float64
end

# Validate via dispatch (no `isa`): a Symbol must be exactly :auto.
_validate_data_precision(dp::Float64) = dp
function _validate_data_precision(dp::Symbol)
    dp === :auto || throw(ArgumentError("data_precision Symbol must be :auto"))
    return dp
end

function AutoTolerance(;
    data_precision::Union{Float64, Symbol} = :auto,
    safety::Float64 = 1.0,
    quantile::Float64 = 0.5,
)
    _validate_data_precision(data_precision)
    return AutoTolerance(data_precision, safety, quantile)
end

# Module-wide default: discover the branch-data precision and sparsify relative
# to it. Used as the default `tol` on every PTDF/LODF constructor.
const DEFAULT_AUTO_TOLERANCE = AutoTolerance()

# Bounds on the relative per-row cutoff fraction α. The floor keeps full-
# precision data from collapsing α to zero (no sparsification at all); the
# ceiling keeps α well below 1 so a row's own peak entry is never dropped.
const MIN_RELATIVE_TOLERANCE = 1e-6
const MAX_RELATIVE_TOLERANCE = 1e-2

# Resolved sparsification rule, produced once at construction and applied to
# each computed row. `AbsoluteCutoff` drops entries below a fixed value (the
# Float64 / backward-compatible path); `RelativeCutoff` drops entries below
# `fraction · max|row|` (the AutoTolerance path), so per-column sparsity is
# independent of the matrix scale and conditioning.
struct AbsoluteCutoff
    value::Float64
end

struct RelativeCutoff
    fraction::Float64
end

const SparsificationCutoff = Union{AbsoluteCutoff, RelativeCutoff}

# Float64 representative of a cutoff, returned by `get_tol`.
cutoff_value(c::AbsoluteCutoff) = c.value
cutoff_value(c::RelativeCutoff) = c.fraction

# Apply the rule to a freshly computed dense row, returning the (possibly sparse)
# row to store in the cache. A cutoff at/below eps() keeps the dense row.
function apply_cutoff(c::AbsoluteCutoff, row::Vector{Float64})
    c.value > eps() || return row
    return sparsify(row, c.value)
end

function apply_cutoff(c::RelativeCutoff, row::Vector{Float64})
    isempty(row) && return row
    threshold = c.fraction * maximum(abs, row)
    threshold > eps() || return row
    return sparsify(row, threshold)
end

# Significant figures of v: smallest s in 1:maxs that reproduces v to within
# rtol. Returns maxs when no small s fits (full precision), 0 for zero/non-finite.
function _sigfigs(v::Float64, maxs::Int, rtol::Float64)
    (iszero(v) || !isfinite(v)) && return 0
    av = abs(v)
    for s in 1:maxs
        abs(round(av; sigdigits = s) - av) <= rtol * av && return s
    end
    return maxs
end

# Linear-interpolated quantile of an already-sorted integer vector. Avoids
# declaring Statistics as a direct dependency for the one place we need a quantile.
function _quantile_sorted(v::Vector{Int}, q::Float64)
    n = length(v)
    n == 1 && return Float64(v[1])
    h = (n - 1) * q + 1.0
    lo = floor(Int, h)
    lo >= n && return Float64(v[n])
    frac = h - lo
    return v[lo] + frac * (v[lo + 1] - v[lo])
end

"""
    discover_data_precision(susceptances; q = 0.5, maxsig = 10, rtol = 1e-9) -> Float64

Estimate relative data precision from branch susceptances ``b_k``. Recovers the
reactances ``x_k = 1/b_k`` (the susceptance hides the original precision; the
reciprocal does not), counts the significant figures of each, and returns
``0.5 \\cdot 10^{-(s-1)}`` at the ``q``-quantile of those counts, clamped to `[eps, 1e-2]`.
`maxsig` is coupled to `rtol`: rounding to ``s`` figures carries a relative error of
``0.5 \\cdot 10^{1-s}``, so `rtol = 1e-9` first accepts ``s = 10`` and no real data resolves
further. Full-precision data (e.g. computed equivalent branches) hits this `maxsig`
cap and yields the tightest precision ``5 \\times 10^{-10}``.
"""
function discover_data_precision(
    susceptances::AbstractVector{Float64};
    q::Float64 = 0.5,
    maxsig::Int = 10,
    rtol::Float64 = 1e-9,
)
    counts = Int[]
    for b in susceptances
        (iszero(b) || !isfinite(b)) && continue
        x = inv(abs(b))
        isfinite(x) || continue
        push!(counts, _sigfigs(x, maxsig, rtol))
    end
    isempty(counts) && return eps(Float64)
    s = clamp(ceil(Int, _quantile_sorted(sort!(counts), q)), 1, maxsig)
    return clamp(0.5 * 10.0^(-(s - 1)), eps(Float64), 1e-2)
end

# Resolve a spec's data_precision to a concrete Float64 via dispatch (no `isa`).
_resolve_delta(dp::Float64, ::AbstractVector{Float64}, ::Float64) = dp
function _resolve_delta(::Symbol, susceptances::AbstractVector{Float64}, q::Float64)
    delta = discover_data_precision(susceptances; q = q)
    @info "AutoTolerance: data precision discovered from branch susceptances" delta maxlog =
        1
    return delta
end

_effective_delta(spec::AutoTolerance, susceptances::AbstractVector{Float64}) =
    _resolve_delta(spec.data_precision, susceptances, spec.quantile)

# Relative per-row cutoff fraction α = clamp(safety · δ, MIN, MAX).
function _relative_alpha(spec::AutoTolerance, susceptances::AbstractVector{Float64})
    delta = _effective_delta(spec, susceptances)
    return clamp(
        spec.safety * delta,
        MIN_RELATIVE_TOLERANCE,
        MAX_RELATIVE_TOLERANCE,
    )
end

# Hager's algorithm for ‖B‖₁ where `applyB!(v)` overwrites v with B·v. B = ABA⁻¹
# is symmetric here, so Bᵀ·v = B·v and only the forward solve is needed. Bounded
# iterations guard against non-convergence; γ is the running lower bound.
function _hager_invnorm1(applyB!, n::Int; itmax::Int = 5)
    x = fill(1.0 / n, n)
    y = similar(x)
    γ = 0.0
    for _ in 1:itmax
        copyto!(y, x)
        applyB!(y)                              # y = B x
        γ = sum(abs, y)
        n == 1 && return γ
        @inbounds for i in 1:n
            if y[i] >= 0.0
                y[i] = 1.0
            else
                y[i] = -1.0
            end
        end
        applyB!(y)                              # y = Bᵀ ξ = B ξ (symmetric)
        j = 1
        m = abs(y[1])
        @inbounds for i in 2:n
            if abs(y[i]) > m
                m = abs(y[i])
                j = i
            end
        end
        zx = 0.0
        @inbounds for i in 1:n
            zx += y[i] * x[i]
        end
        m <= zx && return γ                     # Hager convergence test
        fill!(x, 0.0)
        x[j] = 1.0
    end
    return γ
end

# κ₁(ABA) = ‖ABA‖₁ · ‖ABA⁻¹‖₁. KLU has a native estimate; AA reuses the existing
# factorization through the symmetric Hager estimator above. The annotation
# restores inference, since the untyped libklu cache pointers make `condest!`
# infer as Any.
_condition_estimate(
    cache::KLULinSolveCache{Float64, Int},
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
)::Float64 = condest!(cache)

function _condition_estimate(
    cache::AAFactorCache,
    aba::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    # AccelerateWrapper.solve! overwrites its vector argument with ABA⁻¹·v.
    applyB! = let c = cache
        v -> AccelerateWrapper.solve!(c, v)
    end
    return LinearAlgebra.opnorm(aba, 1) * _hager_invnorm1(applyB!, size(aba, 2))
end

# Compute and log the ABA condition estimate as a diagnostic. The estimate is
# never used in the cutoff (AutoTolerance is purely relative); it is surfaced so
# users can see how ill-conditioned the network is. Reuses the factorization the
# caller already built, so it costs ~two extra solves.
function _log_condition_estimate(
    cache,
    aba::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    kappa = _condition_estimate(cache, aba)
    @info "AutoTolerance: ABA 1-norm condition estimate (diagnostic only)" kappa maxlog =
        1
    return kappa
end

# Dense PTDF/LODF tolerance (all dense constructors share this). The dense matrix
# is the small-system path, so a numeric tol is honored as an absolute cutoff and
# an AutoTolerance is a no-op (eps) — keeping the dense `Matrix{Float64}` type;
# the relative drop is reserved for the on-demand virtual matrices below.
_dense_tol(tol::Float64) = tol
_dense_tol(::AutoTolerance) = eps(Float64)

# ----------------------------------------------------------------------------
# Virtual resolvers: produce a `SparsificationCutoff` stored on the matrix and
# applied per-row in `cached_row_lookup`. A Float64 maps to a fixed absolute
# cutoff (honored at any size). An AutoTolerance maps to a relative cutoff so
# requested columns of large cases come back sparse — but only at or above
# AUTO_TOLERANCE_BUS_LIMIT; on smaller systems it is a no-op (exact rows).
# ----------------------------------------------------------------------------
_resolve_virtual_cutoff(
    tol::Float64,
    _cache,
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
    ::AbstractVector{Float64},
) = AbsoluteCutoff(tol)

function _resolve_virtual_cutoff(
    spec::AutoTolerance,
    cache,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
    susceptances::AbstractVector{Float64},
)
    size(ABA, 2) < AUTO_TOLERANCE_BUS_LIMIT && return AbsoluteCutoff(eps(Float64))
    _log_condition_estimate(cache, ABA)
    return RelativeCutoff(_relative_alpha(spec, susceptances))
end
