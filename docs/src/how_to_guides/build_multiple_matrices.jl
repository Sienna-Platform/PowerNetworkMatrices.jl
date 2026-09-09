# # How to Build Multiple Matrices Without Repeating Work

# Every matrix constructor that takes a [`System`](@extref PowerSystems.System)
# rebuilds the same intermediates from scratch — the [`Ybus`](@ref), the incidence
# matrix `A` ([`IncidenceMatrix`](@ref)), and the susceptance-weighted `BA`
# ([`BA_Matrix`](@ref)). This guide shows how to compute the shared pieces once and
# feed them to the constructors that accept pre-built matrices.

using PowerNetworkMatrices
import PowerSystemCaseBuilder

# Load an example test system with [`PowerSystemCaseBuilder.build_system`](@extref):

sys = PowerSystemCaseBuilder.build_system(
    PowerSystemCaseBuilder.PSITestSystems,
    "c_sys5",
);

# ## Build the shared intermediates once

# The construction dependency chain is
#
# > `Ybus` → `IncidenceMatrix`, `BA_Matrix` →
# > `ABA_Matrix` / `PTDF`, and `PTDF` → `LODF`.
#
# The [`Ybus`](@ref) is the expensive shared root. Build it — and the
# [`IncidenceMatrix`](@ref) and [`BA_Matrix`](@ref) derived from it — exactly once:

ybus = Ybus(sys)
A = IncidenceMatrix(ybus)
BA = BA_Matrix(ybus)

# ## Reuse them across constructors

# [`PTDF`](@ref) accepts the incidence and BA matrices directly, skipping its own
# [`Ybus`](@ref) build:

ptdf = PTDF(A, BA)

# [`LODF`](@ref) can be built straight from a [`PTDF`](@ref) you already have,
# reusing that work too — no second factorization of the network:

lodf = LODF(A, ptdf)

# Alternatively, the factorized [`ABA_Matrix`](@ref) route builds [`LODF`](@ref)
# from the same `A` and `BA`. The `factorize` keyword is what pre-factorizes it;
# [`is_factorized`](@ref) reports the result. All three inputs must share the same
# network reduction — which they do here, because they all descend from one `ybus`:

aba = ABA_Matrix(ybus; factorize = true)
is_factorized(aba)

# That factorized [`ABA_Matrix`](@ref), the incidence matrix, and the BA matrix are all
# [`LODF`](@ref) needs:

lodf_via_aba = LODF(A, aba, BA)

# Virtual matrices likewise accept a pre-built [`Ybus`](@ref), so [`VirtualPTDF`](@ref)
# and its siblings reuse the same root:

vptdf = VirtualPTDF(ybus)

# !!! note "Keep reductions consistent"
#
#     Constructors that combine pre-built matrices (e.g. `LODF(A, ABA, BA)`) require
#     every input to have been built with the **same** `network_reductions`. Because
#     they all derive from a single [`Ybus`](@ref) here, they are automatically
#     consistent. Pass `network_reductions` once, to the [`Ybus`](@ref) call, and
#     everything downstream inherits it. See the [`NetworkReduction`](@ref) docstring
#     for the keyword and its rules.

# ## See also
#
#   - [Matrix overview & indexing](@ref) — every matrix type, its axes, and how the
#     shared intermediates fit together.
#   - [How to Choose a Linear Solver](@ref) — the factorization cost that reuse
#     avoids repeating.
#   - [`NetworkReduction`](@ref) — supplying reductions via `network_reductions`
#     to the shared [`Ybus`](@ref).
