# # How to Build Multiple Matrices Without Repeating Work

# Every matrix constructor that takes a [`System`](@extref PowerSystems.System)
# rebuilds the same intermediates from scratch — the [`Ybus`](@ref), the incidence
# matrix `A`, and the susceptance-weighted `BA`. When you need several matrices for
# the same system, that repeated work adds up. This guide shows how to compute the
# shared pieces once and feed them to the constructors that accept pre-built
# matrices.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Build the shared intermediates once

# The construction dependency chain is
#
# > [`Ybus`](@ref) → ([`IncidenceMatrix`](@ref), [`BA_Matrix`](@ref)) →
# > [`ABA_Matrix`](@ref) / [`PTDF`](@ref), and [`PTDF`](@ref) → [`LODF`](@ref).
#
# The [`Ybus`](@ref) is the expensive shared root. Build it — and the incidence and
# BA matrices derived from it — exactly once:

ybus = PNM.Ybus(sys)
A = PNM.IncidenceMatrix(ybus)
BA = PNM.BA_Matrix(ybus)

# ## Reuse them across constructors

# [`PTDF`](@ref) accepts the incidence and BA matrices directly, skipping its own
# [`Ybus`](@ref) build:

ptdf = PNM.PTDF(A, BA)

# [`LODF`](@ref) can be built straight from a [`PTDF`](@ref) you already have,
# reusing that work too — no second factorization of the network:

lodf = PNM.LODF(A, ptdf)

# Alternatively, the factorized [`ABA_Matrix`](@ref) route builds [`LODF`](@ref)
# from the same `A` and `BA`. All three inputs must share the same network
# reduction — which they do here, because they all descend from one `ybus`:

aba = PNM.ABA_Matrix(ybus; factorize = true)
lodf_via_aba = PNM.LODF(A, aba, BA)

# Virtual matrices likewise accept a pre-built [`Ybus`](@ref), so the lazy forms
# reuse the same root:

vptdf = PNM.VirtualPTDF(ybus)

# !!! note "Keep reductions consistent"
#
#     Constructors that combine pre-built matrices (e.g. `LODF(A, ABA, BA)`) require
#     every input to have been built with the **same** `network_reductions`. Because
#     they all derive from a single [`Ybus`](@ref) here, they are automatically
#     consistent. Pass `network_reductions` once, to the [`Ybus`](@ref) call, and
#     everything downstream inherits it. See
#     [How to Apply Network Reductions at Construction](@ref).

# ## When this matters

# For a small system the savings are negligible — build straight from the
# [`System`](@extref PowerSystems.System) and keep the code simple. Reuse pays off
# when the [`Ybus`](@ref) build and factorization are expensive (large systems) and
# you need more than one matrix: for example a [`PTDF`](@ref) and a [`LODF`](@ref),
# or several matrices under the same network reduction.

# ## See also
#
#   - [Matrix overview & indexing](@ref) — every matrix type, its axes, and how the
#     shared intermediates fit together.
#   - [How to Choose a Linear Solver](@ref) — the factorization cost that reuse
#     avoids repeating.
#   - [How to Apply Network Reductions at Construction](@ref) — supplying reductions
#     to the shared [`Ybus`](@ref).
