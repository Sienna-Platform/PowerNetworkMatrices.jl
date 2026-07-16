# # How to Factorize and Reuse an ABA Matrix

# This guide shows you how to build an [`ABA_Matrix`](@ref) with a stored KLU
# factorization, factorize one after the fact, and check whether a factorization
# is present. The factorization is what makes repeated DC power-flow solves
# cheap: it is computed once and reused for every right-hand side.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` installed
#   - A power system model loaded (see [Getting Started](@ref))

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Factorize at construction

# Pass `factorize = true` to compute and store the KLU factorization while the
# matrix is built:

aba = PNM.ABA_Matrix(sys; factorize = true)

# Confirm the factorization is present with [`is_factorized`](@ref):

PNM.is_factorized(aba)

# ## Factorize an existing matrix

# If you already built an `ABA_Matrix` without a factorization (the default,
# `factorize = false`), add one later with [`factorize`](@ref). This returns a
# new `ABA_Matrix` that carries the factorization; the original is unchanged.

aba_unfactorized = PNM.ABA_Matrix(sys)
PNM.is_factorized(aba_unfactorized)

#

aba_factorized = PNM.factorize(aba_unfactorized)
PNM.is_factorized(aba_factorized)

# ## Reuse the factorization

# The factorization is stored on the matrix and reused automatically by
# downstream calculations (e.g. [`PTDF`](@ref) built from the ABA/incidence
# matrices). Build it once and keep the object around rather than reconstructing
# it per solve.

# ## Notes
#
#   - Reference buses are excluded from `ABA_Matrix` so the matrix is invertible.
#   - `factorize` returns a fresh, factorized copy; it does not mutate its input.
#   - See the [matrix type reference](../reference/matrix_types.md) for the
#     `ABA_Matrix` structure and its factorized/unfactorized type aliases.
