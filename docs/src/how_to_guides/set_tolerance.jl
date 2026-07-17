# # How to Set the Sparsification Tolerance

# The `tol` keyword controls how aggressively [`PTDF`](@ref) / [`LODF`](@ref) rows
# drop near-zero entries. This guide gives the recipe for each common goal. For
# the full settings reference — the [`AutoTolerance`](@ref) fields, the cutoff
# formula, and the bus-count gate — see
# [Tolerance and solver settings](../reference/tolerance_and_solvers.md).

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# `tol` accepts either a `Float64` — a fixed **absolute** cutoff that drops
# entries with `|x| < tol` at any system size — or an [`AutoTolerance`](@ref) — an
# automatic **relative per-row** cutoff, active only on large virtual matrices. It
# defaults to `AutoTolerance()`.

# ## Goal: keep the matrix exact

# On a small or medium system the default already keeps every entry:
# [`AutoTolerance`](@ref) is a no-op below the bus-count gate, so nothing is
# dropped.

ptdf = PNM.PTDF(sys);

# ## Goal: apply a fixed, size-independent cutoff

# Pass a `Float64` to drop any entry below an absolute magnitude, regardless of
# system size:

ptdf = PNM.PTDF(sys; tol = 1e-5);

# ## Goal: sparsify a large virtual matrix more aggressively

# On large virtual matrices, raise the `safety` multiplier to drop more entries,
# trading a little accuracy for memory and speed:

ptdf = PNM.PTDF(sys; tol = PNM.AutoTolerance(; safety = 5.0));

# ## Goal: match your branch-data precision

# If you know how precise your branch parameters are, pin the relative precision
# instead of letting it be discovered — for example, reactances good to 0.1%:

ptdf = PNM.PTDF(sys; tol = PNM.AutoTolerance(; data_precision = 1e-3));

# To see what precision the automatic rule would infer from your data, call
# [`discover_data_precision`](@ref) on the branch **susceptances** directly:

susceptances = [100.0, 25.0, 33.33333]
PNM.discover_data_precision(susceptances)

# ## Where each cutoff applies

# `AutoTolerance` is a **no-op on small systems** (below the bus-count gate), so
# every small test case stays exact; its relative drop is reserved for large
# virtual matrices. A `Float64` `tol` is honored at any size. The dense
# `PTDF`/`LODF` path never sparsifies under `AutoTolerance` (it preserves the
# dense `Matrix{Float64}`), but a `Float64` `tol` still applies there.

# See [Tolerance and solver settings](../reference/tolerance_and_solvers.md) for
# the `AutoTolerance` fields (`data_precision`, `safety`, `quantile`), the exact
# cutoff formula, and the gate constants.
