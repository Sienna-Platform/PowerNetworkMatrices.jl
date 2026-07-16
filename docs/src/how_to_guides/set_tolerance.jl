# # How to Set the Sparsification Tolerance

# This guide shows you how to control how aggressively [`PTDF`](@ref) /
# [`LODF`](@ref) rows are sparsified, using the `tol` keyword. For the full
# settings reference, see
# [Tolerance and solver settings](../reference/tolerance_and_solvers.md).

using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## The `tol` keyword

# `tol` is `Union{Float64, AutoTolerance}` and defaults to
# `DEFAULT_AUTO_TOLERANCE` — an automatic, condition-aware rule, **not** a plain
# float. The two forms behave differently:
#
#   - a `Float64` applies a fixed **absolute** cutoff: entries with `|x| < tol`
#     are dropped;
#   - an [`AutoTolerance`](@ref) applies a **relative per-row** cutoff derived
#     from the branch-data precision.

# ## Fixed absolute tolerance

# Pass a `Float64` for an exact, size-independent cutoff:

ptdf = PNM.PTDF(sys; tol = 1e-5);

# ## Automatic tolerance

# The default is equivalent to constructing an [`AutoTolerance`](@ref) with its
# defaults:

ptdf = PNM.PTDF(sys; tol = PNM.AutoTolerance());

# `AutoTolerance` has three keyword fields:

PNM.AutoTolerance(; data_precision = :auto, safety = 1.0, quantile = 0.5);

#   - `data_precision`: the relative precision `δ` of the branch parameters.
#     `:auto` (default) discovers it from the branch data; a `Float64` sets it
#     explicitly (e.g. `1e-3` for reactances good to 0.1%).
#   - `safety`: an aggressiveness multiplier on `δ`; `> 1` sparsifies more,
#     `< 1` less.
#   - `quantile`: only used when `data_precision = :auto` — which quantile of the
#     per-branch significant-figure counts to adopt.

# Sparsify more aggressively while still discovering the precision:

ptdf = PNM.PTDF(sys; tol = PNM.AutoTolerance(; safety = 5.0));

# Or pin the precision explicitly and skip discovery:

ptdf = PNM.PTDF(sys; tol = PNM.AutoTolerance(; data_precision = 1e-3));

# ## Discovering the data precision directly

# The discovery step is available on its own via
# [`discover_data_precision`](@ref), which takes branch **susceptances** and
# returns the relative precision `δ`:

susceptances = [100.0, 25.0, 33.33333]
PNM.discover_data_precision(susceptances)

# ## Where each cutoff applies

# `AutoTolerance` is a **no-op on small systems** (below the bus-count gate), so
# every small test case stays exact; its relative drop is reserved for large
# virtual matrices. A `Float64` `tol` is honored at any size. The dense
# `PTDF`/`LODF` path never sparsifies under `AutoTolerance` (it preserves the
# dense `Matrix{Float64}`), but a `Float64` `tol` still applies there. See
# [Tolerance and solver settings](../reference/tolerance_and_solvers.md).
