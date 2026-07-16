# # How to Tune and Clear the Virtual-Matrix Cache

# Virtual matrices ([`VirtualPTDF`](@ref), [`VirtualLODF`](@ref),
# [`VirtualMODF`](@ref)) compute rows on demand and store them in an LRU row
# cache instead of materializing the whole matrix. This guide shows you how to
# size that cache, pin hot rows so they are never evicted, and free the memory
# when you are done.

# ## Prerequisites
#
#   - `PowerNetworkMatrices.jl` installed
#   - A power system model loaded (see [Getting Started](@ref))

using PowerNetworkMatrices
using PowerSystemCaseBuilder

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Size the cache

# `max_cache_size` caps the cache in **MiB** (default `MAX_CACHE_SIZE_MiB`). When
# the cache fills, the least-recently-used non-persistent row is evicted:

vptdf = PNM.VirtualPTDF(sys; max_cache_size = 50)

# ## Pin hot rows with `persistent_arcs`

# Pass `persistent_arcs` to compute a set of arcs eagerly at construction and
# keep their rows in the cache permanently (they are exempt from LRU eviction).
# Use this for arcs you query repeatedly:

vptdf = PNM.VirtualPTDF(sys; persistent_arcs = [(1, 2), (1, 4)])

# `VirtualLODF` accepts the same `max_cache_size` and `persistent_arcs` kwargs:

vlodf = PNM.VirtualLODF(sys; max_cache_size = 50, persistent_arcs = [(1, 2)])

# `VirtualMODF` accepts `max_cache_size` (applied per contingency) but not
# `persistent_arcs`:

# ```julia
# vmodf = PNM.VirtualMODF(sys; max_cache_size = 50)
# ```

# ## Read some rows

# Each miss computes and caches a row; subsequent reads of the same row are
# served from the cache:

vptdf[(1, 2), 3]
vlodf[(1, 4), (2, 3)];

# ## Clear the cache

# For `VirtualPTDF` / `VirtualLODF`, empty the row cache in place to reclaim
# memory (persistent rows are also removed, with a warning):

empty!(vptdf.cache)
empty!(vlodf.cache)

# `VirtualMODF` exposes dedicated clearing functions:
#
#   - [`clear_caches!`](@ref) frees the Woodbury and row caches but **keeps** the
#     registered contingencies, so the matrix can still be queried.
#   - [`clear_all_caches!`](@ref) additionally drops the contingency
#     registrations, leaving the matrix empty and unqueryable — rebuild a new
#     `VirtualMODF` from the system to restore it.

# ```julia
# PNM.clear_caches!(vmodf)      # keeps registered contingencies
# PNM.clear_all_caches!(vmodf)  # drops everything, including registrations
# ```

# ## Notes
#
#   - See the [virtual vs. materialized matrices](../explanation/virtual_vs_materialized.md)
#     explanation for when a virtual matrix is the right choice and how the LRU
#     trade-off works.
