# # Analysis at Scale

# The [first tutorial](@ref "Getting Started") answered a single-outage question on a
# small network with a materialized [`LODF`](@ref). This one scales that idea up:
#
# > **I need to screen *every* single-line outage on a large network for overloads —
# > how do I do that without ever building the full matrix?**
#
# On a grid with thousands of branches a dense [`LODF`](@ref) is an `n × n` array of
# `Float64` — easily many gigabytes, most of which a screen never looks at, because it
# reads one outage's row at a time. PNM's **virtual** matrices are built for exactly
# this: they compute rows on demand and keep the recent ones in a bounded cache, so you
# pay for the rows you actually query and nothing more. The technique that makes this
# practical at scale is *managing that cache*, which is what this tutorial is about.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# !!! note
#
#     We use a small five-bus system so the tutorial runs quickly, but read it as if
#     the network were far larger — that is where the virtual approach earns its keep.
#     On a five-bus system a dense [`LODF`](@ref) is trivial; at thousands of branches
#     it is the difference between fitting in memory and not.

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Step 1 — Swap in a virtual matrix

# [`VirtualLODF`](@ref) builds and indexes **exactly** like the materialized
# [`LODF`](@ref) — same constructor call, same `[monitored, outaged]` indexing — but it
# never forms the whole matrix. Rows are computed the first time you ask for them and
# cached; the `max_cache_size` keyword caps that cache in **MiB**:

vlodf = PNM.VirtualLODF(sys; max_cache_size = 50)

# Nothing has been computed yet. The matrix is a promise: ask for a row and it appears.

# ## Step 2 — Run the outage screen

# Now the screen itself — for every branch, trip it and find the branch that inherits
# the largest share of its flow. This is the same loop you would run on a real grid;
# the only difference is that a `VirtualLODF` computes each outage's row on first
# access and serves it from cache if the screen revisits it:

arcs = vlodf.axes[1]
for outaged in arcs
    worst_arc, worst_val = outaged, 0.0
    for monitored in arcs
        monitored == outaged && continue
        v = vlodf[monitored, outaged]
        if abs(v) > abs(worst_val)
            worst_arc, worst_val = monitored, v
        end
    end
    println("outage $outaged → worst responder $worst_arc @ ", round(worst_val; digits = 3))
end

# Identical numbers to a dense [`LODF`](@ref) — several outages here fully redistribute
# onto a single partner line (a factor of `±1.0`) — but we only ever materialized the
# rows the screen touched. On a large network that is the whole game: the full matrix
# never has to exist.

# The same drop-in swap works for sensitivities via [`VirtualPTDF`](@ref):

vptdf = PNM.VirtualPTDF(sys; max_cache_size = 50)
vptdf[(1, 2), 3]

# ## Step 3 — Size the cache to your memory budget

# `max_cache_size` is the one knob that trades memory for recomputation. When the cache
# fills, the **least-recently-used** row is evicted; querying it again recomputes it.
# On a big grid you set this to whatever slice of RAM you can spare:
#
#   - **generous** cache → more rows stay resident, fewer recomputes, more memory held;
#   - **tight** cache → a hard memory ceiling, at the cost of recomputing evicted rows.
#
# A full screen that fits in the cache pays each row's compute exactly once. A screen
# larger than the cache still completes — the evicted rows are simply recomputed if
# revisited.

# ## Step 4 — Pin the rows you always touch

# Some rows are hot: a screen that re-runs at every operating point keeps hitting the
# same critical corridors. Pass those arcs as `persistent_arcs` to compute them eagerly
# at construction and **exempt them from eviction** — they never leave the cache no
# matter how much else churns through it:

vlodf_pinned = PNM.VirtualLODF(sys; max_cache_size = 50, persistent_arcs = [(1, 4), (2, 3)])

# [`VirtualPTDF`](@ref) takes the same keyword. [`VirtualMODF`](@ref) (post-contingency
# factors) accepts `max_cache_size`, applied per contingency, but not `persistent_arcs`.

# ## Step 5 — Reclaim the memory when you are done

# For [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref), empty the row cache in place to free
# its memory (pinned rows go too, with a warning):

empty!(vlodf.cache)

# [`VirtualMODF`](@ref) holds registered contingencies alongside its caches, so it
# exposes two dedicated functions:
#
#   - [`clear_caches!`](@ref) frees the Woodbury and row caches but **keeps** the
#     registered contingencies, so the matrix stays queryable;
#   - [`clear_all_caches!`](@ref) additionally drops the registrations, leaving the
#     matrix empty — rebuild a new [`VirtualMODF`](@ref) from the system to restore it.
#
# ```julia
# PNM.clear_caches!(vmodf)      # keeps registered contingencies
# PNM.clear_all_caches!(vmodf)  # drops everything, including registrations
# ```

# ## Where to go next
#
#   - [Virtual vs. materialized matrices](@ref) — when a virtual matrix is the right
#     call and how the LRU trade-off actually works.
#   - [How to Define and Apply Contingencies](@ref) — [`VirtualMODF`](@ref) for
#     multi-element post-contingency factors, the next step past single-line screening.
#   - [Getting Started](@ref) — the first tutorial, if you skipped straight here.
