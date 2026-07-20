# # Analysis at Scale

# The [first tutorial](@ref "Getting Started") screened one outage on a small network.
# Here we take on a broader reliability question:
#
# > Which single-line outages are the most dangerous — which trip dumps the largest
# > share of its flow onto a single surviving line?
#
# The analysis is ordinary [`LODF`](@ref) work. What is new is the setting: this grid is
# large. A dense [`LODF`](@ref) is an `n × n` array of `Float64` — easily many gigabytes,
# most of which a screen never reads. So we run the whole study on a **virtual** matrix
# that computes rows on demand, and reach for a short sequence of cache techniques — a
# targeted first pass, a bounded full screen, pinned hot rows — to get the answer without
# ever holding the full matrix or recomputing more than we must.

import PowerNetworkMatrices as PNM
import PowerSystemCaseBuilder as PSB

# !!! note
#
#     We use a small five-bus system so the tutorial runs quickly, but read it as if the
#     network were far larger — that is where the virtual approach earns its keep. On a
#     five-bus system a dense [`LODF`](@ref) is trivial; at thousands of branches it is
#     the difference between fitting in memory and not.

sys = PSB.build_system(PSB.PSITestSystems, "c_sys5");

# ## Step 1 — Build the screen on a virtual matrix

# [`VirtualLODF`](@ref) builds and indexes **exactly** like the materialized
# [`LODF`](@ref) — same constructor, same `[monitored, outaged]` indexing — but it never
# forms the whole matrix. A row (one *monitored* line's factors against every outage) is
# computed the first time you ask for it and cached; `max_cache_size` caps that cache in
# **MiB**:

vlodf = PNM.VirtualLODF(sys; max_cache_size = 50)

# Nothing has been computed yet. The matrix is a promise: ask for a row and it appears.

arcs = vlodf.axes[1]

# ## Step 2 — Screen the lines that matter first

# We rarely need every line at once. Start from a **watchlist** of the lines we most want
# to protect, and ask, for each, which outage threatens it worst. Every access
# `vlodf[monitored, outaged]` computes and caches the *monitored* row, so a watchlist of
# two lines touches exactly two rows:

watchlist = [(1, 4), (2, 3)]
for monitored in watchlist
    worst_outage, worst_val = monitored, 0.0
    for outaged in arcs
        outaged == monitored && continue
        v = vlodf[monitored, outaged]
        if abs(v) > abs(worst_val)
            worst_outage, worst_val = outaged, v
        end
    end
    println("line $monitored most exposed to outage of $worst_outage @ ",
        round(worst_val; digits = 3))
end

# That is the first payoff of going virtual: only the two watchlist rows were ever
# computed. The rest of the [`LODF`](@ref) — every other monitored line — never existed.
# On a grid with thousands of branches, a targeted question pays for a handful of rows
# instead of the whole matrix.

# ## Step 3 — Widen to the full N-1 screen

# Now the network-wide version of the question: across *every* outage, which surviving
# line inherits the most flow? This time we monitor every line, so the screen touches
# every row:

for outaged in arcs
    worst_line, worst_val = outaged, 0.0
    for monitored in arcs
        monitored == outaged && continue
        v = vlodf[monitored, outaged]
        if abs(v) > abs(worst_val)
            worst_line, worst_val = monitored, v
        end
    end
    println("outage $outaged → worst-hit line $worst_line @ ",
        round(worst_val; digits = 3))
end

# There is the answer: several outages here fully redistribute onto a single partner line
# (a factor of `±1.0`) — those are the dangerous ones. The numbers are identical to a
# dense [`LODF`](@ref), but the memory story is different. Because `max_cache_size` caps
# the cache, once it fills the **least-recently-used** row is evicted; we visit every row
# yet memory never climbs past the ceiling. The trade is explicit — a hard memory bound
# in exchange for recomputing an evicted row if the screen comes back to it — and it is
# what lets a full N-1 screen run on a grid whose dense [`LODF`](@ref) would never fit.

# ## Step 4 — Pin the rows the study reuses

# This screen is not run once. At each operating point the factors are reused — only the
# base-case flows they multiply change — so every pass re-reads the same watchlist rows.
# Left to the LRU policy, a big full screen could evict them between passes and force a
# recompute each time. Pass the watchlist as `persistent_arcs` to compute those rows
# eagerly at construction and **exempt them from eviction** — they stay resident no
# matter how much else churns through the cache:

vlodf_pinned = PNM.VirtualLODF(sys; max_cache_size = 50, persistent_arcs = watchlist)

# Now the watchlist rows survive every full screen, so repeated studies never pay to
# rebuild them. [`VirtualPTDF`](@ref) takes the same keyword.

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
#     call and how the LRU trade-off actually works. The same drop-in swap applies to
#     sensitivities via [`VirtualPTDF`](@ref).
#   - [How to Define and Apply Contingencies](@ref) — [`VirtualMODF`](@ref) for
#     multi-element post-contingency factors, the next step past single-line screening.
#   - [Getting Started](@ref) — the first tutorial, if you skipped straight here.
