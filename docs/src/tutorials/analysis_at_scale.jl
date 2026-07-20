#src EXECUTE = TRUE
# # Analysis at Scale

# The [first tutorial](@ref "Getting Started") screened a *single* outage on a small
# network with dense matrices. The everyday bulk-reliability job is bigger: the full
# **N-1 screen** — trip every line in turn, and for each trip check whether any surviving
# line is pushed past its rating. Pose it as one question:
#
# > *Across every single-line outage, which contingencies drive a surviving line closest
# > to — or past — its limit?*
#
# The analysis itself is ordinary DC N-1: base-case flows from a [`PTDF`](@ref), then
# post-contingency flows from an [`LODF`](@ref), compared against line ratings. What makes
# it a *scale* problem is the matrices. A dense [`PTDF`](@ref) or [`LODF`](@ref) is an
# `O(N²)` array of `Float64` — on a real interconnection that is tens of gigabytes, and a
# screen reads each row exactly once. So we never materialize them: we run the identical
# study on **virtual** matrices that compute rows on demand, and add a few memory
# disciplines around them.
#
# !!! note "About the example system"
#     We use RTS-GMLC (73 buses) so the whole tutorial runs in seconds. It is small enough
#     that the dense matrices would fit fine — the point is the *method*. Every line of
#     code below is what you would run unchanged on a 20,000-bus grid, where the dense
#     forms never could.

import PowerNetworkMatrices as PNM
import PowerSystems as PSY
import PowerSystemCaseBuilder as PSB
import Logging
using LinearAlgebra: dot
using DataFrames

## Silence the informational build logs so the tutorial output stays readable.
Logging.disable_logging(Logging.Info)

sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys");

# ## Step 1 — Set up the study: base-case flows and limits

# Post-contingency flow needs two ordinary ingredients: the flow each line carries *now*,
# and the limit it must stay under.

# **Base-case flows.** Under the DC approximation a line's flow is its [`PTDF`](@ref) row
# dotted with the vector of net bus injections (generation minus load). We build the
# injection vector straight from the system — accumulating per bus, because a bus can host
# several generators and loads — ordered to match the matrix's bus axis:

vptdf = PNM.VirtualPTDF(sys)
bus_lookup = PNM.get_bus_lookup(vptdf)

injection = zeros(Float64, length(bus_lookup))
for gen in PSY.get_components(
    d -> !isa(d, Union{PSY.ElectricLoad, PSY.SynchronousCondenser}),
    PSY.StaticInjection, sys)
    PSY.get_available(gen) || continue
    injection[bus_lookup[PSY.get_number(PSY.get_bus(gen))]] += PSY.get_active_power(gen)
end
for load in PSY.get_components(d -> !isa(d, PSY.FixedAdmittance), PSY.ElectricLoad, sys)
    PSY.get_available(load) || continue
    injection[bus_lookup[PSY.get_number(PSY.get_bus(load))]] -= PSY.get_active_power(load)
end

# The system defaults to its per-unit *system base*, so these injections — and the ratings
# we read below — are already on the same `100`-MVA base and directly comparable. The
# injections need not sum to zero; the reference bus balances the difference, exactly as
# the [`PTDF`](@ref) assumes.

# Now the base flow on every line. Each `vptdf[arc, :]` computes that line's [`PTDF`](@ref)
# row on first access and caches it — one pass touches each row once:

arcs = vptdf.axes[1]
base_flow = Dict(arc => dot(vptdf[arc, :], injection) for arc in arcs);

# !!! note
#     Building *every* base flow this way touches the whole [`PTDF`](@ref), one row at a
#     time. When only the base flows are needed, the sparse `ABA`/`BA` DC solve
#     ([`ABA_Matrix`](@ref), [`BA_Matrix`](@ref)) gets them in a single factorization; the
#     row-at-a-time route here is what the N-1 screen below needs anyway.

# **Line limits.** Read each branch's rating and key it by arc. Parallel branches share an
# arc, so we sum their ratings into the combined corridor limit:

line_rating = Dict{Tuple{Int, Int}, Float64}()
for branch in PSY.get_components(PSY.ACTransmission, sys)
    arc = PSY.get_arc(branch)
    key = (PSY.get_number(PSY.get_from(arc)), PSY.get_number(PSY.get_to(arc)))
    line_rating[key] = get(line_rating, key, 0.0) + PSY.get_rating(branch)
end

# ## Step 2 — Screen every contingency on a virtual LODF

# The [`LODF`](@ref) gives the redistribution: when `outaged` trips, line `monitored`
# picks up `LODF[monitored, outaged]` of the outaged line's pre-trip flow. So the
# post-contingency flow on `monitored` is `base_flow[monitored] +
# LODF[monitored, outaged] · base_flow[outaged]`, and its **loading** is that over its
# rating.

# [`VirtualLODF`](@ref) builds and indexes exactly like the dense [`LODF`](@ref) — same
# constructor, same `[monitored, outaged]` indexing — but never forms the whole matrix.
# `max_cache_size` caps the row cache in MiB:

vlodf = PNM.VirtualLODF(sys; max_cache_size = 100)

# A **row** — one monitored line's factors against every outage — is the unit the cache
# stores, so we sweep row by row: compute each monitored line's row once, then score it
# against every outage. We keep, for each outage, the single worst-loaded survivor:

lines = vlodf.axes[1]
outage_col = vlodf.lookup[2]

worst = Dict{Tuple{Int, Int}, @NamedTuple{line::Tuple{Int, Int}, loading::Float64}}()
for monitored in lines
    factors = vlodf[monitored, :]
    rating = line_rating[monitored]
    f_m = base_flow[monitored]
    for outaged in lines
        outaged == monitored && continue
        post = f_m + factors[outage_col[outaged]] * base_flow[outaged]
        loading = abs(post) / rating
        if !haskey(worst, outaged) || loading > worst[outaged].loading
            worst[outaged] = (line = monitored, loading = loading)
        end
    end
end

# Rank the outages by the worst loading they cause — the top of this list is the operator's
# work queue:

screen = sort(
    DataFrame(;
        outaged_line = [o for o in keys(worst)],
        worst_monitored = [w.line for w in values(worst)],
        loading = [round(w.loading; digits = 3) for w in values(worst)],
    ),
    :loading; rev = true,
)

# There is the answer. The day-ahead schedule is **not** N-1 secure — most single-line
# outages here push some surviving line over its rating:

count(>(1.0), screen.loading)

# The worst contingency drives a line to roughly `1.3×` its limit. Notice the top pair:
# tripping `(107, 108)` overloads `(107, 203)` and vice versa — they are a tightly coupled
# corridor, each inheriting essentially the *entire* flow of the other. The numbers are
# identical to a dense [`LODF`](@ref); the difference is that we never held one.

# **The memory story.** The sweep visited every row, so all of them are now cached:

length(vlodf.cache)

# On RTS-GMLC that is ~108 short rows — a rounding error in memory, which is why nothing
# was evicted. At real scale it is the opposite: the rows do not all fit, and
# `max_cache_size` is the hard ceiling. Once the cache is full the **least-recently-used**
# row is dropped; a full screen still completes, trading a bounded memory footprint for
# recomputing an evicted row if the screen returns to it. That trade is what lets an N-1
# screen run on a grid whose dense [`LODF`](@ref) would never fit.

# ## Step 3 — Reuse rows across operating points

# This screen is not run once. It reruns every operating point — but the [`LODF`](@ref) is
# a property of the **topology**, not the dispatch: the factors do not change from hour to
# hour, only the base flows they multiply do. So a study that reruns the screen wants its
# rows to *stay* resident rather than be recomputed each pass.

# In practice you monitor a defined set of facilities every cycle — here, the inter-area
# tie corridors. Declare them up front as `persistent_arcs`: those rows are held in the
# cache and **exempt from eviction**, so no amount of churn from the rest of a screen can
# force them to be re-solved:

tie_lines = [(107, 203), (113, 215), (123, 217)]
vlodf_watch = PNM.VirtualLODF(sys; persistent_arcs = tie_lines, max_cache_size = 100)

# Across repeated operating points you now recompute only the base flows (cheap) and read
# the tie-line factors straight from the pinned rows. [`VirtualPTDF`](@ref) takes the same
# keyword. Pinned rows still count against the budget, so the constructor errors if the
# pinned set alone would exceed `max_cache_size`.

# ## Step 4 — Shrink the network to scale further

# A second lever is to make the matrices smaller before you build them. Real grids carry
# buses that add nothing to a redistribution screen — dead-end (radial) buses and
# pass-through (degree-two) buses. Supplying `network_reductions` removes them (see
# [Getting Started](@ref) for the guarantee that surviving sensitivities are unchanged):

reductions = PNM.NetworkReduction[PNM.RadialReduction(), PNM.DegreeTwoReduction()]
vlodf_reduced = PNM.VirtualLODF(sys; network_reductions = reductions)
length(vlodf.axes[1]), length(vlodf_reduced.axes[1])

# RTS-GMLC is compact and meshed, so only a handful of arcs drop here; on a full
# interconnection with long sub-transmission tails, reduction routinely removes a large
# fraction of the network. Either way the screen's answer is untouched — the worst
# contingency's factor is identical before and after:

vlodf[(107, 203), (107, 108)], vlodf_reduced[(107, 203), (107, 108)]

# Both essentially `1.0`: when `(107, 108)` trips, `(107, 203)` still inherits its entire
# flow. (One caveat when screening a *reduced* network against ratings: a degree-two merge
# fuses several branches into one equivalent arc whose limit is an aggregate — reach for
# [`get_single_element_contingency_rating`](@ref) and the related aggregated-rating
# accessors rather than a single branch's `PSY.get_rating`.)

# ## Step 5 — Reclaim the memory when you are done

# For [`VirtualPTDF`](@ref) / [`VirtualLODF`](@ref), empty the row cache in place to free
# it (pinned rows go too, with a warning):

empty!(vlodf.cache)

# ## Where to go next
#
#   - [Reproduce industry DFAX values](@ref) — richer distribution-factor reports
#     (transfer, flowgate, N-k) on this same system.
#   - [How to Define and Apply Contingencies](@ref) — [`VirtualMODF`](@ref) for
#     multi-element post-contingency factors, the next step past single-line screening.
#   - [Computational Considerations](@ref) — the sparsity and complexity behind why the
#     dense matrices are the ones you avoid at scale.
