# # Analysis at Scale

# A dense [`PTDF`](@ref) or [`LODF`](@ref) is an `O(N²)` array of `Float64`, which on a real interconnection is tens of gigabytes, and a screen reads each row exactly once. This tutorial makes use of **virtual** matrices that compute rows on demand to answer the following question:
# > Across every single-line outage, which contingencies drive a surviving line closest to — or past — its limit?
#
# !!! note
#     We use a small network (73 buses) for demonstration purposes.

using PowerNetworkMatrices
import PowerSystems
import PowerSystemCaseBuilder
using LinearAlgebra: dot
using DataFrames

# Load an example test system with [`PowerSystemCaseBuilder.build_system`](@extref):

sys = PowerSystemCaseBuilder.build_system(
    PowerSystemCaseBuilder.PSISystems,
    "RTS_GMLC_DA_sys",
);

# ## Step 1 — Set up the study: base-case flows and limits

# A post-contingency flow requires the flow each line carries at present and the limit it must stay under.

# **Base-case flows.** Under the DC approximation a line's flow is its [`PTDF`](@ref) row dotted with the vector of net bus injections, that is, generation minus load. [`VirtualPTDF`](@ref) supplies those rows one at a time instead of materializing the whole matrix.

vptdf = VirtualPTDF(sys)

# The injection vector has to line up with the matrix's bus dimension, so [`get_bus_lookup`](@ref) is what maps a bus number to its position in that vector:

bus_lookup = get_bus_lookup(vptdf)

# We build the injection vector from the system, accumulating per bus because a bus can host several generators and loads, ordered to match the matrix's bus axis.

injection = zeros(Float64, length(bus_lookup))
for gen in PowerSystems.get_components(
    d -> !isa(
        d,
        Union{PowerSystems.ElectricLoad, PowerSystems.SynchronousCondenser},
    ),
    PowerSystems.StaticInjection,
    sys,
)
    PowerSystems.get_available(gen) || continue
    bus_number = PowerSystems.get_number(PowerSystems.get_bus(gen))
    injection[bus_lookup[bus_number]] += PowerSystems.get_active_power(gen)
end
for load in PowerSystems.get_components(
    d -> !isa(d, PowerSystems.FixedAdmittance),
    PowerSystems.ElectricLoad,
    sys,
)
    PowerSystems.get_available(load) || continue
    bus_number = PowerSystems.get_number(PowerSystems.get_bus(load))
    injection[bus_lookup[bus_number]] -= PowerSystems.get_active_power(load)
end
injection

# The system defaults to its per-unit *system base*, so these injections and the ratings read below are already on the same `100`-MVA base and are directly comparable. The injections do not need to sum to zero because the reference bus balances the difference.

# The base flow on every line follows. [`get_arc_axis`](@ref) enumerates the arcs to sweep, and each `vptdf[arc, :]` computes that line's [`PTDF`](@ref) row on first access and caches it, so one pass touches each row once.

arcs = get_arc_axis(vptdf)
base_flow = Dict(arc => dot(vptdf[arc, :], injection) for arc in arcs)

# !!! note
#     Building *every* base flow this way touches the whole [`PTDF`](@ref), one row at a
#     time. When only the base flows are needed, the sparse `ABA`/`BA` DC solve
#     ([`ABA_Matrix`](@ref), [`BA_Matrix`](@ref)) obtains them in a single factorization;
#     the row-at-a-time route here is what the N-1 screen below requires anyway.

# **Line limits.** Each branch's rating is read and keyed by arc. Parallel branches share an arc, so their ratings are summed into the combined corridor limit.

line_rating = Dict{Tuple{Int, Int}, Float64}()
for branch in PowerSystems.get_components(PowerSystems.ACTransmission, sys)
    arc = PowerSystems.get_arc(branch)
    key = (
        PowerSystems.get_number(PowerSystems.get_from(arc)),
        PowerSystems.get_number(PowerSystems.get_to(arc)),
    )
    line_rating[key] = get(line_rating, key, 0.0) + PowerSystems.get_rating(branch)
end

# ## Step 2 — Screen every contingency on a VirtualLODF

# The [`LODF`](@ref) gives the redistribution: when `outaged` trips, line `monitored` picks up `LODF[monitored, outaged]` of the outaged line's pre-trip flow. The post-contingency flow on `monitored` is therefore `base_flow[monitored] + LODF[monitored, outaged] · base_flow[outaged]`, and its **loading** is that quantity over its rating.

# [`VirtualLODF`](@ref) is constructed and indexed exactly like the dense [`LODF`](@ref), with the same constructor and the same `[monitored, outaged]` indexing, but never forms the whole matrix. `max_cache_size` caps the row cache in MiB.

vlodf = VirtualLODF(sys; max_cache_size = 100)

# A **row**, meaning one monitored line's factors against every outage, is the unit the cache stores, so the sweep proceeds row by row: compute each monitored line's row once, then score it against every outage. Both dimensions of a [`LODF`](@ref) are arcs, so [`get_arc_axis`](@ref) lists the lines to sweep and [`get_arc_lookup`](@ref) turns an outaged arc into its position within a row. For each outage we keep the single worst-loaded survivor.

lines = get_arc_axis(vlodf)
outage_col = get_arc_lookup(vlodf)

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

# Ranking the outages by the worst loading they cause gives the screen's result.

screen = sort(
    DataFrame(;
        outaged_line = [o for o in keys(worst)],
        worst_monitored = [w.line for w in values(worst)],
        loading = [round(w.loading; digits = 3) for w in values(worst)],
    ),
    :loading; rev = true,
)

# The day-ahead schedule is **not** N-1 secure: most single-line outages here push some surviving line over its rating.

count(>(1.0), screen.loading)

# The worst contingency drives a line to roughly `1.3×` its limit. The top pair is reciprocal — tripping `(107, 108)` overloads `(107, 203)` and vice versa — because they form a tightly coupled corridor in which each inherits essentially the entire flow of the other. The numbers are identical to those of a dense [`LODF`](@ref); the difference is that no dense matrix was ever stored.

# The sweep visited every row, so all of them are now cached. For a virtual matrix, [`get_lodf_data`](@ref) returns exactly the rows the cache is holding, keyed by row index:

length(get_lodf_data(vlodf))

# On RTS-GMLC that is approximately 108 short rows, a negligible memory footprint, which is why nothing was evicted. At realistic scale the situation is the opposite: the rows do not all fit, and `max_cache_size` is a hard ceiling. Once the cache is full the **least-recently-used** row is dropped. A full screen still completes, trading a bounded memory footprint for the recomputation of an evicted row should the screen return to it. That trade is what allows an N-1 screen to run on a grid whose dense [`LODF`](@ref) would not fit.

# ## Step 3 — Reuse rows across operating points

# This screen is not run once. It reruns at every operating point, but the [`LODF`](@ref) is a property of the **topology** rather than the dispatch: the factors do not change from hour to hour, only the base flows they multiply. A study that reruns the screen therefore wants its rows to remain resident rather than be recomputed on each pass.

# In practice a defined set of facilities is monitored every cycle, here the inter-area tie corridors. Declaring them up front as the [`VirtualLODF`](@ref) constructor's `persistent_arcs` holds those rows in the cache and makes them **exempt from eviction**, so no amount of churn from the rest of a screen can force them to be re-solved.

tie_lines = [(107, 203), (113, 215), (123, 217)]
vlodf_watch = VirtualLODF(sys; persistent_arcs = tie_lines, max_cache_size = 100)

# Across repeated operating points only the base flows are recomputed, which is cheap, and the tie-line factors are read straight from the pinned rows. [`VirtualPTDF`](@ref) takes the same keyword. Pinned rows still count against the budget, so the constructor errors if the pinned set alone would exceed `max_cache_size`.

# ## Step 4 — Reclaim the memory

# For [`VirtualPTDF`](@ref) and [`VirtualLODF`](@ref), the row cache is emptied in place to free it. Pinned rows are removed too, with a warning.

empty!(vlodf.cache)

# ## Where to go next
#
#   - [Reproduce industry DFAX values](@ref) — richer distribution-factor reports
#     (transfer, flowgate, N-k) on this same system.
#   - [How to Define and Apply Contingencies](@ref) — [`VirtualMODF`](@ref) for
#     multi-element post-contingency factors, the next step past single-line screening.
#   - [Computational Considerations](@ref) — the sparsity and complexity behind why the
#     dense matrices are the ones to avoid at scale.
