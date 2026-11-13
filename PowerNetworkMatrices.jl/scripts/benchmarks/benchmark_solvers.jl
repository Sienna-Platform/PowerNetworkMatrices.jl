# benchmark_solvers.jl
#
# Purpose:
#   Benchmark VirtualPTDF and VirtualMODF construction and per-row query
#   performance for KLU and AppleAccelerateLU solvers on the 10k-bus
#   matpower_ACTIVSg10k_sys case.
#
#   Line outages are injected as PSY.FixedForcedOutage supplemental attributes
#   on ACTransmission branches above a configurable kV threshold, making
#   VirtualMODF contingency-row performance measurable.
#
#   Every measured quantity is sampled over PASSES (default 10) passes and
#   reported as `median [min–max]` so the run-to-run range is visible.
#
# Run command (from repo root):
#   julia --project=test scripts/benchmarks/benchmark_solvers.jl
#
# Runtime note:
#   Each row-batch pass builds a *fresh* (cold-cache) Virtual* object so the
#   row-solve cost is measured without cache hits. With PASSES=10 that means
#   ~11 cold builds per solver per path; on the 10k case this is minutes, not
#   seconds. Lower PASSES / N_ROWS to trade range fidelity for speed.
#
# Note on Project.toml:
#   This script activates the repo `test` project rather than providing a
#   standalone scripts/benchmarks/Project.toml because PowerSystemCaseBuilder
#   is a test-only dependency. Resolving PSB from the registry in a standalone
#   project is unreliable without a dedicated manifest; the test environment
#   already has PSB deved alongside PNM and is guaranteed to be consistent.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "..", "test"))

using PowerNetworkMatrices
using PowerSystemCaseBuilder
import PowerSystems as PSY
import InfrastructureSystems as IS
import SparseArrays, LinearAlgebra, Random
using Printf

const PNM = PowerNetworkMatrices
const PSB = PowerSystemCaseBuilder

# ─────────────────────────────────────────────────────────────────────────────
# Top-level configuration constants
# ─────────────────────────────────────────────────────────────────────────────

const SYSTEM_NAME = "matpower_ACTIVSg10k_sys"
const KV_THRESHOLD = 230.0          # inject outages on arcs above this kV
const N_ROWS = 200            # arc-rows to time per batch
const SOLVERS = ["KLU", "AppleAccelerateLU"]
const PASSES = 10             # timed passes per measurement (range)
const WARMUP = 1              # untimed warmup passes (excluded)

# ─────────────────────────────────────────────────────────────────────────────
# Timing helpers
# ─────────────────────────────────────────────────────────────────────────────

Base.@noinline _no_inline(x) = x

"Min/median/max of a sample vector of nanosecond timings."
struct TimeStats
    tmin::Float64
    tmed::Float64
    tmax::Float64
end

const NA_STATS = TimeStats(NaN, NaN, NaN)

isna(s::TimeStats) = isnan(s.tmed)

"""
    collect_stats(sample; passes=PASSES, warmup=WARMUP) -> TimeStats

Call `sample()` `warmup + passes` times. `sample()` must return its own
elapsed time in nanoseconds (Float64). The first `warmup` results are
discarded; the remaining `passes` are reduced to (min, median, max).
"""
function collect_stats(sample::F; passes::Int = PASSES, warmup::Int = WARMUP) where {F}
    for _ in 1:warmup
        _no_inline(sample())
    end
    times = Vector{Float64}(undef, passes)
    for i in 1:passes
        times[i] = Float64(_no_inline(sample()))
    end
    sort!(times)
    tmin = times[1]
    tmax = times[end]
    tmed = times[(passes + 1) ÷ 2]
    return TimeStats(tmin, tmed, tmax)
end

"Time a single call to `f()` (whole call, GC'd first); returns elapsed ns."
function time_call(f::F) where {F}
    GC.gc()
    t = time_ns()
    _no_inline(f())
    return Float64(time_ns() - t)
end

function fmt_time(t_ns::Float64)
    if t_ns < 1e3
        return @sprintf("%.1f ns", t_ns)
    elseif t_ns < 1e6
        return @sprintf("%.1f μs", t_ns / 1e3)
    elseif t_ns < 1e9
        return @sprintf("%.1f ms", t_ns / 1e6)
    else
        return @sprintf("%.2f s", t_ns / 1e9)
    end
end

"Markdown cell: `median [min–max]`, or `n/a`."
function cell_str(s::TimeStats)
    if isna(s)
        return "n/a"
    end
    return "$(fmt_time(s.tmed)) [$(fmt_time(s.tmin))–$(fmt_time(s.tmax))]"
end

"Ratio of medians, e.g. KLU/AA."
function ratio_str(a::TimeStats, b::TimeStats)
    if isna(a) || isna(b)
        return "n/a"
    end
    return @sprintf("%.2fx", a.tmed / b.tmed)
end

# ─────────────────────────────────────────────────────────────────────────────
# Spread-out index selection
# ─────────────────────────────────────────────────────────────────────────────

"""
    spread_keys(axis, n) -> Vector

Return `n` evenly-spread elements from `axis` (or all elements if fewer than `n`).
"""
function spread_keys(axis::Vector, n::Int)
    total = length(axis)
    n = min(n, total)
    indices = unique(round.(Int, range(1, total; length = n)))
    return [axis[i] for i in indices]
end

# ─────────────────────────────────────────────────────────────────────────────
# Outage injection
# ─────────────────────────────────────────────────────────────────────────────

"""
    inject_line_outages!(sys; kv_threshold=KV_THRESHOLD) -> Int

Attach a `PSY.FixedForcedOutage(outage_status=1.0)` supplemental attribute to
every `PSY.Line` whose highest terminal voltage exceeds `kv_threshold` kV.

Only `PSY.Line` components are considered — transformers and
phase-shifting transformers are intentionally excluded (contingencies are
registered for lines only).

Returns the number of outages injected. Prints a summary with line count,
injected count, and three example (name, kV) pairs.
"""
function inject_line_outages!(sys::PSY.System; kv_threshold::Float64 = KV_THRESHOLD)::Int
    all_branches = collect(PSY.get_components(PSY.Line, sys))
    n_total = length(all_branches)
    injected = 0
    examples = Tuple{String, Float64}[]

    for branch in all_branches
        arc = PSY.get_arc(branch)
        from_kv = PSY.get_base_voltage(PSY.get_from(arc))
        to_kv = PSY.get_base_voltage(PSY.get_to(arc))
        v = max(from_kv, to_kv)
        if v > kv_threshold
            outage = PSY.FixedForcedOutage(; outage_status = 1.0)
            PSY.add_supplemental_attribute!(sys, branch, outage)
            injected += 1
            if length(examples) < 3
                push!(examples, (PSY.get_name(branch), v))
            end
        end
    end

    println(
        "inject_line_outages!: $n_total total PSY.Line components; " *
        "injected $injected outages (kv_threshold = $kv_threshold kV)",
    )
    for (name, kv) in examples
        println("  example: \"$name\" at $(kv) kV")
    end
    return injected
end

# ─────────────────────────────────────────────────────────────────────────────
# Load system and inject outages
# ─────────────────────────────────────────────────────────────────────────────

println("Loading $SYSTEM_NAME …")
sys = PSB.build_system(PSB.MatpowerTestSystems, SYSTEM_NAME)
n_buses = length(collect(PSY.get_components(PSY.ACBus, sys)))
println("System loaded: $n_buses buses")

println("\nInjecting line outages (kv_threshold = $KV_THRESHOLD kV) …")
n_outages = inject_line_outages!(sys; kv_threshold = KV_THRESHOLD)
println("Total outages injected: $n_outages\n")

# ─────────────────────────────────────────────────────────────────────────────
# Result tables
# ─────────────────────────────────────────────────────────────────────────────

const PATHS = [
    "VirtualPTDF build",
    "VirtualPTDF $(N_ROWS)-row batch",
    "VirtualMODF build",
    "VirtualMODF $(N_ROWS)-row batch",
]

results = Dict{String, Dict{String, TimeStats}}()
for p in PATHS
    results[p] = Dict{String, TimeStats}(s => NA_STATS for s in SOLVERS)
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-solver benchmark loop
#
# Build paths : time the whole constructor call, PASSES times.
# Row batches : each pass builds a FRESH (cold-cache) object UNTIMED, then
#               times only the row loop — isolating row-solve cost with no
#               build-time subtraction and no cache hits across passes.
# ─────────────────────────────────────────────────────────────────────────────

for solver in SOLVERS
    println("\n" * "="^60)
    println("Solver: $solver  ($PASSES passes, $WARMUP warmup)")
    println("="^60)

    # ------------------------------------------------------------------
    # VirtualPTDF: build time
    # ------------------------------------------------------------------
    vptdf_ref = nothing
    try
        println("  [VirtualPTDF] timed build …")
        vptdf_ref = PNM.VirtualPTDF(sys; linear_solver = solver)
        s = collect_stats(
            () -> time_call(
                () -> PNM.VirtualPTDF(sys; linear_solver = solver),
            ),
        )
        results["VirtualPTDF build"][solver] = s
        println("  VirtualPTDF build: $(cell_str(s))")
    catch e
        println("  VirtualPTDF build FAILED: $e")
    end

    # ------------------------------------------------------------------
    # VirtualPTDF: N_ROWS-row batch (fresh cold-cache object per pass)
    # ------------------------------------------------------------------
    if vptdf_ref !== nothing
        try
            println("  [VirtualPTDF] $N_ROWS-row batch …")
            arc_keys = spread_keys(PNM.get_arc_axis(vptdf_ref), N_ROWS)
            sample = function ()
                v = PNM.VirtualPTDF(sys; linear_solver = solver)
                GC.gc()
                t = time_ns()
                for arc in arc_keys
                    _ = v[arc, :]
                end
                return Float64(time_ns() - t)
            end
            s = collect_stats(sample)
            results["VirtualPTDF $(N_ROWS)-row batch"][solver] = s
            println("  VirtualPTDF $(N_ROWS)-row batch (rows only): $(cell_str(s))")
        catch e
            println("  VirtualPTDF row batch FAILED: $e")
        end
    end

    # ------------------------------------------------------------------
    # VirtualMODF: build time
    # ------------------------------------------------------------------
    vmodf_ref = nothing
    try
        println("  [VirtualMODF] timed build …")
        vmodf_ref = PNM.VirtualMODF(sys; linear_solver = solver)
        n_ctg = length(PNM.get_registered_contingencies(vmodf_ref))
        println("  VirtualMODF: $n_ctg contingencies registered")
        s = collect_stats(
            () -> time_call(
                () -> PNM.VirtualMODF(sys; linear_solver = solver),
            ),
        )
        results["VirtualMODF build"][solver] = s
        println("  VirtualMODF build: $(cell_str(s))")
    catch e
        println("  VirtualMODF build FAILED: $e")
    end

    # ------------------------------------------------------------------
    # VirtualMODF: N_ROWS-row batch for one contingency
    #
    # Row access: vmodf[arc_idx::Int, contingency::ContingencySpec]
    # where arc_idx = PNM.get_arc_lookup(vmodf)[arc_tuple]
    # and contingency is taken from PNM.get_registered_contingencies(vmodf).
    # ------------------------------------------------------------------
    if vmodf_ref !== nothing
        try
            ctg_dict = PNM.get_registered_contingencies(vmodf_ref)
            if isempty(ctg_dict)
                println("  VirtualMODF: no contingencies registered — skipping row batch.")
                println("  (Re-run inject_line_outages! with a lower kv_threshold.)")
            else
                println("  [VirtualMODF] $N_ROWS-row batch …")
                arc_keys = spread_keys(PNM.get_arc_axis(vmodf_ref), N_ROWS)
                arc_lookup_ref = PNM.get_arc_lookup(vmodf_ref)
                arc_indices = [arc_lookup_ref[arc] for arc in arc_keys]
                sample = function ()
                    v = PNM.VirtualMODF(sys; linear_solver = solver)
                    ctg = first(values(PNM.get_registered_contingencies(v)))
                    GC.gc()
                    t = time_ns()
                    for arc_idx in arc_indices
                        _ = v[arc_idx, ctg]
                    end
                    return Float64(time_ns() - t)
                end
                s = collect_stats(sample)
                results["VirtualMODF $(N_ROWS)-row batch"][solver] = s
                println("  VirtualMODF $(N_ROWS)-row batch (rows only): $(cell_str(s))")
            end
        catch e
            println("  VirtualMODF row batch FAILED: $e")
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Results table
# ─────────────────────────────────────────────────────────────────────────────

println("\n\n" * "="^60)
println("RESULTS")
println("="^60)

println()
println("System:    $SYSTEM_NAME")
println("Buses:     $n_buses")
println("Outages:   $n_outages (kv_threshold = $KV_THRESHOLD kV)")
println("Hardware:  $(Sys.MACHINE)")
println("Julia:     $(VERSION)")
println("Passes:    $PASSES timed ($WARMUP warmup discarded); cells are median [min–max]")
println(
    "Row batch: $N_ROWS spread-out arcs; fresh cold-cache object per pass, row loop only",
)
println()

header = "| Path | KLU | AA_LU | KLU/AA_LU |"
sep = "|---|---|---|---|"
println(header)
println(sep)

for p in PATHS
    s_klu = results[p]["KLU"]
    s_lu = results[p]["AppleAccelerateLU"]
    row = @sprintf(
        "| %s | %s | %s | %s |",
        p,
        cell_str(s_klu),
        cell_str(s_lu),
        ratio_str(s_klu, s_lu),
    )
    println(row)
end

println()
println("Done.")
