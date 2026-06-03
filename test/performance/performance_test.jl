precompile = @timed using PowerNetworkMatrices

open("precompile_time_$(ARGS[1]).txt", "w") do io
    write(io, string(precompile.time))
end

using PowerSystems
using PowerSystemCaseBuilder
using Logging

configure_logging(; console_level = Logging.Error)

# Number of timing samples collected per metric (after a warm-up). Reporting the median and
# range over several samples makes the comparison robust to one-off noise on shared CI
# runners, which a single timing is very sensitive to.
const N_SAMPLES = 5

# Dense PTDF/LODF become infeasible past this many buses; large systems exercise the
# on-demand Virtual variants (VirtualPTDF/VirtualMODF) instead.
const DENSE_MATRIX_BUS_LIMIT = 2000

# Cap the registered contingencies on large systems so the VirtualMODF build/query stays
# within reasonable CI time and memory (one contingency per eligible branch otherwise).
const MAX_LARGE_SYSTEM_CONTINGENCIES = 500

systems = [
    (MatpowerTestSystems, "matpower_ACTIVSg2000_sys"),
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]

# Warm up once (compilation), then collect `n` warm samples of `f()`.
function sample_times(f; n = N_SAMPLES)
    f()
    times = Vector{Float64}(undef, n)
    for i in 1:n
        GC.gc()
        times[i] = @elapsed f()
    end
    return times
end

function record_samples(label, times)
    open("execute_time_$(ARGS[1]).csv", "a") do io
        write(io, "$(label),$(join(times, ";"))\n")
    end
end

function record_failure(label)
    open("execute_time_$(ARGS[1]).csv", "a") do io
        write(io, "$(label),FAILED\n")
    end
end

for (group, name) in systems
    sys = build_system(group, name)
    # Dense PTDF/LODF are infeasible for very large systems; they exercise VirtualPTDF and a
    # contingency-capped VirtualMODF instead so the matrices are still covered.
    is_large = length(get_components(ACBus, sys)) > DENSE_MATRIX_BUS_LIMIT

    try
        record_samples("$(name)-Build Ybus", sample_times(() -> Ybus(sys)))
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Build Ybus")
    end

    if !is_large
        try
            record_samples("$(name)-Build PTDF", sample_times(() -> PTDF(sys)))
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build PTDF")
        end
        try
            record_samples("$(name)-Build LODF", sample_times(() -> LODF(sys)))
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build LODF")
        end
    else
        try
            record_samples("$(name)-Build VirtualPTDF", sample_times(() -> VirtualPTDF(sys)))
            vptdf = VirtualPTDF(sys)
            n_arcs = length(vptdf.axes[1])
            n_query = min(10, n_arcs)
            query = start -> begin
                for m in start:(start + n_query - 1)
                    vptdf[m, 1]
                end
            end
            query(1)  # warm up (compilation)
            qtimes = Float64[]
            # Query a distinct block of rows per sample so each timing is a fresh row
            # computation rather than a row-cache hit.
            for s in 1:N_SAMPLES
                start = s * n_query + 1
                start + n_query - 1 > n_arcs && break
                GC.gc()
                push!(qtimes, @elapsed query(start))
            end
            if !isempty(qtimes)
                record_samples("$(name)-VirtualPTDF Query $(n_query) rows", qtimes)
            end
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build VirtualPTDF")
        end
    end

    try
        # Register one full-outage contingency per eligible branch (capped on large
        # systems). FixedForcedOutage with outage_status = 1.0 marks the branch outaged.
        modf_sys = deepcopy(sys)
        max_ctgs = is_large ? MAX_LARGE_SYSTEM_CONTINGENCIES : typemax(Int)
        added = 0
        for branch in get_components(ACTransmission, modf_sys)
            typeof(branch) <: PhaseShiftingTransformer && continue
            added >= max_ctgs && break
            outage = FixedForcedOutage(; outage_status = 1.0)
            add_supplemental_attribute!(modf_sys, branch, outage)
            added += 1
        end
        record_samples(
            "$(name)-Build VirtualMODF",
            sample_times(() -> VirtualMODF(modf_sys)),
        )
        vmodf = VirtualMODF(modf_sys)
        ctgs = collect(values(get_registered_contingencies(vmodf)))
        if !isempty(ctgs)
            n_query = min(10, length(vmodf.axes[1]))
            query = ctg -> begin
                for m in 1:n_query
                    vmodf[m, ctg]
                end
            end
            # Sample distinct contingencies so each timing is a fresh Woodbury solve
            # rather than a row-cache hit on the same contingency.
            query_ctgs = collect(Iterators.take(ctgs, N_SAMPLES + 1))
            if length(query_ctgs) >= 2
                query(first(query_ctgs))  # warm up (compilation)
                qtimes = Float64[]
                for ctg in query_ctgs[2:end]
                    GC.gc()
                    push!(qtimes, @elapsed query(ctg))
                end
                record_samples("$(name)-VirtualMODF Query $(n_query) rows", qtimes)
            end
        end
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Build VirtualMODF")
    end

    try
        A = IncidenceMatrix(sys)
        record_samples(
            "$(name)-Radial network reduction",
            sample_times(
                () -> PowerNetworkMatrices.get_reduction(A, sys, RadialReduction()),
            ),
        )
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Radial network reduction")
    end
    try
        A = AdjacencyMatrix(sys)
        record_samples(
            "$(name)-Degree two network reduction",
            sample_times(
                () -> PowerNetworkMatrices.get_reduction(A, sys, DegreeTwoReduction()),
            ),
        )
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Degree two network reduction")
    end
end
