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
    # Avoid building ptdf/lodf for large systems
    if length(get_components(ACBus, sys)) > 2000
        build_ptdf = false
        build_lodf = false
    else
        build_ptdf = true
        build_lodf = true
    end
    if build_ptdf
        try
            record_samples("$(name)-Build PTDF", sample_times(() -> PTDF(sys)))
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build PTDF")
        end
    end

    try
        record_samples("$(name)-Build Ybus", sample_times(() -> Ybus(sys)))
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Build Ybus")
    end
    if build_lodf
        try
            record_samples("$(name)-Build LODF", sample_times(() -> LODF(sys)))
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build LODF")
        end
    end
    if build_ptdf
        try
            # Add outages to all eligible branches for VirtualMODF construction
            modf_sys = deepcopy(sys)
            for branch in get_components(ACTransmission, modf_sys)
                typeof(branch) <: PhaseShiftingTransformer && continue
                outage = GeometricDistributionForcedOutage(;
                    mean_time_to_recovery = 0.0,
                    outage_transition_probability = 0.0,
                )
                add_supplemental_attribute!(modf_sys, branch, outage)
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
