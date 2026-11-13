using PowerNetworkMatrices
using PowerSystems
using PowerSystemCaseBuilder
using Logging
import LinearAlgebra

configure_logging(; console_level = Logging.Error)
# The thread count of a CI runner can change between runs. One BLAS thread keeps the timings
# comparable; the workflow also starts Julia with one thread.
LinearAlgebra.BLAS.set_num_threads(1)

# Each metric takes samples until it has at least MIN_SAMPLES samples and MIN_TOTAL_TIME
# seconds of timed work, to a maximum of MAX_SAMPLES samples. Short metrics thus get more
# samples than long metrics.
const MIN_SAMPLES = 5
const MAX_SAMPLES = 30
const MIN_TOTAL_TIME = 2.0

# Dense PTDF/LODF become infeasible past this many buses; large systems exercise the
# on-demand Virtual variants (VirtualPTDF/VirtualMODF) instead.
const DENSE_MATRIX_BUS_LIMIT = 2000

# Cap the registered contingencies on large systems so the VirtualMODF build/query stays
# within reasonable CI time and memory (one contingency per eligible branch otherwise).
const MAX_LARGE_SYSTEM_CONTINGENCIES = 500

# Rows in one query sample. A query sample must be long compared to the timer and GC noise.
const N_QUERY_ROWS = 50

systems = [
    (MatpowerTestSystems, "matpower_ACTIVSg2000_sys"),
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]

# Time `f(setup())`; `setup` is not timed. Run once to compile, then take samples. A query
# needs a new object from `setup` for each sample: the query time changes from one build to
# the next by more than it changes inside one build, and a second query of one object reads
# the row cache.
function sample_times(f; setup = () -> nothing)
    f(setup())
    times = Float64[]
    while length(times) < MAX_SAMPLES &&
        (length(times) < MIN_SAMPLES || sum(times) < MIN_TOTAL_TIME)
        obj = setup()
        GC.gc()
        push!(times, @elapsed f(obj))
    end
    return times
end

# The contingency with the smallest key, so each sample solves the same contingency on a new
# object.
function first_contingency(vmodf)
    ctgs = get_registered_contingencies(vmodf)
    isempty(ctgs) && error(
        "VirtualMODF has no registered contingency: the query metric needs at least one.",
    )
    return ctgs[minimum(keys(ctgs))]
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
        record_samples("$(name)-Build Ybus", sample_times(_ -> Ybus(sys)))
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Build Ybus")
    end

    if !is_large
        try
            record_samples("$(name)-Build PTDF", sample_times(_ -> PTDF(sys)))
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build PTDF")
        end
        try
            record_samples("$(name)-Build LODF", sample_times(_ -> LODF(sys)))
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build LODF")
        end
    else
        try
            record_samples(
                "$(name)-Build VirtualPTDF",
                sample_times(_ -> VirtualPTDF(sys)),
            )
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build VirtualPTDF")
        end
        try
            record_samples(
                "$(name)-VirtualPTDF Query $(N_QUERY_ROWS) rows",
                sample_times(
                    vptdf -> for m in 1:N_QUERY_ROWS
                        vptdf[m, 1]
                    end;
                    setup = () -> VirtualPTDF(sys),
                ),
            )
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-VirtualPTDF Query $(N_QUERY_ROWS) rows")
        end
    end

    # Register one full-outage contingency per eligible branch (capped on large
    # systems). FixedForcedOutage with outage_status = 1.0 marks the branch outaged.
    modf_sys = deepcopy(sys)
    max_ctgs = is_large ? MAX_LARGE_SYSTEM_CONTINGENCIES : typemax(Int)
    added = 0
    for branch in get_components(ACTransmission, modf_sys)
        added >= max_ctgs && break
        outage = FixedForcedOutage(; outage_status = 1.0)
        add_supplemental_attribute!(modf_sys, branch, outage)
        added += 1
    end
    try
        record_samples(
            "$(name)-Build VirtualMODF",
            sample_times(_ -> VirtualMODF(modf_sys)),
        )
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Build VirtualMODF")
    end
    try
        record_samples(
            "$(name)-VirtualMODF Query $(N_QUERY_ROWS) rows",
            sample_times(
                function (vmodf)
                    ctg = first_contingency(vmodf)
                    for m in 1:N_QUERY_ROWS
                        vmodf[m, ctg]
                    end
                end;
                setup = () -> VirtualMODF(modf_sys),
            ),
        )
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-VirtualMODF Query $(N_QUERY_ROWS) rows")
    end

    try
        A = IncidenceMatrix(sys)
        record_samples(
            "$(name)-Radial network reduction",
            sample_times(
                _ -> PowerNetworkMatrices.get_reduction(A, sys, RadialReduction()),
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
                _ -> PowerNetworkMatrices.get_reduction(A, sys, DegreeTwoReduction()),
            ),
        )
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Degree two network reduction")
    end
end
