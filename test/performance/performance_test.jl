precompile = @timed using PowerNetworkMatrices

open("precompile_time_$(ARGS[1]).txt", "w") do io
    write(io, string(precompile.time))
end

using PowerSystems
using PowerSystemCaseBuilder
using Logging

configure_logging(; console_level = Logging.Error)
systems = [
    (MatpowerTestSystems, "matpower_ACTIVSg2000_sys"),
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]

function record_time(label, time)
    open("execute_time_$(ARGS[1]).csv", "a") do io
        write(io, "$(label),$(time)\n")
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
            _, time_build_ptdf1, _, _ = @timed PTDF(sys)
            record_time("$(name)-Build PTDF First", time_build_ptdf1)
            _, time_build_ptdf2, _, _ = @timed PTDF(sys)
            record_time("$(name)-Build PTDF Second", time_build_ptdf2)
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build PTDF")
        end
    end

    try
        _, time_build_ybus1, _, _ = @timed Ybus(sys)
        record_time("$(name)-Build Ybus First", time_build_ybus1)
        _, time_build_ybus2, _, _ = @timed Ybus(sys)
        record_time("$(name)-Build Ybus Second", time_build_ybus2)
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Build Ybus")
    end
    if build_lodf
        try
            _, time_build_LODF1, _, _ = @timed LODF(sys)
            record_time("$(name)-Build LODF First", time_build_LODF1)
            _, time_build_LODF2, _, _ = @timed LODF(sys)
            record_time("$(name)-Build LODF Second", time_build_LODF2)
        catch e
            @error exception = (e, catch_backtrace())
            record_failure("$(name)-Build LODF")
        end
    end
    try
        A = IncidenceMatrix(sys)
        _, time_radial, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, RadialReduction())
        record_time("$(name)-Radial network reduction First", time_radial)
        _, time_radial2, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, RadialReduction())
        record_time("$(name)-Radial network reduction Second", time_radial2)
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Radial network reduction")
    end
    try
        A = AdjacencyMatrix(sys)
        _, time_degreetwo, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, DegreeTwoReduction())
        record_time("$(name)-Degree two network reduction First", time_degreetwo)
        _, time_degreetwo2, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, DegreeTwoReduction())
        record_time("$(name)-Degree two network reduction Second", time_degreetwo2)
    catch e
        @error exception = (e, catch_backtrace())
        record_failure("$(name)-Degree two network reduction")
    end
end
