precompile = @timed using PowerNetworkMatrices

open("precompile_time.txt", "a") do io
    write(io, "| $(ARGS[1]) | $(precompile.time) |\n")
end

using PowerSystems
using PowerSystemCaseBuilder
using Logging

configure_logging(; console_level = Logging.Error)
systems = [
    (MatpowerTestSystems, "matpower_ACTIVSg2000_sys"),
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]
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
            open("execute_time.txt", "a") do io
                write(io, "| $(ARGS[1])-$(name)-Build PTDF First | $(time_build_ptdf1) |\n")
            end
            _, time_build_ptdf2, _, _ = @timed PTDF(sys)
            open("execute_time.txt", "a") do io
                write(
                    io,
                    "| $(ARGS[1])-$(name)-Build PTDF Second | $(time_build_ptdf2) |\n",
                )
            end
        catch e
            @error exception = (e, catch_backtrace())
            open("execute_time.txt", "a") do io
                write(io, "| $(ARGS[1])-$(name)-Build PTDF | FAILED TO TEST |\n")
            end
        end
    end

    try
        _, time_build_ybus1, _, _ = @timed Ybus(sys)
        open("execute_time.txt", "a") do io
            write(io, "| $(ARGS[1])-$(name)-Build Ybus First | $(time_build_ybus1) |\n")
        end
        _, time_build_ybus2, _, _ = @timed Ybus(sys)
        open("execute_time.txt", "a") do io
            write(io, "| $(ARGS[1])-$(name)-Build Ybus Second | $(time_build_ybus2) |\n")
        end
    catch e
        @error exception = (e, catch_backtrace())
        open("execute_time.txt", "a") do io
            write(io, "| $(ARGS[1])-$(name)-Build Ybus | FAILED TO TEST |\n")
        end
    end
    if build_lodf
        try
            _, time_build_LODF1, _, _ = @timed LODF(sys)
            open("execute_time.txt", "a") do io
                write(io, "| $(ARGS[1])-$(name)-Build LODF First | $(time_build_LODF1) |\n")
            end
            _, time_build_LODF2, _, _ = @timed LODF(sys)
            open("execute_time.txt", "a") do io
                write(
                    io,
                    "| $(ARGS[1])-$(name)-Build LODF Second | $(time_build_LODF2) |\n",
                )
            end
        catch e
            @error exception = (e, catch_backtrace())
            open("execute_time.txt", "a") do io
                write(io, "| $(ARGS[1])-$(name)-Build LODF | FAILED TO TEST |\n")
            end
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
            _, time_build_modf1, _, _ = @timed VirtualMODF(modf_sys)
            open("execute_time.txt", "a") do io
                write(
                    io,
                    "| $(ARGS[1])-$(name)-Build VirtualMODF First | $(time_build_modf1) |\n",
                )
            end
            vmodf = nothing
            _, time_build_modf2, _, _ = @timed begin
                vmodf = VirtualMODF(modf_sys)
            end
            open("execute_time.txt", "a") do io
                write(
                    io,
                    "| $(ARGS[1])-$(name)-Build VirtualMODF Second | $(time_build_modf2) |\n",
                )
            end
            # Time querying rows for the first registered contingency
            ctgs = collect(values(get_registered_contingencies(vmodf)))
            if !isempty(ctgs)
                ctg = first(ctgs)
                n_query = min(10, length(vmodf.axes[1]))
                _, time_query, _, _ = @timed begin
                    for m in 1:n_query
                        vmodf[m, ctg]
                    end
                end
                open("execute_time.txt", "a") do io
                    write(
                        io,
                        "| $(ARGS[1])-$(name)-VirtualMODF Query $(n_query) rows | $(time_query) |\n",
                    )
                end
            end
        catch e
            @error exception = (e, catch_backtrace())
            open("execute_time.txt", "a") do io
                write(io, "| $(ARGS[1])-$(name)-Build VirtualMODF | FAILED TO TEST |\n")
            end
        end
    end
    try
        A = IncidenceMatrix(sys)
        _, time_radial, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, RadialReduction())
        open("execute_time.txt", "a") do io
            write(
                io,
                "| $(ARGS[1])-$(name)-Radial network reduction First | $(time_radial) |\n",
            )
        end
        _, time_radial2, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, RadialReduction())
        open("execute_time.txt", "a") do io
            write(
                io,
                "| $(ARGS[1])-$(name)-Radial network reduction Second | $(time_radial2) |\n",
            )
        end
    catch e
        @error exception = (e, catch_backtrace())
        open("execute_time.txt", "a") do io
            write(io, "| $(ARGS[1])-$(name)-Radial network reduction | FAILED TO TEST |\n")
        end
    end
    try
        A = AdjacencyMatrix(sys)
        _, time_degreetwo, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, DegreeTwoReduction())
        open("execute_time.txt", "a") do io
            write(
                io,
                "| $(ARGS[1])-$(name)-Degree two network reduction First | $(time_degreetwo) |\n",
            )
        end
        _, time_degreetwo2, _, _ =
            @timed PowerNetworkMatrices.get_reduction(A, sys, DegreeTwoReduction())
        open("execute_time.txt", "a") do io
            write(
                io,
                "| $(ARGS[1])-$(name)-Degree two network reduction Second | $(time_degreetwo2) |\n",
            )
        end
    catch e
        @error exception = (e, catch_backtrace())
        open("execute_time.txt", "a") do io
            write(
                io,
                "| $(ARGS[1])-$(name)-Degree two network reduction | FAILED TO TEST |\n",
            )
        end
    end
end
