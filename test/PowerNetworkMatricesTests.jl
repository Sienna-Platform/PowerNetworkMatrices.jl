module PowerNetworkMatricesTests

using ReTest
using Logging
using Random
import LinearAlgebra: I, opnorm
using PowerNetworkMatrices
using TimeSeries
using DelimitedFiles
using InteractiveUtils
using PowerSystemCaseBuilder
using PowerNetworkMatrices
import PowerNetworkMatrices as PNM
import InfrastructureSystems as IS
using PowerSystems
import PowerSystems as PSY
using PowerSystemCaseBuilder
import PowerSystemCaseBuilder as PSB

# Aqua tests
import Aqua
Aqua.test_unbound_args(PowerNetworkMatrices)
Aqua.test_undefined_exports(PowerNetworkMatrices)
Aqua.test_ambiguities(PowerNetworkMatrices)
# The two OpenAPI packages are never loaded by PNM. They sit in `[deps]` only because Pkg
# requires every `[sources]` entry to appear in `deps` or `extras`, and PNM must carry its own
# pins for PSY's unregistered OpenAPI deps (see the comment in Project.toml). Drop both from
# this ignore list when the psy6 pins come off at release.
Aqua.test_stale_deps(
    PowerNetworkMatrices;
    ignore = [:Pardiso, :PowerCoreOpenAPIModels, :PowerOperationsOpenAPIModels],
)
Aqua.test_deps_compat(PowerNetworkMatrices)
# `find_persistent_tasks_deps`/`test_persistent_tasks` are deliberately not run: they
# precompile PNM inside a throwaway temp project that does not inherit this repo's
# `[sources]` git pins, so PowerSystems resolves to the *registered* release instead of the
# psy6 branch and PNM fails to load there (`UndefVarError: TransformerCircuit not defined in
# PowerSystems`) for reasons unrelated to persistent tasks. PSY and PowerFlows stop at
# `test_deps_compat` for the same reason; restore these once psy6 is released and the pins
# come off.

const BASE_DIR = dirname(dirname(Base.find_package("PowerNetworkMatrices")))
const TEST_DATA_DIR = joinpath(
    dirname(dirname(Base.find_package("PowerNetworkMatrices"))),
    "test",
    "test_data",
)
const DATA_DIR = PSB.DATA_DIR

const LOG_FILE = "power-network-matrices.log"

# [include test utils here]
include("testing_data.jl")

# [include tests]
for filename in readdir(joinpath(BASE_DIR, "test"))
    if startswith(filename, "test_") && endswith(filename, ".jl")
        include(filename)
    end
end
# include(joinpath(BASE_DIR, "test", "performance", "performance_test.jl"))

# package-independent logging stuff: can be copy-pasted.
function get_logging_level_from_env(env_name::String, default)
    level = get(ENV, env_name, default)
    return IS.get_logging_level(level)
end

function run_tests(args...; kwargs...)
    logger = global_logger()
    try
        logging_config_filename = get(ENV, "SIIP_LOGGING_CONFIG", nothing)
        if !isnothing(logging_config_filename)
            config = IS.LoggingConfiguration(logging_config_filename)
        else
            config = IS.LoggingConfiguration(;
                filename = LOG_FILE,
                file_level = get_logging_level_from_env("SIENNA_FILE_LOG_LEVEL", "Info"),
                console_level = get_logging_level_from_env(
                    "SIENNA_CONSOLE_LOG_LEVEL",
                    "Error",
                ),
            )
        end
        console_logger = Logging.ConsoleLogger(config.console_stream, config.console_level)

        IS.open_file_logger(config.filename, config.file_level) do file_logger
            levels = (Logging.Info, Logging.Warn, Logging.Error)
            multi_logger =
                IS.MultiLogger([console_logger, file_logger], IS.LogEventTracker(levels))
            Logging.global_logger(multi_logger)

            if !isempty(config.group_levels)
                IS.set_group_levels!(multi_logger, config.group_levels)
            end

            @time retest(args...; kwargs...)
            @test length(IS.get_log_events(multi_logger.tracker, Logging.Error)) == 0
            @info IS.report_log_summary(multi_logger)
        end
    finally
        # Guarantee that the global logger is reset.
        global_logger(logger)
        nothing
    end
end

export run_tests

end

using .PowerNetworkMatricesTests
