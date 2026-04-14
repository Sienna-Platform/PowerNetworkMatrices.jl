module PowerNetworkMatricesTests

using ReTest
using Logging
import LinearAlgebra: I
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
Aqua.test_stale_deps(PowerNetworkMatrices; ignore = [:AppleAccelerate, :Pardiso])
Aqua.test_deps_compat(PowerNetworkMatrices)
Aqua.find_persistent_tasks_deps(PowerNetworkMatrices)
Aqua.test_persistent_tasks(PowerNetworkMatrices)

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
        if logging_config_filename !== nothing
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
