# Shared preamble evaluated into every test worker's sandbox module before the test file
# body runs. Load Pardiso before PowerNetworkMatrices so the MKL/Pardiso extension
# precompiles; the Apple Accelerate backend is built in and needs no trigger package.
@static if (Sys.ARCH === :x86_64 || Sys.ARCH === :i686) && !Sys.isapple()
    using Pardiso
end

using Test
using Logging
using Random
import LinearAlgebra: I, opnorm
using PowerNetworkMatrices
using TimeSeries
using DelimitedFiles
using InteractiveUtils
import PowerNetworkMatrices as PNM
import InfrastructureSystems as IS
using PowerSystems
import PowerSystems as PSY
using PowerSystemCaseBuilder
import PowerSystemCaseBuilder as PSB

const BASE_DIR = dirname(dirname(Base.find_package("PowerNetworkMatrices")))
const TEST_DATA_DIR = joinpath(BASE_DIR, "test", "test_data")
const DATA_DIR = PSB.DATA_DIR

# Keep each worker's captured console output to real problems; the runner echoes it back.
Logging.global_logger(Logging.ConsoleLogger(stderr, Logging.Error))

include("testing_data.jl")
