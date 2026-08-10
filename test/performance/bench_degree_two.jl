# Degree-two reduction performance gate. Run with:
#   julia --project=test test/performance/bench_degree_two.jl <label>
using PowerNetworkMatrices, PowerSystems, PowerSystemCaseBuilder, Logging
const PNM = PowerNetworkMatrices
configure_logging(; console_level = Logging.Error)

const N = 7

function sample(f; n = N)
    f()
    ts = Float64[]
    for _ in 1:n
        GC.gc()
        push!(ts, @elapsed f())
    end
    return sort(ts)
end

med(ts) = ts[div(length(ts), 2) + 1]

for (mod, name) in [
    (MatpowerTestSystems, "matpower_ACTIVSg2000_sys"),
    (MatpowerTestSystems, "matpower_ACTIVSg10k_sys"),
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW"),
]
    sys = build_system(mod, name)
    # Kernel only: the degree-two reduction off a prebuilt adjacency matrix.
    A = AdjacencyMatrix(sys)
    kernel = sample(() -> PNM.get_reduction(A, sys, DegreeTwoReduction()))
    # End to end, in the order a consumer uses.
    e2e = sample(
        () -> Ybus(sys;
            network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()]),
    )
    nrd = get_network_reduction_data(
        Ybus(sys;
            network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()]),
    )
    n_series = length(PNM.get_series_branch_map(nrd))
    n_parallel = length(PNM.get_parallel_branch_map(nrd))
    println(
        "$name|kernel_ms=$(round(med(kernel) * 1000, digits = 3))|" *
        "e2e_ms=$(round(med(e2e) * 1000, digits = 1))|" *
        "series=$n_series|parallel=$n_parallel",
    )
end
