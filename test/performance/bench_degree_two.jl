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

# `configure_logging` above sets the process-wide console level to `Error`, so the
# `IterativeTopologyReduction` round count never reaches the console. `with_logger` overrides
# that for the dynamic scope of one call regardless, so this logger is how the round count is
# read back for the comparison below.
mutable struct RoundCapture <: Logging.AbstractLogger
    rounds::Int
end
Logging.min_enabled_level(::RoundCapture) = Logging.Info
Logging.shouldlog(::RoundCapture, args...) = true
Logging.catch_exceptions(::RoundCapture) = true
function Logging.handle_message(
    l::RoundCapture,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    m = match(r"converged after (\d+) productive round", string(message))
    m === nothing || (l.rounds = parse(Int, m.captures[1]))
    return
end

function rounds_and_ybus(sys, reduction)
    cap = RoundCapture(-1)
    ybus =
        Logging.with_logger(cap) do
            Ybus(sys; network_reductions = NetworkReduction[reduction])
        end
    return ybus, cap.rounds
end

for (mod, name, kwargs) in [
    (MatpowerTestSystems, "matpower_ACTIVSg2000_sys", NamedTuple()),
    (MatpowerTestSystems, "matpower_ACTIVSg10k_sys", NamedTuple()),
    (PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW", NamedTuple()),
    (PSYTestSystems, "psse_240_parsing_sys", (; runchecks = false)),
]
    sys = build_system(mod, name; kwargs...)
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

    # `[RadialReduction(), DegreeTwoReduction()]` (one pass each, the order every existing
    # consumer uses) versus `[IterativeTopologyReduction()]` (alternates both to a fixed point).
    # The round count is the decision input for whether an internally-iterating
    # `DegreeTwoReduction` is ever worth building: if every system converges in a couple of
    # rounds, the outer loop already pays for that cost and an internal version would only save
    # passes, not correctness. Both the default (DC) and AC-safe
    # (`reduce_reactive_power_injectors = false`) configurations are measured, since AC consumers
    # take a different path through degree-two reduction.
    for (label, degree_two) in [
        ("DC", DegreeTwoReduction()),
        ("AC", DegreeTwoReduction(; reduce_reactive_power_injectors = false)),
    ]
        sequential = NetworkReduction[RadialReduction(), degree_two]
        seq_time = med(sample(() -> Ybus(sys; network_reductions = sequential)))
        seq_buses = length(PNM.get_bus_axis(Ybus(sys; network_reductions = sequential)))

        iterative = IterativeTopologyReduction(; degree_two = degree_two)
        iter_time =
            med(sample(() -> Ybus(sys; network_reductions = NetworkReduction[iterative])))
        iter_ybus, rounds = rounds_and_ybus(sys, iterative)
        iter_buses = length(PNM.get_bus_axis(iter_ybus))

        println(
            "$name|$label|seq_buses=$seq_buses|iter_buses=$iter_buses|" *
            "diff=$(seq_buses - iter_buses)|rounds=$rounds|" *
            "seq_ms=$(round(seq_time * 1000, digits = 1))|" *
            "iter_ms=$(round(iter_time * 1000, digits = 1))",
        )
    end
end
