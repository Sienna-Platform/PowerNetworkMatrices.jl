using Documenter, PowerNetworkMatrices
import DataStructures: OrderedDict
using Literate
using DocumenterInterLinks

links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/objects.inv",
    "PowerSystems" => "https://sienna-platform.github.io/PowerSystems.jl/stable/",
    "PowerSystemCaseBuilder" => "https://sienna-platform.github.io/PowerSystemCaseBuilder.jl/stable/",
)

include(joinpath(@__DIR__, "make_tutorials.jl"))
make_literate_folder("tutorials")
make_literate_folder("how_to_guides")

pages = OrderedDict(
    "Welcome Page" => "index.md",
    "Tutorials" => Any[
        "Getting Started" => "tutorials/generated_getting_started.md",
    ],
    "How-To Guides" => Any[
        "Build Multiple Matrices" => "how_to_guides/generated_build_multiple_matrices.md",
        "Choose a Linear Solver" => "how_to_guides/generated_choose_linear_solver.md",
        "Reproduce Industry DFAX Values" => "how_to_guides/generated_reproduce_dfax_values.md",
        "Compute Phase Shifter Factors" => "how_to_guides/generated_phase_shifter_factor.md",
        "Apply Network Reductions" => "how_to_guides/generated_apply_network_reductions.md",
        "Set the Sparsification Tolerance" => "how_to_guides/generated_set_tolerance.md",
        "Define and Apply Contingencies" => "how_to_guides/generated_contingencies.md",
        "Diagnose Network Connectivity" => "how_to_guides/generated_diagnose_connectivity.md",
        "Tune and Clear the Virtual-Matrix Cache" => "how_to_guides/generated_virtual_cache.md",
    ],
    "Reference" => Any[
        "Matrix Overview and Indexing" => "reference/network_matrices_overview.md",
        "Matrix Types" => "reference/matrix_types.md",
        "Connectivity and Islands" => "reference/connectivity.md",
        "Contingencies and Modifications" => "reference/contingencies.md",
        "Tolerance and Solvers" => "reference/tolerance_and_solvers.md",
        "Serialization" => "reference/serialization.md",
        "Public API" => "reference/public.md",
        "Internals" => "reference/internals.md",
    ],
    "Explanation" => Any[
        "DC Power Flow Approximation" => "explanation/dc_power_flow_approximation.md",
        "Computational Considerations" => "explanation/computational_considerations.md",
        "Network Reduction Theory" => "explanation/network_reduction_theory.md",
        "Flowgate Methodology" => "explanation/flowgate_methodology.md",
        "Virtual vs. Materialized Matrices" => "explanation/virtual_vs_materialized.md",
        "Concurrency and the KLU Lock" => "explanation/concurrency.md",
        "Equivalent Representation of Reduced Branches" => "explanation/equivalent_branches.md",
        "Slack Distribution and Reference-Bus Conventions" => "explanation/slack_conventions.md",
    ],
)

makedocs(;
    modules = [PowerNetworkMatrices],
    format = Documenter.HTML(;
        mathengine = Documenter.MathJax(),
        prettyurls = haskey(ENV, "GITHUB_ACTIONS"),
        size_threshold_warn = 400 * 2^10,
        size_threshold = 600 * 2^10,
    ),
    sitename = "PowerNetworkMatrices.jl",
    authors = "Jose Daniel Lara, Matt Bossart, Alessandro Francesco Castelli",
    pages = Any[p for p in pages],
    clean = true,
    plugins = [links],
)

deploydocs(;
    repo = "github.com/Sienna-Platform/PowerNetworkMatrices.jl.git",
    target = "build",
    branch = "gh-pages",
    devurl = "dev",
    push_preview = true,
    forcepush = true,
    versions = ["stable" => "v^", "v#.#"],
)
