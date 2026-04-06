using Documenter, PowerNetworkMatrices
import DataStructures: OrderedDict
using DocumenterInterLinks

links = InterLinks(
    "Julia" => "https://docs.julialang.org/en/v1/",
    "InfrastructureSystems" =>
        "https://nrel-sienna.github.io/InfrastructureSystems.jl/stable/",
    "PowerSystems" => "https://nrel-sienna.github.io/PowerSystems.jl/stable/",
    "PowerSimulations" => "https://nrel-sienna.github.io/PowerSimulations.jl/stable/",
)

pages = OrderedDict(
    "Welcome Page" => "index.md",
    "Tutorials" => Any[
        "Network Matrices" => "tutorials/tutorial_network_matrices.md",
        "Virtual Matrices" => "tutorials/tutorial_virtual_matrices.md",
        "Network Reduction" => "tutorials/tutorial_network_reduction.md",
    ],
    "How-To Guides" => Any[
        "Choose a Linear Solver" => "how_to_guides/choose_linear_solver.md",
        "Reduce Repeated Operations" => "how_to_guides/reduce_repeated_operations.md",
    ],
    "Explanation" => Any[
        "Computational Considerations" => "explanation/computational_considerations.md",
        "DC Power Flow Approximation" => "explanation/dc_power_flow_approximation.md",
        "Network Reduction Theory" => "explanation/network_reduction_theory.md",
    ],
    "Reference" => Any[
        "Matrix Overview" => "reference/network_matrices_overview.md",
        "Public API" => "reference/public.md",
    ],
)

makedocs(;
    modules = [PowerNetworkMatrices],
    format = Documenter.HTML(;
        mathengine = Documenter.MathJax(),
        prettyurls = haskey(ENV, "GITHUB_ACTIONS"),
        size_threshold_warn = 200 * 2^10,
        size_threshold = 300 * 2^10,
    ),
    sitename = "PowerNetworkMatrices.jl",
    authors = "Jose Daniel Lara, Matt Bossart, Alessandro Francesco Castelli",
    pages = Any[p for p in pages],
    plugins = [links],
    clean = true,
)

deploydocs(;
    repo = "github.com/NREL-Sienna/PowerNetworkMatrices.jl.git",
    target = "build",
    branch = "gh-pages",
    devurl = "dev",
    push_preview = true,
    forcepush = true,
    versions = ["stable" => "v^", "v#.#"],
)
