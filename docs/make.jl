using Documenter, PowerNetworkMatrices
import DataStructures: OrderedDict
using Literate
using DocumenterInterLinks

links = InterLinks(
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
        "Network Matrices" => "tutorials/generated_network_matrices.md",
        "Incidence, BA and ABA matrices" => "tutorials/generated_tutorial_Incidence_BA_ABA_matrices.md",
        "PTDF matrix" => "tutorials/generated_tutorial_PTDF_matrix.md",
        "VirtualPTDF matrix" => "tutorials/generated_tutorial_VirtualPTDF_matrix.md",
        "LODF matrix" => "tutorials/generated_tutorial_LODF_matrix.md",
        "VirtualLODF matrix" => "tutorials/generated_tutorial_VirtualLODF_matrix.md",
        "Industry DFAX values" => "tutorials/generated_tutorial_DFAX.md",
        "Radial Reduction" => "tutorials/generated_tutorial_RadialReduction.md",
        "Degree Two Reduction" => "tutorials/generated_tutorial_DegreeTwoReduction.md",
    ],
    "How-To Guides" => Any[
        "Compute Network Matrices" => "how_to_guides/generated_compute_network_matrices.md",
        "Choose a Linear Solver" => "how_to_guides/generated_choose_linear_solver.md",
    ],
    "Explanation" => Any[
        "Computational Considertaions" => "explanation/computational_considerations.md",
        "DC Power Flow Approximation" => "explanation/dc_power_flow_approximation.md",
        "Network Reduction Theory" => "explanation/network_reduction_theory.md",
        "Flowgate Methodology" => "explanation/flowgate_methodology.md",
    ],
    "Reference" => Any[
        "Matrix Overview" => "reference/network_matrices_overview.md",
        "Public API" => "reference/public.md",
        "Internals" => "reference/internals.md",
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
