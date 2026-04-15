using Documenter, PowerNetworkMatrices
import DataStructures: OrderedDict

pages = OrderedDict(
    "Welcome Page" => "index.md",
    "Tutorials" => Any[
        "Getting Started" => "tutorials/getting_started.md",
        "Incidence, BA and ABA matrices" => "tutorials/tutorial_Incidence_BA_ABA_matrices.md",
        "PTDF matrix" => "tutorials/tutorial_PTDF_matrix.md",
        "VirtualPTDF matrix" => "tutorials/tutorial_VirtualPTDF_matrix.md",
        "LODF matrix" => "tutorials/tutorial_LODF_matrix.md",
        "VirtualLODF matrix" => "tutorials/tutorial_VirtualLODF_matrix.md",
        "Radial Reduction" => "tutorials/tutorial_RadialReduction.md",
        "Degree Two Reduction" => "tutorials/tutorial_DegreeTwoReduction.md",
    ],
    "How-To Guides" => Any[
        "Compute Network Matrices" => "how_to_guides/compute_network_matrices.md",
        "Choose a Linear Solver" => "how_to_guides/choose_linear_solver.md",
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
