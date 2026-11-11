using Documenter, PowerMatricesCore

pages = Any[
    "Welcome Page" => "index.md",
    "Reference" => Any[
        "Public API" => "reference/public.md",
        "Internals" => "reference/internals.md",
    ],
]

makedocs(;
    modules = [PowerMatricesCore],
    format = Documenter.HTML(;
        mathengine = Documenter.MathJax(),
        prettyurls = haskey(ENV, "GITHUB_ACTIONS"),
    ),
    sitename = "PowerMatricesCore.jl",
    authors = "Jose Daniel Lara, Matt Bossart, Alessandro Francesco Castelli",
    pages = pages,
    clean = true,
)

deploydocs(;
    repo = "github.com/Sienna-Platform/PowerNetworkMatrices.jl.git",
    target = "build",
    branch = "gh-pages",
    dirname = "PowerMatricesCore",
    tag_prefix = "PowerMatricesCore-",
    devurl = "dev",
    push_preview = true,
    versions = ["stable" => "v^", "v#.#"],
)
