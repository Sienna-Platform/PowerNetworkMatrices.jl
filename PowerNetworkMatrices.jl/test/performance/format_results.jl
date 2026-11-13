"""
Reads the per-round CSV results and writes a Markdown comparison table.

The workflow runs two rounds of each version, in the order Main, Branch, Main, Branch. Each
cell gives the median and the (min–max) range of the samples of both rounds. The noise floor
of a test is the larger of two deltas: Main round 1 against Main round 2, and Branch round 1
against Branch round 2. A delta is a change only when it is
larger than `MIN_CHANGE` and larger than `NOISE_FACTOR` times the noise floor.

Usage:
    julia format_results.jl <main1_csv> <main2_csv> <branch1_csv> <branch2_csv> <output_md>
"""

using Statistics

const MIN_CHANGE = 0.10
const NOISE_FACTOR = 2.0

# Test name => samples. A failed test has no samples. Keeps the file order in `order`.
function read_results!(order::Vector{String}, path::AbstractString)
    results = Dict{String, Vector{Float64}}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        name, val = strip.(split(line, ","; limit = 2))
        name in order || push!(order, name)
        if val == "FAILED"
            results[name] = Float64[]
        else
            results[name] = parse.(Float64, split(val, ";"))
        end
    end
    return results
end

function pooled(a, b, name)
    x = get(a, name, Float64[])
    y = get(b, name, Float64[])
    if isempty(x) || isempty(y)
        return Float64[]
    end
    return vcat(x, y)
end

# Relative delta of the median of `y` against the median of `x`; NaN if one has no samples.
function relative_delta(x::Vector{Float64}, y::Vector{Float64})
    if isempty(x) || isempty(y)
        return NaN
    end
    m = median(x)
    return (median(y) - m) / m
end

# Pick a display unit from a magnitude in seconds.
function _unit(t::Float64)
    if t < 1.0
        return (1000.0, "ms")
    end
    return (1.0, "s")
end

function format_cell(samples::Vector{Float64})
    isempty(samples) && return "N/A"
    med = median(samples)
    scale, unit = _unit(med)
    r(x) = round(x * scale; digits = 1)
    return string(r(med), " (", r(minimum(samples)), "–", r(maximum(samples)), ") ", unit)
end

function format_pct(x::Float64)
    isnan(x) && return "N/A"
    pct = round(x * 100; digits = 1)
    if pct > 0
        return "+$(pct)%"
    end
    return "$(pct)%"
end

function verdict(delta::Float64, floor::Float64)
    (isnan(delta) || isnan(floor)) && return "N/A"
    if abs(delta) > max(MIN_CHANGE, NOISE_FACTOR * floor)
        if delta > 0
            return "**slower**"
        end
        return "**faster**"
    end
    return "no change"
end

function main()
    order = String[]
    main1 = read_results!(order, ARGS[1])
    main2 = read_results!(order, ARGS[2])
    branch1 = read_results!(order, ARGS[3])
    branch2 = read_results!(order, ARGS[4])
    main_all = Dict(n => pooled(main1, main2, n) for n in order)
    branch_all = Dict(n => pooled(branch1, branch2, n) for n in order)

    open(ARGS[5], "w") do io
        write(io, "## Performance Results\n\n")
        write(
            io,
            "Two rounds per version, run as Main, Branch, Main, Branch. Cells show the median " *
            "(min–max) of the samples of both rounds. Noise is the larger delta between the two " *
            "rounds of one version. A delta is a change only when it is larger than " *
            "$(round(Int, MIN_CHANGE * 100))% and larger than $(NOISE_FACTOR) times the noise.\n\n",
        )
        write(io, "| Test | Main | This Branch | Delta | Noise | Result |\n")
        write(io, "| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for name in order
            delta = relative_delta(main_all[name], branch_all[name])
            # A slower runner during one round of one version shows in that version's pair.
            floor = max(
                abs(
                    relative_delta(
                        get(main1, name, Float64[]),
                        get(main2, name, Float64[]),
                    ),
                ),
                abs(
                    relative_delta(
                        get(branch1, name, Float64[]),
                        get(branch2, name, Float64[]),
                    ),
                ),
            )
            write(
                io,
                "| $(name) | $(format_cell(main_all[name])) | " *
                "$(format_cell(branch_all[name])) | $(format_pct(delta)) | " *
                "$(format_pct(floor)) | $(verdict(delta, floor)) |\n",
            )
        end
    end
    return
end

main()
