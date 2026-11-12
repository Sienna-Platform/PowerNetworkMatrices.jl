"""
Reads per-version CSV results and writes a Markdown comparison table.

Execution-time values are recorded as `;`-separated samples (see `performance_test.jl`);
each cell reports the median and the (min–max) range over those samples, and the delta
compares medians. Precompile time remains a single value.

Usage:
    julia format_results.jl <main_csv> <branch_csv> <main_precompile_txt> <branch_precompile_txt> <output_md>
"""

using Statistics

function read_results(path::AbstractString)::Vector{Tuple{String, String}}
    results = Tuple{String, String}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        parts = split(line, ","; limit = 2)
        push!(results, (strip(parts[1]), strip(parts[2])))
    end
    return results
end

# `nothing` for FAILED/N/A, otherwise the parsed sample vector.
function parse_samples(val::AbstractString)
    (val == "FAILED" || val == "N/A") && return nothing
    return parse.(Float64, split(val, ";"))
end

# Pick a display unit from a magnitude in seconds.
function _unit(t::Float64)
    if t < 1.0
        return (1000.0, "ms")
    end
    return (1.0, "s")
end

# Single seconds value -> "X ms"/"X s" (used for precompile time).
function format_single(val::AbstractString)::String
    (val == "FAILED" || val == "N/A") && return val
    t = parse(Float64, val)
    scale, unit = _unit(t)
    return string(round(t * scale; digits = 1), " ", unit)
end

# Sample vector -> "median (min–max) unit". The unit is chosen from the median.
function format_cell(val::AbstractString)::String
    samples = parse_samples(val)
    isnothing(samples) && return val
    med = median(samples)
    scale, unit = _unit(med)
    r(x) = round(x * scale; digits = 1)
    return string(r(med), " (", r(minimum(samples)), "–", r(maximum(samples)), ") ", unit)
end

function _median_or_nothing(val::AbstractString)
    samples = parse_samples(val)
    isnothing(samples) && return nothing
    return median(samples)
end

# Delta between branch and main medians, as a signed percentage.
function compute_delta(main_val::AbstractString, branch_val::AbstractString)::String
    main_med = _median_or_nothing(main_val)
    branch_med = _median_or_nothing(branch_val)
    (isnothing(main_med) || isnothing(branch_med) || main_med == 0.0) && return "N/A"
    pct = round((branch_med - main_med) / main_med * 100; digits = 1)
    prefix = ""
    if pct > 0
        prefix = "+"
    end
    return "$(prefix)$(pct)%"
end

# Delta between two single values (precompile time).
function compute_delta_single(
    main_val::AbstractString,
    branch_val::AbstractString,
)::String
    (
        main_val == "FAILED" || branch_val == "FAILED" ||
        main_val == "N/A" || branch_val == "N/A"
    ) && return "N/A"
    main_t = parse(Float64, main_val)
    branch_t = parse(Float64, branch_val)
    main_t == 0.0 && return "N/A"
    pct = round((branch_t - main_t) / main_t * 100; digits = 1)
    prefix = ""
    if pct > 0
        prefix = "+"
    end
    return "$(prefix)$(pct)%"
end

function main()
    main_csv = ARGS[1]
    branch_csv = ARGS[2]
    main_precompile = ARGS[3]
    branch_precompile = ARGS[4]
    output_md = ARGS[5]

    main_results = read_results(main_csv)
    branch_results = read_results(branch_csv)

    main_dict = Dict(name => val for (name, val) in main_results)
    branch_dict = Dict(name => val for (name, val) in branch_results)

    # Preserve order from main results, then add any branch-only tests
    all_tests = String[name for (name, _) in main_results]
    for (name, _) in branch_results
        name in all_tests || push!(all_tests, name)
    end

    main_precompile_time = strip(read(main_precompile, String))
    branch_precompile_time = strip(read(branch_precompile, String))

    open(output_md, "w") do io
        write(io, "## Performance Results\n\n")

        # Precompile table
        precompile_delta =
            compute_delta_single(main_precompile_time, branch_precompile_time)
        write(io, "### Precompile Time\n\n")
        write(io, "| Main | This Branch | Delta |\n")
        write(io, "| :---: | :---: | :---: |\n")
        write(
            io,
            "| $(format_single(main_precompile_time)) | $(format_single(branch_precompile_time)) | $(precompile_delta) |\n",
        )

        # Execution time table
        write(io, "\n### Execution Time\n\n")
        write(
            io,
            "Cells show median (min–max) over $(length(parse_samples_for_count(main_results))) samples; delta compares medians.\n\n",
        )
        write(io, "| Test | Main | This Branch | Delta |\n")
        write(io, "| :--- | :---: | :---: | :---: |\n")
        for test in all_tests
            main_val = get(main_dict, test, "N/A")
            branch_val = get(branch_dict, test, "N/A")
            delta = compute_delta(main_val, branch_val)
            write(
                io,
                "| $(test) | $(format_cell(main_val)) | $(format_cell(branch_val)) | $(delta) |\n",
            )
        end
    end
end

# Best-effort sample count for the header note (first parseable row).
function parse_samples_for_count(results::Vector{Tuple{String, String}})
    for (_, val) in results
        samples = parse_samples(val)
        isnothing(samples) || return samples
    end
    return Float64[]
end

main()
