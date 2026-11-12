"""
    branch_shares(table::BranchArcTable, outaged_rows::Vector{Int} = Int[]) -> Vector{Float64}

Return one value for each row of `table`: the share of the flow of its arc that the row
carries, with the sign of the frame of the row. The share is the derivative of
[`branch_flows`](@ref) with respect to the arc flow, so phase shifts do not change it.
A dead row has the share 0.0. A zero susceptance sum gives the share 0.0.
The rules for `outaged_rows` are the rules of `branch_flows`. The function throws an error
if a row of `outaged_rows` is out of range, or a series sum is zero.
"""
function branch_shares(
    table::BranchArcTable,
    outaged_rows::Vector{Int} = Int[],
)
    n = length(table.kind)
    n_arc = table.n_arc
    dead = falses(n)
    for r in outaged_rows
        1 <= r <= n ||
            error("branch_shares: outaged row $r is not in 1:$n.")
        dead[r] = true
    end
    outaged = copy(dead)
    kind = table.kind
    parent = table.parent
    sign = table.sign
    b = copy(table.b)
    sb = zeros(n)
    sr = zeros(n)
    live_children = zeros(Int, n)
    dead_children = zeros(Int, n)
    arc_sb = zeros(n_arc)
    for r in n:-1:1
        if dead[r]
            # An outaged row carries no flow, so no sum is checked.
        elseif kind[r] == 9
            if iszero(live_children[r])
                dead[r] = true
            elseif iszero(sb[r])
                b[r] = 0.0
            else
                b[r] = sb[r]
            end
        elseif kind[r] == 10
            if !iszero(dead_children[r])
                dead[r] = true
            else
                iszero(sr[r]) &&
                    error("branch_shares: row $r (series chain) has a zero impedance sum.")
                b[r] = 1 / sr[r]
            end
        end
        p = parent[r]
        if p > 0 && kind[p] == 10
            if dead[r]
                dead_children[p] += 1
            else
                sr[p] += 1 / b[r]
            end
        elseif !dead[r]
            if p < 0
                arc_sb[-p] += b[r]
            else
                live_children[p] += 1
                sb[p] += b[r]
            end
        end
    end
    shares = zeros(n)
    inherited = falses(n)
    for r in 1:n
        p = parent[r]
        if p > 0 && (inherited[p] || (outaged[p] && kind[p] == 9))
            dead[r] = true
            inherited[r] = true
        end
        if !dead[r]
            if p < 0
                shares[r] = _split_share(sign[r], b[r], arc_sb[-p], 1.0)
            elseif kind[p] == 9
                shares[r] = _split_share(sign[r], b[r], sb[p], shares[p])
            else
                shares[r] = sign[r] * shares[p]
            end
        end
    end
    return shares
end

function _split_share(sign::Int, b::Float64, b_sum::Float64, share::Float64)
    if iszero(b_sum)
        return 0.0
    end
    return sign * (b / b_sum) * share
end

function _row_arcs(table::BranchArcTable)
    n = length(table.kind)
    arcs = zeros(Int, n)
    for r in 1:n
        p = table.parent[r]
        if p < 0
            arcs[r] = -p
        else
            arcs[r] = arcs[p]
        end
    end
    return arcs
end

function _leaf_rows(table::BranchArcTable)
    rows = Dict{Tuple{Int, Int}, Int}()
    for r in eachindex(table.kind)
        if table.kind[r] <= 8
            rows[(table.id[r], max(table.kind[r] - 5, 0))] = r
        end
    end
    return rows
end

function _empty_outage_table(n_arc::Int)
    return BranchOutageTable(
        Dict{Tuple{String, String}, Tuple{Bool, UnitRange{Int}}}(),
        Int[],
        Float64[],
        Float64[],
        Int[],
        Bool[],
        Int[],
        Dict{Int, Tuple{String, String}}(),
        n_arc,
    )
end

"""
    BranchFactors

The data to answer queries by branch id from a factor file, without PowerSystems. Build it
with [`read_branch_factors`](@ref).

# Fields
- `core`: the `FactorCore` of the file.
- `branch_map`: the [`BranchArcTable`](@ref) of the file.
- `leaf_rows`: the row of the branch map for each key `(id, winding)`.
- `row_arcs`: the arc of each row of the branch map.
- `shares`: the [`branch_shares`](@ref) of the branch map without an outage.
- `outages`: the [`BranchOutageTable`](@ref) of the file. It is empty if `has_outages` is `false`.
- `has_outages`: `true` if the file holds a branch outage table.
- `bus_columns`: the column of the matrix for each bus number, including reduced buses.
- `dist_slack`: the distributed slack weights, normalized to the sum 1. Empty if the file holds none.
"""
struct BranchFactors{C <: FactorCore}
    core::C
    branch_map::BranchArcTable
    leaf_rows::Dict{Tuple{Int, Int}, Int}
    row_arcs::Vector{Int}
    shares::Vector{Float64}
    outages::BranchOutageTable
    has_outages::Bool
    bus_columns::Dict{Int, Int}
    dist_slack::Vector{Float64}
end

function _bus_columns(bus_axis::Vector{Int}, reverse_bus_map::Dict{Int, Int})
    axis_columns = Dict{Int, Int}()
    for (position, bus) in enumerate(bus_axis)
        axis_columns[bus] = position
    end
    columns = copy(axis_columns)
    for (from, to) in reverse_bus_map
        if haskey(axis_columns, to)
            columns[from] = axis_columns[to]
        end
    end
    return columns
end

"""
    read_branch_factors(path::AbstractString; linear_solver = _default_linear_solver()) -> BranchFactors

Read a factor file for queries by branch id. The function factorizes the network again with
`linear_solver`. It reads the branch map, and the branch outage table if the file has one.
It throws an error if the file has no branch map.
"""
function read_branch_factors(
    path::AbstractString;
    linear_solver::Union{String, LinearSolverType} = _default_linear_solver(),
)
    core = read_factor_core(path; linear_solver = linear_solver)
    branch_map = read_branch_map(path)
    has_outages = HDF5.h5open(f -> haskey(f, "branch_outages"), path, "r")
    if has_outages
        outages = read_branch_outages(path)
    else
        outages = _empty_outage_table(get_branch_arc_count(branch_map))
    end
    reverse_bus_map, weights = HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        arrays = read_factor_arrays(f, budget)
        return arrays.reverse_bus_map, read_dist_slack(f, core, budget)
    end
    dist_slack = weights
    if !isempty(weights)
        dist_slack = weights ./ sum(weights)
    end
    return BranchFactors(
        core,
        branch_map,
        _leaf_rows(branch_map),
        _row_arcs(branch_map),
        branch_shares(branch_map),
        outages,
        has_outages,
        _bus_columns(get_bus_axis(core), reverse_bus_map),
        dist_slack,
    )
end

"Return the `FactorCore` of `x`."
get_core(x::BranchFactors) = x.core

"Return the bus numbers of the columns of the matrix of `x`."
get_bus_axis(x::BranchFactors) = get_bus_axis(x.core)

"Return `true` if the bus `number` has a column in the matrix of `x`."
has_bus_column(x::BranchFactors, number::Int) = haskey(x.bus_columns, number)

"""
    bus_column(x::BranchFactors, number::Int) -> Int

Return the column of the bus `number`. A reduced bus has the column of the bus that absorbed
it. For a bus that a Ward reduction removed, the column is the column of one boundary bus, so
the factor is an approximation. The function throws an error if the bus has no column.
"""
function bus_column(x::BranchFactors, number::Int)
    haskey(x.bus_columns, number) || error(
        "Bus $number has no column in the matrix. It is not a bus of the file, or a network reduction removed it.",
    )
    return x.bus_columns[number]
end

"""
    has_branch(x::BranchFactors, id::Int; winding::Int = 0) -> Bool

Return `true` if the branch map of `x` has a row for the branch `id`. For a three-winding
transformer, `winding` is 1, 2 or 3, and `id` is the id of the transformer.
"""
has_branch(x::BranchFactors, id::Int; winding::Int = 0) =
    haskey(x.leaf_rows, (id, winding))

"""
    has_branch_outages(x::BranchFactors) -> Bool

Return `true` if the factor file of `x` holds a branch outage table.
"""
has_branch_outages(x::BranchFactors) = x.has_outages

"""
    has_dist_slack(x::BranchFactors) -> Bool

Return `true` if the factor file of `x` holds distributed slack weights. `branch_ptdf_row`
with a contingency throws an error when `is_islanding(c)` and `has_dist_slack(x)` are both
`true`.
"""
has_dist_slack(x::BranchFactors) = !isempty(x.dist_slack)

"""
    has_branch_outage(x::BranchFactors, id::Int) -> Bool

Return `true` if `id` has a row in the branch outage table of `x` and at least one row in the
branch map, so `id` can be part of a contingency. The result is `false` for every `id` if the
file has no table.
"""
function has_branch_outage(x::BranchFactors, id::Int)
    return haskey(get_outage_ids(x.outages), id) && !isempty(_branch_rows(x, id))
end

function _branch_rows(x::BranchFactors, id::Int)
    rows = Int[]
    for winding in 0:3
        if haskey(x.leaf_rows, (id, winding))
            push!(rows, x.leaf_rows[(id, winding)])
        end
    end
    return rows
end

function _leaf_row(x::BranchFactors, id::Int, winding::Int)
    haskey(x.leaf_rows, (id, winding)) || error(
        "Branch with id $id and winding $winding has no row in the branch map. A network reduction removed it, or it is not in service.",
    )
    return x.leaf_rows[(id, winding)]
end

function _finish_row(x::BranchFactors, row::Vector{Float64}, share::Float64)
    dense = Vector{Float64}(apply_cutoff(get_cutoff(x.core), row))
    dense .*= share
    return dense
end

function _zero_row(x::BranchFactors)
    return zeros(length(get_bus_axis(x.core)))
end

"""
    branch_ptdf_row(x::BranchFactors, id::Int; winding::Int = 0) -> Vector{Float64}

Return the sensitivity factors of the flow of the branch `id` to the injection at each bus,
in the order of `get_bus_axis(x)`. The row has the distributed slack of the file, if any, and
the cutoff of the core. The sign follows the frame of the branch. The function throws an
error if the branch has no row in the branch map.
"""
function branch_ptdf_row(x::BranchFactors, id::Int; winding::Int = 0)
    r = _leaf_row(x, id, winding)
    if iszero(x.shares[r])
        return _zero_row(x)
    end
    row = ptdf_row(x.core, x.row_arcs[r])
    if !isempty(x.dist_slack)
        apply_dist_slack!(row, x.core, x.dist_slack)
    end
    return _finish_row(x, row, x.shares[r])
end

"""
    BranchContingency

The outage of one or more branches, ready for [`branch_ptdf_row`](@ref). Build it with
[`branch_contingency`](@ref). A contingency is valid only with the `BranchFactors` that built it.

# Fields
- `woodbury`: the `WoodburyFactors` of the outage.
- `shares`: the [`branch_shares`](@ref) of the branch map with the tripped branches outaged.
- `branch_map`: the branch map of the `BranchFactors` that built the contingency.
"""
struct BranchContingency
    woodbury::WoodburyFactors
    shares::Vector{Float64}
    branch_map::BranchArcTable
end

function _check_owner(x::BranchFactors, c::BranchContingency)
    if c.branch_map !== x.branch_map
        error("The contingency belongs to another factor file.")
    end
    return
end

function _check_islanding_slack(x::BranchFactors, c::BranchContingency)
    if c.woodbury.is_islanding && has_dist_slack(x)
        error(
            "Distributed slack is not supported for a modification that islands the network.",
        )
    end
    return
end

"""
    branch_contingency(x::BranchFactors, ids::Vector{Int}) -> BranchContingency

Build the outage of the branches `ids`. The function throws an error if the file has no
branch outage table, an id has no row in the outage table, or an id has no row in the
branch map.
"""
function branch_contingency(x::BranchFactors, ids::Vector{Int})
    x.has_outages || error(
        "The factor file has no branch outage table, so a contingency by branch id is not available. Write the file with the system.",
    )
    modification = NetworkModification(x.outages, ids)
    woodbury = compute_woodbury_factors(x.core, modification)
    outaged_rows = Int[]
    for id in ids
        rows = _branch_rows(x, id)
        isempty(rows) && error("Branch with id $id has no row in the branch map.")
        append!(outaged_rows, rows)
    end
    return BranchContingency(
        woodbury,
        branch_shares(x.branch_map, outaged_rows),
        x.branch_map,
    )
end

"Return `true` if the outage `c` splits the network into islands."
is_islanding(c::BranchContingency) = c.woodbury.is_islanding

"""
    branch_ptdf_row(x::BranchFactors, id::Int, c::BranchContingency; winding::Int = 0) -> Vector{Float64}

Return the sensitivity factors of the flow of the branch `id` after the outage `c`, in the
order of `get_bus_axis(x)`. The row of a tripped branch is zero. The function throws an
error in two cases. Case one: `c` belongs to another `BranchFactors`, or the branch has no row
in the branch map. Case two: `c` islands the network and the file holds a distributed slack.
"""
function branch_ptdf_row(
    x::BranchFactors,
    id::Int,
    c::BranchContingency;
    winding::Int = 0,
)
    _check_owner(x, c)
    r = _leaf_row(x, id, winding)
    _check_islanding_slack(x, c)
    if iszero(c.shares[r])
        return _zero_row(x)
    end
    row = apply_woodbury_correction(x.core, x.row_arcs[r], c.woodbury)
    if !isempty(x.dist_slack)
        apply_dist_slack!(row, x.core, x.dist_slack, c.woodbury)
    end
    return _finish_row(x, row, c.shares[r])
end
