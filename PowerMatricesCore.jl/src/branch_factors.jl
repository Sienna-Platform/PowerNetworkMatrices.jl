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
    return _split_rows(
        _ShareRule(),
        table,
        zeros(length(table.kind)),
        ones(table.n_arc),
        outaged_rows,
    )
end

function _split_rule(
    ::_ShareRule,
    ::Int,
    sign::Int,
    b::Float64,
    ::Float64,
    b_sum::Float64,
    share::Float64,
)
    return _split_share(sign, b, b_sum, share)
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
It throws an error if the file has no branch map. It also throws an error if the file holds
distributed slack weights and the network has more than one reference bus. It throws an error
if a series chain of the branch map has a zero impedance sum. It throws an error if a reduced
bus of the file maps to a target that is not on the bus axis.
"""
function read_branch_factors(
    path::AbstractString;
    linear_solver::Union{String, LinearSolverType} = _default_linear_solver(),
)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        arrays = read_factor_arrays(f, budget)
        _check_reduced_targets(f, arrays, budget)
        core = _factor_core(arrays; linear_solver = linear_solver)
        branch_map = read_branch_map(f, budget, length(get_arc_axis(core)))
        has_outages = haskey(f, "branch_outages")
        if has_outages
            outages = read_branch_outages(f, budget, arrays.arc_susceptances)
        else
            outages = _empty_outage_table(get_branch_arc_count(branch_map))
        end
        dist_slack = read_dist_slack(f, core, budget)
        if !isempty(dist_slack)
            _check_dist_slack_network(core)
            dist_slack = dist_slack ./ sum(dist_slack)
        end
        return BranchFactors(
            core,
            branch_map,
            _leaf_rows(branch_map),
            _row_arcs(branch_map),
            branch_shares(branch_map),
            outages,
            has_outages,
            _bus_columns(get_bus_axis(core), arrays.reverse_bus_map),
            dist_slack,
        )
    end
end

# For a file with the group `reduction`, the group reader checks the reduced-bus targets. A
# file without it can name no removed bus, so every target is on the bus axis.
function _check_reduced_targets(f::HDF5.File, arrays::NamedTuple, budget::ReadBudget)
    bus_ax, arc_ax = arrays.axes[2], arrays.axes[1]
    if haskey(f, "reduction")
        read_reduction_records(f, bus_ax, arc_ax, arrays.reverse_bus_map, budget)
        return
    end
    bus_set = Set(bus_ax)
    for (from, to) in arrays.reverse_bus_map
        to in bus_set || error(
            "Factor file entry reduced_bus_to holds bus $to for the reduced bus $from, " *
            "which is not on the bus axis.",
        )
    end
    return
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
    has_branch(x::BranchFactors, id::Int) -> Bool
    has_branch(x::BranchFactors, id::Int, w::Winding) -> Bool

Return `true` if the branch map of `x` has a row for the branch `id`. Use the first method for
a branch with two terminals. Use the second method for a circuit of a three-winding
transformer: `id` is the id of the transformer, and `w` is `PRIMARY`, `SECONDARY` or `TERTIARY`.
"""
has_branch(x::BranchFactors, id::Int) = haskey(x.leaf_rows, (id, 0))

has_branch(x::BranchFactors, id::Int, w::Winding) = haskey(x.leaf_rows, (id, Int(w)))

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
    return [x.leaf_rows[(id, w)] for w in 0:3 if haskey(x.leaf_rows, (id, w))]
end

function _leaf_row(x::BranchFactors, id::Int)
    key = _branch_key(
        x.leaf_rows,
        id,
        "Branch with id $id has no row in the branch map. A network reduction removed it, or it is not in service.",
    )
    return x.leaf_rows[key]
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
    branch_ptdf_row(x::BranchFactors, id::Int) -> Vector{Float64}
    branch_ptdf_row(x::BranchFactors, id::Int, w::Winding) -> Vector{Float64}

Return the sensitivity factors of the flow of the branch `id` to the injection at each bus,
in the order of `get_bus_axis(x)`. The row has the distributed slack of the file, if any, and
the cutoff of the core. The sign follows the frame of the branch. The function throws an
error if the branch has no row in the branch map. Use the first method for a branch with two
terminals. Use the second method for a circuit of a three-winding transformer: `w` is
`PRIMARY`, `SECONDARY` or `TERTIARY`.
"""
function branch_ptdf_row(x::BranchFactors, id::Int)
    return _branch_ptdf_row(x, _leaf_row(x, id))
end

function branch_ptdf_row(x::BranchFactors, id::Int, w::Winding)
    return _branch_ptdf_row(x, _leaf_row(x, id, Int(w)))
end

function _branch_ptdf_row(x::BranchFactors, r::Int)
    if iszero(x.shares[r])
        return _zero_row(x)
    end
    row = ptdf_row(x.core, x.row_arcs[r])
    if has_dist_slack(x)
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

"""
    branch_contingency(x::BranchFactors, ids::Vector{Int}) -> BranchContingency

Build the outage of the branches `ids`. The function throws an error if the file has no
branch outage table, an id has no row in the outage table, or an id has no row in the
branch map. The vector `ids` must not be empty and must not repeat an id. The function throws an error
if the outage gives a series chain a zero impedance sum.
"""
function branch_contingency(x::BranchFactors, ids::Vector{Int})
    x.has_outages || error(
        "The factor file has no branch outage table, so a contingency by branch id is not available. The writer of the file did not add it.",
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
    branch_ptdf_row(x::BranchFactors, id::Int, c::BranchContingency) -> Vector{Float64}
    branch_ptdf_row(x::BranchFactors, id::Int, w::Winding, c::BranchContingency) -> Vector{Float64}

Return the sensitivity factors of the flow of the branch `id` after the outage `c`, in the
order of `get_bus_axis(x)`. The row of a tripped branch is zero. The function throws an
error in two cases. Case one: `c` belongs to another `BranchFactors`, or the branch has no row
in the branch map. Case two: `c` islands the network and the file holds a distributed slack.
Use the second method for a circuit of a three-winding transformer: `w` is `PRIMARY`,
`SECONDARY` or `TERTIARY`.
"""
function branch_ptdf_row(x::BranchFactors, id::Int, c::BranchContingency)
    _check_owner(x, c)
    return _branch_ptdf_row(x, _leaf_row(x, id), c)
end

function branch_ptdf_row(x::BranchFactors, id::Int, w::Winding, c::BranchContingency)
    _check_owner(x, c)
    return _branch_ptdf_row(x, _leaf_row(x, id, Int(w)), c)
end

function _branch_ptdf_row(x::BranchFactors, r::Int, c::BranchContingency)
    if has_dist_slack(x)
        _check_dist_slack_islanding(c.woodbury)
    end
    if iszero(c.shares[r])
        return _zero_row(x)
    end
    row = apply_woodbury_correction(x.core, x.row_arcs[r], c.woodbury)
    if has_dist_slack(x)
        apply_dist_slack!(row, x.core, x.dist_slack)
    end
    return _finish_row(x, row, c.shares[r])
end
