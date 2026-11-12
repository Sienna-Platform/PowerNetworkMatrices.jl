"""
    BranchOutageTable

Branch outages of the group `branch_outages` of a factor file. Read it with
`read_branch_outages(path)`. Build a `NetworkModification` from it with
`NetworkModification(table, type, name)`.

# Fields
- `rows`: for the key `(type, name)`, the islanding flag and the row range.
- `arc_index`, `delta_b`, `delta_shift`, `opened`: the arrays that the row ranges index.
- `opens_arc`: for each arc row, `true` when the outage of that one branch removes all
  admittance of the arc.
- `member`: for each arc row, the position of the tripped member in the parallel group of
  the arc. Zero when the arc is not a parallel group.
- `ids`: the id of each branch, mapped to its key `(type, name)`. Build a modification with
  `NetworkModification(table, id::Int)`.
- `n_arc`: the number of arcs in the file. Each `arc_index` is at most `n_arc`.
"""
struct BranchOutageTable
    rows::Dict{Tuple{String, String}, Tuple{Bool, UnitRange{Int}}}
    arc_index::Vector{Int}
    delta_b::Vector{Float64}
    delta_shift::Vector{Float64}
    opened::Vector{Int}
    opens_arc::Vector{Bool}
    member::Vector{Int}
    ids::Dict{Int, Tuple{String, String}}
    n_arc::Int
end

"Return the map from the key `(type, name)` to the islanding flag and the row range of `t`."
get_outage_rows(t::BranchOutageTable) = t.rows

"Return the map from the id of a branch to its key `(type, name)` in `t`."
get_outage_ids(t::BranchOutageTable) = t.ids

"Return the `opens_arc` column of `t`: `true` when the outage removes all admittance of the arc."
get_outage_opens_arc(t::BranchOutageTable) = t.opens_arc

"Return the `member` column of `t`: the position of the tripped member in the parallel group."
get_outage_member(t::BranchOutageTable) = t.member

function _check_row_values(
    group::String,
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    n_arc::Int,
)
    for (row, i) in enumerate(arc_index)
        1 <= i <= n_arc || error(
            "Factor file entry $group/arc_index has the value $i at row $row; " *
            "expected a value in 1:$n_arc.",
        )
    end
    for (row, b) in enumerate(delta_b)
        (isfinite(b) && !iszero(b) && abs(b) <= MAX_ARC_SUSCEPTANCE) || error(
            "Factor file entry $group/delta_b has the value $b in arc row $row. The value " *
            "must be finite, not zero, and at most $MAX_ARC_SUSCEPTANCE in magnitude.",
        )
    end
    for (row, x) in enumerate(delta_shift)
        isfinite(x) || error(
            "Factor file entry $group/delta_shift has the value $x at row $row; " *
            "the value must be finite.",
        )
    end
    for (row, x) in enumerate(opened)
        x >= 0 || error(
            "Factor file entry $group/opened has the value $x at row $row; " *
            "the value must not be negative.",
        )
    end
    return
end

function _check_row_ranges(
    group::String,
    starts::Vector{Int},
    lens::Vector{Int},
    n_rows::Int,
)
    for (k, s) in enumerate(starts)
        s >= 1 || error(
            "Factor file entry $group/start has the value $s at row $k; expected 1 or more.",
        )
        s <= n_rows + 1 || error(
            "Factor file entry $group/start has the value $s at row $k; " *
            "the group has $n_rows arc rows.",
        )
    end
    for (k, l) in enumerate(lens)
        l >= 0 || error(
            "Factor file entry $group/len has the value $l at row $k; " *
            "the value must not be negative.",
        )
        l <= n_rows || error(
            "Factor file entry $group/len has the value $l at row $k; " *
            "the group has $n_rows arc rows.",
        )
    end
    return
end

# An outage on an arc with zero susceptance divides by zero in the Woodbury factors.
function _check_outage_arcs(
    arc_index::Vector{Int},
    arc_susceptances::Vector{Float64},
)
    for (row, i) in enumerate(arc_index)
        iszero(arc_susceptances[i]) && error(
            "Factor file outage row $row is on arc $i, whose arc_susceptances entry is zero.",
        )
    end
    return
end

function _arc_modifications(
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    rows::Union{Vector{Int}, UnitRange{Int}},
)
    z = zero(YBUS_ELTYPE)
    return ArcModification[
        ArcModification(arc_index[i], delta_b[i], delta_shift[i], z, z, z, z, opened[i])
        for i in rows
    ]
end

"""
    NetworkModification(table::BranchOutageTable, type, name)
    NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})

Build the outage of one branch, or of several branches at once, from a
[`BranchOutageTable`](@ref). `type` is a `String`, for example
`"Line"`. The function groups the arc rows of all keys by arc:

  - If a row opens the whole arc, the arc gets the first such row only.
  - Otherwise the arc gets the first row of each tripped group member. Partial outages of
    different members add. Two lines of one chain in a parallel group remove that chain one
    time.

The modification has no Ybus deltas and no shunt modifications. It serves the DC row
kernels only.
"""
function NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})
    if isempty(keys)
        error("An outage needs at least one branch key.")
    end
    allunique(keys) || error(
        "Duplicate branch key in outage: $(collect(Set(k for k in keys if count(==(k), keys) > 1))).",
    )
    # `arcs` holds the arcs in the order of their first row.
    by_arc = Dict{Int, Vector{Int}}()
    arcs = Int[]
    islanding = false
    for key in keys
        if !haskey(table.rows, key)
            error("Branch $(key[1]) $(key[2]) has no row in the outage table.")
        end
        is_islanding, range = table.rows[key]
        islanding |= is_islanding
        for i in range
            arc = table.arc_index[i]
            haskey(by_arc, arc) || push!(arcs, arc)
            push!(get!(by_arc, arc, Int[]), i)
        end
    end
    picked = Int[]
    for arc in arcs
        append!(picked, _combined_rows(table, by_arc[arc]))
    end
    return NetworkModification(
        join((key[2] for key in keys), "+"),
        _arc_modifications(
            table.arc_index, table.delta_b, table.delta_shift, table.opened, picked,
        ),
        ShuntModification[],
        islanding,
    )
end

# `rows` holds the rows of one arc.
function _combined_rows(table::BranchOutageTable, rows::Vector{Int})
    for i in rows
        if table.opens_arc[i]
            return [i]
        end
    end
    return unique(i -> table.member[i], rows)
end

function NetworkModification(table::BranchOutageTable, type::String, name::String)
    return NetworkModification(table, [(type, name)])
end

"""
    NetworkModification(table::BranchOutageTable, id::Int)

Build the outage of the branch with the id `id`. The function throws an error if the table
has no row for `id`.
"""
function NetworkModification(table::BranchOutageTable, id::Int)
    haskey(table.ids, id) || error("Branch with id $id has no row in the outage table.")
    return NetworkModification(table, [table.ids[id]])
end

"""
    NetworkModification(table::BranchOutageTable, ids::Vector{Int})

Build the outage of the branches with the ids `ids`. The function throws an error if the
table has no row for an id.
"""
function NetworkModification(table::BranchOutageTable, ids::Vector{Int})
    branch_keys = Tuple{String, String}[]
    for id in ids
        haskey(table.ids, id) || error("Branch with id $id has no row in the outage table.")
        push!(branch_keys, table.ids[id])
    end
    return NetworkModification(table, branch_keys)
end

function _write_row_arrays(
    g::HDF5.Group,
    mods::Vector{NetworkModification},
    n_arc::Int,
)
    starts = Int[]
    lens = Int[]
    arc_index = Int[]
    delta_b = Float64[]
    delta_shift = Float64[]
    opened = Int[]
    for mod in mods
        push!(starts, length(arc_index) + 1)
        push!(lens, length(mod.arc_modifications))
        for m in mod.arc_modifications
            push!(arc_index, m.arc_index)
            push!(delta_b, m.delta_b)
            push!(delta_shift, m.delta_shift_injection)
            push!(opened, m.opened)
        end
    end
    _check_row_values(HDF5.name(g), arc_index, delta_b, delta_shift, opened, n_arc)
    g["start"] = starts
    g["len"] = lens
    g["arc_index"] = arc_index
    g["delta_b"] = delta_b
    g["delta_shift"] = delta_shift
    g["opened"] = opened
    return
end

# `extra` holds the arc-row columns that only one group stores. They join the length check.
function _read_row_arrays(
    g::HDF5.Group,
    n_arc::Int,
    budget::ReadBudget,
    extra::Pair{String, <:AbstractVector}...,
)
    starts = _read(g, "start", Vector{Int}, budget)
    lens = _read(g, "len", Vector{Int}, budget)
    arc_index = _read(g, "arc_index", Vector{Int}, budget)
    delta_b = _read(g, "delta_b", Vector{Float64}, budget)
    delta_shift = _read(g, "delta_shift", Vector{Float64}, budget)
    opened = _read(g, "opened", Vector{Int}, budget)
    _check_same_length("rows", "start" => starts, "len" => lens)
    _check_same_length(
        "arc rows",
        "arc_index" => arc_index,
        "delta_b" => delta_b,
        "delta_shift" => delta_shift,
        "opened" => opened,
        extra...,
    )
    _check_row_values(HDF5.name(g), arc_index, delta_b, delta_shift, opened, n_arc)
    n_rows = length(arc_index)
    _check_row_ranges(HDF5.name(g), starts, lens, n_rows)
    ranges = [s:(s + l - 1) for (s, l) in zip(starts, lens)]
    for (k, range) in enumerate(ranges)
        (isempty(range) || checkbounds(Bool, arc_index, range)) ||
            error(
                "Factor file row range $k ($range) is outside the $(length(arc_index)) arc rows.",
            )
    end
    # Overlapping ranges would let a small file allocate one vector per range and row.
    next = 1
    for k in eachindex(ranges)
        starts[k] == next || error(
            "Factor file row range $k starts at $(starts[k]); " *
            "ranges must be contiguous and ordered, so it must start at $next.",
        )
        next += lens[k]
    end
    next == n_rows + 1 || error(
        "Factor file row ranges cover $(next - 1) of the $n_rows arc rows.",
    )
    for (k, range) in enumerate(ranges)
        seen = Set{Int}()
        for row in range
            arc_index[row] in seen && error(
                "Factor file row range $k has the arc_index $(arc_index[row]) more than once, at row $row.",
            )
            push!(seen, arc_index[row])
        end
    end
    return ranges, (; arc_index, delta_b, delta_shift, opened)
end

# The outage readers do not build a `FactorCore`, so they read the arc susceptances.
function _read_arc_susceptances(f::HDF5.File, budget::ReadBudget)
    arc_susceptances = _read(f, "arc_susceptances", Vector{Float64}, budget)
    n_arc = length(_read(f, "arc_from", Vector{Int}, budget))
    length(arc_susceptances) == n_arc || error(
        "Factor file entry arc_susceptances has $(length(arc_susceptances)) values for $n_arc arcs.",
    )
    return arc_susceptances
end

"""
    read_branch_outages(path::AbstractString) -> BranchOutageTable

Read the group `branch_outages` of a factor file. The function does not factorize the
network. It throws an error if the file has no such group. The `type` of a key is a `String`
that the function does not interpret.
"""
function read_branch_outages(path::AbstractString)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        if !haskey(f, "branch_outages")
            error(
                "File $path has no branch outage table. " *
                "The file was written without a system.",
            )
        end
        g = _group(f, "branch_outages")
        arc_susceptances = _read_arc_susceptances(f, budget)
        n_arc = length(arc_susceptances)
        opens_arc = _read(g, "opens_arc", Vector{Bool}, budget)
        member = _read(g, "member", Vector{Int}, budget)
        ranges, arrays = _read_row_arrays(
            g,
            n_arc,
            budget,
            "opens_arc" => opens_arc,
            "member" => member,
        )
        _check_outage_arcs(arrays.arc_index, arc_susceptances)
        # A row that does not open its arc removes one group member, and the reader keys it
        # by that member.
        for (row, m) in enumerate(member)
            opens_arc[row] || m >= 1 ||
                error(
                    "Factor file entry branch_outages/member has the value $m at row $row, " *
                    "a partial row; expected 1 or more.",
                )
        end
        types = _read(g, "type", Vector{String}, budget)
        names = _read(g, "name", Vector{String}, budget)
        ids = _read(g, "id", Vector{Int}, budget)
        is_islanding = _read(g, "is_islanding", Vector{Bool}, budget)
        _check_same_length(
            "branch_outages",
            "start" => ranges,
            "type" => types,
            "name" => names,
            "id" => ids,
            "is_islanding" => is_islanding,
        )
        _check_unique("branch_outages/id", ids)
        _check_unique("branch_outages/(type, name)", collect(zip(types, names)))
        rows = Dict(zip(zip(types, names), zip(is_islanding, ranges)))
        id_keys = Dict(id => (t, n) for (id, t, n) in zip(ids, types, names))
        return BranchOutageTable(
            rows,
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            opens_arc,
            member,
            id_keys,
            n_arc,
        )
    end
end

function _read_registered_outages!(
    f::HDF5.File,
    cache::Dict{Int, ContingencySpec},
    arc_susceptances::Vector{Float64},
    budget::ReadBudget,
)
    g = _group(f, "outages")
    ranges, arrays = _read_row_arrays(g, length(arc_susceptances), budget)
    _check_outage_arcs(arrays.arc_index, arc_susceptances)
    ids = _read(g, "id", Vector{Int}, budget)
    labels = _read(g, "label", Vector{String}, budget)
    is_islanding = _read(g, "is_islanding", Vector{Bool}, budget)
    _check_same_length(
        "outages",
        "start" => ranges,
        "id" => ids,
        "label" => labels,
        "is_islanding" => is_islanding,
    )
    _check_unique("outages/id", ids)
    for (id, label, islanding, range) in zip(ids, labels, is_islanding, ranges)
        mods = _arc_modifications(
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            range,
        )
        cache[id] = ContingencySpec(
            id,
            NetworkModification(label, mods, ShuntModification[], islanding),
        )
    end
    return
end

"""
    read_registered_outages(path::AbstractString) -> Dict{Int, ContingencySpec}

Read the group `outages` of a factor file: the registered contingencies, each under its id.
The result is empty if the file has no such group. The function does not factorize the
network. The modifications have no Ybus deltas and no shunt modifications. They serve the DC
row kernels only.
"""
function read_registered_outages(path::AbstractString)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        cache = Dict{Int, ContingencySpec}()
        if haskey(f, "outages")
            _read_registered_outages!(f, cache, _read_arc_susceptances(f, budget), budget)
        end
        return cache
    end
end
