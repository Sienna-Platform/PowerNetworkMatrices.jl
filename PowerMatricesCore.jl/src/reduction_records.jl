const REDUCTION_CODE_ZERO_IMPEDANCE = 1
const REDUCTION_CODE_RADIAL = 2
const REDUCTION_CODE_DEGREE_TWO = 3
const REDUCTION_CODE_WARD = 4

const _REDUCTION_COLUMNS = (
    :applied,
    :removed_arc_from, :removed_arc_to, :removed_arc_code,
    :composite_from, :composite_to,
    :survivor_from, :survivor_to, :survivor_bus,
    :boundary_bus, :boundary_from, :boundary_to,
    :removed_buses,
    :merged_from, :merged_to,
    :irreducible_buses,
)

const _ReductionRecords = NamedTuple{_REDUCTION_COLUMNS, NTuple{16, Vector{Int}}}

function _check_known_bus(name::String, bus::Int, i::Int, known::Set{Int})
    bus in known || error(
        "Factor file entry reduction/$name holds bus $bus at row $i, which is not on the bus " *
        "axis, in the reduced-bus map or in removed_buses.",
    )
    return
end

function _check_known_buses(name::String, buses::Vector{Int}, known::Set{Int})
    for (i, bus) in enumerate(buses)
        _check_known_bus(name, bus, i, known)
    end
    return
end

# One validator serves the writer and the reader.
function _check_reduction_records(
    rec::_ReductionRecords,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
)
    _check_same_length(
        "reduction removed_arc",
        "removed_arc_from" => rec.removed_arc_from,
        "removed_arc_to" => rec.removed_arc_to,
        "removed_arc_code" => rec.removed_arc_code,
        "composite_from" => rec.composite_from,
        "composite_to" => rec.composite_to,
    )
    _check_same_length(
        "reduction survivor",
        "survivor_from" => rec.survivor_from,
        "survivor_to" => rec.survivor_to,
        "survivor_bus" => rec.survivor_bus,
    )
    _check_same_length(
        "reduction boundary",
        "boundary_bus" => rec.boundary_bus,
        "boundary_from" => rec.boundary_from,
        "boundary_to" => rec.boundary_to,
    )
    _check_same_length(
        "reduction merged",
        "merged_from" => rec.merged_from,
        "merged_to" => rec.merged_to,
    )
    for code in rec.applied
        1 <= code <= 4 ||
            error("Factor file entry reduction/applied holds code $code; expected 1 to 4.")
    end
    _check_unique("reduction/applied", rec.applied)
    iszero(count(==(REDUCTION_CODE_WARD), rec.applied)) ||
        last(rec.applied) == REDUCTION_CODE_WARD ||
        error("Factor file entry reduction/applied has the Ward reduction before the end.")
    for (i, code) in enumerate(rec.removed_arc_code)
        code in rec.applied || error(
            "Factor file entry reduction/removed_arc_code row $i holds code $code, " *
            "which is not in applied.",
        )
    end
    removed_buses = Set(rec.removed_buses)
    _check_unique("reduction/removed_buses", rec.removed_buses)
    bus_set = Set(bus_ax)
    for bus in rec.removed_buses
        bus in bus_set && error(
            "Factor file entry reduction/removed_buses holds bus $bus, which is on the bus axis.",
        )
    end
    known = union(bus_set, keys(reduced), removed_buses)
    for root in values(reduced)
        root in bus_set || root in removed_buses ||
            error(
                "Factor file entry reduced_bus_to holds bus $root, which is not on the bus axis or in removed_buses.",
            )
    end
    removed_arcs = collect(zip(rec.removed_arc_from, rec.removed_arc_to))
    _check_unique("reduction/removed_arc", removed_arcs)
    _check_known_buses("removed_arc_from", rec.removed_arc_from, known)
    _check_known_buses("removed_arc_to", rec.removed_arc_to, known)
    removed_set = Set(removed_arcs)
    arc_set = Set(arc_ax)
    for (i, arc) in enumerate(removed_arcs)
        (arc in arc_set || (arc[2], arc[1]) in arc_set) && error(
            "Factor file entry reduction/removed_arc row $i holds arc $arc, which is on the arc axis.",
        )
    end
    code_of_arc = Dict(zip(removed_arcs, rec.removed_arc_code))
    for (i, code) in enumerate(rec.removed_arc_code)
        pair = (rec.composite_from[i], rec.composite_to[i])
        if code == REDUCTION_CODE_DEGREE_TWO
            (iszero(pair[1]) || iszero(pair[2])) && error(
                "Factor file entry reduction/composite row $i is empty for a degree-two arc.",
            )
            _check_known_bus("composite_from", pair[1], i, known)
            _check_known_bus("composite_to", pair[2], i, known)
            any(key -> key in arc_set || key in removed_set, (pair, (pair[2], pair[1]))) ||
                error(
                    "Factor file entry reduction/composite row $i holds arc $pair, which " *
                    "is neither on the arc axis nor a removed arc.",
                )
            # One degree-two step exists, so a composite arc is never a degree-two member.
            # This also stops the reduction lookup from following a loop.
            for key in (pair, (pair[2], pair[1]))
                get(code_of_arc, key, 0) == REDUCTION_CODE_DEGREE_TWO && error(
                    "Factor file entry reduction/composite row $i holds the degree-two arc $key.",
                )
            end
        else
            pair == (0, 0) || error(
                "Factor file entry reduction/composite row $i holds an arc, but code $code is not the degree-two code.",
            )
        end
    end
    survivor_arcs = collect(zip(rec.survivor_from, rec.survivor_to))
    _check_unique("reduction/survivor", survivor_arcs)
    _check_known_buses("survivor_bus", rec.survivor_bus, known)
    for arc in survivor_arcs
        arc in removed_set || error(
            "Factor file entry reduction/survivor holds arc $arc, which is not a removed arc.",
        )
    end
    boundary = collect(zip(rec.boundary_bus, rec.boundary_from, rec.boundary_to))
    _check_unique("reduction/boundary", boundary)
    _check_known_buses("boundary_bus", rec.boundary_bus, known)
    for (bus, from, to) in boundary
        (from, to) in removed_set || error(
            "Factor file entry reduction/boundary holds arc $((from, to)), which is not a removed arc.",
        )
    end
    _check_unique("reduction/merged_from", rec.merged_from)
    for bus in rec.merged_from
        bus in bus_set && error(
            "Factor file entry reduction/merged_from holds bus $bus, which is on the bus axis.",
        )
    end
    _check_known_buses("merged_from", rec.merged_from, known)
    _check_known_buses("merged_to", rec.merged_to, known)
    merged_from_set = Set(rec.merged_from)
    for (i, bus) in enumerate(rec.merged_to)
        bus in merged_from_set && error(
            "Factor file entry reduction/merged_to holds bus $bus at row $i, which is also in merged_from.",
        )
    end
    _check_unique("reduction/irreducible_buses", rec.irreducible_buses)
    _check_known_buses("irreducible_buses", rec.irreducible_buses, known)
    return
end

"""
    write_reduction_records!(f::FactorFile, rec, bus_ax, arc_ax, reduced)

Write the group `reduction` of the factor file `f` from the records `rec`. The function checks
`rec` against the bus axis `bus_ax`, the arc axis `arc_ax` and the map `reduced` of reduced
buses first, and it throws an error if a column breaks a rule of the group.
"""
function write_reduction_records!(
    f::FactorFile,
    rec::_ReductionRecords,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
)
    _check_reduction_records(rec, bus_ax, arc_ax, reduced)
    g = HDF5.create_group(f, "reduction")
    for name in _REDUCTION_COLUMNS
        g[string(name)] = rec[name]
    end
    return
end

"""
    read_reduction_records(f::FactorFile, bus_ax, arc_ax, reduced, budget::ReadBudget)

Read the group `reduction` of the factor file `f` and return its columns as a `NamedTuple` of
`Vector{Int}`. The function checks the columns against the bus axis `bus_ax`, the arc axis
`arc_ax` and the map `reduced` of reduced buses, and it throws an error if a column breaks a
rule of the group or if the file has no such group.
"""
function read_reduction_records(
    f::FactorFile,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
    budget::ReadBudget,
)
    g = _group(f, "reduction")
    rec = NamedTuple{_REDUCTION_COLUMNS}(
        map(name -> _read(g, string(name), Vector{Int}, budget), _REDUCTION_COLUMNS),
    )
    _check_reduction_records(rec, bus_ax, arc_ax, reduced)
    return rec
end
