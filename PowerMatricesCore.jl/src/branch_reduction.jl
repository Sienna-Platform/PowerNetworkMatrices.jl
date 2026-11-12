const BRANCH_REDUCTION_HAS_ARC = 0
const BRANCH_REDUCTION_RADIAL = 1
const BRANCH_REDUCTION_ZERO_IMPEDANCE = 2
const BRANCH_REDUCTION_WARD = 3
const BRANCH_REDUCTION_NOT_IN_NETWORK = 4

"""
    BranchReductionTable

For each branch of a system, the code that tells whether the branch has an arc in a matrix,
and the surviving bus of a branch that the radial reduction absorbed. The key is
(`id`, winding), the key of a leaf row of [`BranchArcTable`](@ref): the integer id of the
branch, and winding 0 for a branch or 1 to 3 for a circuit of a three-winding transformer.
The table has one row for each branch.

Read it with `read_branch_reductions(path)`. Query it with [`branch_reduction`](@ref) and
[`branch_surviving_bus`](@ref).
"""
struct BranchReductionTable
    rows::Dict{Tuple{Int, Int}, Tuple{Int, Int}}
end

function _branch_reduction_row(t::BranchReductionTable, id::Int)
    key = _branch_key(
        t.rows,
        id,
        "The branch reduction table has no branch with id $id. " *
        "It holds the branches of the system that built it.",
    )
    return t.rows[key]
end

function _branch_reduction_row(t::BranchReductionTable, key::Tuple{Int, Int})
    haskey(t.rows, key) || error(
        "The branch reduction table has no branch with id $(key[1]) and winding $(key[2]). " *
        "It holds the branches of the system that built it.",
    )
    return t.rows[key]
end

"""
    branch_reduction(table::BranchReductionTable, id::Int) -> Int
    branch_reduction(table::BranchReductionTable, id::Int, w::Winding) -> Int

Tell what the network reductions did to the branch with the integer id `id`. Use the first
method for a branch with two terminals. Use the second method for a circuit of a
three-winding transformer: `w` is `PRIMARY`, `SECONDARY` or `TERTIARY`, and `id` is the id of
the transformer. The function returns one of these codes:

  - `BRANCH_REDUCTION_HAS_ARC` (0): the branch has an arc, alone or inside a parallel group or series chain.
  - `BRANCH_REDUCTION_RADIAL` (1): the radial reduction absorbed the branch. Call [`branch_surviving_bus`](@ref) for the bus that kept it.
  - `BRANCH_REDUCTION_ZERO_IMPEDANCE` (2): the zero-impedance reduction merged both ends into one bus.
  - `BRANCH_REDUCTION_WARD` (3): the Ward reduction eliminated the branch.
  - `BRANCH_REDUCTION_NOT_IN_NETWORK` (4): the branch is out of service, or it touches a bus that the matrix excluded.

These codes are the `BRANCH_REDUCTION_*` constants. They are not the `REDUCTION_CODE_*` set
of `NetworkReductionData`: the values differ (the radial code is 1 here and 2 there).

The function throws an error if the table has no row for the branch.
"""
function branch_reduction(t::BranchReductionTable, id::Int)
    return first(_branch_reduction_row(t, id))
end

function branch_reduction(t::BranchReductionTable, id::Int, w::Winding)
    return first(_branch_reduction_row(t, (id, Int(w))))
end

"""
    branch_surviving_bus(table::BranchReductionTable, id::Int) -> Int
    branch_surviving_bus(table::BranchReductionTable, id::Int, w::Winding) -> Int

The bus that kept the branch when the radial reduction absorbed it, in the numbering after
the later reductions. A later degree-two reduction can remove that bus. Use the second
method for a circuit of a three-winding transformer, as in [`branch_reduction`](@ref). The
function throws an error if the table has no row for the branch, or if the radial reduction
did not absorb the branch.
"""
function branch_surviving_bus(t::BranchReductionTable, id::Int)
    code, bus = _branch_reduction_row(t, id)
    code == BRANCH_REDUCTION_RADIAL || error(
        "The radial reduction did not remove the branch with id $id; its code is $code.",
    )
    return bus
end

function branch_surviving_bus(t::BranchReductionTable, id::Int, w::Winding)
    code, bus = _branch_reduction_row(t, (id, Int(w)))
    code == BRANCH_REDUCTION_RADIAL || error(
        "The radial reduction did not remove the branch with id $id and winding $(Int(w)); its code is $code.",
    )
    return bus
end

function _check_branch_reduction(
    id::Vector{Int},
    winding::Vector{Int},
    code::Vector{Int},
    surviving_bus::Vector{Int},
)
    _check_same_length(
        "branch_reduction",
        "id" => id, "winding" => winding, "code" => code,
        "surviving_bus" => surviving_bus,
    )
    _check_unique("branch_reduction/(id, winding)", collect(zip(id, winding)))
    for r in eachindex(id)
        fail(what) = error("Factor file table branch_reduction row $r: $what.")
        id[r] > 0 || fail("id $(id[r]) is not positive")
        0 <= winding[r] <= 3 || fail("winding $(winding[r]) is not in 0:3")
        BRANCH_REDUCTION_HAS_ARC <= code[r] <= BRANCH_REDUCTION_NOT_IN_NETWORK ||
            fail("code $(code[r]) is not in 0:4")
        if code[r] == BRANCH_REDUCTION_RADIAL
            surviving_bus[r] > 0 ||
                fail("surviving_bus is not positive for a radial branch")
        else
            iszero(surviving_bus[r]) ||
                fail(
                    "surviving_bus is $(surviving_bus[r]); expected 0 for code $(code[r])",
                )
        end
    end
    return
end

function _write_branch_reduction!(f::HDF5.File, t::BranchReductionTable)
    rows = sort!(collect(t.rows); by = first)
    id = [first(first(r)) for r in rows]
    winding = [last(first(r)) for r in rows]
    code = [first(last(r)) for r in rows]
    surviving_bus = [last(last(r)) for r in rows]
    _check_branch_reduction(id, winding, code, surviving_bus)
    g = HDF5.create_group(f, "branch_reduction")
    g["id"] = id
    g["winding"] = winding
    g["code"] = code
    g["surviving_bus"] = surviving_bus
    return
end

"""
    read_branch_reductions(path::AbstractString) -> BranchReductionTable

Read the `branch_reduction` group of a factor file. The function throws an error if the file has
no such group, because a file written without a system cannot name its branches. It also
throws an error if the format version is not 1 or a column breaks a rule of the table.
"""
function read_branch_reductions(path::AbstractString)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        haskey(f, "branch_reduction") || error(
            "Factor file $path has no group branch_reduction: it was written without a " *
            "System. Write it with `to_hdf5(mat, filename, sys)`.",
        )
        g = _group(f, "branch_reduction")
        id = _read(g, "id", Vector{Int}, budget)
        winding = _read(g, "winding", Vector{Int}, budget)
        code = _read(g, "code", Vector{Int}, budget)
        surviving_bus = _read(g, "surviving_bus", Vector{Int}, budget)
        _check_branch_reduction(id, winding, code, surviving_bus)
        return BranchReductionTable(
            Dict(zip(zip(id, winding), zip(code, surviving_bus))),
        )
    end
end
