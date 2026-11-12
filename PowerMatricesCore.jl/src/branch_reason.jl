const NO_ARC_HAS_ARC = 0
const NO_ARC_RADIAL = 1
const NO_ARC_ZERO_IMPEDANCE = 2
const NO_ARC_WARD = 3
const NO_ARC_NOT_IN_NETWORK = 4

"""
    BranchReasonTable

For each branch of a system, the code that tells whether the branch has an arc in a matrix,
and the surviving bus of a branch that the radial reduction absorbed. The key is
(`id`, winding), the key of a leaf row of [`BranchArcTable`](@ref): the integer id of the
branch, and winding 0 for a branch or 1 to 3 for a circuit of a three-winding transformer.

Read it with `read_branch_reasons(path)`. Query it with [`no_arc_reason`](@ref) and
[`no_arc_surviving_bus`](@ref).
"""
struct BranchReasonTable
    rows::Dict{Tuple{Int, Int}, Tuple{Int, Int}}
end

function _branch_reason(t::BranchReasonTable, id::Int, winding::Int)
    haskey(t.rows, (id, winding)) || error(
        "The branch reason table has no branch with id $id and winding $winding. " *
        "It holds the branches of the system that built it.",
    )
    return t.rows[(id, winding)]
end

"""
    no_arc_reason(table::BranchReasonTable, id::Int, winding::Int = 0) -> Int

Tell why the branch with the integer id `id` has or lacks an arc. Use `winding` 1 to 3
for a circuit of a three-winding transformer. The function returns one of these codes:

  - `NO_ARC_HAS_ARC` (0): the branch has an arc, alone or inside a parallel group or series chain.
  - `NO_ARC_RADIAL` (1): the radial reduction absorbed the branch. Call [`no_arc_surviving_bus`](@ref) for the bus that kept it.
  - `NO_ARC_ZERO_IMPEDANCE` (2): the zero-impedance reduction merged both ends into one bus.
  - `NO_ARC_WARD` (3): the Ward reduction eliminated the branch.
  - `NO_ARC_NOT_IN_NETWORK` (4): the branch is out of service, or it touches a bus that the matrix excluded.

The function throws an error if the table has no row for (`id`, `winding`).
"""
no_arc_reason(t::BranchReasonTable, id::Int, winding::Int = 0) =
    first(_branch_reason(t, id, winding))

"""
    no_arc_surviving_bus(table::BranchReasonTable, id::Int, winding::Int = 0) -> Int

The bus that kept the branch when the radial reduction absorbed it, in the numbering after
the later reductions. A later degree-two reduction can remove that bus. The function throws
an error if the table has no row for (`id`, `winding`), or if the radial reduction did not
absorb the branch.
"""
function no_arc_surviving_bus(t::BranchReasonTable, id::Int, winding::Int = 0)
    code, bus = _branch_reason(t, id, winding)
    code == NO_ARC_RADIAL || error(
        "The radial reduction did not remove the branch with id $id and winding $winding; " *
        "its code is $code.",
    )
    return bus
end

function _check_branch_reason(
    id::Vector{Int},
    winding::Vector{Int},
    reason::Vector{Int},
    surviving_bus::Vector{Int},
)
    _check_same_length(
        "branch_reason",
        "id" => id, "winding" => winding, "reason" => reason,
        "surviving_bus" => surviving_bus,
    )
    _check_unique("branch_reason/(id, winding)", collect(zip(id, winding)))
    for r in eachindex(id)
        fail(what) = error("Factor file table branch_reason row $r: $what.")
        id[r] > 0 || fail("id $(id[r]) is not positive")
        0 <= winding[r] <= 3 || fail("winding $(winding[r]) is not in 0:3")
        NO_ARC_HAS_ARC <= reason[r] <= NO_ARC_NOT_IN_NETWORK ||
            fail("reason $(reason[r]) is not in 0:4")
        if reason[r] == NO_ARC_RADIAL
            surviving_bus[r] > 0 ||
                fail("surviving_bus is not positive for a radial branch")
        else
            iszero(surviving_bus[r]) ||
                fail(
                    "surviving_bus is $(surviving_bus[r]); expected 0 for reason $(reason[r])",
                )
        end
    end
    return
end

function _write_branch_reason!(f::HDF5.File, t::BranchReasonTable)
    rows = sort!(collect(t.rows); by = first)
    id = [first(first(r)) for r in rows]
    winding = [last(first(r)) for r in rows]
    reason = [first(last(r)) for r in rows]
    surviving_bus = [last(last(r)) for r in rows]
    _check_branch_reason(id, winding, reason, surviving_bus)
    g = HDF5.create_group(f, "branch_reason")
    g["id"] = id
    g["winding"] = winding
    g["reason"] = reason
    g["surviving_bus"] = surviving_bus
    return
end

"""
    read_branch_reasons(path::AbstractString) -> BranchReasonTable

Read the `branch_reason` group of a factor file. The function throws an error if the file has
no such group, because a file written without a system cannot name its branches. It also
throws an error if the format version is not 1 or a column breaks a rule of the table.
"""
function read_branch_reasons(path::AbstractString)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        haskey(f, "branch_reason") || error(
            "Factor file $path has no group branch_reason: it was written without a " *
            "System. Write it with `to_hdf5(mat, filename, sys)`.",
        )
        g = _group(f, "branch_reason")
        id = _read(g, "id", Vector{Int}, budget)
        winding = _read(g, "winding", Vector{Int}, budget)
        reason = _read(g, "reason", Vector{Int}, budget)
        surviving_bus = _read(g, "surviving_bus", Vector{Int}, budget)
        _check_branch_reason(id, winding, reason, surviving_bus)
        return BranchReasonTable(
            Dict(zip(zip(id, winding), zip(reason, surviving_bus))),
        )
    end
end
