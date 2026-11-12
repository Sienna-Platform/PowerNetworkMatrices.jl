import HDF5

function _reason_table()
    return PMC.BranchReasonTable(
        Dict(
            (101, 0) => (PMC.NO_ARC_HAS_ARC, 0),
            (102, 0) => (PMC.NO_ARC_RADIAL, 7),
            (103, 0) => (PMC.NO_ARC_ZERO_IMPEDANCE, 0),
            (104, 0) => (PMC.NO_ARC_WARD, 0),
            (105, 0) => (PMC.NO_ARC_NOT_IN_NETWORK, 0),
            (106, 2) => (PMC.NO_ARC_HAS_ARC, 0),
        ),
    )
end

@testset "Branch reasons: round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        @test_throws "has no group branch_reason" PMC.read_branch_reasons(path)
        table = _reason_table()
        HDF5.h5open(f -> PMC._write_branch_reason!(f, table), path, "r+")
        restored = PMC.read_branch_reasons(path)
        @test restored.rows == table.rows
        @test PMC.no_arc_reason(restored, 101) == PMC.NO_ARC_HAS_ARC
        @test PMC.no_arc_reason(restored, 102) == PMC.NO_ARC_RADIAL
        @test PMC.no_arc_reason(restored, 103) == PMC.NO_ARC_ZERO_IMPEDANCE
        @test PMC.no_arc_reason(restored, 104) == PMC.NO_ARC_WARD
        @test PMC.no_arc_reason(restored, 105) == PMC.NO_ARC_NOT_IN_NETWORK
        @test PMC.no_arc_reason(restored, 106, 2) == PMC.NO_ARC_HAS_ARC
        @test PMC.no_arc_surviving_bus(restored, 102) == 7
        @test_throws "no branch with id 999 and winding 0" PMC.no_arc_reason(restored, 999)
        @test_throws "no branch with id 106 and winding 0" PMC.no_arc_surviving_bus(
            restored,
            106,
        )
        @test_throws "did not remove the branch with id 101 and winding 0; its code is 0" PMC.no_arc_surviving_bus(
            restored,
            101,
        )
    end
end

function _hostile_branch_reason(change!, expected::String)
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        HDF5.h5open(f -> PMC._write_branch_reason!(f, _reason_table()), path, "r+")
        HDF5.h5open(change!, path, "r+")
        @test_throws expected PMC.read_branch_reasons(path)
    end
    return
end

function _set_reason_column!(f, name::String, row::Int, value)
    g = f["branch_reason"]
    column = read(g[name])
    column[row] = value
    _replace_entry!(g, name, column)
    return
end

# Rows sort by (id, winding): 101, 102, 103, 104, 105, then (106, 2).
@testset "Branch reasons: hostile input" begin
    for (name, row, value, expected) in (
        ("id", 3, 102, "has the value (102, 0) more than once"),
        ("id", 1, 0, "id 0 is not positive"),
        ("id", 1, -4, "id -4 is not positive"),
        ("winding", 1, 4, "winding 4 is not in 0:3"),
        ("winding", 1, -1, "winding -1 is not in 0:3"),
        ("reason", 1, 5, "reason 5 is not in 0:4"),
        ("reason", 1, -1, "reason -1 is not in 0:4"),
        ("surviving_bus", 2, 0, "surviving_bus is not positive for a radial branch"),
        ("surviving_bus", 2, -3, "surviving_bus is not positive for a radial branch"),
        ("surviving_bus", 1, 7, "surviving_bus is 7; expected 0 for reason 0"),
        ("surviving_bus", 4, 2, "surviving_bus is 2; expected 0 for reason 3"),
    )
        _hostile_branch_reason(f -> _set_reason_column!(f, name, row, value), expected)
    end
    _hostile_branch_reason("columns of unequal length") do f
        _replace_entry!(f["branch_reason"], "reason", [1])
    end
    _hostile_branch_reason("entry /branch_reason is a dataset; expected a group") do f
        HDF5.delete_object(f, "branch_reason")
        f["branch_reason"] = [1, 2]
    end
    _hostile_branch_reason("Unsupported factor file format version") do f
        HDF5.delete_attribute(f, "format_version")
        HDF5.attributes(f)["format_version"] = 2
    end
end
