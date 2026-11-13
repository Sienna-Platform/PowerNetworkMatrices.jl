import HDF5

function _reduction_table()
    return PMC.BranchReductionTable(
        Dict(
            (101, 0) => (PMC.BRANCH_REDUCTION_HAS_ARC, 0),
            (102, 0) => (PMC.BRANCH_REDUCTION_RADIAL, 7),
            (103, 0) => (PMC.BRANCH_REDUCTION_ZERO_IMPEDANCE, 0),
            (104, 0) => (PMC.BRANCH_REDUCTION_WARD, 0),
            (105, 0) => (PMC.BRANCH_REDUCTION_NOT_IN_NETWORK, 0),
            (106, 2) => (PMC.BRANCH_REDUCTION_HAS_ARC, 0),
            (107, 3) => (PMC.BRANCH_REDUCTION_RADIAL, 9),
        ),
    )
end

@testset "Branch reductions: round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        @test_throws "has no group branch_reduction" PMC.read_branch_reductions(path)
        table = _reduction_table()
        HDF5.h5open(f -> PMC._write_branch_reduction!(f, table), path, "r+")
        restored = PMC.read_branch_reductions(path)
        @test restored.rows == table.rows
        @test PMC.branch_reduction(restored, 101) == PMC.BRANCH_REDUCTION_HAS_ARC
        @test PMC.branch_reduction(restored, 102) == PMC.BRANCH_REDUCTION_RADIAL
        @test PMC.branch_reduction(restored, 103) == PMC.BRANCH_REDUCTION_ZERO_IMPEDANCE
        @test PMC.branch_reduction(restored, 104) == PMC.BRANCH_REDUCTION_WARD
        @test PMC.branch_reduction(restored, 105) == PMC.BRANCH_REDUCTION_NOT_IN_NETWORK
        @test PMC.branch_reduction(restored, 106, PMC.SECONDARY) ==
              PMC.BRANCH_REDUCTION_HAS_ARC
        @test PMC.branch_surviving_bus(restored, 102) == 7
        @test PMC.branch_surviving_bus(restored, 107, PMC.TERTIARY) == 9
        @test_throws "The id 106 is a three-winding transformer. Pass a Winding" PMC.branch_reduction(
            restored,
            106,
        )
        @test_throws "no branch with id 102 and winding 1" PMC.branch_reduction(
            restored,
            102,
            PMC.PRIMARY,
        )
        @test_throws "no branch with id 102 and winding 1" PMC.branch_surviving_bus(
            restored,
            102,
            PMC.PRIMARY,
        )
        @test_throws "did not remove the branch with id 106 and winding 2; its code is 0" PMC.branch_surviving_bus(
            restored,
            106,
            PMC.SECONDARY,
        )
        @test_throws "has no branch with id 999." PMC.branch_reduction(
            restored,
            999,
        )
        @test_throws "The id 106 is a three-winding transformer. Pass a Winding" PMC.branch_surviving_bus(
            restored,
            106,
        )
        @test_throws "did not remove the branch with id 101; its code is 0" PMC.branch_surviving_bus(
            restored,
            101,
        )
    end
end

function _write_branch_reduction_file(path)
    _write_five_bus(path)
    HDF5.h5open(f -> PMC._write_branch_reduction!(f, _reduction_table()), path, "r+")
    return
end

# Rows sort by (id, winding): 101 to 105, then (106, 2) and (107, 3).
@testset "Branch reductions: hostile input" begin
    for (name, row, value, expected) in (
        ("id", 3, 102, "has the value (102, 0) more than once"),
        ("id", 1, 0, "id 0 is not positive"),
        ("id", 1, -4, "id -4 is not positive"),
        ("winding", 1, 4, "winding 4 is not in 0:3"),
        ("winding", 1, -1, "winding -1 is not in 0:3"),
        ("code", 1, 5, "code 5 is not in 0:4"),
        ("code", 1, -1, "code -1 is not in 0:4"),
        ("surviving_bus", 2, 0, "surviving_bus is not positive for a radial branch"),
        ("surviving_bus", 2, -3, "surviving_bus is not positive for a radial branch"),
        ("surviving_bus", 1, 7, "surviving_bus is 7; expected 0 for code 0"),
        ("surviving_bus", 4, 2, "surviving_bus is 2; expected 0 for code 3"),
    )
        _hostile(
            f -> _set_column!(f["branch_reduction"], name, row, value),
            expected,
            PMC.read_branch_reductions,
            _write_branch_reduction_file,
        )
    end
    _hostile(
        "columns of unequal length",
        PMC.read_branch_reductions,
        _write_branch_reduction_file,
    ) do f
        _replace_entry!(f["branch_reduction"], "code", [1])
    end
    _hostile(
        "entry /branch_reduction is a dataset; expected a group",
        PMC.read_branch_reductions,
        _write_branch_reduction_file,
    ) do f
        HDF5.delete_object(f, "branch_reduction")
        f["branch_reduction"] = [1, 2]
    end
    _hostile(
        "Unsupported factor file format version",
        PMC.read_branch_reductions,
        _write_branch_reduction_file,
    ) do f
        HDF5.delete_attribute(f, "format_version")
        HDF5.attributes(f)["format_version"] = PMC.FACTOR_FILE_FORMAT_VERSION + 1
    end
end
