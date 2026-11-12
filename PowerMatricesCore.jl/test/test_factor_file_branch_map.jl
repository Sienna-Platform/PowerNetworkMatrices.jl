import HDF5

# Arc 1 holds a parallel group of two branches; the second one points against the arc. The
# arcs 2 to 5 hold one branch each. Arc 6 holds a series chain of two segments.
function _six_arc_table(; alpha3::Float64 = 0.0)
    return PMC.BranchArcTable(
        [9, 1, 1, 1, 2, 1, 1, 10, 1, 1],
        [0, 101, 102, 103, 104, 105, 106, 0, 107, 108],
        [-1, 1, 1, -2, -3, -4, -5, -6, 8, 8],
        [1, 1, -1, 1, 1, 1, 1, 1, 1, -1],
        [0.0, 4.0, 6.0, 5.0, 8.0, 4.0, 6.0, 0.0, 14.0, 14.0],
        [0.0, 0.0, alpha3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6,
    )
end

const _SIX_ARC_FLOWS = [10.0, 3.0, 1.0, -1.0, 2.0, -2.0]

@testset "Branch map: arc flows split into branch flows" begin
    table = _six_arc_table()
    @test PMC.branch_flows(table, _SIX_ARC_FLOWS) ≈
          [10.0, 4.0, -6.0, 3.0, 1.0, -1.0, 2.0, -2.0, -2.0, 2.0]
    # The live member of the group carries the arc flow.
    @test PMC.branch_flows(table, _SIX_ARC_FLOWS, [2]) ≈
          [10.0, 0.0, -10.0, 3.0, 1.0, -1.0, 2.0, -2.0, -2.0, 2.0]
    # An outaged segment stops the flow of its chain.
    @test PMC.branch_flows(table, _SIX_ARC_FLOWS, [9])[8:10] == zeros(3)
    # A phase shift in one member gives a loop flow of 0.24 in the group.
    shifted = PMC.branch_flows(_six_arc_table(; alpha3 = 0.1), _SIX_ARC_FLOWS)
    @test shifted[1] ≈ 10.0
    @test shifted[2] ≈ 3.76
    @test shifted[3] ≈ -6.24
    @test shifted[4:10] ≈ [3.0, 1.0, -1.0, 2.0, -2.0, -2.0, 2.0]
    @test_throws "arc_flows has 2 values; expected 6" PMC.branch_flows(table, [1.0, 2.0])
    @test_throws "outaged row 11 is not in 1:10" PMC.branch_flows(
        table,
        _SIX_ARC_FLOWS,
        [11],
    )
end

@testset "Branch map: round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        @test_throws "no entry /branch_map" PMC.read_branch_map(path)
        table = _six_arc_table(; alpha3 = 0.1)
        HDF5.h5open(f -> PMC._write_branch_map!(f, table), path, "r+")
        restored = PMC.read_branch_map(path)
        for getter in (
            PMC.get_branch_kind,
            PMC.get_branch_id,
            PMC.get_branch_parent,
            PMC.get_branch_sign,
            PMC.get_branch_susceptance,
            PMC.get_branch_alpha,
            PMC.get_branch_arc_count,
        )
            @test getter(restored) == getter(table)
        end
        @test PMC.branch_flows(restored, _SIX_ARC_FLOWS) ==
              PMC.branch_flows(table, _SIX_ARC_FLOWS)
    end
end

function _write_branch_map_file(path)
    _write_five_bus(path)
    HDF5.h5open(f -> PMC._write_branch_map!(f, _six_arc_table()), path, "r+")
    return
end

@testset "Branch map: hostile input" begin
    for (name, row, value, expected) in (
        ("kind", 1, 12, "kind is not in 1:11"),
        ("kind", 2, 11, "a Ward equivalent must be a top node"),
        ("parent", 2, 0, "parent is 0"),
        ("parent", 2, 3, "parent 3 is not below the row"),
        ("parent", 4, -7, "parent -7 is not an arc of the 6 arcs"),
        ("parent", 3, 2, "parent row 2 has kind 1, not an aggregate"),
        ("parent", 7, -4, "arc 5 has no top node"),
        ("sign", 4, 0, "sign is not 1 or -1"),
        ("id", 4, 0, "id is not positive"),
        ("id", 3, 101, "id and winding (101, 0) repeat"),
        ("id", 1, 5, "id is not 0"),
        ("b", 4, NaN, "b is NaN"),
        ("b", 1, 1.0, "b is not 0"),
        ("alpha", 4, Inf, "alpha is Inf"),
    )
        _hostile(
            f -> _set_column!(f["branch_map"], name, row, value),
            expected,
            PMC.read_branch_map,
            _write_branch_map_file,
        )
    end
    _hostile("row 8 (kind 10): no child", PMC.read_branch_map, _write_branch_map_file) do f
        _set_column!(f["branch_map"], "parent", 9, 1)
        _set_column!(f["branch_map"], "parent", 10, 1)
    end
    _hostile("columns of unequal length", PMC.read_branch_map, _write_branch_map_file) do f
        _replace_entry!(f["branch_map"], "b", [1.0])
    end
    _hostile("has the HDF5 class", PMC.read_branch_map, _write_branch_map_file) do f
        _replace_entry!(f["branch_map"], "kind", ones(10))
    end
    _hostile(
        "entry /branch_map is a dataset; expected a group",
        PMC.read_branch_map,
        _write_branch_map_file,
    ) do f
        HDF5.delete_object(f, "branch_map")
        f["branch_map"] = [1, 2]
    end
    _hostile(
        "Unsupported factor file format version",
        PMC.read_branch_map,
        _write_branch_map_file,
    ) do f
        HDF5.delete_attribute(f, "format_version")
        HDF5.attributes(f)["format_version"] = PMC.FACTOR_FILE_FORMAT_VERSION + 1
    end
end
