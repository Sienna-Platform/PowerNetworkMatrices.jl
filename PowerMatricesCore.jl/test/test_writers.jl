import HDF5

function _group_content(path, group)
    return HDF5.h5open(path, "r") do f
        g = f[group]
        return Dict(k => read(g[k]) for k in keys(g))
    end
end

@testset "Writers: open_factor_file and the handle type" begin
    @test PMC.FactorFile === HDF5.File
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        @test PMC.open_factor_file(f -> haskey(f, "arc_from"), path)
        @test !PMC.open_factor_file(f -> haskey(f, "names"), path)
        @test_throws Exception PMC.open_factor_file(f -> 1, joinpath(dir, "missing.h5"))
        @test_throws Exception PMC.open_factor_file(f -> (f["v"] = 1), path)
    end
end

@testset "Writers: atomic write" begin
    mktempdir() do dir
        path = joinpath(dir, "out.h5")
        put(value) = f -> (f["v"] = value; nothing)
        @test PMC._write_factor_file_atomically(put([1]), path, false) == path
        @test _group_content(path, "/") == Dict("v" => [1])
        @test_throws "exists. Pass `force = true`" PMC._write_factor_file_atomically(
            put([2]),
            path,
            false,
        )
        @test _group_content(path, "/") == Dict("v" => [1])
        @test PMC._write_factor_file_atomically(put([3]), path, true) == path
        @test _group_content(path, "/") == Dict("v" => [3])
        @test_throws "write failed" PMC._write_factor_file_atomically(path, true) do f
            error("write failed")
        end
        @test _group_content(path, "/") == Dict("v" => [3])
        @test readdir(dir) == ["out.h5"]
        @test_throws "not a regular file" PMC._write_factor_file_atomically(
            put([4]),
            dir,
            true,
        )
    end
end

@testset "Writers: names group" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path)
        HDF5.h5open(path, "r+") do f
            PMC.write_branch_names!(f, ["Line", "Line"], ["a", "b"], [1, 2], [2, 4])
        end
        @test PMC.read_branch_names(path) ==
              Dict(("Line", "a") => (1, 2), ("Line", "b") => (2, 4))
        HDF5.h5open(path, "r+") do f
            @test_throws "unequal length" PMC.write_branch_names!(
                f,
                ["Line"],
                ["a", "b"],
                [1],
                [2],
            )
        end
    end
end

function _base_records()
    return (
        applied = [1, 2],
        removed_arc_from = [3],
        removed_arc_to = [98],
        removed_arc_code = [2],
        composite_from = [0],
        composite_to = [0],
        survivor_from = [3],
        survivor_to = [98],
        survivor_bus = [3],
        boundary_bus = [3],
        boundary_from = [3],
        boundary_to = [98],
        removed_buses = [98],
        merged_from = [99],
        merged_to = [1],
        irreducible_buses = [2],
    )
end

const _REDUCED = Dict(99 => 1)
const _BUS_AX = [1, 2, 3, 4, 5]
const _ARC_AX = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]

@testset "Writers: reduction group round trip" begin
    mktempdir() do dir
        rec = _base_records()
        path = joinpath(dir, "core.h5")
        _write_five_bus(path; reverse_bus_map = _REDUCED)
        HDF5.h5open(path, "r+") do f
            PMC.write_reduction_records!(f, rec, _BUS_AX, _ARC_AX, _REDUCED)
        end
        back = PMC.open_factor_file(path) do f
            return PMC.read_reduction_records(
                f,
                _BUS_AX,
                _ARC_AX,
                _REDUCED,
                PMC.ReadBudget(path),
            )
        end
        @test back == rec
        @test Set(keys(_group_content(path, "reduction"))) ==
              Set(string.(PMC._REDUCTION_COLUMNS))
        @test PMC.REDUCTION_CODE_ZERO_IMPEDANCE == 1
        @test PMC.REDUCTION_CODE_RADIAL == 2
        @test PMC.REDUCTION_CODE_DEGREE_TWO == 3
        @test PMC.REDUCTION_CODE_WARD == 4
    end
end

const _HOSTILE_RECORDS = [
    "removed_arc" => (removed_arc_to = Int[],),
    "survivor" => (survivor_bus = Int[],),
    "boundary" => (boundary_bus = Int[],),
    "merged" => (merged_to = Int[],),
    "expected 1 to 4" => (applied = [1, 2, 7],),
    "reduction/applied" => (applied = [1, 1, 2],),
    "Ward reduction before the end" => (applied = [4, 1, 2],),
    "not in applied" => (applied = [1],),
    "reduction/removed_buses" => (removed_buses = [98, 98],),
    "removed_buses holds bus 2, which is on the bus axis" =>
        (removed_buses = [98, 2],),
    "reduction/removed_arc" => (
        removed_arc_from = [3, 3],
        removed_arc_to = [98, 98],
        removed_arc_code = [2, 2],
        composite_from = [0, 0],
        composite_to = [0, 0],
    ),
    "removed_arc_to holds bus 97" => (removed_arc_to = [97],),
    "which is on the arc axis" => (removed_arc_from = [1], removed_arc_to = [2]),
    "is empty for a degree-two arc" =>
        (applied = [2, 3], removed_arc_code = [3]),
    "is not the degree-two code" => (composite_from = [1], composite_to = [2]),
    "neither on the arc axis nor a removed arc" => (
        applied = [2, 3],
        removed_arc_code = [3],
        composite_from = [1],
        composite_to = [5],
    ),
    "holds the degree-two arc" => (
        applied = [2, 3],
        removed_buses = [98, 97],
        removed_arc_from = [3, 3],
        removed_arc_to = [98, 97],
        removed_arc_code = [3, 3],
        composite_from = [1, 3],
        composite_to = [2, 98],
    ),
    "survivor holds arc (1, 2), which is not a removed arc" =>
        (survivor_from = [1], survivor_to = [2]),
    "survivor_bus holds bus 55" => (survivor_bus = [55],),
    "reduction/survivor" => (
        survivor_from = [3, 3],
        survivor_to = [98, 98],
        survivor_bus = [3, 3],
    ),
    "boundary holds arc (1, 2), which is not a removed arc" =>
        (boundary_from = [1], boundary_to = [2]),
    "boundary_bus holds bus 55" => (boundary_bus = [55],),
    "reduction/boundary" => (
        boundary_bus = [3, 3],
        boundary_from = [3, 3],
        boundary_to = [98, 98],
    ),
    "merged_from holds bus 2, which is on the bus axis" => (merged_from = [2],),
    "merged_to holds bus 99 at row 2, which is also in merged_from" =>
        (merged_from = [99, 98], merged_to = [1, 99]),
    "reduction/merged_from" => (merged_from = [99, 99], merged_to = [1, 1]),
    "irreducible_buses holds bus 55" => (irreducible_buses = [55],),
    "reduction/irreducible_buses" => (irreducible_buses = [2, 2],),
]

@testset "Writers: reduction group rejects hostile records" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path; reverse_bus_map = _REDUCED)
        for (expected, change) in _HOSTILE_RECORDS
            rec = merge(_base_records(), change)
            HDF5.h5open(path, "r+") do f
                @test_throws expected PMC.write_reduction_records!(
                    f,
                    rec,
                    _BUS_AX,
                    _ARC_AX,
                    _REDUCED,
                )
                @test !haskey(f, "reduction")
            end
        end
        @test_throws "reduced_bus_to holds bus 77" HDF5.h5open(path, "r+") do f
            PMC.write_reduction_records!(
                f,
                _base_records(),
                _BUS_AX,
                _ARC_AX,
                Dict(99 => 77),
            )
        end
    end
end

@testset "Writers: reduction reader rejects a bad stored column" begin
    mktempdir() do dir
        path = joinpath(dir, "core.h5")
        _write_five_bus(path; reverse_bus_map = _REDUCED)
        HDF5.h5open(path, "r+") do f
            PMC.write_reduction_records!(f, _base_records(), _BUS_AX, _ARC_AX, _REDUCED)
            HDF5.delete_object(f["reduction"], "applied")
            f["reduction"]["applied"] = [1, 2, 7]
        end
        @test_throws "expected 1 to 4" PMC.open_factor_file(path) do f
            return PMC.read_reduction_records(
                f,
                _BUS_AX,
                _ARC_AX,
                _REDUCED,
                PMC.ReadBudget(path),
            )
        end
    end
end

@testset "Writers: outage groups equal the groups of the file layout" begin
    mktempdir() do dir
        reference = joinpath(dir, "reference.h5")
        core, mods = _write_outage_file(reference)
        path = joinpath(dir, "written.h5")
        _write_five_bus(path; arc_live_members = [2, 1, 1, 1, 1, 1])
        n_arc = length(PMC.get_arc_axis(core))
        HDF5.h5open(path, "r+") do f
            PMC.write_branch_outages!(
                f,
                fill("Line", length(mods)),
                _OUTAGE_NAMES,
                _OUTAGE_IDS,
                mods,
                [false, false, true, true, true, true],
                [1, 2, 1, 0, 0, 0],
                n_arc,
            )
            PMC.write_registered_outages!(f, [21, 22], mods[[4, 5]], n_arc)
        end
        @test _group_content(path, "branch_outages") ==
              _group_content(reference, "branch_outages")
        written = _group_content(path, "outages")
        @test written["label"] == ["c", "d"]
        @test delete!(written, "label") ==
              delete!(_group_content(reference, "outages"), "label")
        table = PMC.read_branch_outages(path)
        @test collect(keys(PMC.get_outage_rows(table))) ==
              collect(keys(PMC.get_outage_rows(PMC.read_branch_outages(reference))))
        @test sort!(collect(keys(PMC.read_registered_outages(path)))) == [21, 22]
        HDF5.h5open(path, "r+") do f
            @test_throws "unequal length" PMC.write_registered_outages!(
                f,
                [1],
                mods[[4, 5]],
                n_arc,
            )
            for (opens, members) in (
                ([true], [1, 2, 1, 0, 0, 0]),
                ([false, false, true, true, true, true], [1]),
            )
                @test_throws "unequal length" PMC.write_branch_outages!(
                    f,
                    fill("Line", length(mods)),
                    _OUTAGE_NAMES,
                    _OUTAGE_IDS,
                    mods,
                    opens,
                    members,
                    n_arc,
                )
            end
        end
    end
end
