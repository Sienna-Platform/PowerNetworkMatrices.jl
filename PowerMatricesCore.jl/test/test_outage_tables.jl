import HDF5

function _post_outage_rows(core, mod)
    wf = PMC.compute_woodbury_factors(core, mod)
    return [
        PMC.apply_woodbury_correction(core, e, wf) for
        e in 1:length(PMC.get_arc_axis(core))
    ]
end

@testset "Outage tables: branch outages round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "outages.h5")
        core, mods = _write_outage_file(path)
        table = PMC.read_branch_outages(path)
        @test table.n_arc == 6
        @test sort!(collect(keys(PMC.get_outage_rows(table)))) ==
              sort!([("Line", n) for n in _OUTAGE_NAMES])
        @test PMC.get_outage_ids(table) ==
              Dict(id => ("Line", n) for (id, n) in zip(_OUTAGE_IDS, _OUTAGE_NAMES))
        @test PMC.get_outage_opens_arc(table) == [false, false, true, true, true, true]
        @test PMC.get_outage_member(table) == [1, 2, 1, 0, 0, 0]
        restored = PMC.read_factor_core(path; linear_solver = "KLU")
        for (name, id, mod) in zip(_OUTAGE_NAMES, _OUTAGE_IDS, mods)
            from_table = PMC.NetworkModification(table, "Line", name)
            @test from_table == mod
            @test from_table.label == name
            @test PMC.NetworkModification(table, id) == mod
            @test _post_outage_rows(restored, from_table) == _post_outage_rows(core, mod)
        end
    end
end

@testset "Outage tables: several branches in one outage" begin
    mktempdir() do dir
        path = joinpath(dir, "outages.h5")
        core, mods = _write_outage_file(path)
        table = PMC.read_branch_outages(path)
        key(name) = ("Line", name)
        # The members of a group do not add without the core.
        @test_throws "trips 2 members of the parallel group on arc 1. Pass the factor core" PMC.NetworkModification(
            table,
            [key("a"), key("b")],
        )
        both = PMC.NetworkModification(table, [key("a"), key("b")], core)
        @test both.label == "a+b"
        @test only(both.arc_modifications).delta_b == -10.0
        @test only(both.arc_modifications).opened == 2
        # A lossy group: the member rows sum to -10.0, but the arc holds 12.0. Opening the
        # arc removes the arc susceptance, not the sum.
        lossy = _five_bus_core(;
            arc_live_members = [2, 1, 1, 1, 1, 1],
            b = [12.0, 5.0, 8.0, 4.0, 6.0, 7.0],
        )
        @test only(
            PMC.NetworkModification(table, [key("a"), key("b")], lossy).arc_modifications,
        ).delta_b == -12.0
        # Two of three members do not open the arc, and the rows cannot combine.
        three = _five_bus_core(; arc_live_members = [3, 1, 1, 1, 1, 1])
        @test_throws "trips 2 of the 3 members of the parallel group on arc 1 and does not open the arc" PMC.NetworkModification(
            table,
            [key("a"), key("b")],
            three,
        )
        # One member with the core gives the row of the table.
        @test PMC.NetworkModification(table, [key("a")], three) == mods[1]
        # A row that opens the arc replaces the partial rows of that arc.
        @test PMC.NetworkModification(table, [key("a"), key("ab")]) == mods[3]
        # Two rows that open one arc count one time. The islanding flags combine.
        cd = PMC.NetworkModification(table, [key("c"), key("d")])
        @test cd == mods[5]
        @test cd.is_islanding
        @test_throws "at least one branch key" PMC.NetworkModification(
            table,
            Tuple{String, String}[],
        )
        @test_throws "Duplicate branch key" PMC.NetworkModification(
            table,
            [key("a"), key("a")],
        )
        @test_throws "has no row in the outage table" PMC.NetworkModification(
            table,
            "Line",
            "x",
        )
        @test_throws "id 99 has no row" PMC.NetworkModification(table, 99)
    end
end

@testset "Outage tables: registered outages round trip" begin
    mktempdir() do dir
        path = joinpath(dir, "outages.h5")
        _, mods = _write_outage_file(path)
        specs = PMC.read_registered_outages(path)
        @test sort!(collect(keys(specs))) == [21, 22]
        @test specs[21].id == 21
        @test specs[21].modification == mods[4]
        @test specs[21].modification.label == "first"
        @test specs[22].modification == mods[5]
        @test specs[22].modification.is_islanding
        bare = joinpath(dir, "bare.h5")
        _write_five_bus(bare)
        @test isempty(PMC.read_registered_outages(bare))
        @test_throws "no branch outage table" PMC.read_branch_outages(bare)
    end
end

@testset "Outage tables: writers apply the rules of the readers" begin
    mods = _outage_mods(_five_bus_core())
    partial = mods[1:2]
    lines = ["Line", "Line"]
    mktempdir() do dir
        HDF5.h5open(joinpath(dir, "bad.h5"), "w") do f
            @test_throws "branch_outages/(type, name)" PMC.write_branch_outages!(
                f, lines, ["a", "a"], [1, 2], partial, [false, false], [1, 2], 6,
            )
            @test_throws "branch_outages/id" PMC.write_branch_outages!(
                f, lines, ["a", "b"], [1, 1], partial, [false, false], [1, 2], 6,
            )
            @test_throws "a partial row; expected 1 or more" PMC.write_branch_outages!(
                f, lines, ["a", "b"], [1, 2], partial, [false, false], [1, 0], 6,
            )
            @test_throws "outages/id" PMC.write_registered_outages!(f, [1, 1], partial, 6)
            @test_throws "names/(type, name)" PMC.write_branch_names!(
                f, lines, ["x", "x"], [1, 1], [2, 2],
            )
            good = _six_arc_table()
            bad = PMC.BranchArcTable(
                [12; good.kind[2:end]],
                good.id,
                good.parent,
                good.sign,
                good.b,
                good.alpha,
                good.n_arc,
            )
            @test_throws "kind is not in 1:11" PMC._write_branch_map!(f, bad)
            @test isempty(keys(f))
        end
    end
end

@testset "Outage tables: hostile input" begin
    for (group, reader) in (
        ("branch_outages", PMC.read_branch_outages),
        ("outages", PMC.read_registered_outages),
    )
        for (change!, expected) in (
            (g -> _replace_entry!(g, "arc_index", fill(7, length(g["arc_index"]))),
                "expected a value in 1:6"),
            (g -> _replace_entry!(g, "delta_b", fill(NaN, length(g["delta_b"]))),
                "must be finite, not zero"),
            (g -> _replace_entry!(g, "delta_shift", fill(Inf, length(g["delta_shift"]))),
                "the value must be finite"),
            (g -> _replace_entry!(g, "opened", fill(-1, length(g["opened"]))),
                "must not be negative"),
            (g -> _replace_entry!(g, "len", zeros(Int, length(g["len"]))),
                "contiguous and ordered"),
            (g -> _replace_entry!(g, "delta_b", [1.0]), "columns of unequal length"),
            (g -> _replace_entry!(g, "id", fill(7, length(g["id"]))), "more than once"),
            (g -> _replace_entry!(g, "id", Float64[1.0]), "has the HDF5 class"),
        )
            _hostile(f -> change!(f[group]), expected, reader, _write_outage_file)
        end
        _hostile(
            "entry /$group is a dataset; expected a group",
            reader,
            _write_outage_file,
        ) do f
            HDF5.delete_object(f, group)
            f[group] = [1, 2]
        end
        _hostile("arc_susceptances entry is zero", reader, _write_outage_file) do f
            b = read(f["arc_susceptances"])
            b[4] = 0.0
            _replace_entry!(f, "arc_susceptances", b)
        end
        _hostile("Unsupported factor file format version", reader, _write_outage_file) do f
            HDF5.delete_attribute(f, "format_version")
            HDF5.attributes(f)["format_version"] = PMC.FACTOR_FILE_FORMAT_VERSION + 1
        end
    end
    _hostile("columns of unequal length", PMC.read_branch_outages, _write_outage_file) do f
        _replace_entry!(f["branch_outages"], "member", [1, 2])
    end
    _hostile(
        "branch_outages/member has the value 0 at row 1",
        PMC.read_branch_outages,
        _write_outage_file,
    ) do f
        _replace_entry!(f["branch_outages"], "member", zeros(Int, 6))
    end
    _hostile("more than once", PMC.read_branch_outages, _write_outage_file) do f
        HDF5.delete_object(f["branch_outages"], "name")
        PMC._write_strings!(f["branch_outages"], "name", fill("a", 5))
    end
end
