import HDF5
import Random

@testset "Queries by id: branch shares" begin
    table = _six_arc_table()
    @test PMC.branch_shares(table) ≈ [1, 0.4, -0.6, 1, 1, 1, 1, 1, 1, -1]
    @test PMC.branch_shares(table, [2])[1:3] ≈ [1, 0, -1]
    @test PMC.branch_shares(table, [9])[8:10] == zeros(3)
    # An outaged group makes its members dead.
    @test PMC.branch_shares(table, [1])[1:3] == zeros(3)
    @test PMC.branch_shares(_six_arc_table(; alpha3 = 0.1)) ≈ PMC.branch_shares(table)
    @test_throws "branch_shares: outaged row 11 is not in 1:10" PMC.branch_shares(
        table,
        [11],
    )
end

@testset "Queries by id: shares match branch flows" begin
    t = _six_arc_table(; alpha3 = 0.1)
    rng = Random.MersenneTwister(11)
    arcs = PMC._row_arcs(t)
    @test arcs == [1, 1, 1, 2, 3, 4, 5, 6, 6, 6]
    for out in (Int[], [1], [2], [3], [9], [2, 9])
        F = randn(rng, 6)
        @test PMC.branch_flows(t, F, out) ≈
              PMC.branch_shares(t, out) .* F[arcs] .+ PMC.branch_flows(t, zeros(6), out)
    end
end

@testset "Queries by id: zero susceptance and zero chain sum" begin
    flat = PMC.BranchArcTable([1], [101], [-1], [1], [0.0], [0.0], 1)
    @test PMC.branch_shares(flat) == [0.0]
    chain = PMC.BranchArcTable(
        [10, 1, 1],
        [0, 1, 2],
        [-1, 1, 1],
        [1, 1, 1],
        [0.0, 4.0, -4.0],
        zeros(3),
        1,
    )
    @test_throws "branch_shares: row 1 (series chain) has a zero impedance sum" PMC.branch_shares(
        chain,
    )
end

@testset "Queries by id: outage of several ids" begin
    mktempdir() do dir
        path = joinpath(dir, "outages.h5")
        _write_outage_file(path)
        table = PMC.read_branch_outages(path)
        ids = [11, 14]
        @test PMC.NetworkModification(table, ids) ==
              PMC.NetworkModification(table, [("Line", "a"), ("Line", "c")])
        @test_throws "Branch with id 99 has no row in the outage table." PMC.NetworkModification(
            table,
            [11, 99],
        )
    end
end

const _QUERY_IDS = [101, 102, 13, 105, 106]

function _write_query_file(
    path;
    with_outages::Bool = true,
    reverse_bus_map = Dict{Int, Int}(),
    tol::Float64 = 0.0,
    table = _six_arc_table(),
)
    core = _write_five_bus(
        path;
        arc_live_members = [2, 1, 1, 1, 1, 1],
        reverse_bus_map = reverse_bus_map,
        tol = tol,
    )
    HDF5.h5open(f -> PMC._write_branch_map!(f, table), path, "r+")
    if with_outages
        mods = _outage_mods(core)
        n_arc = length(PMC.get_arc_axis(core))
        HDF5.h5open(path, "r+") do f
            g = HDF5.create_group(f, "branch_outages")
            PMC._write_strings!(g, "type", fill("Line", length(mods)))
            PMC._write_strings!(g, "name", _OUTAGE_NAMES)
            g["id"] = _QUERY_IDS
            g["is_islanding"] = [m.is_islanding for m in mods]
            PMC._write_row_arrays(g, mods, n_arc)
            g["opens_arc"] = [false, false, true, true, true, true]
            g["member"] = [1, 2, 1, 0, 0, 0]
        end
    end
    return core
end

# The distributed slack weights of `_write_five_bus`, normalized.
const _QUERY_SLACK = [0.0, 1.0, 1.0, 0.0, 2.0] ./ 4.0

function _slack_row(core, arc::Int)
    return PMC.apply_dist_slack!(PMC.ptdf_row(core, arc), core, _QUERY_SLACK)
end

@testset "Queries by id: rows without a contingency" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        core = _write_query_file(path)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test x.has_outages
        @test PMC.get_core(x) === x.core
        @test PMC.get_bus_axis(x) == [1, 2, 3, 4, 5]
        @test x.dist_slack ≈ _QUERY_SLACK
        @test PMC.branch_ptdf_row(x, 101) ≈ 0.4 .* _slack_row(core, 1)
        @test PMC.branch_ptdf_row(x, 102) ≈ -0.6 .* _slack_row(core, 1)
        @test PMC.branch_ptdf_row(x, 107) ≈ _slack_row(core, 6)
        @test PMC.branch_ptdf_row(x, 108) ≈ -_slack_row(core, 6)
        @test typeof(PMC.branch_ptdf_row(x, 101)) == Vector{Float64}
    end
end

@testset "Queries by id: rows with a contingency" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        core = _write_query_file(path)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        core6 = _five_bus_core(; b = [6.0, 5.0, 8.0, 4.0, 6.0, 7.0])
        c = PMC.branch_contingency(x, [101])
        @test !PMC.is_islanding(c)
        @test PMC.branch_ptdf_row(x, 102, c) ≈ -1 .* _slack_row(core6, 1)
        @test iszero(PMC.branch_ptdf_row(x, 101, c))
        c2 = PMC.branch_contingency(x, [101, 102])
        wf = PMC.compute_woodbury_factors(
            core,
            PMC.NetworkModification(x.outages, [101, 102]),
        )
        expected = PMC.apply_dist_slack!(
            PMC.apply_woodbury_correction(core, 2, wf),
            core,
            _QUERY_SLACK,
            wf,
        )
        @test iszero(PMC.branch_ptdf_row(x, 101, c2))
        @test iszero(PMC.branch_ptdf_row(x, 102, c2))
        @test PMC.branch_ptdf_row(x, 103, c2) ≈ expected
        @test PMC.is_islanding(PMC.branch_contingency(x, [106]))
        island = PMC.branch_contingency(x, [106])
        for id in (103, 106)
            @test_throws "Distributed slack is not supported for a modification that islands" PMC.branch_ptdf_row(
                x,
                id,
                island,
            )
        end
    end
end

@testset "Queries by id: lookups and errors" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path; reverse_bus_map = Dict(98 => 3))
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.has_branch(x, 101)
        @test !PMC.has_branch(x, 101, PMC.PRIMARY)
        @test !PMC.has_branch(x, 999)
        @test PMC.has_bus_column(x, 4)
        @test PMC.bus_column(x, 4) == 4
        @test PMC.has_bus_column(x, 98)
        @test PMC.bus_column(x, 98) == 3
        @test !PMC.has_bus_column(x, 77)
        @test_throws "Branch with id 999 has no row in the branch map. A network reduction removed it, or it is not in service." PMC.branch_ptdf_row(
            x,
            999,
        )
        @test_throws "Bus 77 has no column in the matrix. It is not a bus of the file, or a network reduction removed it." PMC.bus_column(
            x,
            77,
        )
        @test_throws "Branch with id 999 has no row in the outage table." PMC.branch_contingency(
            x,
            [999],
        )
        # An id with an outage row and no leaf row in the branch map.
        HDF5.h5open(path, "r+") do f
            _replace_entry!(f["branch_outages"], "id", [101, 102, 13, 105, 109])
        end
        y = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test_throws "Branch with id 109 has no row in the branch map." PMC.branch_contingency(
            y,
            [109],
        )
    end
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        core = _write_query_file(path; with_outages = false)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test !x.has_outages
        @test PMC.branch_ptdf_row(x, 101) ≈ 0.4 .* _slack_row(core, 1)
        @test_throws "The factor file has no branch outage table, so a contingency by branch id is not available. The writer of the file did not add it." PMC.branch_contingency(
            x,
            [101],
        )
    end
end

# The cutoff drops the entries of the row at or below `tol`, before the multiplication by the
# share.
# Without the cutoff call, an entry at or below `tol` stays (bus 4 of the first row). With the
# cutoff after the share, an entry above `tol` and at or below `tol / share` drops (bus 3 of the
# first row).
function _check_cutoff_order(raw::Vector{Float64}, share::Float64, tol::Float64)
    @test any(v -> abs(v) <= tol, raw)
    @test any(v -> abs(v) > tol && abs(share * v) <= tol, raw)
    return raw .* (abs.(raw) .> tol) .* share
end

@testset "Queries by id: the cutoff applies before the share" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        tol = 0.1
        core = _write_query_file(path; tol = tol)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.get_tol(PMC.get_core(x)) == tol
        expected = _check_cutoff_order(_slack_row(core, 1), 0.4, tol)
        @test PMC.branch_ptdf_row(x, 101) ≈ expected
        # Arc 4 is open. Id 101 keeps the share 0.4 of arc 1.
        c = PMC.branch_contingency(x, [105])
        wf = PMC.compute_woodbury_factors(core, PMC.NetworkModification(x.outages, [105]))
        raw = PMC.apply_dist_slack!(
            PMC.apply_woodbury_correction(core, 1, wf),
            core,
            _QUERY_SLACK,
            wf,
        )
        expected = _check_cutoff_order(raw, 0.4, tol)
        @test PMC.branch_ptdf_row(x, 101, c) ≈ expected
    end
end

@testset "Queries by id: a zero share gives a zero row" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        t = _six_arc_table()
        b = copy(t.b)
        b[7] = 0.0
        table = PMC.BranchArcTable(t.kind, t.id, t.parent, t.sign, b, t.alpha, t.n_arc)
        _write_query_file(path; table = table)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.branch_ptdf_row(x, 106) == zeros(5)
        c = PMC.branch_contingency(x, [105])
        @test PMC.branch_ptdf_row(x, 106, c) == zeros(5)
        @test PMC.branch_ptdf_row(x, 105, c) == zeros(5)
    end
end

@testset "Queries by id: a contingency belongs to its factor file" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        y = PMC.read_branch_factors(path; linear_solver = "KLU")
        text = "The contingency belongs to another factor file."
        c = PMC.branch_contingency(x, [101])
        island = PMC.branch_contingency(x, [106])
        @test PMC.is_islanding(island)
        @test_throws text PMC.branch_ptdf_row(y, 102, c)
        @test_throws text PMC.branch_ptdf_row(y, 103, island)
        @test length(PMC.branch_ptdf_row(x, 102, c)) == length(PMC.get_bus_axis(x))
        @test_throws "Distributed slack is not supported" PMC.branch_ptdf_row(
            x,
            103,
            island,
        )
        @test_throws text PMC.branch_ptdf_row(y, 999, c)
        @test_throws text PMC.branch_ptdf_row(y, 999, PMC.PRIMARY, c)
    end
end

@testset "Queries by id: a circuit takes its winding" begin
    circuits = PMC.BranchArcTable(
        [1, 6, 7, 8, 1, 1],
        [101, 201, 201, 201, 104, 105],
        [-1, -2, -3, -4, -5, -6],
        [1, 1, 1, 1, 1, 1],
        [4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        zeros(6),
        6,
    )
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        core = _write_query_file(path; table = circuits)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.has_branch(x, 101)
        @test !PMC.has_branch(x, 201)
        c = PMC.branch_contingency(x, [101])
        wf = PMC.compute_woodbury_factors(core, PMC.NetworkModification(x.outages, [101]))
        r = x.leaf_rows[(201, 2)]
        expected =
            c.shares[r] .* PMC.apply_dist_slack!(
                PMC.apply_woodbury_correction(core, x.row_arcs[r], wf),
                core,
                _QUERY_SLACK,
                wf,
            )
        @test PMC.branch_ptdf_row(x, 201, PMC.SECONDARY, c) ≈ expected
        @test !iszero(expected)
        for w in (PMC.PRIMARY, PMC.SECONDARY, PMC.TERTIARY)
            @test PMC.has_branch(x, 201, w)
            @test !PMC.has_branch(x, 101, w)
            @test length(PMC.branch_ptdf_row(x, 201, w)) == length(PMC.get_bus_axis(x))
        end
        @test PMC.branch_ptdf_row(x, 201, PMC.SECONDARY) !=
              PMC.branch_ptdf_row(x, 201, PMC.PRIMARY)
        @test_throws "The id 201 is a three-winding transformer. Pass a Winding" PMC.branch_ptdf_row(
            x,
            201,
        )
    end
end

@testset "Queries by id: bus columns resolve one step" begin
    @test PMC._bus_columns([1, 2, 3], Dict(98 => 2)) ==
          Dict(1 => 1, 2 => 2, 3 => 3, 98 => 2)
    @test PMC._bus_columns([3], Dict(1 => 2, 2 => 3)) == Dict(3 => 1, 2 => 1)
end

@testset "Queries by id: outage predicates" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.has_branch_outages(x)
        @test PMC.has_branch_outage(x, 101)
        @test !PMC.has_branch_outage(x, 999)
        for id in (101, 999)
            if PMC.has_branch_outage(x, id)
                @test !PMC.is_islanding(PMC.branch_contingency(x, [id]))
            end
        end
        @test_throws "has no row in the outage table" PMC.branch_contingency(x, [999])
    end
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        core = _write_query_file(path; with_outages = false)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test !PMC.has_branch_outages(x)
        for id in (101, 102, 999)
            @test !PMC.has_branch_outage(x, id)
        end
        @test PMC.branch_ptdf_row(x, 101) ≈ 0.4 .* _slack_row(core, 1)
    end
end

@testset "Queries by id: slack and branch map predicates" begin
    islanding_text = "Distributed slack is not supported for a modification that islands"
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.has_dist_slack(x)
        island = PMC.branch_contingency(x, [106])
        @test PMC.is_islanding(island) && PMC.has_dist_slack(x)
        @test_throws islanding_text PMC.branch_ptdf_row(x, 103, island)
        @test_throws islanding_text PMC.branch_ptdf_row(x, 106, island)
        # An outage row with no branch map row.
        HDF5.h5open(path, "r+") do f
            _replace_entry!(f["branch_outages"], "id", [101, 102, 13, 105, 109])
        end
        y = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test PMC.has_branch_outage(y, 101)
        @test !PMC.has_branch_outage(y, 109)
        @test_throws "Branch with id 109 has no row in the branch map." PMC.branch_contingency(
            y,
            [109],
        )
    end
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path)
        HDF5.h5open(path, "r+") do f
            _replace_entry!(f, "dist_slack", Float64[])
        end
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test !PMC.has_dist_slack(x)
        island = PMC.branch_contingency(x, [106])
        @test PMC.is_islanding(island) && !PMC.has_dist_slack(x)
        @test length(PMC.branch_ptdf_row(x, 103, island)) == length(PMC.get_bus_axis(x))
    end
end

@testset "Queries by id: distributed slack needs one reference bus" begin
    arc_ax = [(1, 2), (3, 4)]
    bus_ax = [1, 2, 3, 4]
    core = PMC.FactorCore(
        [1.0, 1.0],
        Set([1, 3]),
        (arc_ax, bus_ax),
        Dict(1 => ([(1, 2)], [1, 2]), 3 => ([(3, 4)], [3, 4])),
        [1, 1];
        network_data = nothing,
        linear_solver = "KLU",
        tol = 0.0,
    )
    table =
        PMC.BranchArcTable([1, 1], [101, 102], [-1, -2], [1, 1], [1.0, 1.0], zeros(2), 2)
    mktempdir() do dir
        for (name, weights) in (("slack.h5", fill(1.0, 4)), ("plain.h5", Float64[]))
            path = joinpath(dir, name)
            HDF5.h5open(path, "w") do f
                PMC.write_factor_core(
                    f,
                    core;
                    shift_injection = zeros(2),
                    dist_slack = weights,
                )
                PMC._write_branch_map!(f, table)
            end
            if isempty(weights)
                x = PMC.read_branch_factors(path; linear_solver = "KLU")
                @test !PMC.has_dist_slack(x)
                @test length(PMC.branch_ptdf_row(x, 101)) == 4
            else
                @test_throws "Distributed slack is not supported for systems with multiple reference buses." PMC.read_branch_factors(
                    path;
                    linear_solver = "KLU",
                )
            end
        end
    end
end

@testset "Queries by id: contingency ids are not empty and do not repeat" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path)
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test_throws "An outage needs at least one branch key." PMC.branch_contingency(
            x,
            Int[],
        )
        @test_throws "Duplicate branch key in outage" PMC.branch_contingency(
            x,
            [101, 101],
        )
    end
end

@testset "Queries by id: a zero impedance sum in a file throws" begin
    chain = PMC.BranchArcTable(
        [9, 1, 1, 1, 2, 1, 1, 10, 1, 1],
        [0, 101, 102, 103, 104, 105, 106, 0, 107, 108],
        [-1, 1, 1, -2, -3, -4, -5, -6, 8, 8],
        [1, 1, -1, 1, 1, 1, 1, 1, 1, -1],
        [0.0, 4.0, 6.0, 5.0, 8.0, 4.0, 6.0, 0.0, 10.0, -10.0],
        zeros(10),
        6,
    )
    mktempdir() do dir
        path = joinpath(dir, "zero_sum.h5")
        _write_query_file(path; with_outages = false, table = chain)
        @test_throws "branch_shares: row 8 (series chain) has a zero impedance sum" PMC.read_branch_factors(
            path;
            linear_solver = "KLU",
        )
    end
end

@testset "Queries by id: one read budget for the whole file" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path)
        HDF5.h5open(path, "r") do f
            probe = PMC.ReadBudget(path)
            PMC.read_factor_arrays(f, probe)
            core_bytes = filesize(path) - probe.remaining
            budget = PMC.ReadBudget(core_bytes)
            PMC.read_factor_arrays(f, budget)
            @test_throws "branch_map" PMC.read_branch_map(f, budget, 6)
        end
    end
end

@testset "Queries by id: the branch map must match the arcs of the core" begin
    t = _six_arc_table()
    seven = PMC.BranchArcTable(
        [t.kind; 1],
        [t.id; 109],
        [t.parent; -7],
        [t.sign; 1],
        [t.b; 3.0],
        [t.alpha; 0.0],
        7,
    )
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path; table = seven)
        @test_throws "parent -7 is not an arc of the 6 arcs" PMC.read_branch_factors(
            path;
            linear_solver = "KLU",
        )
    end
end

@testset "Queries by id: a reduced bus names a bus of the file" begin
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        _write_query_file(path; reverse_bus_map = Dict(98 => 3))
        HDF5.h5open(path, "r+") do f
            _replace_entry!(f, "reduced_bus_to", [77])
        end
        @test_throws "reduced_bus_to holds bus 77 for the reduced bus 98" PMC.read_branch_factors(
            path;
            linear_solver = "KLU",
        )
    end
    mktempdir() do dir
        path = joinpath(dir, "q.h5")
        reduced = Dict(99 => 98)
        _write_query_file(path; reverse_bus_map = reduced)
        rec = (
            applied = [2],
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
            merged_to = [98],
            irreducible_buses = Int[],
        )
        arc_ax = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
        HDF5.h5open(path, "r+") do f
            PMC.write_reduction_records!(f, rec, [1, 2, 3, 4, 5], arc_ax, reduced)
        end
        x = PMC.read_branch_factors(path; linear_solver = "KLU")
        @test !PMC.has_bus_column(x, 99)
        @test !PMC.has_bus_column(x, 98)
        @test PMC.has_bus_column(x, 3)
        HDF5.h5open(path, "r+") do f
            _replace_entry!(f, "reduced_bus_to", [77])
        end
        @test_throws "reduced_bus_to holds bus 77" PMC.read_branch_factors(
            path;
            linear_solver = "KLU",
        )
    end
end
