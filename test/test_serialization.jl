function _roundtrip_ptdf(sys; kwargs...)
    vptdf = VirtualPTDF(sys; kwargs...)
    path = joinpath(mktempdir(), "ptdf.h5")
    to_hdf5(vptdf, path)
    return vptdf, from_hdf5(VirtualPTDF, path), path
end

function _modf_roundtrip(vmodf)
    path = joinpath(mktempdir(), "modf.h5")
    to_hdf5(vmodf, path)
    return from_hdf5(VirtualMODF, path)
end

function _outage_roundtrip(vmodf, sys)
    path = joinpath(mktempdir(), "outages.h5")
    to_hdf5(vmodf, path, sys)
    return from_hdf5(VirtualMODF, path), from_hdf5(BranchOutageTable, path)
end

function _test_ptdf_rows_equal(a, b, arcs)
    for arc in arcs
        @test isapprox(
            collect(PNM.get_ptdf_row(a, arc)),
            collect(PNM.get_ptdf_row(b, arc));
            atol = 1e-12,
        )
    end
    return
end

function _test_same_core(a, b)
    @test PNM.get_tol(a) == PNM.get_tol(b)
    @test PNM.get_cutoff(PNM.get_core(a)) == PNM.get_cutoff(PNM.get_core(b))
    @test PNM.get_tol_spec(PNM.get_core(a)) == PNM.get_tol_spec(PNM.get_core(b))
    @test PNM.get_bus_axis(a) == PNM.get_bus_axis(b)
    @test PNM.get_arc_axis(a) == PNM.get_arc_axis(b)
    @test PNM.get_ref_bus_position(a) == PNM.get_ref_bus_position(b)
    @test PNM.get_system_uuid(a) == PNM.get_system_uuid(b)
    @test PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(a))) ==
          PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(b)))
    @test PNM.get_core(a).arc_live_members == PNM.get_core(b).arc_live_members
    return
end

# Every monitored row of `expected` on `vmodf` equals the row of `actual` on `restored`.
function _test_modf_rows_equal(vmodf, restored, expected, actual = expected)
    @test actual.is_islanding == expected.is_islanding
    for monitored in 1:length(PNM.get_arc_axis(vmodf))
        @test isapprox(
            collect(vmodf[monitored, expected]),
            collect(restored[monitored, actual]);
            atol = 1e-9,
        )
    end
    return
end

function _add_outage!(sys, components...)
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    for component in components
        PSY.add_supplemental_attribute!(sys, component, outage)
    end
    return outage
end

const RTS_REDUCTIONS = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
const RTS_SYS = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
const RTS_VMODF = VirtualMODF(RTS_SYS; network_reductions = RTS_REDUCTIONS)
const RTS_RESTORED, RTS_TABLE = _outage_roundtrip(RTS_VMODF, RTS_SYS)
const RTS_NR = PNM.get_network_reduction_data(RTS_VMODF)

@testset "factor file" begin
    @testset "VirtualPTDF c_sys14" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf, restored, _ = _roundtrip_ptdf(sys)
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        @test isempty(PNM.get_dist_slack(restored))
    end

    @testset "VirtualPTDF with dist_slack" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        buses = sort!([PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)])
        ds = Dict{Int, Float64}(buses[2] => 0.3, buses[5] => 0.7)
        vptdf, restored, _ = _roundtrip_ptdf(sys; dist_slack = ds)
        @test !isempty(PNM.get_dist_slack(restored))
        @test PNM.get_dist_slack(restored) == PNM.get_dist_slack(vptdf)
        @test PNM.get_dist_slack_normalized(restored) ≈
              PNM.get_dist_slack_normalized(vptdf)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
    end

    @testset "VirtualPTDF with an explicit AutoTolerance" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        spec = AutoTolerance(; data_precision = 1e-8, safety = 2.0, quantile = 0.9)
        vptdf, restored, _ = _roundtrip_ptdf(sys; tol = spec)
        @test PNM.get_tol_spec(PNM.get_core(restored)) == spec
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
    end

    @testset "VirtualMODF RTS-GMLC with reductions" begin
        @test PNM.get_reverse_bus_search_map(
            PNM.get_network_reduction_data(RTS_RESTORED),
        ) ==
              PNM.get_reverse_bus_search_map(RTS_NR)
        _test_same_core(RTS_VMODF, RTS_RESTORED)
        _test_ptdf_rows_equal(
            VirtualPTDF(PNM.get_core(RTS_VMODF)),
            VirtualPTDF(PNM.get_core(RTS_RESTORED)),
            PNM.get_arc_axis(RTS_VMODF),
        )
        for map in (
            RTS_NR.reverse_direct_branch_map,
            RTS_NR.reverse_parallel_branch_map,
            RTS_NR.reverse_series_branch_map,
        )
            mod = NetworkModification(RTS_VMODF, first(keys(map)))
            _test_modf_rows_equal(RTS_VMODF, RTS_RESTORED, mod)
        end
    end

    @testset "case10 radial and series reductions with 3WT outage" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf = VirtualMODF(sys; network_reductions = RTS_REDUCTIONS)
        restored = _modf_roundtrip(vmodf)
        _test_same_core(vmodf, restored)
        _test_ptdf_rows_equal(
            VirtualPTDF(PNM.get_core(vmodf)),
            VirtualPTDF(PNM.get_core(restored)),
            PNM.get_arc_axis(vmodf),
        )
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        _test_modf_rows_equal(vmodf, restored, NetworkModification(vmodf, trf))
    end

    @testset "ACTIVSg2000 large system" begin
        sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
        vptdf, restored, _ = _roundtrip_ptdf(sys; network_reductions = RTS_REDUCTIONS)
        _test_same_core(vptdf, restored)
        arcs = PNM.get_arc_axis(vptdf)
        _test_ptdf_rows_equal(
            vptdf,
            restored,
            arcs[round.(Int, range(1, length(arcs); length = 5))],
        )
        vmodf =
            VirtualMODF(PNM.get_core(vptdf), sys; automatically_register_outages = false)
        restored_modf = _modf_roundtrip(vmodf)
        nr = PNM.get_network_reduction_data(vmodf)
        mod = NetworkModification(vmodf, first(keys(nr.reverse_direct_branch_map)))
        monitored = length(arcs) ÷ 2
        @test isapprox(
            collect(vmodf[monitored, mod]),
            collect(restored_modf[monitored, mod]);
            atol = 1e-10,
        )
    end

    @testset "phase shifter" begin
        sys = _mk_line_pst_parallel_system()
        vptdf, restored, path = _roundtrip_ptdf(sys)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        shift = PNM.HDF5.h5open(path, "r") do f
            return read(f["shift_injection"])
        end
        @test any(!iszero, shift)
    end

    @testset "islanding outage" begin
        sys = PSB.build_system(
            PSB.PSSEParsingTestSystems,
            "psse_14_network_reduction_test_system",
        )
        vmodf =
            VirtualMODF(sys; linear_solver = "KLU", automatically_register_outages = false)
        restored = _modf_roundtrip(vmodf)
        bridges = [
            e for
            e in eachindex(vmodf.PTDF_A_diag) if abs(vmodf.PTDF_A_diag[e] - 1.0) < 1e-6
        ]
        @test !isempty(bridges)
        e = first(bridges)
        mod = NetworkModification(
            "bridge",
            [ArcModification(e, -vmodf.arc_susceptances[e])],
            ShuntModification[],
            true,
        )
        monitored = first(i for i in eachindex(vmodf.PTDF_A_diag) if i != e)
        base_row = PNM.get_ptdf_row(
            VirtualPTDF(PNM.get_core(vmodf)),
            PNM.get_arc_axis(vmodf)[monitored],
        )
        @test !isapprox(collect(vmodf[monitored, mod]), collect(base_row); atol = 1e-9)
        _test_modf_rows_equal(vmodf, restored, mod)
    end

    @testset "errors" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf, _, path = _roundtrip_ptdf(sys)
        @test_throws ErrorException to_hdf5(vptdf, path)
        to_hdf5(vptdf, path; force = true)
        PNM.HDF5.h5open(path, "r+") do f
            PNM.HDF5.delete_attribute(f, "tol_kind")
            PNM.HDF5.write_attribute(f, "tol_kind", "other")
        end
        @test_throws ErrorException from_hdf5(VirtualPTDF, path)
        PNM.HDF5.h5open(path, "r+") do f
            PNM.HDF5.delete_attribute(f, "format_version")
            PNM.HDF5.write_attribute(f, "format_version", 2)
        end
        @test_throws ErrorException from_hdf5(VirtualPTDF, path)
        @test_throws ErrorException from_hdf5(VirtualMODF, path)
    end
end

@testset "factor file outages" begin
    @testset "RTS-GMLC outage by name" begin
        for (map, tag) in (
            (RTS_NR.reverse_direct_branch_map, "direct"),
            (RTS_NR.reverse_series_branch_map, "series"),
            (RTS_NR.reverse_parallel_branch_map, "parallel"),
        )
            branch = first(keys(map))
            key = PNM._outage_key(branch)
            @test RTS_TABLE.rows[key][1] == tag
            @test NetworkModification(RTS_TABLE, typeof(branch), key[2]) ==
                  NetworkModification(RTS_TABLE, key...)
            _test_modf_rows_equal(
                RTS_VMODF,
                RTS_RESTORED,
                NetworkModification(RTS_VMODF, branch),
                NetworkModification(RTS_TABLE, key...),
            )
        end
        @test_throws ErrorException NetworkModification(RTS_TABLE, "Line", "no such line")
        @test_throws ErrorException NetworkModification(RTS_TABLE, Tuple{String, String}[])
    end

    @testset "case10 3WT outage and skipped series chain" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf = VirtualMODF(sys; network_reductions = RTS_REDUCTIONS)
        path = joinpath(mktempdir(), "case10.h5")
        @test_logs (:warn,) match_mode = :any to_hdf5(vmodf, path, sys)
        restored = from_hdf5(VirtualMODF, path)
        table = from_hdf5(BranchOutageTable, path)
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        key = PNM._outage_key(trf)
        @test table.rows[key][1] == "3wt"
        _test_modf_rows_equal(
            vmodf,
            restored,
            NetworkModification(vmodf, trf),
            NetworkModification(table, key...),
        )
        nr = PNM.get_network_reduction_data(vmodf)
        chain = [
            line for line in PSY.get_components(PSY.Line, sys) if
            haskey(nr.reverse_series_branch_map, line)
        ]
        @test length(chain) == 2
        for branch in chain
            @test !haskey(table.rows, PNM._outage_key(branch))
        end
    end

    @testset "multi-branch outage" begin
        by_arc = Dict{Tuple{Int, Int}, Vector{PSY.ACTransmission}}()
        for (branch, arc) in RTS_NR.reverse_series_branch_map
            push!(get!(by_arc, arc, PSY.ACTransmission[]), branch)
        end
        chain = first(v for v in values(by_arc) if length(v) >= 2)
        direct = first(keys(RTS_NR.reverse_direct_branch_map))
        for pair in (chain[1:2], [chain[1], direct])
            sys_outage = _add_outage!(RTS_SYS, pair...)
            oracle = NetworkModification(RTS_VMODF, RTS_SYS, sys_outage)
            actual = NetworkModification(RTS_TABLE, [PNM._outage_key(b) for b in pair])
            @test length(actual.arc_modifications) == length(oracle.arc_modifications)
            _test_modf_rows_equal(RTS_VMODF, RTS_RESTORED, oracle, actual)
        end
        chain_mod = NetworkModification(RTS_TABLE, [PNM._outage_key(b) for b in chain[1:2]])
        @test length(chain_mod.arc_modifications) == 1
    end

    @testset "registered outages round trip" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        lines = collect(PSY.get_components(PSY.Line, sys))
        _add_outage!(sys, lines[1])
        _add_outage!(sys, lines[2])
        vmodf = VirtualMODF(sys)
        restored, _ = _outage_roundtrip(vmodf, sys)
        original = PNM.get_registered_contingencies(vmodf)
        copies = PNM.get_registered_contingencies(restored)
        @test length(original) == 2
        @test sort(collect(keys(copies))) == sort(collect(keys(original)))
        for (id, spec) in original
            @test copies[id].modification.label == spec.modification.label
            _test_modf_rows_equal(
                vmodf,
                restored,
                spec.modification,
                copies[id].modification,
            )
        end
        @test isempty(PNM.get_registered_contingencies(_modf_roundtrip(vmodf)))
    end

    @testset "phase shifter outage" begin
        sys = _mk_line_pst_parallel_system()
        vmodf = VirtualMODF(sys)
        restored, table = _outage_roundtrip(vmodf, sys)
        pst = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")
        key = PNM._outage_key(pst)
        @test any(!iszero, table.delta_shift[table.rows[key][3]])
        _test_modf_rows_equal(
            vmodf,
            restored,
            NetworkModification(vmodf, pst),
            NetworkModification(table, key...),
        )
    end

    @testset "file without system" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        _, _, path = _roundtrip_ptdf(sys)
        @test_throws ErrorException from_hdf5(BranchOutageTable, path)
    end
end
