function _roundtrip_ptdf(sys; kwargs...)
    vptdf = VirtualPTDF(sys; kwargs...)
    path = joinpath(mktempdir(), "ptdf.h5")
    to_hdf5(vptdf, path)
    return vptdf, from_hdf5(VirtualPTDF, path), path
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
    @test typeof(PNM.get_cutoff(PNM.get_core(a))) == typeof(PNM.get_cutoff(PNM.get_core(b)))
    @test PNM.get_bus_axis(a) == PNM.get_bus_axis(b)
    @test PNM.get_arc_axis(a) == PNM.get_arc_axis(b)
    @test PNM.get_ref_bus_position(a) == PNM.get_ref_bus_position(b)
    @test PNM.get_system_uuid(a) == PNM.get_system_uuid(b)
    @test PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(a))) ==
          PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(b)))
    @test PNM.get_core(a).arc_live_members == PNM.get_core(b).arc_live_members
    return
end

function _modf_roundtrip(vmodf)
    path = joinpath(mktempdir(), "modf.h5")
    to_hdf5(vmodf, path)
    return from_hdf5(VirtualMODF, path)
end

function _test_modf_rows_equal(vmodf, restored, mod)
    for monitored in 1:length(PNM.get_arc_axis(vmodf))
        @test isapprox(
            collect(vmodf[monitored, mod]),
            collect(restored[monitored, mod]);
            atol = 1e-10,
        )
    end
    return
end

@testset "factor file" begin
    @testset "VirtualPTDF c_sys14" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf, restored, _ = _roundtrip_ptdf(sys)
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        @test isempty(restored.dist_slack)
    end

    @testset "VirtualPTDF with dist_slack" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        buses = PNM.get_bus_axis(VirtualPTDF(sys))
        ds = Dict{Int, Float64}(buses[2] => 0.3, buses[5] => 0.7)
        vptdf, restored, _ = _roundtrip_ptdf(sys; dist_slack = ds)
        @test !isempty(restored.dist_slack)
        @test restored.dist_slack == vptdf.dist_slack
        @test restored.dist_slack_normalized ≈ vptdf.dist_slack_normalized
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
    end

    @testset "VirtualMODF RTS-GMLC with reductions" begin
        sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
        reds = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vmodf = VirtualMODF(sys; network_reductions = reds)
        restored = _modf_roundtrip(vmodf)
        nr = PNM.get_network_reduction_data(vmodf)
        @test PNM.get_network_reduction_data(restored).reverse_bus_search_map ==
              nr.reverse_bus_search_map
        _test_same_core(vmodf, restored)
        _test_ptdf_rows_equal(
            VirtualPTDF(PNM.get_core(vmodf)),
            VirtualPTDF(PNM.get_core(restored)),
            PNM.get_arc_axis(vmodf),
        )
        direct = first(keys(nr.reverse_direct_branch_map))
        parallel = first(keys(nr.reverse_parallel_branch_map))
        series = first(keys(nr.reverse_series_branch_map))
        for branch in (direct, parallel, series)
            mod = NetworkModification(vmodf, branch)
            _test_modf_rows_equal(vmodf, restored, mod)
        end
    end

    @testset "case10 radial and series reductions with 3WT outage" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        reds = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vmodf = VirtualMODF(sys; network_reductions = reds)
        restored = _modf_roundtrip(vmodf)
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
        reds = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vptdf, restored, _ = _roundtrip_ptdf(sys; network_reductions = reds)
        @test PNM.get_cutoff(PNM.get_core(restored)) == PNM.get_cutoff(PNM.get_core(vptdf))
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
        branch = first(keys(nr.reverse_direct_branch_map))
        mod = NetworkModification(vmodf, branch)
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
        @test mod.is_islanding
        _test_modf_rows_equal(vmodf, restored, mod)
    end

    @testset "VirtualPTDF with relative cutoff" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        ybus = Ybus(sys)
        A = IncidenceMatrix(ybus)
        core = PNM.VirtualFactorCore(
            BA_Matrix(ybus).data,
            A.data,
            Set(PNM.get_ref_bus_position(ybus)),
            A.axes,
            A.subnetwork_axes,
            copy(PNM.get_arc_live_members(ybus)),
            PNM.get_branch_catalog(ybus);
            tol = PNM.RelativeCutoff(0.01),
        )
        vptdf = VirtualPTDF(core)
        path = joinpath(mktempdir(), "relative.h5")
        to_hdf5(vptdf, path)
        restored = from_hdf5(VirtualPTDF, path)
        @test PNM.get_cutoff(PNM.get_core(restored)) == PNM.RelativeCutoff(0.01)
        @test PNM.get_tol(restored) == 0.01
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
    end

    @testset "cutoff kinds" begin
        @test PNM._cutoff_kind(PNM.RelativeCutoff(0.01)) == "relative"
        @test PNM._restore_cutoff(Val(:relative), 0.01) == PNM.RelativeCutoff(0.01)
        @test PNM._restore_cutoff(Val(:absolute), 1e-9) == PNM.AbsoluteCutoff(1e-9)
        @test_throws ErrorException PNM._restore_cutoff(Val(:other), 1.0)
    end

    @testset "errors" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf, _, path = _roundtrip_ptdf(sys)
        @test_throws ErrorException to_hdf5(vptdf, path)
        to_hdf5(vptdf, path; force = true)
        PNM.HDF5.h5open(path, "r+") do f
            PNM.HDF5.delete_attribute(f, "format_version")
            PNM.HDF5.write_attribute(f, "format_version", 2)
        end
        @test_throws ErrorException from_hdf5(VirtualPTDF, path)
        @test_throws ErrorException from_hdf5(VirtualMODF, path)
    end
end

_outage_key(branch) = (string(nameof(typeof(branch))), PSY.get_name(branch))

function _test_outage_rows_equal(vmodf, restored, expected, actual)
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

function _outage_roundtrip(vmodf, sys)
    path = joinpath(mktempdir(), "outages.h5")
    to_hdf5(vmodf, path, sys)
    return from_hdf5(VirtualMODF, path), from_hdf5(BranchOutageTable, path)
end

function _add_outage!(sys, components...)
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    for component in components
        PSY.add_supplemental_attribute!(sys, component, outage)
    end
    return outage
end

@testset "factor file outages" begin
    @testset "RTS-GMLC outage by name" begin
        sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
        reds = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vmodf = VirtualMODF(sys; network_reductions = reds)
        restored, table = _outage_roundtrip(vmodf, sys)
        nr = PNM.get_network_reduction_data(vmodf)
        for (map, tag) in (
            (nr.reverse_direct_branch_map, "direct"),
            (nr.reverse_series_branch_map, "series"),
            (nr.reverse_parallel_branch_map, "parallel"),
        )
            branch = first(keys(map))
            key = _outage_key(branch)
            @test table.rows[key][1] == tag
            @test NetworkModification(table, typeof(branch), key[2]) ==
                  NetworkModification(table, key...)
            _test_outage_rows_equal(
                vmodf,
                restored,
                NetworkModification(vmodf, branch),
                NetworkModification(table, key...),
            )
        end
    end

    @testset "case10 3WT outage and skipped series chain" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        reds = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vmodf = VirtualMODF(sys; network_reductions = reds)
        path = joinpath(mktempdir(), "case10.h5")
        @test_logs (:warn,) match_mode = :any to_hdf5(vmodf, path, sys)
        restored = from_hdf5(VirtualMODF, path)
        table = from_hdf5(BranchOutageTable, path)
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        @test table.rows[_outage_key(trf)][1] == "3wt"
        _test_outage_rows_equal(
            vmodf,
            restored,
            NetworkModification(vmodf, trf),
            NetworkModification(table, _outage_key(trf)...),
        )
        nr = PNM.get_network_reduction_data(vmodf)
        chain = [
            line for line in PSY.get_components(PSY.Line, sys) if
            haskey(nr.reverse_series_branch_map, line)
        ]
        @test length(chain) == 2
        for branch in chain
            @test !haskey(table.rows, _outage_key(branch))
        end
    end

    @testset "multi-branch outage" begin
        sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
        reds = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vmodf = VirtualMODF(sys; network_reductions = reds)
        restored, table = _outage_roundtrip(vmodf, sys)
        nr = PNM.get_network_reduction_data(vmodf)
        by_arc = Dict{Tuple{Int, Int}, Vector{PSY.ACTransmission}}()
        for (branch, arc) in nr.reverse_series_branch_map
            push!(get!(by_arc, arc, PSY.ACTransmission[]), branch)
        end
        chain = first(v for v in values(by_arc) if length(v) >= 2)
        direct = first(keys(nr.reverse_direct_branch_map))
        for pair in (chain[1:2], [chain[1], direct])
            sys_outage = _add_outage!(sys, pair...)
            oracle = NetworkModification(vmodf, sys, sys_outage)
            keys_ = [_outage_key(b) for b in pair]
            actual = NetworkModification(table, keys_)
            @test length(actual.arc_modifications) == length(oracle.arc_modifications)
            _test_outage_rows_equal(vmodf, restored, oracle, actual)
        end
        @test length(
            NetworkModification(
                table,
                [_outage_key(b) for b in chain[1:2]],
            ).arc_modifications,
        ) ==
              1
    end

    @testset "registered outages round trip" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        lines = collect(PSY.get_components(PSY.Line, sys))
        _add_outage!(sys, lines[1])
        _add_outage!(sys, lines[2])
        vmodf = VirtualMODF(sys)
        path = joinpath(mktempdir(), "registered.h5")
        to_hdf5(vmodf, path, sys)
        restored = from_hdf5(VirtualMODF, path)
        original = PNM.get_registered_contingencies(vmodf)
        copies = PNM.get_registered_contingencies(restored)
        @test length(original) == 2
        @test sort(collect(keys(copies))) == sort(collect(keys(original)))
        for (id, spec) in original
            @test copies[id].modification.label == spec.modification.label
            @test copies[id].modification.is_islanding == spec.modification.is_islanding
            for monitored in 1:length(PNM.get_arc_axis(vmodf))
                @test isapprox(
                    collect(vmodf[monitored, spec]),
                    collect(restored[monitored, copies[id]]);
                    atol = 1e-9,
                )
            end
        end
        no_sys = joinpath(mktempdir(), "plain.h5")
        to_hdf5(vmodf, no_sys)
        @test isempty(PNM.get_registered_contingencies(from_hdf5(VirtualMODF, no_sys)))
    end

    @testset "phase shifter outage" begin
        sys = _mk_line_pst_parallel_system()
        vmodf = VirtualMODF(sys)
        restored, table = _outage_roundtrip(vmodf, sys)
        pst = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")
        key = _outage_key(pst)
        @test any(!iszero, table.delta_shift[table.rows[key][3]])
        _test_outage_rows_equal(
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
