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
