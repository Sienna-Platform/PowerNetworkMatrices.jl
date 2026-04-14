import Dates

function _make_test_time_series(name::String)
    resolution = Dates.Hour(1)
    dates = collect(
        Dates.DateTime("2020-01-01T00:00:00"):resolution:Dates.DateTime(
            "2020-01-01T23:00:00",
        ),
    )
    ta = TimeSeries.TimeArray(dates, ones(length(dates)))
    return PSY.SingleTimeSeries(; name = name, data = ta)
end

@testset "is_a_reduction predicate" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    line = first(PSY.get_components(PSY.Line, sys))
    @test PNM.is_a_reduction(line) == false
    @test PNM.is_a_reduction(PNM.BranchesParallel([line])) == true

    bs = PNM.BranchesSeries()
    PNM.add_branch!(bs, line, :FromTo)
    @test PNM.is_a_reduction(bs) == true

    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    @test PNM.is_a_reduction(PNM.ThreeWindingTransformerWinding(trf, 1)) == true
end

@testset "has_time_series" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    lines = collect(PSY.get_components(PSY.Line, sys))
    line1, line2 = lines[1], lines[2]
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))

    bp = PNM.BranchesParallel([line1, line2])
    bs = PNM.BranchesSeries()
    PNM.add_branch!(bs, line1, :FromTo)
    PNM.add_branch!(bs, line2, :FromTo)
    tww1 = PNM.ThreeWindingTransformerWinding(trf, 1)
    tww2 = PNM.ThreeWindingTransformerWinding(trf, 2)

    # No time series attached to any component
    @test PNM.has_time_series(bp, PSY.SingleTimeSeries, "rating") == false
    @test PNM.has_time_series(bs, PSY.SingleTimeSeries, "rating") == false
    @test PNM.has_time_series(tww1, PSY.SingleTimeSeries, "rating") == false

    # Add time series to line1 — BranchesParallel and BranchesSeries should detect it
    PSY.add_time_series!(sys, line1, _make_test_time_series("rating"))
    @test PNM.has_time_series(bp, PSY.SingleTimeSeries, "rating") == true
    @test PNM.has_time_series(bs, PSY.SingleTimeSeries, "rating") == true
    @test PNM.has_time_series(bp, PSY.SingleTimeSeries, "nonexistent") == false

    # ThreeWindingTransformerWinding delegates to parent transformer
    PSY.add_time_series!(sys, trf, _make_test_time_series("rating"))
    @test PNM.has_time_series(tww1, PSY.SingleTimeSeries, "rating") == true
    @test PNM.has_time_series(tww2, PSY.SingleTimeSeries, "rating") == true
    @test PNM.has_time_series(tww1, PSY.SingleTimeSeries, "nonexistent") == false
end

@testset "get_device_with_time_series" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    lines = collect(PSY.get_components(PSY.Line, sys))
    line1, line2 = lines[1], lines[2]
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))

    bp = PNM.BranchesParallel([line1, line2])
    bs = PNM.BranchesSeries()
    PNM.add_branch!(bs, line1, :FromTo)
    PNM.add_branch!(bs, line2, :FromTo)
    tww1 = PNM.ThreeWindingTransformerWinding(trf, 1)
    tww2 = PNM.ThreeWindingTransformerWinding(trf, 2)

    # No time series attached — should return nothing
    @test PNM.get_device_with_time_series(bp, PSY.SingleTimeSeries, "rating") === nothing
    @test PNM.get_device_with_time_series(bs, PSY.SingleTimeSeries, "rating") === nothing
    @test PNM.get_device_with_time_series(tww1, PSY.SingleTimeSeries, "rating") === nothing

    # Add time series to line1 in BranchesParallel
    PSY.add_time_series!(sys, line1, _make_test_time_series("rating"))
    @test PNM.get_device_with_time_series(bp, PSY.SingleTimeSeries, "rating") === line1
    @test PNM.get_device_with_time_series(bp, PSY.SingleTimeSeries, "nonexistent") ===
          nothing

    # Add time series to line1 in BranchesSeries
    @test PNM.get_device_with_time_series(bs, PSY.SingleTimeSeries, "rating") === line1
    @test PNM.get_device_with_time_series(bs, PSY.SingleTimeSeries, "nonexistent") ===
          nothing

    # Add time series to transformer — ThreeWindingTransformerWinding should delegate to parent
    PSY.add_time_series!(sys, trf, _make_test_time_series("rating"))
    @test PNM.get_device_with_time_series(tww1, PSY.SingleTimeSeries, "rating") === trf
    @test PNM.get_device_with_time_series(tww2, PSY.SingleTimeSeries, "rating") === trf
    @test PNM.get_device_with_time_series(tww1, PSY.SingleTimeSeries, "nonexistent") ===
          nothing
    @test PNM.get_device_with_time_series(tww2, PSY.SingleTimeSeries, "nonexistent") ===
          nothing
end
