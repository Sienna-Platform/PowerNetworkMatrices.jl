import SparseArrays

@testset "Flowgate accumulator rating-weighted" begin
    # rating = [10, 100, 50]; contingency 1 hits arcs 2 and 3 with raw |LODF| = 0.5.
    # impact = |LODF| * rating_c / rating_m : onto 2 -> 0.5*10/100=0.05 ; onto 3 -> 0.5*10/50=0.10
    P = [
        -1.0 0.5 0.5
        0.5 -1.0 0.5
        0.0 0.0 -1.0
    ]
    rating = [10.0, 100.0, 50.0]
    inv_rating = inv.(rating)
    for parent_matrix in (P, SparseArrays.sparse(P))
        acc = PNM.FlowgateAccumulator(3, 5, 0.0)
        PNM._accumulate!(acc, parent_matrix, rating, inv_rating)
        @test acc.max_off[1] == 0.5                  # raw, rating-independent
        @test iszero(acc.max_off[3])                 # radial
        @test isapprox(acc.l1[1], 0.05 + 0.10)
        @test isapprox(acc.l2sq[1], 0.05^2 + 0.10^2)
        @test acc.cnt[1] == 2
        @test sort(abs.(acc.buf_val[1:acc.buf_cnt[1], 1])) ≈ [0.05, 0.10]
    end
end

@testset "Flowgate accumulator buf_min fast-path" begin
    # In _accumulate!, P[i,j] = LODF for contingency i, monitored j.  Contingency i=1
    # accumulates from all columns j≠1.  Build a 4-arc system where all ratings=1 so
    # impact == |LODF|.  Give contingency 1 four non-zero off-diagonal entries across
    # columns 2,3,4,5 (row 1) to fill a top_n=2 buffer and exercise the fast-path.
    n = 5
    rating = ones(n)
    inv_rating = ones(n)
    # P[i,j]: row=contingency, col=monitored.
    # Row 1 contributions via columns 2..5: 0.9, 0.8, 0.5, 0.7
    # Expected top-2: {0.9, 0.8}; 0.5 and 0.7 rejected by buf_min=0.8 fast path.
    P = Float64[-1 0.9 0.8 0.5 0.7
        0 -1 0 0 0
        0 0 -1 0 0
        0 0 0 -1 0
        0 0 0 0 -1]
    acc = PNM.FlowgateAccumulator(n, 2, 0.0)
    PNM._accumulate!(acc, P, rating, inv_rating)
    kept = sort(abs.(acc.buf_val[1:acc.buf_cnt[1], 1]); rev = true)
    @test kept ≈ [0.9, 0.8]
    @test acc.buf_min[1] == 0.8

    # Tie: two entries with |impact|=0.5; first-encountered must be retained.
    # Column j=2 is processed before j=3 in the outer loop, so j=2 wins the tie.
    P2 = Float64[-1 0.5 0.5
        0 -1 0
        0 0 -1]
    rating2 = ones(3)
    inv_rating2 = ones(3)
    acc2 = PNM.FlowgateAccumulator(3, 1, 0.0)
    PNM._accumulate!(acc2, P2, rating2, inv_rating2)
    @test acc2.buf_cnt[1] == 1
    # First-encountered tie winner: j=2 (the first column processed by _accumulate!)
    @test acc2.buf_idx[1, 1] == 2
end

@testset "Flowgate arc ratings and branch resolution" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    lodf = LODF(sys5)
    nr = PNM.get_network_reduction_data(lodf)
    arcs = PNM.get_arc_axis(lodf)

    ratings = PNM._build_arc_ratings(lodf)
    @test length(ratings) == length(arcs)
    @test all(r -> isfinite(r) && r > 0, ratings)

    for (k, arc) in enumerate(arcs)
        br = PNM.get_arc_branch(nr, arc)
        @test br isa PSY.ACTransmission
        @test isapprox(PNM._arc_rating(nr, arc), ratings[k])
    end

    @test isnothing(PNM.get_arc_branch(nr, (-1, -2)))
    # An arc with no branch has no rating.
    @test isnothing(PNM._arc_rating(nr, (-1, -2)))
end

@testset "Flowgate ranking helper minmax" begin
    @test PNM._minmax_normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    @test PNM._minmax_normalize([7.0]) == [1.0]
    @test PNM._minmax_normalize([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]
end

@testset "Flowgates on c_sys5 (rating-weighted, FixedForcedOutage)" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    lodf = LODF(sys5)
    nr = PNM.get_network_reduction_data(lodf)

    fgs = flowgates(lodf)
    @test fgs isa Vector{PNM.FlowgateResult}
    @test !isempty(fgs)

    # Retrieve flowgate parts through the getters (Sienna style: no dot access).
    fg = first(fgs)
    @test get_flowgate_contingency_branch(fg) isa PSY.ACTransmission
    @test get_flowgate_outage(fg) isa PSY.FixedForcedOutage
    @test PSY.get_outage_status(get_flowgate_outage(fg)) == 1.0
    @test get_flowgate_monitored(fg) isa Vector{PSY.ACTransmission}
    @test length(get_flowgate_monitored(fg)) == length(get_flowgate_impacts(fg))

    @test get_flowgate_breadth(fg) isa Float64
    @test get_flowgate_magnitude(fg) isa Float64

    scores = get_flowgate_score.(fgs)
    @test issorted(scores; rev = true)
    @test all(s -> 0.0 <= s <= 1.0, scores)

    # Impacts are rating-weighted: cross-check against |LODF[m,c]| * rating_c / rating_m.
    arcs = PNM.get_arc_axis(lodf)
    ratings = PNM._build_arc_ratings(lodf)
    arc_index = Dict(a => k for (k, a) in enumerate(arcs))
    rev = nr.reverse_direct_branch_map
    c_arc = rev[get_flowgate_contingency_branch(fg)]
    ci = arc_index[c_arc]
    for (mbr, imp) in zip(get_flowgate_monitored(fg), get_flowgate_impacts(fg))
        m_arc = rev[mbr]
        mi = arc_index[m_arc]
        raw = lodf[m_arc, c_arc]
        expected = raw * ratings[ci] / ratings[mi]
        @test isapprox(imp, expected; atol = 1e-9)
    end

    # Monitored set ranked by |impact| descending.
    @test issorted(abs.(get_flowgate_impacts(fg)); rev = true)

    @test all(f -> length(get_flowgate_monitored(f)) <= 2, flowgates(lodf; top_n = 2))
    a = flowgates(lodf)
    b = flowgates(lodf)
    @test get_flowgate_score.(a) == get_flowgate_score.(b)
    @test_throws Exception flowgates(lodf; top_n = 0)
    @test_throws Exception flowgates(lodf; breadth_weight = 1.5)
    @test_throws Exception flowgates(lodf; breadth_weight = -0.1)
    # Fix 4: tolerance keyword validation
    @test_throws Exception flowgates(lodf; monitor_tol = -0.1)
    @test_throws Exception flowgates(lodf; radial_tol = -1.0)
    @test isempty(flowgates(lodf; radial_tol = 999.0))
    @test (@inferred flowgates(lodf)) isa Vector{PNM.FlowgateResult}

    # Fix 1: FixedForcedOutage monitored_components populated from the monitored set.
    fg = first(flowgates(lodf))
    expected_uuids = Set(IS.get_uuid.(get_flowgate_monitored(fg)))
    @test PSY.get_monitored_components(get_flowgate_outage(fg)) == expected_uuids

    # Fix 2: all returned flowgates have non-empty monitored sets.
    @test all(f -> !isempty(get_flowgate_monitored(f)), flowgates(lodf))

    # Fix 2: huge monitor_tol gates out every impact -> no flowgates returned.
    @test isempty(flowgates(lodf; monitor_tol = 1.0e6))
end

@testset "Flowgate outage selection and attachment" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    lodf = LODF(sys5)
    fgs = flowgates(lodf; top_n = 3)

    # Selecting flowgates and attaching their outages to the system works.
    selected = first(fgs, 2)
    for f in selected
        PSY.add_supplemental_attribute!(
            sys5,
            get_flowgate_contingency_branch(f),
            get_flowgate_outage(f),
        )
    end
    attached = collect(PSY.get_supplemental_attributes(PSY.FixedForcedOutage, sys5))
    @test length(attached) == length(selected)
end

@testset "Flowgates unrated transformer base-power fallback (unit)" begin
    arc = PSY.Arc(nothing)
    line_rated = PSY.Line(;
        name = "rated_line",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = arc,
        r = 0.1,
        x = 0.2,
        b = (from = 0.01, to = 0.01),
        g = (from = 0.0, to = 0.0),
        rating = 100.0,
        angle_limits = (min = -π / 2, max = π / 2),
    )
    # A rated branch resolves directly to its rating, no fallback/warning.
    @test isfinite(PNM._rating_value(line_rated)) && PNM._rating_value(line_rated) > 0

    # An attached unrated transformer is approximated from its base power, in system-base pu
    # (base_power / system_base), and emits a warning. The fallback now lives in
    # get_equivalent_rating, so it applies both directly (here) and via members inside
    # parallel/series equivalents (see test_equivalent_getters.jl).
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    buses = collect(PSY.get_components(PSY.ACBus, sys))
    tap_attached = PSY.TapTransformer(;
        name = "unrated_tap_attached",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = PSY.Arc(; from = buses[1], to = buses[2]),
        r = 0.01,
        x = 0.1,
        primary_shunt = 0.0 + 0.0im,
        tap = 1.0,
        rating = nothing,
        base_power = 50.0,
        winding_group_number = PSY.WindingGroupNumber.GROUP_11,
    )
    # PSY rejects an unrated transformer on validation; we add it deliberately to exercise the
    # base-power fallback, so skip the rating check.
    PSY.add_component!(sys, tap_attached; skip_validation = true)
    expected = 50.0 / PSY.get_system_base_power(tap_attached)
    fallback = @test_logs (:warn,) match_mode = :any PNM._rating_value(tap_attached)
    @test fallback ≈ expected
end

@testset "Flowgates radial exclusion and sparse path" begin
    sys14 = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    lodf14 = LODF(sys14)
    nr14 = PNM.get_network_reduction_data(lodf14)
    fgs = flowgates(lodf14)
    M = PNM.get_lodf_data(lodf14)
    arcs = PNM.get_arc_axis(lodf14)
    arc_index = Dict(a => k for (k, a) in enumerate(arcs))
    for fg in fgs
        c = arc_index[nr14.reverse_direct_branch_map[get_flowgate_contingency_branch(fg)]]
        col = M[:, c]
        col[c] = 0.0
        @test maximum(abs, col) > 1e-5
    end

    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    lodf_sparse = LODF(sys5; tol = 1e-3)
    @test parent(PNM.get_lodf_data(lodf_sparse)) isa SparseArrays.SparseMatrixCSC
    @test !isempty(flowgates(lodf_sparse))
end

@testset "Flowgates effective_count = false uses literal count" begin
    # Unit-level: verify that the cnt branch of the flowgates loop fires correctly.
    # Build a synthetic 4-arc parent where the participation ratio differs from the
    # literal count so the two modes produce distinct breadth values.
    P = Float64[
        -1.0 0.9 0.1 0.0
        0.6 -1.0 0.0 0.0
        0.1 0.1 -1.0 0.4
        0.0 0.0 0.3 -1.0
    ]
    rating = [10.0, 20.0, 30.0, 40.0]
    inv_r = inv.(rating)
    # Contingency i=3 (row 3 in P) has 3 off-diagonal impacts: columns 1, 2, 4.
    acc = PNM.FlowgateAccumulator(4, 5, 0.0)
    PNM._accumulate!(acc, P, rating, inv_r)
    # Participation ratio for i=3 is l1^2/l2sq; literal count is cnt[3]=3.
    part_ratio = acc.l1[3]^2 / acc.l2sq[3]
    literal_cnt = Float64(acc.cnt[3])
    @test literal_cnt == 3.0
    @test part_ratio != literal_cnt   # they must differ so the two modes are distinguishable

    # System-level on c_sys14 (meshed, varied impact structure).
    sys14 = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    lodf14 = LODF(sys14)
    fgs_ec_false = flowgates(lodf14; effective_count = false)
    fgs_ec_true = flowgates(lodf14; effective_count = true)
    @test !isempty(fgs_ec_false)

    # effective_count=false breadths are integer-valued (they are Float64(cnt)).
    @test all(fg -> isinteger(get_flowgate_breadth(fg)), fgs_ec_false)

    # effective_count=true uses the participation ratio, which is not always integer.
    @test any(fg -> !isinteger(get_flowgate_breadth(fg)), fgs_ec_true)

    # The two modes produce different breadth values for at least one flowgate, confirming
    # the literal-count branch executes and is distinct from the participation-ratio path.
    ec_false_breadths = get_flowgate_breadth.(fgs_ec_false)
    ec_true_breadths = get_flowgate_breadth.(fgs_ec_true)
    @test ec_false_breadths != ec_true_breadths
end

@testset "Flowgates breadth_weight extremes give pure orderings" begin
    sys5 = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    lodf = LODF(sys5)

    # breadth_weight=1.0 → score = breadth_norm → result must be sorted by breadth desc.
    fgs_bw1 = flowgates(lodf; breadth_weight = 1.0)
    @test !isempty(fgs_bw1)
    @test issorted(get_flowgate_breadth.(fgs_bw1); rev = true)

    # breadth_weight=0.0 → score = magnitude_norm → result must be sorted by magnitude desc.
    fgs_bw0 = flowgates(lodf; breadth_weight = 0.0)
    @test !isempty(fgs_bw0)
    @test issorted(get_flowgate_magnitude.(fgs_bw0); rev = true)

    # The two extreme orderings differ (c_sys5 has spread across breadth and magnitude).
    names_bw1 = PSY.get_name.(get_flowgate_contingency_branch.(fgs_bw1))
    names_bw0 = PSY.get_name.(get_flowgate_contingency_branch.(fgs_bw0))
    @test names_bw1 != names_bw0
end

@testset "Flowgates self-arc never in monitored set" begin
    # The outaged (contingency) arc must never appear in its own monitored set.
    # The diagonal skip is positional (row == col); test it for both the dense
    # _accumulate! path (default LODF) and the sparse path (tol=1e-3).
    sys14 = PSB.build_system(PSB.PSITestSystems, "c_sys14")

    for lodf14 in (LODF(sys14), LODF(sys14; tol = 1e-3))
        fgs = flowgates(lodf14)
        @test !isempty(fgs)
        for fg in fgs
            cb_uuid = IS.get_uuid(get_flowgate_contingency_branch(fg))
            monitored_uuids = IS.get_uuid.(get_flowgate_monitored(fg))
            @test cb_uuid ∉ monitored_uuids
        end
    end
end
