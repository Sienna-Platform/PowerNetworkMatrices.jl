"""
    _clone_transformer_with_name(xfmr, new_name) -> TapTransformer

Clone a TapTransformer with a new name, sharing the same buses and identical
electrical parameters (R, X, tap, primary_shunt, rating, base_power,
base_voltage_primary, base_voltage_secondary). The returned object has no
supplemental attributes and is not attached to any system.

Used to create a second circuit on a bus pair so that `BA_Matrix` treats that
pair as a 2-circuit parallel group, enabling N-2 parallel-circuit tests.
"""
function _clone_transformer_with_name(xfmr::PSY.TapTransformer, new_name::String)
    old_arc = PSY.get_arc(xfmr)
    # Deep-copy to reproduce every electrical parameter exactly (avoids re-reading
    # each field in a particular unit basis), then give it a fresh identity, name,
    # and an arc that shares the original buses rather than the deep-copied ones.
    clone = deepcopy(xfmr)
    clone.internal = IS.InfrastructureSystemsInternal()
    PSY.set_name!(clone, new_name)
    PSY.set_arc!(clone, PSY.Arc(; from = old_arc.from, to = old_arc.to))
    return clone
end

"""
    _canonical_arc_key(d, k) -> Tuple{Int,Int}

Return `k` if it is a key of `d`, otherwise return the reversed tuple `(k[2], k[1])`.
Used when an arc may be stored under either orientation in a lookup dictionary.
"""
function _canonical_arc_key(d::AbstractDict, k::Tuple{Int, Int})
    if haskey(d, k)
        return k
    else
        return (k[2], k[1])
    end
end

"""
    _ptdf_post_row_builder(ptdf_post, pre_bus_axis) -> Function

Return a closure that accepts an `arc_key::Tuple{Int,Int}` and returns the
corresponding row of `ptdf_post` permuted to match `pre_bus_axis`.  The arc key
is canonicalised via `_canonical_arc_key` so either orientation is accepted.
"""
function _ptdf_post_row_builder(ptdf_post, pre_bus_axis)
    post_bus_axis = PNM.get_bus_axis(ptdf_post)
    post_bus_lookup = Dict(bn => i for (i, bn) in enumerate(post_bus_axis))
    bus_perm = [post_bus_lookup[bn] for bn in pre_bus_axis]
    post_arc_lookup = Dict{Tuple{Int, Int}, Int}(
        a => i for (i, a) in enumerate(PNM.get_arc_axis(ptdf_post))
    )
    return function (arc_key::Tuple{Int, Int})
        k = _canonical_arc_key(post_arc_lookup, arc_key)
        return ptdf_post[post_arc_lookup[k], :][bus_perm]
    end
end

"""
Helper: verify the N-1 identity for a single contingency.

Builds a ContingencySpec, computes the LODF reference (standard column for full
outage, `get_partial_lodf_row` for partial), and asserts:

    VirtualMODF[m, ctg] ≈ PTDF[m, :] + lodf_col[m] * PTDF[e, :]

for every monitored arc m.

For partial outages (delta_b ≠ -b_arc), the self-monitoring case (m == arc_idx)
requires an additional susceptance ratio correction:

    post_PTDF[e, :] = (b_post / b_pre) * (pre_PTDF[e, :] + lodf_col[e] * pre_PTDF[e, :])

This follows from the Sherman-Morrison expansion of B_m⁻¹ where the monitoring
arc's own susceptance changes.
"""
function verify_modf_lodf_identity(
    vmodf::VirtualMODF,
    vlodf::VirtualLODF,
    ptdf::PTDF,
    arc_idx::Int,
    delta_b::Float64;
    atol = 1e-6,
)
    n_arcs = size(vlodf, 1)
    b_arc = vmodf.arc_susceptances[arc_idx]

    # Build and register ContingencySpec
    ctg_uuid = Base.UUID(UInt128(hash((arc_idx, delta_b))))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification("test_arc_$(arc_idx)", [ArcModification(arc_idx, delta_b)]),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    # LODF reference: standard column for full outage, partial otherwise
    is_full_outage = isapprox(delta_b, -b_arc; atol = 1e-12)
    if is_full_outage
        lodf_col = [vlodf[m, arc_idx] for m in 1:n_arcs]
    else
        lodf_col = PNM.get_partial_lodf_row(vlodf, arc_idx, delta_b)
    end

    # Verify identity for every monitored arc
    for m in 1:n_arcs
        modf_row = vmodf[m, ctg]
        correction = ptdf[m, :] .+ lodf_col[m] .* ptdf[arc_idx, :]
        if !is_full_outage && m == arc_idx
            # Self-monitoring arc with partial outage: susceptance ratio correction.
            b_post = b_arc + delta_b
            expected = (b_post / b_arc) .* correction
        else
            expected = correction
        end
        @test isapprox(modf_row, expected; atol = atol)
    end

    # Clean up Woodbury cache so next call starts fresh
    empty!(vmodf.woodbury_cache)
    return
end

"""
Verify the N-2 simultaneous-outage identity for VirtualMODF.

For full outage of arcs `e1` and `e2` simultaneously, checks:

    VirtualMODF[m, ctg] ≈ PTDF[m,:] + eff1*PTDF[e1,:] + eff2*PTDF[e2,:]

for every monitored arc m, where eff1/eff2 come from the 2×2 LODF
correction (Woodbury identity on the pre-contingency LODF matrix):

    denom = 1 - LODF[e1,e2] * LODF[e2,e1]
    eff1  = (LODF[m,e1] + LODF[e2,e1] * LODF[m,e2]) / denom
    eff2  = (LODF[e1,e2] * LODF[m,e1] + LODF[m,e2]) / denom

Raises an error if the pair {e1, e2} is an N-2 islanding pair
(denom ≈ 0), since the closed-form formula is undefined in that case.
"""
function verify_modf_n2_lodf_identity(
    vmodf::VirtualMODF,
    vlodf::VirtualLODF,
    ptdf::PTDF,
    arc_idx1::Int,
    arc_idx2::Int;
    atol = 1e-6,
)
    n_arcs = size(vlodf, 1)
    b_arc1 = vmodf.arc_susceptances[arc_idx1]
    b_arc2 = vmodf.arc_susceptances[arc_idx2]

    L12 = vlodf[arc_idx1, arc_idx2]
    L21 = vlodf[arc_idx2, arc_idx1]
    denom = 1 - L12 * L21
    abs(denom) < 1e-6 &&
        error(
            "Arc pair ($arc_idx1, $arc_idx2) is an N-2 islanding pair (denom=$denom); use a non-islanding pair",
        )

    ctg_uuid = Base.UUID(UInt128(hash((arc_idx1, arc_idx2, :n2))))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification(
            "n2_test_$(arc_idx1)_$(arc_idx2)",
            [ArcModification(arc_idx1, -b_arc1), ArcModification(arc_idx2, -b_arc2)],
        ),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    for m in 1:n_arcs
        Lm1 = vlodf[m, arc_idx1]
        Lm2 = vlodf[m, arc_idx2]
        eff1 = (Lm1 + L21 * Lm2) / denom
        eff2 = (L12 * Lm1 + Lm2) / denom

        modf_row = vmodf[m, ctg]
        expected = ptdf[m, :] .+ eff1 .* ptdf[arc_idx1, :] .+ eff2 .* ptdf[arc_idx2, :]
        @test isapprox(modf_row, expected; atol = atol)
    end

    empty!(vmodf.woodbury_cache)
    return
end

"""
Return two distinct non-islanding arcs from `branch_map` whose simultaneous
outage does not island the network (L12*L21 ≠ 1 in the pre-contingency LODF).
Raises an error if no such pair exists in the map.
"""
function _find_two_non_islanding_arcs(vmodf, vlodf, branch_map)
    candidates = Tuple{Any, Int}[]
    for (arc_tuple, _) in branch_map
        arc_idx = vmodf.lookup[1][arc_tuple]
        if abs(vmodf.PTDF_A_diag[arc_idx]) < 1.0 - 1e-6
            push!(candidates, (arc_tuple, arc_idx))
        end
    end
    for i in eachindex(candidates), j in (i + 1):length(candidates)
        _, idx1 = candidates[i]
        _, idx2 = candidates[j]
        denom = 1 - vlodf[idx1, idx2] * vlodf[idx2, idx1]
        abs(denom) >= 1e-6 && return candidates[i], candidates[j]
    end
    error("No non-islanding arc pair found in map ($(length(candidates)) candidates)")
end

"""
Find a representative non-islanding arc from a branch map.

Skips arcs where PTDF_A_diag ≈ 1.0 (bridge arcs), since those produce
islanding contingencies where both MODF and LODF give degenerate results.
"""
function _find_non_islanding_arc(vmodf, branch_map)
    for (arc_tuple, _) in branch_map
        arc_idx = vmodf.lookup[1][arc_tuple]
        if abs(vmodf.PTDF_A_diag[arc_idx]) < 1.0 - 1e-6
            return arc_tuple, arc_idx
        end
    end
    error("No non-islanding branch found in map")
end

@testset "MODF vs LODF: no reductions" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[]
    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    @testset "direct branch" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.direct_branch_map)
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    @testset "parallel single-circuit" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.parallel_branch_map)
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches), PSY.SU)
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end

@testset "MODF vs LODF: radial reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[RadialReduction()]
    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    @testset "direct branch" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.direct_branch_map)
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    @testset "parallel single-circuit" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.parallel_branch_map)
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches), PSY.SU)
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end

@testset "MODF vs LODF: degree-two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[DegreeTwoReduction()]
    # reductions is modified by adding irreducible buses; pass a copy to ensure consistency.
    ptdf = PTDF(sys; network_reductions = deepcopy(reductions))
    vlodf = VirtualLODF(sys; network_reductions = deepcopy(reductions))
    vmodf = VirtualMODF(sys; network_reductions = deepcopy(reductions))
    nrd = get_network_reduction_data(vmodf)

    @testset "direct branch" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.direct_branch_map)
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    @testset "parallel single-circuit" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.parallel_branch_map)
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches), PSY.SU)
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    @testset "series segment" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.series_branch_map)
        series_chain = nrd.series_branch_map[arc_tuple]
        segment = first(series_chain)
        delta_b = PNM._compute_series_outage_delta_b(series_chain, segment)
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end

@testset "MODF vs LODF: radial + degree-two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
    # reductions is modified by adding irreducible buses; pass a copy to ensure consistency.
    ptdf = PTDF(sys; network_reductions = deepcopy(reductions))
    vlodf = VirtualLODF(sys; network_reductions = deepcopy(reductions))
    vmodf = VirtualMODF(sys; network_reductions = deepcopy(reductions))
    nrd = get_network_reduction_data(vmodf)

    @testset "direct branch" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.direct_branch_map)
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    @testset "parallel single-circuit" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.parallel_branch_map)
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches), PSY.SU)
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    @testset "series segment" begin
        arc_tuple, arc_idx = _find_non_islanding_arc(vmodf, nrd.series_branch_map)
        series_chain = nrd.series_branch_map[arc_tuple]
        segment = first(series_chain)
        delta_b = PNM._compute_series_outage_delta_b(series_chain, segment)
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end

@testset "MODF vs LODF N-2: direct branch pair" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[]
    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    (_, arc_idx1), (_, arc_idx2) =
        _find_two_non_islanding_arcs(vmodf, vlodf, nrd.direct_branch_map)
    verify_modf_n2_lodf_identity(vmodf, vlodf, ptdf, arc_idx1, arc_idx2)
end

@testset "VirtualMODF parallel-group N-k tests on RTS-GMLC" begin
    # Shared fixture: RTS-GMLC system augmented with a second circuit on the A14
    # transformer (bus 109 → 111) to create a 2-circuit parallel group.
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    xfmr_a14 = PSY.get_component(PSY.TapTransformer, sys, "A14")
    PSY.add_component!(sys, _clone_transformer_with_name(xfmr_a14, "A14_b"))

    # Resolve the canonical arc key for the parallel group once.  A lightweight
    # VirtualMODF is constructed here only to query the reduction maps; each
    # child testset builds its own vmodf so contingency caches stay isolated.
    nrd_shared = get_network_reduction_data(VirtualMODF(sys))
    par_key = if haskey(nrd_shared.parallel_branch_map, (109, 111))
        (109, 111)
    else
        (111, 109)
    end

    @testset "N-2 trip of one parallel transformer + a line" begin
        vmodf = VirtualMODF(sys)
        nrd = get_network_reduction_data(vmodf)

        @test haskey(nrd.parallel_branch_map, par_key)
        @test length(nrd.parallel_branch_map[par_key].branches) == 2

        # N-2 partner line: A18 (bus 111 → 113), selected as the line incident on
        # bus 111 with the largest |LODF| coupling with the parallel-group arc
        # (|LODF| ≈ 0.428 vs A19's 0.231).  Confirmed non-islanding by building
        # PTDF(sys_post) without error.
        #
        # Selection probe (run once, result baked in):
        #   vlodf = VirtualLODF(sys)
        #   |vlodf[(111,113), par_arc_idx]| ≈ 0.4285  <- largest
        #   |vlodf[(111,114), par_arc_idx]| ≈ 0.2310
        sel_line_name = "A18"

        A14_b_branch = PSY.get_component(PSY.TapTransformer, sys, "A14_b")
        line_branch = PSY.get_component(PSY.Line, sys, sel_line_name)

        mod_a14b = NetworkModification(vmodf, A14_b_branch)
        mod_a18 = NetworkModification(vmodf, line_branch)

        n2_mods = vcat(
            collect(mod_a14b.arc_modifications),
            collect(mod_a18.arc_modifications),
        )
        n2_mod = NetworkModification("n2_parallel_xfmr_plus_line", n2_mods)

        ctg_uuid = Base.UUID(UInt128(515151))
        ctg = ContingencySpec(ctg_uuid, n2_mod)
        vmodf.contingency_cache[ctg_uuid] = ctg

        sys_post = deepcopy(sys)
        PSY.remove_component!(
            sys_post,
            PSY.get_component(PSY.TapTransformer, sys_post, "A14_b"),
        )
        PSY.remove_component!(
            sys_post,
            PSY.get_component(PSY.Line, sys_post, sel_line_name),
        )
        ptdf_post = PTDF(sys_post)

        pre_bus_axis = collect(PNM.get_bus_axis(vmodf))
        ptdf_post_row = _ptdf_post_row_builder(ptdf_post, pre_bus_axis)
        post_arc_lookup = Dict{Tuple{Int, Int}, Int}(
            a => i for (i, a) in enumerate(PNM.get_arc_axis(ptdf_post))
        )

        atol = 1e-6
        for (label, arc_key) in [
            ("parallel pair (109,111)", par_key),
            ("A19 other 230kV off bus 111", (111, 114)),
            ("A12-1 138kV off bus 109", (108, 109)),
            ("B8 area-200", (204, 209)),
            ("C5 area-300", (302, 306)),
        ]
            mon_idx = PNM.get_arc_lookup(vmodf)[arc_key]
            modf_row = vmodf[mon_idx, ctg]
            @test isapprox(modf_row, ptdf_post_row(arc_key); atol = atol)
        end
    end

    @testset "N-3 trip of one parallel transformer + two lines" begin
        # N-3 outage: {A14_b, A18, A19}
        #   A18 (bus 111 -> 113): |LODF| ≈ 0.428 with the parallel-group arc.
        #   A19 (bus 111 -> 114): |LODF| ≈ 0.231 with the parallel-group arc.
        # Both lines are off bus 111 (area-100 hub).  PTDF(sys_post) confirmed
        # non-islanding (size (73, 106) after removing three branches).
        vmodf = VirtualMODF(sys)
        nrd = get_network_reduction_data(vmodf)

        @test haskey(nrd.parallel_branch_map, par_key)
        @test length(nrd.parallel_branch_map[par_key].branches) == 2

        A14_b_branch = PSY.get_component(PSY.TapTransformer, sys, "A14_b")
        a18_branch = PSY.get_component(PSY.Line, sys, "A18")
        a19_branch = PSY.get_component(PSY.Line, sys, "A19")

        mod_a14b = NetworkModification(vmodf, A14_b_branch)
        mod_a18 = NetworkModification(vmodf, a18_branch)
        mod_a19 = NetworkModification(vmodf, a19_branch)

        n3_mods = vcat(
            collect(mod_a14b.arc_modifications),
            collect(mod_a18.arc_modifications),
            collect(mod_a19.arc_modifications),
        )
        n3_mod = NetworkModification("n3_parallel_xfmr_plus_two_lines", n3_mods)

        ctg_uuid = Base.UUID(UInt128(515251))
        ctg = ContingencySpec(ctg_uuid, n3_mod)
        vmodf.contingency_cache[ctg_uuid] = ctg

        sys_post = deepcopy(sys)
        PSY.remove_component!(
            sys_post,
            PSY.get_component(PSY.TapTransformer, sys_post, "A14_b"),
        )
        PSY.remove_component!(sys_post, PSY.get_component(PSY.Line, sys_post, "A18"))
        PSY.remove_component!(sys_post, PSY.get_component(PSY.Line, sys_post, "A19"))
        ptdf_post = PTDF(sys_post)

        pre_bus_axis = collect(PNM.get_bus_axis(vmodf))
        ptdf_post_row = _ptdf_post_row_builder(ptdf_post, pre_bus_axis)
        post_arc_lookup = Dict{Tuple{Int, Int}, Int}(
            a => i for (i, a) in enumerate(PNM.get_arc_axis(ptdf_post))
        )

        # (111, 114) is A19 which is outaged and absent from sys_post, so it is
        # not included here.
        atol = 1e-6
        for (label, arc_key) in [
            ("parallel pair (109,111)", par_key),
            ("A20 230kV hub (112,113)", (112, 113)),
            ("A12-1 138kV off bus 109 (108,109)", (108, 109)),
            ("B8 area-200 (204,209)", (204, 209)),
            ("C5 area-300 (302,306)", (302, 306)),
        ]
            if !haskey(post_arc_lookup, arc_key) &&
               !haskey(post_arc_lookup, (arc_key[2], arc_key[1]))
                @info "Skipping $label: arc $arc_key not present in sys_post"
                continue
            end
            mon_idx = PNM.get_arc_lookup(vmodf)[arc_key]
            modf_row = vmodf[mon_idx, ctg]
            @test isapprox(modf_row, ptdf_post_row(arc_key); atol = atol)
        end
    end

    @testset "N-3 trip of three regular lines (Woodbury M=3)" begin
        # N-3 selection: {A23, B20, C20} — three PSY.Line instances from three
        # different areas (area 1, 2, 3) so the Woodbury W matrix is genuinely
        # M=3 with full rank.  None is in a parallel group or incident on a bus
        # whose removal would island the network.  PTDF(sys_post) confirmed
        # non-islanding (size (73, 105) after removing three branches).
        #   A23: area 1, bus 114 -> 116, 230 kV
        #   B20: area 2, bus 212 -> 213, 230 kV
        #   C20: area 3, bus 312 -> 313, 230 kV
        vmodf = VirtualMODF(sys)
        nrd = get_network_reduction_data(vmodf)

        a23_branch = PSY.get_component(PSY.Line, sys, "A23")
        b20_branch = PSY.get_component(PSY.Line, sys, "B20")
        c20_branch = PSY.get_component(PSY.Line, sys, "C20")

        mod_a23 = NetworkModification(vmodf, a23_branch)
        mod_b20 = NetworkModification(vmodf, b20_branch)
        mod_c20 = NetworkModification(vmodf, c20_branch)

        n3_mods = vcat(
            collect(mod_a23.arc_modifications),
            collect(mod_b20.arc_modifications),
            collect(mod_c20.arc_modifications),
        )
        n3_mod = NetworkModification("n3_three_regular_lines", n3_mods)

        ctg_uuid = Base.UUID(UInt128(515252))
        ctg = ContingencySpec(ctg_uuid, n3_mod)
        vmodf.contingency_cache[ctg_uuid] = ctg

        sys_post = deepcopy(sys)
        PSY.remove_component!(sys_post, PSY.get_component(PSY.Line, sys_post, "A23"))
        PSY.remove_component!(sys_post, PSY.get_component(PSY.Line, sys_post, "B20"))
        PSY.remove_component!(sys_post, PSY.get_component(PSY.Line, sys_post, "C20"))
        ptdf_post = PTDF(sys_post)

        pre_bus_axis = collect(PNM.get_bus_axis(vmodf))
        ptdf_post_row = _ptdf_post_row_builder(ptdf_post, pre_bus_axis)
        post_arc_lookup = Dict{Tuple{Int, Int}, Int}(
            a => i for (i, a) in enumerate(PNM.get_arc_axis(ptdf_post))
        )

        # Monitored arcs:
        #   1. (116, 119) A28: adjacent to A23's to-bus (116), area 1.
        #   2. (212, 223) B21: adjacent to B20's from-bus (212), area 2.
        #   3. (312, 323) C21: adjacent to C20's from-bus (312), area 3.
        #   4. (109, 111) parallel pair: sanity check that the parallel-augmented
        #      system still behaves correctly when A23/B20/C20 are not part of the N-3.
        #   5. (204, 209) B8: unrelated area-200 line.
        #   6. (302, 306) C5: unrelated area-300 line.
        atol = 1e-6
        for (label, arc_key) in [
            ("A28 adj A23 to-bus 116 (116,119)", (116, 119)),
            ("B21 adj B20 from-bus 212 (212,223)", (212, 223)),
            ("C21 adj C20 from-bus 312 (312,323)", (312, 323)),
            ("parallel pair sanity (109,111)", par_key),
            ("B8 area-200 (204,209)", (204, 209)),
            ("C5 area-300 (302,306)", (302, 306)),
        ]
            if !haskey(post_arc_lookup, arc_key) &&
               !haskey(post_arc_lookup, (arc_key[2], arc_key[1]))
                @info "Skipping $label: arc $arc_key not present in sys_post"
                continue
            end
            mon_idx = PNM.get_arc_lookup(vmodf)[arc_key]
            modf_row = vmodf[mon_idx, ctg]
            @test isapprox(modf_row, ptdf_post_row(arc_key); atol = atol)
        end
    end
end
