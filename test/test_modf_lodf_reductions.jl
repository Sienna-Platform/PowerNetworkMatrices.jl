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
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
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
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end

@testset "MODF vs LODF: degree-two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[DegreeTwoReduction()]
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
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
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
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
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
