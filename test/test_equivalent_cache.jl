# Cached two-port on reduction aggregates: warm-query cost, invalidation on every mutation
# path, and the storage-precision guards (the cache is ComplexF64 by design; Ybus stays
# YBUS_ELTYPE).

@testset "equivalent cache: warm query skips the two-port rebuild" begin
    sys = _mk_line_pst_parallel_system()
    nr = get_network_reduction_data(Ybus(sys))
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]

    @test !bp.equivalent_ybus_populated
    cold = PNM.get_equivalent_physical_branch_parameters(bp, nr)
    @test bp.equivalent_ybus_populated
    warm = PNM.get_equivalent_physical_branch_parameters(bp, nr)
    # Same answer, and the recovered fields are bit-identical -- the cache must not perturb.
    @test PNM.get_equivalent_r(warm) === PNM.get_equivalent_r(cold)
    @test PNM.get_equivalent_x(warm) === PNM.get_equivalent_x(cold)
    @test PNM.get_equivalent_shift(warm) === PNM.get_equivalent_shift(cold)
    # The cached value is exactly what `ybus_branch_entries` computes.
    @test collect(bp.equivalent_ybus) ≈
          collect(ComplexF64.(PNM.ybus_branch_entries(bp, nr))) rtol = 1e-15
end

@testset "equivalent cache: adding a member invalidates" begin
    sys = _mk_line_pst_parallel_system()
    nr = get_network_reduction_data(Ybus(sys))
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    before = PNM.get_equivalent_physical_branch_parameters(bp, nr)
    @test bp.equivalent_ybus_populated

    PNM.add_branch!(bp, PSY.get_component(Line, sys, "L2"))
    @test !bp.equivalent_ybus_populated
    after = PNM.get_equivalent_physical_branch_parameters(bp, nr)
    # A third member changes the equivalent; a stale cache would return `before` unchanged.
    @test PNM.get_equivalent_x(after) != PNM.get_equivalent_x(before)
end

@testset "equivalent cache: series chain invalidates on segment append" begin
    sys = _mk_line_pst_parallel_system()
    nr = get_network_reduction_data(Ybus(sys))
    bs = PNM.BranchesSeries((1, 2))
    PNM.add_branch!(bs, PSY.get_component(Line, sys, "L2"), :FromTo)
    @test !bs.equivalent_ybus_populated
    PNM.populate_equivalent_ybus!(bs, nr)
    @test bs.equivalent_ybus_populated
    PNM.add_branch!(bs, PSY.get_component(Line, sys, "L1"), :FromTo)
    @test !bs.equivalent_ybus_populated
end

@testset "equivalent cache: field type is ComplexF64 for both aggregate kinds" begin
    sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr = get_network_reduction_data(Ybus(sys))
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    @test typeof(bp.equivalent_ybus) === PNM.CACHED_TWO_PORT
    @test PNM.CACHED_TWO_PORT === NTuple{4, ComplexF64}
    # `ybus_branch_entries` hands back ComplexF32 for series chains and ComplexF64 for parallel
    # groups; the declared field type is the one place that difference is normalized.
    @test eltype(PNM.ybus_branch_entries(bp, nr)) === ComplexF64
    PNM.populate_equivalent_ybus!(bp, nr)
    @test eltype(bp.equivalent_ybus) === ComplexF64
end

@testset "equivalent cache: Ybus storage stays at YBUS_ELTYPE" begin
    for reductions in (
        NetworkReduction[],
        NetworkReduction[DegreeTwoReduction()],
    )
        sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
        ybus = Ybus(sys; network_reductions = reductions)
        @test eltype(ybus.data) === PNM.YBUS_ELTYPE
        nr = get_network_reduction_data(ybus)
        # Warm every cache, then re-assert: populating a ComplexF64 cache must not promote the
        # stored ComplexF32 Ybus (a promoted `V` in `sparse(I, J, V)` doubles memory silently).
        for arc in PNM.get_arc_axis(nr)
            PNM.arc_dc_resistance(nr, arc)
        end
        @test eltype(ybus.data) === PNM.YBUS_ELTYPE
        for entry in values(PNM.get_series_branch_map(nr))
            @test eltype(entry.equivalent_ybus) === ComplexF64
        end
    end
end
