@testset "IterativeTopologyReduction spec and validation" begin
    r = IterativeTopologyReduction()
    @test PNM.get_radial_reduction(r) == RadialReduction()
    @test PNM.get_degree_two_reduction(r) == DegreeTwoReduction()
    @test PNM.get_max_rounds(r) > 1

    # The AC path needs reactive-injector hosts kept, so the spec must carry that through.
    r_ac = IterativeTopologyReduction(;
        degree_two = DegreeTwoReduction(; reduce_reactive_power_injectors = false),
    )
    @test !PNM.get_reduce_reactive_power_injectors(PNM.get_degree_two_reduction(r_ac))

    # Rejected as a duplicate of itself.
    prior = PNM.ReductionContainer(; iterative_topology_reduction = r)
    @test_throws PNM.IS.DataFormatError PNM.validate_reduction_type(r, prior)

    # Rejected after Ward, like every other reduction.
    after_ward = PNM.ReductionContainer(; ward_reduction = WardReduction([1]))
    @test_throws PNM.IS.DataFormatError PNM.validate_reduction_type(r, after_ward)

    # Rejected when a primitive it owns has already been applied, so the two cannot interleave.
    after_d2 = PNM.ReductionContainer(; degree_two_reduction = DegreeTwoReduction())
    @test_throws PNM.IS.DataFormatError PNM.validate_reduction_type(r, after_d2)

    # Rejected in reverse order too: a standalone primitive after IterativeTopologyReduction
    # is a guaranteed no-op (the fixed point already includes it) and would leave the
    # container's slots ambiguous about which spec produced what.
    after_itr = PNM.ReductionContainer(; iterative_topology_reduction = r)
    @test_throws PNM.IS.DataFormatError PNM.validate_reduction_type(
        RadialReduction(),
        after_itr,
    )
    @test_throws PNM.IS.DataFormatError PNM.validate_reduction_type(
        DegreeTwoReduction(),
        after_itr,
    )
end
