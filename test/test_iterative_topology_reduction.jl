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

@testset "IterativeTopologyReduction converges past a single pass" begin
    sys = build_iterative_convergence_system()
    # Buses 2 and 3 must survive every round. Their loads pin them against
    # `DegreeTwoReduction`'s system-derived set, but `RadialReduction` only exempts reference
    # buses and this caller-supplied set — see `build_iterative_convergence_system`'s docstring.
    pinned = Set([2, 3])

    # Precondition: one fixed sequence leaves a degree-two bus behind.
    y_once = Ybus(sys; irreducible_buses = pinned,
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()])
    A_once = AdjacencyMatrix(y_once)
    leftover = [
        b for b in PNM.get_bus_axis(y_once)
        if length(SparseArrays.nzrange(A_once.data, A_once.lookup[1][b])) == 2
    ]
    @test !isempty(leftover)

    # The iterative reduction eliminates it.
    y_iter = Ybus(sys; irreducible_buses = pinned,
        network_reductions = NetworkReduction[IterativeTopologyReduction()])
    A_iter = AdjacencyMatrix(y_iter)
    @test all(
        length(SparseArrays.nzrange(A_iter.data, A_iter.lookup[1][b])) > 2 ||
        b in PNM.get_irreducible_buses(PNM.get_network_reduction_data(y_iter))
        for b in PNM.get_bus_axis(y_iter)
    )
    @test length(PNM.get_bus_axis(y_iter)) < length(PNM.get_bus_axis(y_once))
end
