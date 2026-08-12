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

    # A productive IterativeTopologyReduction round legitimately stamps both its own slot and
    # the primitive's, so a container coming out of a real run has both `radial_reduction` (or
    # `degree_two_reduction`) and `iterative_topology_reduction` set on the same object. The
    # tailored ITR message must still win over the generic duplicate-application one, which the
    # primitive's own `has_radial_reduction`/`has_degree_two_reduction` check would otherwise
    # reach first.
    after_itr_and_radial = PNM.ReductionContainer(;
        radial_reduction = RadialReduction(),
        iterative_topology_reduction = r,
    )
    radial_err = try
        PNM.validate_reduction_type(RadialReduction(), after_itr_and_radial)
        nothing
    catch e
        e
    end
    @test radial_err isa PNM.IS.DataFormatError
    @test occursin(
        "IterativeTopologyReduction applies RadialReduction internally",
        radial_err.msg,
    )

    after_itr_and_d2 = PNM.ReductionContainer(;
        degree_two_reduction = DegreeTwoReduction(),
        iterative_topology_reduction = r,
    )
    d2_err = try
        PNM.validate_reduction_type(DegreeTwoReduction(), after_itr_and_d2)
        nothing
    catch e
        e
    end
    @test d2_err isa PNM.IS.DataFormatError
    @test occursin(
        "IterativeTopologyReduction applies DegreeTwoReduction internally",
        d2_err.msg,
    )
end

@testset "IterativeTopologyReduction converges past a single pass" begin
    sys = build_iterative_convergence_system()
    # Buses 2 and 3 must survive every round. Their loads pin them against
    # `DegreeTwoReduction`'s system-derived set, but `RadialReduction` only exempts reference
    # buses and this caller-supplied set — see `build_iterative_convergence_system`'s docstring.
    pinned = Set([2, 3])

    # Precondition: one fixed sequence leaves a degree-two bus behind, and that bus must not be
    # one of the two pinned buses — `leftover` alone includes buses 2 and 3 too (they sit at
    # degree two as chain terminals), so excluding the pinned set is what actually pins this
    # assertion to bus 1, the one this fixture exists to exercise.
    y_once = Ybus(sys; irreducible_buses = pinned,
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()])
    A_once = AdjacencyMatrix(y_once)
    leftover = [
        b for b in PNM.get_bus_axis(y_once)
        if length(SparseArrays.nzrange(A_once.data, A_once.lookup[1][b])) == 2
    ]
    non_pinned_leftover =
        setdiff(leftover, PNM.get_irreducible_buses(PNM.get_network_reduction_data(y_once)))
    @test !isempty(non_pinned_leftover)

    # The iterative reduction eliminates it, leaving exactly the two pinned buses.
    y_iter = Ybus(sys; irreducible_buses = pinned,
        network_reductions = NetworkReduction[IterativeTopologyReduction()])
    A_iter = AdjacencyMatrix(y_iter)
    @test Set(PNM.get_bus_axis(y_iter)) == Set([2, 3])
    @test all(
        length(SparseArrays.nzrange(A_iter.data, A_iter.lookup[1][b])) > 2 ||
        b in PNM.get_irreducible_buses(PNM.get_network_reduction_data(y_iter))
        for b in PNM.get_bus_axis(y_iter)
    )
    @test length(PNM.get_bus_axis(y_iter)) < length(PNM.get_bus_axis(y_once))
end

@testset "IterativeTopologyReduction accumulates irreducible_buses on a no-op round" begin
    sys = build_radial_only_tree_system()

    # Precondition: DegreeTwoReduction genuinely finds nothing to do, even after RadialReduction
    # has already peeled every leaf. Without this, the test below would prove nothing.
    y_radial_only = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    nrd_d2 = PNM.get_reduction(y_radial_only, sys, DegreeTwoReduction())
    @test isempty(nrd_d2.removed_buses)

    # DegreeTwoReduction's own exempt set is the system-derived complement, computed independently
    # of whether it ever fires productively.
    system_derived = PNM._system_derived_irreducible_buses(sys, true)
    @test !isempty(system_derived)

    # The no-op DegreeTwoReduction round it takes every iteration must still fold its exempt set
    # into the accumulated NetworkReductionData, even though `_apply_reduction` is never called
    # for it on this fixture.
    y_iter = Ybus(sys; network_reductions = NetworkReduction[IterativeTopologyReduction()])
    final_irreducible = PNM.get_irreducible_buses(PNM.get_network_reduction_data(y_iter))
    @test issubset(system_derived, final_irreducible)
end
