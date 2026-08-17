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

@testset "IterativeTopologyReduction is idempotent" begin
    sys = build_iterative_convergence_system()
    # Pin buses 2 and 3, as `build_iterative_convergence_system`'s docstring requires, so
    # convergence lands on a genuine two-bus fixed point rather than collapsing the whole
    # network onto the reference bus.
    y1 = Ybus(sys; irreducible_buses = Set([2, 3]),
        network_reductions = NetworkReduction[IterativeTopologyReduction()])
    # A second application must find nothing: `_apply_primitive_reduction` returns the same
    # object for an empty step, so a converged Ybus is a fixed point of both primitives. The
    # user-supplied irreducible set lives on `y1`'s own `NetworkReductionData` and is read from
    # there, so it does not need to be passed again here.
    y2 = PNM._apply_primitive_reduction(y1, sys, RadialReduction())
    @test y2 === y1
    y3 = PNM._apply_primitive_reduction(y1, sys, DegreeTwoReduction())
    @test y3 === y1
end

@testset "IterativeTopologyReduction matches the unreduced network" begin
    # `build_iterative_convergence_system` needs its pinned buses repeated here: without them,
    # `RadialReduction` — which exempts only reference buses and the caller's explicit
    # `irreducible_buses`, never injector hosts — peels bus 3's load onto bus 2 once the hub
    # (bus 1) collapses, eliminating an injection-hosting bus and violating the Kron oracle's
    # injection-free precondition for every eliminated bus. The other two fixtures carry no
    # injectors on any bus a `DegreeTwoReduction` chain interior can reach, so they need no
    # pinning.
    for (builder, irreducible_buses) in (
        (build_iterative_convergence_system, Set([2, 3])),
        (build_two_parallel_degree_two_chains, Set{Int}()),
        (build_reversed_asymmetric_degree_two_chains, Set{Int}()),
    )
        sys = builder()
        y_full = Ybus(sys)
        y_red = Ybus(sys; irreducible_buses = irreducible_buses,
            network_reductions = NetworkReduction[IterativeTopologyReduction()])
        _test_kron_oracle(y_full, y_red)
    end
end

@testset "IterativeTopologyReduction chains through a lone composite arc" begin
    sys = build_nested_chain_segment_system()
    irreducible = Set([2, 4])

    # Precondition: one round must file the 1-2 chain as a LONE entry in `series_branch_map` and
    # leave bus 1 at degree two. A grouped arc instead would exercise the parallel path, and a
    # bus 1 above degree two would mean no second round chains through anything.
    y_once = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
        irreducible_buses = irreducible,
    )
    nrd_once = PNM.get_network_reduction_data(y_once)
    @test collect(keys(PNM.get_series_branch_map(nrd_once))) == [(1, 2)]
    A_once = AdjacencyMatrix(y_once)
    @test length(SparseArrays.nzrange(A_once.data, A_once.lookup[1][1])) == 2

    y_iter = Ybus(
        sys;
        network_reductions = NetworkReduction[IterativeTopologyReduction()],
        irreducible_buses = irreducible,
    )
    @test Set(PNM.get_bus_axis(y_iter)) == Set([2, 4])

    # The second round reaches the lone composite arc as a segment, in the reverse orientation:
    # its own key is (1, 2) while the pass-two segment is (2, 1). Assert the nesting is really
    # present, so this testset cannot quietly stop covering it.
    nrd_iter = PNM.get_network_reduction_data(y_iter)
    group = PNM.get_parallel_branch_map(nrd_iter)[(2, 4)]
    chains = [m for m in group if m isa PNM.BranchesSeries]
    @test length(chains) == 1
    outer = only(chains)
    @test any(segment -> segment isa PNM.BranchesSeries, outer)
    @test :ToFrom in outer.segment_orientations

    # The numeric gate. A chain segment enters as its own two-port, which is what a composite arc
    # contributes to Ybus and what is backed out for it; taking its constituent branches' raw
    # admittances instead leaves an error of order the branch admittances themselves.
    _test_kron_oracle(Ybus(sys), y_iter)
end
