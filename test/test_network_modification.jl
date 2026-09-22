@testset "NetworkModification: single-line outage matches rebuilt PTDF" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)

    arc_ax = PNM.get_arc_axis(vptdf)
    bus_ax = PNM.get_bus_axis(vptdf)
    n_arcs = length(arc_ax)
    n_buses = length(bus_ax)
    arc_lookup = PNM.get_arc_lookup(vptdf)

    for e in 1:n_arcs
        outaged_arc = arc_ax[e]

        # Woodbury-corrected PTDF rows via the new API
        mod = NetworkModification(vptdf, outaged_arc)
        wf = compute_woodbury_factors(vptdf, mod)

        # The outaged arc row should be all zeros (b_mon_post = 0)
        outaged_row = apply_woodbury_correction(vptdf, e, wf)
        @test all(abs.(outaged_row) .< 1e-10)

        # Rebuild PTDF from a fresh system with the line disabled
        sys_mod = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        lines = collect(PSY.get_components(PSY.ACBranch, sys_mod))
        outaged_line = nothing
        for l in lines
            arc = PSY.get_arc(l)
            arc_tuple = (arc.from.number, arc.to.number)
            if arc_tuple == outaged_arc
                outaged_line = l
                break
            end
        end
        @test !isnothing(outaged_line)
        PSY.set_available!(outaged_line, false)
        ptdf_rebuilt = PTDF(sys_mod)
        rebuilt_arc_ax = PNM.get_arc_axis(ptdf_rebuilt)
        rebuilt_arc_lookup = PNM.get_arc_lookup(ptdf_rebuilt)
        rebuilt_bus_lookup = PNM.get_bus_lookup(ptdf_rebuilt)

        # Compare each surviving arc's PTDF row
        for m in 1:n_arcs
            m == e && continue  # skip outaged arc
            monitored_arc = arc_ax[m]

            # Woodbury-corrected row
            wb_row = apply_woodbury_correction(vptdf, m, wf)

            # Find matching row in rebuilt PTDF
            if !haskey(rebuilt_arc_lookup, monitored_arc)
                continue
            end
            rebuilt_m = rebuilt_arc_lookup[monitored_arc]

            # Compare bus-by-bus using bus number matching
            for (b_idx, bus_num) in enumerate(bus_ax)
                if haskey(rebuilt_bus_lookup, bus_num)
                    rb_idx = rebuilt_bus_lookup[bus_num]
                    @test isapprox(
                        wb_row[b_idx],
                        ptdf_rebuilt[rebuilt_m, rb_idx],
                        atol = 1e-6,
                    )
                end
            end
        end
    end
end

@testset "NetworkModification: multi-line outage matches rebuilt PTDF" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)

    arc_ax = PNM.get_arc_axis(vptdf)
    bus_ax = PNM.get_bus_axis(vptdf)
    n_arcs = length(arc_ax)
    arc_lookup = PNM.get_arc_lookup(vptdf)
    arc_sus = vptdf.arc_susceptances

    # Outage arcs 1 and 2 simultaneously
    e1, e2 = 1, 2
    mods = [
        ArcModification(e1, -arc_sus[e1]),
        ArcModification(e2, -arc_sus[e2]),
    ]
    mod = NetworkModification("multi_outage", mods)
    wf = compute_woodbury_factors(vptdf, mod)

    # Rebuild PTDF with both lines disabled
    sys_mod = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    outaged_arcs = Set([arc_ax[e1], arc_ax[e2]])
    for l in PSY.get_components(PSY.ACBranch, sys_mod)
        arc = PSY.get_arc(l)
        arc_tuple = (arc.from.number, arc.to.number)
        if arc_tuple ∈ outaged_arcs
            PSY.set_available!(l, false)
        end
    end
    ptdf_rebuilt = PTDF(sys_mod)
    rebuilt_arc_lookup = PNM.get_arc_lookup(ptdf_rebuilt)
    rebuilt_bus_lookup = PNM.get_bus_lookup(ptdf_rebuilt)

    # Compare surviving arcs
    for m in 1:n_arcs
        m ∈ (e1, e2) && continue
        monitored_arc = arc_ax[m]

        wb_row = apply_woodbury_correction(vptdf, m, wf)

        if !haskey(rebuilt_arc_lookup, monitored_arc)
            continue
        end
        rebuilt_m = rebuilt_arc_lookup[monitored_arc]

        for (b_idx, bus_num) in enumerate(bus_ax)
            if haskey(rebuilt_bus_lookup, bus_num)
                rb_idx = rebuilt_bus_lookup[bus_num]
                @test isapprox(
                    wb_row[b_idx],
                    ptdf_rebuilt[rebuilt_m, rb_idx],
                    atol = 1e-6,
                )
            end
        end
    end

    # Outaged arcs should be zeros
    for e in (e1, e2)
        outaged_row = apply_woodbury_correction(vptdf, e, wf)
        @test all(abs.(outaged_row) .< 1e-10)
    end
end

@testset "NetworkModification: one-shot API and getindex consistency" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)

    arc_ax = PNM.get_arc_axis(vptdf)
    arc_tuple = arc_ax[1]

    mod = NetworkModification(vptdf, arc_tuple)

    # Two-step API
    wf = compute_woodbury_factors(vptdf, mod)
    row_twostep = apply_woodbury_correction(vptdf, 2, wf)

    # One-shot API
    row_oneshot = get_post_modification_ptdf_row(vptdf, 2, mod)

    # getindex API
    row_getindex = vptdf[2, mod]

    # Tuple-indexed getindex
    row_tuple = vptdf[arc_ax[2], mod]

    @test isapprox(row_twostep, row_oneshot, atol = 1e-14)
    @test isapprox(row_twostep, row_getindex, atol = 1e-14)
    @test isapprox(row_twostep, row_tuple, atol = 1e-14)
end

@testset "NetworkModification: matches VirtualMODF for N-1" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(sys)

    n_arcs = length(PNM.get_arc_axis(vptdf))

    for e in 1:n_arcs
        b_e = vptdf.arc_susceptances[e]
        ctg_id = e + 10000
        ctg = ContingencySpec(
            ctg_id,
            NetworkModification("test_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_id] = ctg

        mod = NetworkModification(ctg)
        wf = compute_woodbury_factors(vptdf, mod)

        for m in 1:n_arcs
            modf_row = PNM._compute_modf_entry(vmodf, m, ctg.modification)
            wb_row = apply_woodbury_correction(vptdf, m, wf)
            @test isapprox(wb_row, modf_row, atol = 1e-10)
        end
        empty!(vmodf.woodbury_cache)
    end
end

@testset "NetworkModification: from branch component" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys)

    line = first(PSY.get_components(PSY.Line, sys))
    mod = NetworkModification(vptdf, line)

    @test !isempty(mod.arc_modifications)
    @test mod.label == PSY.get_name(line)

    # Should produce a valid row
    row = get_post_modification_ptdf_row(vptdf, 1, mod)
    @test length(row) == length(PNM.get_bus_axis(vptdf))
end

@testset "NetworkModification: 14-bus system single-line outage" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    vptdf = VirtualPTDF(sys)

    arc_ax = PNM.get_arc_axis(vptdf)
    bus_ax = PNM.get_bus_axis(vptdf)
    arc_lookup = PNM.get_arc_lookup(vptdf)

    # Outage first arc
    outaged_arc = arc_ax[1]
    mod = NetworkModification(vptdf, outaged_arc)
    wf = compute_woodbury_factors(vptdf, mod)

    # Rebuild with line disabled
    sys_mod = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    for l in PSY.get_components(PSY.ACBranch, sys_mod)
        arc = PSY.get_arc(l)
        if (arc.from.number, arc.to.number) == outaged_arc
            PSY.set_available!(l, false)
            break
        end
    end
    ptdf_rebuilt = PTDF(sys_mod)
    rebuilt_arc_lookup = PNM.get_arc_lookup(ptdf_rebuilt)
    rebuilt_bus_lookup = PNM.get_bus_lookup(ptdf_rebuilt)

    # Compare surviving arcs
    n_arcs = length(arc_ax)
    for m in 2:n_arcs
        monitored_arc = arc_ax[m]
        wb_row = apply_woodbury_correction(vptdf, m, wf)

        if !haskey(rebuilt_arc_lookup, monitored_arc)
            continue
        end
        rebuilt_m = rebuilt_arc_lookup[monitored_arc]

        for (b_idx, bus_num) in enumerate(bus_ax)
            if haskey(rebuilt_bus_lookup, bus_num)
                rb_idx = rebuilt_bus_lookup[bus_num]
                @test isapprox(
                    wb_row[b_idx],
                    ptdf_rebuilt[rebuilt_m, rb_idx],
                    atol = 1e-6,
                )
            end
        end
    end
end

@testset "NetworkModification: full ThreeWindingTransformer outage" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    vptdf = VirtualPTDF(sys)

    # Full 3WT outage should produce 3 arc modifications (one per winding)
    mod = NetworkModification(vptdf, trf)
    @test length(mod.arc_modifications) == 3
    @test mod.label == PSY.get_name(trf)

    # Each modification should have negative delta_b (removing susceptance)
    for am in mod.arc_modifications
        @test am.delta_b < 0
    end

    # Should produce valid PTDF rows
    row = get_post_modification_ptdf_row(vptdf, 1, mod)
    @test length(row) == length(PNM.get_bus_axis(vptdf))
end

@testset "NetworkModification: single ThreeWindingTransformerCircuit outage" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    vptdf = VirtualPTDF(sys)
    nr = PNM.get_network_reduction_data(vptdf)
    arc_lookup = PNM.get_arc_lookup(vptdf)

    # Single winding outage: exactly one arc modification, targeting THIS winding's arc
    # (siblings and the star point are untouched), with a Ybus delta equal to the negated
    # Pi-model of just that winding.
    winding_arc_indices =
        Dict(
            w => arc_lookup[PNM.get_arc_tuple(
                PNM.ThreeWindingTransformerCircuit(trf, w),
                nr,
            )] for w in 1:3
        )
    for w in 1:3
        winding = PNM.ThreeWindingTransformerCircuit(trf, w)
        mod = NetworkModification(vptdf, winding)
        @test length(mod.arc_modifications) == 1
        am = mod.arc_modifications[1]
        @test am.delta_b < 0
        @test am.arc_index == winding_arc_indices[w]
        sibling_indices = [winding_arc_indices[s] for s in 1:3 if s != w]
        @test am.arc_index ∉ sibling_indices
        Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(winding, nr)
        @test am.delta_y11 ≈ ComplexF32(-Y11)
        @test am.delta_y12 ≈ ComplexF32(-Y12)
        @test am.delta_y21 ≈ ComplexF32(-Y21)
        @test am.delta_y22 ≈ ComplexF32(-Y22)
    end
end

@testset "NetworkModification: partial ThreeWindingTransformer (disabled winding)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))

    # Disable one winding before building the matrix
    PSY.set_available!(PSY.get_secondary_circuit(trf), false)
    vptdf = VirtualPTDF(sys)

    # Full 3WT outage should only produce 2 mods (secondary is unavailable)
    mod = NetworkModification(vptdf, trf)
    @test length(mod.arc_modifications) == 2
end

@testset "NetworkModification: ThreeWindingTransformer via Outage attribute" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))

    # Attach an outage supplemental attribute to the 3WT
    outage = GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    add_supplemental_attribute!(sys, trf, outage)
    vptdf = VirtualPTDF(sys)

    # Construct modification via Outage path
    mod = NetworkModification(vptdf, sys, outage)
    @test length(mod.arc_modifications) == 3
    @test !isempty(mod.label)

    # Should produce valid PTDF rows
    row = get_post_modification_ptdf_row(vptdf, 1, mod)
    @test length(row) == length(PNM.get_bus_axis(vptdf))
end

@testset "Woodbury correction concurrent across arcs (AppleAccelerate backend)" begin
    # Validates the AA-backend concurrency claim documented in
    # `src/virtual_ptdf_modification.jl`: many tasks calling
    # `apply_woodbury_correction` on a single VirtualPTDF should agree with the
    # serial baseline. The KLU path is exercised by the threaded testsets in
    # `test/test_virtual_modf.jl`; this complements that coverage on the
    # AppleAccelerate path.
    if !PowerNetworkMatrices._has_apple_accelerate_backend()
        @info "Skipping: AppleAccelerate backend not available on this platform."
        return
    end
    if Threads.nthreads() < 2
        @info "Skipping: requires Threads.nthreads() ≥ 2 to exercise concurrent access."
        return
    end

    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    vptdf = VirtualPTDF(sys; linear_solver = "AppleAccelerate")

    arc_ax = PNM.get_arc_axis(vptdf)
    n_arcs = length(arc_ax)
    @test n_arcs ≥ 2

    outaged_arc = arc_ax[1]
    mod = NetworkModification(vptdf, outaged_arc)
    wf = compute_woodbury_factors(vptdf, mod)

    monitored_indices = collect(2:n_arcs)
    serial = [apply_woodbury_correction(vptdf, m, wf) for m in monitored_indices]

    for iter in 1:5
        parallel = Vector{Vector{Float64}}(undef, length(monitored_indices))
        Threads.@threads :dynamic for i in eachindex(monitored_indices)
            parallel[i] = apply_woodbury_correction(vptdf, monitored_indices[i], wf)
        end
        for i in eachindex(monitored_indices)
            @test parallel[i] ≈ serial[i]
        end
    end
end

@testset "NetworkModification: arc susceptance is the key's own, not the bus pair's" begin
    sys = build_antiparallel_chain_segment_system()
    y = Ybus(sys)
    nr = PNM.get_network_reduction_data(y)
    ba = BA_Matrix(y)
    bus_lookup = PNM.get_bus_lookup(ba)
    vptdf = VirtualPTDF(sys)

    pair_arcs = [a for a in PNM.get_arc_axis(nr) if Set(a) == Set([10, 3])]
    @test length(pair_arcs) == 2

    for arc in pair_arcs
        ix = PNM.get_arc_lookup(vptdf)[arc]
        entry = PNM.get_direct_branch_map(nr)[arc]
        # One BA column is one arc key's own participation in the DC network. Reading the
        # summed Ybus off-diagonal gave both twins the pair total, which ABA counted twice.
        @test isapprox(
            PNM._ba_arc_susceptance(entry, nr),
            ba.data[bus_lookup[arc[1]], ix];
            rtol = 1e-5,
        )
        @test isapprox(PNM._ba_arc_susceptance(entry, nr), 1 / PSY.get_x(entry, PSY.SU))

        # With the two sides agreeing, a full-arc outage negates the π-model rather than
        # scaling it by the ratio of the pair total to one twin.
        mod = NetworkModification(vptdf, arc)
        am = only(mod.arc_modifications)
        Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(entry, nr)
        @test am.delta_y11 ≈ PNM.YBUS_ELTYPE(-Y11)
        @test am.delta_y12 ≈ PNM.YBUS_ELTYPE(-Y12)
        @test am.delta_y21 ≈ PNM.YBUS_ELTYPE(-Y21)
        @test am.delta_y22 ≈ PNM.YBUS_ELTYPE(-Y22)
    end
end

@testset "NetworkModification: grouped degree-two chain leaf trips its whole chain" begin
    sys = build_two_parallel_degree_two_chains()
    reductions = NetworkReduction[
        DegreeTwoReduction(; reduce_reactive_power_injectors = false),
    ]
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(vptdf)
    leaf = PSY.get_component(Line, sys, "L_1_10")

    # Sibling chains collapse into one `BranchesParallel{BranchesSeries}`, so the leaf is
    # filed on the group arc without being a member of the group.
    @test PNM._resolve_branch_arc(nr, leaf) == (:parallel, (1, 3))

    mod = NetworkModification(vptdf, leaf)
    am = only(mod.arc_modifications)
    @test am.arc_index == PNM.get_arc_lookup(vptdf)[(1, 3)]

    bp = PNM.get_parallel_branch_map(nr)[(1, 3)]
    chain = PNM._outaged_group_member(bp, leaf)
    @test any(l === leaf for l in PNM.leaf_components(chain))
    @test am.delta_b ≈ -PNM._finite_series_susceptance(chain, nr)
    @test iszero(am.delta_shift_injection)

    # The arc loses exactly the tripped chain's two-port, not one segment's.
    Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(chain, nr)
    @test am.delta_y11 ≈ PNM.YBUS_ELTYPE(-Y11)
    @test am.delta_y12 ≈ PNM.YBUS_ELTYPE(-Y12)
    @test am.delta_y21 ≈ PNM.YBUS_ELTYPE(-Y21)
    @test am.delta_y22 ≈ PNM.YBUS_ELTYPE(-Y22)

    # The surviving sibling still carries the arc: this is not a full-arc outage.
    @test abs(PNM._get_arc_susceptances(vptdf)[am.arc_index] + am.delta_b) > 1.0

    # A branch nested below the group is not a group member, and the shift injection says so
    # instead of answering with a sign-flipped angle.
    @test_throws ErrorException PNM._member_shift_injection(bp, nr, leaf)
    @test_throws "L_1_10" PNM._member_shift_injection(bp, nr, leaf)
end

@testset "NetworkModification: nested aggregate member requires an all-or-none trip" begin
    # Chain segment 2 is `MixedBranchesParallel[Line, BranchesParallel{Line}]`: the
    # anti-parallel twin is a direct member and the same-direction pair nests below it.
    sys = build_antiparallel_chain_segment_nested_parallel_system()
    reductions = NetworkReduction[
        DegreeTwoReduction(; reduce_reactive_power_injectors = false),
    ]
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(vptdf)
    chain = only(values(PNM.get_series_branch_map(nr)))
    b_old = PNM._finite_series_susceptance(chain, nr)

    p1 = PSY.get_component(Line, sys, "L_10_3")
    p2 = PSY.get_component(Line, sys, "L_10_3_b")
    ap = PSY.get_component(Line, sys, "L_3_10")

    # One nested leaf tripped leaves the `BranchesParallel` it sits inside partly in service;
    # that has no representation on the composite arc it was grouped onto.
    @test_throws "BranchesParallel" PNM._compute_series_outage_delta_b(chain, p1, nr)
    @test_throws r"partly in service" PNM._compute_series_outage_delta_b(chain, p1, nr)

    # Every leaf of the mixed segment tripped -- the nested pair and its anti-parallel twin
    # -- is a full outage of that segment, and therefore of the whole series arc.
    delta_b = PNM._compute_series_outage_delta_b(chain, [p1, p2, ap], nr)
    @test delta_b ≈ -b_old rtol = 1e-8
end

# Two identical phase shifters on the same bus pair, one written each way. Their individual
# two-ports are asymmetric, but the pair's summed off-diagonals are equal, so the pair is
# symmetric only when each member is read in the shared arc frame.
function _mk_antiparallel_identical_pst_system(; alpha = 0.15, x = 0.2, tap = 1.0)
    sys, buses = _mk_bus_system(3)
    function _add_pst!(name, from, to)
        arc = Arc(; from = buses[from], to = buses[to])
        add_component!(sys, arc)
        add_component!(
            sys,
            PSY.TwoWindingTransformer(;
                name = name,
                circuit = PSY.TransformerCircuit(;
                    arc = arc, tap = tap, α = alpha, available = true,
                    active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                    base_power = 100.0, base_voltage_primary = 230.0, r = 0.0, x = x,
                ),
                magnetizing_shunt = Complex(0.0, 0.0),
            ),
        )
        return nothing
    end
    _add_pst!("PST_A", 1, 2)
    _add_pst!("PST_B", 2, 1)
    arc = Arc(; from = buses[2], to = buses[3])
    add_component!(sys, arc)
    add_component!(
        sys,
        Line(;
            name = "L23", available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = arc, r = 0.0, x = 0.1,
            b = (from = 0.0, to = 0.0), rating = 1.0,
            angle_limits = (min = -1.5, max = 1.5),
        ),
    )
    return sys
end

@testset "NetworkModification: anti-parallel pair is summed in the shared arc frame" begin
    sys = _mk_antiparallel_identical_pst_system()
    y = Ybus(sys)
    nr = PNM.get_network_reduction_data(y)
    ba = BA_Matrix(y)
    bus_lookup = PNM.get_bus_lookup(ba)
    vptdf = VirtualPTDF(sys)

    pair_arcs = [a for a in PNM.get_arc_axis(nr) if Set(a) == Set([1, 2])]
    @test length(pair_arcs) == 2

    # Each member alone is asymmetric; the pair read in one shared frame is symmetric only
    # because `_subset_two_port` transposes the anti-frame twin. Flipping that transpose
    # leaves the sum asymmetric, and nothing else in the suite notices.
    entries = [PNM.get_direct_branch_map(nr)[a] for a in pair_arcs]
    _, pair_Y12, pair_Y21, _ = PNM._subset_two_port(entries, first(pair_arcs), nr)
    @test isapprox(pair_Y12, pair_Y21)

    for arc in pair_arcs
        entry = PNM.get_direct_branch_map(nr)[arc]
        _, own_Y12, own_Y21, _ = PNM.ybus_branch_entries(entry, nr)
        @test own_Y12 != own_Y21
        # A shifter's own column takes the phase-independent component value, and it is that
        # key's alone: summing the pair here gave both keys the pair total.
        @test isapprox(
            PNM._ba_arc_susceptance(entry, nr),
            PNM._finite_series_susceptance(entry, nr),
        )
        @test isapprox(
            PNM._ba_arc_susceptance(entry, nr),
            ba.data[bus_lookup[arc[1]], PNM.get_arc_lookup(vptdf)[arc]];
            rtol = 1e-5,
        )

        mod = NetworkModification(vptdf, arc)
        am = only(mod.arc_modifications)
        Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(entry, nr)
        @test am.delta_y11 ≈ PNM.YBUS_ELTYPE(-Y11)
        @test am.delta_y12 ≈ PNM.YBUS_ELTYPE(-Y12)
        @test am.delta_y21 ≈ PNM.YBUS_ELTYPE(-Y21)
        @test am.delta_y22 ≈ PNM.YBUS_ELTYPE(-Y22)
    end
end

@testset "NetworkModification: full outage of a negative-susceptance arc negates its Pi-model" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = PSY.get_component(Line, sys, "Line10")
    PSY.set_r!(line, 0.0 * PSY.SU)
    PSY.set_x!(line, -0.1 * PSY.SU)
    vptdf = VirtualPTDF(sys)
    nr = PNM.get_network_reduction_data(vptdf)
    arc = PNM.get_arc_tuple(line, nr)

    # `_extract_arc_susceptances` takes the magnitude of the BA column, so every Δb reaching
    # the outage handlers is `-|b|` while the arc's own susceptance keeps its sign. The two
    # conventions only differ on an arc with net negative reactance, which is what makes
    # this fixture bite.
    b_arc = PNM._ba_arc_susceptance(line, nr)
    @test b_arc < 0
    @test PNM._get_arc_susceptances(vptdf)[PNM.get_arc_lookup(vptdf)[arc]] ≈ abs(b_arc)

    mod = NetworkModification(vptdf, line)
    am = only(mod.arc_modifications)
    # Pins the magnitude convention as it stands, not as established truth: the MODF Woodbury
    # update disagrees with a direct oracle on negative-susceptance arcs.
    @test am.delta_b ≈ -abs(b_arc)
    Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(line, nr)
    @test am.delta_y11 ≈ PNM.YBUS_ELTYPE(-Y11)
    @test am.delta_y12 ≈ PNM.YBUS_ELTYPE(-Y12)
    @test am.delta_y21 ≈ PNM.YBUS_ELTYPE(-Y21)
    @test am.delta_y22 ≈ PNM.YBUS_ELTYPE(-Y22)

    # The Ybus the outage leaves behind has no trace of the branch. Scaling by `delta_b /
    # b_arc` on a signed `b_arc` returns `+1`, which doubles the branch instead.
    modified = apply_ybus_modification(Ybus(sys), mod)
    bus_lookup = PNM.get_bus_lookup(vptdf)
    f = bus_lookup[arc[1]]
    t = bus_lookup[arc[2]]
    @test abs(modified[f, t]) < 1e-5
end

# `build_two_parallel_degree_two_chains` with chain A's second segment replaced by a phase
# shifter, so exactly one of the two sibling chains carries an angle.
function _mk_shifted_grouped_chain_system(; alpha = 0.15, pst_x = 0.2)
    sys = _build_degree_two_chain_system([
        (1, 2, 0.0, 0.05, 0.0, 0.0), (2, 3, 0.0, 0.06, 0.0, 0.0),
        (3, 4, 0.0, 0.07, 0.0, 0.0), (4, 1, 0.0, 0.08, 0.0, 0.0),
        (2, 4, 0.0, 0.09, 0.0, 0.0),
        (1, 10, 0.0, 0.10, 0.0, 0.0),
        (1, 20, 0.0, 0.20, 0.0, 0.0), (20, 3, 0.0, 0.21, 0.0, 0.0),
    ])
    arc = Arc(;
        from = PSY.get_component(ACBus, sys, "Bus 10"),
        to = PSY.get_component(ACBus, sys, "Bus 3"),
    )
    add_component!(sys, arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST_10_3",
            circuit = PSY.TransformerCircuit(;
                arc = arc, tap = 1.0, α = alpha, available = true,
                active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                base_power = 100.0, base_voltage_primary = 230.0, r = 0.0, x = pst_x,
            ),
            magnetizing_shunt = Complex(0.0, 0.0),
        ),
    )
    return sys
end

@testset "NetworkModification: grouped chain loses its share of the arc shift injection" begin
    sys = _mk_shifted_grouped_chain_system()
    reductions = NetworkReduction[
        DegreeTwoReduction(; reduce_reactive_power_injectors = false),
    ]
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(vptdf)
    bp = PNM.get_parallel_branch_map(nr)[(1, 3)]
    @test length(bp) == 2

    # Only one sibling is shifted, so the whole arc injection is that chain's share. This is
    # the independent oracle: `arc_dc_shift_injection` reaches it through the group's
    # susceptance-weighted equivalent angle, `_member_shift_injection` through the member's
    # own `b·α`, and the two must agree.
    arc_injection = PNM.arc_dc_shift_injection(nr, (1, 3))
    @test arc_injection ≈ 0.5
    @test !iszero(arc_injection)

    shifted = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST_10_3")
    unshifted = PSY.get_component(Line, sys, "L_1_20")

    for leaf in (PSY.get_component(Line, sys, "L_1_10"), shifted)
        am = only(NetworkModification(vptdf, leaf).arc_modifications)
        @test am.delta_shift_injection ≈ -arc_injection
        @test am.delta_b ≈ -1 / (1 / 10.0 + 1 / 5.0)
    end

    # Tripping the unshifted sibling leaves the arc's whole injection in place.
    am = only(NetworkModification(vptdf, unshifted).arc_modifications)
    @test iszero(am.delta_shift_injection)
    @test am.delta_b ≈ -1 / (1 / 5.0 + 1 / (1 / 0.21))
end
