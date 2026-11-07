_c_sys14() = PSB.build_system(PSB.PSITestSystems, "c_sys14")

function _max_row_error(rows, builder, out_names, vptdf; skip_arc = nothing)
    sys_ref = builder()
    for n in out_names
        PSY.set_available!(PSY.get_component(PSY.Line, sys_ref, n), false)
    end
    ref = PTDF(sys_ref)
    arc_lookup = PNM.get_arc_lookup(ref)
    bus_lookup = PNM.get_bus_lookup(ref)
    err = 0.0
    for arc in PNM.get_arc_axis(vptdf)
        (!haskey(arc_lookup, arc) || arc == skip_arc) && continue
        row = rows(arc)
        for (j, bus) in enumerate(PNM.get_bus_axis(vptdf))
            err = max(err, abs(row[j] - ref[arc_lookup[arc], bus_lookup[bus]]))
        end
    end
    return err
end

@testset "Parallel group outages with unequal r/x match a rebuilt PTDF" begin
    for r_scale in (1.0, 5.0)
        builder = () -> _c_sys14_with_parallel_pair(; r_scale = r_scale)[1]
        sys, base, dup = _c_sys14_with_parallel_pair(; r_scale = r_scale)
        vptdf = VirtualPTDF(sys)
        arc = _arc_numbers(base)
        nr = PNM.get_network_reduction_data(vptdf)
        @test haskey(PNM.get_parallel_branch_map(nr), arc)
        arc_idx = PNM.get_arc_lookup(vptdf)[arc]

        partial = NetworkModification(vptdf, sys, [dup])
        full = NetworkModification(vptdf, sys, [base, dup])
        @test only(partial.arc_modifications).opened == 1
        @test only(full.arc_modifications).opened == 2
        b_arc = PNM._get_arc_susceptances(vptdf)[arc_idx]
        @test only(full.arc_modifications).delta_b == -b_arc
        # a single-branch constructor goes through the same joint rule
        @test NetworkModification(vptdf, dup).arc_modifications ==
              partial.arc_modifications

        for (mod, names, skip) in
            ((partial, ["dup"], nothing), (full, ["dup", PSY.get_name(base)], arc))
            err = _max_row_error(
                a -> vptdf[a, mod], builder, names, vptdf; skip_arc = skip,
            )
            @test err < 1e-9
        end
    end
end

@testset "Ward equivalent on the arc of a tripped branch keeps conducting" begin
    sys = _c_sys14()
    # Buses 4 and 5 are boundary buses joined by a line, so the Ward equivalent of the
    # external network lands on the line's own arc.
    reductions = NetworkReduction[WardReduction([1, 2, 3, 4, 5])]
    vptdf = VirtualPTDF(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(vptdf)
    @test length(PNM.get_bus_axis(vptdf)) >= 4
    arc = (4, 5)
    line = PNM.get_direct_branch_map(nr)[arc]
    @test haskey(PNM.get_added_arc_impedance_map(nr), arc)
    arc_idx = PNM.get_arc_lookup(vptdf)[arc]
    b_arc = PNM._get_arc_susceptances(vptdf)[arc_idx]
    b_ward = PNM._added_arc_susceptance(nr, arc)
    @test b_ward > 0

    mod = NetworkModification(vptdf, sys, [line])
    m = only(mod.arc_modifications)
    @test m.delta_b ≈ b_ward - b_arc
    @test b_arc + m.delta_b ≈ b_ward
    @test PNM.get_equivalent_rating(PNM.get_added_arc_impedance_map(nr)[arc]) == 1e6

    # The line sits inside the study area, so removing it leaves the Ward equivalent as is.
    PSY.set_available!(line, false)
    ref = VirtualPTDF(sys; network_reductions = reductions)
    PSY.set_available!(line, true)
    @test Set(PNM.get_arc_axis(ref)) == Set(PNM.get_arc_axis(vptdf))
    @test PNM.get_bus_axis(ref) == PNM.get_bus_axis(vptdf)
    @test all(
        isapprox(vptdf[a, mod], Vector(get_ptdf_row(ref, a)); atol = 1e-9) for
        a in PNM.get_arc_axis(vptdf)
    )
    @test !isapprox(vptdf[arc, mod], Vector(get_ptdf_row(vptdf, arc)); atol = 1e-6)
end
