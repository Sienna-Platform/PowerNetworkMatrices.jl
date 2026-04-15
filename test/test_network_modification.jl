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
        @test outaged_line !== nothing
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
        ctg_uuid = Base.UUID(UInt128(e + 10000))
        ctg = ContingencySpec(
            ctg_uuid,
            NetworkModification("test_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_uuid] = ctg

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
