function _uniform_dist_slack(sys)
    buses = [PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)]
    return Dict(b => 1.0 / length(buses) for b in buses)
end

@testset "Post-modification rows honor distributed slack" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    dist = _uniform_dist_slack(sys)
    vptdf = VirtualPTDF(sys; dist_slack = dist)
    line = first(sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name))
    wf = compute_woodbury_factors(vptdf, NetworkModification(vptdf, line))

    sys_ref = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    PSY.set_available!(PSY.get_component(PSY.Line, sys_ref, PSY.get_name(line)), false)
    ptdf_ref = PTDF(sys_ref; dist_slack = _uniform_dist_slack(sys_ref))
    ref_arc_lookup = PNM.get_arc_lookup(ptdf_ref)
    ref_bus_lookup = PNM.get_bus_lookup(ptdf_ref)
    for (m, arc) in enumerate(PNM.get_arc_axis(vptdf))
        haskey(ref_arc_lookup, arc) || continue
        row = apply_woodbury_correction(vptdf, m, wf)
        for (b, bus) in enumerate(PNM.get_bus_axis(vptdf))
            @test isapprox(
                row[b],
                ptdf_ref[ref_arc_lookup[arc], ref_bus_lookup[bus]];
                atol = 1e-6,
            )
        end
    end
end

@testset "Distributed slack with an islanding modification errors" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys; dist_slack = _uniform_dist_slack(sys))
    islanding_found = false
    for line in sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
        mod = NetworkModification(vptdf, line)
        isempty(mod.arc_modifications) && continue
        wf = compute_woodbury_factors(vptdf, mod)
        wf.is_islanding || continue
        @test_throws ErrorException apply_woodbury_correction(vptdf, 1, wf)
        islanding_found = true
        break
    end
    @test islanding_found
end

@testset "Rows on several modifications honor distributed slack bit for bit" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys; dist_slack = _uniform_dist_slack(sys))
    lines = sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    mods = [
        NetworkModification("none", PNM.ArcModification[]);
        [NetworkModification(vptdf, line) for line in lines];
        NetworkModification(vptdf, sys, lines[1:2])
    ]
    for arc in PNM.get_arc_axis(vptdf)
        rows = get_post_modification_ptdf_rows(vptdf, arc, mods)
        for (mod, row) in zip(mods, rows)
            expected = vptdf[arc, mod]
            @test typeof(row) == typeof(expected) && isequal(row, expected)
        end
    end
end

@testset "Rows on several modifications with distributed slack reject islanding" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys; dist_slack = _uniform_dist_slack(sys))
    mods = [
        NetworkModification(vptdf, line) for
        line in sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
    ]
    filter!(mod -> !isempty(mod.arc_modifications), mods)
    @test any(mod -> compute_woodbury_factors(vptdf, mod).is_islanding, mods)
    @test_throws ErrorException get_post_modification_ptdf_rows(vptdf, 1, mods)
end

_row_batch_component(sys, name) = something(
    PSY.get_component(PSY.Line, sys, name),
    PSY.get_component(PSY.TwoWindingTransformer, sys, name),
)

const _ORACLE_CASES = [
    String[],
    ["Line7"],
    ["dup"],
    ["shifter"],
    ["Line1", "dup"],
    ["Line7", "dup", "shifter"],
]

@testset "Rows on several modifications with distributed slack match rebuilt systems" begin
    sys = _c_sys14_for_row_batches()
    uniform(s) = Dict(
        PSY.get_number(b) => 1.0 / length(PSY.get_components(PSY.ACBus, s)) for
        b in PSY.get_components(PSY.ACBus, s)
    )
    vptdf = VirtualPTDF(sys; dist_slack = uniform(sys))
    mods = [
        NetworkModification(vptdf, sys, [_row_batch_component(sys, n) for n in names])
        for
        names in _ORACLE_CASES
    ]
    arcs = PNM.get_arc_axis(vptdf)
    rows = Dict(arc => get_post_modification_ptdf_rows(vptdf, arc, mods) for arc in arcs)
    for arc in arcs, (k, mod) in enumerate(mods)
        @test _bitwise_equal(rows[arc][k], vptdf[arc, mod])
    end
    for (k, names) in enumerate(_ORACLE_CASES)
        sys_ref = _c_sys14_for_row_batches()
        err = _max_error_against_rebuild(
            a -> rows[a][k],
            arcs,
            PNM.get_bus_axis(vptdf),
            names;
            dist_slack = uniform(sys_ref),
        )
        @test err < 1e-10
    end
end
