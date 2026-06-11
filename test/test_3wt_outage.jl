import LinearAlgebra as LA

@testset "full 3WT outage matches deflated-ABA reference (case10)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys)
    bus_ax = PNM.get_bus_axis(vptdf)
    arc_ax = PNM.get_arc_axis(vptdf)
    BA = vptdf.BA
    A = vptdf.A
    valid = vptdf.valid_ix
    nbus = length(bus_ax)

    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    mod = NetworkModification(vptdf, trf)
    wind = sort([am.arc_index for am in mod.arc_modifications])

    # Ground truth: post-outage ABA over valid buses (winding arcs removed),
    # solved with the Moore-Penrose pseudo-inverse, which is the convention the
    # production pinv-based Woodbury path implements for islanding.
    BAm = Matrix(BA)
    for e in wind
        BAm[:, e] .= 0.0
    end
    ABAm = (BAm * A)[valid, valid]
    mon = first(setdiff(1:length(arc_ax), wind))
    rhs = Vector(BA[:, mon])[valid]
    x_ref = LA.pinv(Matrix(ABAm); atol = 1e-8) * rhs
    ref = zeros(nbus)
    ref[valid] .= x_ref

    row = get_post_modification_ptdf_row(vptdf, mon, mod)

    # Compare only on buses still connected post-outage (non-zero ABA row).
    keep = [valid[k] for k in 1:length(valid) if !all(iszero, ABAm[k, :])]
    @test maximum(abs.(row[keep] .- ref[keep])) < 1e-7
end

@testset "star bus column is ~zero in post-mod row" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys)
    bus_ax = PNM.get_bus_axis(vptdf)
    arc_ax = PNM.get_arc_axis(vptdf)
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    mod = NetworkModification(vptdf, trf)
    wind = sort([am.arc_index for am in mod.arc_modifications])
    star_num = unique([arc_ax[e][2] for e in wind])[1]
    star_pos = findfirst(==(star_num), bus_ax)
    mon = first(setdiff(1:length(arc_ax), wind))
    row = get_post_modification_ptdf_row(vptdf, mon, mod)
    # Isolated zero-injection star bus carries no sensitivity.
    @test isapprox(row[star_pos], 0.0; atol = 1e-9)
end

@testset "is_islanding is true for radial-terminal full 3WT outage" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys)
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    mod = NetworkModification(vptdf, trf)
    @test mod.is_islanding == true
end

@testset "is_islanding via Outage attribute path" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    outage = PSY.GeometricDistributionForcedOutage(;
        mean_time_to_recovery = 0.0,
        outage_transition_probability = 0.0,
    )
    PSY.add_supplemental_attribute!(sys, trf, outage)
    vptdf = VirtualPTDF(sys)
    mod = NetworkModification(vptdf, sys, outage)
    @test mod.is_islanding == true
end

@testset "outaged winding arc returns zero row" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys)
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    mod = NetworkModification(vptdf, trf)
    wind = [am.arc_index for am in mod.arc_modifications]
    for e in wind
        row = get_post_modification_ptdf_row(vptdf, e, mod)
        @test all(iszero, row)
    end
end

@testset "VirtualMODF cache path equals one-shot (full 3WT)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(sys)
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    mod = NetworkModification(vptdf, trf)
    arc_ax = PNM.get_arc_axis(vptdf)
    wind = Set(am.arc_index for am in mod.arc_modifications)
    mon = first(setdiff(1:length(arc_ax), wind))
    oneshot = get_post_modification_ptdf_row(vptdf, mon, mod)
    cached = vmodf[mon, mod]
    @test isapprox(oneshot, cached; atol = 1e-9)
end

@testset "Ybus delta matches star-surgery rebuild on surviving buses" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    ybus0 = Ybus(sys)
    vptdf = VirtualPTDF(sys)
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    mod = NetworkModification(vptdf, trf)
    arc_ax = PNM.get_arc_axis(vptdf)
    wind = [am.arc_index for am in mod.arc_modifications]
    star_num = unique([arc_ax[e][2] for e in wind])[1]

    # Raw post-modification Ybus (original-system positions, star row/col zeroed).
    ybus_mod = apply_ybus_modification(ybus0, mod)
    lookup_mod = PNM.get_bus_lookup(ybus0)

    # Star-surgery rebuild: remove 3WT + orphan star bus.
    sys2 = deepcopy(sys)
    trf2 = first(PSY.get_components(PSY.ThreeWindingTransformer, sys2))
    PSY.remove_component!(sys2, trf2)
    star_bus = first(
        b for b in PSY.get_components(PSY.ACBus, sys2)
        if PSY.get_number(b) == star_num
    )
    PSY.remove_component!(sys2, star_bus)
    ybus_ref = Ybus(sys2)
    ref_data = ybus_ref.data
    lookup_ref = PNM.get_bus_lookup(ybus_ref)

    # Compare entries for every real bus pair present in the surgery system.
    bus_ax_ref = PNM.get_bus_axis(ybus_ref)
    for i in bus_ax_ref, j in bus_ax_ref
        # ComplexF32 entries: compare with float32-appropriate relative tol.
        @test isapprox(
            ybus_mod[lookup_mod[i], lookup_mod[j]],
            ref_data[lookup_ref[i], lookup_ref[j]];
            atol = 1e-4,
            rtol = 1e-4,
        )
    end
end
