# Removing a surviving arc with very small reactance leaves W ≈ -x_s²/x_alt, far below the
# islanding tolerance in absolute terms, even though the network stays connected.

function _low_impedance_rows_match_rebuild(sys_builder, line_names::Vector{String})
    sys = sys_builder()
    for name in line_names
        line = PSY.get_component(PSY.Line, sys, name)
        PSY.set_r!(line, 1e-4u"SU")
        PSY.set_x!(line, 1e-6u"SU")
    end
    vptdf = VirtualPTDF(sys)
    mods = ArcModification[]
    for name in line_names
        line = PSY.get_component(PSY.Line, sys, name)
        append!(mods, NetworkModification(vptdf, line).arc_modifications)
    end
    wf = compute_woodbury_factors(vptdf, NetworkModification("low_x", mods))
    @test !wf.is_islanding

    sys_ref = sys_builder()
    for name in line_names
        line = PSY.get_component(PSY.Line, sys_ref, name)
        PSY.set_r!(line, 1e-4u"SU")
        PSY.set_x!(line, 1e-6u"SU")
        PSY.set_available!(line, false)
    end
    ptdf_ref = PTDF(sys_ref)
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
    return
end

_c_sys5() = PSB.build_system(PSB.PSITestSystems, "c_sys5")
_c_sys14() = PSB.build_system(PSB.PSITestSystems, "c_sys14")

# The first `k` lines (by name) whose joint removal keeps the network connected.
function _connected_removal(sys_builder, k::Int)
    names = sort!([PSY.get_name(l) for l in PSY.get_components(PSY.Line, sys_builder())])
    chosen = String[]
    for name in names
        sys_try = sys_builder()
        for n in vcat(chosen, name)
            PSY.set_available!(PSY.get_component(PSY.Line, sys_try, n), false)
        end
        length(PNM.find_subnetworks(PNM.Ybus(sys_try))) == 1 && push!(chosen, name)
        length(chosen) == k && return chosen
    end
    error("No connected removal of $k lines found")
end

@testset "Low-impedance arc removal is not islanding (M = 1)" begin
    _low_impedance_rows_match_rebuild(_c_sys5, _connected_removal(_c_sys5, 1))
end

@testset "Low-impedance arc removal is not islanding (M = 2)" begin
    _low_impedance_rows_match_rebuild(_c_sys5, _connected_removal(_c_sys5, 2))
end

@testset "Low-impedance arc removal is not islanding (M = 3)" begin
    _low_impedance_rows_match_rebuild(_c_sys14, _connected_removal(_c_sys14, 3))
end

@testset "A true bridge is still detected as islanding" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    vptdf = VirtualPTDF(sys)
    bridge_found = false
    for line in sort!(collect(PSY.get_components(PSY.Line, sys)); by = PSY.get_name)
        sys_try = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        PSY.set_available!(PSY.get_component(PSY.Line, sys_try, PSY.get_name(line)), false)
        length(PNM.find_subnetworks(PNM.Ybus(sys_try))) > 1 || continue
        mod = NetworkModification(vptdf, line)
        isempty(mod.arc_modifications) && continue
        wf = compute_woodbury_factors(vptdf, mod)
        @test wf.is_islanding
        bridge_found = true
        break
    end
    @test bridge_found
end
