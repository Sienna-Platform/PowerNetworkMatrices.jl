# Distributed slack must use one convention per matrix object: post-modification
# (Woodbury) rows apply the same slack adjustment as base rows.

function _set_arc_unavailable!(sys, arc_tuple::Tuple{Int, Int})
    for l in PSY.get_components(PSY.ACBranch, sys)
        arc = PSY.get_arc(l)
        if (arc.from.number, arc.to.number) == arc_tuple
            PSY.set_available!(l, false)
            return
        end
    end
    return error("No branch with arc $arc_tuple found in system")
end

# Non-uniform weights keyed by bus number; a missed slack adjustment cannot
# cancel out under these.
function _nonuniform_slack_dict(bus_ax::Vector{Int})
    return Dict(n => Float64(i) for (i, n) in enumerate(bus_ax))
end

@testset "VirtualPTDF post-modification rows apply dist_slack" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    A = IncidenceMatrix(sys)
    arc_ax = PNM.get_arc_axis(A)
    bus_ax = PNM.get_bus_axis(A)
    slack_dict = _nonuniform_slack_dict(bus_ax)
    vptdf = VirtualPTDF(sys; dist_slack = slack_dict)

    for e in eachindex(arc_ax)
        outaged_arc = arc_ax[e]
        mod = NetworkModification(vptdf, outaged_arc)

        # Reference: dense PTDF with the same dist_slack on a system with the
        # outaged line disabled.
        sys_mod = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        _set_arc_unavailable!(sys_mod, outaged_arc)
        ptdf_rebuilt = PTDF(sys_mod; dist_slack = slack_dict)
        rebuilt_arc_lookup = PNM.get_arc_lookup(ptdf_rebuilt)
        rebuilt_bus_lookup = PNM.get_bus_lookup(ptdf_rebuilt)

        for m in eachindex(arc_ax)
            m == e && continue
            monitored_arc = arc_ax[m]
            if !haskey(rebuilt_arc_lookup, monitored_arc)
                continue
            end
            wb_row = vptdf[m, mod]
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
end

@testset "VirtualMODF rows apply dist_slack" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    A = IncidenceMatrix(sys)
    arc_ax = PNM.get_arc_axis(A)
    bus_ax = PNM.get_bus_axis(A)
    slack_dict = _nonuniform_slack_dict(bus_ax)
    # VirtualMODF takes positional weights aligned with its bus axis.
    weights = [slack_dict[n] for n in bus_ax]
    vmodf = VirtualMODF(sys; dist_slack = weights)

    e = 2
    b_e = vmodf.arc_susceptances[e]
    ctg_uuid = Base.UUID(UInt128(4242))
    ctg = ContingencySpec(
        ctg_uuid,
        NetworkModification("outage_arc_$e", [ArcModification(e, -b_e)]),
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    sys_mod = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    _set_arc_unavailable!(sys_mod, arc_ax[e])
    ptdf_rebuilt = PTDF(sys_mod; dist_slack = slack_dict)
    rebuilt_arc_lookup = PNM.get_arc_lookup(ptdf_rebuilt)
    rebuilt_bus_lookup = PNM.get_bus_lookup(ptdf_rebuilt)

    for m in eachindex(arc_ax)
        m == e && continue
        monitored_arc = arc_ax[m]
        if !haskey(rebuilt_arc_lookup, monitored_arc)
            continue
        end
        row = vmodf[m, ctg]
        rebuilt_m = rebuilt_arc_lookup[monitored_arc]
        for (b_idx, bus_num) in enumerate(bus_ax)
            if haskey(rebuilt_bus_lookup, bus_num)
                rb_idx = rebuilt_bus_lookup[bus_num]
                @test isapprox(
                    row[b_idx],
                    ptdf_rebuilt[rebuilt_m, rb_idx],
                    atol = 1e-6,
                )
            end
        end
    end
end

@testset "VirtualMODF dist_slack validation at construction" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    n_bus = length(PNM.get_bus_axis(IncidenceMatrix(sys)))
    # Weight vector length must match the (reduced) bus axis.
    @test_throws ErrorException VirtualMODF(sys; dist_slack = ones(n_bus + 1))

    # Distributed slack requires a single reference bus.
    sys2 = PSB.build_system(PSISystems, "2Area 5 Bus System")
    n_bus2 = length(PNM.get_bus_axis(IncidenceMatrix(sys2)))
    @test_throws ErrorException VirtualMODF(sys2; dist_slack = ones(n_bus2))
end

@testset "dist_slack with islanding modification errors" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    bus_ax = PNM.get_bus_axis(IncidenceMatrix(sys))
    slack_dict = _nonuniform_slack_dict(bus_ax)
    vptdf = VirtualPTDF(sys; dist_slack = slack_dict)

    # Full outage of a bridge arc islands part of the network; a distributed
    # slack over the whole pre-contingency network is ill-defined there.
    vmodf_probe = VirtualMODF(sys)
    e = _find_bridge_arc(vmodf_probe)
    outaged_arc = PNM.get_arc_axis(vptdf)[e]
    mod = NetworkModification(vptdf, outaged_arc)
    monitored = e == 1 ? 2 : 1
    @test_throws ErrorException vptdf[monitored, mod]
end
