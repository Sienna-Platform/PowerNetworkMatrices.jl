# Tests that a VirtualMODF built with outages + network reductions keeps every
# outaged component AND every monitored component (declared on the outage via
# `monitored_components`) expressible as an arc, so post-contingency PTDF queries
# never silently return the unmodified base row or crash with a raw KeyError.

# --- helpers ---------------------------------------------------------------

# Build a reduced VirtualMODF and an unreduced one on a system with NO outages
# yet, then return the set of arcs the reduction eliminates (present full, absent
# reduced). With no outages there is nothing to protect, so the reduction runs
# fully — this is exactly the set of arcs that WOULD be lost without protection.
function _rts_reduced_away_arcs()
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    base = VirtualMODF(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )
    full = VirtualMODF(sys)
    reduced_away = setdiff(
        Set(keys(PNM.get_arc_lookup(full))),
        Set(keys(PNM.get_arc_lookup(base))),
    )
    return sys, full, reduced_away
end

# First non-phase-shifter branch whose reduction-mapped arc is in `arcs`.
function _branch_on_arcs(sys, nrd, arcs)
    for br in PSY.get_components(PSY.ACTransmission, sys)
        _is_phase_shifting_2w(br) && continue
        if PNM.get_arc_tuple(br, nrd) in arcs
            return br
        end
    end
    return nothing
end

_fixed_outage(; monitored = Int[]) =
    PSY.FixedForcedOutage(; outage_status = 0.0, monitored_components = monitored)

function _arc_buses(branch)
    arc = PSY.get_arc(branch)
    return PSY.get_number(PSY.get_from(arc)), PSY.get_number(PSY.get_to(arc))
end

# The protected buses are not stored separately on the MODF. The reduction data
# is the single source of truth for which buses survived reduction: the keys of
# `bus_reduction_map` are exactly the retained buses, for every reduction type
# (Radial/DegreeTwo via `irreducible_buses`, Ward via `study_buses`).
_retained_buses(vmodf) =
    Set(keys(PNM.get_bus_reduction_map(PNM.get_network_reduction_data(vmodf))))

# --- Phase 1: pure helpers -------------------------------------------------

@testset "collect protected buses from outaged component" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    outaged = PSY.ACTransmission[]
    for name in ["1", "2"]
        line = PSY.get_component(PSY.ACTransmission, sys, name)
        @test line !== nothing
        push!(outaged, line)
        PSY.add_supplemental_attribute!(sys, line, _fixed_outage())
    end

    protected = PNM._collect_protected_buses(sys)
    for line in outaged
        fb, tb = _arc_buses(line)
        @test fb in protected
        @test tb in protected
    end
end

@testset "collect protected buses from monitored components on the outage" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    outaged = PSY.get_component(PSY.ACTransmission, sys, "1")
    monitored = PSY.get_component(PSY.ACTransmission, sys, "4")
    # The monitored set is carried ON the outage as device UUIDs.
    PSY.add_supplemental_attribute!(
        sys,
        outaged,
        _fixed_outage(; monitored = [monitored]),
    )

    protected = PNM._collect_protected_buses(sys)
    # Both the outaged branch and its monitored branch contribute their buses.
    for branch in (outaged, monitored)
        fb, tb = _arc_buses(branch)
        @test fb in protected
        @test tb in protected
    end
end

@testset "collect protected buses handles ThreeWindingTransformer outage without crashing" begin
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_4_zero_impedance_3wt_test_system",
    )
    t3w = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
    PSY.add_supplemental_attribute!(sys, t3w, _fixed_outage())

    # Regression: Transformer3W <: ACTransmission but has no `get_arc`, so the
    # generic ACTransmission path would throw MethodError. It must route to the
    # 3WT-specific method and protect all of the transformer's buses.
    protected = PNM._collect_protected_buses(sys)
    @test !isempty(protected)
    for arc in PSY.get_arc.(PSY.get_circuits(t3w))
        @test PSY.get_number(PSY.get_from(arc)) in protected
        @test PSY.get_number(PSY.get_to(arc)) in protected
    end
end

# --- Phase 2/3: constructor wiring + validation ----------------------------

@testset "VirtualMODF auto-protects outaged branch under reduction (RTS)" begin
    sys, full, reduced_away = _rts_reduced_away_arcs()
    @test !isempty(reduced_away)  # sanity: reduction actually removed arcs

    target = _branch_on_arcs(sys, get_network_reduction_data(full), reduced_away)
    @test target !== nothing

    PSY.add_supplemental_attribute!(sys, target, _fixed_outage())
    protected_modf = VirtualMODF(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )

    fb, tb = _arc_buses(target)
    @test fb in _retained_buses(protected_modf)
    @test tb in _retained_buses(protected_modf)

    arc_lookup = PNM.get_arc_lookup(protected_modf)
    @test haskey(arc_lookup, (fb, tb)) || haskey(arc_lookup, (tb, fb))

    outage = PSY.get_supplemental_attributes(target)[1]
    ctg = get_registered_contingencies(protected_modf)[IS.get_id(outage)]
    @test !isempty(ctg.modification.arc_modifications)
end

@testset "VirtualMODF protects monitored component declared on the outage (RTS)" begin
    sys, full, reduced_away = _rts_reduced_away_arcs()
    nrd_full = get_network_reduction_data(full)

    # Monitored branch sits on a reduced-away arc; the outaged branch can be any
    # retained branch.
    monitored = _branch_on_arcs(sys, nrd_full, reduced_away)
    @test monitored !== nothing
    outaged = nothing
    for br in PSY.get_components(PSY.ACTransmission, sys)
        _is_phase_shifting_2w(br) && continue
        br === monitored && continue
        outaged = br
        break
    end
    @test outaged !== nothing

    PSY.add_supplemental_attribute!(
        sys,
        outaged,
        _fixed_outage(; monitored = [monitored]),
    )
    vmodf = VirtualMODF(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )

    mfb, mtb = _arc_buses(monitored)
    @test mfb in _retained_buses(vmodf)
    @test mtb in _retained_buses(vmodf)
    arc_lookup = PNM.get_arc_lookup(vmodf)
    @test haskey(arc_lookup, (mfb, mtb)) || haskey(arc_lookup, (mtb, mfb))
end

@testset "VirtualMODF protects monitored branch endpoint merged by zero-impedance reduction" begin
    # `psse_14_network_reduction_test_system` has a zero-impedance branch (104, 105);
    # bus 105 is merged into bus 104 by the auto-applied ZeroImpedanceBranchReduction
    # when neither bus is irreducible (see "ZeroImpedanceBranchReduction respects
    # irreducible buses" in test_ybus_reductions.jl). The monitored branch below has
    # bus 105 as an endpoint, so its (102, 105) arc must survive — bus 105 needs to be
    # irreducible *before* the base Ybus (and its ZIBR pass) is built, not after.
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    monitored = PSY.get_component(PSY.ACTransmission, sys, "BUS 102_102-BUS 105_105-i_1")
    outaged = PSY.get_component(PSY.ACTransmission, sys, "BUS 107_107-BUS 108_108-i_1")
    @test monitored !== nothing
    @test outaged !== nothing

    PSY.add_supplemental_attribute!(
        sys,
        outaged,
        _fixed_outage(; monitored = [monitored]),
    )
    vmodf = VirtualMODF(sys)

    mfb, mtb = _arc_buses(monitored)
    @test mfb in _retained_buses(vmodf)
    @test mtb in _retained_buses(vmodf)
    arc_lookup = PNM.get_arc_lookup(vmodf)
    @test haskey(arc_lookup, (mfb, mtb)) || haskey(arc_lookup, (mtb, mfb))
end

# Minimal capturing logger: ReTest cannot `record` a `@test_logs` failure, so we
# capture log records directly and assert on them.
mutable struct _CollectLogs <: Logging.AbstractLogger
    records::Vector{Tuple{Any, String}}
end
Logging.min_enabled_level(::_CollectLogs) = Logging.Debug
Logging.shouldlog(::_CollectLogs, args...) = true
Logging.catch_exceptions(::_CollectLogs) = false
function Logging.handle_message(
    l::_CollectLogs,
    level,
    message,
    _module,
    group,
    id,
    file,
    line;
    kwargs...,
)
    push!(l.records, (level, string(message)))
    return
end

@testset "VirtualMODF warns only when a transmission outage is dropped, not for generators" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    line = PSY.get_component(PSY.ACTransmission, sys, "1")
    line_outage = _fixed_outage()
    PSY.add_supplemental_attribute!(sys, line, line_outage)
    gen = first(PSY.get_components(PSY.Generator, sys))
    gen_outage = _fixed_outage()
    PSY.add_supplemental_attribute!(sys, gen, gen_outage)

    # A transmission outage whose arc was dropped (empty modification) MUST warn.
    empty_mod = NetworkModification("dropped_line", ArcModification[])
    collector = _CollectLogs(Tuple{Any, String}[])
    Logging.with_logger(collector) do
        PNM._warn_if_transmission_dropped(sys, line_outage, empty_mod)
    end
    @test any(
        r -> r[1] == Logging.Warn && occursin("transmission components", r[2]),
        collector.records,
    )

    # A generator-only outage with an empty modification is benign — NO warning.
    collector_gen = _CollectLogs(Tuple{Any, String}[])
    Logging.with_logger(collector_gen) do
        PNM._warn_if_transmission_dropped(sys, gen_outage, empty_mod)
    end
    @test !any(r -> r[1] == Logging.Warn, collector_gen.records)

    # A transmission outage that DID resolve to an arc modification must NOT warn.
    real_mod = NetworkModification("real_line", [ArcModification(1, -1.0)])
    collector_ok = _CollectLogs(Tuple{Any, String}[])
    Logging.with_logger(collector_ok) do
        PNM._warn_if_transmission_dropped(sys, line_outage, real_mod)
    end
    @test !any(r -> r[1] == Logging.Warn, collector_ok.records)
end

@testset "VirtualMODF tuple getindex gives a clear error for a reduced arc" begin
    # No outages on `sys`, so the reduction runs fully and `reduced_away` arcs are
    # genuinely absent from the reduced network.
    sys, _, reduced_away = _rts_reduced_away_arcs()
    base = VirtualMODF(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )
    missing_arc = first(reduced_away)

    e = 1
    b_e = base.arc_susceptances[e]
    ctg = ContingencySpec(
        123456,
        NetworkModification("ctg", [ArcModification(e, -b_e)]),
    )
    base.contingency_cache[ctg.id] = ctg

    err = try
        base[missing_arc, ctg]
        nothing
    catch ex
        ex
    end
    @test err isa ErrorException
    @test occursin("not present in the reduced network", err.msg)
end

# --- Phase 4: numerical correctness ----------------------------------------

@testset "VirtualMODF Ward rejects an outaged branch outside the study area (RTS)" begin
    # Ward retains exactly its study area, so a contingency on a branch outside it cannot
    # survive and the specification is rejected. RTS carries forced-outage data on 120
    # branches spanning all 73 buses, so no proper subset of them is a valid study area.
    sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")
    bus_numbers = [PSY.get_number(x) for x in PSY.get_components(PSY.ACBus, sys)]
    # Study area = area 1 (leading digit 1), matching test_ward_reduction.jl.
    study_buses = filter(x -> digits(x)[end] == 1, bus_numbers)
    @test !isempty(study_buses)
    @test length(study_buses) < length(bus_numbers)

    @test_throws IS.ConflictingInputsError VirtualMODF(
        sys;
        network_reductions = NetworkReduction[WardReduction(study_buses)],
    )
end

@testset "VirtualMODF reduced+protected matches unreduced at common buses (RTS)" begin
    # Seed the candidate set with a branch that WOULD be reduced away, so the
    # protection path is exercised numerically and not just on retained backbone
    # branches (otherwise the test could pass without the feature working).
    sys, full_no_outage, reduced_away = _rts_reduced_away_arcs()
    target = _branch_on_arcs(sys, get_network_reduction_data(full_no_outage), reduced_away)
    @test target !== nothing

    candidate = PSY.ACTransmission[target]
    for br in PSY.get_components(PSY.ACTransmission, sys)
        _is_phase_shifting_2w(br) && continue
        br === target && continue
        push!(candidate, br)
        length(candidate) >= 8 && break
    end
    for br in candidate
        PSY.add_supplemental_attribute!(sys, br, _fixed_outage())
    end

    full = VirtualMODF(sys)                                   # ground truth
    reduced = VirtualMODF(                                    # reduced + protected
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )

    # The would-be-reduced target must have survived in the reduced network.
    tfb, ttb = _arc_buses(target)
    target_arc = haskey(PNM.get_arc_lookup(reduced), (tfb, ttb)) ? (tfb, ttb) : (ttb, tfb)
    @test haskey(PNM.get_arc_lookup(reduced), target_arc)
    target_id = IS.get_id(PSY.get_supplemental_attributes(target)[1])

    bus_lookup_full = PNM.get_bus_lookup(full)
    nrd_full = get_network_reduction_data(full)
    bus_lookup_red = PNM.get_bus_lookup(reduced)
    nrd_red = get_network_reduction_data(reduced)

    arcs_to_compare = collect(keys(PNM.get_arc_lookup(reduced)))
    buses_to_compare = collect(keys(nrd_red.bus_reduction_map))

    tested_any = false
    tested_target = false
    for br in candidate
        outage = PSY.get_supplemental_attributes(br)[1]
        id = IS.get_id(outage)
        haskey(get_registered_contingencies(full), id) || continue
        haskey(get_registered_contingencies(reduced), id) || continue
        ctg_full = get_registered_contingencies(full)[id]
        ctg_red = get_registered_contingencies(reduced)[id]
        for arc in arcs_to_compare
            haskey(PNM.get_arc_lookup(full), arc) || continue
            ix_full = PNM.get_arc_lookup(full)[arc]
            ix_red = PNM.get_arc_lookup(reduced)[arc]
            row_full = full[ix_full, ctg_full]
            row_red = reduced[ix_red, ctg_red]
            for bus in buses_to_compare
                ib_full = PNM.get_bus_index(bus, bus_lookup_full, nrd_full)
                ib_red = PNM.get_bus_index(bus, bus_lookup_red, nrd_red)
                @test isapprox(row_red[ib_red], row_full[ib_full]; atol = 1e-6)
                tested_any = true
                id == target_id && (tested_target = true)
            end
        end
    end
    @test tested_any
    # The would-be-reduced branch's own contingency was numerically validated.
    @test tested_target
end
