# c_sys14 with a lossy "dup" line parallel to Line1 and an unplanned outage on each of
# `outage_lines`. Returns the system, a VirtualMODF with the outages registered, and the
# outage id of each line by name.
function _two_tier_system(outage_lines::Vector{String})
    sys, _, _ = _c_sys14_with_parallel_pair(; r_scale = 2.0)
    ids = Dict{String, Int}()
    for name in outage_lines
        outage = PSY.FixedForcedOutage(; outage_status = 1.0)
        PSY.add_supplemental_attribute!(sys, PSY.get_component(PSY.Line, sys, name), outage)
        ids[name] = IS.get_id(outage)
    end
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    return sys, vmodf, ids
end

_two_tier_line(sys, name) = PSY.get_component(PSY.Line, sys, name)

_arc_fields(m::PNM.ArcModification) = (
    m.arc_index,
    m.delta_b,
    m.delta_shift_injection,
    m.delta_y11,
    m.delta_y12,
    m.delta_y21,
    m.delta_y22,
    m.opened,
)

@testset "tier-2 modification adds back to the post-contingency modification" begin
    sys, vmodf, ids = _two_tier_system(["Line1", "Line3"])
    pair_arc = PNM.get_arc_lookup(vmodf)[_arc_numbers(_two_tier_line(sys, "Line1"))]
    for (k_name, c_name) in (("dup", "Line1"), ("Line7", "Line3"))
        ts = TopologySchedule(
            vmodf,
            sys,
            [Set{PSY.Component}([_two_tier_line(sys, k_name)])],
        )
        network = get_post_contingency_network(ts, 1, ids[c_name])
        @test network isa PostContingencyNetwork
        @test network.contingency_id == ids[c_name]
        @test network.topology == get_network_modification(ts, 1)
        @test network.post_contingency == NetworkModification(
            vmodf,
            sys,
            [_two_tier_line(sys, k_name), _two_tier_line(sys, c_name)],
        )
        @test network.tier2.label == network.post_contingency.label
        @test network.tier2.is_islanding == network.post_contingency.is_islanding
        @test all(
            m -> any(!iszero, _arc_fields(m)[2:end]),
            network.tier2.arc_modifications,
        )
        merged = PNM._merge_arc_modifications(
            vcat(
                collect(network.topology.arc_modifications),
                collect(network.tier2.arc_modifications),
            ),
        )
        post = collect(network.post_contingency.arc_modifications)
        @test length(merged) == length(post)
        for (m, p) in zip(merged, post)
            @test all(_arc_fields(m) .== _arc_fields(p))
        end
    end
    # The contingency opens the remaining member of a group the topology opened partly.
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}([_two_tier_line(sys, "dup")])])
    network = get_post_contingency_network(ts, 1, ids["Line1"])
    tier2_pair = only(network.tier2.arc_modifications)
    @test tier2_pair.arc_index == pair_arc
    @test tier2_pair.opened == 1
    @test only(network.post_contingency.arc_modifications).opened == 2
end

@testset "a contingency already out gives no network" begin
    sys, vmodf, ids = _two_tier_system(["Line3"])
    ts = TopologySchedule(
        vmodf,
        sys,
        [Set{PSY.Component}([_two_tier_line(sys, "Line3")])],
    )
    @test isnothing(get_post_contingency_network(ts, 1, ids["Line3"]))
end

@testset "steps on one topology share one network" begin
    sys, vmodf, ids = _two_tier_system(["Line3"])
    line7 = _two_tier_line(sys, "Line7")
    ts = TopologySchedule(
        vmodf,
        sys,
        [
            Set{PSY.Component}([line7]),
            Set{PSY.Component}(),
            Set{PSY.Component}([line7]),
        ],
    )
    first_step = get_post_contingency_network(ts, 1, ids["Line3"])
    third_step = get_post_contingency_network(ts, 3, ids["Line3"])
    @test first_step == third_step
    @test hash(first_step) == hash(third_step)
    base_step = get_post_contingency_network(ts, 2, ids["Line3"])
    @test base_step != first_step
    # A network on the base topology is the registered contingency's network.
    @test base_step == PostContingencyNetwork(vmodf, ids["Line3"])
    @test hash(base_step) == hash(PostContingencyNetwork(vmodf, ids["Line3"]))
end

@testset "a network hashes by value without walking its modifications" begin
    sys, vmodf, ids = _two_tier_system(["Line1", "Line3"])
    line7 = _two_tier_line(sys, "Line7")
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}([line7])])
    network = get_post_contingency_network(ts, 1, ids["Line3"])
    relabeled = PostContingencyNetwork(
        PNM._Tier2Derived(),
        network.topology,
        network.post_contingency,
        network.tier2,
        ids["Line1"],
    )
    @test relabeled == network
    @test hash(relabeled) == hash(network)
    @test fieldtype(PostContingencyNetwork, :key_hash) === UInt
    @test network.key_hash == hash((network.topology, network.tier2))
    @test hash(network, UInt(7)) == hash(network.key_hash, UInt(7))
    other = get_post_contingency_network(ts, 1, ids["Line1"])
    @test other != network
    @test other.key_hash != network.key_hash
    lookup = Dict(network => 1, other => 2)
    @test lookup[relabeled] == 1
end

@testset "a network's tier-2 modification is always derived from its other two" begin
    sys, vmodf, ids = _two_tier_system(["Line3"])
    ts = TopologySchedule(vmodf, sys, [Set{PSY.Component}([_two_tier_line(sys, "Line7")])])
    network = get_post_contingency_network(ts, 1, ids["Line3"])
    @test network.tier2 ==
          PNM._tier2_modification(network.post_contingency, network.topology)
    @test network.tier2 != network.post_contingency
    @test_throws MethodError PostContingencyNetwork(
        network.topology,
        network.post_contingency,
        network.post_contingency,
        ids["Line3"],
    )
end

@testset "a base-topology network is the registered contingency" begin
    sys, vmodf, ids = _two_tier_system(["Line1", "Line3"])
    registered = get_registered_contingencies(vmodf)
    for id in values(ids)
        network = PostContingencyNetwork(vmodf, id)
        @test network.tier2 == registered[id].modification
        @test network.post_contingency == registered[id].modification
        @test network.topology == PNM._BASE_NETWORK
        @test network.contingency_id == id
    end
    @test_throws ErrorException PostContingencyNetwork(vmodf, -1)
end

_transmission(sys, name) = PSY.get_component(PSY.ACTransmission, sys, name)

_out_set(sys, names) = Set{PSY.Component}(_transmission(sys, n) for n in names)

_case10() = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")

_case10_beside() = first(_case10_with_transformer_beside_winding())

# c_sys14 with "dup" parallel to Line1 at minus twice its reactance: the pair conducts with
# Line1's susceptance sign, and Line1 out leaves the series-compensated member, of the
# opposite sign.
_c_sys14_series_compensated() = first(_c_sys14_with_parallel_pair(; x_scale = -2.0))

# `builder()` with one registered outage per distinct entry of `contingencies` (component
# names). Returns the system, a VirtualMODF on it and the outage id of each entry.
function _two_tier_fixture(builder, contingencies)
    sys = builder()
    ids = Dict{Vector{String}, Int}()
    for names in contingencies
        haskey(ids, names) && continue
        outage = PSY.FixedForcedOutage(; outage_status = 1.0)
        for name in names
            PSY.add_supplemental_attribute!(sys, _transmission(sys, name), outage)
        end
        ids[names] = IS.get_id(outage)
    end
    return sys, VirtualMODF(VirtualPTDF(sys), sys), ids
end

# Bus sets of the islands holding a reference bus once the `names` components are out, and
# the total number of islands.
function _reference_islands(vmodf, sys, names)
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, names)])
    islands, refs = get_islands(ts, 1)
    return [Set(islands[r]) for r in refs], length(islands)
end

# (system builder, components out at K, contingency components, whether K ∪ c islands)
const _TWO_TIER_CASES = [
    (_c_sys14_for_row_batches, String[], ["Line3"], false),
    (_c_sys14_for_row_batches, String[], ["Line5", "Line9"], false),
    (_c_sys14_for_row_batches, ["Line7"], ["Line13"], false),
    (_c_sys14_for_row_batches, ["Line7"], ["Line4"], false),
    (_c_sys14_for_row_batches, ["Line7"], ["Line7", "Line13"], false),
    (_c_sys14_for_row_batches, ["dup"], ["Line1"], false),
    (_c_sys14_for_row_batches, ["dup"], ["Line10"], false),
    (_c_sys14_for_row_batches, ["Line7", "dup", "shifter"], ["Line12"], false),
    (_c_sys14_for_row_batches, ["Line7", "dup", "shifter"], ["Line6"], false),
    (_c_sys14_for_row_batches, ["Line7", "dup"], ["shifter"], false),
    (_c_sys14_for_row_batches, String[], ["Trans4"], true),
    (_c_sys14_for_row_batches, ["Line16"], ["Trans3"], true),
    (_case10, String[], ["LOOP1-LOOP2-i_1"], false),
    (_case10, String[], ["HV-LV-MV-i_1"], true),
    (_case10, ["LV-RADIAL1-i_1"], ["SERIES1-SERIES2-i_1"], false),
    (_case10, ["MV-SERIES1-i_1", "MV-SERIES1-i_2"], ["SERIES1-SERIES2-i_1"], true),
    (_case10, ["MV-SERIES1-i_1", "MV-SERIES1-i_2"], ["HV-LOOP2-i_1"], true),
    (_case10, ["MV-SERIES1-i_1", "MV-SERIES1-i_2"], ["RADIAL1-RADIAL2-i_1"], true),
    (_case10_beside, ["beside_winding"], ["HV-LOOP2-i_1"], false),
    (_case10_beside, ["beside_winding"], ["SERIES1-SERIES3-i_1"], false),
    (_case10_beside, ["HV-LV-MV-i_1"], ["HV-LOOP2-i_1"], true),
    (_case10_beside, ["HV-LV-MV-i_1"], ["beside_winding"], true),
]

# Every case on `builder`, resolved on one VirtualMODF and one schedule, with each arc's rows
# on every case's network from one call.
function _two_tier_rows(builder)
    cases = [c for c in _TWO_TIER_CASES if c[1] === builder]
    sys, vmodf, ids = _two_tier_fixture(builder, [c[3] for c in cases])
    topologies = unique([c[2] for c in cases])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, k) for k in topologies])
    nets = [
        get_post_contingency_network(ts, findfirst(==(c[2]), topologies), ids[c[3]])
        for
        c in cases
    ]
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(arc => get_post_contingency_ptdf_rows(vmodf, arc, nets) for arc in arcs)
    return (; cases, sys, vmodf, nets, arcs, rows)
end

_case_islands(r, case) =
    case[4] ? first(_reference_islands(r.vmodf, r.sys, union(case[2], case[3]))) : nothing

@testset "two-tier rows match rebuilt systems" begin
    for builder in unique(first.(_TWO_TIER_CASES))
        r = _two_tier_rows(builder)
        @test all(!isnothing, r.nets)
        bus_axis = PNM.get_bus_axis(r.vmodf)
        # Fresh caches on the same factorization, so the comparison is exact on any backend.
        fresh = VirtualMODF(PNM.get_core(r.vmodf), r.sys)
        for (i, case) in enumerate(r.cases)
            names = union(case[2], case[3])
            _, n_islands = _reference_islands(r.vmodf, r.sys, names)
            @test (n_islands > 1) == case[4]
            err = _max_error_against_rebuild(
                a -> r.rows[a][i],
                r.arcs,
                bus_axis,
                names;
                builder = builder,
                islands = _case_islands(r, case),
            )
            @test err < 1e-10
            # On the base topology the rows are the registered contingency's rows, bit for bit.
            if r.nets[i].topology == PNM._BASE_NETWORK
                for a in r.arcs
                    @test _bitwise_equal(r.rows[a][i], fresh[a, r.nets[i].tier2])
                end
            end
        end
    end
end

@testset "opening the last member of a partly out group takes the arc out" begin
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_for_row_batches, [["Line1"]])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, ["dup"])])
    net = get_post_contingency_network(ts, 1, ids[["Line1"]])
    pair = _arc_numbers(_transmission(sys, "dup"))
    m = PNM.get_arc_lookup(vmodf)[pair]
    @test only(net.tier2.arc_modifications).arc_index == m
    @test only(net.tier2.arc_modifications).opened == 1
    arc_sus = PNM._get_arc_susceptances(vmodf)
    @test PNM._post_contingency_arc_out(arc_sus, PNM.get_core(vmodf), m, net)
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    @test all(iszero, rows[pair])
    err = _max_error_against_rebuild(
        a -> rows[a],
        arcs,
        PNM.get_bus_axis(vmodf),
        ["dup", "Line1"],
    )
    @test err < 1e-10
end

@testset "a monitored arc partly out on K is scaled by its K ∪ c susceptance" begin
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_for_row_batches, [["Line10"]])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, ["dup"])])
    net = get_post_contingency_network(ts, 1, ids[["Line10"]])
    pair = _arc_numbers(_transmission(sys, "dup"))
    m = PNM.get_arc_lookup(vmodf)[pair]
    arc_sus = PNM._get_arc_susceptances(vmodf)
    k_change = only(x for x in net.topology.arc_modifications if x.arc_index == m).delta_b
    @test all(x -> x.arc_index != m, net.tier2.arc_modifications)
    b_post = PNM._post_contingency_susceptance(arc_sus, m, net)
    @test b_post == arc_sus[m] + k_change
    @test !isapprox(b_post, arc_sus[m]; rtol = 1e-3)
    @test !PNM._post_contingency_arc_out(arc_sus, PNM.get_core(vmodf), m, net)
    row = only(get_post_contingency_ptdf_rows(vmodf, pair, [net]))
    err = _max_error_against_rebuild(
        _ -> row,
        [pair],
        PNM.get_bus_axis(vmodf),
        ["dup", "Line10"],
    )
    @test err < 1e-10
end

const _MV_PAIR = ["MV-SERIES1-i_1", "MV-SERIES1-i_2"]

@testset "a contingency inside a dead section leaves the topology rows" begin
    sys, vmodf, ids = _two_tier_fixture(_case10, [["SERIES1-SERIES2-i_1"]])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, _MV_PAIR)])
    islands, refs = get_islands(ts, 1)
    dead = Set(bus for (k, island) in enumerate(islands) if k ∉ refs for bus in island)
    @test length(dead) > 2
    @test all(in(dead), _arc_numbers(_transmission(sys, "SERIES1-SERIES2-i_1")))
    net = get_post_contingency_network(ts, 1, ids[["SERIES1-SERIES2-i_1"]])
    @test isempty(PNM.get_woodbury_cache(vmodf))
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    @test PNM.get_woodbury_cache(vmodf)[net.topology].is_islanding
    @test isempty(PNM.get_tier2_factor_cache(vmodf))
    tripped = _arc_numbers(_transmission(sys, "SERIES1-SERIES2-i_1"))
    @test all(iszero, rows[tripped])
    for a in arcs
        a == tripped && continue
        @test _bitwise_equal(rows[a], vmodf[a, net.topology])
    end
end

@testset "a contingency across the islands of an islanded topology errors" begin
    c = ["SERIES1-SERIES2-i_1", "HV-LOOP2-i_1"]
    sys, vmodf, ids = _two_tier_fixture(_case10, [c])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, _MV_PAIR)])
    net = get_post_contingency_network(ts, 1, ids[c])
    err = try
        get_post_contingency_ptdf_rows(vmodf, (101, 104), [net])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(net.topology.label, err.msg)
    @test occursin(string(ids[c]), err.msg)
    @test occursin("(106, 107)", err.msg)
    @test occursin("(101, 105)", err.msg)
    @test isempty(PNM.get_tier2_factor_cache(vmodf))
    @test isempty(PNM.get_tier2_row_caches(vmodf))
end

# case10 with LOOP1-LOOP2 at zero reactance: a loop arc with no DC susceptance.
function _case10_resistive_loop()
    sys = _case10()
    PSY.set_x!(_transmission(sys, "LOOP1-LOOP2-i_1"), 0.0 * u"CU")
    return sys
end

# case10 with SERIES1-SERIES2 at zero reactance: an arc with no DC susceptance in the section
# the MV-SERIES1 pair feeds.
function _case10_resistive_series()
    sys = _case10()
    PSY.set_x!(_transmission(sys, "SERIES1-SERIES2-i_1"), 0.0 * u"CU")
    return sys
end

@testset "an arc with no susceptance change does not count in the islanding rule" begin
    # The first arc of each contingency changes no susceptance: in the islands holding a
    # reference bus, then in the dead section. HV-LOOP2 is the live arc.
    for (builder, c) in (
        (_case10_resistive_loop, ["LOOP1-LOOP2-i_1", "HV-LOOP2-i_1"]),
        (_case10_resistive_series, ["SERIES1-SERIES2-i_1", "HV-LOOP2-i_1"]),
    )
        sys, vmodf, ids = _two_tier_fixture(builder, [c])
        ts = TopologySchedule(vmodf, sys, [_out_set(sys, _MV_PAIR)])
        net = get_post_contingency_network(ts, 1, ids[c])
        lookup = PNM.get_arc_lookup(vmodf)
        unchanged = lookup[_arc_numbers(_transmission(sys, first(c)))]
        live = lookup[_arc_numbers(_transmission(sys, last(c)))]
        @test [m.arc_index for m in net.tier2.arc_modifications] == sort!([unchanged, live])
        @test iszero(
            only(
                m for m in net.tier2.arc_modifications if m.arc_index == unchanged
            ).delta_b,
        )
        arcs = PNM.get_arc_axis(vmodf)
        rows =
            Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
        @test PNM.get_tier2_factor_cache(vmodf)[net].arc_indices == [live]
        @test all(row -> all(isfinite, row), values(rows))
        names = union(_MV_PAIR, c)
        err = _max_error_against_rebuild(
            a -> rows[a],
            arcs,
            PNM.get_bus_axis(vmodf),
            names;
            builder = builder,
            islands = first(_reference_islands(vmodf, sys, names)),
        )
        @test err < 1e-10
    end
end

@testset "a contingency that islands K further zeroes the disconnected buses" begin
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_for_row_batches, [["Trans3"]])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, ["Line16"])])
    @test length(first(get_islands(ts, 1))) == 1
    net = get_post_contingency_network(ts, 1, ids[["Trans3"]])
    islands, n_islands = _reference_islands(vmodf, sys, ["Line16", "Trans3"])
    @test n_islands == 2
    arcs = PNM.get_arc_axis(vmodf)
    bus_axis = PNM.get_bus_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    @test !PNM.get_woodbury_cache(vmodf)[net.topology].is_islanding
    @test PNM.get_tier2_factor_cache(vmodf)[net].is_islanding
    cut_off = [j for (j, bus) in enumerate(bus_axis) if bus ∉ only(islands)]
    @test sort!(bus_axis[cut_off]) == [7, 8]
    for a in arcs
        isempty(_compared_positions(a, bus_axis, islands)) && continue
        @test all(==(0.0), rows[a][cut_off])
    end
    err = _max_error_against_rebuild(
        a -> rows[a],
        arcs,
        bus_axis,
        ["Line16", "Trans3"];
        islands = islands,
    )
    @test err < 1e-10
end

@testset "a series-compensated group partly out keeps its sign frame" begin
    sys, vmodf, ids =
        _two_tier_fixture(_c_sys14_series_compensated, [["Line10"], ["dup"], ["Line1"]])
    pair = _arc_numbers(_transmission(sys, "dup"))
    m = PNM.get_arc_lookup(vmodf)[pair]
    arc_sus = PNM._get_arc_susceptances(vmodf)
    signs = PNM.get_core(vmodf).arc_susceptance_signs
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, ["Line1"]), Set{PSY.Component}()])
    k_change = only(ts.modifications[1].arc_modifications).delta_b
    # Line1 out flips the sign of the group's remaining susceptance.
    @test signs[m] * (arc_sus[m] + k_change) < 0 < signs[m] * arc_sus[m]
    arcs = PNM.get_arc_axis(vmodf)
    bus_axis = PNM.get_bus_axis(vmodf)
    for (t, k_names, c_names) in
        ((1, ["Line1"], ["Line10"]), (1, ["Line1"], ["dup"]), (2, String[], ["Line1"]))
        net = get_post_contingency_network(ts, t, ids[c_names])
        rows =
            Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
        err = _max_error_against_rebuild(
            a -> rows[a],
            arcs,
            bus_axis,
            union(k_names, c_names);
            builder = _c_sys14_series_compensated,
        )
        @test err < 1e-10
    end
end

@testset "steps on one topology share tier-2 factors and rows" begin
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_for_row_batches, [["Line3"]])
    k = _out_set(sys, ["Line7"])
    ts = TopologySchedule(vmodf, sys, [k, Set{PSY.Component}(), k])
    first_step, third_step =
        (get_post_contingency_network(ts, t, ids[["Line3"]]) for t in (1, 3))
    rows = get_post_contingency_ptdf_rows(vmodf, (2, 4), [first_step, third_step])
    @test _bitwise_equal(rows[1], rows[2])
    @test rows[1] !== rows[2]
    @test length(PNM.get_tier2_factor_cache(vmodf)) == 1
    @test length(PNM.get_tier2_row_caches(vmodf)) == 1
end

@testset "retained post-contingency networks keep their factors" begin
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_for_row_batches, [["Line3"]])
    ts = TopologySchedule(
        vmodf,
        sys,
        [_out_set(sys, ["Line7"]), _out_set(sys, ["dup"]), Set{PSY.Component}()],
    )
    net, other, base = (get_post_contingency_network(ts, t, ids[["Line3"]]) for t in 1:3)
    arc = (2, 5)
    rows = get_post_contingency_ptdf_rows(vmodf, arc, [net, other, base])
    tier2_factors = PNM.get_tier2_factor_cache(vmodf)
    woodbury = PNM.get_woodbury_cache(vmodf)
    @test Set(keys(tier2_factors)) == Set([net, other])
    @test Set(keys(woodbury)) == Set([net.topology, other.topology, base.tier2])
    @test haskey(PNM.get_row_caches(vmodf), base.tier2)

    retain_modifications!(vmodf, [net.topology, net])
    @test Set(keys(tier2_factors)) == Set([net])
    @test Set(keys(PNM.get_tier2_row_caches(vmodf))) == Set([net])
    @test Set(keys(woodbury)) == Set([net.topology])
    @test !haskey(PNM.get_row_caches(vmodf), base.tier2)
    kept = tier2_factors[net]
    @test _bitwise_equal(only(get_post_contingency_ptdf_rows(vmodf, arc, [net])), rows[1])
    @test tier2_factors[net] === kept

    # A retained network retains its topology without naming it.
    get_post_contingency_ptdf_rows(vmodf, arc, [other, base])
    retain_modifications!(vmodf, [other, base])
    @test Set(keys(tier2_factors)) == Set([other])
    @test Set(keys(woodbury)) == Set([other.topology, base.tier2])
    @test haskey(PNM.get_row_caches(vmodf), base.tier2)

    clear_caches!(vmodf)
    @test isempty(tier2_factors)
    @test isempty(PNM.get_tier2_row_caches(vmodf))
    @test isempty(woodbury)
    again = get_post_contingency_ptdf_rows(vmodf, arc, [net, other, base])
    @test all(_bitwise_equal.(again, rows))

    PNM.clear_all_caches!(vmodf)
    @test isempty(tier2_factors)
    @test isempty(PNM.get_tier2_row_caches(vmodf))
end

# c_sys14 with "dup" parallel to Line1 at zero reactance: a lossy member with no DC
# susceptance of its own.
_c_sys14_resistive_member() =
    first(_c_sys14_with_parallel_pair(; r_scale = 2.0, x_scale = 0.0))

# c_sys14 with Line13 at zero reactance: an arc with no DC susceptance.
function _c_sys14_resistive_line()
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    PSY.set_x!(PSY.get_component(PSY.Line, sys, "Line13"), 0.0 * u"CU")
    return sys
end

@testset "a tier-2 arc whose susceptance is unchanged leaves the Woodbury set" begin
    c = ["dup", "Line10"]
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_resistive_member, [c])
    ts = TopologySchedule(vmodf, sys, [_out_set(sys, ["Line1"])])
    net = get_post_contingency_network(ts, 1, ids[c])
    lookup = PNM.get_arc_lookup(vmodf)
    pair = lookup[_arc_numbers(_transmission(sys, "dup"))]
    line10 = lookup[_arc_numbers(_transmission(sys, "Line10"))]
    # Line1 out leaves the resistive member alone on the arc, so opening it changes the
    # arc's Pi model but not its susceptance.
    @test [m.arc_index for m in net.tier2.arc_modifications] == sort!([pair, line10])
    unchanged = only(m for m in net.tier2.arc_modifications if m.arc_index == pair)
    @test iszero(unchanged.delta_b)
    @test unchanged.opened == 1
    @test !iszero(unchanged.delta_y12)
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    @test PNM.get_tier2_factor_cache(vmodf)[net].arc_indices == [line10]
    @test all(row -> all(isfinite, row), values(rows))
    err = _max_error_against_rebuild(
        a -> rows[a],
        arcs,
        PNM.get_bus_axis(vmodf),
        union(["Line1"], c);
        builder = _c_sys14_resistive_member,
    )
    @test err < 1e-10
end

@testset "a contingency arc with no susceptance leaves the Woodbury set" begin
    c = ["Line13", "Line10"]
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_resistive_line, [c])
    mod = get_registered_contingencies(vmodf)[ids[c]].modification
    lookup = PNM.get_arc_lookup(vmodf)
    line13 = lookup[_arc_numbers(_transmission(sys, "Line13"))]
    line10 = lookup[_arc_numbers(_transmission(sys, "Line10"))]
    @test iszero(PNM._get_arc_susceptances(vmodf)[line13])
    @test [m.arc_index for m in mod.arc_modifications] == sort!([line13, line10])
    outage = PSY.get_supplemental_attribute(sys, ids[c])
    arcs = PNM.get_arc_axis(vmodf)
    bus_axis = PNM.get_bus_axis(vmodf)
    rows = Dict(a => vmodf[a, outage] for a in arcs)
    @test PNM.get_woodbury_cache(vmodf)[mod].arc_indices == [line10]
    @test all(row -> all(isfinite, row), values(rows))
    err = _max_error_against_rebuild(
        a -> rows[a],
        arcs,
        bus_axis,
        c;
        builder = _c_sys14_resistive_line,
    )
    @test err < 1e-10
    vptdf = VirtualPTDF(sys)
    err = _max_error_against_rebuild(
        a -> vptdf[a, mod],
        arcs,
        bus_axis,
        c;
        builder = _c_sys14_resistive_line,
    )
    @test err < 1e-10
end

@testset "base-topology contingency rows are the registered contingency's rows" begin
    sys = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        outage = PSY.FixedForcedOutage(; outage_status = 1.0)
        PSY.add_supplemental_attribute!(sys, branch, outage)
    end
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    registered = get_registered_contingencies(vmodf)
    @test length(registered) > 100
    ids = sort!(collect(keys(registered)))
    outages = Dict(id => PSY.get_supplemental_attribute(sys, id) for id in ids)
    arcs = PNM.get_arc_axis(vmodf)
    lookup = PNM.get_arc_lookup(vmodf)
    # Fresh caches on the same factorization, so the comparison is exact on any backend.
    core = PNM.get_core(vmodf)
    oracle = VirtualMODF(core, sys)
    mismatches = Tuple{Int, Tuple{Int, Int}}[]
    for id in ids, arc in arcs
        expected = only(
            get_post_contingency_ptdf_rows(
                oracle,
                arc,
                [PostContingencyNetwork(oracle, id)],
            ),
        )
        _bitwise_equal(vmodf[arc, outages[id]], expected) &&
            _bitwise_equal(vmodf[lookup[arc], outages[id]], expected) &&
            _bitwise_equal(vmodf[arc, registered[id]], expected) &&
            _bitwise_equal(vmodf[lookup[arc], registered[id]], expected) ||
            push!(mismatches, (id, arc))
    end
    @test isempty(mismatches)

    # Rows pinned by populate_cache are what both paths return, with no solve: the cached row
    # objects are the ones populate_cache stored.
    pinned = VirtualMODF(core, sys)
    pinned_ids = ids[1:5]
    monitored = arcs[1:10]
    populate_cache(pinned, [outages[id] for id in pinned_ids]; monitored = monitored)
    row_caches = PNM.get_row_caches(pinned)
    n_factors = length(PNM.get_woodbury_cache(pinned))
    for id in pinned_ids
        rc = row_caches[registered[id].modification]
        n_rows = length(rc)
        net = PostContingencyNetwork(pinned, id)
        stored = Dict(arc => rc[lookup[arc]] for arc in monitored)
        for arc in monitored
            @test _bitwise_equal(pinned[arc, outages[id]], stored[arc])
            @test _bitwise_equal(
                only(get_post_contingency_ptdf_rows(pinned, arc, [net])),
                stored[arc],
            )
        end
        @test all(rc[lookup[arc]] === stored[arc] for arc in monitored)
        @test length(rc) == n_rows
        @test row_caches[registered[id].modification] === rc
    end
    @test length(row_caches) == length(pinned_ids)
    @test length(PNM.get_woodbury_cache(pinned)) == n_factors
end

@testset "base-topology contingency rows honor distributed slack" begin
    cases = (
        (_c_sys14_for_row_batches, [["Line3"], ["Line5", "Line9"], ["shifter"]]),
        (_case10, [["LOOP1-LOOP2-i_1"]]),
    )
    for (builder, contingencies) in cases
        sys, _, ids = _two_tier_fixture(builder, contingencies)
        buses = sort!([PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)])
        slack = Dict(b => Float64(k) for (k, b) in enumerate(buses))
        vptdf = VirtualPTDF(sys; dist_slack = slack)
        vmodf = VirtualMODF(vptdf, sys)
        weights = PNM.get_dist_slack_normalized(vptdf)
        arcs = PNM.get_arc_axis(vptdf)
        bus_axis = PNM.get_bus_axis(vptdf)
        for names in contingencies
            outage = PSY.get_supplemental_attribute(sys, ids[names])
            mod = NetworkModification(vptdf, sys, outage)
            @test mod == get_registered_contingencies(vmodf)[ids[names]].modification
            rows = Dict(a => vptdf[a, mod] for a in arcs)
            err = _max_error_against_rebuild(
                a -> rows[a],
                arcs,
                bus_axis,
                names;
                builder = builder,
                dist_slack = slack,
            )
            @test err < 1e-10
            # The slack shifts the registered contingency's reference-bus row.
            for a in arcs
                reference_row = vmodf[a, outage]
                @test _bitwise_equal(
                    rows[a],
                    reference_row .- PNM.LinearAlgebra.dot(reference_row, weights),
                )
                @test _bitwise_equal(
                    only(get_post_modification_ptdf_rows(vptdf, a, [mod])),
                    rows[a],
                )
            end
        end
    end
    # Distributed slack is not supported for a contingency that islands the network.
    sys, _, ids = _two_tier_fixture(_case10, [["HV-LV-MV-i_1"]])
    buses = [PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)]
    vptdf = VirtualPTDF(sys; dist_slack = Dict(b => 1.0 for b in buses))
    outage = PSY.get_supplemental_attribute(sys, ids[["HV-LV-MV-i_1"]])
    @test_throws ErrorException vptdf[1, NetworkModification(vptdf, sys, outage)]
end

# Every entry point computing a base-topology row, each on its own fresh caches over `core`,
# keyed by name. `mod` is the network's modification and `net` its base-topology network.
function _base_topology_entry_point_rows(core, sys, vptdf, mod, net, arcs)
    lookup = PNM.get_arc_lookup(vptdf)
    oracle = VirtualMODF(core, sys)
    by_getindex = VirtualMODF(core, sys)
    by_rows = VirtualMODF(core, sys)
    by_worker = VirtualMODF(core, sys)
    by_worker_factors = VirtualMODF(core, sys)
    pinned = VirtualMODF(core, sys)
    populate_cache(pinned, [mod]; monitored = arcs)
    worker = PNM.worker_core(by_worker)
    worker_wf = PNM.get_woodbury_factors(by_worker_factors, mod, worker)
    vptdf_wf = compute_woodbury_factors(vptdf, mod)
    rows = Dict{String, Dict{Tuple{Int, Int}, Any}}()
    entry_points = (
        "row function" => a -> only(get_post_contingency_ptdf_rows(oracle, a, [net])),
        "getindex" => a -> by_getindex[a, mod],
        "getindex by index" => a -> by_getindex[lookup[a], mod],
        "get_post_modification_ptdf_rows" =>
            a -> only(get_post_modification_ptdf_rows(by_rows, a, [mod])),
        "populate_cache" => a -> PNM.get_row_caches(pinned)[mod][lookup[a]],
        "worker row" => a -> get_post_modification_ptdf_row(by_worker, a, mod, worker),
        "worker apply_woodbury_correction" =>
            a -> PNM.apply_woodbury_correction(worker, lookup[a], worker_wf),
        "VirtualPTDF getindex" => a -> vptdf[a, mod],
        "VirtualPTDF rows" =>
            a -> only(get_post_modification_ptdf_rows(vptdf, a, [mod])),
        "VirtualPTDF apply_woodbury_correction" =>
            a -> apply_woodbury_correction(vptdf, a, vptdf_wf),
    )
    for (name, f) in entry_points
        rows[name] = Dict(a => f(a) for a in arcs)
    end
    return rows
end

# Adds a FixedAdmittance named "shunt" at the lowest-numbered bus of `sys`.
function _add_shunt!(sys)
    bus = argmin(PSY.get_number, PSY.get_components(PSY.ACBus, sys))
    PSY.add_component!(
        sys,
        PSY.FixedAdmittance(;
            name = "shunt",
            available = true,
            bus = bus,
            Y = 0.01 + 0.05im,
        ),
    )
    return sys
end

# The outaged component named `name`: a branch, the "shunt" admittance, or, for "generator",
# the first generator by name.
function _outage_component(sys, name)
    name == "shunt" && return PSY.get_component(PSY.FixedAdmittance, sys, name)
    if name == "generator"
        return argmin(PSY.get_name, PSY.get_components(PSY.Generator, sys))
    end
    return _transmission(sys, name)
end

# Entry points that take only Woodbury factors, so cannot tell the base network from a
# modification whose factors have no columns.
const _FACTORS_ONLY_ENTRY_POINTS =
    ("worker apply_woodbury_correction", "VirtualPTDF apply_woodbury_correction")

@testset "every base-topology entry point returns the row function's rows" begin
    rts = _add_shunt!(PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys"))
    lines = sort!([PSY.get_name(l) for l in PSY.get_components(PSY.Line, rts)])
    transformers = sort!([
        PSY.get_name(t) for t in PSY.get_components(PSY.TwoWindingTransformer, rts)
    ])
    cases = (
        (
            () -> rts,
            [
                [lines[1]],
                [lines[2], lines[3]],
                [transformers[1]],
                [lines[4], transformers[2]],
                ["shunt"],
                ["generator"],
            ],
        ),
        (
            () -> _add_shunt!(_case10_resistive_loop()),
            [
                ["LOOP1-LOOP2-i_1"],
                ["LOOP1-LOOP2-i_1", "SERIES1-SERIES3-i_1"],
                ["HV-LOOP1-i_1"],
                ["HV-LV-MV-i_1"],
                ["shunt"],
                ["generator"],
            ],
        ),
    )
    for (builder, contingencies) in cases
        sys = builder()
        ids = Dict{Vector{String}, Int}()
        for names in contingencies
            outage = PSY.FixedForcedOutage(; outage_status = 1.0)
            for name in names
                PSY.add_supplemental_attribute!(sys, _outage_component(sys, name), outage)
            end
            ids[names] = IS.get_id(outage)
        end
        vptdf = VirtualPTDF(sys; linear_solver = "KLU")
        core = PNM.get_core(vptdf)
        registered = get_registered_contingencies(VirtualMODF(core, sys))
        arcs = PNM.get_arc_axis(vptdf)
        networks = Pair{String, PostContingencyNetwork}[
            join(names, ", ") => PNM._base_topology_network(
                registered[ids[names]].modification,
                ids[names],
            ) for names in contingencies
        ]
        push!(networks, "base network" => PNM._base_topology_network(PNM._BASE_NETWORK))
        # The arithmetic of a correction with no Woodbury columns: the base solve divided and
        # multiplied by the arc susceptance; the zero row for an arc with no susceptance.
        lookup = PNM.get_arc_lookup(vptdf)
        arc_sus = core.arc_susceptances
        base_rows = Dict(a => get_ptdf_row(VirtualMODF(core, sys), a) for a in arcs)
        scaled = Dict(
            a => begin
                b = arc_sus[lookup[a]]
                iszero(b) ? zeros(length(base_rows[a])) : (base_rows[a] ./ b) .* b
            end for a in arcs
        )
        @test any(a -> !_bitwise_equal(scaled[a], base_rows[a]), arcs)
        mismatches = Tuple{String, String, Tuple{Int, Int}}[]
        for (label, net) in networks
            rows = _base_topology_entry_point_rows(core, sys, vptdf, net.tier2, net, arcs)
            expected = rows["row function"]
            on_base = net.tier2 == PNM._BASE_NETWORK
            @test all(row -> all(isfinite, row), values(expected))
            for (name, by_arc) in rows, a in arcs
                want =
                    on_base && name in _FACTORS_ONLY_ENTRY_POINTS ? scaled[a] : expected[a]
                _bitwise_equal(by_arc[a], want) || push!(mismatches, (label, name, a))
            end
            if on_base
                for a in arcs
                    _bitwise_equal(expected[a], base_rows[a]) ||
                        push!(mismatches, (label, "get_ptdf_row", a))
                end
            elseif isempty(net.tier2.arc_modifications)
                @test !isempty(net.tier2.shunt_modifications)
                for a in arcs
                    _bitwise_equal(expected[a], scaled[a]) ||
                        push!(mismatches, (label, "(x / b) * b", a))
                end
            end
        end
        @test registered[ids[["generator"]]].modification == PNM._BASE_NETWORK
        @test isempty(mismatches)
    end
end

@testset "a contingency whose every arc has no susceptance leaves the base rows" begin
    sys, vmodf, ids = _two_tier_fixture(_c_sys14_resistive_line, [["Line13"]])
    mod = get_registered_contingencies(vmodf)[ids[["Line13"]]].modification
    line13 = _arc_numbers(_transmission(sys, "Line13"))
    @test iszero(PNM._get_arc_susceptances(vmodf)[PNM.get_arc_lookup(vmodf)[line13]])
    outage = PSY.get_supplemental_attribute(sys, ids[["Line13"]])
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => vmodf[a, outage] for a in arcs)
    @test isempty(PNM.get_woodbury_cache(vmodf)[mod].arc_indices)
    @test all(row -> all(isfinite, row), values(rows))
    @test all(iszero, rows[line13])
    for a in arcs
        a == line13 && continue
        @test maximum(abs.(rows[a] .- get_ptdf_row(vmodf, a))) < 1e-12
    end
end

@testset "a contingency's components resolve once per schedule" begin
    function builder()
        sys = first(_c_sys14_with_parallel_pair(; r_scale = 2.0))
        PSY.set_available!(PSY.get_component(PSY.Line, sys, "Line5"), false)
        return sys
    end
    with_unavailable = ["Line3", "Line5"]
    sys, vmodf, ids = _two_tier_fixture(builder, [["Line3"], with_unavailable])
    line7 = _transmission(sys, "Line7")
    removed = [
        Set{PSY.Component}([line7]),
        Set{PSY.Component}(),
        Set{PSY.Component}([line7]),
    ]
    ts = TopologySchedule(vmodf, sys, removed)
    id = ids[with_unavailable]
    first_network = get_post_contingency_network(ts, 1, id)
    memoized = ts.contingency_components[id]
    networks = [
        first_network,
        [get_post_contingency_network(ts, t, id) for t in 2:length(removed)]...,
    ]
    @test ts.contingency_components[id] === memoized
    @test collect(keys(ts.contingency_components)) == [ids[with_unavailable]]
    @test only(values(ts.contingency_components)) ==
          PSY.Component[_transmission(sys, "Line3")]
    # The unavailable member is dropped, so on every topology the outage's network is the
    # one a fresh schedule resolves for Line3 alone.
    for (t, network) in enumerate(networks)
        fresh = TopologySchedule(vmodf, sys, removed)
        @test network isa PostContingencyNetwork
        @test network == get_post_contingency_network(fresh, t, ids[["Line3"]])
    end
end
