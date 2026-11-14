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
        ts = TopologySequence(
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
    ts = TopologySequence(vmodf, sys, [Set{PSY.Component}([_two_tier_line(sys, "dup")])])
    network = get_post_contingency_network(ts, 1, ids["Line1"])
    tier2_pair = only(network.tier2.arc_modifications)
    @test tier2_pair.arc_index == pair_arc
    @test tier2_pair.opened == 1
    @test only(network.post_contingency.arc_modifications).opened == 2
end

@testset "a contingency already out gives no network" begin
    sys, vmodf, ids = _two_tier_system(["Line3"])
    ts = TopologySequence(
        vmodf,
        sys,
        [Set{PSY.Component}([_two_tier_line(sys, "Line3")])],
    )
    @test isnothing(get_post_contingency_network(ts, 1, ids["Line3"]))
end

@testset "steps on one topology share one network" begin
    sys, vmodf, ids = _two_tier_system(["Line3"])
    line7 = _two_tier_line(sys, "Line7")
    ts = TopologySequence(
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
    ts = TopologySequence(vmodf, sys, [Set{PSY.Component}([line7])])
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
    ts = TopologySequence(vmodf, sys, [Set{PSY.Component}([_two_tier_line(sys, "Line7")])])
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

@testset "a contingency's components resolve once per sequence" begin
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
    ts = TopologySequence(vmodf, sys, removed)
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
    # one a fresh sequence resolves for Line3 alone.
    for (t, network) in enumerate(networks)
        fresh = TopologySequence(vmodf, sys, removed)
        @test network isa PostContingencyNetwork
        @test network == get_post_contingency_network(fresh, t, ids[["Line3"]])
    end
end
