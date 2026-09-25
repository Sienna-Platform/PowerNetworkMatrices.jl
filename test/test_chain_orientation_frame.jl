#=
`get_segment_orientations` replaces the second return value of the removed `_get_chain_data`,
which PowerOperationsModels still calls to sign interface flow contributions. The reference
below is that function's body verbatim — the point is to prove the stored vector agrees with
it, not to keep the algorithm.
=#

function _reference_chain_data(equivalent_arc, chain, nrd)
    ordered_bus_numbers = [equivalent_arc[1]]
    segment_orientation = Vector{Symbol}()
    for segment in chain
        arc_tuple = PNM.get_arc_tuple(segment, nrd)
        if arc_tuple[1] in ordered_bus_numbers
            push!(ordered_bus_numbers, arc_tuple[2])
            push!(segment_orientation, :FromTo)
        elseif arc_tuple[2] in ordered_bus_numbers
            push!(ordered_bus_numbers, arc_tuple[1])
            push!(segment_orientation, :ToFrom)
        else
            error("Found disconnected series chain")
        end
    end
    @assert ordered_bus_numbers[end] == equivalent_arc[2]
    return ordered_bus_numbers, segment_orientation
end

# Orientation of each STORED segment relative to an arbitrary frame, derived from bus
# positions along the chain's path rather than from the implementation: segment (u, v) reads
# `:FromTo` exactly when u precedes v walking the requested frame. Independent of
# `get_segment_orientations`, so it can serve as its oracle in a reframed frame, where
# `_reference_chain_data` cannot (it walks in segment-storage order and errors on a reversal).
function _orientations_in_frame(equivalent_arc, chain, nrd)
    key = PNM.get_arc_key(chain)
    path, _ = _reference_chain_data(key, chain, nrd)
    if equivalent_arc == reverse(key)
        path = reverse(path)
    elseif equivalent_arc != key
        error("Frame $equivalent_arc is neither $key nor its reverse")
    end
    position = Dict(bus => i for (i, bus) in enumerate(path))
    orientations = Symbol[]
    for segment in chain
        from, to = PNM.get_arc_tuple(segment, nrd)
        if position[from] < position[to]
            push!(orientations, :FromTo)
        else
            push!(orientations, :ToFrom)
        end
    end
    return orientations
end

_is_chain(::PNM.BranchesSeries) = true
_is_chain(::PSY.ACTransmission) = false

@testset "get_segment_orientations matches the removed _get_chain_data walk" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    series_map = PNM.get_series_branch_map(nrd)
    @test !isempty(series_map)
    for (arc, chain) in series_map
        # The map key is the chain's own frame, which is what the old caller passed.
        @test PNM.get_arc_key(chain) == arc
        @test PNM.get_segment_orientations(chain) ==
              _reference_chain_data(arc, chain, nrd)[2]
        @test PNM.get_segment_orientations(chain, arc) ==
              _reference_chain_data(arc, chain, nrd)[2]
    end
end

@testset "get_segment_orientations matches the removed walk for chains inside a parallel group" begin
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    groups = [
        (arc, grp) for (arc, grp) in PNM.get_parallel_branch_map(nrd) if
        all(_is_chain, grp)
    ]
    @test !isempty(groups)
    for (_, grp) in groups, chain in grp
        key = PNM.get_arc_key(chain)
        @test PNM.get_segment_orientations(chain) ==
              _reference_chain_data(key, chain, nrd)[2]
        @test PNM.get_segment_orientations(chain, key) ==
              _reference_chain_data(key, chain, nrd)[2]
    end
end

@testset "get_segment_orientations reframes a reversed arc and rejects an unrelated one" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    arc, chain = first(PNM.get_series_branch_map(nrd))
    reversed = (arc[2], arc[1])

    own = PNM.get_segment_orientations(chain, arc)
    flipped = PNM.get_segment_orientations(chain, reversed)
    @test flipped == _orientations_in_frame(reversed, chain, nrd)
    # Every orientation negates and the segment order is preserved: callers zip this against
    # the chain's members in storage order.
    @test length(flipped) == length(own)
    @test all(a !== b for (a, b) in zip(own, flipped))

    # An arc that is neither the key nor its reverse remains a caller bug.
    @test_throws ErrorException PNM.get_segment_orientations(
        chain,
        (arc[1], arc[2] + 10_000),
    )

    # Neither frame may alias the stored field, which `add_branch!` keeps mutating.
    @test PNM.get_segment_orientations(chain, arc) !==
          PNM.get_segment_orientations(chain)
    @test PNM.get_segment_orientations(chain, reversed) !==
          PNM.get_segment_orientations(chain)
end

@testset "a chain reached through a parallel group reframes to the group's arc" begin
    # The group is framed `(1, 3)` but one member is keyed `(3, 1)`. POM's `_get_direction`
    # forwards the GROUP's arc to every member, so refusing the mismatch made every
    # `TransmissionInterface` over a grouped chain throw; reframing is the contract.
    sys = build_reversed_asymmetric_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    group_arc = (1, 3)
    group = PNM.get_parallel_branch_map(nrd)[group_arc]
    @test Set(PNM.get_arc_key(m) for m in group) == Set([(1, 3), (3, 1)])

    for chain in group
        key = PNM.get_arc_key(chain)
        # Each member agrees with the removed walk in ITS OWN frame.
        @test PNM.get_segment_orientations(chain, key) ==
              _reference_chain_data(key, chain, nrd)[2]
        # And the group's frame resolves for both members, reframed where the keys disagree.
        @test PNM.get_segment_orientations(chain, group_arc) ==
              _orientations_in_frame(group_arc, chain, nrd)
        if key != group_arc
            @test PNM.get_segment_orientations(chain, group_arc) !=
                  PNM.get_segment_orientations(chain, key)
        end
    end
end
