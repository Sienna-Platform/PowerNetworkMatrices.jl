#=
`PNM._get_chain_data(equivalent_arc, chain, nrd)` was deleted in `e67ee3b`; PowerOperationsModels
still calls it to sign interface flow contributions. `get_segment_orientations` replaces its
second return value. These tests pin the equivalence, because the replacement reads a stored
field where the old code recomputed the walk.

The reference below is the removed function's body verbatim. It lives here, not in `src/`: the
point is to prove the stored vector agrees with it, not to keep the algorithm.
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

@testset "get_segment_orientations matches for chains nested in a parallel group" begin
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    groups = [
        (arc, grp) for (arc, grp) in PNM.get_parallel_branch_map(nrd) if
        all(m isa PNM.BranchesSeries for m in grp)
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

@testset "get_segment_orientations rejects an arc that is not the chain's frame" begin
    sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = PNM.get_network_reduction_data(ybus)

    arc, chain = first(PNM.get_series_branch_map(nrd))
    reversed = (arc[2], arc[1])

    # The old function walked from `equivalent_arc[1]` through segments stored in one order, so a
    # reversed arc threw "disconnected series chain" rather than returning flipped orientations.
    # Erroring here keeps that contract instead of inventing a reversal.
    @test_throws ErrorException PNM.get_segment_orientations(chain, reversed)
    @test_throws ErrorException _reference_chain_data(reversed, chain, nrd)
end

@testset "get_segment_orientations is framed per member, not per group" begin
    # The group is framed `(1, 3)` but one member is keyed `(3, 1)`. This is the case that
    # motivated checking the frame at all: a caller holding the group's arc must not silently
    # receive a member's orientations recorded against the opposite traversal.
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
        if key != group_arc
            # And the group's frame is rejected for the member keyed the other way, matching
            # the old function, which threw "disconnected series chain" on that input.
            @test_throws ErrorException PNM.get_segment_orientations(chain, group_arc)
            @test_throws ErrorException _reference_chain_data(group_arc, chain, nrd)
        end
    end
end
