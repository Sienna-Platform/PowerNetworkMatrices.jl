import SparseArrays

@testset "Large 2d Reduction Test" begin
    sys = build_system(
        MatpowerTestSystems,
        "matpower_ACTIVSg10k_sys",
    )
    ybus = Ybus(sys)
    reduction = PNM.get_reduction(ybus, sys, DegreeTwoReduction())
    @test !isempty(reduction.series_branch_map)
end

# Five-bus meshed system whose degree-two reduction folds a 3W-transformer winding into a
# series chain that lies inside a loop:
#
#   B1 --L12-- B2
#   |           \
#   L1_10      winding2
#   |             \
#   B10 -winding1- STAR(99) -winding3- B3 (radial)
#
# B10 hosts no injection and has degree 2 (one line + one winding), so DegreeTwoReduction
# folds [L1_10, winding1] into composite arc (1, 99). That arc closes the loop
# 1 - 2 - STAR - 1, so reduced DC sensitivities are impedance-weighted through the chain
# equivalent — unlike a radial chain, where PTDF entries are topological (0/±1) and
# insensitive to the chain's susceptance value.
function _build_meshed_3wt_loop_system()
    sys = PSY.System(100.0)
    mkbus(n, name, bt) = PSY.ACBus(;
        number = n,
        name = name,
        available = true,
        bustype = bt,
        angle = 0.0,
        magnitude = 1.0,
        voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 138.0,
    )
    b1 = mkbus(1, "B1", PSY.ACBusTypes.REF)
    b2 = mkbus(2, "B2", PSY.ACBusTypes.PV)
    b10 = mkbus(10, "B10", PSY.ACBusTypes.PQ)
    b3 = mkbus(3, "B3", PSY.ACBusTypes.PQ)
    star = mkbus(99, "STAR", PSY.ACBusTypes.PQ)
    foreach(b -> PSY.add_component!(sys, b), (b1, b2, b10, b3, star))
    for (bus, name) in ((b1, "g1"), (b2, "g2"))
        gen = PSY.ThermalStandard(;
            name = name,
            available = true,
            status = true,
            bus = bus,
            active_power = 1.0,
            reactive_power = 0.0,
            rating = 5.0,
            active_power_limits = (min = 0.0, max = 5.0),
            reactive_power_limits = (min = -5.0, max = 5.0),
            ramp_limits = nothing,
            time_limits = nothing,
            operation_cost = PSY.ThermalGenerationCost(nothing),
            base_power = 100.0,
            prime_mover_type = PSY.PrimeMovers.ST,
            fuel = PSY.ThermalFuels.COAL,
        )
        PSY.add_component!(sys, gen)
    end
    load = PSY.PowerLoad(;
        name = "load_star",
        available = true,
        bus = star,
        active_power = 1.5,
        reactive_power = 0.0,
        base_power = 100.0,
        max_active_power = 1.5,
        max_reactive_power = 0.0,
    )
    PSY.add_component!(sys, load)
    function mkline(name, f, t, x)
        arc = PSY.Arc(; from = f, to = t)
        PSY.add_component!(sys, arc)
        line = PSY.Line(;
            name = name,
            available = true,
            active_power_flow = 0.0,
            reactive_power_flow = 0.0,
            arc = arc,
            r = 0.01,
            x = x,
            b = (from = 0.0, to = 0.0),
            rating = 10.0,
            angle_limits = (min = -1.5, max = 1.5),
        )
        PSY.add_component!(sys, line)
        return line
    end
    mkline("L12", b1, b2, 0.10)
    mkline("L1_10", b1, b10, 0.05)
    arcs = (
        PSY.Arc(; from = b10, to = star),
        PSY.Arc(; from = b2, to = star),
        PSY.Arc(; from = b3, to = star),
    )
    foreach(a -> PSY.add_component!(sys, a), arcs)
    # Winding-resident star-leg impedances derived from the pairwise data (as PFFP does at
    # parse): z1 = (z12 + z31 - z23)/2, z2 = (z12 + z23 - z31)/2, z3 = (z31 + z23 - z12)/2.
    z12, z23, z31 = complex(0.01, 0.10), complex(0.01, 0.12), complex(0.01, 0.08)
    legs = (
        (z12 + z31 - z23) / 2,
        (z12 + z23 - z31) / 2,
        (z31 + z23 - z12) / 2,
    )
    circuits = ntuple(
        i -> PSY.TransformerCircuit(;
            arc = arcs[i],
            available = true,
            base_power = 100.0,
            base_voltage_primary = 138.0,
            r = real(legs[i]),
            x = imag(legs[i]),
            rating = 5.0,
        ),
        3,
    )
    t3w = PSY.ThreeWindingTransformer(;
        name = "T3W",
        primary_circuit = circuits[1],
        secondary_circuit = circuits[2],
        tertiary_circuit = circuits[3],
        star_bus = star,
        r_12 = 0.01, x_12 = 0.10,
        r_23 = 0.01, x_23 = 0.12,
        r_31 = 0.01, x_31 = 0.08,
        base_power_12 = 100.0, base_power_23 = 100.0, base_power_31 = 100.0,
    )
    PSY.add_component!(sys, t3w)
    return sys
end

@testset "2d reduction of a 3W winding in a loop preserves PTDF" begin
    sys = _build_meshed_3wt_loop_system()
    reductions = PNM.NetworkReduction[PNM.RadialReduction(), PNM.DegreeTwoReduction()]

    # Guard the fixture: the reduction must fold a ThreeWindingTransformerCircuit into a
    # series chain, otherwise this testset silently stops covering the winding path.
    ybus = PNM.Ybus(sys; network_reductions = reductions)
    nr = PNM.get_network_reduction_data(ybus)
    series_map = PNM.get_series_branch_map(nr)
    @test any(
        any(m isa PNM.ThreeWindingTransformerCircuit for m in chain)
        for chain in values(series_map)
    )

    # Reduced PTDF must reproduce the full PTDF on every arc surviving the reduction
    # (matching either orientation, with the sign flipped for a reversed arc).
    ptdf_full = PNM.PTDF(sys)
    ptdf_red = PNM.PTDF(sys; network_reductions = reductions)
    full_buses, full_arcs = ptdf_full.axes
    red_buses, red_arcs = ptdf_red.axes
    red_arc_lookup = Dict(a => i for (i, a) in enumerate(red_arcs))
    red_bus_lookup = Dict(b => i for (i, b) in enumerate(red_buses))
    common_buses = intersect(Set(full_buses), Set(red_buses))
    n_compared = 0
    for (ia, arc) in enumerate(full_arcs)
        sgn = 1.0
        ja = get(red_arc_lookup, arc, 0)
        if ja == 0
            ja = get(red_arc_lookup, (arc[2], arc[1]), 0)
            sgn = -1.0
        end
        ja == 0 && continue
        for bus in common_buses
            v_full = ptdf_full.data[findfirst(==(bus), full_buses), ia]
            v_red = sgn * ptdf_red.data[red_bus_lookup[bus], ja]
            @test isapprox(v_full, v_red; atol = 1e-6)
            n_compared += 1
        end
    end
    @test n_compared > 0
end

@testset "BranchesSeries carries an arc identity" begin
    bs = PNM.BranchesSeries((7, 11))
    @test PNM.get_arc_tuple(bs) == (7, 11)
    # Remapped through a reduction the same way a parallel group's arc_key is.
    nrd = PNM.NetworkReductionData(; reverse_bus_search_map = Dict(11 => 4))
    @test PNM.get_arc_tuple(bs, nrd) == (7, 4)
end

@testset "parallel group of series chains yields the summed two-port" begin
    # Two three-segment chains between Bus 1 and Bus 4 of c_sys14, built by hand so the
    # aggregate can be exercised without depending on the reduction that will create it.
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    nrd = PNM.NetworkReductionData()
    lines = collect(get_components(Line, sys))
    chain_a = PNM.BranchesSeries((1, 4))
    PNM.add_branch!(chain_a, lines[1], :FromTo)
    PNM.add_branch!(chain_a, lines[2], :FromTo)
    chain_b = PNM.BranchesSeries((1, 4))
    PNM.add_branch!(chain_b, lines[3], :FromTo)
    PNM.add_branch!(chain_b, lines[4], :FromTo)

    group = PNM.BranchesParallel(PNM.BranchesSeries[chain_a, chain_b])
    @test PNM.get_arc_tuple(group, nrd) == (1, 4)

    ya = PNM.ybus_branch_entries(chain_a, nrd)
    yb = PNM.ybus_branch_entries(chain_b, nrd)
    yg = PNM.ybus_branch_entries(group, nrd)
    # Parallel members in the same arc frame add entry-wise. A single chain's two-port comes
    # back as ComplexF32 (Ybus storage precision), while the group accumulates in ComplexF64;
    # promote before summing so the comparison isn't dominated by a second, avoidable F32
    # rounding of the addition itself.
    for i in 1:4
        @test isapprox(yg[i], ComplexF64(ya[i]) + ComplexF64(yb[i]); rtol = 1e-10)
    end
end

@testset "parallel group reports a nested chain's phase shift" begin
    sys = build_system(PSITestSystems, "c_sys14"; add_forecasts = false)
    nrd = PNM.NetworkReductionData()
    lines = collect(get_components(Line, sys))
    pst = first(get_components(PSY.TwoWindingTransformer, sys))
    PSY.set_α!(PSY.get_circuit(pst), 0.15)

    # One chain carries the shifter, the other is purely lossless line segments.
    shifted = PNM.BranchesSeries((1, 4))
    PNM.add_branch!(shifted, pst, :FromTo)
    PNM.add_branch!(shifted, lines[1], :FromTo)
    plain = PNM.BranchesSeries((1, 4))
    PNM.add_branch!(plain, lines[2], :FromTo)
    PNM.add_branch!(plain, lines[3], :FromTo)

    # The chain's own accumulated shift is the reference value.
    chain_alpha = PNM.get_series_phase_shift(shifted, nrd)
    @test chain_alpha != 0.0

    group = PNM.BranchesParallel(PNM.BranchesSeries[shifted, plain])
    b_shifted = PNM.get_series_susceptance(shifted, PSY.SU)
    b_plain = PNM.get_series_susceptance(plain, PSY.SU)
    expected = b_shifted * chain_alpha / (b_shifted + b_plain)
    @test isapprox(PNM.get_series_phase_shift(group, nrd), expected; rtol = 1e-10)
end

# Symmetric value-based adjacency from an undirected edge list.
function _adjacency_from_edges(edges, n)
    I = Int[]
    J = Int[]
    for (a, b) in edges
        push!(I, a, b)
        push!(J, b, a)
    end
    return SparseArrays.sparse(I, J, ones(Int8, length(I)), n, n)
end

@testset "find_degree2_chains returns every sibling chain on one endpoint pair" begin
    # Buses 1 and 2 are held above degree two by a triangle core on 7 and 8, so they are chain
    # terminals while 7 and 8 are degree three. Two interior paths connect them, one through
    # 3-4 and one through 5-6, and both are valid chains on the endpoint pair (1, 2).
    edges = [(1, 3), (3, 4), (4, 2), (1, 5), (5, 6), (6, 2),
        (1, 7), (1, 8), (2, 7), (2, 8), (7, 8)]
    A = _adjacency_from_edges(edges, 8)
    chains = PNM.find_degree2_chains(A, Set{Int}())
    @test length(chains) == 2
    @test Set(Set(c) for c in chains) == Set([Set([1, 3, 4, 2]), Set([1, 5, 6, 2])])
end

@testset "find_degree2_chains returns opposite-traversal siblings separately" begin
    # Same topology, but the second chain's interior numbering makes it traverse 2 -> 1.
    edges = [(1, 3), (3, 4), (4, 2), (2, 5), (5, 6), (6, 1),
        (1, 7), (1, 8), (2, 7), (2, 8), (7, 8)]
    A = _adjacency_from_edges(edges, 8)
    chains = PNM.find_degree2_chains(A, Set{Int}())
    @test length(chains) == 2
    # Endpoint pair is the same unordered set regardless of traversal direction.
    @test all(Set([c[1], c[end]]) == Set([1, 2]) for c in chains)
end

@testset "find_degree2_chains keeps a chain whose endpoints already share an arc" begin
    # Buses 1 and 2 are held above degree two by a triangle core on 7 and 8. One interior path
    # runs 1 - 3 - 4 - 2 and a plain arc spans (1, 2), so the chain is electrically parallel to
    # that arc and must still be returned whole.
    edges = [(1, 3), (3, 4), (4, 2), (1, 2),
        (1, 7), (1, 8), (2, 7), (2, 8), (7, 8)]
    A = _adjacency_from_edges(edges, 8)
    chains = PNM.find_degree2_chains(A, Set{Int}())
    @test length(chains) == 1
    @test Set(only(chains)) == Set([1, 3, 4, 2])
end

@testset "find_degree2_chains truncates a circular chain by one node" begin
    # A four-bus ring with only bus 1 held irreducible: the traversal starts and ends on bus 1,
    # which is the one shape that is still not a chain. The longest valid subchain drops the
    # repeated endpoint, leaving bus 4 as the second terminal.
    A = _adjacency_from_edges([(1, 2), (2, 3), (3, 4), (4, 1)], 4)
    chains = PNM.find_degree2_chains(A, Set{Int}([1]))
    @test length(chains) == 1
    @test only(chains) == [1, 2, 3, 4]
end

@testset "DegreeTwoReduction: a chain groups with a pre-existing direct arc" begin
    sys = build_chain_parallel_to_direct_line()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)

    # Both chain interiors are eliminated even though buses 1 and 3 already carry a line.
    @test Set(PNM.get_bus_axis(ybus)) == Set([1, 2, 3, 4])
    for b in (10, 11)
        @test b in PNM.get_removed_buses(nrd)
    end

    # The bus pair keeps exactly one forward-map entry: a group holding the line and the chain.
    @test Set(keys(PNM.get_parallel_branch_map(nrd))) == Set([(1, 3)])
    @test !haskey(PNM.get_series_branch_map(nrd), (1, 3))
    @test !haskey(PNM.get_series_branch_map(nrd), (3, 1))
    @test !haskey(PNM.get_direct_branch_map(nrd), (1, 3))
    group = PNM.get_parallel_branch_map(nrd)[(1, 3)]
    @test group isa PNM.MixedBranchesParallel
    @test count(m -> m isa PNM.BranchesSeries, group) == 1
    @test count(m -> m isa Line, group) == 1
    @test PNM.get_arc_tuple(group, nrd) == (1, 3)

    # The line and every chain segment resolve to the composite arc through the parallel map,
    # and each physical branch is registered in exactly one reverse map.
    reverse_series_map = PNM.get_reverse_series_branch_map(nrd)
    reverse_direct_map = PNM.get_reverse_direct_branch_map(nrd)
    for name in ["L_1_3", "L_1_10", "L_10_11", "L_11_3"]
        br = get_component(Line, sys, name)
        tag, arc = PNM._resolve_branch_arc(nrd, br)
        @test tag == :parallel
        @test arc == (1, 3)
        @test !haskey(reverse_series_map, br)
        @test !haskey(reverse_direct_map, br)
    end
end

@testset "DegreeTwoReduction: a chain groups with a reversed pre-existing arc" begin
    sys = build_reversed_chain_parallel_to_direct_line()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)

    @test Set(PNM.get_bus_axis(ybus)) == Set([1, 2, 3, 4])

    # The line's established key wins, so the group is filed under (3, 1) even though the chain
    # was discovered running 1 -> 3, and the chain member is then in the transposed frame.
    @test Set(keys(PNM.get_parallel_branch_map(nrd))) == Set([(3, 1)])
    group = PNM.get_parallel_branch_map(nrd)[(3, 1)]
    @test PNM.get_arc_tuple(group, nrd) == (3, 1)
    chain = only(m for m in group if m isa PNM.BranchesSeries)
    @test PNM.get_arc_tuple(chain, nrd) == (1, 3)

    # The chain's own two-port is asymmetric, so a missing transpose would change the group's.
    c11, c12, c21, c22 = PNM.ybus_branch_entries(chain, nrd)
    @test abs(c11 - c22) > 1e-3
    line = get_component(Line, sys, "L_3_1")
    l11, l12, l21, l22 = PNM.ybus_branch_entries(line, nrd)
    g11, g12, g21, g22 = PNM.ybus_branch_entries(group, nrd)
    @test isapprox(g11, l11 + c22; rtol = 1e-6)
    @test isapprox(g12, l12 + c21; rtol = 1e-6)
    @test isapprox(g21, l21 + c12; rtol = 1e-6)
    @test isapprox(g22, l22 + c11; rtol = 1e-6)
end

@testset "DegreeTwoReduction: sibling chains join an existing parallel group" begin
    # Two parallel lines on (1, 3) make the pair a parallel group before any reduction runs, and
    # the two sibling chains between the same buses form a group of their own. The two groups
    # must end up as one entry holding all four members.
    sys = build_two_parallel_degree_two_chains()
    arc = Arc(
        get_component(ACBus, sys, "Bus 1"),
        get_component(ACBus, sys, "Bus 3"),
    )
    PSY.add_component!(sys, arc)
    for (name, x) in (("L_1_3_a", 0.30), ("L_1_3_b", 0.35))
        PSY.add_component!(
            sys,
            Line(name, true, 0.0, 0.0, arc, 0.0, x, (from = 0.0, to = 0.0), 2.0,
                (-1.6, 1.6)),
        )
    end
    y_full = Ybus(sys)
    y_red = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(y_red)

    @test Set(PNM.get_bus_axis(y_red)) == Set([1, 2, 3, 4])
    @test Set(keys(PNM.get_parallel_branch_map(nrd))) == Set([(1, 3)])
    @test isempty(PNM.get_series_branch_map(nrd))
    group = PNM.get_parallel_branch_map(nrd)[(1, 3)]
    @test group isa PNM.MixedBranchesParallel
    @test count(m -> m isa Line, group) == 2
    @test count(m -> m isa PNM.BranchesSeries, group) == 2

    keep = [PNM.get_bus_lookup(y_full)[b] for b in PNM.get_bus_axis(y_red)]
    drop = setdiff(1:size(y_full.data, 1), keep)
    Ykk = Matrix(y_full.data[keep, keep])
    Ykd = Matrix(y_full.data[keep, drop])
    Ydk = Matrix(y_full.data[drop, keep])
    Ydd = Matrix(y_full.data[drop, drop])
    expected = Ykk - Ykd * (Ydd \ Ydk)
    reduced = Matrix(y_red.data)
    for i in axes(reduced, 1), j in axes(reduced, 2)
        @test isapprox(reduced[i, j], expected[i, j]; rtol = 1e-4, atol = 1e-4)
    end
end

@testset "composite raw two-port is complete over a mixed group's member types" begin
    # `_composite_raw_two_port`'s `PSY.ACTransmission` arm is a completeness arm: the Ybus
    # assembly path stamps composite arcs before any physical branch joins the group, so it
    # reaches only the chain and group arms. This exercises all three directly.
    sys = build_chain_parallel_to_direct_line()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)
    group = PNM.get_parallel_branch_map(nrd)[(1, 3)]
    line = only(m for m in group if m isa Line)
    chain = only(m for m in group if m isa PNM.BranchesSeries)

    # A physical branch's raw contribution is its whole two-port: it really is connected across
    # the composite arc in the unreduced Ybus.
    line_raw = PNM._composite_raw_two_port(line, nrd)
    @test all(
        isapprox(line_raw[i], PNM.ybus_branch_entries(line, nrd)[i]; rtol = 1e-6) for
        i in 1:4
    )

    # A chain contributes only diagonals, so the group's raw off-diagonals are the line's alone.
    chain_raw = PNM._composite_raw_two_port(chain, nrd)
    @test iszero(chain_raw[2])
    @test iszero(chain_raw[3])
    group_raw = PNM._composite_raw_two_port(group, nrd)
    @test isapprox(group_raw[2], line_raw[2]; rtol = 1e-6)
    @test isapprox(group_raw[3], line_raw[3]; rtol = 1e-6)
    @test isapprox(group_raw[1], line_raw[1] + chain_raw[1]; rtol = 1e-6)
    @test isapprox(group_raw[4], line_raw[4] + chain_raw[4]; rtol = 1e-6)
end

@testset "DegreeTwoReduction: sibling chains on one arc become a parallel group" begin
    # A meshed core (buses 1-4) with two independent two-bus chains between buses 1 and 3.
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)

    # Every interior bus of both chains is eliminated; neither chain is left behind.
    for b in (10, 11, 20, 21)
        @test b in PNM.get_removed_buses(nrd)
        @test b ∉ keys(PNM.get_bus_lookup(ybus))
    end

    # The composite arc carries a parallel group of two chains, not a bare chain.
    group_arcs = [
        k for (k, v) in PNM.get_parallel_branch_map(nrd)
        if all(m isa PNM.BranchesSeries for m in v)
    ]
    @test length(group_arcs) == 1
    group = PNM.get_parallel_branch_map(nrd)[only(group_arcs)]
    @test length(group) == 2
    @test Set(only(group_arcs)) == Set([1, 3])
    # No composite arc is left in the series map for that pair, in either orientation.
    @test !haskey(PNM.get_series_branch_map(nrd), (1, 3))
    @test !haskey(PNM.get_series_branch_map(nrd), (3, 1))
end

@testset "grouped chain members resolve to their composite arc" begin
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)

    # Every line that was folded into either chain must resolve to the composite arc, by
    # physical component identity, not by the aggregate wrapper.
    for name in ["L_1_10", "L_10_11", "L_11_3", "L_1_20", "L_20_21", "L_21_3"]
        br = get_component(Line, sys, name)
        tag, arc = PNM._resolve_branch_arc(nrd, br)
        @test tag == :parallel
        @test Set(arc) == Set([1, 3])
    end
end

"""
Meshed core on buses 1-4 with TWO independent grouped degree-two composite arcs: one on
(1, 3) (interiors 10-11 and 20-21) and one on (2, 4) (interiors 40-41 and 42-43). Neither
composite arc's endpoints host an injector, so a `WardReduction` naming only [1, 2, 3] as
study buses drops bus 4 and, with it, the whole (2, 4) composite arc — while (1, 3) survives
untouched in `parallel_branch_map`.

This is the system used to exercise `_remake_reverse_parallel_branch_map!`: removing the
(2, 4) composite arc forces the reverse-map rebuild, and since that rebuild recomputes the
map from every surviving entry in `parallel_branch_map`, the (1, 3) composite arc's
recursive registration goes through the rebuild too even though it was never itself removed.
"""
function _build_two_composite_arcs_system()
    edges = [
        (1, 2, 0.0, 0.05, 0.0, 0.0), (2, 3, 0.0, 0.06, 0.0, 0.0),
        (3, 4, 0.0, 0.07, 0.0, 0.0), (4, 1, 0.0, 0.08, 0.0, 0.0),
        (1, 10, 0.0, 0.10, 0.0, 0.0), (10, 11, 0.0, 0.11, 0.0, 0.0),
        (11, 3, 0.0, 0.12, 0.0, 0.0),
        (1, 20, 0.0, 0.20, 0.0, 0.0), (20, 21, 0.0, 0.21, 0.0, 0.0),
        (21, 3, 0.0, 0.22, 0.0, 0.0),
        (2, 40, 0.0, 0.30, 0.0, 0.0), (40, 41, 0.0, 0.31, 0.0, 0.0),
        (41, 4, 0.0, 0.32, 0.0, 0.0),
        (2, 42, 0.0, 0.40, 0.0, 0.0), (42, 43, 0.0, 0.41, 0.0, 0.0),
        (43, 4, 0.0, 0.42, 0.0, 0.0),
    ]
    sys = PSY.System(100.0)
    buses = Dict{Int, ACBus}()
    for n in (1, 2, 3, 4, 10, 11, 20, 21, 40, 41, 42, 43)
        b = ACBus(;
            number = n,
            name = "Bus $n",
            available = true,
            bustype = n == 1 ? ACBusTypes.REF : ACBusTypes.PQ,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        )
        PSY.add_component!(sys, b)
        buses[n] = b
    end
    for (f, t, r, x, b_from, b_to) in edges
        arc = Arc(buses[f], buses[t])
        PSY.add_component!(sys, arc)
        PSY.add_component!(
            sys,
            Line("L_$(f)_$(t)", true, 0.0, 0.0, arc, r, x,
                (from = b_from, to = b_to), 2.0, (-1.6, 1.6)),
        )
    end
    PSY.add_component!(
        sys,
        ThermalStandard(; name = "G1", available = true, status = true, bus = buses[1],
            active_power = 1.0, reactive_power = 0.0, rating = 2.0,
            prime_mover_type =
            PSY.PrimeMovers.OT, fuel = PSY.ThermalFuels.OTHER,
            active_power_limits = (min = 0.0, max = 2.0),
            reactive_power_limits = nothing, ramp_limits = nothing,
            time_limits = nothing, base_power = 100.0,
            operation_cost = ThermalGenerationCost(nothing)),
    )
    PSY.add_component!(
        sys,
        PowerLoad(; name = "D3", available = true, bus = buses[3], active_power = 1.0,
            reactive_power = 0.0, base_power = 100.0, max_active_power = 1.0,
            max_reactive_power = 0.0),
    )
    return sys
end

@testset "grouped chain members survive a later rebuild of the reverse parallel map" begin
    sys = _build_two_composite_arcs_system()
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[
            DegreeTwoReduction(),
            WardReduction([1, 2, 3]),
        ],
    )
    nrd = get_network_reduction_data(ybus)

    # The (2, 4) composite arc was dropped by the Ward reduction; only (1, 3) remains.
    parallel_map = PNM.get_parallel_branch_map(nrd)
    @test Set(keys(parallel_map)) == Set([(1, 3)])

    # Every line folded into the surviving (1, 3) composite arc resolves to it by physical
    # component identity: the reverse map is keyed by each physical `Line`, not by the
    # `BranchesSeries` chain wrapping it.
    reverse_series_map = PNM.get_reverse_series_branch_map(nrd)
    for name in ["L_1_10", "L_10_11", "L_11_3", "L_1_20", "L_20_21", "L_21_3"]
        br = get_component(Line, sys, name)
        tag, arc = PNM._resolve_branch_arc(nrd, br)
        @test tag == :parallel
        @test Set(arc) == Set([1, 3])
        # A branch must be registered in exactly one reverse map, or `_resolve_branch_arc`'s
        # probe order would silently mask a duplicate registration.
        @test !haskey(reverse_series_map, br)
    end
end

@testset "series branch maps from successive reductions merge" begin
    sys = build_two_parallel_degree_two_chains()
    lines = collect(get_components(Line, sys))

    nr = PNM.NetworkReductionData()
    chain_a = PNM.BranchesSeries((1, 3))
    PNM.add_branch!(chain_a, lines[1], :FromTo)
    PNM.add_branch!(chain_a, lines[2], :FromTo)
    nr.series_branch_map[(1, 3)] = chain_a
    nr.reverse_series_branch_map =
        PNM._make_reverse_series_branch_map(nr.series_branch_map)

    nr_new = PNM.NetworkReductionData()
    chain_b = PNM.BranchesSeries((2, 4))
    PNM.add_branch!(chain_b, lines[3], :FromTo)
    PNM.add_branch!(chain_b, lines[4], :FromTo)
    nr_new.series_branch_map[(2, 4)] = chain_b
    nr_new.reverse_series_branch_map =
        PNM._make_reverse_series_branch_map(nr_new.series_branch_map)

    PNM._apply_composite_branch_maps!(nr, nr_new)

    @test Set(keys(nr.series_branch_map)) == Set([(1, 3), (2, 4)])
    # Every physical member of both chains resolves to its own composite arc.
    @test PNM._resolve_branch_arc(nr, lines[1]) == (:series, (1, 3))
    @test PNM._resolve_branch_arc(nr, lines[4]) == (:series, (2, 4))
end
