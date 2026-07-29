# Characterization tests for the six reduction-map traversals. These pin CURRENT behavior across
# four resolution cases each -- forward hit, reverse hit, added-Ward-arc hit, unmapped miss --
# so a consolidation onto a shared walker cannot silently flatten the differences between them.
# The functions deliberately disagree: only `arc_dc_phase_shift` negates on a reverse hit, only
# `_reduced_arc_equivalent_branch` returns `nothing` instead of throwing, and the three DC
# accessors each treat an added Ward arc differently.

# Forward/reverse/miss cases: L1 ∥ PST on (1, 2), plain L2 on (2, 3). No reduction, so the
# parallel group is reachable both ways and every direct arc is forward-only.
function _arc_resolution_fixtures()
    sys = _mk_line_pst_parallel_system()
    return Ybus(sys).network_reduction_data
end

# Added Ward arcs: Ward populates `added_arc_impedance_map` with `GenericArcImpedance` entries.
function _ward_added_arc_nr()
    sys = PSB.build_system(PSB.PSIDTestSystems, "psid_test_ieee_9bus")
    ybus = Ybus(sys; network_reductions = NetworkReduction[WardReduction([1, 2, 5, 4, 7])])
    return ybus.network_reduction_data
end

@testset "arc resolution: forward hits" begin
    nr = _arc_resolution_fixtures()
    parallel_arc = (1, 2)
    direct_arc = (2, 3)

    # Parallel group, forward key.
    @test PNM.arc_dc_phase_shift(nr, parallel_arc) > 0.0
    @test PNM._arc_dc_susceptance(nr, parallel_arc) ≈ 15.0
    @test PNM.arc_dc_resistance(nr, parallel_arc) ≈ 0.0
    @test PNM.arc_equivalent_branch(nr, parallel_arc) isa PNM.EquivalentBranch
    @test length(PNM.arc_equivalent_branches(nr, parallel_arc)) == 1
    @test PNM._reduced_arc_equivalent_branch(nr, parallel_arc) isa PNM.EquivalentBranch

    # Direct branch, forward key.
    @test iszero(PNM.arc_dc_phase_shift(nr, direct_arc))
    @test PNM._arc_dc_susceptance(nr, direct_arc) ≈ 10.0
    @test PNM.arc_dc_resistance(nr, direct_arc) ≈ 0.0
    @test PNM.arc_equivalent_branch(nr, direct_arc) isa PNM.EquivalentBranch
    @test length(PNM.arc_equivalent_branches(nr, direct_arc)) == 1
    # A direct arc is NOT "reduced": this returns nothing, it does not throw.
    @test isnothing(PNM._reduced_arc_equivalent_branch(nr, direct_arc))
end

@testset "arc resolution: reverse hits" begin
    nr = _arc_resolution_fixtures()
    forward = (1, 2)
    reverse = (2, 1)

    # Only the group maps are probed in reverse, and only the phase shift negates.
    @test PNM.arc_dc_phase_shift(nr, reverse) ≈ -PNM.arc_dc_phase_shift(nr, forward)
    @test PNM._arc_dc_susceptance(nr, reverse) ≈ PNM._arc_dc_susceptance(nr, forward)
    @test PNM.arc_dc_resistance(nr, reverse) ≈ PNM.arc_dc_resistance(nr, forward)
    fwd_eb = PNM.arc_equivalent_branch(nr, forward)
    rev_eb = PNM.arc_equivalent_branch(nr, reverse)
    @test PNM.get_equivalent_shift(rev_eb) ≈ -PNM.get_equivalent_shift(fwd_eb)
    @test PNM.get_equivalent_x(rev_eb) ≈ PNM.get_equivalent_x(fwd_eb)
    @test length(PNM.arc_equivalent_branches(nr, reverse)) ==
          length(PNM.arc_equivalent_branches(nr, forward))
    @test PNM._reduced_arc_equivalent_branch(nr, reverse) isa PNM.EquivalentBranch

    # The DIRECT map is probed forward only, so a reversed direct arc is a miss, not a hit.
    @test_throws ErrorException PNM.arc_dc_phase_shift(nr, (3, 2))
    @test_throws ErrorException PNM.arc_equivalent_branch(nr, (3, 2))
    @test isnothing(PNM._reduced_arc_equivalent_branch(nr, (3, 2)))
end

@testset "arc resolution: added Ward arcs" begin
    nr = _ward_added_arc_nr()
    added = PNM.get_added_arc_impedance_map(nr)
    @test !isempty(added)
    arc = first(keys(added))

    # The three DC accessors disagree here, deliberately.
    @test iszero(PNM.arc_dc_phase_shift(nr, arc))          # GenericArcImpedance never shifts
    @test PNM.arc_dc_resistance(nr, arc) isa Float64        # returns the added arc's r
    @test_throws ErrorException PNM._arc_dc_susceptance(nr, arc)  # not probed: errors
    @test PNM.arc_equivalent_branch(nr, arc) isa PNM.EquivalentBranch
    @test length(PNM.arc_equivalent_branches(nr, arc)) == 1
    # Added arcs are not in the series/parallel maps, so this reports "not reduced".
    @test isnothing(PNM._reduced_arc_equivalent_branch(nr, arc))
end

@testset "arc resolution: unmapped arcs" begin
    nr = _arc_resolution_fixtures()
    missing_arc = (7, 9)

    @test_throws ErrorException PNM.arc_dc_phase_shift(nr, missing_arc)
    @test_throws ErrorException PNM._arc_dc_susceptance(nr, missing_arc)
    @test_throws ErrorException PNM.arc_dc_resistance(nr, missing_arc)
    @test_throws ErrorException PNM.arc_equivalent_branch(nr, missing_arc)
    @test_throws ErrorException PNM.arc_equivalent_branches(nr, missing_arc)
    # The lone non-throwing member of the family.
    @test isnothing(PNM._reduced_arc_equivalent_branch(nr, missing_arc))
end

@testset "arc resolution: series chain hits" begin
    # DegreeTwoReduction folds bus 2, producing a series chain keyed (1, 3).
    sys = _mk_line_pst_parallel_system()
    nr =
        Ybus(
            sys;
            network_reductions = NetworkReduction[DegreeTwoReduction()],
        ).network_reduction_data
    arc = first(keys(PNM.get_series_branch_map(nr)))

    @test PNM.arc_dc_phase_shift(nr, arc) isa Float64
    @test PNM._arc_dc_susceptance(nr, arc) > 0.0
    @test PNM.arc_dc_resistance(nr, arc) isa Float64
    @test PNM.arc_equivalent_branch(nr, arc) isa PNM.EquivalentBranch
    @test length(PNM.arc_equivalent_branches(nr, arc)) == 1
    @test PNM._reduced_arc_equivalent_branch(nr, arc) isa PNM.EquivalentBranch
    # Reverse key resolves too, with the shift negated.
    rev = (arc[2], arc[1])
    @test PNM.arc_dc_phase_shift(nr, rev) ≈ -PNM.arc_dc_phase_shift(nr, arc)
end
