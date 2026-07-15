@testset "Invalid reduction combinations" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    @test_throws IS.DataFormatError("Radial reduction is applied twice to the same system") Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), RadialReduction()],
    )
    @test_throws IS.DataFormatError("Ward reduction must be the last applied reduction") Ybus(
        sys;
        network_reductions = NetworkReduction[WardReduction([1, 2, 4]), RadialReduction()],
    )
    @test_logs (
        :warn,
        r"When applying both RadialReduction and DegreeTwoReduction, it is likely beneficial to apply RadialReduction first",
    ) match_mode = :any Ybus(
        sys;
        network_reductions = NetworkReduction[DegreeTwoReduction(), RadialReduction()],
    )
end

function check_bus_arc_axis_consistency(A::IncidenceMatrix)
    arc_axis_numbers = Set()
    for arc in PNM.get_arc_axis(A)
        push!(arc_axis_numbers, arc[1])
        push!(arc_axis_numbers, arc[2])
    end
    bus_axis_numbers = Set(A.axes[2])
    @test isempty(setdiff(arc_axis_numbers, bus_axis_numbers))
    @test isempty(setdiff(bus_axis_numbers, arc_axis_numbers))
end

@testset "14 bus; default reductions" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    A = IncidenceMatrix(sys)
    check_bus_arc_axis_consistency(A)
    ybus = Ybus(sys)
    nrd = get_network_reduction_data(ybus)
    @test nrd.irreducible_buses == Set{Int}()
    @test length(keys(nrd.bus_reduction_map)) == 18
    @test nrd.bus_reduction_map[112] == Set([113])
    @test nrd.bus_reduction_map[104] == Set([105])
    @test nrd.reverse_bus_search_map == Dict{Int, Int}(105 => 104, 113 => 112)
    # 14 non-winding direct branches plus 6 three-winding windings, all in direct_branch_map.
    @test length(keys(nrd.direct_branch_map)) == 20
    @test count(
        v -> v isa PNM.ThreeWindingTransformerCircuit,
        values(nrd.direct_branch_map),
    ) == 6
    @test length(keys(nrd.parallel_branch_map)) == 3
    @test length(keys(nrd.series_branch_map)) == 0
    @test nrd.removed_buses == Set{Int}()
    @test nrd.removed_arcs == Set([(112, 113), (104, 105)])
    @test Set(keys(nrd.added_admittance_map)) == Set{Int}()
    @test Set(keys(nrd.added_arc_impedance_map)) == Set{Tuple{Int, Int}}()
    @test Set(A.axes[1]) == union(
        Set(keys(nrd.direct_branch_map)),
        Set(keys(nrd.parallel_branch_map)),
        Set(keys(nrd.series_branch_map)),
    )
    @test Set(A.axes[2]) == Set(keys(nrd.bus_reduction_map))
end

@testset "14 bus; + radial reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    A = IncidenceMatrix(sys; network_reductions = NetworkReduction[RadialReduction()])
    check_bus_arc_axis_consistency(A)
    ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    nrd = get_network_reduction_data(ybus)
    @test nrd.irreducible_buses == Set{Int}()
    @test length(keys(nrd.bus_reduction_map)) == 15
    @test nrd.bus_reduction_map[112] == Set([113])
    @test nrd.bus_reduction_map[104] == Set([105])
    @test nrd.bus_reduction_map[103] == Set([116])
    @test nrd.bus_reduction_map[1001] == Set([107, 108])
    @test nrd.reverse_bus_search_map ==
          Dict(113 => 112, 105 => 104, 116 => 103, 108 => 1001, 107 => 1001)
    # 12 non-winding direct branches plus 5 three-winding windings, all in direct_branch_map.
    @test length(keys(nrd.direct_branch_map)) == 17
    @test count(
        v -> v isa PNM.ThreeWindingTransformerCircuit,
        values(nrd.direct_branch_map),
    ) == 5
    @test length(keys(nrd.parallel_branch_map)) == 3
    @test length(keys(nrd.series_branch_map)) == 0
    @test nrd.removed_buses == Set{Int}()
    @test nrd.removed_arcs ==
          Set([(107, 108), (107, 1001), (103, 116), (112, 113), (104, 105)])
    @test Set(keys(nrd.added_admittance_map)) == Set{Int}()
    @test Set(keys(nrd.added_arc_impedance_map)) == Set{Tuple{Int, Int}}()
    @test Set(A.axes[1]) == union(
        Set(keys(nrd.direct_branch_map)),
        Set(keys(nrd.parallel_branch_map)),
        Set(keys(nrd.series_branch_map)),
    )
    @test Set(A.axes[2]) == Set(keys(nrd.bus_reduction_map))
end

@testset "14 bus; + degree two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    A = IncidenceMatrix(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    check_bus_arc_axis_consistency(A)
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd = get_network_reduction_data(ybus)
    @test nrd.irreducible_buses ==
          Set([112, 101, 114, 110, 105, 108, 103, 102, 111, 113, 104, 106, 109])
    @test length(keys(nrd.bus_reduction_map)) == 14
    @test nrd.bus_reduction_map[112] == Set([113])
    @test nrd.bus_reduction_map[104] == Set([105])
    @test nrd.reverse_bus_search_map == Dict(113 => 112, 105 => 104)
    # 9 non-winding direct branches plus 5 three-winding windings, all in direct_branch_map.
    @test length(keys(nrd.direct_branch_map)) == 14
    @test count(
        v -> v isa PNM.ThreeWindingTransformerCircuit,
        values(nrd.direct_branch_map),
    ) == 5
    @test length(keys(nrd.parallel_branch_map)) == 2
    @test length(keys(nrd.series_branch_map)) == 3
    @test length(keys(nrd.reverse_series_branch_map)) == 8
    @test nrd.removed_buses == Set([117, 107, 115, 118])
    @test nrd.removed_arcs == Set([
        (107, 108),
        (107, 1001),
        (118, 104),
        (115, 102),
        (101, 117),
        (112, 113),
        (104, 105),
        (101, 115),
        (117, 118),
    ])
    @test Set(keys(nrd.added_admittance_map)) == Set{Int}()
    @test Set(keys(nrd.added_arc_impedance_map)) == Set{Tuple{Int, Int}}()
    @test Set(A.axes[1]) == union(
        Set(keys(nrd.direct_branch_map)),
        Set(keys(nrd.parallel_branch_map)),
        Set(keys(nrd.series_branch_map)),
    )
    @test Set(A.axes[2]) == Set(keys(nrd.bus_reduction_map))
    ybus_full = Ybus(sys)
    # sqrt(eps) tolerance: reduced Ybus goes through a dense solve, so composed
    # numerical error is expected up to O(sqrt(eps)) for well-conditioned systems.
    @test isapprox(
        ybus[108, 1001]^-1,
        (ybus_full[108, 107])^-1 + ybus_full[107, 1001]^-1;
        rtol = sqrt(eps(real(YBUS_ELTYPE))),
    )
end

@testset "14 bus; radial + degree two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    A_both = IncidenceMatrix(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )
    check_bus_arc_axis_consistency(A_both)
    ybus_both = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()],
    )
    nrd_both = get_network_reduction_data(ybus_both)
    A_radial =
        IncidenceMatrix(sys; network_reductions = NetworkReduction[RadialReduction()])
    ybus_radial = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    nrd_radial = get_network_reduction_data(ybus_radial)
    A_d2 = IncidenceMatrix(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    ybus_d2 = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    nrd_d2 = get_network_reduction_data(ybus_d2)

    @test nrd_both.bus_reduction_map != nrd_radial.bus_reduction_map
    @test nrd_both.reverse_bus_search_map == nrd_radial.reverse_bus_search_map
    @test isempty(nrd_radial.removed_buses)
    @test !isempty(nrd_both.removed_buses)
end

@testset "14 bus; Ward reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    study_buses = [101, 114, 110, 111]
    boundary_buses = [101, 114, 110, 111]
    A = IncidenceMatrix(
        sys;
        network_reductions = NetworkReduction[WardReduction(study_buses)],
    )
    check_bus_arc_axis_consistency(A)
    ybus = Ybus(sys; network_reductions = NetworkReduction[WardReduction(study_buses)])
    nrd = get_network_reduction_data(ybus)
    @test Set(ybus.axes[1]) == Set(study_buses)
    @test length(nrd.added_admittance_map) == length(boundary_buses)
    @test length(nrd.added_arc_impedance_map) == factorial(length(boundary_buses) - 1)
    ybus_full = Ybus(sys)
    for i in study_buses, j in study_buses
        if i ∈ boundary_buses && j ∈ boundary_buses
            @test ybus_full[i, j] != ybus[i, j]
        else
            @test ybus_full[i, j] == ybus[i, j]
        end
    end
    @test Set(A.axes[1]) == union(
        Set(keys(nrd.direct_branch_map)),
        Set(keys(nrd.parallel_branch_map)),
        Set(keys(nrd.series_branch_map)),
        Set(keys(nrd.added_arc_impedance_map)),
    )
    @test Set(A.axes[2]) == Set(keys(nrd.bus_reduction_map))
end

@testset "Test irreducible_buses" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    # Test irreducible bus input for radial reduction
    ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    @test haskey(ybus.network_reduction_data.reverse_bus_search_map, 116)
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set([116]),
    )
    @test !haskey(ybus.network_reduction_data.reverse_bus_search_map, 116)
    @test ybus.network_reduction_data.irreducible_buses == Set{Int}(116)

    # Test irreducible bus input for degree two reduction
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    @test 117 ∈ ybus.network_reduction_data.removed_buses
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[DegreeTwoReduction()],
        irreducible_buses = Set([117]),
    )
    @test 117 ∉ ybus.network_reduction_data.removed_buses
    @test ybus.network_reduction_data.irreducible_buses ==
          Set{Int}([112, 101, 114, 110, 105, 108, 103, 102, 111, 113, 117, 104, 106, 109])
end

function set_radial_removed_arcs_to_unavailable!(sys, radial_removed_arcs, rbsm)
    for l in get_components(ACTransmission, sys)
        if typeof(l) <: ThreeWindingTransformer
            # Star arcs are the per-winding arcs; availability is per winding.
            for winding in get_circuits(l)
                star_arc = get_arc(winding)
                if ((star_arc.from.number, star_arc.to.number) ∈ radial_removed_arcs) ||
                   ((star_arc.to.number, star_arc.from.number) ∈ radial_removed_arcs)
                    set_available!(winding, false)
                    if star_arc.from.number ∈ keys(rbsm)
                        set_available!(star_arc.from, false)
                    end
                    if star_arc.to.number ∈ keys(rbsm)
                        set_available!(star_arc.to, false)
                    end
                end
            end
        else
            arc = get_arc(l)
            if (arc.from.number, arc.to.number) ∈ radial_removed_arcs
                set_available!(l, false)
                if arc.from.number ∈ keys(rbsm)
                    set_available!(arc.from, false)
                end
                if arc.to.number ∈ keys(rbsm)
                    set_available!(arc.to, false)
                end
            end
        end
    end
    return
end

# This test is designed to test the Ybus modifications needed when removing radial branches
@testset "Test compare Ybus matrices with radial reduction and manually removing radial components" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    ybus_1 = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    # Take the setdiff to ignore removed_arcs from breaker/switch reduction:
    radial_removed_arcs = setdiff(
        ybus_1.network_reduction_data.removed_arcs,
        Ybus(sys).network_reduction_data.removed_arcs,
    )
    rbsm = ybus_1.network_reduction_data.reverse_bus_search_map
    set_radial_removed_arcs_to_unavailable!(sys, radial_removed_arcs, rbsm)
    ybus_2 = Ybus(sys)

    for ix in PNM.get_bus_axis(ybus_1)
        for jx in PNM.get_bus_axis(ybus_1)
            @test isapprox(ybus_1[ix, jx], ybus_2[ix, jx])
        end
    end
end

function _set_zero_impedance!(branch)
    set_r!(branch, 0.0 * PSY.SU)
    set_x!(branch, 1e-5 * PSY.SU)
end

@testset "ZeroImpedanceBranchReduction: chained bus merge" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    _set_zero_impedance!(get_component(Line, sys, "Line3"))  # bus 2 → 3
    _set_zero_impedance!(get_component(Line, sys, "Line6"))  # bus 3 → 4

    ybus = Ybus(sys)
    nrd = ybus.network_reduction_data
    rbsm = nrd.reverse_bus_search_map

    @test get(rbsm, 3, nothing) == 2
    @test get(rbsm, 4, nothing) == 2

    @test 3 ∉ PNM.get_bus_axis(ybus)
    @test 4 ∉ PNM.get_bus_axis(ybus)
    @test length(PNM.get_bus_axis(ybus)) == 14 - 2
end

@testset "ZeroImpedanceBranchReduction: transformer arcs are excluded" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    t = get_component(TwoWindingTransformer, sys, "Trans4")  # from=7, to=8
    set_r!(t, 0.0 * PSY.SU)
    set_x!(t, 1e-5 * PSY.SU)

    ybus = Ybus(sys)
    nrd = ybus.network_reduction_data

    @test !haskey(nrd.reverse_bus_search_map, 7)
    @test !haskey(nrd.reverse_bus_search_map, 8)

    @test 7 ∈ PNM.get_bus_axis(ybus)
    @test 8 ∈ PNM.get_bus_axis(ybus)
end

@testset "ZeroImpedanceBranchReduction: degenerate 3WT merge promotes windings to a parallel group" begin
    # A NON-winding zero-impedance branch between two REAL terminal buses of the same
    # three-winding transformer gets ZIR-merged. Both winding arcs then remap to the same
    # (merged, star) arc and collide in the direct map, promoting the two windings into a
    # BranchesParallel group. Pin that the group's equivalent Ybus entry is the sum of the
    # two windings' Pi-models (physically, the windings really are in parallel between the
    # merged bus and the star point) and matches the merged matrix entry.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5_ml")
    busD = PSY.get_component(PSY.ACBus, sys, "nodeD")
    sec_bus, ter_bus, star_bus = _add_star_buses!(sys, busD)
    t3w = _add_three_winding_transformer!(
        sys, busD, sec_bus, ter_bus, star_bus; name = "T3W_degenerate",
    )
    zi_arc = PSY.Arc(; from = busD, to = sec_bus)
    PSY.add_component!(sys, zi_arc)
    zi_line = PSY.Line(;
        name = "zi_line",
        available = true,
        active_power_flow = 0.0,
        reactive_power_flow = 0.0,
        arc = zi_arc,
        r = 0.0,
        x = 1.0e-5,
        b = (from = 0.0, to = 0.0),
        rating = 10.0,
        angle_limits = (min = -1.57, max = 1.57),
    )
    PSY.add_component!(sys, zi_line)

    ybus = Ybus(sys)
    nrd = PNM.get_network_reduction_data(ybus)
    busD_no = PSY.get_number(busD)
    sec_no = PSY.get_number(sec_bus)
    star_no = PSY.get_number(star_bus)
    ter_no = PSY.get_number(ter_bus)

    # The zero-impedance line merged the secondary terminal into the primary terminal.
    @test get(nrd.reverse_bus_search_map, sec_no, nothing) == busD_no

    # After the merge, windings 1 and 2 share the (busD, star) arc as a parallel group.
    merged_arc = (busD_no, star_no)
    @test haskey(nrd.parallel_branch_map, merged_arc)
    bp = nrd.parallel_branch_map[merged_arc]
    @test bp isa PNM.BranchesParallel{PNM.ThreeWindingTransformerCircuit}
    @test Set(PNM.get_name.(bp.branches)) ==
          Set(["T3W_degenerate_winding_1", "T3W_degenerate_winding_2"])
    @test !haskey(nrd.direct_branch_map, merged_arc)
    # Winding 3 keeps its own direct arc to the star point.
    @test nrd.direct_branch_map[(ter_no, star_no)] isa PNM.ThreeWindingTransformerCircuit

    # Hand computation: with r12=r23=r31=0.01 and x12=x23=x31=0.1 (the helper defaults) each
    # star leg is z_leg = (0.01 + 0.1im)/2, so two identical legs in parallel give
    # y = 2 / z_leg with the standard Pi-model sign pattern (unit taps, no shunt).
    y_expected = 2 / (0.005 + 0.05im)
    Y11, Y12, Y21, Y22 = PNM.ybus_branch_entries(bp, nrd)
    @test Y11 ≈ y_expected atol = 1e-8
    @test Y12 ≈ -y_expected atol = 1e-8
    @test Y21 ≈ -y_expected atol = 1e-8
    @test Y22 ≈ y_expected atol = 1e-8
    # The group equivalent equals the sum of the member windings' own Pi-models.
    w1 = PNM.ThreeWindingTransformerCircuit(t3w, 1)
    w2 = PNM.ThreeWindingTransformerCircuit(t3w, 2)
    expected_sum = PNM.ybus_branch_entries(w1) .+ PNM.ybus_branch_entries(w2)
    @test all(isapprox.((Y11, Y12, Y21, Y22), expected_sum; atol = 1e-10))
    # And the merged matrix entry agrees with the group's off-diagonal.
    @test isapprox(ybus[busD_no, star_no], Y12; atol = 1e-4)
end

@testset "ZeroImpedanceBranchReduction respects irreducible buses" begin
    # The psse_14_network_reduction_test_system has two ZI branches:
    #   arc (112, 113) → bus 113 merged into 112 by default
    #   arc (104, 105) → bus 105 merged into 104 by default
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")

    # Baseline: confirm default merge direction.
    ybus_default = Ybus(sys)
    nrd_default = ybus_default.network_reduction_data
    @test get(nrd_default.reverse_bus_search_map, 113, nothing) == 112
    @test 112 ∉ keys(nrd_default.reverse_bus_search_map)

    # Mark bus 113 irreducible via a downstream RadialReduction.
    # ZIR should flip the (112,113) merge so 113 survives and 112 is removed.
    ybus_flip = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set([113]),
    )
    nrd_flip = ybus_flip.network_reduction_data
    @test 113 ∉ keys(nrd_flip.reverse_bus_search_map)   # 113 survived
    @test get(nrd_flip.reverse_bus_search_map, 112, nothing) == 113  # 112 removed → 113
    # The other ZI branch (104, 105) is unaffected.
    @test get(nrd_flip.reverse_bus_search_map, 105, nothing) == 104

    # Mark both sides of the (112, 113) arc irreducible: merge should be skipped.
    @test_logs (:warn, r"irreducible buses (112 and 113|113 and 112)") match_mode = :any Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set([112, 113]),
    )
    ybus_skip = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set([112, 113]),
    )
    nrd_skip = ybus_skip.network_reduction_data
    @test 112 ∉ keys(nrd_skip.reverse_bus_search_map)   # neither bus removed
    @test 113 ∉ keys(nrd_skip.reverse_bus_search_map)
    @test 112 ∈ PNM.get_bus_axis(ybus_skip)
    @test 113 ∈ PNM.get_bus_axis(ybus_skip)
end

@testset "ZeroImpedanceBranchReduction: custom susceptance_threshold" begin
    # Line3 (bus 2 → 3) gets r=0 and a reactance giving |imag(Y)| ≈ 1e3, below the
    # default 1e4 threshold, so it is NOT merged by default. A custom, lower threshold
    # makes it qualify and bus 3 merges into bus 2.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = get_component(Line, sys, "Line3")
    set_r!(line, 0.0 * PSY.SU)
    set_x!(line, 1e-3 * PSY.SU)  # susceptance ≈ 1 / 1e-3 = 1e3

    ybus_default = Ybus(sys)
    @test 3 ∈ PNM.get_bus_axis(ybus_default)
    @test !haskey(ybus_default.network_reduction_data.reverse_bus_search_map, 3)

    ybus_custom = Ybus(
        sys;
        zero_impedance_reduction = PNM.ZeroImpedanceBranchReduction(;
            susceptance_threshold = 1e2,
        ),
    )
    @test 3 ∉ PNM.get_bus_axis(ybus_custom)
    @test get(ybus_custom.network_reduction_data.reverse_bus_search_map, 3, nothing) == 2
end

@testset "ZeroImpedanceBranchReduction: custom minimum_retained_impedance" begin
    # A branch with r == x == 0 has its reactance replaced by `minimum_retained_impedance`
    # during assembly. With the default (1e-6) the substituted susceptance (~1e6) exceeds
    # the threshold and the branch is merged; a large custom value (1.0 → susceptance ~1.0)
    # drops below the threshold, so the branch is retained instead.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    line = get_component(Line, sys, "Line3")  # bus 2 → 3
    set_r!(line, 0.0 * PSY.SU)
    set_x!(line, 0.0 * PSY.SU)

    ybus_default = Ybus(sys)
    @test 3 ∉ PNM.get_bus_axis(ybus_default)  # merged with the default tiny substitute reactance

    ybus_custom = Ybus(
        sys;
        zero_impedance_reduction = PNM.ZeroImpedanceBranchReduction(;
            minimum_retained_impedance = 1.0,
        ),
    )
    @test 3 ∈ PNM.get_bus_axis(ybus_custom)  # retained: substituted susceptance below threshold
    @test !haskey(ybus_custom.network_reduction_data.reverse_bus_search_map, 3)
end

@testset "ZeroImpedanceBranchReduction: stub off a merged junction (no fake island)" begin
    # Regression for the chained zero-impedance topology
    #     grid ──normal── M(903) ──[ZI]──► J(901) ◄──[ZI]── S(902, stub)
    # where both zero-impedance branches are oriented INTO the junction J. The
    # junction must fold the WHOLE cluster {S, J, M} into one survivor. Previously
    # J was merged twice (its reverse-map entry overwritten), dropping the stub's
    # arc from the branch maps -> S stranded in BA (fake island) -> singular ABA.
    sys = PSB.build_system(PSITestSystems, "c_sys5")
    template = first(get_components(ACBus, sys))
    grid_bus = first(
        b for b in get_components(ACBus, sys) if get_bustype(b) != PSY.ACBusTypes.REF &&
        get_bustype(b) != PSY.ACBusTypes.ISOLATED
    )
    function _mk_bus(num, name)
        b = deepcopy(template)
        b.internal = IS.InfrastructureSystemsInternal()
        set_number!(b, num)
        set_name!(b, name)
        set_bustype!(b, PSY.ACBusTypes.PQ)
        return b
    end
    J = _mk_bus(901, "JUNCTION")
    S = _mk_bus(902, "STUB")
    M = _mk_bus(903, "MID")
    foreach(b -> add_component!(sys, b), (J, S, M))
    function _mk_line(name, from, to, r, x)
        arc = Arc(from, to)
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(
                name,
                true,
                0.0,
                0.0,
                arc,
                r,
                x,
                (from = 0.0, to = 0.0),
                100.0,
                (-1.5, 1.5),
            ),
        )
    end
    _mk_line("MID_grid", M, grid_bus, 0.01, 0.10)  # normal: M joins the grid
    _mk_line("ZI_M_J", M, J, 0.0, 1e-5)            # zero-impedance, into J
    _mk_line("ZI_S_J", S, J, 0.0, 1e-5)            # zero-impedance, into J

    Y = Ybus(sys)
    nr = Y.network_reduction_data
    # The whole zero-impedance cluster collapses to a single surviving bus.
    surv = PNM.get_mapped_bus_number(nr, 901)
    @test PNM.get_mapped_bus_number(nr, 902) == surv
    @test PNM.get_mapped_bus_number(nr, 903) == surv
    @test 901 ∉ PNM.get_bus_axis(Y)
    @test 902 ∉ PNM.get_bus_axis(Y)   # the stub is merged, not stranded
    # The ABA must be non-singular: a KLU PTDF builds without a singular solve.
    ptdf = PTDF(sys; linear_solver = "KLU")
    @test all(isfinite, ptdf.data)
end

@testset "ZeroImpedanceBranchReduction: column merge symmetric when survivor index > removed" begin
    # When ZIR merges a removed bus into a survivor whose bus index is GREATER than the
    # removed bus's (i > j), `_accumulate_csc_col_into!` must still copy every mutual from
    # the removed column. The old offset bookkeeping only held for i < j, so for i > j a
    # remapped arc's mutual was dropped on one side -> asymmetric Ybus -> BA computes
    # 1/imag(1/0) = NaN. Regression for that index-ordering bug.
    sys = PSB.build_system(PSITestSystems, "c_sys5")
    template = first(get_components(ACBus, sys))
    grid_bus = first(
        b for b in get_components(ACBus, sys) if get_bustype(b) != PSY.ACBusTypes.REF &&
        get_bustype(b) != PSY.ACBusTypes.ISOLATED
    )
    function _mk_bus(num, name)
        b = deepcopy(template)
        b.internal = IS.InfrastructureSystemsInternal()
        set_number!(b, num)
        set_name!(b, name)
        set_bustype!(b, PSY.ACBusTypes.PQ)
        return b
    end
    lo = _mk_bus(910, "ZI_LOW")
    hi = _mk_bus(920, "ZI_HIGH")
    foreach(b -> add_component!(sys, b), (lo, hi))
    function _mk_line(name, from, to, r, x)
        arc = Arc(from, to)
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(
                name,
                true,
                0.0,
                0.0,
                arc,
                r,
                x,
                (from = 0.0, to = 0.0),
                100.0,
                (-1.5, 1.5),
            ),
        )
    end
    # Zero-impedance arc HIGH(920) -> LOW(910): survivor = from = 920, removed = to = 910,
    # so the surviving column index exceeds the removed one (the i > j case).
    _mk_line("ZI_hi_lo", hi, lo, 0.0, 1e-5)
    # A normal line to the REMOVED bus; after the merge it remaps onto the survivor and
    # needs a brand-new structural entry in the survivor's column.
    _mk_line("grid_lo", grid_bus, lo, 0.01, 0.10)

    yb = Ybus(sys)
    @test PNM.get_mapped_bus_number(yb.network_reduction_data, 910) == 920
    bl = yb.lookup[1]
    g = bl[get_number(grid_bus)]
    s = bl[920]
    # The remapped mutual must be present and symmetric (the bug zeroed one side).
    @test yb.data[g, s] != 0
    @test yb.data[g, s] == yb.data[s, g]
    @test all(isfinite, BA_Matrix(yb).data.nzval)
    @test all(isfinite, ABA_Matrix(yb; factorize = false).data.nzval)
end

@testset "ZeroImpedanceBranchReduction: anti-parallel merge keeps a bus's degree" begin
    # Regression: ZIB L13 merges bus 3 into bus 1, making L23 anti-parallel to L12; their
    # signed-adjacency entries cancel and hide the 1-2 edge. Without the repair, D2 wrongly
    # folds bus 2 (looks degree-2) and IncidenceMatrix then KeyErrors on the orphaned arc.
    sys = System(100.0)
    buses = ACBus[]
    for i in 1:5
        b = ACBus(;
            number = i,
            name = "b$i",
            available = true,
            bustype = i == 1 ? ACBusTypes.REF : ACBusTypes.PV,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        )
        add_component!(sys, b)
        push!(buses, b)
    end
    function _mk_line(name, f, t, x)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = arc,
                r = 0.0,
                x = x,
                b = (from = 0.0, to = 0.0),
                rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
    end
    _mk_line("L12", 1, 2, 0.1)
    _mk_line("L23", 2, 3, 0.1)
    _mk_line("L13", 1, 3, 1e-5)   # zero-impedance: merges bus 3 into bus 1
    _mk_line("L24", 2, 4, 0.1)
    _mk_line("L25", 2, 5, 0.1)
    _mk_line("L14", 1, 4, 0.1)    # keeps bus 1 connected after bus 2's neighbors

    ybus = Ybus(
        sys;
        irreducible_buses = [1, 4, 5],
        network_reductions = NetworkReduction[DegreeTwoReduction()],
    )
    nr = ybus.network_reduction_data
    # Bus 2 stays degree-3 (1 via the parallel group, plus 4 and 5), so it is not folded.
    @test 2 ∉ nr.removed_buses
    @test 2 ∈ PNM.get_bus_axis(ybus)
    # Every arc endpoint must resolve, i.e. IncidenceMatrix builds without a KeyError.
    @test IncidenceMatrix(ybus) isa IncidenceMatrix
end

@testset "anti-parallel asymmetric group: orientation-correct equivalent Ybus" begin
    # Regression: ZIB merges bus 3 into bus 1, making asymmetric L2 (b.from != b.to)
    # anti-parallel to L1. The equivalent must swap L2's 2x2 into the key frame; the old
    # positional sum gave a wrong Y11. Oracle: the independently merged ybus.data.
    sys = System(100.0)
    buses = ACBus[]
    for i in 1:3
        b = ACBus(;
            number = i,
            name = "b$i",
            available = true,
            bustype = i == 1 ? ACBusTypes.REF : ACBusTypes.PV,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        )
        add_component!(sys, b)
        push!(buses, b)
    end
    function _mk_line(name, f, t, x, b_from)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = arc,
                r = 0.0,
                x = x,
                b = (from = b_from, to = 0.0),
                rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
    end
    _mk_line("L1", 1, 2, 0.1, 0.0)     # symmetric
    _mk_line("L2", 2, 3, 0.2, 0.3)     # asymmetric shunt; anti-parallel after the merge
    _mk_line("ZIB", 1, 3, 1e-5, 0.0)   # zero-impedance: merges bus 3 into bus 1

    ybus = Ybus(sys)
    nr = ybus.network_reduction_data
    @test length(nr.parallel_branch_map) == 1
    key, bp = first(nr.parallel_branch_map)
    bl = ybus.lookup[1]
    ip = bl[key[1]]
    iq = bl[key[2]]

    aware = PNM.ybus_branch_entries(bp, nr)
    # Oracle: ybus.data is built by accumulation (an independent code path).
    @test aware[1] ≈ ybus.data[ip, ip]
    @test aware[2] ≈ ybus.data[ip, iq]
    @test aware[3] ≈ ybus.data[iq, ip]
    @test aware[4] ≈ ybus.data[iq, iq]
    # The old orientation-blind positional sum mis-places the reversed member's self admittance.
    blind11 = zero(eltype(ybus.data))
    for br in bp.branches
        blind11 += PNM.ybus_branch_entries(br)[1]
    end
    @test !isapprox(blind11, ybus.data[ip, ip])
end

@testset "anti-parallel asymmetric group: phase-shifting transformer member" begin
    # Regression: a PhaseShiftingTransformer makes both diagonals (tap: Y11 != Y22) and
    # off-diagonals (phase shift: Y12 != Y21) asymmetric. When ZIB merges bus 3 into bus 1
    # the transformer goes anti-parallel to L1, so the equivalent must swap its full 2x2 into
    # the key frame. Unlike the shunt-only case above, here the off-diagonal swap matters too.
    sys = System(100.0)
    buses = ACBus[]
    for i in 1:3
        bustype = ACBusTypes.PV
        if i == 1
            bustype = ACBusTypes.REF
        end
        b = ACBus(;
            number = i,
            name = "b$i",
            available = true,
            bustype = bustype,
            angle = 0.0,
            magnitude = 1.0,
            voltage_limits = (min = 0.9, max = 1.1),
            base_voltage = 230.0,
        )
        add_component!(sys, b)
        push!(buses, b)
    end
    function _mk_line(name, f, t, x)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                arc = arc,
                r = 0.0,
                x = x,
                b = (from = 0.0, to = 0.0),
                rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
    end
    _mk_line("L1", 1, 2, 0.1)        # symmetric
    _mk_line("ZIB", 1, 3, 1e-5)      # zero-impedance: merges bus 3 into bus 1
    # Asymmetric member oriented (2, 3); becomes anti-parallel to L1 after the merge. tap != 1
    # breaks Y11 == Y22 and α != 0 breaks Y12 == Y21, so the whole block must be reoriented.
    arc = Arc(; from = buses[2], to = buses[3])
    add_component!(sys, arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = arc,
                tap = 1.05,
                α = 0.15,
                available = true,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                rating = 1.0,
                base_power = 100.0,
                base_voltage_primary = 230.0,
                r = 0.0,
                x = 0.2,
            ),
            magnetizing_shunt = Complex(0.0, 0.3),
        ),
    )

    ybus = Ybus(sys)
    nr = ybus.network_reduction_data
    @test length(nr.parallel_branch_map) == 1
    key, bp = first(nr.parallel_branch_map)
    bl = ybus.lookup[1]
    ip = bl[key[1]]
    iq = bl[key[2]]

    aware = PNM.ybus_branch_entries(bp, nr)
    # The phase shift makes the equivalent itself off-diagonal-asymmetric.
    @test !isapprox(aware[2], aware[3])
    # Oracle: ybus.data is built by accumulation (an independent code path).
    @test aware[1] ≈ ybus.data[ip, ip]
    @test aware[2] ≈ ybus.data[ip, iq]
    @test aware[3] ≈ ybus.data[iq, ip]
    @test aware[4] ≈ ybus.data[iq, iq]
    # The old orientation-blind positional sum mis-places both the self admittance (tap) and
    # the off-diagonal (phase shift) of the reversed member.
    blind11 = zero(eltype(ybus.data))
    blind12 = zero(eltype(ybus.data))
    for br in bp.branches
        entries = PNM.ybus_branch_entries(br)
        blind11 += entries[1]
        blind12 += entries[2]
    end
    @test !isapprox(blind11, ybus.data[ip, ip])
    @test !isapprox(blind12, ybus.data[ip, iq])
end
