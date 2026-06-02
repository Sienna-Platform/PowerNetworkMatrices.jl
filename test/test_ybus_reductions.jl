@testset "Invalid reduction combinations" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    @test_throws IS.DataFormatError("Radial reduction is applied twice to the same system") Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction(), RadialReduction()],
    )
    @test_throws IS.DataFormatError(
        "Ward reduction must be the last applied reduction",
    ) Ybus(
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
    @test length(keys(nrd.direct_branch_map)) == 14
    @test length(keys(nrd.parallel_branch_map)) == 3
    @test length(keys(nrd.series_branch_map)) == 0
    @test length(keys(nrd.transformer3W_map)) == 6
    @test nrd.removed_buses == Set{Int}()
    @test nrd.removed_arcs == Set([(112, 113), (104, 105)])
    @test Set(keys(nrd.added_admittance_map)) == Set{Int}()
    @test Set(keys(nrd.added_arc_impedance_map)) == Set{Tuple{Int, Int}}()
    @test Set(A.axes[1]) == union(
        Set(keys(nrd.direct_branch_map)),
        Set(keys(nrd.parallel_branch_map)),
        Set(keys(nrd.series_branch_map)),
        Set(keys(nrd.transformer3W_map)),
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
    @test length(keys(nrd.direct_branch_map)) == 12
    @test length(keys(nrd.parallel_branch_map)) == 3
    @test length(keys(nrd.series_branch_map)) == 0
    @test length(keys(nrd.transformer3W_map)) == 5
    @test nrd.removed_buses == Set{Int}()
    @test nrd.removed_arcs ==
          Set([(107, 108), (107, 1001), (103, 116), (112, 113), (104, 105)])
    @test Set(keys(nrd.added_admittance_map)) == Set{Int}()
    @test Set(keys(nrd.added_arc_impedance_map)) == Set{Tuple{Int, Int}}()
    @test Set(A.axes[1]) == union(
        Set(keys(nrd.direct_branch_map)),
        Set(keys(nrd.parallel_branch_map)),
        Set(keys(nrd.series_branch_map)),
        Set(keys(nrd.transformer3W_map)),
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
    @test length(keys(nrd.direct_branch_map)) == 9
    @test length(keys(nrd.parallel_branch_map)) == 2
    @test length(keys(nrd.series_branch_map)) == 3
    @test length(keys(nrd.transformer3W_map)) == 5
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
        Set(keys(nrd.transformer3W_map)),
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
        Set(keys(nrd.transformer3W_map)),
        Set(keys(nrd.added_arc_impedance_map)),
    )
    @test Set(A.axes[2]) == Set(keys(nrd.bus_reduction_map))
end

@testset "Test irreducible_buses" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    # Test irreducible bus input for radial reduction
    ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()])
    @test haskey(ybus.network_reduction_data.reverse_bus_search_map, 116)
    ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction()],
        irreducible_buses = Set([116]))
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
            primary_star_arc = get_primary_star_arc(l)
            if (
                (primary_star_arc.from.number, primary_star_arc.to.number) ∈
                radial_removed_arcs
            ) ||
               (
                (primary_star_arc.to.number, primary_star_arc.from.number) ∈
                radial_removed_arcs
            )
                set_available_primary!(l, false)
                if primary_star_arc.from.number ∈ keys(rbsm)
                    set_available!(primary_star_arc.from, false)
                end
                if primary_star_arc.to.number ∈ keys(rbsm)
                    set_available!(primary_star_arc.to, false)
                end
            end
            secondary_star_arc = get_secondary_star_arc(l)
            if (
                (secondary_star_arc.from.number, secondary_star_arc.to.number) ∈
                radial_removed_arcs
            ) ||
               (
                (secondary_star_arc.to.number, secondary_star_arc.from.number) ∈
                radial_removed_arcs
            )
                set_available_secondary!(l, false)
                if secondary_star_arc.from.number ∈ keys(rbsm)
                    set_available!(secondary_star_arc.from, false)
                end
                if secondary_star_arc.to.number ∈ keys(rbsm)
                    set_available!(secondary_star_arc.to, false)
                end
            end
            tertiary_star_arc = get_tertiary_star_arc(l)
            if (
                (tertiary_star_arc.from.number, tertiary_star_arc.to.number) ∈
                radial_removed_arcs
            ) ||
               (
                (tertiary_star_arc.to.number, tertiary_star_arc.from.number) ∈
                radial_removed_arcs
            )
                set_available_tertiary!(l, false)
                if tertiary_star_arc.from.number ∈ keys(rbsm)
                    set_available!(tertiary_star_arc.from, false)
                end
                if tertiary_star_arc.to.number ∈ keys(rbsm)
                    set_available!(tertiary_star_arc.to, false)
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
    ybus_1 = Ybus(
        sys;
        network_reductions = NetworkReduction[RadialReduction()],
    )
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
    t = get_component(Transformer2W, sys, "Trans4")  # from=7, to=8
    set_r!(t, 0.0 * PSY.SU)
    set_x!(t, 1e-5 * PSY.SU)

    ybus = Ybus(sys)
    nrd = ybus.network_reduction_data

    @test !haskey(nrd.reverse_bus_search_map, 7)
    @test !haskey(nrd.reverse_bus_search_map, 8)

    @test 7 ∈ PNM.get_bus_axis(ybus)
    @test 8 ∈ PNM.get_bus_axis(ybus)
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
        b for b in get_components(ACBus, sys) if
        get_bustype(b) != PSY.ACBusTypes.REF && get_bustype(b) != PSY.ACBusTypes.ISOLATED
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
        b for b in get_components(ACBus, sys) if
        get_bustype(b) != PSY.ACBusTypes.REF && get_bustype(b) != PSY.ACBusTypes.ISOLATED
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
