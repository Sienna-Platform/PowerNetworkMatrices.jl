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
    ybus = Ybus(sys; network_reductions = NetworkReduction[RadialReduction([116])])
    @test !haskey(ybus.network_reduction_data.reverse_bus_search_map, 116)
    @test ybus.network_reduction_data.irreducible_buses == Set{Int}(116)

    # Test irreducible bus input for degree two reduction
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    @test 117 ∈ ybus.network_reduction_data.removed_buses
    ybus = Ybus(
        sys;
        network_reductions = NetworkReduction[DegreeTwoReduction(;
            irreducible_buses = [117],
        )],
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
