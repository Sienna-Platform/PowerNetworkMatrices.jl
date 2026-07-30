# Exact equivalents for parallel groups containing phase-shifting circuits (issue #231).

@testset "issue 231: equivalent recovery is Float64-accurate" begin
    sys = _mk_line_pst_parallel_system()
    nr = Ybus(sys).network_reduction_data
    eb = PNM.arc_equivalent_branch(nr, (1, 2))
    # The recovery is imag(log(Y21/Y12))/2, a nonlinear function of the two summed phasors
    # (b1=10 at angle 0, b2=5 at angle 0.15) -- NOT the susceptance-weighted average of member
    # angles (5*0.15/15 = 0.05). A BigFloat evaluation of the same closed-form phasor sum gives
    # the true value below to 16 digits; the old ComplexF32 round-trip recovered 0.0499582514,
    # off from it by ~7e-8 relative. Float64 recovery must land within 1e-9 of the true value.
    @test PNM.get_equivalent_shift(eb) ≈ 0.04995825507139971 atol = 1e-9
end

@testset "issue 231: members partition by impedance angle" begin
    # Lossless line ∥ lossless PST: both purely reactive -> one bucket, even though
    # their alphas differ (0.0 vs 0.15).
    sys = _mk_line_pst_parallel_system()
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    @test length(PNM._partition_members_by_impedance_angle(bp)) == 1
    @test PNM._is_single_pi_representable(PNM.ybus_branch_entries(bp, nr)[2:3]...)

    # Lossy PST ∥ lossless line: different impedance angles -> two buckets, and the
    # whole-group two-port is NOT single-pi representable.
    sys_lossy = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr_lossy = Ybus(sys_lossy).network_reduction_data
    bp_lossy = PNM.get_parallel_branch_map(nr_lossy)[(1, 2)]
    parts = PNM._partition_members_by_impedance_angle(bp_lossy)
    @test length(parts) == 2
    @test all(p -> length(p) == 1, parts)
    @test !PNM._is_single_pi_representable(
        PNM.ybus_branch_entries(bp_lossy, nr_lossy)[2:3]...,
    )

    # Each bucket, taken alone, IS representable -- this is the property that makes the
    # partition exact.
    reference = PNM.get_arc_tuple(bp_lossy, nr_lossy)
    for p in parts
        (_, y12, y21, _) = PNM._subset_two_port(p, reference, nr_lossy)
        @test PNM._is_single_pi_representable(y12, y21)
    end
end

@testset "issue 231: subset two-port sums to the group two-port" begin
    sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    reference = PNM.get_arc_tuple(bp, nr)
    total = zeros(ComplexF64, 4)
    for p in PNM._partition_members_by_impedance_angle(bp)
        total .+= collect(PNM._subset_two_port(p, reference, nr))
    end
    @test all(
        isapprox.(total, collect(PNM.ybus_branch_entries(bp, nr)); rtol = 1e-12),
    )
end

@testset "issue 231: exact partition equivalents for a lossy shifted group" begin
    sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]

    @test !PNM.has_single_pi_equivalent(bp, nr)
    partitions = PNM.equivalent_partitions(bp, nr)
    @test length(partitions) == 2
    # Every member appears exactly once across the partitions.
    all_members = reduce(vcat, PNM.get_members.(partitions))
    @test length(all_members) == length(bp)
    @test all(br -> any(m -> m === br, all_members), collect(bp))

    # THE exactness property: the partitions' pi-models sum back to the group's two-port.
    total = zeros(ComplexF64, 4)
    for pe in partitions
        adm = PNM._to_admittance(PNM.get_equivalent(pe))
        total .+= collect(PNM._pi_to_ybus(adm))
    end
    @test all(isapprox.(total, collect(PNM.ybus_branch_entries(bp, nr)); rtol = 1e-9))
end

@testset "issue 231: representable groups yield exactly one partition" begin
    sys = _mk_line_pst_parallel_system()
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]

    @test PNM.has_single_pi_equivalent(bp, nr)
    partitions = PNM.equivalent_partitions(bp, nr)
    @test length(partitions) == 1
    # And it agrees with the singular accessor bit-for-bit.
    single = PNM.get_equivalent_physical_branch_parameters(bp, nr)
    only_eq = PNM.get_equivalent(only(partitions))
    @test PNM.get_equivalent_r(only_eq) == PNM.get_equivalent_r(single)
    @test PNM.get_equivalent_x(only_eq) == PNM.get_equivalent_x(single)
    @test PNM.get_equivalent_shift(only_eq) == PNM.get_equivalent_shift(single)
end

@testset "issue 231: partition ratings sum per partition" begin
    sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    for pe in PNM.equivalent_partitions(bp, nr)
        expected = sum(PNM.get_equivalent_rating(m) for m in PNM.get_members(pe))
        @test PNM.get_partition_rating(pe) ≈ expected
    end
end

@testset "issue 231: arc_equivalent_branches is total" begin
    sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr = Ybus(sys).network_reduction_data

    # The lossy shifted group: singular accessor throws and steers to the plural one, which
    # returns both branches.
    @test_throws ErrorException PNM.arc_equivalent_branch(nr, (1, 2))
    @test_throws "arc_equivalent_branches" PNM.arc_equivalent_branch(nr, (1, 2))

    ebs = PNM.arc_equivalent_branches(nr, (1, 2))
    @test length(ebs) == 2
    @test all(eb -> isfinite(PNM.get_equivalent_x(eb)), ebs)

    # A plain direct line: exactly one, equal to its own equivalent.
    direct = PNM.arc_equivalent_branches(nr, (2, 3))
    @test length(direct) == 1
    @test PNM.get_equivalent_x(only(direct)) ==
          PNM.get_equivalent_x(PNM.equivalent_branch(PNM.get_direct_branch_map(nr)[(2, 3)]))

    # Reversed orientation negates the shift on every returned branch.
    fwd = PNM.arc_equivalent_branches(nr, (1, 2))
    rev = PNM.arc_equivalent_branches(nr, (2, 1))
    @test length(rev) == length(fwd)
    @test sum(PNM.get_equivalent_shift.(rev)) ≈ -sum(PNM.get_equivalent_shift.(fwd))

    @test_throws ErrorException PNM.arc_equivalent_branches(nr, (7, 9))
end

# `_mk_line_pst_parallel_system(; pst_r = 0.05)` is a 3-bus radial path whose bus 2 has no
# injector and degree two, so DegreeTwoReduction folds it away, leaving a series chain on
# composite arc (1, 3) whose first segment is the non-representable parallel group. Buses 1 and
# 3 are natural degree-one endpoints, so no direct branch is also keyed (1, 3) -- which matters,
# because a direct branch on that key would resolve first and mask the error under test.
@testset "issue 231: series chain containing a non-representable group errors precisely" begin
    sys = _mk_line_pst_parallel_system(; pst_r = 0.05)
    nr =
        Ybus(
            sys;
            network_reductions = NetworkReduction[DegreeTwoReduction()],
        ).network_reduction_data
    arc, bs = first(PNM.get_series_branch_map(nr))

    @test !PNM.has_single_pi_equivalent(bs, nr)
    @test_throws ErrorException PNM.arc_equivalent_branches(nr, arc)
    # The message must name the chain, name the offending segment, and say what to do.
    @test_throws PNM.get_name(bs) PNM.arc_equivalent_branches(nr, arc)
    @test_throws "Series chain" PNM.arc_equivalent_branches(nr, arc)
end

# Task 5: coverage sweep across transformer varieties. Every case asserts the exactness
# property -- the partitions' pi-models, run through `_to_admittance`/`_pi_to_ybus` and
# summed, must reproduce `ybus_branch_entries(bp, nr)` -- in addition to the shape-specific
# assertions the plan calls for.
function _partition_exactness_residual(bp, nr)
    total = zeros(ComplexF64, 4)
    for pe in PNM.equivalent_partitions(bp, nr)
        adm = PNM._to_admittance(PNM.get_equivalent(pe))
        total .+= collect(PNM._pi_to_ybus(adm))
    end
    reference = collect(PNM.ybus_branch_entries(bp, nr))
    return maximum(abs.(total .- reference) ./ abs.(reference))
end

# Two `TwoWindingTransformer`s sharing one arc.
function _mk_two_transformer_parallel_system(; alpha1, alpha2, r1, x1, r2, x2)
    sys, buses = _mk_bus_system(2)
    arc = Arc(; from = buses[1], to = buses[2])
    add_component!(sys, arc)
    function _mk_xfmr(name, α, r, x)
        add_component!(
            sys,
            PSY.TwoWindingTransformer(;
                name = name,
                circuit = PSY.TransformerCircuit(;
                    arc = arc, tap = 1.0, α = α, available = true,
                    active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                    base_power = 100.0, base_voltage_primary = 230.0, r = r, x = x,
                ),
                magnetizing_shunt = Complex(0.0, 0.0),
            ),
        )
        return nothing
    end
    _mk_xfmr("T1", alpha1, r1, x1)
    _mk_xfmr("T2", alpha2, r2, x2)
    return sys
end

@testset "issue 231: uniform α with mixed R/X collapses to one partition despite two impedance-angle buckets" begin
    # T1: r=0, x=0.1 (impedance angle 90°); T2: r=0.05, x=0.2 (impedance angle atan(4)≈76°).
    # Different impedance angles would partition-first into 2 groups; representability-first
    # must still find the whole group single-π because α is uniform (0.15 for both).
    sys = _mk_two_transformer_parallel_system(;
        alpha1 = 0.15, alpha2 = 0.15, r1 = 0.0, x1 = 0.1, r2 = 0.05, x2 = 0.2,
    )
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]

    @test length(PNM._partition_members_by_impedance_angle(bp)) == 2
    @test PNM.has_single_pi_equivalent(bp, nr)
    @test length(PNM.equivalent_partitions(bp, nr)) == 1
    @test _partition_exactness_residual(bp, nr) < 1e-9
end

@testset "issue 231: shared impedance angle with different α collapses to one partition" begin
    # T1: r=0.05, x=0.10; T2: r=0.10, x=0.20 -- identical r/x ratio (0.5), so one
    # impedance-angle bucket even though the shift angles differ (0.1 vs 0.3).
    sys = _mk_two_transformer_parallel_system(;
        alpha1 = 0.1, alpha2 = 0.3, r1 = 0.05, x1 = 0.10, r2 = 0.10, x2 = 0.20,
    )
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]

    @test length(PNM._partition_members_by_impedance_angle(bp)) == 1
    @test length(PNM.equivalent_partitions(bp, nr)) == 1
    @test _partition_exactness_residual(bp, nr) < 1e-9
end

# Two lossless lines and a lossy PST sharing one arc.
function _mk_two_lines_one_pst_parallel_system(;
    pst_alpha = 0.15, pst_r = 0.05, pst_x = 0.2, x_line1 = 0.1, x_line2 = 0.25,
)
    sys, buses = _mk_bus_system(2)
    arc = Arc(; from = buses[1], to = buses[2])
    add_component!(sys, arc)
    function _mk_pll_line(name, x)
        add_component!(
            sys,
            Line(;
                name = name, available = true, active_power_flow = 0.0,
                reactive_power_flow = 0.0, arc = arc, r = 0.0, x = x,
                b = (from = 0.0, to = 0.0), rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
        return nothing
    end
    _mk_pll_line("L1", x_line1)
    _mk_pll_line("L2", x_line2)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = arc, tap = 1.0, α = pst_alpha, available = true,
                active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                base_power = 100.0, base_voltage_primary = 230.0, r = pst_r, x = pst_x,
            ),
            magnetizing_shunt = Complex(0.0, 0.0),
        ),
    )
    return sys
end

@testset "issue 231: three members partition into two impedance-angle groups" begin
    sys = _mk_two_lines_one_pst_parallel_system()
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 2)]
    l1 = PSY.get_component(PSY.Line, sys, "L1")
    l2 = PSY.get_component(PSY.Line, sys, "L2")

    @test length(bp) == 3
    partitions = PNM.equivalent_partitions(bp, nr)
    @test length(partitions) == 2
    two_member = only(filter(p -> length(PNM.get_members(p)) == 2, partitions))
    @test any(m -> m === l1, PNM.get_members(two_member))
    @test any(m -> m === l2, PNM.get_members(two_member))
    @test _partition_exactness_residual(bp, nr) < 1e-9
end

# 3-bus anti-parallel fixture from `test_ybus_reductions.jl:725`, parameterized by pst_r.
function _mk_antiparallel_pst_system(; pst_r)
    sys, buses = _mk_bus_system(3)
    function _mk_line(name, f, t, x)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name, available = true, active_power_flow = 0.0,
                reactive_power_flow = 0.0, arc = arc, r = 0.0, x = x,
                b = (from = 0.0, to = 0.0), rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
        return nothing
    end
    _mk_line("L1", 1, 2, 0.1)        # symmetric
    _mk_line("ZIB", 1, 3, 1e-5)      # zero-impedance: merges bus 3 into bus 1
    # Oriented (2, 3); becomes anti-parallel to L1 after the merge.
    arc = Arc(; from = buses[2], to = buses[3])
    add_component!(sys, arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = arc, tap = 1.05, α = 0.15, available = true,
                active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                base_power = 100.0, base_voltage_primary = 230.0, r = pst_r, x = 0.2,
            ),
            magnetizing_shunt = Complex(0.0, 0.3),
        ),
    )
    return sys
end

@testset "issue 231: anti-parallel member exercises per-partition orientation swap" begin
    sys = _mk_antiparallel_pst_system(; pst_r = 0.05)
    nr = Ybus(sys).network_reduction_data
    _, bp = first(PNM.get_parallel_branch_map(nr))

    @test !PNM.has_single_pi_equivalent(bp, nr)
    @test length(PNM.equivalent_partitions(bp, nr)) == 2
    @test _partition_exactness_residual(bp, nr) < 1e-9
end

# One winding of a `ThreeWindingTransformer` sharing an arc with a lossy `Line`. The winding
# leg has r=0.005, x=0.05 (impedance angle atan(10)≈84.3°); `line_r`/`line_x` default to a
# different impedance angle (atan(3)≈71.6°) so the group is genuinely non-representable, and
# the winding carries α=0.15 -- both are needed: a phase shift alone (with all impedance
# angles equal) or an impedance-angle split alone (with no phase shift anywhere) is still
# single-π, per the earlier testsets in this file.
function _mk_3w_winding_line_parallel_system(; line_r = 0.05, line_x = 0.15)
    sys, buses = _mk_bus_system(3)
    star = ACBus(;
        number = 4, name = "star", available = true, bustype = ACBusTypes.PQ,
        angle = 0.0, magnitude = 1.0, voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
    add_component!(sys, star)
    arc1 = Arc(; from = buses[1], to = star)
    arc2 = Arc(; from = buses[2], to = star)
    arc3 = Arc(; from = buses[3], to = star)
    foreach(a -> add_component!(sys, a), (arc1, arc2, arc3))
    z12, z23, z31 = complex(0.01, 0.1), complex(0.01, 0.1), complex(0.01, 0.1)
    legs = ((z12 + z31 - z23) / 2, (z12 + z23 - z31) / 2, (z31 + z23 - z12) / 2)
    arcs = (arc1, arc2, arc3)
    # Only the primary winding shifts: a group with no phase shift anywhere is trivially
    # single-π, so the non-representable case needs α on the winding that parallels the line.
    alphas = (0.15, 0.0, 0.0)
    circuits = ntuple(
        i -> PSY.TransformerCircuit(;
            arc = arcs[i], available = true, base_power = 100.0,
            base_voltage_primary = PSY.get_base_voltage(PSY.get_from(arcs[i])),
            r = real(legs[i]), x = imag(legs[i]), rating = 1.0,
            α = alphas[i],
        ),
        3,
    )
    t3w = PSY.ThreeWindingTransformer(;
        name = "T3W", primary_circuit = circuits[1], secondary_circuit = circuits[2],
        tertiary_circuit = circuits[3], star_bus = star,
        r_12 = 0.01, x_12 = 0.1, r_23 = 0.01, x_23 = 0.1, r_31 = 0.01, x_31 = 0.1,
        base_power_12 = 100.0, base_power_23 = 100.0, base_power_31 = 100.0,
        magnetizing_shunt = Complex(0.0, 0.0),
        shunt_location = PSY.ThreeWindingTransformerShuntLocation.PRIMARY,
    )
    add_component!(sys, t3w)
    add_component!(
        sys,
        Line(;
            name = "Lstar", available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = arc1, r = line_r, x = line_x,
            b = (from = 0.0, to = 0.0), rating = 1.0,
            angle_limits = (min = -1.5, max = 1.5),
        ),
    )
    return sys
end

@testset "issue 231: three-winding transformer winding sharing an arc with a lossy line" begin
    sys = _mk_3w_winding_line_parallel_system()
    nr = Ybus(sys).network_reduction_data
    bp = PNM.get_parallel_branch_map(nr)[(1, 4)]
    @test length(bp) == 2

    @test !PNM.has_single_pi_equivalent(bp, nr)
    partitions = PNM.equivalent_partitions(bp, nr)
    @test length(partitions) == 2
    @test all(
        pe ->
            isfinite(PNM.get_equivalent_r(PNM.get_equivalent(pe))) &&
                isfinite(PNM.get_equivalent_x(PNM.get_equivalent(pe))),
        partitions,
    )
    @test _partition_exactness_residual(bp, nr) < 1e-9
end
