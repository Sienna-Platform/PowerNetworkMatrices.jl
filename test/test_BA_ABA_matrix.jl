@testset "Test A, BA, and ABA matrix creation" begin
    # test on 5 and 14 bus system, and RTS.
    for (category, name) in [
        (PSITestSystems, "c_sys5"),
        (PSITestSystems, "c_sys14"),
        (PSISystems, "RTS_GMLC_DA_sys"),
    ]
        sys = PSB.build_system(category, name)
        # at first let's see if factorization flag works
        ABA_no_lu = ABA_Matrix(sys)
        @test isnothing(ABA_no_lu.K)
        # check the is_factorized function
        @test is_factorized(ABA_no_lu) == false
        # factorize the ABA matrix
        ABA_no_lu = factorize(ABA_no_lu)
        @test is_factorized(ABA_no_lu) == true
        # get the ABA matrix with the current method
        ABA_lu = ABA_Matrix(sys; factorize = true)
        # check the is_factorized function
        @test is_factorized(ABA_no_lu) == true
        # evaluate if the ABA matrix is correct
        A = IncidenceMatrix(sys)
        BA = BA_Matrix(sys)
        ABA_2 = transpose(A.data) * BA.data'
        @test isapprox(
            ABA_lu.data,
            ABA_2[
                setdiff(1:end, PNM.get_ref_bus_position(A)),
                setdiff(1:end, PNM.get_ref_bus_position(A)),
            ],
            atol = 1e-8,
        )

        # evaluate if the LU factorization evaluates a correct PTDF matrix
        ptdf_1 = PTDF(sys)
        Ix = Matrix(1.0I, size(ABA_lu.data, 1), size(ABA_lu.data, 1))
        ABA_inv = ABA_lu.K \ Ix
        ptdf_2 = BA.data[setdiff(1:end, PNM.get_ref_bus_position(A)), :]' * ABA_inv
        @test isapprox(
            ptdf_1.data[setdiff(1:end, PNM.get_ref_bus_position(A)), :],
            ptdf_2',
            atol = 1e-8,
        )
    end
end

@testset "Test BA and ABA matrix indexing" begin
    sys = PSB.build_system(PSB.PSISystems, "RTS_GMLC_DA_sys")

    # get the matrices
    ba = BA_Matrix(sys)
    aba = ABA_Matrix(sys)

    # check if indexing for the BA is correct (row and column indices)
    # ba matrix is stored as transposed
    for i in 1:size(ba, 2)
        for j in 1:size(ba, 1)
            @test isapprox(ba[i, j], ba.data[j, i])
        end
    end
    # check if indexing for the BA is correct (line names and bus numbers)
    lookup1 = PNM.get_lookup(ba)
    for i in axes(ba, 2)
        for j in axes(ba, 1)
            @test isapprox(ba[i, j], ba.data[lookup1[1][j], lookup1[2][i]])
        end
    end

    # check indexing for the ABA matrix
    lookup2 = aba.lookup[1]
    for i in axes(aba, 1)
        @test aba[i, :] == aba.data[lookup2[i], :]
    end

    # test if error is correctly thrown when ref bus is called
    buses = get_components(PSY.ACBus, sys)
    rb = [bus for bus in buses if get_bustype(bus) == PSY.ACBusTypes.REF][1]
    @test_throws ErrorException aba[rb, :]
end

# 3-bus system: ref bus 1 tied to bus 2 by a normal line, plus parallel members on arc (2, 3)
# carrying the supplied (r, x) pairs (used to force a degenerate net series admittance).
function _mk_parallel_cancel_sys(member_rx::Vector{Tuple{Float64, Float64}})
    sys, buses = _mk_bus_system(3)
    arc12 = Arc(; from = buses[1], to = buses[2])
    add_component!(sys, arc12)
    _add_test_line!(sys, "L12", arc12, 0.01, 0.1)  # keeps the ref bus connected
    # Parallel members share one Arc (2, 3), as real parallel branches do.
    arc23 = Arc(; from = buses[2], to = buses[3])
    add_component!(sys, arc23)
    for (k, (r, x)) in enumerate(member_rx)
        _add_test_line!(sys, "L23_$k", arc23, r, x)
    end
    return sys
end

@testset "BA/ABA: degenerate net series admittance stays finite" begin
    # b = 1 / imag(1 / Yt) blows up for two degenerate parallel combinations; both must give
    # finite b = 0. Case 1: admittances cancel (Yt = 0 -> NaN). Case 2: reactances cancel but
    # resistance does not (Yt purely real -> imag(1 / Yt) = 0 -> Inf).
    for member_rx in (
        [(0.0, 0.1), (0.0, -0.1)],     # case 1: Yt = 0
        [(0.01, 0.1), (0.01, -0.1)],   # case 2: zero net reactance, nonzero conductance
    )
        ybus = Ybus(_mk_parallel_cancel_sys(member_rx))
        @test all(isfinite, BA_Matrix(ybus).data.nzval)
        @test all(isfinite, ABA_Matrix(ybus; factorize = false).data.nzval)
    end
end

@testset "BA: phase-shifting transformer susceptance ignores the phase angle" begin
    # A phase shifter's off-diagonal Ybus is asymmetric, so a single entry folds α into b. BA
    # must match the phase-independent PNM.get_series_susceptance = 1/(a x): b independent of α.
    function _mk_pst_sys(α)
        sys, buses = _mk_bus_system(3)
        arc12 = Arc(; from = buses[1], to = buses[2])
        add_component!(sys, arc12)
        _add_test_line!(sys, "L12", arc12, 0.0, 0.1)
        arc23 = Arc(; from = buses[2], to = buses[3])
        add_component!(sys, arc23)
        pst = PSY.TwoWindingTransformer(;
            name = "PST23",
            circuit = PSY.TransformerCircuit(;
                available = true,
                arc = arc23,
                tap = 1.05,
                α = α,
                r = 0.01,
                x = 0.2,
                active_power_flow = 0.0,
                reactive_power_flow = 0.0,
                rating = 1.0,
                base_power = 100.0,
                base_voltage_primary = 1.0,
            ),
            magnetizing_shunt = 0.0,
        )
        add_component!(sys, pst)
        return sys, pst
    end

    # b for the PST arc (2, 3): BA.data is bus x arc, so the (from-bus, arc) entry holds +b.
    function _pst_susceptance(sys)
        ba = BA_Matrix(Ybus(sys))
        bus_lookup, arc_lookup = PNM.get_lookup(ba)
        return ba.data[bus_lookup[2], arc_lookup[(2, 3)]]
    end

    _, pst = _mk_pst_sys(0.0)
    target = PNM.get_series_susceptance(pst, PSY.SU)  # 1 / (tap * x) = 1 / (1.05 * 0.2)
    for α in (0.0, 0.3, -0.5)
        sys, _ = _mk_pst_sys(α)
        b = _pst_susceptance(sys)
        @test isfinite(b)
        @test isapprox(abs(b), target; rtol = sqrt(eps(real(YBUS_ELTYPE))))
    end
end

@testset "Test show for A, BA and ABA matrix" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    a = IncidenceMatrix(sys)
    ba = BA_Matrix(sys)
    aba = ABA_Matrix(sys)

    for mat in [a, ba, aba]
        test_value = false
        try
            show(@eval $mat)
            test_value = true
        catch err
            @error err
            test_value = false
        end
        @test test_value
    end
end
