
@testset "Test LODF with radial branches" begin
    # check if the flows obtained with original LODF and reduced one are the same
    for name in ["c_sys14", "test_RTS_GMLC_sys"]
        # load the system
        sys = PSB.build_system(PSB.PSITestSystems, name)

        # get A, BA and ABA matrix with radial branches
        A_rad =
            IncidenceMatrix(sys; network_reductions = NetworkReduction[RadialReduction()])
        BA_rad = BA_Matrix(sys; network_reductions = NetworkReduction[RadialReduction()])
        ABA_rad = ABA_Matrix(
            sys;
            network_reductions = NetworkReduction[RadialReduction()],
            factorize = true,
        )
        ptdf = PTDF(sys)
        ptdf_rad = PTDF(sys; network_reductions = NetworkReduction[RadialReduction()])

        # get the original and reduced PTDF matrices (consider different methods)
        lodf = LODF(sys)
        lodf_rad = LODF(sys; network_reductions = NetworkReduction[RadialReduction()])
        lodf_rad_A_BA_ABA = LODF(A_rad, ABA_rad, BA_rad)
        lodf_rad_A_PTDF = LODF(A_rad, ptdf_rad)

        rb = A_rad.network_reduction_data

        # at first check if all the matrices are the same
        @test isapprox(lodf_rad.data, lodf_rad_A_BA_ABA.data, atol = 1e-10)
        @test isapprox(lodf_rad_A_PTDF.data, lodf_rad_A_BA_ABA.data, atol = 1e-10)

        # now check if the flows are the same

        # first evaluate the flows on the original and reduced system
        bus_numbers = []
        for i in keys(rb.bus_reduction_map)
            append!(bus_numbers, collect(rb.bus_reduction_map[i]))
        end
        bus_idx = setdiff(
            1:size(ptdf.data, 1),
            append!([ptdf.lookup[1][i] for i in bus_numbers]),
        )
        br_idx =
            setdiff(1:size(ptdf.data, 2), [ptdf.lookup[2][i] for i in rb.removed_arcs])
        # now get the injections from the system
        n_buses = length(axes(ptdf, 1))
        bus_lookup = ptdf.lookup[1]
        branch_flow_values = zeros(Float64, length(axes(ptdf, 2)))
        bus_activepower_injection = zeros(Float64, n_buses)
        sources =
            PSY.get_components(
                d -> !isa(d, Union{PSY.ElectricLoad, PSY.SynchronousCondenser}),
                PSY.StaticInjection,
                sys,
            )
        for source in sources
            !PSY.get_available(source) && continue
            bus = PSY.get_bus(source)
            bus_ix = bus_lookup[PSY.get_number(bus)]
            bus_activepower_injection[bus_ix] += PSY.get_active_power(source, Float64)
        end
        bus_activepower_withdrawals = zeros(Float64, n_buses)
        loads = PSY.get_components(x -> !isa(x, PSY.FixedAdmittance), PSY.ElectricLoad, sys)
        for l in loads
            !PSY.get_available(l) && continue
            bus = PSY.get_bus(l)
            bus_ix = bus_lookup[PSY.get_number(bus)]
            bus_activepower_withdrawals[bus_ix] += PSY.get_active_power(l, Float64)
        end
        power_injection =
            deepcopy(bus_activepower_injection - bus_activepower_withdrawals)
        # get the flows with the PTDF matrix
        ref_flow_values = transpose(ptdf.data) * power_injection
        # evaluate according to the matrix with no radial branches
        reduce_flow_values = zeros((length(br_idx),))
        # change power injection for aggregated leaf buses
        power_injection2 = deepcopy(power_injection)
        for i in keys(rb.bus_reduction_map)
            for j in rb.bus_reduction_map[i]
                power_injection2[ptdf.lookup[1][i]] += power_injection[ptdf.lookup[1][j]]
            end
        end
        reduced_flow_values = transpose(ptdf_rad.data) * power_injection2[bus_idx]

        # now evaluate the flows with the reduced LODF matrix
        ref_lodf_flows = transpose(lodf.data) * ref_flow_values
        ref_lodf_flows_reduced1 = transpose(lodf_rad.data) * reduced_flow_values
        ref_lodf_flows_reduced2 = transpose(lodf_rad.data) * ref_flow_values[br_idx]

        @test isapprox(ref_lodf_flows_reduced1, ref_lodf_flows_reduced2, atol = 1e-10)
        @test isapprox(ref_lodf_flows[br_idx], ref_lodf_flows_reduced1, atol = 1e-10)
    end
end

@testset "Test LODF errors" begin
    # load the system
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    # get A, BA and ABA matrix with radial branches
    A = IncidenceMatrix(sys)
    BA_rad = BA_Matrix(sys; network_reductions = NetworkReduction[RadialReduction()])
    ABA = ABA_Matrix(sys; factorize = true)
    ptdf = PTDF(sys)
    ptdf_rad = PTDF(sys; network_reductions = NetworkReduction[RadialReduction()])
    # test LODF from A, ABA and BA
    test_value = false
    try
        lodf_rad_A_BA_ABA = LODF(A, ABA, BA_rad)
    catch err
        if err isa Exception
            test_value = true
        end
    end
    @test test_value

    # test LODF from A, PTDF
    test_value = false
    try
        lodf_rad_A_PTDF = LODF(A_rad, ptdf)
    catch err
        if err isa Exception
            test_value = true
        end
    end
    @test test_value

    test_value = false
    try
        lodf_rad_A_PTDF = LODF(A_rad, ptdf_rad)
    catch err
        if err isa Exception
            test_value = true
        end
    end
    @test test_value
end
