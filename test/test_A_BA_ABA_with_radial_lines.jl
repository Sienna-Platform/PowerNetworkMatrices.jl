@testset "Test BA matrix with radial lines" begin
    for name in ["c_sys14", "test_RTS_GMLC_sys"]
        sys = PSB.build_system(PSB.PSITestSystems, name)
        BA = BA_Matrix(sys)
        BA_rad = BA_Matrix(sys; network_reductions = NetworkReduction[RadialReduction()])
        nr = BA_rad.network_reduction_data
        bus_numbers = []
        for i in keys(nr.bus_reduction_map)
            append!(bus_numbers, collect(nr.bus_reduction_map[i]))
        end
        bus_idx = setdiff(1:size(BA.data, 1), [BA.lookup[1][i] for i in bus_numbers])
        br_idx = setdiff(1:size(BA.data, 2), [BA.lookup[2][i] for i in nr.removed_arcs])
        # now extract A matrix manually and compare
        @test all(isapprox.(BA.data[bus_idx, br_idx], BA_rad.data))
    end
end

@testset "Test ABA matrix with radial lines" begin
    # to test the if the ABA matrix contains the same information, the power
    # flows must be evaluated
    for name in ["c_sys14", "test_RTS_GMLC_sys"]
        # load the system
        sys = PSB.build_system(PSB.PSITestSystems, name)

        # get the original and reduced IncidenceMatrix, BA and ABA
        A = IncidenceMatrix(sys)
        A_rad =
            IncidenceMatrix(sys; network_reductions = NetworkReduction[RadialReduction()])
        BA = BA_Matrix(sys)
        BA_rad = BA_Matrix(sys; network_reductions = NetworkReduction[RadialReduction()])
        ABA = ABA_Matrix(sys; factorize = true)
        ABA_rad = ABA_Matrix(
            sys;
            factorize = true,
            network_reductions = NetworkReduction[RadialReduction()],
        )

        nr = A_rad.network_reduction_data
        # check if the same angles and flows are computed with the matrices of the reduced systems
        # get the indices for the reduced system
        bus_numbers = []
        for i in keys(nr.bus_reduction_map)
            append!(bus_numbers, collect(nr.bus_reduction_map[i]))
        end
        bus_idx = setdiff(
            1:size(A.data, 2),
            append!([A.lookup[2][i] for i in bus_numbers], PNM.get_ref_bus_position(A)),
        )
        br_idx = setdiff(1:size(A.data, 1), [A.lookup[1][i] for i in nr.removed_arcs])

        # now get the injenctions from the system
        n_buses = length(axes(BA, 1))
        bus_lookup = BA.lookup[1]
        branch_lookup = BA.lookup[2]
        bus_angles = zeros(Float64, n_buses)
        branch_flow_values = zeros(Float64, length(axes(BA, 2)))
        temp_bus_map = Dict{Int, String}(
            PSY.get_number(b) => PSY.get_name(b) for
            b in PSY.get_components(PSY.ACBus, sys)
        )
        for (bus_no, ix) in bus_lookup
            bus_name = temp_bus_map[bus_no]
            bus = PSY.get_component(PSY.Bus, sys, bus_name)
            if PSY.get_bustype(bus) == PSY.ACBusTypes.REF
                bus_angles[ix] = 0.0
            else
                bus_angles[ix] = PSY.get_angle(bus)
            end
        end
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
        valid_ix = setdiff(1:length(power_injection), PNM.get_ref_bus_position(BA))
        ref_bus_angles = deepcopy(bus_angles)
        ref_flow_values = deepcopy(branch_flow_values)

        # now get the reference power flows and angle evaluated with BA and ABA
        ref_bus_angles[valid_ix] = ABA.K \ power_injection[valid_ix]
        ref_flow_values = transpose(BA.data) * ref_bus_angles

        # evaluate according to the matrices with no radial branches
        reduced_bus_angles = zeros((length(bus_idx) + length(PNM.get_ref_bus_position(A)),))
        reduce_flow_values = zeros((length(br_idx),))
        # change power injection for affected leaf buses
        power_injection2 = deepcopy(power_injection)
        for i in keys(nr.bus_reduction_map)
            for j in nr.bus_reduction_map[i]
                power_injection2[BA.lookup[1][i]] += power_injection[BA.lookup[1][j]]
            end
        end
        valid_ix2 = setdiff(1:size(BA_rad.data, 1), PNM.get_ref_bus_position(BA_rad))
        reduced_bus_angles[valid_ix2] = ABA_rad.K \ power_injection2[bus_idx]
        reduced_flow_values = transpose(BA_rad.data) * reduced_bus_angles

        # now check if flows and angles are the same
        @test isapprox(ref_bus_angles[bus_idx], reduced_bus_angles[valid_ix2])
        @test isapprox(ref_flow_values[br_idx], reduced_flow_values)
    end
end
