using SparseArrays
using LinearAlgebra

@testset "YbusSplit" begin

    # --- No transformers: y_series + Diagonal(y_shunt) should equal the normal Ybus ---
    @testset "No transformers: recombined split equals Ybus" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        ybus = Ybus(sys)
        ybus_split = YbusSplit(sys)

        y_recombined = ybus_split.y_series + spdiagm(ybus_split.y_shunt)
        # dropzeros so the comparison is clean
        dropzeros!(y_recombined)

        @test size(ybus.data) == size(y_recombined)
        I_ref, J_ref, V_ref = findnz(ybus.data)
        I_split, J_split, V_split = findnz(y_recombined)
        @test Set(zip(I_ref, J_ref)) == Set(zip(I_split, J_split))
        for (i, j, v) in zip(I_ref, J_ref, V_ref)
            @test isapprox(y_recombined[i, j], v; atol = 1e-5)
        end
    end

    # --- With transformers: same sparsity structure, off-diagonal match away from TX buses ---
    @testset "With transformers: structure and non-TX entries" begin
        sys = build_system(MatpowerTestSystems, "matpower_case5_transformer")
        ybus = Ybus(sys)
        ybus_split = YbusSplit(sys)

        y_recombined = ybus_split.y_series + spdiagm(ybus_split.y_shunt)

        # Sparsity pattern of y_series should be a superset of the normal Ybus
        # (transformer arcs are structurally present but numerically zero in y_series)
        I_ref, J_ref, _ = findnz(ybus.data)
        ref_pattern = Set(zip(I_ref, J_ref))
        I_split, J_split, _ = findnz(ybus_split.y_series)
        split_pattern = Set(zip(I_split, J_split))
        @test ref_pattern ⊆ split_pattern

        # Find buses connected to transformers so we can exclude them
        bus_lookup = PNM.get_bus_lookup(ybus)
        tx_bus_indices = Set{Int}()
        for br in PSY.get_components(
            x -> PSY.get_available(x),
            PSY.TwoWindingTransformer,
            sys,
        )
            arc = PSY.get_arc(br)
            from_no = PSY.get_number(PSY.get_from(arc))
            to_no = PSY.get_number(PSY.get_to(arc))
            push!(tx_bus_indices, bus_lookup[from_no])
            push!(tx_bus_indices, bus_lookup[to_no])
        end

        # Off-diagonal entries not touching any transformer bus should match exactly
        for (i, j, v) in zip(I_ref, J_ref, findnz(ybus.data)[3])
            i == j && continue
            (i ∈ tx_bus_indices || j ∈ tx_bus_indices) && continue
            @test isapprox(y_recombined[i, j], v; atol = 1e-5)
        end
    end
end
