using SparseArrays

function parse_psse_ybus(path)
    data = readlines(path)
    row_buses, col_buses, y_values = Int[], Int[], YBUS_ELTYPE[]
    reduced_bus_pairs = Vector{Vector{Int}}()
    values_done = false
    for line in data
        if line == " ZERO IMPEDANCE LINE CONNECTED BUSES (BUS IN MATRIX LISTED FIRST):"
            values_done = true
        end
        if !isempty(line) &&
           line != " ZERO IMPEDANCE LINE CONNECTED BUSES (BUS IN MATRIX LISTED FIRST):"
            if values_done == true
                vals = split(strip(line), r",\s*")
                push!(reduced_bus_pairs, [parse(Int, x) for x in vals])
            else
                vals = split(strip(line), r",\s*")
                row_bus = parse(Int, vals[1])
                col_bus = parse(Int, vals[2])
                y_real = parse(real(YBUS_ELTYPE), vals[3])
                y_imag = parse(real(YBUS_ELTYPE), vals[4])
                push!(row_buses, row_bus)
                push!(col_buses, col_bus)
                push!(y_values, y_real + im * y_imag)
            end
        end
    end
    unique_buses = sort(unique(vcat(row_buses, col_buses)))
    bus_to_ix_map = Dict(bus => i for (i, bus) in enumerate(unique_buses))
    row_indices = [bus_to_ix_map[f] for f in row_buses]
    col_indices = [bus_to_ix_map[t] for t in col_buses]
    return sparse(row_buses, col_buses, y_values),
    bus_to_ix_map,
    row_buses,
    col_buses,
    y_values,
    reduced_bus_pairs
end

function _test_psse_reduction_row(psse_row, reverse_bus_search_map)
    reduced = zeros(length(psse_row))
    reduced_to = zeros(length(psse_row))
    for (ix, x) in enumerate(psse_row)
        if haskey(reverse_bus_search_map, x)
            reduced[ix] = 1
            reduced_to[ix] = reverse_bus_search_map[x]
        end
    end
    @test sum(reduced) == length(psse_row) - 1   #all but one entry should be reduced 
    ix = findfirst(isequal(0), reduced)
    splice!(reduced_to, ix)
    @test all(reduced_to .== psse_row[ix])       #all entries should be reduced to the one entry that was not reduced
end

@testset "14 bus system" begin
    sys = build_system(PSSEParsingTestSystems, "psse_ybus_14_test_system")
    Ybus_pnm = Ybus(sys)
    ref_bus_numbers = [
        get_number(x) for
        x in get_components(x -> get_bustype(x) == PSY.ACBusTypes.REF, ACBus, sys)
    ]
    skip_indices = indexin(ref_bus_numbers, Ybus_pnm.axes[1])
    n_ref_bus_elements = nnz(Ybus_pnm.data[skip_indices, :])
    nr = Ybus_pnm.network_reduction_data

    #Test adjacency and Ybus have same non-zero elements
    findnz(Ybus_pnm.data)[1] == findnz(Ybus_pnm.adjacency_data)[1]
    findnz(Ybus_pnm.data)[2] == findnz(Ybus_pnm.adjacency_data)[2]

    # Compare with PSSE
    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(joinpath(TEST_DATA_DIR, "14bus_Ymx.txt"))         #NOTE - required manual remapping of star buses
    # Compare breaker/switch reductions
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end
    #Test number of nonzero elements matches
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_ref_bus_elements
    #Test values match 
    for (row_bus, col_bus, val) in zip(row_buses, col_buses, y_values)
        if row_bus ∈ keys(nr.reverse_bus_search_map)
            row_bus = nr.reverse_bus_search_map[row_bus]
        end
        if col_bus ∈ keys(nr.reverse_bus_search_map)
            col_bus = nr.reverse_bus_search_map[col_bus]
        end
        @test isapprox(Ybus_pnm[row_bus, col_bus], val, rtol = 2 * eps(Float32), atol = 0.0)
    end
end

@testset "WECC 240 bus" begin
    sys = PSB.build_system(PSYTestSystems, "psse_240_parsing_sys"; runchecks = false)
    Ybus_pnm = Ybus(sys; include_constant_impedance_loads = true)
    nr = Ybus_pnm.network_reduction_data
    ref_bus_numbers = [
        get_number(x) for
        x in get_components(x -> get_bustype(x) == PSY.ACBusTypes.REF, ACBus, sys)
    ]
    skip_indices = indexin(ref_bus_numbers, Ybus_pnm.axes[1])
    n_ref_bus_elements = nnz(Ybus_pnm.data[skip_indices, :])

    # Compare with PSSE
    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(joinpath(TEST_DATA_DIR, "240busWECC_2018_PSS33_Ymatrix.txt"))

    # Compare breaker/switch reductions
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end
    #Test number of nonzero elements matches
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_ref_bus_elements
    #Test values match 
    for (row_bus, col_bus, val) in zip(row_buses, col_buses, y_values)
        if row_bus ∈ keys(nr.reverse_bus_search_map)
            row_bus = nr.reverse_bus_search_map[row_bus]
        end
        if col_bus ∈ keys(nr.reverse_bus_search_map)
            col_bus = nr.reverse_bus_search_map[col_bus]
        end
        @test isapprox(Ybus_pnm[row_bus, col_bus], val, rtol = 4 * eps(Float32), atol = 0.0)
    end
end

@testset "Base_Eastern_Interconnect_515GW" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "Base_Eastern_Interconnect_515GW")
    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(
            joinpath(TEST_DATA_DIR, "Base_Eastern_Interconnect_515GW_Ymatrix.dat"),
        )

    Ybus_pnm = Ybus(sys; include_constant_impedance_loads = true)
    nr = Ybus_pnm.network_reduction_data
    ref_bus_numbers = [
        get_number(x) for
        x in get_components(x -> get_bustype(x) == PSY.ACBusTypes.REF, ACBus, sys)
    ]

    skip_indices = indexin(ref_bus_numbers, Ybus_pnm.axes[1])
    n_ref_bus_elements = nnz(Ybus_pnm.data[skip_indices, :])

    # Compare breaker/switch reductions
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end
    #Test number of nonzero elements matches
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_ref_bus_elements
    #Test values match 
    # PSSE does not write values in the REF bus row, so we need to add them manually
    ref_row =
        get_number(first(get_components(x -> get_bustype(x) == ACBusTypes.REF, ACBus, sys)))
    Ybus_psse[ref_row, Ybus_pnm.axes[2]] = Ybus_pnm[ref_row, :]
    # PSS/E reference data has Float32 precision, so tolerance must reflect that.
    @test isapprox(
        Ybus_pnm.data,
        Ybus_psse[Ybus_pnm.axes[1], Ybus_pnm.axes[2]],
        rtol = 2 * eps(Float32),
        atol = 0.0,
    )

    rows, cols, vals = findnz(Ybus_psse)
    r, c, v = findnz(Ybus_pnm.data)
    @test isapprox(v, vals, rtol = 2 * eps(Float32), atol = 0.0) # test all values first
    # Now also test that both the structure and the values match
    @test isapprox(
        Ybus_pnm.data[r, c],
        Ybus_psse[rows, cols],
        rtol = 2 * eps(Float32),
        atol = 0.0,
    )
end

@testset "14 bus system with phase shifting 3wt" begin
    sys = build_system(PSSEParsingTestSystems, "pti_case14_with_pst3w_sys")
    Ybus_pnm = Ybus(sys)
    ref_bus_numbers = [
        get_number(x) for
        x in get_components(x -> get_bustype(x) == PSY.ACBusTypes.REF, ACBus, sys)
    ]
    skip_indices = indexin(ref_bus_numbers, Ybus_pnm.axes[1])
    n_ref_bus_elements = nnz(Ybus_pnm.data[skip_indices, :])
    nr = Ybus_pnm.network_reduction_data

    #Test adjacency and Ybus have same non-zero elements
    findnz(Ybus_pnm.data)[1] == findnz(Ybus_pnm.adjacency_data)[1]
    findnz(Ybus_pnm.data)[2] == findnz(Ybus_pnm.adjacency_data)[2]

    # Compare with PSSE
    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(joinpath(TEST_DATA_DIR, "case14_with_pst3w_ybus.txt"))         #NOTE - required manual remapping of star buses
    # Compare breaker/switch reductions
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end
    #Test number of nonzero elements matches
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_ref_bus_elements
    #Test values match 
    for (row_bus, col_bus, val) in zip(row_buses, col_buses, y_values)
        if row_bus ∈ keys(nr.reverse_bus_search_map)
            row_bus = nr.reverse_bus_search_map[row_bus]
        end
        if col_bus ∈ keys(nr.reverse_bus_search_map)
            col_bus = nr.reverse_bus_search_map[col_bus]
        end
        @test isapprox(Ybus_pnm[row_bus, col_bus], val; rtol = 9 * eps(Float32), atol = 0.0)
    end
end

@testset "4 bus system with zero impedance correction for three winding transformer" begin
    sys = build_system(PSSEParsingTestSystems, "psse_4_zero_impedance_3wt_test_system")
    Ybus_pnm = Ybus(sys)
    ref_bus_numbers = [
        get_number(x) for
        x in get_components(x -> get_bustype(x) == PSY.ACBusTypes.REF, ACBus, sys)
    ]
    skip_indices = indexin(ref_bus_numbers, Ybus_pnm.axes[1])
    n_ref_bus_elements = nnz(Ybus_pnm.data[skip_indices, :])
    nr = Ybus_pnm.network_reduction_data

    #Test adjacency and Ybus have same non-zero elements
    findnz(Ybus_pnm.data)[1] == findnz(Ybus_pnm.adjacency_data)[1]
    findnz(Ybus_pnm.data)[2] == findnz(Ybus_pnm.adjacency_data)[2]

    # Compare with PSSE
    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(joinpath(TEST_DATA_DIR, "case4_zero_impedance_3wt_Ymatrix.txt"))         #NOTE - required manual remapping of star buses
    # Compare breaker/switch reductions
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end
    #Test number of nonzero elements matches
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_ref_bus_elements
    #Test values match 
    for (row_bus, col_bus, val) in zip(row_buses, col_buses, y_values)
        if row_bus ∈ keys(nr.reverse_bus_search_map)
            row_bus = nr.reverse_bus_search_map[row_bus]
        end
        if col_bus ∈ keys(nr.reverse_bus_search_map)
            col_bus = nr.reverse_bus_search_map[col_bus]
        end
        @test isapprox(Ybus_pnm[row_bus, col_bus], val; rtol = 2 * eps(Float32), atol = 0.0)
    end
end

@testset "psse_modified_14bus_off_nominal_icd" begin
    # Three transformers have off-nominal settings that activate their ICTs:
    #   BUS 109-BUS 104-i_1  (TapTransformer):          WINDV1=0.95, table 4 → F=1.030
    #   BUS 110-BUS 109-i_1  (PhaseShiftingTransformer): ANG1=12.2°,  table 3 → F=1.082
    #   BUS 113-BUS 110-BUS 114-i_1 winding 1 (PST3W):  ANG1=10.0°,  table 9 → F=1.100
    sys = build_system(PSSEParsingTestSystems, "psse_modified_14bus_off_nominal_icd")
    Ybus_pnm = Ybus(sys)
    nr = Ybus_pnm.network_reduction_data

    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(
            joinpath(TEST_DATA_DIR, "modified_14bus_system_off_nominal_icd_ymatrix.txt"),
        )
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end

    psse_ref_bus_numbers = [101, 108]
    psse_skip_indices = indexin(psse_ref_bus_numbers, Ybus_pnm.axes[1])
    n_psse_excluded_elements = nnz(Ybus_pnm.data[psse_skip_indices, :])
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_psse_excluded_elements

    psse_to_pnm_star_bus = Dict(1100001 => 1002, 1100002 => 1001)
    for (row_bus, col_bus, val) in zip(row_buses, col_buses, y_values)
        row_bus = get(psse_to_pnm_star_bus, row_bus, row_bus)
        col_bus = get(psse_to_pnm_star_bus, col_bus, col_bus)
        if row_bus ∈ keys(nr.reverse_bus_search_map)
            row_bus = nr.reverse_bus_search_map[row_bus]
        end
        if col_bus ∈ keys(nr.reverse_bus_search_map)
            col_bus = nr.reverse_bus_search_map[col_bus]
        end
        @test isapprox(Ybus_pnm[row_bus, col_bus], val; rtol = 2 * eps(Float32), atol = 0.0)
    end
end

@testset "psse_modified_14bus_nominal_icd" begin
    sys = build_system(PSSEParsingTestSystems, "psse_modified_14bus_nominal_icd")
    Ybus_pnm = Ybus(sys)
    nr = Ybus_pnm.network_reduction_data

    Ybus_psse, b_ix_psse, row_buses, col_buses, y_values, reduced_bus_pairs_psse =
        parse_psse_ybus(joinpath(TEST_DATA_DIR, "modified_14bus_system_no_icd.txt"))
    for x in reduced_bus_pairs_psse
        _test_psse_reduction_row(x, nr.reverse_bus_search_map)
    end

    # PSS/E excluded rows for both type-3 buses it saw (101 and 108); count those
    # PNM entries to reconcile the NNZ check.
    psse_ref_bus_numbers = [101, 108]
    psse_skip_indices = indexin(psse_ref_bus_numbers, Ybus_pnm.axes[1])
    n_psse_excluded_elements = nnz(Ybus_pnm.data[psse_skip_indices, :])
    @test nnz(Ybus_pnm.data) == length(filter(!iszero, y_values)) + n_psse_excluded_elements

    # PSS/E numbers its switch star-buses as 1100001/1100002; PNM assigns 1002/1001.
    # Adjacency-derived mapping: PSS/E 1100001 ↔ {114,112,110} = PNM 1002;
    #                            PSS/E 1100002 ↔ {107,104,109} = PNM 1001.
    psse_to_pnm_star_bus = Dict(1100001 => 1002, 1100002 => 1001)
    for (row_bus, col_bus, val) in zip(row_buses, col_buses, y_values)
        row_bus = get(psse_to_pnm_star_bus, row_bus, row_bus)
        col_bus = get(psse_to_pnm_star_bus, col_bus, col_bus)
        if row_bus ∈ keys(nr.reverse_bus_search_map)
            row_bus = nr.reverse_bus_search_map[row_bus]
        end
        if col_bus ∈ keys(nr.reverse_bus_search_map)
            col_bus = nr.reverse_bus_search_map[col_bus]
        end
        @test isapprox(Ybus_pnm[row_bus, col_bus], val; rtol = 2 * eps(Float32), atol = 0.0)
    end
end
