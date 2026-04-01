function types_in_series_reduction(nrd::PNM.NetworkReductionData)
    types = Set{DataType}()
    for segments in values(PNM.get_series_branch_map(nrd))
        for comp in segments
            push!(types, typeof(comp))
        end
    end
    return types
end

function find_parallel_arc(sys::System)
    arcs_seen = Set{Tuple{Int, Int}}()
    for br in PSY.get_components(
        x -> !(typeof(x) <: PSY.ThreeWindingTransformer),
        PSY.ACBranch,
        sys,
    )
        arc = PNM.get_arc_tuple(br)
        if arc in arcs_seen
            return arc
        else
            push!(arcs_seen, arc)
        end
    end
    error("No parallel arcs found")
    return (-1, -1)
end

function test_all_subtypes(sys::System, network_reductions)
    for T in subtypes(PNM.PowerNetworkMatrix)
        # arc admittance matrix constructor has different args.
        (T == PNM.Ybus || T == PNM.ArcAdmittanceMatrix || T == PNM.YbusSplit) && continue
        M = T(sys; network_reductions = deepcopy(network_reductions))
        # test that it runs without error
        @test M isa T
    end
    # return Ybus so we can inspect the network reduction data.
    Y = Ybus(sys; network_reductions = deepcopy(network_reductions))
    @test Y isa PNM.Ybus
    #Y_split = YbusSplit(sys)
    #@test Y_split isa PNM.YbusSplit
    return Y
end

@testset "3WT + radial" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[RadialReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the 3WT arc was actually reduced
    trf = first(get_components(PSY.Transformer3W, sys))
    trf_arcs = Tuple{Int, Int}[
        PNM.get_arc_tuple(PSY.get_primary_star_arc(trf)),
        PNM.get_arc_tuple(PSY.get_secondary_star_arc(trf)),
        PNM.get_arc_tuple(PSY.get_tertiary_star_arc(trf)),
    ]
    nrd = PNM.get_network_reduction_data(Y)
    @test any(arc in PNM.get_removed_arcs(nrd) for arc in trf_arcs) ||
          any(reverse(arc) in PNM.get_removed_arcs(nrd) for arc in trf_arcs)
end

@testset "3WT + degree-2" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[DegreeTwoReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the 3WT arc was actually reduced
    nrd = PNM.get_network_reduction_data(Y)
    @test PNM.ThreeWindingTransformerWinding{Transformer3W} in
          types_in_series_reduction(nrd)
end

@testset "Parallel lines + radial" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[RadialReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the parallel lines were reduced
    nrd = PNM.get_network_reduction_data(Y)
    parallel_arc = find_parallel_arc(sys)
    @test parallel_arc in keys(nrd.parallel_branch_map)
end

@testset "Parallel lines + degree-2" begin
    sys = build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    network_reductions = NetworkReduction[DegreeTwoReduction()]
    Y = test_all_subtypes(sys, network_reductions)
    # test that the parallel lines were reduced
    nrd = PNM.get_network_reduction_data(Y)
    @test PNM.BranchesParallel{Line} in types_in_series_reduction(nrd)
end

@testset "Test Reductions with filters" begin
    sys_rts_da = build_system(PSISystems, "modified_RTS_GMLC_DA_sys")

    ptdf = VirtualPTDF(
        sys_rts_da;
        network_reductions = NetworkReduction[
            RadialReduction(),
            DegreeTwoReduction(),
        ],
    )
    PowerNetworkMatrices.populate_branch_maps_by_type!(PNM.get_network_reduction_data(ptdf),
        Dict(Line => x -> occursin("B", get_name(x)),
            TapTransformer => x -> occursin("B", get_name(x))))
    @test PNM.has_filtered_branches(PNM.get_network_reduction_data(ptdf))
    for k in keys(PNM.get_network_reduction_data(ptdf).name_to_arc_map[Line])
        @test occursin("B", k)
    end
    PNM.empty!(PNM.get_network_reduction_data(ptdf))
    @test isempty(PNM.get_network_reduction_data(ptdf))
end
