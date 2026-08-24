# Find a bridge arc (PTDF_A_diag ≈ 1.0) not in `exclude`.
function _find_bridge_arc(vmodf; exclude = Set{Int}())
    for e in eachindex(vmodf.PTDF_A_diag)
        if abs(vmodf.PTDF_A_diag[e] - 1.0) < 1e-6 && !(e in exclude)
            return e
        end
    end
    error("No bridge arc found")
end

# Find a non-bridge arc (PTDF_A_diag well below 1.0) not in `exclude`.
function _find_non_bridge_arc(vmodf; exclude = Set{Int}())
    for e in eachindex(vmodf.PTDF_A_diag)
        if abs(vmodf.PTDF_A_diag[e]) < 1.0 - 1e-6 && !(e in exclude)
            return e
        end
    end
    error("No non-bridge arc found")
end

# Bus positions disconnected from the monitored arc after outaging `outaged`,
# computed independently via plain DFS over the incidence matrix (a cross-check
# for the production zeroing logic, which uses a separate union-find over BA).
function _islanded_positions(vmodf, outaged::Vector{Int}, monitored::Int)
    A = vmodf.A
    keep = setdiff(1:size(A, 1), outaged)
    Ak = A[keep, :]
    nbus = size(A, 2)
    neighbors = [Set{Int}() for _ in 1:nbus]
    for r in 1:size(Ak, 1)
        buses = findall(!iszero, Ak[r, :])
        for i in buses, j in buses
            i == j || push!(neighbors[i], j)
        end
    end
    start = first(findall(!iszero, A[monitored, :]))
    seen = falses(nbus)
    seen[start] = true
    stack = [start]
    while !isempty(stack)
        c = pop!(stack)
        for r in neighbors[c]
            if !seen[r]
                seen[r] = true
                push!(stack, r)
            end
        end
    end
    return findall(!, seen)
end

# After islanding, entries of buses disconnected from the monitored arc must be
# exactly zero; connected entries are left as the Woodbury solve produced them.
function _check_islanded_zeros(row, vmodf, outaged, monitored)
    islanded = _islanded_positions(vmodf, outaged, monitored)
    @test !isempty(islanded)                       # the contingency really islands
    @test all(==(0.0), row[islanded])              # exactly zero, not just small
    return islanded
end

@testset "MODF islanding" begin
    sys14 = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vmodf = VirtualMODF(sys14)
    ptdf_ref = PTDF(sys14)

    # Compute shared arc indices once — deterministic since PTDF_A_diag is fixed
    e_bridge1 = _find_bridge_arc(vmodf)
    e_bridge2 = _find_bridge_arc(vmodf; exclude = Set([e_bridge1]))
    e_other = _find_non_bridge_arc(vmodf; exclude = Set([e_bridge1, e_bridge2]))
    monitored =
        _find_non_bridge_arc(vmodf; exclude = Set([e_bridge1, e_bridge2, e_other]))

    @testset "M=1 single bridge arc" begin
        b = vmodf.arc_susceptances[e_bridge1]
        ctg = ContingencySpec(
            Base.UUID(50001),
            NetworkModification("island_m1", [ArcModification(e_bridge1, -b)]),
        )
        vmodf.contingency_cache[ctg.id] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test all(isfinite, row)
        # Islanded buses are forced to exactly zero; the connected buses keep the
        # pre-contingency PTDF row (a single bridge does not change intra-component
        # sensitivities, so the Woodbury correction there is zero).
        islanded = _check_islanded_zeros(row, vmodf, [e_bridge1], monitored)
        connected = setdiff(1:length(row), islanded)
        ref_row = collect(ptdf_ref[monitored, :])
        @test isapprox(row[connected], ref_row[connected]; atol = 1e-6)
        # The public getindex path (which also applies the sparsification cutoff)
        # preserves the exact zeros.
        public_row = vmodf[monitored, ctg]
        @test all(==(0.0), public_row[islanded])
        PNM.clear_caches!(vmodf)
    end

    @testset "M=2 one bridge + one non-bridge" begin
        b_br = vmodf.arc_susceptances[e_bridge1]
        b_ot = vmodf.arc_susceptances[e_other]
        ctg = ContingencySpec(
            Base.UUID(50002),
            NetworkModification(
                "island_m2",
                [ArcModification(e_bridge1, -b_br), ArcModification(e_other, -b_ot)],
            ),
        )
        vmodf.contingency_cache[ctg.id] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test !all(x -> abs(x) < 1e-10, row)
        @test all(isfinite, row)
        _check_islanded_zeros(row, vmodf, [e_bridge1, e_other], monitored)
        PNM.clear_caches!(vmodf)
    end

    @testset "M=2 two bridges, fully null W" begin
        b1 = vmodf.arc_susceptances[e_bridge1]
        b2 = vmodf.arc_susceptances[e_bridge2]
        ctg = ContingencySpec(
            Base.UUID(50003),
            NetworkModification(
                "island_2bridge",
                [ArcModification(e_bridge1, -b1), ArcModification(e_bridge2, -b2)],
            ),
        )
        vmodf.contingency_cache[ctg.id] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test all(isfinite, row)
        # Both bridges island a bus apiece; those entries are zeroed, the rest of
        # the row keeps the pre-contingency PTDF values.
        islanded = _check_islanded_zeros(row, vmodf, [e_bridge1, e_bridge2], monitored)
        connected = setdiff(1:length(row), islanded)
        ref_row = collect(ptdf_ref[monitored, :])
        @test isapprox(row[connected], ref_row[connected]; atol = 1e-6)
        PNM.clear_caches!(vmodf)
    end

    @testset "M=3 two bridges + one non-bridge" begin
        b1 = vmodf.arc_susceptances[e_bridge1]
        b2 = vmodf.arc_susceptances[e_bridge2]
        b3 = vmodf.arc_susceptances[e_other]
        ctg = ContingencySpec(
            Base.UUID(50004),
            NetworkModification(
                "island_m3",
                [
                    ArcModification(e_bridge1, -b1),
                    ArcModification(e_bridge2, -b2),
                    ArcModification(e_other, -b3),
                ],
            ),
        )
        vmodf.contingency_cache[ctg.id] = ctg

        row = PNM._compute_modf_entry(vmodf, monitored, ctg.modification)
        @test !all(x -> abs(x) < 1e-10, row)
        @test all(isfinite, row)
        _check_islanded_zeros(row, vmodf, [e_bridge1, e_bridge2, e_other], monitored)
        PNM.clear_caches!(vmodf)
    end
end

@testset "MODF non-islanding: pinv changes do not affect normal path" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vlodf = VirtualLODF(sys)
    ptdf_ref = PTDF(sys)
    vmodf = VirtualMODF(sys)

    n_arcs = size(vlodf, 1)

    for e in 1:n_arcs
        @test abs(vmodf.PTDF_A_diag[e]) < 1.0 - 1e-6
    end

    for e in 1:n_arcs
        b_e = vmodf.arc_susceptances[e]
        ctg_id = Base.UUID(60000 + e)
        ctg = ContingencySpec(
            ctg_id,
            NetworkModification("regression_$e", [ArcModification(e, -b_e)]),
        )
        vmodf.contingency_cache[ctg_id] = ctg

        for m in 1:n_arcs
            modf_row = PNM._compute_modf_entry(vmodf, m, ctg.modification)
            expected = ptdf_ref[m, :] .+ vlodf[m, e] .* ptdf_ref[e, :]
            @test isapprox(modf_row, expected; atol = 1e-6)
        end
        PNM.clear_caches!(vmodf)
    end
end

@testset "VirtualPTDF islanding: disconnected entries are exactly zero" begin
    # The zeroing lives in the shared Woodbury kernel, so VirtualPTDF's
    # post-modification rows get the same exact-zero guarantee as VirtualMODF.
    sys = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    vptdf = VirtualPTDF(sys)
    vmodf = VirtualMODF(sys)        # only used to locate a bridge arc via PTDF_A_diag
    ptdf_ref = PTDF(sys)

    # Arc indices align between VirtualPTDF and VirtualMODF of the same system.
    e_bridge = _find_bridge_arc(vmodf)
    monitored = _find_non_bridge_arc(vmodf; exclude = Set([e_bridge]))

    b = vptdf.arc_susceptances[e_bridge]
    mod = NetworkModification("ptdf_island", [ArcModification(e_bridge, -b)])
    row = get_post_modification_ptdf_row(vptdf, monitored, mod)

    @test all(isfinite, row)
    islanded = _check_islanded_zeros(row, vptdf, [e_bridge], monitored)
    connected = setdiff(1:length(row), islanded)
    ref_row = collect(ptdf_ref[monitored, :])
    @test isapprox(row[connected], ref_row[connected]; atol = 1e-6)
end
