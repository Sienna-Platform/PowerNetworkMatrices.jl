# Several testsets below assert that AutoTolerance is a no-op by comparing a matrix built with
# it against one built at eps(). Two separate builds are not bitwise reproducible -- the
# AppleAccelerate backend drifts by a ULP on roughly 1 run in 20 -- so these compare with a
# tolerance. 1e-10 still proves the no-op: genuinely sparsifying would zero entries, moving
# them by their full magnitude (>=1e-3 in these systems).
const NO_OP_BUILD_ATOL = 1e-10

@testset "condition estimate: KLU/AA parity" begin
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
    A = IncidenceMatrix(sys)
    BA = BA_Matrix(sys)
    ref = Set(PNM.get_ref_bus_position(BA))
    ABA = PNM.calculate_ABA_matrix(A.data, BA.data, ref)

    klu_cache = PNM.klu_factorize(ABA)
    kappa_klu = PNM._condition_estimate(klu_cache, ABA)
    @test isfinite(kappa_klu) && kappa_klu > 1.0
    @test (@inferred PNM._condition_estimate(klu_cache, ABA)) isa Float64

    if PNM._has_apple_accelerate_backend()
        aa_cache = PNM.AccelerateWrapper.aa_factorize(ABA)
        kappa_aa = PNM._condition_estimate(aa_cache, ABA)
        @test 0.5 <= kappa_aa / kappa_klu <= 2.0
    end
end

@testset "PTDF backward compatibility + AutoTolerance (dense)" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    A = IncidenceMatrix(sys)
    BA = BA_Matrix(sys)

    # Explicit Float64 is an absolute cutoff, honored verbatim (backward compat).
    p_num = PTDF(A, BA; tol = 1e-3)
    @test PNM.get_tol(p_num) == 1e-3
    p_eps = PTDF(A, BA; tol = eps())
    @test PNM.get_tol(p_eps) == eps()

    # The dense PTDF is the small-system path: AutoTolerance is a NO-OP there. It
    # stays an exact, dense `Matrix{Float64}` (so `DC_PTDF_Matrix` and downstream
    # dispatch are preserved); sparsification is reserved for VirtualPTDF.
    p_auto = PTDF(A, BA; tol = AutoTolerance(; data_precision = 1e-2))
    @test PNM.get_tol(p_auto) == eps()
    @test isapprox(get_ptdf_data(p_auto), get_ptdf_data(p_eps); atol = NO_OP_BUILD_ATOL)
    # Dense storage is preserved -> matches the DC_PTDF_Matrix alias (the data
    # field is a Matrix{Float64}, so downstream dispatch is unaffected).
    @test p_auto isa PNM.DC_PTDF_Matrix
    @test PTDF(A, BA) isa PNM.DC_PTDF_Matrix
end

@testset "VirtualPTDF small system: AutoTolerance is a no-op; Float64 sparsifies" begin
    # ACTIVSg2000 (1999 buses) is below the gate, so the default AutoTolerance is
    # exact; a Float64 tol still sparsifies columns at any size.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")

    v_exact = VirtualPTDF(sys; tol = eps())
    # The sparsification check needs a meshed arc: a radial stub's row is a single unit
    # entry with the rest at numerical noise straddling `eps()`, so nothing is left to drop.
    arc_ax = PNM.get_arc_axis(v_exact)
    arc = arc_ax[findfirst(a -> count(>(1e-9), abs.(v_exact[a, :])) > 1, arc_ax)]
    nnz_dense = count(!iszero, v_exact[arc, :])

    v_auto = VirtualPTDF(sys)                           # default, below gate
    @test v_auto.tol isa PNM.AbsoluteCutoff
    @test PNM.get_tol(v_auto) == eps()
    # No entries dropped; the `count` is what proves nothing was sparsified away.
    @test isapprox(v_auto[arc, :], v_exact[arc, :]; atol = NO_OP_BUILD_ATOL)
    @test count(!iszero, v_auto[arc, :]) == nnz_dense

    # Explicit Float64 -> absolute cutoff -> genuinely sparse, faithful column.
    v_num = VirtualPTDF(sys; tol = 1e-2)
    @test v_num.tol isa PNM.AbsoluteCutoff
    sparse_row = v_num[arc, :]
    @test count(!iszero, sparse_row) < nnz_dense
    dense_row = v_exact[arc, :]
    for i in eachindex(sparse_row)
        iszero(sparse_row[i]) ||
            @test isapprox(sparse_row[i], dense_row[i]; rtol = sqrt(eps()))
    end
end

@testset "LODF AutoTolerance" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    A = IncidenceMatrix(sys)
    BA = BA_Matrix(sys)
    ABA = ABA_Matrix(sys; factorize = true)

    l_num = LODF(A, ABA, BA; tol = 1e-3)
    @test PNM.get_tol(l_num) == 1e-3

    # The dense LODF is the small-system path: AutoTolerance is a no-op (eps), so
    # the result is identical to the exact LODF. Sparsification is a VirtualLODF
    # feature; pass a Float64 tol to sparsify a dense LODF explicitly.
    l_exact = LODF(A, ABA, BA; tol = eps())
    l_auto = LODF(A, ABA, BA; tol = AutoTolerance(; data_precision = 1e-2))
    @test PNM.get_tol(l_auto) == eps()
    @test isapprox(l_auto.data, l_exact.data; atol = NO_OP_BUILD_ATOL)

    # System constructor: AutoTolerance routes through the ABA path and is a no-op.
    l_sys_auto = LODF(sys; tol = AutoTolerance(; data_precision = 1e-2))
    @test PNM.get_tol(l_sys_auto) == eps()

    # from-PTDF constructor: AutoTolerance is a no-op (eps), so the default
    # LODF(A, ptdf) keeps working and stays exact.
    P = PTDF(A, BA; tol = eps())
    l_from_ptdf = LODF(A, P; tol = AutoTolerance())
    @test PNM.get_tol(l_from_ptdf) == eps()
end

@testset "LODF preserves -1.0 diagonal under aggressive tol" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    A = IncidenceMatrix(sys)
    BA = BA_Matrix(sys)
    ABA = ABA_Matrix(sys; factorize = true)
    # A tol >= 1.0 would let droptol! drop the structural -1.0 diagonal; it must
    # be re-asserted so the LODF contract holds for downstream N-1 analysis.
    l = LODF(A, ABA, BA; tol = 2.0)
    d = PNM.SparseArrays.sparse(l.data)
    @test all(d[i, i] == -1.0 for i in axes(d, 1))
    # Same protection on the from-PTDF path.
    l2 = LODF(A, PTDF(A, BA; tol = eps()); tol = 2.0)
    d2 = PNM.SparseArrays.sparse(l2.data)
    @test all(d2[i, i] == -1.0 for i in axes(d2, 1))
end

@testset "VirtualLODF / VirtualMODF accept AutoTolerance" begin
    # c_sys5 is below the gate, so the default AutoTolerance is a no-op (exact);
    # an explicit Float64 tol is an absolute cutoff. Large-system sparsification
    # is covered by the "bus-count gate" testset and demonstrated in _dev.jl.
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")

    vl_num = VirtualLODF(sys; tol = 1e-3)
    @test PNM.get_tol(vl_num) == 1e-3
    @test vl_num.tol isa PNM.AbsoluteCutoff
    vl_auto = VirtualLODF(sys)
    @test vl_auto.tol isa PNM.AbsoluteCutoff
    @test PNM.get_tol(vl_auto) == eps()

    vm_num = VirtualMODF(sys; tol = 1e-3)
    @test vm_num.tol isa PNM.AbsoluteCutoff
    vm_auto = VirtualMODF(sys)
    @test vm_auto.tol isa PNM.AbsoluteCutoff
    @test PNM.get_tol(vm_auto) == eps()
end

@testset "dense PTDF AutoTolerance is a no-op across solvers" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    A = IncidenceMatrix(sys)
    BA = BA_Matrix(sys)
    # The dense PTDF stays exact under AutoTolerance regardless of solver: the
    # result equals the eps() build and is a dense DC_PTDF_Matrix.
    for solver in ("KLU", "Dense")
        p_auto =
            PTDF(
                A,
                BA;
                linear_solver = solver,
                tol = AutoTolerance(; data_precision = 1e-3),
            )
        p_eps = PTDF(A, BA; linear_solver = solver, tol = eps())
        @test PNM.get_tol(p_auto) == eps()
        @test isapprox(
            get_ptdf_data(p_auto),
            get_ptdf_data(p_eps);
            atol = NO_OP_BUILD_ATOL,
        )
        @test p_auto isa PNM.DC_PTDF_Matrix
    end
end
