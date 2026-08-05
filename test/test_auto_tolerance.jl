# Several testsets below assert that AutoTolerance is a no-op by comparing a matrix built with
# it against one built at eps(). Two separate builds are not bitwise reproducible -- the
# AppleAccelerate backend drifts by a ULP on roughly 1 run in 20 -- so these compare with a
# tolerance. 1e-10 still proves the no-op: genuinely sparsifying would zero entries, moving
# them by their full magnitude (>=1e-3 in these systems).
const NO_OP_BUILD_ATOL = 1e-10

@testset "AutoTolerance type construction" begin
    t = AutoTolerance()
    @test t.data_precision === :auto          # default discovers precision
    @test t.safety == 1.0
    @test t.quantile == 0.5
    @test AutoTolerance(; data_precision = 1e-3).data_precision == 1e-3
    @test AutoTolerance(; data_precision = :auto).data_precision === :auto
    @test AutoTolerance(; safety = 5.0).safety == 5.0
    @test_throws ArgumentError AutoTolerance(; data_precision = :nonsense)
    # The module-wide default is an auto-discovering AutoTolerance.
    @test PNM.DEFAULT_AUTO_TOLERANCE isa AutoTolerance
    @test PNM.DEFAULT_AUTO_TOLERANCE.data_precision === :auto
end

@testset "significant-figure counting" begin
    @test PNM._sigfigs(0.0, 12, 1e-9) == 0
    @test PNM._sigfigs(0.25, 12, 1e-9) == 2     # 2.5e-1
    @test PNM._sigfigs(0.1234, 12, 1e-9) == 4
    # 1/3 = 0.333... reproduces to within rtol=1e-9 at 9 sig figs.
    @test PNM._sigfigs(1.0 / 3.0, 12, 1e-9) == 9
end

@testset "discover_data_precision accuracy" begin
    # Reactances rounded to s=4 sig figs -> delta ≈ 0.5e-3.
    x = round.(0.01 .+ 0.5 .* rand(MersenneTwister(1), 200); sigdigits = 4)
    b = inv.(x)
    @test isapprox(PNM.discover_data_precision(b), 0.5e-3; rtol = 0.5)
    # Full-precision reactances -> collapses toward the floor (<= 1e-2 clamp, tiny).
    bf = inv.(0.01 .+ 0.5 .* rand(MersenneTwister(2), 200))
    @test PNM.discover_data_precision(bf) <= 1e-6
    # Type stability.
    @test (@inferred PNM.discover_data_precision(b)) isa Float64
end

@testset "relative-cutoff fraction + clamping" begin
    # Explicit data_precision drives the fraction directly (safety = 1).
    x = round.(0.01 .+ 0.5 .* rand(MersenneTwister(3), 50); sigdigits = 4)
    b = inv.(x)
    spec = AutoTolerance(; data_precision = 1e-3)
    @test PNM._relative_alpha(spec, b) == 1e-3
    # safety scales the fraction.
    @test PNM._relative_alpha(AutoTolerance(; data_precision = 1e-3, safety = 5.0), b) ==
          5e-3
    # Clamped to [1e-6, 1e-2]: tiny precision floors, huge precision caps.
    @test PNM._relative_alpha(AutoTolerance(; data_precision = 1e-12), b) ==
          PNM.MIN_RELATIVE_TOLERANCE
    @test PNM._relative_alpha(AutoTolerance(; data_precision = 1.0), b) ==
          PNM.MAX_RELATIVE_TOLERANCE
    @test (@inferred PNM._relative_alpha(spec, b)) isa Float64
end

@testset "cutoff dispatch: absolute vs relative" begin
    row = [1.0, 0.5, 1e-3, 1e-7, -0.2]
    # Absolute cutoff drops below a fixed value.
    abs_cut = PNM.AbsoluteCutoff(1e-2)
    @test PNM.cutoff_value(abs_cut) == 1e-2
    sa = PNM.apply_cutoff(abs_cut, copy(row))
    @test count(!iszero, sa) == 3                 # 1.0, 0.5, -0.2 survive
    # Relative cutoff drops below fraction * max|row| (= 1e-2 * 1.0 here).
    rel_cut = PNM.RelativeCutoff(1e-2)
    @test PNM.cutoff_value(rel_cut) == 1e-2
    sr = PNM.apply_cutoff(rel_cut, copy(row))
    @test count(!iszero, sr) == 3
    # A relative cutoff is scale-invariant: scaling the row keeps the same pattern.
    sr_scaled = PNM.apply_cutoff(rel_cut, 1000.0 .* row)
    @test count(!iszero, sr_scaled) == count(!iszero, sr)
    # A cutoff at/below eps() keeps the dense row untouched (no sparsification).
    kept = PNM.apply_cutoff(PNM.AbsoluteCutoff(0.0), copy(row))
    @test kept == row
end

@testset "sparsify (vector) keeps |x| > tol, matches sparsevec+droptol!" begin
    # The fused count-then-fill must equal the reference sparsevec + droptol!.
    ref(v, tol) = (s = PNM.SparseArrays.sparsevec(v); PNM.SparseArrays.droptol!(s, tol); s)
    rng = MersenneTwister(7)
    for n in (1, 4, 500), tol in (0.0, eps(), 1e-6, 1e-3, 0.5, 10.0)
        v = randn(rng, n)
        v[1:min(n, 2)] .= 1e-12                       # guarantee some droppable entries
        s = PNM.sparsify(copy(v), tol)
        r = ref(copy(v), tol)
        @test s isa PNM.SparseArrays.SparseVector{Float64, Int}
        @test s.n == length(v)
        @test s.nzind == r.nzind                      # sorted indices, same survivors
        @test s.nzval == r.nzval
        @test all(abs(x) > tol for x in s.nzval)      # the documented contract
    end
    # Edge cases: all kept, all dropped.
    @test PNM.SparseArrays.nnz(PNM.sparsify(fill(1.0, 10), 1e-6)) == 10
    @test PNM.SparseArrays.nnz(PNM.sparsify(fill(1e-9, 10), 1e-6)) == 0
end

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
    @test PNM.get_tol(p_num)[] == 1e-3
    p_eps = PTDF(A, BA; tol = eps())
    @test PNM.get_tol(p_eps)[] == eps()

    # The dense PTDF is the small-system path: AutoTolerance is a NO-OP there. It
    # stays an exact, dense `Matrix{Float64}` (so `DC_PTDF_Matrix` and downstream
    # dispatch are preserved); sparsification is reserved for VirtualPTDF.
    p_auto = PTDF(A, BA; tol = AutoTolerance(; data_precision = 1e-2))
    @test PNM.get_tol(p_auto)[] == eps()
    # The AppleAccelerate LU backend is not bit-reproducible run-to-run (threaded
    # libSparse solve; observed ~1 ULP diffs under CPU contention), so an exact
    # `==` between two independent dense builds is invalid. A real sparsification
    # regression (AutoTolerance no longer a no-op) would drop entries at the
    # `data_precision = 1e-2` scale, many orders above this tolerance.
    @test isapprox(get_ptdf_data(p_auto), get_ptdf_data(p_eps); atol = 1e-9, rtol = 1e-9)
    # Dense storage is preserved -> matches the DC_PTDF_Matrix alias (the data
    # field is a Matrix{Float64}, so downstream dispatch is unaffected).
    @test p_auto isa PNM.DC_PTDF_Matrix
    @test PTDF(A, BA) isa PNM.DC_PTDF_Matrix
end

# Build a grounded path-graph Laplacian of size n: a strictly diagonally dominant
# (hence SPD, nonsingular) sparse tridiagonal, used to exercise the bus-count
# gate without materializing a real large system.
function _synthetic_aba(n::Int)
    return PNM.SparseArrays.spdiagm(
        -1 => fill(-1.0, n - 1),
        0 => fill(2.1, n),
        1 => fill(-1.0, n - 1),
    )
end

@testset "virtual AutoTolerance: bus-count gate" begin
    # Below the gate -> no-op (exact rows), so small systems are never sparsified.
    small = _synthetic_aba(64)
    cache_s = PNM.klu_factorize(small)
    cut_s = PNM._resolve_virtual_cutoff(AutoTolerance(), cache_s, small, [1.0])
    @test cut_s isa PNM.AbsoluteCutoff
    @test PNM.cutoff_value(cut_s) == eps()

    # At/above the gate -> relative cutoff driven by data precision.
    big = _synthetic_aba(PNM.AUTO_TOLERANCE_BUS_LIMIT)
    cache_b = PNM.klu_factorize(big)
    cut_b = PNM._resolve_virtual_cutoff(
        AutoTolerance(; data_precision = 1e-3),
        cache_b,
        big,
        [1.0],
    )
    @test cut_b isa PNM.RelativeCutoff
    @test PNM.cutoff_value(cut_b) == 1e-3

    # A Float64 tol is an absolute cutoff at any size (backward compatible).
    @test PNM._resolve_virtual_cutoff(1e-2, cache_s, small, [1.0]) ==
          PNM.AbsoluteCutoff(1e-2)
end

@testset "VirtualPTDF small system: AutoTolerance is a no-op; Float64 sparsifies" begin
    # ACTIVSg2000 (1999 buses) is below the gate, so the default AutoTolerance is
    # exact; a Float64 tol still sparsifies columns at any size.
    sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")

    v_exact = VirtualPTDF(sys; tol = eps())
    arc = first(PNM.get_arc_axis(v_exact))
    nnz_dense = count(!iszero, v_exact[arc, :])

    v_auto = VirtualPTDF(sys)                           # default, below gate
    @test v_auto.tol isa PNM.AbsoluteCutoff
    @test PNM.get_tol(v_auto) == eps()
    # No entries dropped (AutoTolerance is a no-op below the gate). The two VirtualPTDFs are
    # independently KLU-factorized, so kept values agree only to floating-point tolerance, not
    # bit-for-bit — `==` here was flaky (ULP-level, run-to-run nondeterministic). The `count`
    # check below is what actually verifies nothing was sparsified away.
    @test isapprox(v_auto[arc, :], v_exact[arc, :]; atol = 1e-9)
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
    @test PNM.get_tol(l_num)[] == 1e-3

    # The dense LODF is the small-system path: AutoTolerance is a no-op (eps), so
    # the result is identical to the exact LODF. Sparsification is a VirtualLODF
    # feature; pass a Float64 tol to sparsify a dense LODF explicitly.
    l_exact = LODF(A, ABA, BA; tol = eps())
    l_auto = LODF(A, ABA, BA; tol = AutoTolerance(; data_precision = 1e-2))
    @test PNM.get_tol(l_auto)[] == eps()
    @test isapprox(l_auto.data, l_exact.data; atol = NO_OP_BUILD_ATOL)

    # System constructor: AutoTolerance routes through the ABA path and is a no-op.
    l_sys_auto = LODF(sys; tol = AutoTolerance(; data_precision = 1e-2))
    @test PNM.get_tol(l_sys_auto)[] == eps()

    # from-PTDF constructor: AutoTolerance is a no-op (eps), so the default
    # LODF(A, ptdf) keeps working and stays exact.
    P = PTDF(A, BA; tol = eps())
    l_from_ptdf = LODF(A, P; tol = AutoTolerance())
    @test PNM.get_tol(l_from_ptdf)[] == eps()
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
        @test PNM.get_tol(p_auto)[] == eps()
        @test isapprox(
            get_ptdf_data(p_auto),
            get_ptdf_data(p_eps);
            atol = NO_OP_BUILD_ATOL,
        )
        @test p_auto isa PNM.DC_PTDF_Matrix
    end
end
