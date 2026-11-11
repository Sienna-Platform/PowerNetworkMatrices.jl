using Random: MersenneTwister
import SparseArrays

@testset "AutoTolerance type construction" begin
    t = PMC.AutoTolerance()
    @test t.data_precision === :auto          # default discovers precision
    @test t.safety == 1.0
    @test t.quantile == 0.5
    @test PMC.AutoTolerance(; data_precision = 1e-3).data_precision == 1e-3
    @test PMC.AutoTolerance(; data_precision = :auto).data_precision === :auto
    @test PMC.AutoTolerance(; safety = 5.0).safety == 5.0
    @test_throws ArgumentError PMC.AutoTolerance(; data_precision = :nonsense)
    @test_throws ArgumentError PMC.AutoTolerance(; data_precision = NaN)
    @test_throws ArgumentError PMC.AutoTolerance(; data_precision = -1e-3)
    @test_throws ArgumentError PMC.AutoTolerance(; safety = 0.0)
    @test_throws ArgumentError PMC.AutoTolerance(; quantile = 1.5)
    # The module-wide default is an auto-discovering AutoTolerance.
    @test PMC.DEFAULT_AUTO_TOLERANCE isa PMC.AutoTolerance
    @test PMC.DEFAULT_AUTO_TOLERANCE.data_precision === :auto
end

@testset "significant-figure counting" begin
    @test PMC._sigfigs(0.0, 12, 1e-9) == 0
    @test PMC._sigfigs(0.25, 12, 1e-9) == 2     # 2.5e-1
    @test PMC._sigfigs(0.1234, 12, 1e-9) == 4
    # 1/3 = 0.333... reproduces to within rtol=1e-9 at 9 sig figs.
    @test PMC._sigfigs(1.0 / 3.0, 12, 1e-9) == 9
end

@testset "discover_data_precision accuracy" begin
    # Reactances rounded to s=4 sig figs -> delta ≈ 0.5e-3.
    x = round.(0.01 .+ 0.5 .* rand(MersenneTwister(1), 200); sigdigits = 4)
    b = inv.(x)
    @test isapprox(PMC.discover_data_precision(b), 0.5e-3; rtol = 0.5)
    # Full-precision reactances -> collapses toward the floor (<= 1e-2 clamp, tiny).
    bf = inv.(0.01 .+ 0.5 .* rand(MersenneTwister(2), 200))
    @test PMC.discover_data_precision(bf) <= 1e-6
    # Type stability.
    @test (@inferred PMC.discover_data_precision(b)) isa Float64
end

@testset "relative-cutoff fraction + clamping" begin
    # Explicit data_precision drives the fraction directly (safety = 1).
    x = round.(0.01 .+ 0.5 .* rand(MersenneTwister(3), 50); sigdigits = 4)
    b = inv.(x)
    spec = PMC.AutoTolerance(; data_precision = 1e-3)
    @test PMC._relative_alpha(spec, b) == 1e-3
    # safety scales the fraction.
    @test PMC._relative_alpha(
        PMC.AutoTolerance(; data_precision = 1e-3, safety = 5.0),
        b,
    ) ==
          5e-3
    # Clamped to [1e-6, 1e-2]: tiny precision floors, huge precision caps.
    @test PMC._relative_alpha(PMC.AutoTolerance(; data_precision = 1e-12), b) ==
          PMC.MIN_RELATIVE_TOLERANCE
    @test PMC._relative_alpha(PMC.AutoTolerance(; data_precision = 1.0), b) ==
          PMC.MAX_RELATIVE_TOLERANCE
    @test (@inferred PMC._relative_alpha(spec, b)) isa Float64
end

@testset "cutoff dispatch: absolute vs relative" begin
    row = [1.0, 0.5, 1e-3, 1e-7, -0.2]
    # Absolute cutoff drops below a fixed value.
    abs_cut = PMC.AbsoluteCutoff(1e-2)
    @test PMC.cutoff_value(abs_cut) == 1e-2
    sa = PMC.apply_cutoff(abs_cut, copy(row))
    @test count(!iszero, sa) == 3                 # 1.0, 0.5, -0.2 survive
    # Relative cutoff drops below fraction * max|row| (= 1e-2 * 1.0 here).
    rel_cut = PMC.RelativeCutoff(1e-2)
    @test PMC.cutoff_value(rel_cut) == 1e-2
    sr = PMC.apply_cutoff(rel_cut, copy(row))
    @test count(!iszero, sr) == 3
    # A relative cutoff is scale-invariant: scaling the row keeps the same pattern.
    sr_scaled = PMC.apply_cutoff(rel_cut, 1000.0 .* row)
    @test count(!iszero, sr_scaled) == count(!iszero, sr)
    # A cutoff at/below eps() keeps the dense row untouched (no sparsification).
    kept = PMC.apply_cutoff(PMC.AbsoluteCutoff(0.0), copy(row))
    @test kept == row
end

@testset "sparsify (vector) keeps |x| > tol, matches sparsevec+droptol!" begin
    # The fused count-then-fill must equal the reference sparsevec + droptol!.
    ref(v, tol) = (s = SparseArrays.sparsevec(v); SparseArrays.droptol!(s, tol); s)
    rng = MersenneTwister(7)
    for n in (1, 4, 500), tol in (0.0, eps(), 1e-6, 1e-3, 0.5, 10.0)
        v = randn(rng, n)
        v[1:min(n, 2)] .= 1e-12                       # guarantee some droppable entries
        s = PMC.sparsify(copy(v), tol)
        r = ref(copy(v), tol)
        @test s isa SparseArrays.SparseVector{Float64, Int}
        @test s.n == length(v)
        @test s.nzind == r.nzind                      # sorted indices, same survivors
        @test s.nzval == r.nzval
        @test all(abs(x) > tol for x in s.nzval)      # the documented contract
    end
    # Edge cases: all kept, all dropped.
    @test SparseArrays.nnz(PMC.sparsify(fill(1.0, 10), 1e-6)) == 10
    @test SparseArrays.nnz(PMC.sparsify(fill(1e-9, 10), 1e-6)) == 0
end

# Build a grounded path-graph Laplacian of size n: a strictly diagonally dominant
# (hence SPD, nonsingular) sparse tridiagonal, used to exercise the bus-count
# gate without materializing a real large system.
function _synthetic_aba(n::Int)
    return SparseArrays.spdiagm(
        -1 => fill(-1.0, n - 1),
        0 => fill(2.1, n),
        1 => fill(-1.0, n - 1),
    )
end

@testset "virtual AutoTolerance: bus-count gate" begin
    # Below the gate -> no-op (exact rows), so small systems are never sparsified.
    small = _synthetic_aba(64)
    cache_s = PMC.klu_factorize(small)
    cut_s = PMC._resolve_virtual_cutoff(PMC.AutoTolerance(), cache_s, small, [1.0])
    @test cut_s isa PMC.AbsoluteCutoff
    @test PMC.cutoff_value(cut_s) == eps()

    # At/above the gate -> relative cutoff driven by data precision.
    big = _synthetic_aba(PMC.AUTO_TOLERANCE_BUS_LIMIT)
    cache_b = PMC.klu_factorize(big)
    cut_b = PMC._resolve_virtual_cutoff(
        PMC.AutoTolerance(; data_precision = 1e-3),
        cache_b,
        big,
        [1.0],
    )
    @test cut_b isa PMC.RelativeCutoff
    @test PMC.cutoff_value(cut_b) == 1e-3

    # A Float64 tol is an absolute cutoff at any size (backward compatible).
    @test PMC._resolve_virtual_cutoff(1e-2, cache_s, small, [1.0]) ==
          PMC.AbsoluteCutoff(1e-2)
end
