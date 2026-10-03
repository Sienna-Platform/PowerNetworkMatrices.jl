import SparseArrays
import SparseArrays: SparseMatrixCSC, nonzeros, rowvals, getcolptr, nnz
import LinearAlgebra: norm
import Random

const KW = PNM.KLUWrapper

function _lean_convection_diffusion(m; c = 0.3)
    T = SparseArrays.spdiagm(
        0 => fill(2.0, m), 1 => fill(-1.0 + c, m - 1), -1 => fill(-1.0 - c, m - 1),
    )
    Im = SparseArrays.sparse(1.0 * I, m, m)
    return kron(Im, T) + kron(T, Im)
end

# Full 2n polar Jacobian [dP/dθ dP/d|V|; dQ/dθ dQ/d|V|] of `Y` at `V`, with a pattern that
# does not depend on `V` (explicit zeros kept) and bus 1's rows replaced by identity rows.
function _lean_polar_jacobian(Y::SparseMatrixCSC, V::Vector{ComplexF64})
    n = size(Y, 1)
    Ibus = Y * V
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    function push4!(i, j, da, dm)
        append!(rows, (i, n + i, i, n + i))
        append!(cols, (j, j, n + j, n + j))
        append!(vals, (real(da), imag(da), real(dm), imag(dm)))
        return
    end
    for j in 1:n, e in SparseArrays.nzrange(Y, j)
        i = rowvals(Y)[e]
        y = ComplexF64(nonzeros(Y)[e])
        push4!(i, j, -im * V[i] * conj(y * V[j]), V[i] * conj(y * V[j] / abs(V[j])))
    end
    for i in 1:n
        push4!(i, i, im * V[i] * conj(Ibus[i]), conj(Ibus[i]) * V[i] / abs(V[i]))
    end
    J = SparseArrays.sparse(rows, cols, vals, 2n, 2n)
    for e in eachindex(rowvals(J))
        r = rowvals(J)[e]
        if r == 1 || r == n + 1
            J.nzval[e] = 0.0
        end
    end
    J[1, 1] = 1.0
    J[n + 1, n + 1] = 1.0
    return J
end

function _lean_jacobians(sys; nsteps = 4, seed = 1)
    Y = PNM.Ybus(sys).data
    n = size(Y, 1)
    rng = Random.Xoshiro(seed)
    J0 = _lean_polar_jacobian(Y, ones(ComplexF64, n))
    steps = [
        _lean_polar_jacobian(
            Y, (1.0 .+ 0.03 .* randn(rng, n)) .* cis.(0.1 .* randn(rng, n)),
        ) for _ in 1:nsteps
    ]
    @assert all(J -> getcolptr(J) == getcolptr(J0) && rowvals(J) == rowvals(J0), steps)
    return J0, steps
end

function _lean_perturbed(A, rng; scale = 0.1)
    B = copy(A)
    nonzeros(B) .*= 1.0 .+ scale .* (rand(rng, nnz(B)) .- 0.5)
    return B
end

# KLU (Int32, btf off) factor of A0, then klu_refactor on A1 with A0's pivot order:
# the reference the lean kernel reproduces. Returns the unscaled factors as sparse
# matrices in pivot space, plus a klu_solve of b.
function _klu_static_reference(A0, A1, b)
    n = size(A0, 1)
    ap = Int32.(getcolptr(A0) .- 1)
    ai = Int32.(rowvals(A0) .- 1)
    common = Ref(KW.KluCommon())
    KW.klu_defaults!(common)
    common[].btf = Cint(0)
    sym = KW.klu_analyze(Int32(n), pointer(ap), pointer(ai), common)
    x0 = copy(nonzeros(A0))
    num = KW.klu_factor(pointer(ap), pointer(ai), pointer(x0), sym, common)
    x1 = copy(nonzeros(A1))
    @assert KW.klu_refactor(pointer(ap), pointer(ai), pointer(x1), sym, num, common) == 1
    head = unsafe_load(Ptr{KW.KluNumericHead{Cint}}(num))
    lp = Vector{Cint}(undef, n + 1)
    li = Vector{Cint}(undef, head.lnz)
    lx = Vector{Float64}(undef, head.lnz)
    up = Vector{Cint}(undef, n + 1)
    ui = Vector{Cint}(undef, head.unz)
    ux = Vector{Float64}(undef, head.unz)
    rs = Vector{Float64}(undef, n)
    p = Vector{Cint}(undef, n)
    q = Vector{Cint}(undef, n)
    @assert KW.klu_extract(
        num, sym, pointer(lp), pointer(li), pointer(lx), pointer(up), pointer(ui),
        pointer(ux), Ptr{Cint}(C_NULL), Ptr{Cint}(C_NULL), Ptr{Cdouble}(C_NULL),
        pointer(p), pointer(q), pointer(rs), Ptr{Cint}(C_NULL), common,
    ) == 1
    x = copy(b)
    @assert KW.klu_solve(sym, num, Cint(n), Cint(1), pointer(x), common) == 1
    KW.klu_free_numeric!(Ref(num), common)
    KW.klu_free_symbolic!(Ref(sym), common)
    L = SparseMatrixCSC(n, n, Int.(lp) .+ 1, Int.(li) .+ 1, lx)
    U = SparseMatrixCSC(n, n, Int.(up) .+ 1, Int.(ui) .+ 1, ux)
    # KLU factors the row-scaled matrix; the unscaled factors on the same order are
    # (D L D⁻¹, D U), with D the row scales in pivot order.
    D = SparseArrays.spdiagm(0 => rs)
    Dinv = SparseArrays.spdiagm(0 => 1.0 ./ rs)
    return (L = D * L * Dinv, U = D * U, x = x, p = Int.(p) .+ 1, q = Int.(q) .+ 1)
end

function _lean_factors(plan, vals)
    n = Int(plan.n)
    I = Int[]
    J = Int[]
    lv = Float64[]
    uI = Int[]
    uJ = Int[]
    uv = Float64[]
    for k in 1:n
        push!(I, k)
        push!(J, k)
        push!(lv, 1.0)
        for s in plan.cp[k]:(plan.cp[k + 1] - 1)
            if s <= plan.dpos[k]
                push!(uI, plan.row[s])
                push!(uJ, k)
                push!(uv, vals.lu[s])
            else
                push!(I, plan.row[s])
                push!(J, k)
                push!(lv, vals.lu[s])
            end
        end
    end
    return SparseArrays.sparse(I, J, lv, n, n), SparseArrays.sparse(uI, uJ, uv, n, n)
end

_lean_plan_hash(plan) = hash(Tuple(getfield(plan, f) for f in fieldnames(typeof(plan))))

_lean_refactor_alloc(vals, plan, x) = @allocated KW.lean_refactor!(vals, plan, x)
_lean_solve_alloc(b, plan, vals) = @allocated KW.lean_solve!(b, plan, vals)

const _LEAN_SYS14 = PSB.build_system(PSB.PSITestSystems, "c_sys14")
const _LEAN_SYS2000 =
    PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")

@testset "_factor_stats" begin
    J0, _ = _lean_jacobians(_LEAN_SYS14; nsteps = 1)
    for A in (J0, _lean_convection_diffusion(30))
        n = size(A, 1)
        stats = KW._factor_stats(PNM.klu_factorize(A))
        @test stats.lnz >= n && stats.unz >= n
        @test stats.flops > 0
        @test stats.nblocks >= 1
        # With BTF off the counts must match the lean plan's flat pattern.
        cache = PNM.KLULinSolveCache(A)
        cache.common[].btf = Cint(0)
        PNM.full_factor!(cache, A)
        flat = KW._factor_stats(cache)
        @test flat.nblocks == 1 && iszero(flat.nzoff)
        @test flat.lnz + flat.unz - n == KW._lean_nnz(KW.build_lean_plan(A))
    end
    A32 = SparseMatrixCSC{Float64, Int32}(_lean_convection_diffusion(10))
    @test KW._factor_stats(PNM.klu_factorize(A32)).flops > 0
    @test_throws ErrorException KW._factor_stats(PNM.KLULinSolveCache(A32))
end

@testset "Lean LU matches KLU on its frozen order" begin
    J0, steps = _lean_jacobians(_LEAN_SYS2000)
    rng = Random.Xoshiro(7)
    cd = _lean_convection_diffusion(60)
    # KLU factors the row-scaled matrix and the lean kernel the unscaled one, so their
    # roundings differ; element growth (|L| up to ~500 on the flat-start order of the
    # perturbed Jacobians) amplifies that to ~1e-10 in the factors and ~1e-11 in the
    # solves. The lean backward error is no worse than KLU's.
    cases = [
        ("ACTIVSg2000 J", J0, steps, 1e-9, 1e-11),
        ("convection-diffusion", cd, [_lean_perturbed(cd, rng) for _ in 1:4], 1e-12, 1e-12),
    ]
    for (label, A0, seq, ftol, xtol) in cases
        @testset "$label" begin
            plan = KW.build_lean_plan(A0)
            vals = KW.LeanLUValues(plan)
            @test plan.rcond0 > 0
            for A1 in seq
                b = randn(rng, size(A0, 1))
                ratio = KW.lean_refactor!(vals, plan, nonzeros(A1))
                @test ratio >= KW.LEAN_REJECT_RATIO * plan.rcond0
                @test all(iszero, vals.work)
                ref = _klu_static_reference(A0, A1, b)
                @test ref.p == plan.p && ref.q == plan.q
                L, U = _lean_factors(plan, vals)
                @test norm(L - ref.L) <= ftol * norm(ref.L)
                @test norm(U - ref.U) <= ftol * norm(ref.U)
                Ap = A1[plan.p, plan.q]
                @test norm(Ap - L * U) <= 1e-13 * norm(Ap)
                @test norm(Ap - L * U) <= 2 * norm(Ap - ref.L * ref.U)
                x = KW.lean_solve!(copy(b), plan, vals)
                @test norm(x - ref.x) <= xtol * norm(ref.x)
                @test norm(A1 * x - b) <= 1e-9 * norm(b)
            end
            A1 = seq[end]
            b = randn(rng, size(A0, 1))
            @test iszero(_lean_refactor_alloc(vals, plan, nonzeros(A1)))
            @test iszero(_lean_solve_alloc(b, plan, vals))
        end
    end
end

@testset "Lean LU rejects zero and non-finite pivots" begin
    A = _lean_convection_diffusion(8)
    plan = KW.build_lean_plan(A)
    vals = KW.LeanLUValues(plan)
    @test iszero(KW.lean_refactor!(vals, plan, zeros(nnz(A))))
    @test all(iszero, vals.work)
    bad = copy(nonzeros(A))
    bad[1] = NaN
    @test iszero(KW.lean_refactor!(vals, plan, bad))
    @test all(iszero, vals.work)
    bad[1] = Inf
    @test iszero(KW.lean_refactor!(vals, plan, bad))
    @test KW.lean_refactor!(vals, plan, nonzeros(A)) ≈ plan.rcond0
    @test_throws DimensionMismatch KW.lean_refactor!(vals, plan, zeros(3))
    @test_throws PNM.LinearAlgebra.SingularException KW.build_lean_plan(
        SparseArrays.spzeros(3, 3),
    )
end

# Structurally nonsymmetric, diagonally dominant, irreducible (has a full cycle).
function _lean_random_matrix(rng, n, ::Type{Ti}) where {Ti}
    A =
        SparseArrays.sprand(rng, n, n, 4 / n) +
        SparseArrays.spdiagm(1 => ones(n - 1), -(n - 1) => ones(1)) +
        SparseArrays.spdiagm(0 => fill(10.0, n))
    return SparseMatrixCSC{Float64, Ti}(A)
end

@testset "Lean LU refactor and solve match KLU ($Ti indices)" for Ti in (Int32, Int64)
    rng = Random.MersenneTwister(1)
    n = 300
    A = _lean_random_matrix(rng, n, Ti)
    plan = KW.build_lean_plan(A)
    vals = KW.LeanLUValues(plan)
    cache = PNM.KLULinSolveCache(A)
    PNM.full_factor!(cache, A)
    b = randn(rng, n)
    for _ in 1:5
        A2 = copy(A)
        nonzeros(A2) .*= 1 .+ 0.2 .* randn(rng, nnz(A2))
        @test KW.lean_refactor!(vals, plan, nonzeros(A2)) >= 1e-3
        x = KW.lean_solve!(copy(b), plan, vals)
        PNM.numeric_refactor!(cache, A2)
        @test x ≈ PNM.solve!(cache, copy(b)) rtol = 1e-10
        @test all(iszero, vals.work)
    end
    @test_throws DimensionMismatch KW.lean_solve!(zeros(3), plan, vals)
end

@testset "Lean LU on reducible and block-diagonal matrices" begin
    U = SparseArrays.spdiagm(0 => fill(2.0, 5), 1 => ones(4))
    planU = KW.build_lean_plan(U)
    valsU = KW.LeanLUValues(planU)
    KW.lean_refactor!(valsU, planU, nonzeros(U))
    @test KW.lean_solve!(ones(5), planU, valsU) ≈ U \ ones(5)
    A = _lean_random_matrix(Random.MersenneTwister(3), 50, Int64)
    D = SparseArrays.blockdiag(A, A)
    planD = KW.build_lean_plan(D)
    valsD = KW.LeanLUValues(planD)
    KW.lean_refactor!(valsD, planD, nonzeros(D))
    @test KW.lean_solve!(ones(100), planD, valsD) ≈ D \ ones(100)
end

_counts(c) = Tuple(KW.lean_counts(c))

@testset "KLULinSolveCache lean path and fallback" begin
    A0 = SparseArrays.sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    plan = KW.build_lean_plan(A0)
    cache = PNM.KLULinSolveCache(A0)
    @test !KW.has_lean_plan(cache)
    KW.set_lean_plan!(cache, plan)
    @test KW.has_lean_plan(cache)
    PNM.full_factor!(cache, A0)
    @test cache.lean_active && cache.numeric == C_NULL
    @test PNM.is_factored(cache)
    b = [1.0, 2.0, 3.0]
    @test PNM.solve!(cache, copy(b)) ≈ Matrix(A0) \ b
    B = [b 2b]
    @test PNM.solve!(cache, copy(B)) ≈ Matrix(A0) \ B
    @test_throws ErrorException PNM.tsolve!(cache, copy(b))
    @test_throws ErrorException PNM.condest!(cache, A0)
    @test _counts(cache) == (1, 0, 0, 0)

    # A first pivot that collapses on the frozen order falls back to a pivoting klu_factor.
    A1 = copy(A0)
    A1[plan.p[1], plan.q[1]] = 1e-14
    PNM.numeric_refactor!(cache, A1)
    @test !cache.lean_active && cache.numeric != C_NULL
    @test _counts(cache) == (2, 1, 0, 0)
    @test PNM.solve!(cache, copy(b)) ≈ Matrix(A1) \ b
    @test PNM.tsolve!(cache, copy(b)) ≈ transpose(Matrix(A1)) \ b

    # Back on the lean path for well-conditioned values; plan survives a same-pattern
    # re-analysis and is shared by reference.
    PNM.numeric_refactor!(cache, A0)
    @test cache.lean_active
    # Re-setting the plan must not expose the stale KLU factors of A1.
    KW.set_lean_plan!(cache, plan)
    @test cache.numeric == C_NULL && !PNM.is_factored(cache)
    @test_throws ErrorException PNM.solve!(cache, copy(b))
    PNM.numeric_refactor!(cache, A0)
    @test cache.lean_active
    @test_throws ErrorException KW._factor_stats(cache)
    PNM.symbolic_factor!(cache, A0)
    @test KW.has_lean_plan(cache) && !cache.lean_active && !PNM.is_factored(cache)
    other = PNM.KLULinSolveCache(A0)
    KW.share_lean_plan!(other, cache)
    @test other.lean_plan === cache.lean_plan
    @test other.lean_vals !== cache.lean_vals
    @test_throws ArgumentError KW.share_lean_plan!(cache, PNM.KLULinSolveCache(A0))

    A2 = SparseArrays.sparse([4.0 1.0 1.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    @test_throws ArgumentError KW.set_lean_plan!(PNM.KLULinSolveCache(A2), plan)
    PNM.symbolic_factor!(other, A2)
    @test !KW.has_lean_plan(other)
end

@testset "KLULinSolveCache lean drop, pivoted factor and pause" begin
    A0 = SparseArrays.sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    plan = KW.build_lean_plan(A0)
    cache = PNM.KLULinSolveCache(A0)
    KW.set_lean_plan!(cache, plan)
    PNM.full_factor!(cache, A0)
    @test KW.lean_active(cache)

    KW.drop_numeric!(cache)
    @test !KW.lean_active(cache) && !PNM.is_factored(cache)
    @test KW.has_lean_plan(cache)
    PNM.numeric_refactor!(cache, A0)
    @test KW.lean_active(cache)

    # Pivoted: KLU's own factors, so tsolve!/condest! work; the plan stays for later.
    KW.pivoted_factor!(cache, A0)
    @test !KW.lean_active(cache) && cache.numeric != C_NULL
    @test PNM.tsolve!(cache, copy(b)) ≈ transpose(Matrix(A0)) \ b
    @test isfinite(PNM.condest!(cache, A0))
    @test KW.has_lean_plan(cache)
    @test _counts(cache) == (2, 0, 0, 0)

    # Paused: refactors stay on KLU (a lean factorization in place is dropped first).
    PNM.numeric_refactor!(cache, A0)
    @test KW.lean_active(cache)
    KW.pause_lean!(cache, true)
    @test !KW.lean_active(cache) && cache.numeric == C_NULL
    PNM.numeric_refactor!(cache, A0)
    @test !KW.lean_active(cache) && cache.numeric != C_NULL
    PNM.numeric_refactor!(cache, A0)
    @test _counts(cache) == (3, 0, 0, 0)
    @test PNM.solve!(cache, copy(b)) ≈ Matrix(A0) \ b
    KW.pause_lean!(cache, false)
    PNM.numeric_refactor!(cache, A0)
    @test KW.lean_active(cache)
    @test _counts(cache) == (4, 0, 0, 0)
    KW.pause_lean!(cache, true)
    KW.set_lean_plan!(cache, plan)
    PNM.numeric_refactor!(cache, A0)
    @test KW.lean_active(cache)
end

# Every column of a dense 4x4 pattern, explicit zeros kept.
_dense_pattern(M) =
    SparseArrays.sparse(repeat(1:4, 4), repeat(1:4; inner = 4), vec(M), 4, 4)

@testset "KLULinSolveCache lean column swap" begin
    # Columns 1-2 are a power-flow bus: PV (Q, θ) in the plan, REF (P, Q) after. The plan pivots
    # the PV Q column on row 2 (its only entry), so the REF columns put an exact zero there.
    pv = [0.0 10.0 1.0 -10.0; -1.0 2.0 -3.0 1.0; 0.0 -10.0 2.0 12.0; 0.0 -2.0 9.0 -1.0]
    ref = copy(pv)
    ref[:, 1] = [-1.0, 0.0, 0.0, 0.0]
    ref[:, 2] = [0.0, -1.0, 0.0, 0.0]
    A0 = _dense_pattern(pv)
    A1 = _dense_pattern(ref)
    plan = KW.build_lean_plan(A0)
    cache = PNM.KLULinSolveCache(A0)
    KW.set_lean_plan!(cache, plan)
    PNM.symbolic_factor!(cache, A0)
    PNM.numeric_refactor!(cache, A1)
    @test !cache.lean_active
    @test _counts(cache) == (1, 1, 0, 0)

    KW.swap_lean_columns!(cache, plan, ((1, 2),))
    @test cache.lean_plan.q != plan.q && cache.lean_plan.row === plan.row
    PNM.numeric_refactor!(cache, A1)
    @test cache.lean_active
    @test _counts(cache) == (2, 1, 0, 0)
    b = [1.0, 2.0, 3.0, 4.0]
    @test PNM.solve!(cache, copy(b)) ≈ Matrix(A1) \ b
    # Re-aligning invalidates the current lean factors; an empty set restores the plan.
    KW.swap_lean_columns!(cache, plan, ())
    @test cache.lean_plan === plan && !cache.lean_active
    PNM.numeric_refactor!(cache, A0)
    @test cache.lean_active
    @test PNM.solve!(cache, copy(b)) ≈ Matrix(A0) \ b

    # A pair whose columns differ in pattern stays in place.
    sparse_pv = SparseArrays.sparse(pv)
    other = PNM.KLULinSolveCache(sparse_pv)
    p2 = KW.build_lean_plan(sparse_pv)
    KW.set_lean_plan!(other, p2)
    KW.swap_lean_columns!(other, p2, ((1, 2),))
    @test other.lean_plan.q == p2.q
    @test_throws ArgumentError KW.swap_lean_columns!(cache, p2, ())
end

@testset "KLULinSolveCache deferred analysis and re-pivot" begin
    A0 = SparseArrays.sparse([4.0 1.0 0.0; 1.0 4.0 1.0; 0.0 1.0 4.0])
    b = [1.0, 2.0, 3.0]
    plan = KW.build_lean_plan(A0)
    cache = PNM.KLULinSolveCache(A0)
    KW.defer_symbolic!(cache, A0)
    KW.set_lean_plan!(cache, plan)
    PNM.numeric_refactor!(cache, A0)
    @test cache.lean_active && cache.symbolic == C_NULL && PNM.is_factored(cache)
    @test PNM.solve!(cache, copy(b)) ≈ Matrix(A0) \ b
    @test _counts(cache) == (1, 0, 0, 0)

    # A lean solve the caller rejects re-pivots on the late analysis and pauses the plan.
    KW.repivot!(cache, A0)
    @test !cache.lean_active && cache.symbolic != C_NULL && cache.numeric != C_NULL
    @test _counts(cache) == (1, 0, 1, 1)
    @test PNM.tsolve!(cache, copy(b)) ≈ transpose(Matrix(A0)) \ b
    PNM.numeric_refactor!(cache, A0)
    @test !cache.lean_active
    KW.repivot!(cache, A0)
    @test _counts(cache) == (1, 0, 1, 1)

    # A lean reject pays the deferred analysis in-line.
    other = PNM.KLULinSolveCache(A0)
    KW.defer_symbolic!(other, A0)
    KW.set_lean_plan!(other, plan)
    A1 = copy(A0)
    A1[plan.p[1], plan.q[1]] = 1e-14
    PNM.numeric_refactor!(other, A1)
    @test !other.lean_active && other.numeric != C_NULL
    @test _counts(other) == (1, 1, 0, 1)
    @test PNM.solve!(other, copy(b)) ≈ Matrix(A1) \ b

    # Without a plan the first numeric_refactor! analyzes; freed handles drop a pending one.
    plain = PNM.KLULinSolveCache(A0)
    KW.defer_symbolic!(plain, A0)
    PNM.numeric_refactor!(plain, A0)
    @test _counts(plain) == (0, 0, 0, 1)
    @test PNM.solve!(plain, copy(b)) ≈ Matrix(A0) \ b
    KW.defer_symbolic!(plain, A0)
    finalize(plain)
    @test_throws ErrorException PNM.numeric_refactor!(plain, A0)
end

function _lean_task(plan, A0, t)
    vals = KW.LeanLUValues(plan)
    rng = Random.Xoshiro(t)
    out = Float64[]
    for _ in 1:20
        A1 = _lean_perturbed(A0, rng)
        KW.lean_refactor!(vals, plan, nonzeros(A1))
        append!(out, KW.lean_solve!(ones(size(A0, 1)), plan, vals))
    end
    return out
end

@testset "One lean plan shared by concurrent tasks" begin
    A0 = _lean_convection_diffusion(40)
    plan = KW.build_lean_plan(A0)
    h = _lean_plan_hash(plan)
    serial = [_lean_task(plan, A0, t) for t in 1:4]
    tasks = [Threads.@spawn _lean_task(plan, A0, t) for t in 1:4]
    @test fetch.(tasks) == serial
    @test _lean_plan_hash(plan) == h

    # The test worker runs one thread; repeat on four real threads in a subprocess.
    script = """
    using PowerNetworkMatrices, SparseArrays, LinearAlgebra, Random
    const KW = PowerNetworkMatrices.KLUWrapper
    m = 40
    T = spdiagm(0 => fill(2.0, m), 1 => fill(-0.7, m - 1), -1 => fill(-1.3, m - 1))
    A0 = kron(sparse(1.0I, m, m), T) + kron(T, sparse(1.0I, m, m))
    plan = KW.build_lean_plan(A0)
    phash(p) = hash(Tuple(getfield(p, f) for f in fieldnames(typeof(p))))
    h = phash(plan)
    function run(t)
        vals = KW.LeanLUValues(plan)
        rng = Xoshiro(t)
        out = Float64[]
        for _ in 1:200
            A1 = copy(A0)
            nonzeros(A1) .*= 1.0 .+ 0.1 .* (rand(rng, nnz(A1)) .- 0.5)
            KW.lean_refactor!(vals, plan, nonzeros(A1))
            append!(out, KW.lean_solve!(ones(size(A0, 1)), plan, vals))
        end
        return out
    end
    serial = [run(t) for t in 1:4]
    par = fetch.([Threads.@spawn run(t) for t in 1:4])
    println("nthreads=", Threads.nthreads(), " same=", par == serial, " hash=", phash(plan) == h)
    """
    project = "--project=$(Base.active_project())"
    cmd = `$(Base.julia_cmd()) -t 4 --startup-file=no $project -e $script`
    output = read(ignorestatus(cmd), String)
    @test occursin("nthreads=4 same=true hash=true", output)
end
