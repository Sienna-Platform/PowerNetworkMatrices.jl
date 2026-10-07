function _klu_small(n = 6)
    return SparseArrays.spdiagm(
        0 => collect(1.0:n) .+ 2.0,
        1 => fill(0.3, n - 1),
        -1 => fill(0.2, n - 1),
    )
end

@testset "Re-entry is allowed" begin
    A = _klu_small()
    b = ones(6)
    cache = PNM.KLULinSolveCache(A)
    PNM.full_factor!(cache, A)
    @test iszero(cache.owner[])
    PNM.full_refactor!(cache, A)
    @test iszero(cache.owner[])
    cache2 = PNM.klu_factorize(A)
    @test iszero(cache2.owner[])
    @test cache2 \ b ≈ A \ b
    @test iszero(cache2.owner[])
end

@testset "Foreign owner raises" begin
    A = _klu_small()
    b = ones(6)
    cache = PNM.klu_factorize(A)
    cache.owner[] = UInt(1)
    @test_throws ErrorException PNM.solve!(cache, copy(b))
    @test_throws ErrorException PNM.numeric_refactor!(cache, A)
    @test_throws ErrorException PNM.full_factor!(cache, A)
    cache.owner[] = UInt(0)
    @test PNM.solve!(cache, copy(b)) ≈ A \ b
end

@testset "Owner released on throw" begin
    A = SparseArrays.sparse([1, 2, 1, 2], [1, 1, 2, 2], [2.0, 1.0, 1.0, 2.0])
    cache = PNM.klu_factorize(A)
    A0 = copy(A)
    A0.nzval[1] = 0.0
    @test_throws PNM.LinearAlgebra.SingularException PNM.numeric_refactor!(cache, A0)
    @test iszero(cache.owner[])
end

const _KLU_STRESS_FULL = get(ENV, "PNM_KLU_STRESS", "") == "full"

function _convection_diffusion(m::Int)
    T = SparseArrays.spdiagm(
        -1 => fill(-1.2, m - 1),
        0 => fill(2.0, m),
        1 => fill(-0.8, m - 1),
    )
    I_m = SparseArrays.sparse(I, m, m)
    return kron(I_m, T) + kron(T, I_m) + 0.1 * SparseArrays.sparse(I, m * m, m * m)
end

function _fold(h::UInt64, x::AbstractArray{Float64})
    for w in reinterpret(UInt64, vec(x))
        h = (h ⊻ w) * 0x00000100000001b3
    end
    return h
end

# Each task owns its matrix copy and an Int64 and an Int32 cache (distinct libklu triples).
# Per iteration: perturb values, refactor both caches, then solve!/tsolve!/solve_sparse!
# with nrhs = 1 on each. Every 10 iterations both caches are rebuilt and the old ones left
# to the finalizer, so frees run on one thread while factorizations run on others.
function _klu_stress_task(
    t::Int,
    base::SparseArrays.SparseMatrixCSC{Float64, Int},
    niter::Int,
)
    A64 = copy(base)
    A32 = SparseArrays.SparseMatrixCSC{Float64, Int32}(A64)
    n = size(A64, 1)
    p = randn(Random.Xoshiro(t), SparseArrays.nnz(A64))
    rhs = randn(Random.Xoshiro(1000 + t), n)
    c64 = PNM.klu_factorize(A64)
    c32 = PNM.klu_factorize(A32)
    x = similar(rhs)
    out = zeros(n, 1)
    digest = zeros(UInt64, niter)
    for i in 1:niter
        s = 1e-4 * sin(i + t)
        A64.nzval .= base.nzval .* (1 .+ s .* p)
        copyto!(A32.nzval, A64.nzval)
        PNM.numeric_refactor!(c64, A64)
        PNM.numeric_refactor!(c32, A32)
        col = SparseArrays.sparse(
            [1 + (7i + t) % n, 1 + (13i) % n],
            [1, 1],
            [1.0 + s, -0.5],
            n,
            1,
        )
        h = UInt64(t)
        for c in (c64, c32)
            x .= rhs .* (1 + s)
            PNM.solve!(c, x)
            h = _fold(h, x)
            x .= rhs .* (1 - s)
            PNM.tsolve!(c, x)
            h = _fold(h, x)
            PNM.solve_sparse!(c, col, out)
            h = _fold(h, out)
        end
        digest[i] = h
        if iszero(i % 10)
            c64 = PNM.klu_factorize(A64)
            c32 = PNM.klu_factorize(A32)
            GC.gc(iszero(i % 50))
        end
    end
    return digest
end

@testset "Distinct caches in parallel" begin
    if Threads.nthreads() == 1
        @test_skip false
    else
        bases = [_convection_diffusion(50)]
        niter = 40
        if _KLU_STRESS_FULL
            sys = build_system(MatpowerTestSystems, "matpower_ACTIVSg10k_sys")
            bases = [_convection_diffusion(100), PNM.ABA_Matrix(sys).data]
            niter = 100
        end
        ntasks = 2 * Threads.nthreads()
        base_of(t) = bases[1 + t % length(bases)]
        serial = [_klu_stress_task(t, base_of(t), niter) for t in 1:ntasks]
        tasks = [Threads.@spawn _klu_stress_task(t, base_of(t), niter) for t in 1:ntasks]
        @test fetch.(tasks) == serial
    end
end

function _run_subprocess(script::String; threads::Int = 1, timeout::Real = 300)
    project = "--project=$(Base.active_project())"
    cmd = `$(Base.julia_cmd()) -t $threads --startup-file=no $project -e $script`
    out = IOBuffer()
    proc = run(pipeline(ignorestatus(cmd); stdout = out, stderr = out); wait = false)
    if timedwait(() -> process_exited(proc), timeout) == :timed_out
        kill(proc)
        wait(proc)
        output = String(take!(out)) * "\n[timed out]"
        @error "KLU subprocess timed out" output
        return (ok = false, output = output)
    end
    output = String(take!(out))
    ok = success(proc)
    if !ok
        @error "KLU subprocess failed" output
    end
    return (ok = ok, output = output)
end

const _KLU_SUBPROCESS_PRELUDE = """
using PowerNetworkMatrices, SparseArrays, LinearAlgebra
import PowerNetworkMatrices as PNM
function grid(m)
    T = spdiagm(-1 => fill(-1.2, m - 1), 0 => fill(2.0, m), 1 => fill(-0.8, m - 1))
    I_m = sparse(I, m, m)
    return kron(I_m, T) + kron(T, I_m) + 0.1 * sparse(I, m * m, m * m)
end
"""

@testset "Allocator pinned at load" begin
    # Without the pin, freeing these libc-born blocks after the sparse qr switches the
    # allocator sets off a GC on nearly every allocation (the process effectively hangs).
    script = _KLU_SUBPROCESS_PRELUDE * """
    lib = SparseArrays.LibSuiteSparse.libsuitesparseconfig
    p = ccall((:SuiteSparse_config_malloc_func_get, lib), Ptr{Cvoid}, ())
    println("jl_malloc=", p == cglobal(:jl_malloc))
    A = grid(200)
    caches = [PNM.klu_factorize(A) for _ in 1:20]
    qr(sparse([2.0 1.0; 1.0 3.0]))
    foreach(Base.finalize, caches)
    GC.gc()
    const SINK = Ref{Any}()
    g0 = Base.gc_num().pause
    for _ in 1:200_000
        SINK[] = Vector{Float64}(undef, 8)
    end
    println("gcs=", Base.gc_num().pause - g0)
    """
    r = _run_subprocess(script; timeout = 120)
    @test r.ok
    @test occursin("jl_malloc=true", r.output)
    @test occursin(r"gcs=\d+", r.output)
    if occursin(r"gcs=\d+", r.output)
        @test parse(Int, match(r"gcs=(\d+)", r.output)[1]) < 10
    end
end

@testset "No crash at exit with unawaited tasks" begin
    script = _KLU_SUBPROCESS_PRELUDE * """
    A = grid(60)
    for k in 1:8
        Threads.@spawn begin
            B = copy(A)
            c = PNM.klu_factorize(B)
            x = ones(size(B, 1))
            while true
                PNM.numeric_refactor!(c, B)
                PNM.solve!(c, x)
                x .= 1.0
                c = PNM.klu_factorize(B)
                # Without a yield point these loops hold every thread, and exit
                # waits forever on Base's profile-listener task.
                yield()
            end
        end
    end
    sleep(0.5)
    println("main done")
    """
    nruns = 3
    if _KLU_STRESS_FULL
        nruns = 20
    end
    for _ in 1:nruns
        r = _run_subprocess(script; threads = 4, timeout = 300)
        @test r.ok
        @test occursin("main done", r.output)
    end
end
