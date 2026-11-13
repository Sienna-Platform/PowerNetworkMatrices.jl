# Per-worker factorizations for parallel queries on the virtual matrices. A `worker_core` shares
# a core's read-only topology but owns its factorization, scratch and lock, so workers solve
# without contending on the shared `solver_lock`. Every function here takes any object with a
# `get_core` and solves on that core.

"""
    get_core(mat) -> FactorCore

Return the factorization core that `mat` solves on.
"""
function get_core end
get_core(core::FactorCore) = core

# Apple Accelerate calls have segfaulted when concurrent, even on distinct factorizations, so a
# worker core on that backend shares its parent's lock: workers serialize. KLU's distinct caches
# are safe to use concurrently, so each worker gets its own lock. An AA worker also shares the
# parent's factorization: with a shared lock, a second one would never solve concurrently.
_worker_lock(::KLULinSolveCache{Float64}, ::ReentrantLock) = ReentrantLock()
_worker_lock(::AAFactorCache, parent::ReentrantLock) = parent

# The KLU worker factors its own ABA. The AA worker reuses the parent's factorization: AA's
# internal scratch is safe only because every access holds the shared lock. Nothing refactors a
# core's `K` after it is built.
function _worker_factorization(
    ::KLULinSolveCache{Float64},
    core::FactorCore,
    lock::ReentrantLock,
)
    ref = Set{Int}(_ref_bus_positions(core))
    ABA = calculate_ABA_matrix(core.A, core.BA, ref)
    return @lock lock _create_factorization(KLUSolver(), ABA)
end
_worker_factorization(::AAFactorCache, core::FactorCore, ::ReentrantLock) = core.K

"""
    worker_core(core::FactorCore, network_data) -> FactorCore

A copy of `core` that shares its read-only topology and holds `network_data`, with its own
solve scratch. On KLU the copy has its own factorization of the same ABA matrix and its own
`solver_lock`; on Apple Accelerate it shares the factorization and the lock of `core`. The lazy
`PTDF_A_diag` starts empty.
"""
function worker_core(core::FactorCore, network_data)
    n_bus = size(core.BA, 1)
    lock = _worker_lock(core.K, core.solver_lock)
    K = _worker_factorization(core.K, core, lock)
    return FactorCore(
        K,
        core.BA,
        core.A,
        core.arc_susceptances,
        core.arc_susceptance_signs,
        core.arc_live_members,
        core.axes,
        core.lookup,
        core.valid_ix,
        core.bus_to_valid_idx,
        core.subnetwork_axes,
        core.tol,
        core.tol_spec,
        [zeros(n_bus)],
        [zeros(length(core.valid_ix))],
        lock,
        core.system_uuid,
        Float64[],
        Threads.Atomic{Bool}(false),
        network_data,
    )
end

"""
    worker_network_data(network_data) -> Any

Return the network data that a [`worker_core`](@ref) copy holds. The default shares
`network_data` with the parent core. A network data type that keeps lazy per-core state must
add a method that returns a new copy. Without it, workers that solve concurrently race on
that state.
"""
worker_network_data(network_data) = network_data

"""
    worker_core(mat) -> FactorCore

A new core sharing the read-only topology of `mat`'s core (a `FactorCore` or a matrix with a
`get_core` method) with its own solve scratch. On KLU the worker also has its own
factorization of the same ABA matrix and its own `solver_lock`. The factorization is
deterministic, so every solve on the worker equals the same solve on the original bit for bit,
and workers solve concurrently. On Apple Accelerate the worker shares the parent's
factorization and lock, so its solves serialize with the parent's and every other worker's.
The lazy `PTDF_A_diag` starts empty. The worker holds the result of
[`worker_network_data`](@ref) for the network data of the parent core.
"""
function worker_core(mat)
    core = get_core(mat)
    return worker_core(core, worker_network_data(get_network_data(core)))
end

function _check_worker_core(core::FactorCore, solver::FactorCore)
    solver.BA === core.BA || error(
        "The solver core does not share this matrix's topology; build it with " *
        "worker_core(matrix).",
    )
    return
end

# --- Factor-free flow algebra on angles ---

function _check_bus_rows(core::FactorCore, x::AbstractVecOrMat, name::String)
    size(x, 1) == size(core.BA, 1) || throw(
        DimensionMismatch(
            "$name has $(size(x, 1)) rows; the network has $(size(core.BA, 1)) buses.",
        ),
    )
    return
end

"""
    solve_bus_angles!(θ, mat, p) -> θ

`θ = B⁻¹ p` on the reference-bus-reduced ABA matrix `B` of `mat` (a `FactorCore` or a
matrix with a `get_core` method), in full-bus space with the reference-bus entries 0;
`p` (injections, per bus position) is read at the non-reference buses only. Vectors or
matrices (one column per case). Solves on `mat`'s core under its `solver_lock`; pass a
[`worker_core`](@ref) to solve concurrently. `arc_flows!(…, θ, …)` then gives the flows of `p`
balanced at each island's reference bus, i.e. the PTDF rows dotted with `p`.
"""
function solve_bus_angles!(
    θ::AbstractVector{Float64},
    mat,
    p::AbstractVector{Float64},
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_bus_rows(core, p, "p")
    valid_ix = core.valid_ix
    return with_solver(
        core.K, core.work_ba_col, core.temp_data, core.solver_lock,
    ) do K_solver, work_ba_col, _
        @inbounds for (i, b) in enumerate(valid_ix)
            work_ba_col[i] = p[b]
        end
        lin_solve = _solve_factorization(K_solver, work_ba_col)
        return _gather_to_buses!(θ, valid_ix, lin_solve)
    end
end

solve_bus_angles!(θ::AbstractMatrix{Float64}, mat, p::AbstractMatrix{Float64}) =
    solve_bus_angles!(
        θ, mat, p, Matrix{Float64}(undef, length(get_core(mat).valid_ix), size(p, 2)),
    )

"""
    solve_bus_angles!(θ::AbstractMatrix, mat, p::AbstractMatrix, rhs::Matrix{Float64}) -> θ

The matrix form solving in `rhs`, a caller buffer of size (non-reference buses, columns of `p`).
"""
function solve_bus_angles!(
    θ::AbstractMatrix{Float64},
    mat,
    p::AbstractMatrix{Float64},
    rhs::Matrix{Float64},
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_bus_rows(core, p, "p")
    size(θ, 2) == size(p, 2) ||
        throw(DimensionMismatch("θ has $(size(θ, 2)) columns, p has $(size(p, 2))."))
    valid_ix = core.valid_ix
    size(rhs) == (length(valid_ix), size(p, 2)) || throw(
        DimensionMismatch(
            "rhs is $(size(rhs)); expected ($(length(valid_ix)), $(size(p, 2))).",
        ),
    )
    for t in axes(p, 2), (i, b) in enumerate(valid_ix)
        rhs[i, t] = p[b, t]
    end
    sol = @lock core.solver_lock _solve_factorization(core.K, rhs)
    for t in axes(p, 2)
        _gather_to_buses!(view(θ, :, t), valid_ix, view(sol, :, t))
    end
    return θ
end

# BA[:, arc] ⋅ θ[:, t].
function _ba_dot(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc::Int,
    θ::AbstractVecOrMat{Float64},
    t::Int,
)
    rv = SparseArrays.rowvals(BA)
    nzv = SparseArrays.nonzeros(BA)
    acc = 0.0
    @inbounds for k in SparseArrays.nzrange(BA, arc)
        acc += nzv[k] * θ[rv[k], t]
    end
    return acc
end

function _check_flows(
    flows::AbstractVecOrMat,
    θ::AbstractVecOrMat,
    arcs::AbstractVector{Int},
)
    size(flows, 1) == length(arcs) && size(flows, 2) == size(θ, 2) || throw(
        DimensionMismatch(
            "flows is $(size(flows)); expected ($(length(arcs)), $(size(θ, 2))).",
        ),
    )
    return
end

"""
    arc_flows!(flows, mat, θ, arcs) -> flows
    arc_flows!(flows, mat, θ, arcs, wf::WoodburyFactors) -> flows

Flows on the arc indices `arcs` from angles `θ` (from [`solve_bus_angles!`](@ref); a vector, or
a matrix with one column per case, `flows[q, t]` for arc `arcs[q]`). Without `wf`,
`flows = BA[:, arcs]ᵀ θ`. With `wf` the flows are post-modification by Woodbury:
`flows[q, t] = b_post * (ν_mᵀ θ_t - (ν_mᵀ Z) ⋅ W⁻ᵀ Uᵀ θ_t)`, which for `θ = B⁻¹ p` equals
`vmodf[arcs[q], mod] ⋅ p` before its islanding zero-out (an arc the modification removes reads
exactly 0.0). Under islanding, entries of other islands are not zeroed: pass `θ = B⁻¹ p_I` with
`p` restricted to the arc's island `I` (see `wf.bus_island_labels`). Pure algebra on `BA` and
`wf`, O(length(arcs) · M) per column; no solve, no lock, safe from any number of tasks.
"""
function arc_flows!(
    flows::AbstractVecOrMat{Float64},
    mat,
    θ::AbstractVecOrMat{Float64},
    arcs::AbstractVector{Int},
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_flows(flows, θ, arcs)
    for t in axes(θ, 2), (q, m) in enumerate(arcs)
        flows[q, t] = _ba_dot(core.BA, m, θ, t)
    end
    return flows
end

arc_flows!(
    flows::AbstractVecOrMat{Float64},
    mat,
    θ::AbstractVecOrMat{Float64},
    arcs::AbstractVector{Int},
    wf::WoodburyFactors,
) = arc_flows!(
    flows, mat, θ, arcs, wf, WoodburyFlowScratch(_woodbury_rank(wf), size(θ, 2)),
)

"""
    WoodburyFlowScratch(max_rank, max_columns = 1)

Buffers of the Woodbury [`arc_flows!`](@ref) for updates of rank up to `max_rank` (at most
the number of modified arcs) and angles of up to `max_columns` columns, so a caller reusing
one allocates nothing.
"""
struct WoodburyFlowScratch
    u::Matrix{Float64}
    c::Matrix{Float64}
    zm_Z::Vector{Float64}
end

WoodburyFlowScratch(max_rank::Int, max_columns::Int = 1) = WoodburyFlowScratch(
    Matrix{Float64}(undef, max_rank, max_columns),
    Matrix{Float64}(undef, max_rank, max_columns),
    Vector{Float64}(undef, max_rank),
)

"""
    arc_flows!(flows, mat, θ, arcs, wf::WoodburyFactors, scratch::WoodburyFlowScratch) -> flows

The Woodbury form on caller-owned buffers; allocation-free.
"""
function arc_flows!(
    flows::AbstractVecOrMat{Float64},
    mat,
    θ::AbstractVecOrMat{Float64},
    arcs::AbstractVector{Int},
    wf::WoodburyFactors,
    scratch::WoodburyFlowScratch,
)
    core = get_core(mat)
    _check_bus_rows(core, θ, "θ")
    _check_flows(flows, θ, arcs)
    BA = core.BA
    arc_sus = core.arc_susceptances
    M = _woodbury_rank(wf)
    T = size(θ, 2)
    M <= length(scratch.zm_Z) && T <= size(scratch.u, 2) || throw(
        DimensionMismatch(
            "The WoodburyFlowScratch holds rank $(length(scratch.zm_Z)) and " *
            "$(size(scratch.u, 2)) columns; this call needs $M and $T.",
        ),
    )
    # c[:, t] = W⁻ᵀ Uᵀ θ_t, with (Uᵀ θ)_j = ν_jᵀ θ = BA[:, e_j]ᵀ θ / b_{e_j}. Transposed so the
    # result matches the row form `z_m - Z W⁻¹ (Zᵀ ν_m)` dotted with p in floating point too.
    u = view(scratch.u, 1:M, 1:T)
    for t in 1:T, j in 1:M
        e = wf.arc_indices[j]
        u[j, t] = _ba_dot(BA, e, θ, t) / arc_sus[e]
    end
    c = view(scratch.c, 1:M, 1:T)
    LinearAlgebra.mul!(c, transpose(wf.W_inv), u)
    zm_Z = view(scratch.zm_Z, 1:M)
    for (q, m) in enumerate(arcs)
        if _monitored_arc_out(arc_sus, m, wf)
            for t in 1:T
                flows[q, t] = 0.0
            end
            continue
        end
        b_pre = arc_sus[m]
        b_post = _post_modification_susceptance(arc_sus, m, wf)
        _monitored_Z!(zm_Z, BA, m, b_pre, wf.Z)
        for t in 1:T
            acc = _ba_dot(BA, m, θ, t) / b_pre
            for j in 1:M
                acc -= zm_Z[j] * c[j, t]
            end
            flows[q, t] = b_post * acc
        end
    end
    return flows
end

"""
    compute_woodbury_factors(mat, mod[, labeler]) -> WoodburyFactors

Uncached Woodbury factors of `mod`, solved on `mat`'s core (`mat` a `FactorCore`, e.g. a
[`worker_core`](@ref), or a matrix with a `get_core` method).
`labeler(BA, arc_sus, modifications, n_bus)` gives the island labels when `mod` islands; a
`BridgeLabels` gives the same partition as the default in O(n_bus) for a single bridge, with
different representatives.
"""
function compute_woodbury_factors(
    mat,
    mod::NetworkModification,
    labeler::F = _post_contingency_bus_labels,
) where {F}
    return _compute_woodbury_factors(get_core(mat), _arc_modification_vector(mod), labeler)
end

"""
    apply_woodbury_correction(core::FactorCore, monitored_idx::Int, wf) -> Vector{Float64}

The PTDF-matrix form solved on `core`, e.g. a [`worker_core`](@ref).
"""
apply_woodbury_correction(
    core::FactorCore,
    monitored_idx::Int,
    wf::WoodburyFactors,
) =
    _apply_woodbury_correction(core, monitored_idx, wf)
