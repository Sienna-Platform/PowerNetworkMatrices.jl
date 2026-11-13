# --- Factorization creation (solver dispatch) ---

function _create_factorization(
    ::KLUSolver,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    return klu_factorize(ABA)
end

function _create_factorization(
    ::AppleAccelerateLUSolver,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    _has_apple_accelerate_backend() || error(_apple_accelerate_unavailable_error())
    return AccelerateWrapper.aa_factorize(ABA)
end

function _create_factorization(
    ::MKLPardisoSolver,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    cache = PardisoLinSolveCache(ABA)
    full_factor!(cache, ABA)
    return cache
end

function _create_factorization(
    solver::LinearSolverType,
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    return error(
        "Only the KLU, AppleAccelerateLU, and MKLPardiso solvers are supported for this " *
        "factorization; got $(typeof(solver)).",
    )
end

# --- In-place solves (backend dispatch) ---
# Both overloads solve in place (zero-allocation hot path). KLU and Apple
# Accelerate are the only supported backends; adding a new backend requires
# extending this method.
function _solve_factorization(K::KLULinSolveCache{Float64}, b::VecOrMat{Float64})
    solve!(K, b)
    return b
end

function _solve_factorization(K::AAFactorCache, b::VecOrMat{Float64})
    AccelerateWrapper.solve!(K, b)
    return b
end

# --- Per-arc susceptance extraction ---

# BA columns always have the structure [+b, -b] (from-bus and to-bus entries), so the first
# nonzero's magnitude is the arc's susceptance.
function _ba_column_susceptance(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    nzv::Vector{Float64},
    j::Int,
)::Float64
    rng = nzrange(BA, j)
    isempty(rng) && return 0.0
    return abs(nzv[first(rng)])
end

"""
Extract the effective susceptance for each arc from the BA matrix: the absolute value of
the first nonzero in BA column j.
"""
function _extract_arc_susceptances(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
)::Vector{Float64}
    n_arcs = size(BA, 2)
    nzv = SparseArrays.nonzeros(BA)
    return Float64[_ba_column_susceptance(BA, nzv, j) for j in 1:n_arcs]
end

# --- BA-column scatter + solve ---

"""
    _solve_ba_column!(K, work_ba_col, BA, bus_to_valid_idx, col)

Scatter BA[:, col]'s non-zeros into work_ba_col and solve. Capture the return: only KLU
solves in place.
"""
function _solve_ba_column!(
    K,
    work_ba_col::Vector{Float64},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    bus_to_valid_idx::Vector{Int},
    col::Int,
)
    fill!(work_ba_col, 0.0)
    ba_rv = SparseArrays.rowvals(BA)
    ba_nz = SparseArrays.nonzeros(BA)
    @inbounds for k in SparseArrays.nzrange(BA, col)
        valid_i = bus_to_valid_idx[ba_rv[k]]
        valid_i > 0 || continue
        work_ba_col[valid_i] = ba_nz[k]
    end
    return _solve_factorization(K, work_ba_col)
end

# --- PTDF·A diagonal ---

"""
    _get_PTDF_A_diag(K, BA, A, valid_ix, bus_to_valid_idx) -> Vector{Float64}

Compute `diag(PTDF · A)`. Each row of `A` has exactly two nonzeros (+1 at the
from-bus, -1 at the to-bus), so the per-arc dot product reduces to two indexed
reads into the solved PTDF row after a one-time transpose of `A`.
"""
function _get_PTDF_A_diag(
    K,
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    valid_ix::Vector{Int},
    bus_to_valid_idx::Vector{Int},
)
    n_branches = size(BA, 2)
    diag_ = zeros(n_branches)
    n_valid = length(valid_ix)

    # Per-arc (from_valid, to_valid) via one transpose of A; 0 = ref bus.
    A_T = SparseArrays.sparse(transpose(A))
    arc_from_valid = Vector{Int}(undef, n_branches)
    arc_to_valid = Vector{Int}(undef, n_branches)
    at_rv = SparseArrays.rowvals(A_T)
    at_nz = SparseArrays.nonzeros(A_T)
    for i in 1:n_branches
        f_valid = 0
        t_valid = 0
        @inbounds for k in SparseArrays.nzrange(A_T, i)
            bus_ix = at_rv[k]
            v = at_nz[k]
            valid_i = bus_to_valid_idx[bus_ix]
            if v > 0
                f_valid = valid_i
            elseif v < 0
                t_valid = valid_i
            end
        end
        arc_from_valid[i] = f_valid
        arc_to_valid[i] = t_valid
    end

    ba_col = zeros(n_valid)

    for i in 1:n_branches
        lin_solve = _solve_ba_column!(K, ba_col, BA, bus_to_valid_idx, i)

        # H[e,e] = ptdf[from] - ptdf[to]; ref-bus entries are 0.
        f = arc_from_valid[i]
        t = arc_to_valid[i]
        v_f = if f > 0
            lin_solve[f]
        else
            0.0
        end
        v_t = if t > 0
            lin_solve[t]
        else
            0.0
        end
        @inbounds diag_[i] = v_f - v_t
    end
    return diag_
end
