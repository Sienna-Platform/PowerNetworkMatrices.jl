# Shared low-level helpers for the virtual matrices and their factorization core.
#
# These functions are intentionally defined early in the module load order
# (before `virtual_factor_core.jl` and the `Virtual{PTDF,LODF,MODF}` wrappers)
# so the core constructor and the wrappers can all call them. They were
# previously split between `virtual_ptdf_calculations.jl`
# (`_create_factorization`, `_solve_factorization`) and
# `virtual_lodf_calculations.jl` (`_extract_arc_susceptances`,
# `_extract_branch_susceptances_by_arc`, `_get_PTDF_A_diag`); they live here now
# so a single shared `VirtualFactorCore` can build/solve the factorization once.

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
    ::LinearSolverType,
    ::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    return error(
        "Only KLU and AppleAccelerateLU solvers are supported for VirtualPTDF factorization.",
    )
end

# --- In-place solves (backend dispatch) ---
# Both overloads solve in place (zero-allocation hot path). KLU and Apple
# Accelerate are the only supported backends; adding a new backend requires
# extending this method.
function _solve_factorization(K::KLULinSolveCache{Float64}, b::Vector{Float64})
    solve!(K, b)
    return b
end

function _solve_factorization(K::AAFactorCache, b::Vector{Float64})
    AccelerateWrapper.solve!(K, b)
    return b
end

# --- Per-arc susceptance extraction ---

"""
Extract the effective susceptance for each arc from the BA matrix.
For arc j, the susceptance is the absolute value of the first nonzero in BA column j.
BA columns always have the structure [+b, -b] (from-bus and to-bus entries),
so both nonzeros have the same magnitude.
"""
function _extract_arc_susceptances(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
)::Vector{Float64}
    n_arcs = size(BA, 2)
    b = Vector{Float64}(undef, n_arcs)
    nzv = SparseArrays.nonzeros(BA)
    for j in 1:n_arcs
        rng = nzrange(BA, j)
        b[j] = isempty(rng) ? 0.0 : abs(nzv[first(rng)])
    end
    return b
end

"""
    _extract_branch_susceptances_by_arc(BA, arc_ax, nr_data) -> Vector{Vector{Float64}}

Extract per-branch susceptances for each arc. For arcs with a single branch,
returns a one-element vector equal to the arc susceptance. For arcs with
parallel branches (double circuits), returns one entry per branch. For arcs
with series-reduced branches (D2 reduction), returns one entry per segment.

This enables single-branch contingencies on parallel and series-reduced arcs.
"""
function _extract_branch_susceptances_by_arc(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_ax::Vector{Tuple{Int, Int}},
    nr_data::NetworkReductionData,
)::Vector{Vector{Float64}}
    n_arcs = size(BA, 2)
    nzv = SparseArrays.nonzeros(BA)
    result = Vector{Vector{Float64}}(undef, n_arcs)

    for j in 1:n_arcs
        arc = arc_ax[j]
        rng = nzrange(BA, j)
        arc_b = isempty(rng) ? 0.0 : abs(nzv[first(rng)])

        if haskey(nr_data.parallel_branch_map, arc)
            bp = nr_data.parallel_branch_map[arc]
            result[j] = Float64[
                get_series_susceptance(branch) for branch in bp.branches
            ]
        elseif haskey(nr_data.series_branch_map, arc)
            bs = nr_data.series_branch_map[arc]
            result[j] = Float64[
                get_series_susceptance(segment) for segment in bs
            ]
        else
            result[j] = [arc_b]
        end
    end

    return result
end

# --- PTDF·A diagonal ---

"""
    _get_PTDF_A_diag(K, BA, A, ref_bus_positions) -> Vector{Float64}

Compute `diag(PTDF · A)`. Each row of `A` has exactly two nonzeros (+1 at the
from-bus, -1 at the to-bus), so the per-arc dot product reduces to two indexed
reads into the solved PTDF row after a one-time transpose of `A`.
"""
function _get_PTDF_A_diag(
    K,
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    ref_bus_positions::Set{Int},
)
    n_branches = size(BA, 2)
    n_buses = size(BA, 1)
    diag_ = zeros(n_branches)

    valid_ix = setdiff(1:n_buses, ref_bus_positions)
    n_valid = length(valid_ix)
    bus_to_valid_idx = _build_bus_to_valid_idx(n_buses, valid_ix)

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
    ba_rv = SparseArrays.rowvals(BA)
    ba_nz = SparseArrays.nonzeros(BA)

    for i in 1:n_branches
        fill!(ba_col, 0.0)
        @inbounds for k in SparseArrays.nzrange(BA, i)
            valid_i = bus_to_valid_idx[ba_rv[k]]
            valid_i > 0 || continue
            ba_col[valid_i] = ba_nz[k]
        end

        # Read PTDF row from the returned buffer — backend-agnostic
        # (KLU mutates `ba_col` and returns it; other backends may
        # return a fresh vector, so capture the return value).
        lin_solve = _solve_factorization(K, ba_col)

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
