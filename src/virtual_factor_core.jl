"""
    VirtualFactorCore{Ax, L, K}

Private, unexported container for the shared state of the virtual sensitivity
matrices (`VirtualPTDF`, `VirtualMODF`, `VirtualLODF`). It owns the single
factorization, the topology, the solve scratch buffers, and the solver lock so
that all three wrappers can be built on top of one factorization instead of
each rebuilding (and re-factorizing) the same DC network.

The two expensive/shared derived quantities (`PTDF_A_diag` and
`branch_susceptances_by_arc`) are computed lazily and cached here, so that
sharing one core between, say, a `VirtualMODF` and a `VirtualLODF` computes them
only once.

The type parameter order `{Ax, L, K}` matches `VirtualPTDF` so the PTDF/MODF
wrappers can carry `core::VirtualFactorCore{Ax, L, K}` with the same parameters.

# Thread-safety
All libklu solves serialize through the process-wide `_LIBKLU_LOCK` and the
per-core `solver_lock`. The single scratch slot in `temp_data`/`work_ba_col` is
protected by `solver_lock` (acquired via `with_solver`). A core shared between
multiple wrappers therefore serializes their solves, which is the existing
single-scratch model.

# Fields
- `K`: ABA factorization (`KLULinSolveCache{Float64}` or `AAFactorCache`).
- `BA::SparseMatrixCSC{Float64, Int}`: BA matrix.
- `A::SparseMatrixCSC{Int8, Int}`: incidence matrix data.
- `arc_susceptances::Vector{Float64}`: effective susceptance per arc.
- `axes::Ax`: `(arc_axis, bus_axis)`.
- `lookup::L`: `(arc_lookup, bus_lookup)`.
- `valid_ix::Vector{Int}`: non-reference bus indices.
- `bus_to_valid_idx::Vector{Int}`: inverse of `valid_ix` (0 for reference buses).
- `subnetwork_axes::Dict{Int, Ax}`: per-reference-bus subnetwork axes (bus form).
- `tol::SparsificationCutoff`: resolved per-row sparsification rule (absolute
  cutoff for a `Float64` `tol`, relative cutoff for an [`AutoTolerance`](@ref)).
- `network_reduction_data::NetworkReductionData`: reduction maps.
- `temp_data::Vector{Vector{Float64}}`: single-slot scratch (size n_buses).
- `work_ba_col::Vector{Vector{Float64}}`: single-slot scratch (size n_valid).
- `solver_lock::ReentrantLock`: serializes solves + scratch access.
- `system_uuid::Union{Base.UUID, Nothing}`: originating system UUID.
- `PTDF_A_diag::Vector{Float64}`: raw `H[e,e]`; empty until first
  `get_PTDF_A_diag(core)`.
- `branch_susceptances_by_arc::Vector{Vector{Float64}}`: per-branch susceptances;
  empty until first `get_branch_susceptances_by_arc(core)`.
"""
struct VirtualFactorCore{Ax, L <: NTuple{2, Dict}, K}
    K::K
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    arc_susceptances::Vector{Float64}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    bus_to_valid_idx::Vector{Int}
    subnetwork_axes::Dict{Int, Ax}
    tol::SparsificationCutoff
    network_reduction_data::NetworkReductionData
    temp_data::Vector{Vector{Float64}}
    work_ba_col::Vector{Vector{Float64}}
    solver_lock::ReentrantLock
    system_uuid::Union{Base.UUID, Nothing}
    PTDF_A_diag::Vector{Float64}
    branch_susceptances_by_arc::Vector{Vector{Float64}}
end

# --- Accessors (the wrappers forward to these) ---

get_axes(c::VirtualFactorCore) = c.axes
get_lookup(c::VirtualFactorCore) = c.lookup
get_arc_lookup(c::VirtualFactorCore) = c.lookup[1]
get_bus_lookup(c::VirtualFactorCore) = c.lookup[2]
get_arc_axis(c::VirtualFactorCore) = c.axes[1]
get_bus_axis(c::VirtualFactorCore) = c.axes[2]
get_ref_bus(c::VirtualFactorCore) = sort!(collect(keys(c.subnetwork_axes)))
# A subnetwork's representative can itself be merged away by a later reduction (e.g.
# ZeroImpedanceBranchReduction folding a swing into another bus); resolve it through the
# reduction's reverse map to the surviving bus it now shares a position with. Mirrors the
# generic get_ref_bus_position(::PowerNetworkMatrix); the core is not a PowerNetworkMatrix,
# so it needs its own method for the wrappers (VirtualPTDF/VirtualMODF) to delegate to.
function get_ref_bus_position(c::VirtualFactorCore)
    bus_lookup = get_bus_lookup(c)
    nr = get_network_reduction_data(c)
    return [get_bus_index(x, bus_lookup, nr) for x in keys(c.subnetwork_axes)]
end
get_network_reduction_data(c::VirtualFactorCore) = c.network_reduction_data
get_system_uuid(c::VirtualFactorCore) = c.system_uuid
# Resolved cutoff object (used for per-row sparsification); `get_tol` returns its
# Float64 representative for display/serialization.
get_cutoff(c::VirtualFactorCore) = c.tol
get_tol(c::VirtualFactorCore) = cutoff_value(c.tol)

# Woodbury / NetworkModification kernel accessors.
_get_BA(c::VirtualFactorCore) = c.BA
_get_arc_susceptances(c::VirtualFactorCore) = c.arc_susceptances
_get_valid_ix(c::VirtualFactorCore) = c.valid_ix

function _ref_bus_positions(c::VirtualFactorCore)
    n_buses = length(c.axes[2])
    return Set{Int}(setdiff(1:n_buses, c.valid_ix))
end

# --- Lazy shared derived quantities ---

"""
    get_PTDF_A_diag(c::VirtualFactorCore) -> Vector{Float64}

Return the raw diagonal `H[e, e]` of `PTDF · A`, computing it (one solve per
arc) on first access and caching it on the core. Subsequent calls — including
from other wrappers sharing this core — return the cached vector.
"""
function get_PTDF_A_diag(c::VirtualFactorCore)
    diag = c.PTDF_A_diag
    !isempty(diag) && return diag
    @lock c.solver_lock begin
        diag = c.PTDF_A_diag
        !isempty(diag) && return diag
        n_arcs = length(c.axes[1])
        @info "Computing PTDF_A_diag on first access ($n_arcs arcs)."
        t0 = time_ns()
        new_diag = _get_PTDF_A_diag(c.K, c.BA, c.A, _ref_bus_positions(c))
        resize!(diag, length(new_diag))
        copyto!(diag, new_diag)
        elapsed = (time_ns() - t0) / 1e9
        @info "Computed PTDF_A_diag in $(round(elapsed; digits = 2)) s (cached)."
        return diag
    end
end

"""
    get_branch_susceptances_by_arc(c::VirtualFactorCore) -> Vector{Vector{Float64}}

Return the per-branch susceptances for each arc, computing them on first access
and caching them on the core.
"""
function get_branch_susceptances_by_arc(c::VirtualFactorCore)
    bs = c.branch_susceptances_by_arc
    !isempty(bs) && return bs
    @lock c.solver_lock begin
        bs = c.branch_susceptances_by_arc
        !isempty(bs) && return bs
        new_bs = _extract_branch_susceptances_by_arc(
            c.BA, c.axes[1], c.network_reduction_data,
        )
        resize!(bs, length(new_bs))
        copyto!(bs, new_bs)
        return bs
    end
end

# --- Constructor ---

"""
    VirtualFactorCore(ybus::Ybus; linear_solver, tol, system_uuid) -> VirtualFactorCore

Build the shared factorization core from a `Ybus`. This is the single place that
constructs the incidence matrix, BA matrix, ABA matrix, and its factorization.
"""
function VirtualFactorCore(
    ybus::Ybus;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    system_uuid::Union{Base.UUID, Nothing} = nothing,
)
    solver = resolve_linear_solver(linear_solver)
    ref_bus_positions = get_ref_bus_position(ybus)
    A = IncidenceMatrix(ybus)
    BA = BA_Matrix(ybus)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    bus_ax = get_bus_axis(A)
    axes = A.axes
    look_up = A.lookup
    subnetwork_axes = A.subnetwork_axes
    if length(subnetwork_axes) > 1
        @info "Network is not connected, using subnetworks"
    end

    K = _create_factorization(solver, ABA)

    # Resolve the per-row sparsification rule once: a Float64 becomes an absolute
    # cutoff, an AutoTolerance a relative cutoff (κ logged via the reused K).
    cutoff = _resolve_virtual_cutoff(tol, K, ABA, SparseArrays.nonzeros(BA.data))

    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)
    bus_to_valid_idx = _build_bus_to_valid_idx(length(bus_ax), valid_ix)

    # Single scratch slot — solves serialize through `solver_lock` +
    # `_LIBKLU_LOCK`, so per-worker scratch is unnecessary. Kept as a
    # `Vector{Vector{Float64}}` so the `with_solver` callback signature
    # stays uniform across solver backends.
    temp_data = [zeros(length(bus_ax))]
    work_ba_col = [zeros(length(valid_ix))]

    arc_susceptances = _extract_arc_susceptances(BA.data)

    return VirtualFactorCore(
        K,
        BA.data,
        A.data,
        arc_susceptances,
        axes,
        look_up,
        valid_ix,
        bus_to_valid_idx,
        subnetwork_axes,
        cutoff,
        ybus.network_reduction_data,
        temp_data,
        work_ba_col,
        ReentrantLock(),
        system_uuid,
        Float64[],
        Vector{Vector{Float64}}(),
    )
end
