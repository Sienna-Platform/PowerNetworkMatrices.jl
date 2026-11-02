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
Solves on one core serialize through its `solver_lock` (acquired via
`with_solver`), which protects the single scratch slot in
`temp_data`/`work_ba_col` and the core's one factorization. A core shared between
multiple wrappers therefore serializes their solves, which is the existing
single-scratch model.

# Fields
- `K`: ABA factorization (`KLULinSolveCache{Float64}` or `AAFactorCache`).
- `BA::SparseMatrixCSC{Float64, Int}`: BA matrix.
- `A::SparseMatrixCSC{Int8, Int}`: incidence matrix data.
- `arc_susceptances::Vector{Float64}`: effective susceptance per arc.
- `arc_susceptance_signs::Vector{Float64}`: sign of each arc's DC susceptance
  (`_arc_susceptance_sign`).
- `arc_live_members::Vector{Int}`: members an outage can open on each arc, as
  `Ybus.arc_live_members`.
- `axes::Ax`: `(arc_axis, bus_axis)`.
- `lookup::L`: `(arc_lookup, bus_lookup)`.
- `valid_ix::Vector{Int}`: non-reference bus indices.
- `bus_to_valid_idx::Vector{Int}`: inverse of `valid_ix` (0 for reference buses).
- `subnetwork_axes::Dict{Int, Ax}`: per-reference-bus subnetwork axes (bus form).
- `tol::SparsificationCutoff`: resolved per-row sparsification rule (absolute
  cutoff for a `Float64` `tol`, relative cutoff for an [`AutoTolerance`](@ref)).
- `branch_catalog::BranchCatalog`: reduction maps.
- `temp_data::Vector{Vector{Float64}}`: single-slot scratch (size n_buses).
- `work_ba_col::Vector{Vector{Float64}}`: single-slot scratch (size n_valid).
- `solver_lock::ReentrantLock`: serializes solves + scratch access.
- `system_uuid::Union{Base.UUID, Nothing}`: originating system UUID.
- `PTDF_A_diag::Vector{Float64}`: raw `H[e,e]`; empty until first
  `get_PTDF_A_diag(core)`.
- `PTDF_A_diag_ready::Threads.Atomic{Bool}`: publication flag for `PTDF_A_diag`.
- `branch_susceptances_by_arc::Vector{Vector{Float64}}`: per-branch susceptances;
  empty until first `get_branch_susceptances_by_arc(core)`.
- `branch_susceptances_ready::Threads.Atomic{Bool}`: publication flag for
  `branch_susceptances_by_arc`.

# Publication of the lazy fields
Both lazy vectors are filled under solver_lock and published by their
Atomic{Bool} flag, the only thing the lock-free path reads: a vector is
non-empty before its contents are written.
"""
struct VirtualFactorCore{Ax, L <: NTuple{2, Dict}, K}
    K::K
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    arc_susceptances::Vector{Float64}
    arc_susceptance_signs::Vector{Float64}
    arc_live_members::Vector{Int}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    bus_to_valid_idx::Vector{Int}
    subnetwork_axes::Dict{Int, Ax}
    tol::SparsificationCutoff
    branch_catalog::BranchCatalog
    temp_data::Vector{Vector{Float64}}
    work_ba_col::Vector{Vector{Float64}}
    solver_lock::ReentrantLock
    system_uuid::Union{Base.UUID, Nothing}
    PTDF_A_diag::Vector{Float64}
    PTDF_A_diag_ready::Threads.Atomic{Bool}
    branch_susceptances_by_arc::Vector{Vector{Float64}}
    branch_susceptances_ready::Threads.Atomic{Bool}
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
get_branch_catalog(c::VirtualFactorCore) = c.branch_catalog
get_network_reduction_data(c::VirtualFactorCore) =
    get_network_reduction_data(get_branch_catalog(c))
get_system_uuid(c::VirtualFactorCore) = c.system_uuid
# Resolved cutoff object (used for per-row sparsification); `get_tol` returns its
# Float64 representative for display/serialization.
get_cutoff(c::VirtualFactorCore) = c.tol
get_tol(c::VirtualFactorCore) = cutoff_value(c.tol)

# Woodbury / NetworkModification kernel accessors.
_get_BA(c::VirtualFactorCore) = c.BA
_get_arc_susceptances(c::VirtualFactorCore) = c.arc_susceptances
_get_valid_ix(c::VirtualFactorCore) = c.valid_ix

# --- Lazy shared derived quantities ---

# Double-checked publish for a lazy shared vector: `compute` runs once under `lock` and
# `dest` is filled in place before `ready` is set, so a throw mid-compute leaves the flag
# unset and a retry does not append a second copy.
function _publish_once!(
    compute,
    dest::Vector,
    ready::Threads.Atomic{Bool},
    lock::ReentrantLock,
)
    ready[] && return dest
    @lock lock begin
        ready[] && return dest
        empty!(dest)
        append!(dest, compute())
        ready[] = true
        return dest
    end
end

"""
    get_PTDF_A_diag(c::VirtualFactorCore) -> Vector{Float64}

Return the raw diagonal `H[e, e]` of `PTDF · A`, computing it (one solve per
arc) on first access and caching it on the core. Subsequent calls — including
from other wrappers sharing this core — return the cached vector.
"""
function get_PTDF_A_diag(c::VirtualFactorCore)
    return _publish_once!(c.PTDF_A_diag, c.PTDF_A_diag_ready, c.solver_lock) do
        n_arcs = length(c.axes[1])
        @info "Computing PTDF_A_diag on first access ($n_arcs arcs)."
        t0 = time_ns()
        new_diag = _get_PTDF_A_diag(c.K, c.BA, c.A, c.valid_ix, c.bus_to_valid_idx)
        elapsed = (time_ns() - t0) / 1e9
        @info "Computed PTDF_A_diag in $(round(elapsed; digits = 2)) s (cached)."
        new_diag
    end
end

"""
    get_branch_susceptances_by_arc(c::VirtualFactorCore) -> Vector{Vector{Float64}}

Return the per-branch susceptances for each arc, computing them on first access
and caching them on the core.
"""
function get_branch_susceptances_by_arc(c::VirtualFactorCore)
    return _publish_once!(
        c.branch_susceptances_by_arc,
        c.branch_susceptances_ready,
        c.solver_lock,
    ) do
        _extract_branch_susceptances_by_arc(c.BA, c.axes[1], get_network_reduction_data(c))
    end
end

# --- Constructor ---

# The virtual matrices solve through `_solve_factorization`, which has KLU and
# AppleAccelerate arms only.
_check_virtual_solver(::LinearSolverType) = nothing

function _check_virtual_solver(::MKLPardisoSolver)
    return error(
        "VirtualPTDF, VirtualLODF, and VirtualMODF support the \"KLU\" and " *
        "\"AppleAccelerateLU\" solvers; got \"MKLPardiso\". Use " *
        "`PTDF(sys; linear_solver = \"MKLPardiso\")` for a dense matrix.",
    )
end

"""
    VirtualFactorCore(BA, A, ref_bus_positions, axes, subnetwork_axes, arc_live_members,
        branch_catalog; linear_solver, tol, system_uuid) -> VirtualFactorCore

Build the shared factorization core from the DC arrays. No `System` is needed.

`BA` is the `n_bus × n_arc` BA matrix. `A` is the `n_arc × n_bus` incidence matrix with
entries in `{-1, 0, 1}`. Each entry of `ref_bus_positions` is an index into the bus axis.
`axes` is `(arc_axis, bus_axis)`. `arc_live_members[i]` is the number of members that an
outage can open on arc `i`. The constructor holds `branch_catalog` and does not read it.

The `tol` keyword accepts a `Float64`, an `AutoTolerance`, or a resolved cutoff. A
resolved cutoff is stored as given, so a saved cutoff restores unchanged.
"""
function VirtualFactorCore(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    ref_bus_positions::Set{Int},
    axes::Tuple{Vector{Tuple{Int, Int}}, Vector{Int}},
    subnetwork_axes::Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}},
    arc_live_members::Vector{Int},
    branch_catalog::BranchCatalog;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance, SparsificationCutoff} = DEFAULT_AUTO_TOLERANCE,
    system_uuid::Union{Base.UUID, Nothing} = nothing,
)
    n_arc = length(axes[1])
    n_bus = length(axes[2])
    size(BA) == (n_bus, n_arc) || error(
        "BA has size $(size(BA)); expected (n_bus, n_arc) = ($n_bus, $n_arc).",
    )
    size(A) == (n_arc, n_bus) || error(
        "A has size $(size(A)); expected (n_arc, n_bus) = ($n_arc, $n_bus).",
    )
    length(arc_live_members) == n_arc || error(
        "arc_live_members has length $(length(arc_live_members)); expected n_arc = $n_arc.",
    )
    for pos in ref_bus_positions
        1 <= pos <= n_bus ||
            error("Reference bus position $pos is outside 1:$n_bus.")
    end
    solver = resolve_linear_solver(linear_solver)
    _check_virtual_solver(solver)
    ABA = calculate_ABA_matrix(A, BA, ref_bus_positions)
    lookup = (make_ax_ref(axes[1]), make_ax_ref(axes[2]))
    if length(subnetwork_axes) > 1
        @info "Network is not connected, using subnetworks"
    end

    K = _create_factorization(solver, ABA)

    # Resolve the per-row sparsification rule once: a Float64 becomes an absolute
    # cutoff, an AutoTolerance a relative cutoff (κ logged via the reused K).
    cutoff = _resolve_virtual_cutoff(tol, K, ABA, SparseArrays.nonzeros(BA))

    valid_ix = setdiff(1:n_bus, ref_bus_positions)
    bus_to_valid_idx = _build_bus_to_valid_idx(n_bus, valid_ix)

    # Single scratch slot: solves serialize through `solver_lock`. Kept as a
    # `Vector{Vector{Float64}}` so the `with_solver` callback signature
    # stays uniform across solver backends.
    temp_data = [zeros(n_bus)]
    work_ba_col = [zeros(length(valid_ix))]

    arc_susceptances = _extract_arc_susceptances(BA)
    arc_susceptance_signs =
        Float64[_arc_susceptance_sign(BA, A, l) for l in 1:size(BA, 2)]

    return VirtualFactorCore(
        K,
        BA,
        A,
        arc_susceptances,
        arc_susceptance_signs,
        arc_live_members,
        axes,
        lookup,
        valid_ix,
        bus_to_valid_idx,
        subnetwork_axes,
        cutoff,
        branch_catalog,
        temp_data,
        work_ba_col,
        ReentrantLock(),
        system_uuid,
        Float64[],
        Threads.Atomic{Bool}(false),
        Vector{Vector{Float64}}(),
        Threads.Atomic{Bool}(false),
    )
end

"""
    VirtualFactorCore(ybus::Ybus; linear_solver, tol, system_uuid) -> VirtualFactorCore

Build the shared factorization core from a `Ybus`: the incidence matrix, BA matrix, ABA matrix,
and its factorization. `worker_core` rebuilds the ABA matrix and factorization of an existing core.
"""
function VirtualFactorCore(
    ybus::Ybus;
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    system_uuid::Union{Base.UUID, Nothing} = nothing,
)
    A = IncidenceMatrix(ybus)
    BA = BA_Matrix(ybus)
    return VirtualFactorCore(
        BA.data,
        A.data,
        Set(get_ref_bus_position(ybus)),
        A.axes,
        A.subnetwork_axes,
        copy(get_arc_live_members(ybus)),
        get_branch_catalog(ybus);
        linear_solver = linear_solver,
        tol = tol,
        system_uuid = system_uuid,
    )
end
