"""
    FactorCore{Ax, L, K, N}

Container for the shared state of the virtual sensitivity matrices. It owns the single
factorization, the topology, the solve scratch buffers, and the solver lock so that several
wrappers can be built on top of one factorization instead of each rebuilding (and
re-factorizing) the same DC network.

The expensive shared derived quantity `PTDF_A_diag` is computed lazily and cached here, so
that sharing one core between several wrappers computes it only once.

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
- `arc_live_members::Vector{Int}`: members an outage can open on each arc.
- `axes::Ax`: `(arc_axis, bus_axis)`.
- `lookup::L`: `(arc_lookup, bus_lookup)`.
- `valid_ix::Vector{Int}`: non-reference bus indices.
- `bus_to_valid_idx::Vector{Int}`: inverse of `valid_ix` (0 for reference buses).
- `subnetwork_axes::Dict{Int, Ax}`: per-reference-bus subnetwork axes (bus form).
- `tol::SparsificationCutoff`: resolved per-row sparsification rule (absolute
  cutoff for a `Float64` `tol`, relative cutoff for an [`AutoTolerance`](@ref)).
- `tol_spec::Union{Float64, AutoTolerance}`: the `tol` keyword as given; a factor
  file stores it and a reader resolves it again.
- `temp_data::Vector{Vector{Float64}}`: single-slot scratch (size n_buses).
- `work_ba_col::Vector{Vector{Float64}}`: single-slot scratch (size n_valid).
- `solver_lock::ReentrantLock`: serializes solves + scratch access.
- `system_uuid::Union{Base.UUID, Nothing}`: originating system UUID.
- `PTDF_A_diag::Vector{Float64}`: raw `H[e,e]`; empty until first
  `get_PTDF_A_diag(core)`.
- `PTDF_A_diag_ready::Threads.Atomic{Bool}`: publication flag for `PTDF_A_diag`.
- `network_data::N`: data the caller attaches; the core does not read it.

# Publication of the lazy fields
A lazy vector is filled under solver_lock and published by its Atomic{Bool} flag, the only
thing the lock-free path reads: a vector is non-empty before its contents are written.
"""
struct FactorCore{Ax, L <: NTuple{2, Dict}, K, N}
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
    tol_spec::Union{Float64, AutoTolerance}
    temp_data::Vector{Vector{Float64}}
    work_ba_col::Vector{Vector{Float64}}
    solver_lock::ReentrantLock
    system_uuid::Union{Base.UUID, Nothing}
    PTDF_A_diag::Vector{Float64}
    PTDF_A_diag_ready::Threads.Atomic{Bool}
    network_data::N
end

# --- Accessors (the wrappers forward to these) ---

get_axes(c::FactorCore) = c.axes
get_lookup(c::FactorCore) = c.lookup
get_arc_lookup(c::FactorCore) = c.lookup[1]
get_bus_lookup(c::FactorCore) = c.lookup[2]
get_arc_axis(c::FactorCore) = c.axes[1]
get_bus_axis(c::FactorCore) = c.axes[2]
get_ref_bus(c::FactorCore) = sort!(collect(keys(c.subnetwork_axes)))
get_system_uuid(c::FactorCore) = c.system_uuid
# Resolved cutoff object (used for per-row sparsification); `get_tol` returns its
# Float64 representative for display/serialization.
get_cutoff(c::FactorCore) = c.tol
get_tol(c::FactorCore) = cutoff_value(c.tol)
get_tol_spec(c::FactorCore) = c.tol_spec
get_network_data(c::FactorCore) = c.network_data

# Woodbury / NetworkModification kernel accessors.
_get_BA(c::FactorCore) = c.BA
_get_arc_susceptances(c::FactorCore) = c.arc_susceptances
_get_valid_ix(c::FactorCore) = c.valid_ix

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
    get_PTDF_A_diag(c::FactorCore) -> Vector{Float64}

Return the raw diagonal `H[e, e]` of `PTDF · A`, computing it (one solve per
arc) on first access and caching it on the core. Subsequent calls — including
from other wrappers sharing this core — return the cached vector.
"""
function get_PTDF_A_diag(c::FactorCore)
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

# --- Index and arc helpers ---

"""
Checks if repetitions are present in the dictionary mapping buses and branches.

# Arguments
- `ax::AbstractVector`:
        generic abstract vector
"""
function make_ax_ref(ax::AbstractVector)
    ref = Dict{eltype(ax), Int}()
    for (ix, el) in enumerate(ax)
        if haskey(ref, el)
            @error("Repeated index element $el. Index sets must have unique elements.")
        end
        ref[el] = ix
    end
    return ref
end

# Sign of an arc's DC susceptance: BA holds it at the arc's from bus, the bus `A` marks +1.
function _arc_susceptance_sign(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    e::Int,
)::Float64
    rng = SparseArrays.nzrange(BA, e)
    isempty(rng) && return 1.0
    p = first(rng)
    return sign(SparseArrays.nonzeros(BA)[p]) * A[e, SparseArrays.rowvals(BA)[p]]
end

# Scatter a valid-bus solve `lin_solve` (optionally divided) back into full-bus-space `dest`,
# zeroing the reference-bus entries first.
function _gather_to_buses!(
    dest::AbstractVector{Float64},
    valid_ix::Vector{Int},
    lin_solve,
    divisor::Float64 = 1.0,
)
    fill!(dest, 0.0)
    @inbounds for i in eachindex(valid_ix)
        dest[valid_ix[i]] = lin_solve[i] / divisor
    end
    return dest
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

# ponytail: no real network has |b| near 1e12 (the smallest reactance gives about 1e6); raise
# the limit if a valid case needs more.
const MAX_ARC_SUSCEPTANCE = 1e12

# The residual is measured against the right-hand side, so a garbage solution of large norm
# cannot hide it. A healthy solve on ACTIVSg2000 leaves 4e-11 (KLU) and 6e-11 (AppleAccelerate),
# and a singular factorization that a solver does not report leaves a residual of order 1.
# A near-singular but non-singular ABA passes: its solution is correct to the data's conditioning.
# A backward-stable solve leaves about eps * |ABA| * |x|, which grows with the largest
# susceptance (ACTIVSg10k: 2e-9, and 8e-8 with one arc at b = 1e6). The limit therefore scales
# with that product, between a floor of 1e-6 and an absolute cap that a singular solve exceeds.
const MIN_FACTORIZATION_RESIDUAL = 1e-6
const MAX_FACTORIZATION_RESIDUAL = 1e-1
const FACTORIZATION_RESIDUAL_SLACK = 1e3

_is_singular_error(::LinearAlgebra.SingularException) = true
_is_singular_error(::Exception) = false

# AppleAccelerate reports no column, so its `SingularException` carries 0.
_singular_column_text(info::Integer) = ifelse(info > 0, " (column $info)", "")

function _create_verified_factorization(
    solver::LinearSolverType,
    ABA::SparseArrays.SparseMatrixCSC{Float64, Int},
)
    K = try
        _create_factorization(solver, ABA)
    catch e
        _is_singular_error(e) || rethrow()
        error(
            "The ABA matrix of the factor file is singular$(_singular_column_text(e.info)). " *
            "The file has a susceptance that disconnects a subnetwork or that cancels another.",
        )
    end
    rhs = ones(size(ABA, 1))
    x = copy(_solve_factorization(K, copy(rhs)))
    residual = LinearAlgebra.norm(ABA * x - rhs, Inf) / LinearAlgebra.norm(rhs, Inf)
    limit = clamp(
        FACTORIZATION_RESIDUAL_SLACK * eps() * LinearAlgebra.opnorm(ABA, Inf) *
        LinearAlgebra.norm(x, Inf),
        MIN_FACTORIZATION_RESIDUAL,
        MAX_FACTORIZATION_RESIDUAL,
    )
    (isfinite(residual) && residual <= limit) || error(
        "The factorization of the ABA matrix of the factor file failed its check: relative " *
        "residual $residual, limit $limit. The matrix is singular or " *
        "near singular.",
    )
    return K
end

function _position_map(axis::Vector{T}, what::String) where {T}
    positions = Dict{T, Int}()
    sizehint!(positions, length(axis))
    for (ix, x) in enumerate(axis)
        haskey(positions, x) &&
            error(
                "The $what axis has the entry $x twice (positions $(positions[x]) and $ix).",
            )
        positions[x] = ix
    end
    return positions
end

function _find_root!(parent::Vector{Int}, x::Int)
    while parent[x] != x
        parent[x] = parent[parent[x]]
        x = parent[x]
    end
    return x
end

"""
    _validate_dc_arrays(arc_susceptances, ref_bus_positions, axes, subnetwork_axes,
        arc_live_members, reverse_bus_map)

Check that the DC arrays describe a consistent network, before the ABA matrix is built. The
function throws an error with context if an axis has a duplicate entry, an arc is a self-loop
or has an endpoint outside the bus axis, a susceptance is not finite or is larger than
`MAX_ARC_SUSCEPTANCE`, or `arc_live_members` has an entry below 1. A subnetwork key resolves
through `reverse_bus_map` to a bus of the axis, and that bus must be the reference bus. It
also throws an error if the subnetworks do not partition the buses and the arcs, a subnetwork
does not hold exactly one reference bus, or the buses of a subnetwork are not exactly one
connected component of the graph of arcs with non-zero susceptance. The cost is O(n α(n)).
"""
function _validate_dc_arrays(
    arc_susceptances::Vector{Float64},
    ref_bus_positions::Set{Int},
    axes::Tuple{Vector{Tuple{Int, Int}}, Vector{Int}},
    subnetwork_axes::Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}},
    arc_live_members::Vector{Int},
    reverse_bus_map::Dict{Int, Int},
)
    arc_ax, bus_ax = axes
    n_bus = length(bus_ax)
    n_arc = length(arc_ax)
    bus_position = _position_map(bus_ax, "bus")
    arc_position = _position_map(arc_ax, "arc")
    for arc in arc_ax
        (haskey(bus_position, arc[1]) && haskey(bus_position, arc[2])) ||
            error("Arc $arc has an endpoint outside the bus axis.")
        arc[1] == arc[2] && error("Arc $arc is a self-loop.")
    end
    length(arc_susceptances) == n_arc || error(
        "arc_susceptances has length $(length(arc_susceptances)); expected n_arc = $n_arc.",
    )
    for (i, b) in enumerate(arc_susceptances)
        abs(b) <= MAX_ARC_SUSCEPTANCE || error(
            "The DC susceptance of the arc $(arc_ax[i]) at position $i is $b; " *
            "it is not finite or has a magnitude above $MAX_ARC_SUSCEPTANCE.",
        )
    end
    length(arc_live_members) == n_arc || error(
        "arc_live_members has length $(length(arc_live_members)); expected n_arc = $n_arc.",
    )
    bad_live = findfirst(<(1), arc_live_members)
    isnothing(bad_live) || error(
        "The arc $(arc_ax[bad_live]) has arc_live_members $(arc_live_members[bad_live]), " *
        "below 1. A factor file cannot hold an opened arc.",
    )
    length(ref_bus_positions) == length(subnetwork_axes) || error(
        "$(length(ref_bus_positions)) reference bus positions for " *
        "$(length(subnetwork_axes)) subnetworks; expected one for each subnetwork.",
    )
    all(pos -> 1 <= pos <= n_bus, ref_bus_positions) ||
        error("Reference bus positions $ref_bus_positions must lie in 1:$n_bus.")

    subnetwork_of_bus = zeros(Int, n_bus)
    for (s, (key, (_, sub_buses))) in enumerate(subnetwork_axes)
        isempty(sub_buses) && error("Subnetwork $key has no buses.")
        for bus in sub_buses
            pos = get(bus_position, bus, 0)
            iszero(pos) && error("Subnetwork $key has the bus $bus outside the bus axis.")
            iszero(subnetwork_of_bus[pos]) ||
                error("The bus $bus is in more than one subnetwork.")
            subnetwork_of_bus[pos] = s
        end
    end
    all(!iszero, subnetwork_of_bus) ||
        error("Bus $(bus_ax[findfirst(iszero, subnetwork_of_bus)]) is in no subnetwork.")

    reference_of_subnetwork = zeros(Int, length(subnetwork_axes))
    for pos in ref_bus_positions
        s = subnetwork_of_bus[pos]
        iszero(reference_of_subnetwork[s]) || error(
            "Subnetwork with the reference bus $(bus_ax[pos]) has a second reference bus.",
        )
        reference_of_subnetwork[s] = pos
    end
    for (s, key) in enumerate(keys(subnetwork_axes))
        pos = get(bus_position, get(reverse_bus_map, key, key), 0)
        iszero(pos) && error(
            "Subnetwork key $key is not on the bus axis and no reduction maps it onto the axis.",
        )
        pos == reference_of_subnetwork[s] || error(
            "Subnetwork key $key resolves to a bus that is not the reference bus of its subnetwork.",
        )
    end

    seen = falses(n_arc)
    for (s, (key, (sub_arcs, _))) in enumerate(subnetwork_axes)
        for arc in sub_arcs
            ix = get(arc_position, arc, 0)
            iszero(ix) && error("Subnetwork $key has the arc $arc outside the arc axis.")
            seen[ix] && error("The arc $arc is in more than one subnetwork.")
            seen[ix] = true
            (
                subnetwork_of_bus[bus_position[arc[1]]] == s &&
                subnetwork_of_bus[bus_position[arc[2]]] == s
            ) || error("The arc $arc of subnetwork $key has an endpoint outside it.")
        end
    end
    all(seen) || error("Arc $(arc_ax[findfirst(!, seen)]) is in no subnetwork.")

    parent = collect(1:n_bus)
    for (ix, arc) in enumerate(arc_ax)
        iszero(arc_susceptances[ix]) && continue
        parent[_find_root!(parent, bus_position[arc[1]])] =
            _find_root!(parent, bus_position[arc[2]])
    end
    component_size = zeros(Int, n_bus)
    for pos in 1:n_bus
        component_size[_find_root!(parent, pos)] += 1
    end
    for (key, (_, sub_buses)) in subnetwork_axes
        root = _find_root!(parent, bus_position[first(sub_buses)])
        (
            component_size[root] == length(sub_buses) &&
            all(bus -> _find_root!(parent, bus_position[bus]) == root, sub_buses)
        ) || error(
            "The buses of subnetwork $key are not one connected component of the " *
            "graph of arcs with non-zero susceptance.",
        )
    end
    return
end

"""
    FactorCore(BA, A, ref_bus_positions, axes, subnetwork_axes, arc_live_members;
        network_data, linear_solver, tol, system_uuid, verify_factorization) -> FactorCore

Build the shared factorization core from the DC arrays.

`BA` is the `n_bus × n_arc` BA matrix. `A` is the `n_arc × n_bus` incidence matrix with
entries in `{-1, 0, 1}`. Each entry of `ref_bus_positions` is an index into the bus axis.
`axes` is `(arc_axis, bus_axis)`. `arc_live_members[i]` is the number of members that an
outage can open on arc `i`. The constructor holds `network_data` and does not read it.

The `tol` keyword is the sparsification spec, a `Float64` or an `AutoTolerance`. The
constructor keeps it in `tol_spec` and resolves the per-row cutoff from it. With
`verify_factorization = true`, the constructor checks the residual of a solve against the ABA
matrix and throws an error if the factorization is singular or near singular.
"""
function FactorCore(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    ref_bus_positions::Set{Int},
    axes::Tuple{Vector{Tuple{Int, Int}}, Vector{Int}},
    subnetwork_axes::Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}},
    arc_live_members::Vector{Int};
    network_data,
    linear_solver::String = _default_linear_solver(),
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    system_uuid::Union{Base.UUID, Nothing} = nothing,
    verify_factorization::Bool = false,
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
    all(pos -> 1 <= pos <= n_bus, ref_bus_positions) ||
        error("Reference bus positions $ref_bus_positions must lie in 1:$n_bus.")
    solver = resolve_linear_solver(linear_solver)
    _check_virtual_solver(solver)
    ABA = calculate_ABA_matrix(A, BA, ref_bus_positions)
    iszero(size(ABA, 1)) && error(
        "The ABA matrix is empty: every bus is a reference bus.",
    )
    lookup = (make_ax_ref(axes[1]), make_ax_ref(axes[2]))
    if length(subnetwork_axes) > 1
        @info "Network is not connected, using subnetworks"
    end

    K = if verify_factorization
        _create_verified_factorization(solver, ABA)
    else
        _create_factorization(solver, ABA)
    end

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

    return FactorCore(
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
        tol,
        temp_data,
        work_ba_col,
        ReentrantLock(),
        system_uuid,
        Float64[],
        Threads.Atomic{Bool}(false),
        network_data,
    )
end

# --- PTDF row kernel ---

"""
    ptdf_row!(dest, K, work_ba_col, core, row) -> dest

Write the PTDF row `row` of `core` into `dest`, in full-bus space with the reference-bus
entries 0. `K` and `work_ba_col` are the factorization and scratch that `with_solver` gives;
the caller holds the solver lock.
"""
function ptdf_row!(
    dest::Vector{Float64},
    K,
    work_ba_col::Vector{Float64},
    core::FactorCore,
    row::Int,
)
    lin_solve = _solve_ba_column!(K, work_ba_col, core.BA, core.bus_to_valid_idx, row)
    return _gather_to_buses!(dest, core.valid_ix, lin_solve)
end
