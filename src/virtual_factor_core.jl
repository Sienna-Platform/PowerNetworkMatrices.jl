"""
Branch data that PNM attaches to a [`VirtualFactorCore`](@ref). The factorization core
holds it and does not read it.

# Fields
- `branch_catalog::BranchCatalog`: reduction maps.
- `branch_susceptances_by_arc::Vector{Vector{Float64}}`: per-branch susceptances;
  empty until first `get_branch_susceptances_by_arc(core)`.
- `branch_susceptances_ready::Threads.Atomic{Bool}`: publication flag for
  `branch_susceptances_by_arc`.
"""
struct CoreNetworkData
    branch_catalog::BranchCatalog
    branch_susceptances_by_arc::Vector{Vector{Float64}}
    branch_susceptances_ready::Threads.Atomic{Bool}
end

CoreNetworkData(branch_catalog::BranchCatalog) =
    CoreNetworkData(branch_catalog, Vector{Vector{Float64}}(), Threads.Atomic{Bool}(false))

"""
    VirtualFactorCore{Ax, L, K}

The factorization core of the virtual matrices (`VirtualPTDF`, `VirtualMODF`,
`VirtualLODF`) with PNM's branch data attached: a `PowerMatricesCore.FactorCore` whose
`network_data` is a [`CoreNetworkData`](@ref). The core owns the single factorization, the
topology, the solve scratch, and the solver lock, so that all three wrappers share one
factorization of the DC network. The per-branch susceptances are computed lazily and cached
in the branch data, so a core shared between wrappers computes them only once.

The type parameter order `{Ax, L, K}` matches `VirtualPTDF` so the PTDF/MODF
wrappers can carry `core::VirtualFactorCore{Ax, L, K}` with the same parameters.
"""
const VirtualFactorCore =
    PMC.FactorCore{Ax, L, K, CoreNetworkData} where {Ax, L <: NTuple{2, Dict}, K}

# --- Accessors that need the branch data ---

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
get_branch_catalog(c::VirtualFactorCore) = get_network_data(c).branch_catalog
holds_components(c::VirtualFactorCore) = holds_components(get_branch_catalog(c))
get_network_reduction_data(c::VirtualFactorCore) =
    get_network_reduction_data(get_branch_catalog(c))

# Field reads that the wrappers forward to the core. The branch data is not a field of the
# core, so its names resolve through the attached `CoreNetworkData`.
function _core_property(c::VirtualFactorCore, name::Symbol)
    if name in fieldnames(CoreNetworkData)
        return getfield(get_network_data(c), name)
    end
    return getfield(c, name)
end

"""
    get_branch_susceptances_by_arc(c::VirtualFactorCore) -> Vector{Vector{Float64}}

Return the per-branch susceptances for each arc, computing them on first access
and caching them on the core.
"""
function get_branch_susceptances_by_arc(c::VirtualFactorCore)
    holds_components(c) || error(
        "get_branch_susceptances_by_arc needs the branches of each arc, and a matrix " *
        "restored from a factor file has none.",
    )
    data = get_network_data(c)
    return _publish_once!(
        data.branch_susceptances_by_arc,
        data.branch_susceptances_ready,
        c.solver_lock,
    ) do
        _extract_branch_susceptances_by_arc(c.BA, c.axes[1], get_network_reduction_data(c))
    end
end

# --- Constructors ---

"""
    VirtualFactorCore(BA, A, ref_bus_positions, axes, subnetwork_axes, arc_live_members,
        branch_catalog; linear_solver, tol, system_uuid) -> VirtualFactorCore

Build the shared factorization core from the DC arrays. No `System` is needed.

`BA` is the `n_bus × n_arc` BA matrix. `A` is the `n_arc × n_bus` incidence matrix with
entries in `{-1, 0, 1}`. Each entry of `ref_bus_positions` is an index into the bus axis.
`axes` is `(arc_axis, bus_axis)`. `arc_live_members[i]` is the number of members that an
outage can open on arc `i`. The constructor holds `branch_catalog` and does not read it.

The `tol` keyword is the sparsification spec, a `Float64` or an `AutoTolerance`. The
constructor keeps it in `tol_spec` and resolves the per-row cutoff from it. With
`verify_factorization = true`, the constructor checks the residual of a solve against the ABA
matrix and throws an error if the factorization is singular or near singular.
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
    tol::Union{Float64, AutoTolerance} = DEFAULT_AUTO_TOLERANCE,
    system_uuid::Union{Base.UUID, Nothing} = nothing,
    verify_factorization::Bool = false,
)
    return PMC.FactorCore(
        BA,
        A,
        ref_bus_positions,
        axes,
        subnetwork_axes,
        arc_live_members;
        network_data = CoreNetworkData(branch_catalog),
        linear_solver = linear_solver,
        tol = tol,
        system_uuid = system_uuid,
        verify_factorization = verify_factorization,
    )
end

"""
    VirtualFactorCore(arc_susceptances, ref_bus_positions, axes, subnetwork_axes,
        arc_live_members, branch_catalog; linear_solver, tol, system_uuid) -> VirtualFactorCore

Build the shared factorization core from one signed DC susceptance per arc. The method builds
the `BA` and `A` matrices from `axes` and `arc_susceptances`, then calls the matrix method.
Entry `i` of `arc_susceptances` is the susceptance of arc `i`, positive from the first bus of
the arc to the second.
"""
function VirtualFactorCore(
    arc_susceptances::Vector{Float64},
    ref_bus_positions::Set{Int},
    axes::Tuple{Vector{Tuple{Int, Int}}, Vector{Int}},
    subnetwork_axes::Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}},
    arc_live_members::Vector{Int},
    branch_catalog::BranchCatalog;
    kwargs...,
)
    return PMC.FactorCore(
        arc_susceptances,
        ref_bus_positions,
        axes,
        subnetwork_axes,
        arc_live_members;
        network_data = CoreNetworkData(branch_catalog),
        reverse_bus_map = get_reverse_bus_search_map(
            get_network_reduction_data(branch_catalog),
        ),
        kwargs...,
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
