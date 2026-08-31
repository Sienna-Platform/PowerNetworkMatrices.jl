"""
    Ybus{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{YBUS_ELTYPE}

Nodal admittance matrix (Y-bus) representing the electrical admittance relationships between
buses in a power system. This N×N sparse complex matrix encodes the network topology and
electrical parameters needed for power flow calculations and network analysis.

# Fields
- `data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int}`: Sparse Y-bus matrix with complex admittance values
- `adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int}`: Network connectivity information
- `axes::Ax`: Tuple of bus axis vectors for indexing (bus_numbers, bus_numbers)
- `lookup::L`: Tuple of lookup dictionaries mapping bus numbers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`: Bus axes for each electrical island/subnetwork
- `arc_subnetwork_axis::Dict{Int, Vector{Tuple{Int, Int}}}`: Arc axes for each subnetwork
- `branch_catalog::BranchCatalog`: Metadata from network reduction operations
- `arc_admittance_from_to::Union{ArcAdmittanceMatrix, Nothing}`: From-to arc admittance matrix
- `arc_admittance_to_from::Union{ArcAdmittanceMatrix, Nothing}`: To-from arc admittance matrix

# Key Features
- Indexed by bus numbers (non-sequential numbering supported)
- Supports network reductions (radial, degree-two, Ward)
- Handles multiple electrical islands/subnetworks
- Optional arc admittance matrices for power flow calculations
- Sparse matrix representation for computational efficiency

# Usage
The Y-bus is fundamental for:
- Power flow analysis: V = Y⁻¹I
- Short circuit calculations
- Network impedance analysis
- Sensitivity analysis (PTDF/LODF)

# Examples
```julia
# Basic Y-bus construction
ybus = Ybus(system)

# With arc admittance matrices for power flow
ybus = Ybus(system; make_arc_admittance_matrices=true)

# With network reductions
ybus = Ybus(system; network_reductions=[RadialReduction(), DegreeTwoReduction()])
```

# See Also
- [`PTDF`](@ref): Power Transfer Distribution Factors
- [`LODF`](@ref): Line Outage Distribution Factors
- [`NetworkReduction`](@ref): Network reduction algorithms
"""
struct Ybus{Ax <: NTuple{2, Vector}, L <: NTuple{2, Dict}} <:
       PowerNetworkMatrix{YBUS_ELTYPE}
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int}
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    arc_subnetwork_axis::Dict{Int, Vector{Tuple{Int, Int}}}
    branch_catalog::BranchCatalog
    arc_admittance_from_to::Union{ArcAdmittanceMatrix, Nothing}
    arc_admittance_to_from::Union{ArcAdmittanceMatrix, Nothing}
end

get_axes(M::Ybus) = M.axes
get_lookup(M::Ybus) = M.lookup
get_ref_bus(M::Ybus) = sort!(collect(keys(M.subnetwork_axes)))

"""Get the [`BranchCatalog`](@ref) recording every branch merged into this matrix."""
get_branch_catalog(M::Ybus) = M.branch_catalog
get_bus_axis(M::Ybus) = M.axes[1]
get_bus_lookup(M::Ybus) = M.lookup[1]

"""
    get_isolated_buses(M::Ybus) -> Vector{Int}

Return bus numbers that form isolated single-node subnetworks in the Y-bus matrix.

Isolated buses are electrical islands containing only one bus with no connections to
other parts of the network. These typically represent buses that were disconnected
during network reduction operations or buses with no active branches.

# Arguments
- `M::Ybus`: Y-bus matrix to analyze

# Returns
- `Vector{Int}`: Vector of bus numbers that form isolated single-node subnetworks

# Examples
```julia
ybus = Ybus(system)
isolated = get_isolated_buses(ybus)
println("Isolated buses: ", isolated)
```
"""
function get_isolated_buses(M::Ybus)
    return [x for x in keys(M.subnetwork_axes) if length(M.subnetwork_axes[x][1]) == 1]
end

"""
    get_default_reduction(sys::PSY.System) -> NetworkReductionData

Build a Y-bus matrix from the system and return its default network reduction data.

This function constructs a Y-bus matrix with no network reductions applied and returns
the resulting `NetworkReductionData`, which contains the basic bus and branch mappings
for the system without any reduction algorithms.

# Arguments
- `sys::PSY.System`: Power system to analyze

# Returns
- `NetworkReductionData`: Default network reduction data with basic system mappings

# Examples
```julia
system = System("system.json")
reduction_data = get_default_reduction(system)
println("Number of buses: ", length(get_bus_reduction_map(reduction_data)))
```

# See Also
- [`Ybus`](@ref): Y-bus matrix construction
- [`NetworkReductionData`](@ref): Network reduction data structure
"""
function get_default_reduction(sys::PSY.System)
    ybus = Ybus(sys)
    return get_network_reduction_data(ybus)
end

"""
    get_reduction(ybus::Ybus, sys::PSY.System, reduction::RadialReduction) -> NetworkReductionData

Apply radial network reduction to a Y-bus matrix.

Radial reduction eliminates radial (dangling) buses that have only one connection.
These buses do not affect power flows in the rest of the network and can be safely
removed to reduce computational complexity.

# Arguments
- `ybus::Ybus`: Y-bus matrix to reduce
- `sys::PSY.System`: Power system for validation
- `reduction::RadialReduction`: Radial reduction configuration

# Returns
- `NetworkReductionData`: Reduction data containing eliminated buses and updated mappings

# Examples
```julia
ybus = Ybus(system; irreducible_buses=Set([101, 205]))
reduction = RadialReduction()
reduction_data = get_reduction(ybus, system, reduction)
```

# See Also
- [`RadialReduction`](@ref): Radial reduction configuration
- [`get_reduction`](@ref): Other reduction methods
"""
function get_reduction(ybus::Ybus, sys::PSY.System, reduction::RadialReduction)
    A = IncidenceMatrix(ybus)
    return get_reduction(A, sys, reduction)
end

function _make_parallel_branch_pair(
    a::T,
    b::T,
    ::Tuple{Int, Int},
) where {T <: PSY.ACTransmission}
    return BranchesParallel(T[a, b])
end

function _make_parallel_branch_pair(
    a::PSY.ACTransmission,
    b::PSY.ACTransmission,
    arc_tuple::Tuple{Int, Int},
)
    @warn "Mismatch in parallel device types for arc $(arc_tuple). This could indicate issues in the network data."
    return MixedBranchesParallel(PSY.ACTransmission[a, b])
end

function _push_parallel_branch!(
    parallel_branch_map::Dict,
    arc_tuple::Tuple{Int, Int},
    br::PSY.ACTransmission,
)
    existing = parallel_branch_map[arc_tuple]
    _push_parallel_branch_dispatch!(parallel_branch_map, arc_tuple, existing, br)
    return
end

function _push_parallel_branch_dispatch!(
    ::Dict,
    ::Tuple{Int, Int},
    existing::BranchesParallel{T},
    br::T,
) where {T <: PSY.ACTransmission}
    add_branch!(existing, br)
    return
end

function _push_parallel_branch_dispatch!(
    parallel_branch_map::Dict,
    arc_tuple::Tuple{Int, Int},
    existing::BranchesParallel,
    br::PSY.ACTransmission,
)
    @warn "Mismatch in parallel device types for arc $(arc_tuple). This could indicate issues in the network data."
    parallel_branch_map[arc_tuple] = MixedBranchesParallel(
        PSY.ACTransmission[existing.branches..., br],
        existing.arc_key,
        EMPTY_TWO_PORT,
        false,
    )
    return
end

function _push_parallel_branch_dispatch!(
    ::Dict,
    arc_tuple::Tuple{Int, Int},
    existing::MixedBranchesParallel,
    br::PSY.ACTransmission,
)
    if !any(typeof(b) === typeof(br) for b in existing.branches)
        @warn "Mismatch in parallel device types for arc $(arc_tuple). This could indicate issues in the network data."
    end
    add_branch!(existing, br)
    return
end

"""
    add_to_branch_maps!(nr::NetworkReductionData, arc::PSY.Arc, br::PSY.ACTransmission)

Add an AC transmission branch to the appropriate branch mapping in NetworkReductionData.

This function categorizes branches as direct (one-to-one), parallel (multiple branches
between same buses), or creates new mappings as needed. It maintains both forward and
reverse lookup dictionaries for efficient access.

# Arguments
- `nr::NetworkReductionData`: Network reduction data to modify
- `arc::PSY.Arc`: Arc representing the branch connection
- `br::PSY.ACTransmission`: AC transmission branch to add

# Implementation Details
- If arc already has a parallel group, pushes `br` into it (`_push_parallel_branch!`);
  a type mismatch against the group promotes it to `MixedBranchesParallel` and emits a `@warn`
- If arc already has a direct branch, promotes both branches into a new group
  (`_make_parallel_branch_pair`): homogeneous `BranchesParallel{T}` when types match,
  `MixedBranchesParallel` with a `@warn` otherwise
- Otherwise creates a new direct mapping
- Phase-shifting members are grouped like any other branch — never dropped or forced direct
  (issue #305)
- Maintains reverse lookup consistency
"""
function add_to_branch_maps!(
    nr::NetworkReductionData,
    arc::PSY.Arc,
    br::T,
) where {T <: PSY.ACTransmission}
    direct_branch_map = get_direct_branch_map(nr)
    reverse_direct_branch_map = get_reverse_direct_branch_map(nr)
    parallel_branch_map = get_parallel_branch_map(nr)
    reverse_parallel_branch_map = get_reverse_parallel_branch_map(nr)
    arc_tuple = get_arc_tuple(arc, nr)
    if haskey(parallel_branch_map, arc_tuple)
        _push_parallel_branch!(parallel_branch_map, arc_tuple, br)
        reverse_parallel_branch_map[br] = arc_tuple
    elseif haskey(direct_branch_map, arc_tuple)
        corresponding_branch = direct_branch_map[arc_tuple]
        delete!(direct_branch_map, arc_tuple)
        delete!(reverse_direct_branch_map, corresponding_branch)
        parallel_branch_map[arc_tuple] =
            _make_parallel_branch_pair(corresponding_branch, br, arc_tuple)
        reverse_parallel_branch_map[corresponding_branch] = arc_tuple
        reverse_parallel_branch_map[br] = arc_tuple
    else
        direct_branch_map[arc_tuple] = br
        reverse_direct_branch_map[br] = arc_tuple
    end
    return
end

"""
    add_to_branch_maps!(nr::NetworkReductionData, br::PSY.ThreeWindingTransformer)

Add a three-winding transformer to the branch maps in NetworkReductionData.

Three-winding transformers are modeled using a star (wye) configuration with one arc per
circuit connecting to a virtual star bus.

Each circuit is filed through the same merge-aware path as any other `PSY.ACTransmission`
(the 3-arg `add_to_branch_maps!`), so a winding whose star-point arc coincides with an
already-registered branch (a `Line`, another winding, or an existing parallel group) is
merged into a parallel group rather than silently overwriting the earlier entry.

Arc keys resolve through `get_arc_tuple`, which remaps both endpoints, so two winding
circuits could land on the same key if a reduction merges the transformer's terminal buses.

# Arguments
- `nr::NetworkReductionData`: Network reduction data to modify
- `br::PSY.ThreeWindingTransformer`: Three-winding transformer to add

# Implementation Details
- Only adds arcs for available circuits (per-circuit `PSY.get_available`)
- Circuits are numbered in `PSY.get_circuits` order (1=primary, 2=secondary, 3=tertiary)
"""
function add_to_branch_maps!(
    nr::NetworkReductionData,
    br::PSY.ThreeWindingTransformer,
)
    for (i, circuit) in enumerate(PSY.get_circuits(br))
        if PSY.get_available(circuit)
            winding = ThreeWindingTransformerCircuit(br, circuit, i)
            add_to_branch_maps!(nr, PSY.get_arc(circuit), winding)
        end
    end
    return
end

"""
    add_branch_entries_to_ybus!(
        y11::Vector{YBUS_ELTYPE},
        y12::Vector{YBUS_ELTYPE},
        y21::Vector{YBUS_ELTYPE},
        y22::Vector{YBUS_ELTYPE},
        branch_ix::Int,
        br::PSY.ACTransmission
    )

Add Y-bus matrix entries for an AC transmission branch to the admittance vectors.

This function calculates the 2×2 Y-bus entries for a branch using `ybus_branch_entries()`
and stores them in the provided vectors at the specified index. The entries represent
the Pi-model admittances: Y11 (from-bus self), Y12 (from-to mutual), Y21 (to-from mutual),
and Y22 (to-bus self).

# Arguments
- `y11::Vector{YBUS_ELTYPE}`: Vector for from-bus self admittances
- `y12::Vector{YBUS_ELTYPE}`: Vector for from-to mutual admittances
- `y21::Vector{YBUS_ELTYPE}`: Vector for to-from mutual admittances
- `y22::Vector{YBUS_ELTYPE}`: Vector for to-bus self admittances
- `branch_ix::Int`: Index where to store the branch entries
- `br::PSY.ACTransmission`: AC transmission branch

# Implementation Details
- Calls `ybus_branch_entries()` to compute Pi-model parameters
- Stores results directly in the provided vectors
- Used during Y-bus matrix assembly process
"""
function add_branch_entries_to_ybus!(
    y11::Vector{YBUS_ELTYPE},
    y12::Vector{YBUS_ELTYPE},
    y21::Vector{YBUS_ELTYPE},
    y22::Vector{YBUS_ELTYPE},
    branch_ix::Int,
    br::PSY.ACTransmission;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    Y11, Y12, Y21, Y22 = ybus_branch_entries(br; min_x_eps = min_x_eps)
    y11[branch_ix] = Y11
    y12[branch_ix] = Y12
    y21[branch_ix] = Y21
    y22[branch_ix] = Y22
    return
end

"""
    add_branch_entries_to_indexing_maps!(
        num_bus::Dict{Int, Int},
        branch_ix::Int,
        nr::NetworkReductionData,
        fb::Vector{Int},
        tb::Vector{Int},
        br::PSY.ACTransmission
    )

Update indexing structures when adding an AC transmission branch to the Y-bus.

This function handles the bookkeeping required when adding a branch: updates network
reduction mappings and records from/to bus indices for the branch in the Y-bus
construction vectors.

# Arguments
- `num_bus::Dict{Int, Int}`: Mapping from bus numbers to matrix indices
- `branch_ix::Int`: Branch index in the vectors
- `nr::NetworkReductionData`: Network reduction data to update
- `fb::Vector{Int}`: Vector of from-bus indices
- `tb::Vector{Int}`: Vector of to-bus indices
- `br::PSY.ACTransmission`: AC transmission branch to add

# Implementation Details
- Calls `add_to_branch_maps!()` to update reduction mappings
- Records bus indices in from/to vectors for sparse matrix construction
"""
function add_branch_entries_to_indexing_maps!(
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    nr::NetworkReductionData,
    fb::Vector{Int},
    tb::Vector{Int},
    br::PSY.ACTransmission,
)
    arc = PSY.get_arc(br)
    add_to_branch_maps!(nr, arc, br)
    bus_from_no, bus_to_no = get_bus_indices(arc, num_bus, nr)
    fb[branch_ix] = bus_from_no
    tb[branch_ix] = bus_to_no
    return
end

"""Ybus 2x2 for any single branch — line, Ward equivalent, or transformer circuit of either
arity. The π-model comes from [`branch_admittance`](@ref), the single source of truth;
`min_x_eps` substitutes for `x` when `r == x == 0`. Aggregates (parallel groups, series
chains) have their own methods below: for those Ybus is the primitive and the π-model is
derived from it, not the reverse."""
function ybus_branch_entries(
    br::PSY.ACTransmission;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    adm = branch_admittance(br; min_x_eps = min_x_eps)
    Y11, Y12, Y21, Y22 = _pi_to_ybus(adm)
    if !isfinite(Y11) || !isfinite(complex(adm.g, adm.b))
        error(
            "Data in $(get_name(br)) gives a non-finite Ybus entry. " *
            "g = $(adm.g), b = $(adm.b), tap = $(adm.tap), shift = $(adm.shift)",
        )
    end
    return (Y11, Y12, Y21, Y22)
end

# A single branch has unambiguous orientation; this `nr` overload lets callers iterating
# heterogeneous segments (single branches and parallel groups) pass `nr` uniformly.
function ybus_branch_entries(
    br::PSY.ACTransmission,
    ::NetworkReductionData;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    return ybus_branch_entries(br; min_x_eps = min_x_eps)
end

function ybus_branch_entries(
    parallel_br::AbstractBranchesParallel,
    nr::NetworkReductionData;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    # Pass the group itself, not `collect(parallel_br)`: `collect` yields a `Vector{Any}`
    # (only `BranchesSeries` defines `eltype`), which both allocates per call on the Ybus
    # assembly path and forces the loop's calls dynamic.
    return _subset_two_port(parallel_br, get_arc_tuple(parallel_br, nr), nr)
end

function ybus_branch_entries(
    br::BranchesSeries,
    nr::NetworkReductionData;
    min_x_eps::Float64 = ZERO_IMPEDANCE_X_EPSILON,
)
    ybus_chain = _build_chain_ybus(br, nr)
    ybus_reduced = _reduce_internal_nodes(ybus_chain)
    return ybus_reduced[1, 1], ybus_reduced[1, 2], ybus_reduced[2, 1], ybus_reduced[2, 2]
end

# ZIBR's substitute reactance for r=x=0 branches; falls back when no spec is on the NRD.
function _minimum_retained_impedance(nr::NetworkReductionData)
    zir = get_zero_impedance_reduction(get_reductions(nr))
    if isnothing(zir)
        return ZERO_IMPEDANCE_X_EPSILON
    end
    return get_minimum_retained_impedance(zir)
end

"""Handles ybus entries for most 2-node AC branches. The types handled here are:
`Line`, `DiscreteControlledACBranch`, and `TwoWindingTransformer`.
"""
function _ybus!(
    y11::Vector{YBUS_ELTYPE},
    y12::Vector{YBUS_ELTYPE},
    y21::Vector{YBUS_ELTYPE},
    y22::Vector{YBUS_ELTYPE},
    br::PSY.ACTransmission,
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    nr::NetworkReductionData,
)
    add_branch_entries_to_indexing_maps!(num_bus, branch_ix, nr, fb, tb, br)
    add_branch_entries_to_ybus!(
        y11, y12, y21, y22, branch_ix, br;
        min_x_eps = _minimum_retained_impedance(nr),
    )
    return
end

function _ybus!(
    y11::Vector{YBUS_ELTYPE},
    y12::Vector{YBUS_ELTYPE},
    y21::Vector{YBUS_ELTYPE},
    y22::Vector{YBUS_ELTYPE},
    br::PSY.ThreeWindingTransformer,
    num_bus::Dict{Int, Int},
    offset_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    nr::NetworkReductionData,
)
    add_to_branch_maps!(nr, br)
    min_x_eps = _minimum_retained_impedance(nr)
    n_entries = 0
    for (i, circuit) in enumerate(PSY.get_circuits(br))
        PSY.get_available(circuit) || continue
        term_ix, star_ix = get_bus_indices(PSY.get_arc(circuit), num_bus, nr)
        fb[offset_ix + ix + n_entries] = term_ix
        tb[offset_ix + ix + n_entries] = star_ix
        (Y11, Y12, Y21, Y22) = ybus_branch_entries(
            ThreeWindingTransformerCircuit(br, circuit, i);
            min_x_eps = min_x_eps,
        )
        y11[offset_ix + ix + n_entries] = Y11
        y12[offset_ix + ix + n_entries] = Y12
        y21[offset_ix + ix + n_entries] = Y21
        y22[offset_ix + ix + n_entries] = Y22
        n_entries += 1
    end
    return n_entries
end

function _ybus!(
    ysh::Vector{YBUS_ELTYPE},
    fa::PSY.FixedAdmittance,
    num_bus::Dict{Int, Int},
    fa_ix::Int,
    sb::Vector{Int},
    nr::NetworkReductionData,
)
    bus_no = get_bus_index(fa, num_bus, nr)
    Y = PSY.get_Y(fa)
    sb[fa_ix] = bus_no
    if !isfinite(Y)
        error("Data in $(PSY.get_name(fa)) is incorrect. Y = $(Y)")
    end
    ysh[fa_ix] = Y
    return
end

#Note - PSSE does not include switched admittances in ymatrix
function _ybus!(
    ysh::Vector{YBUS_ELTYPE},
    fa::PSY.SwitchedAdmittance,
    num_bus::Dict{Int, Int},
    fa_ix::Int,
    sb::Vector{Int},
    nr::NetworkReductionData,
)
    bus_no = get_bus_index(fa, num_bus, nr)
    sb[fa_ix] = bus_no
    ysh[fa_ix] = 0.0
    return
end

function _ybus!(
    ysh::Vector{YBUS_ELTYPE},
    fa::PSY.StandardLoad,
    num_bus::Dict{Int, Int},
    fa_ix::Int,
    sb::Vector{Int},
    nr::NetworkReductionData,
)
    bus_no = get_bus_index(fa, num_bus, nr)
    Y =
        PSY.get_impedance_active_power(fa, PSY.SU) -
        im * PSY.get_impedance_reactive_power(fa, PSY.SU)
    if !isfinite(Y)
        error("Data in $(PSY.get_name(fa)) is incorrect. Y = $(Y)")
    end
    sb[fa_ix] = bus_no
    ysh[fa_ix] = Y
    return
end

function _buildybus!(
    network_reduction_data::NetworkReductionData,
    branches::YbusACBranches,
    transformer_3w::Vector{PSY.ThreeWindingTransformer},
    num_bus::Dict{Int, Int},
    fixed_admittances::Vector{PSY.FixedAdmittance},
    switched_admittances::Vector{PSY.SwitchedAdmittance},
    standard_loads::Vector{PSY.StandardLoad},
)
    branch_entries_transformer_3w = 0
    for br in transformer_3w
        branch_entries_transformer_3w += count(PSY.get_available, PSY.get_circuits(br))
    end
    branchcount = length(branches) + branch_entries_transformer_3w
    branchcount_no_3w = length(branches)
    fa_count = length(fixed_admittances)
    sa_count = length(switched_admittances)
    sl_count = length(standard_loads)
    fb = zeros(Int, branchcount)
    tb = zeros(Int, branchcount)
    sb = zeros(Int, fa_count + sa_count + sl_count)

    y11 = zeros(YBUS_ELTYPE, branchcount)
    y12 = zeros(YBUS_ELTYPE, branchcount)
    y21 = zeros(YBUS_ELTYPE, branchcount)
    y22 = zeros(YBUS_ELTYPE, branchcount)
    ysh = zeros(YBUS_ELTYPE, fa_count + sa_count + sl_count)

    _foreach_ybus_branch(branches) do b, ix
        if PSY.get_name(b) == "init"
            throw(DataFormatError("The data in Branch is invalid"))
        end
        _ybus!(y11, y12, y21, y22, b, num_bus, ix, fb, tb, network_reduction_data)
    end

    ix = 1
    for b in transformer_3w
        if PSY.get_name(b) == "init"
            throw(DataFormatError("The data in ThreeWindingTransformer is invalid"))
        end
        n_entries = _ybus!(
            y11,
            y12,
            y21,
            y22,
            b,
            num_bus,
            branchcount_no_3w,
            fb,
            tb,
            ix,
            network_reduction_data,
        )
        ix += n_entries
    end
    shunt_ix = 0
    for fa in fixed_admittances
        shunt_ix += 1
        _ybus!(ysh, fa, num_bus, shunt_ix, sb, network_reduction_data)
    end
    for sa in switched_admittances
        shunt_ix += 1
        _ybus!(ysh, sa, num_bus, shunt_ix, sb, network_reduction_data)
    end
    for sl in standard_loads
        shunt_ix += 1
        _ybus!(ysh, sl, num_bus, shunt_ix, sb, network_reduction_data)
    end
    return (y11, y12, y21, y22, ysh, fb, tb, sb)
end

function _is_available_shunt(x::PSY.StaticInjection)::Bool
    return PSY.get_available(x) && PSY.get_bustype(PSY.get_bus(x)) != ACBusTypes.ISOLATED
end

function _get_filtered_components(
    ::Type{T},
    sys::PSY.System,
    predicate::F,
)::Vector{T} where {T <: PSY.Component, F <: Function}
    iter = PSY.get_components(T, sys)
    components = sizehint!(Vector{T}(), length(iter))
    for comp in iter
        predicate(comp) && push!(components, comp)
    end
    return components
end

"""
    Ybus(
        sys::PSY.System;
        make_arc_admittance_matrices::Bool = false,
        network_reductions::Vector{NetworkReduction} = NetworkReduction[],
        include_constant_impedance_loads::Bool = true,
        subnetwork_algorithm = iterative_union_find,
        kwargs...
    ) -> Ybus

Construct a nodal admittance matrix (Y-bus) from a power system.

Builds the sparse complex Y-bus matrix representing the electrical admittance relationships
between buses in the power system. Handles AC branches, transformers, shunt elements,
and network reductions while maintaining connectivity analysis.

# Arguments
- `sys::PSY.System`: Power system to build Y-bus from

# Keyword arguments
- `make_arc_admittance_matrices::Bool=false`: Whether to construct arc admittance matrices for power flow
- `network_reductions::Vector{NetworkReduction}=[]`: Network reduction algorithms to apply
- `include_constant_impedance_loads::Bool=true`: Whether to include constant impedance loads as shunt admittances
- `subnetwork_algorithm=iterative_union_find`: Algorithm for finding electrical islands

# Returns
- `Ybus`: Constructed Y-bus matrix with network topology and electrical parameters

# Features
- **Branch Support**: Lines, transformers, phase shifters, three-winding transformers
- **Shunt Elements**: Fixed admittances, switched admittances, constant impedance loads
- **Network Reductions**: Radial, degree-two, Ward reductions for computational efficiency
- **Multiple Islands**: Handles disconnected network components with separate reference buses
- **Branch Matrices**: Optional from-to/to-from admittance matrices for power flow calculations

# Examples
```julia
# Basic Y-bus construction
ybus = Ybus(system)

# With arc admittance matrices for power flow
ybus = Ybus(system; make_arc_admittance_matrices=true)

# Apply network reductions for computational efficiency
reductions = [RadialReduction(), DegreeTwoReduction()]
ybus = Ybus(system; network_reductions=reductions)

# Exclude constant impedance loads
ybus = Ybus(system; include_constant_impedance_loads=false)
```

# See Also
- [`NetworkReduction`](@ref): Network reduction algorithms
- [`PTDF`](@ref): Power transfer distribution factors
- [`LODF`](@ref): Line outage distribution factors
"""
# Re-impose the original last-write-wins orientation (`adj[i,j]=1; adj[j,i]=-1` per
# branch) for bus pairs carrying anti-parallel branches, whose +1/-1 contributions
# cancel or flip in the summed (COO) adjacency. Entries are set explicitly, so a
# cancelled position is restored to ±1 whether or not a zero was stored there —
# guaranteeing the connection survives a later `dropzeros!`. Warns once per affected
# bus pair.
function _resolve_antiparallel_adjacency!(
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    fb::Vector{Int},
    tb::Vector{Int},
    bus_ax::Vector{Int},
)
    last_orientation = Dict{Tuple{Int, Int}, Tuple{Int, Int}}()
    antiparallel = Set{Tuple{Int, Int}}()
    for k in eachindex(fb)
        i = fb[k]
        j = tb[k]
        i == j && continue
        canonical = minmax(i, j)
        prev = get(last_orientation, canonical, (0, 0))
        if !iszero(prev[1]) && prev != (i, j)
            push!(antiparallel, canonical)
        end
        last_orientation[canonical] = (i, j)  # last write wins, matching the original
    end
    for canonical in antiparallel
        i, j = last_orientation[canonical]
        adj[i, j] = one(Int8)
        adj[j, i] = -one(Int8)
        @warn "Anti-parallel branches between buses $(bus_ax[i]) and $(bus_ax[j]) " *
              "detected; retaining the last branch's orientation in the bus " *
              "adjacency matrix used for connectivity checks."
    end
    return
end

function Ybus(
    sys::PSY.System;
    make_arc_admittance_matrices::Bool = false,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    irreducible_buses = Set{Int}(),
    include_constant_impedance_loads = true,
    subnetwork_algorithm = iterative_union_find,
)
    network_reductions, zero_impedance_reduction =
        _split_zero_impedance_reduction(network_reductions)
    user_irreducible = Set{Int}(irreducible_buses)
    ref_bus_numbers = Set{Int}()
    # Stored angles of the swing (REF) buses, used to pick the smallest-angle swing as the
    # representative when an island holds more than one (see `assign_reference_buses!`).
    ref_bus_angles = Dict{Int, Float64}()
    # Seed the user set and ZIBR spec into the container so every reduction step and
    # the assembly path can read them.
    nr = NetworkReductionData(;
        reductions = ReductionContainer(;
            user_irreducible_buses = user_irreducible,
            zero_impedance_reduction = zero_impedance_reduction,
        ),
    )
    bus_reduction_map = get_bus_reduction_map(nr)
    reverse_bus_search_map = get_reverse_bus_search_map(nr)

    #Checking for isolated buses; building bus map.
    for b in PSY.get_components(PSY.ACBus, sys)
        !PSY.get_available(b) && continue
        if PSY.get_bustype(b) != ACBusTypes.ISOLATED
            bus_reduction_map[PSY.get_number(b)] = Set{Int}()
            if PSY.get_bustype(b) == ACBusTypes.REF
                push!(ref_bus_numbers, PSY.get_number(b))
                ref_bus_angles[PSY.get_number(b)] = PSY.get_angle(b)
            end
        else
            @debug "Found available isolated bus $(PSY.get_name(b)) with number $(PSY.get_number(b)). This is excluded from the Ybus build."
            push!(nr.removed_buses, PSY.get_number(b))
        end
    end

    bus_ax = sort!(collect(keys(bus_reduction_map)))
    axes = (bus_ax, bus_ax)
    bus_lookup = Dict{Int, Int}()
    lookup = (bus_lookup, bus_lookup)
    busnumber = length(bus_ax)
    for (ix, b) in enumerate(bus_ax)
        bus_lookup[b] = ix
    end
    branches = _get_ybus_two_terminal_ac_branches(sys)
    transformer_3W =
        _get_filtered_components(PSY.ThreeWindingTransformer, sys, PSY.get_available)
    fixed_admittances =
        _get_filtered_components(PSY.FixedAdmittance, sys, _is_available_shunt)
    switched_admittances =
        _get_filtered_components(PSY.SwitchedAdmittance, sys, _is_available_shunt)
    standard_loads = if include_constant_impedance_loads
        _get_filtered_components(PSY.StandardLoad, sys, _is_available_shunt)
    else
        PSY.StandardLoad[]
    end
    y11, y12, y21, y22, ysh, fb, tb, sb = _buildybus!(
        nr,
        branches,
        transformer_3W,
        bus_lookup,
        fixed_admittances,
        switched_admittances,
        standard_loads,
    )
    # The branch maps are final here; the arc-admittance matrices below share this index.
    catalog = BranchCatalog(nr)
    # Build adjacency matrix from COO triplets in a single sparse() call to avoid
    # ~2×branchcount structural insertions into a growing CSC matrix.
    # Values: diagonal = +1, forward arc (from→to) = +1, reverse arc (to→from) = -1.
    # Parallel branches (same orientation) produce duplicate (i,j) entries that sum;
    # clamp back to ±1 via sign. Anti-parallel branches are reconciled separately below.
    branchcount = length(fb)
    adj_I = Vector{Int}(undef, busnumber + 2 * branchcount)
    adj_J = Vector{Int}(undef, busnumber + 2 * branchcount)
    adj_V = Vector{Int8}(undef, busnumber + 2 * branchcount)
    @inbounds for k in 1:busnumber
        adj_I[k] = k
        adj_J[k] = k
        adj_V[k] = one(Int8)
    end
    @inbounds for k in 1:branchcount
        adj_I[busnumber + k] = fb[k]
        adj_J[busnumber + k] = tb[k]
        adj_V[busnumber + k] = one(Int8)
        adj_I[busnumber + branchcount + k] = tb[k]
        adj_J[busnumber + branchcount + k] = fb[k]
        adj_V[busnumber + branchcount + k] = -one(Int8)
    end
    adj = SparseArrays.sparse(adj_I, adj_J, adj_V, busnumber, busnumber)
    map!(sign, adj.nzval, adj.nzval)
    # Anti-parallel bus pairs cancel to a zero in the summed adjacency. The diagonal is
    # always +1, so any zero here flags such a cancellation. When that happens, re-impose
    # ±1 (last-write-wins, matching the original incremental build) right now — before
    # `adj` is used for connectivity or copied/`dropzeros!`-ed by AdjacencyMatrix — so the
    # connection is never carried by a zero that could be eliminated. The check is a
    # cheap scan of `adj.nzval` (no per-branch bookkeeping); `_resolve_*` sets the entries
    # explicitly and emits the per-pair warning. (We use the zero as a signal here, but do
    # not depend on it surviving downstream.)
    any(iszero, adj.nzval) && _resolve_antiparallel_adjacency!(adj, fb, tb, bus_ax)
    ybus = SparseArrays.sparse(
        [fb; fb; tb; tb; sb],  # row indices
        [fb; tb; fb; tb; sb],  # column indices
        [y11; y12; y21; y22; ysh],  # values
        busnumber,  # size (rows) - setting this explicitly is necessary for the case there are no branches
        busnumber,  # size (columns) - setting this explicitly is necessary for the case there are no branches
    )
    SparseArrays.dropzeros!(ybus)

    if make_arc_admittance_matrices
        arc_axis = get_arc_axis(fb, tb, bus_ax)
        arc_count = length(arc_axis)
        arc_lookup = sizehint!(Dict{Tuple{Int, Int}, Int}(), arc_count)
        for (ix, arc_tuple) in enumerate(arc_axis)
            arc_lookup[arc_tuple] = ix
        end
        rows_ix = [arc_lookup[(x, y)] for (x, y) in zip(bus_ax[fb], bus_ax[tb])]
        rows_ix_nnz = vcat(rows_ix, rows_ix)
        yft_data =
            SparseArrays.sparse(rows_ix_nnz, [fb; tb], [y11; y12], arc_count, busnumber)
        ytf_data =
            SparseArrays.sparse(rows_ix_nnz, [tb; fb], [y22; y21], arc_count, busnumber)
        arc_admittance_from_to = ArcAdmittanceMatrix(
            yft_data,
            (arc_axis, bus_ax),
            (arc_lookup, bus_lookup),
            catalog,
            :FromTo,
        )
        arc_admittance_to_from = ArcAdmittanceMatrix(
            ytf_data,
            (arc_axis, bus_ax),
            (arc_lookup, bus_lookup),
            catalog,
            :ToFrom,
        )
    else
        arc_admittance_from_to = nothing
        arc_admittance_to_from = nothing
    end
    if length(bus_lookup) > 1
        subnetworks = assign_reference_buses!(
            find_subnetworks(ybus, bus_ax; subnetwork_algorithm = subnetwork_algorithm),
            ref_bus_numbers,
            ref_bus_angles,
        )
        if length(subnetworks) > 1
            @warn "More than one island found; Network is not connected"
        end
    else
        subnetworks = Dict{Int, Set{Int}}(only(bus_ax) => Set(bus_ax))
    end
    subnetwork_axes = _make_bus_subnetwork_axes(subnetworks)
    arc_subnetwork_axis = _make_arc_subnetwork_axis(subnetworks, nr)
    ybus = Ybus(
        ybus,
        adj,
        axes,
        lookup,
        subnetwork_axes,
        arc_subnetwork_axis,
        catalog,
        arc_admittance_from_to,
        arc_admittance_to_from,
    )
    ybus = build_reduced_ybus(ybus, sys, zero_impedance_reduction)
    for reduction in network_reductions
        ybus = build_reduced_ybus(ybus, sys, reduction)
    end
    return ybus
end

"""
    get_arc_axis(fb::Vector{Int}, tb::Vector{Int}, bus_axis::Vector{Int}) -> Vector{Tuple{Int, Int}}

Generate unique arc axis from from-bus and to-bus index vectors.

Creates a vector of unique (from_bus, to_bus) tuples representing the arcs (branches)
in the system. Used for constructing arc admittance matrices and organizing
network topology data.

# Arguments
- `fb::Vector{Int}`: Vector of from-bus indices into bus_axis
- `tb::Vector{Int}`: Vector of to-bus indices into bus_axis
- `bus_axis::Vector{Int}`: Vector of bus numbers

# Returns
- `Vector{Tuple{Int, Int}}`: Unique arcs as (from_bus_number, to_bus_number) tuples

# Examples
```julia
fb = [1, 2, 1]  # indices into bus_axis
tb = [2, 3, 3]  # indices into bus_axis
bus_axis = [101, 102, 103]  # bus numbers
arcs = get_arc_axis(fb, tb, bus_axis)
# Returns: [(101, 102), (102, 103), (101, 103)]
```

# Implementation Details
- Maps indices to actual bus numbers using bus_axis
- Removes duplicates with unique()
- Preserves arc direction (from → to)
"""
function get_arc_axis(fb::Vector{Int}, tb::Vector{Int}, bus_axis::Vector{Int})
    #TODO - handle arc axis consistently between BranchAdmittanceMatrices and IncidenceMatrix
    return unique(collect(zip(bus_axis[fb], bus_axis[tb])))
end

"""
    make_bus_arc_subnetwork_axes(ybus::Ybus) -> Dict{Int, Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}}

Create subnetwork axes for BA_Matrix construction from a Y-bus matrix.

Generates subnetwork-specific axes combining bus and arc information needed for
constructing Bus-Arc (BA) matrices. Each subnetwork gets its own bus list and
corresponding arc list for matrix indexing.

# Arguments
- `ybus::Ybus`: Y-bus matrix containing subnetwork information

# Returns
- `Dict{Int, Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}}`: Dictionary mapping reference bus numbers to (bus_axis, arc_axis) tuples for each subnetwork

# Implementation Details
- Combines bus axes from `ybus.subnetwork_axes` with arc axes from `ybus.arc_subnetwork_axis`
- Maintains consistency between bus and arc indexing within each electrical island
- Used for constructing BA matrices that relate bus injections to branch flows

# See Also
- [`BA_Matrix`](@ref): Bus-Arc matrix construction
- [`make_arc_bus_subnetwork_axes`](@ref): Arc-Bus variant
"""
function make_bus_arc_subnetwork_axes(ybus::Ybus)
    subnetwork_count = length(ybus.subnetwork_axes)
    subnetwork_axes =
        sizehint!(
            Dict{Int, Tuple{Vector{Int}, Vector{Tuple{Int, Int}}}}(),
            subnetwork_count,
        )
    for key in keys(ybus.subnetwork_axes)
        subnetwork_axes[key] = (ybus.subnetwork_axes[key][1], ybus.arc_subnetwork_axis[key])
    end
    return subnetwork_axes
end

"""
    make_arc_bus_subnetwork_axes(ybus::Ybus) -> Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}}

Create subnetwork axes for IncidenceMatrix construction from a Y-bus matrix.

Generates subnetwork-specific axes with arc-bus ordering needed for constructing
incidence matrices. Each subnetwork gets its own arc list and corresponding bus
list for matrix indexing.

# Arguments
- `ybus::Ybus`: Y-bus matrix containing subnetwork information

# Returns
- `Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}}`: Dictionary mapping reference bus numbers to (arc_axis, bus_axis) tuples for each subnetwork

# Implementation Details
- Swaps order compared to `make_bus_arc_subnetwork_axes` (arc first, bus second)
- Uses same underlying data from `ybus.subnetwork_axes` and `ybus.arc_subnetwork_axis`
- Used for constructing incidence matrices that relate branch connectivity to bus topology

# See Also
- [`IncidenceMatrix`](@ref): Network incidence matrix
- [`make_bus_arc_subnetwork_axes`](@ref): Bus-Arc variant
"""
function make_arc_bus_subnetwork_axes(ybus::Ybus)
    subnetwork_axes = Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}}()
    for key in keys(ybus.subnetwork_axes)
        subnetwork_axes[key] = (ybus.arc_subnetwork_axis[key], ybus.subnetwork_axes[key][1])
    end
    return subnetwork_axes
end

function _make_bus_subnetwork_axes(subnetworks::Dict{Int, Set{Int}})
    subnetwork_axes = Dict{Int, Tuple{Vector{Int}, Vector{Int}}}()
    for (k, v) in subnetworks
        subnetwork_axes[k] = (collect(v), collect(v))
    end
    return subnetwork_axes
end

function _make_arc_subnetwork_axis(
    subnetworks::Dict{Int, Set{Int}},
    nr::NetworkReductionData,
)
    arc_ax = get_arc_axis(nr)
    arc_subnetwork_axis = Dict{Int, Vector{Tuple{Int, Int}}}()
    for k in keys(subnetworks)
        arc_subnetwork_axis[k] = Vector{Tuple{Int, Int}}()
    end
    for arc in arc_ax
        for (k, v) in subnetworks
            if arc[1] ∈ v || arc[2] in v
                subnetwork = get!(arc_subnetwork_axis, k, Vector{Tuple{Int, Int}}())
                push!(subnetwork, arc)
                break
            end
        end
    end
    return arc_subnetwork_axis
end

"""
    build_reduced_ybus(
        ybus::Ybus,
        sys::PSY.System,
        network_reduction::NetworkReduction
    ) -> Ybus

Apply a network reduction algorithm to a Y-bus matrix.

Computes the network reduction data using the specified reduction algorithm and
then applies the reduction to create a new Y-bus matrix with eliminated buses
and branches. The electrical behavior of the remaining network is preserved.

# Arguments
- `ybus::Ybus`: Original Y-bus matrix to reduce
- `sys::PSY.System`: Power system for validation and data access
- `network_reduction::NetworkReduction`: Reduction algorithm to apply

# Returns
- `Ybus`: New reduced Y-bus matrix with eliminated elements

# Implementation Details
- Calls `get_reduction()` to compute elimination data
- Applies reduction via `_apply_reduction()`
- Preserves electrical equivalence of remaining network
- Updates all indexing and mapping structures

# Examples
```julia
ybus = Ybus(system)
reduction = RadialReduction()
reduced_ybus = build_reduced_ybus(ybus, system, reduction)
println("Original buses: ", length(get_bus_axis(ybus)))
println("Reduced buses: ", length(get_bus_axis(reduced_ybus)))
```

# See Also
- [`NetworkReduction`](@ref): Reduction algorithm types
- [`get_reduction`](@ref): Reduction data computation
"""
function build_reduced_ybus(
    ybus::Ybus,
    sys::PSY.System,
    network_reduction::NetworkReduction,
)
    validate_reduction_type(
        network_reduction,
        get_reductions(get_network_reduction_data(ybus)),
    )
    network_reduction_data = get_reduction(ybus, sys, network_reduction)
    isempty(network_reduction_data) && return ybus
    return _apply_reduction(ybus, network_reduction_data)
end

function _resolve_arc_admittance(
    new_y_ft::ArcAdmittanceMatrix,
    new_y_tf::ArcAdmittanceMatrix,
    existing_ft::Union{ArcAdmittanceMatrix, Nothing},
    existing_tf::Union{ArcAdmittanceMatrix, Nothing},
    removed_arcs::Set{Tuple{Int, Int}},
    bus_ax::Vector{Int},
    bus_lookup::Dict{Int, Int},
    bus_ix::Vector{Int},
    nr::NetworkReductionData,
    merged_bus_pairs::Dict{Int, Int} = Dict{Int, Int}(),
)
    arc_ax = setdiff(get_arc_axis(new_y_ft), removed_arcs)
    arc_remove_ixs = indexin(removed_arcs, get_arc_axis(new_y_ft))
    arc_keep_ixs = setdiff(collect(1:length(get_arc_axis(new_y_ft))), arc_remove_ixs)
    # Remap arc endpoint labels to surviving bus numbers. Column data was already merged
    # in _merge_arc_admittance_bus_columns before this call; only the axis labels need
    # updating so downstream reductions can match arcs by their new bus numbers.
    if !isempty(merged_bus_pairs)
        for k in eachindex(arc_ax)
            arc = arc_ax[k]
            new_from = get(merged_bus_pairs, arc[1], arc[1])
            new_to = get(merged_bus_pairs, arc[2], arc[2])
            (new_from == arc[1] && new_to == arc[2]) || (arc_ax[k] = (new_from, new_to))
        end
        # Drop self-loops that can arise if both endpoints map to the same surviving bus.
        valid = findall(arc -> arc[1] != arc[2], arc_ax)
        arc_ax = arc_ax[valid]
        arc_keep_ixs = arc_keep_ixs[valid]
        # Collapse duplicates that appear when a bus merge maps two winding arcs to the
        # same (from, to) label (e.g. primary and secondary windings both become (X, S)).
        # Sum their rows so the combined admittance is preserved, then drop the extras.
        seen_arc_positions = Dict{Tuple{Int, Int}, Int}()
        rows_to_drop = Int[]
        for k in eachindex(arc_ax)
            arc = arc_ax[k]
            if haskey(seen_arc_positions, arc)
                first_k = seen_arc_positions[arc]
                new_y_ft.data[arc_keep_ixs[first_k], :] += new_y_ft.data[arc_keep_ixs[k], :]
                new_y_tf.data[arc_keep_ixs[first_k], :] += new_y_tf.data[arc_keep_ixs[k], :]
                push!(rows_to_drop, k)
            else
                seen_arc_positions[arc] = k
            end
        end
        if !isempty(rows_to_drop)
            keep = setdiff(eachindex(arc_ax), rows_to_drop)
            arc_ax = arc_ax[keep]
            arc_keep_ixs = arc_keep_ixs[keep]
        end
    end
    arc_lookup = make_ax_ref(arc_ax)
    yft_data = new_y_ft.data[arc_keep_ixs, bus_ix]
    ytf_data = new_y_tf.data[arc_keep_ixs, bus_ix]

    catalog = BranchCatalog(nr)
    arc_admittance_from_to = ArcAdmittanceMatrix(
        yft_data,
        (arc_ax, bus_ax),
        (arc_lookup, bus_lookup),
        catalog,
        :FromTo,
    )
    arc_admittance_to_from = ArcAdmittanceMatrix(
        ytf_data,
        (arc_ax, bus_ax),
        (arc_lookup, bus_lookup),
        catalog,
        :ToFrom,
    )
    return arc_admittance_from_to, arc_admittance_to_from
end

function _resolve_arc_admittance(
    ::Nothing,
    ::Nothing,
    existing_ft::Union{ArcAdmittanceMatrix, Nothing},
    existing_tf::Union{ArcAdmittanceMatrix, Nothing},
    removed_arcs::Set{Tuple{Int, Int}},
    bus_ax::Vector{Int},
    bus_lookup::Dict{Int, Int},
    bus_ix::Vector{Int},
    nr::NetworkReductionData,
    merged_bus_pairs::Dict{Int, Int} = Dict{Int, Int}(),
)
    return existing_ft, existing_tf
end

# Add every removed bus's column into its surviving bus's column on one arc×bus admittance
# matrix, in a single O(nnz) relabel-and-sum pass. Each stored entry is emitted at its own
# column and, when that column is a removed bus, also at its survivor's column; `sparse` sums
# the collisions. Removed columns keep their own entries, since the bus-removal slice later
# drops them.
function _merge_arc_admittance_columns(
    M::SparseArrays.SparseMatrixCSC{T, Int},
    bus_lookup::Dict{Int, Int},
    merged_bus_pairs::Dict{Int, Int},
) where {T}
    ncols = size(M, 2)
    col_survivor = collect(1:ncols)
    for (removed_bus, surviving_bus) in merged_bus_pairs
        col_survivor[bus_lookup[removed_bus]] = bus_lookup[surviving_bus]
    end
    rows = SparseArrays.rowvals(M)
    vals = SparseArrays.nonzeros(M)
    cap = 2 * length(vals)
    I = Vector{Int}(undef, cap)
    J = Vector{Int}(undef, cap)
    V = Vector{T}(undef, cap)
    n = 0
    for col in 1:ncols
        s = col_survivor[col]
        for k in SparseArrays.nzrange(M, col)
            r = rows[k]
            v = vals[k]
            n += 1
            I[n] = r
            J[n] = col
            V[n] = v
            if s != col
                n += 1
                I[n] = r
                J[n] = s
                V[n] = v
            end
        end
    end
    return SparseArrays.sparse(
        resize!(I, n), resize!(J, n), resize!(V, n), size(M, 1), ncols)
end

# Merge the removed-bus columns into their survivors on both arc admittance matrices, returning
# new matrices. This transfer keeps the to-bus admittance entries for arcs terminating at a
# removed bus, which the later bus-removal slice would otherwise drop.
function _merge_arc_admittance_bus_columns(
    yft::ArcAdmittanceMatrix,
    ytf::ArcAdmittanceMatrix,
    bus_lookup::Dict{Int, Int},
    merged_bus_pairs::Dict{Int, Int},
)
    new_yft = ArcAdmittanceMatrix(
        _merge_arc_admittance_columns(yft.data, bus_lookup, merged_bus_pairs),
        yft.axes, yft.lookup, get_branch_catalog(yft), yft.direction)
    new_ytf = ArcAdmittanceMatrix(
        _merge_arc_admittance_columns(ytf.data, bus_lookup, merged_bus_pairs),
        ytf.axes, ytf.lookup, get_branch_catalog(ytf), ytf.direction)
    return new_yft, new_ytf
end

_merge_arc_admittance_bus_columns(
    ::Nothing,
    ::Nothing,
    ::Dict{Int, Int},
    ::Dict{Int, Int},
) = (nothing, nothing)

function _accumulate_csc_row_into!(M::SparseArrays.SparseMatrixCSC, i::Int, j::Int)
    rows = SparseArrays.rowvals(M)
    vals = SparseArrays.nonzeros(M)
    for col in 1:size(M, 2)
        v_j = zero(eltype(M))
        for k in SparseArrays.nzrange(M, col)
            rows[k] == j && (v_j = vals[k])
        end
        iszero(v_j) && continue
        found = false
        for k in SparseArrays.nzrange(M, col)
            if rows[k] == i
                vals[k] += v_j
                found = true
                break
            end
        end
        found || (M[i, col] += v_j)
    end
    return
end

function _accumulate_csc_col_into!(M::SparseArrays.SparseMatrixCSC, i::Int, j::Int)
    rows = SparseArrays.rowvals(M)
    vals = SparseArrays.nonzeros(M)
    # Snapshot column j up front. Structural inserts into column i shift the CSC backing
    # store, and the shift direction depends on whether i precedes or follows j; iterating
    # a captured copy of column j stays correct for either ordering. An offset-based version
    # is only valid when i < j (for i > j it drops a mutual term, giving an asymmetric Ybus).
    j_entries = [(rows[k], vals[k]) for k in SparseArrays.nzrange(M, j)]
    for (r, v_j) in j_entries
        iszero(v_j) && continue
        found = false
        # Re-read i_range each iteration: a structural insert changes where column i lives.
        i_range = SparseArrays.nzrange(M, i)
        for k_i in i_range
            if rows[k_i] == r
                vals[k_i] += v_j
                found = true
                break
            end
        end
        found || (M[r, i] += v_j)
    end
    return
end

function _merge_ybus_buses!(
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int},
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int},
    bus_lookup::Dict{Int, Int},
    merged_bus_pairs::Dict{Int, Int},
)
    for (removed_bus, surviving_bus) in merged_bus_pairs
        i = bus_lookup[surviving_bus]
        j = bus_lookup[removed_bus]
        _accumulate_csc_row_into!(data, i, j)
        _accumulate_csc_col_into!(data, i, j)
        _accumulate_csc_row_into!(adjacency_data, i, j)
        _accumulate_csc_col_into!(adjacency_data, i, j)
    end
    # Anti-parallel branches cancel in the signed adjacency (+1/-1 sum to 0) and get dropped
    # by `dropzeros!`, hiding a real edge from DegreeTwoReduction. The complex Ybus data sums
    # and never cancels structurally, so re-derive each surviving bus's adjacency from it.
    _repair_merged_adjacencies!(adjacency_data, data, bus_lookup, merged_bus_pairs)
    return
end

# Re-derive every surviving bus's signed adjacency from the complex Ybus data. Shared by the
# in-place merge (`_merge_ybus_buses!`) and the fused pure-merge rebuild so the two paths
# cannot drift. A merged off-diagonal can sum to a stored zero (e.g. a series capacitor
# cancelling a line of equal magnitude), but that still marks a real edge between the buses,
# so it must produce an adjacency entry; do not drop those zeros.
function _repair_merged_adjacencies!(
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int},
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int},
    bus_lookup::Dict{Int, Int},
    merged_bus_pairs::Dict{Int, Int},
)
    for surviving_bus in Set(values(merged_bus_pairs))
        _repair_merged_adjacency!(adjacency_data, data, bus_lookup[surviving_bus])
    end
    return
end

function _repair_merged_adjacency!(
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int},
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int},
    i::Int,
)
    drows = SparseArrays.rowvals(data)
    for k in SparseArrays.nzrange(data, i)
        r = drows[k]
        r == i && continue
        iszero(adjacency_data[i, r]) && (adjacency_data[i, r] = one(Int8))
        iszero(adjacency_data[r, i]) && (adjacency_data[r, i] = -one(Int8))
    end
    return
end

# A bus merge identifies the removed bus with its survivor: their rows/columns add and the
# removed bus is dropped. The general path does this with in-place CSC structural inserts
# (`_merge_ybus_buses!`) plus a later `data[bus_ix, bus_ix]` slice; on a large sparse matrix
# every insert shifts the whole backing store, so the cost scales with merges × nnz. When a
# reduction is a pure merge (no eliminations or admittance additions), the merge and the
# slice are both just an index relabeling, so we can do them together in one O(nnz) pass:
# relabel each entry's row/column to the surviving index and let `sparse()` sum collisions.
# Returns `true` when the fast path applies; only ZIBR produces such a reduction today
# (radial/Ward add admittances, degree-two adds series branches).
function _is_pure_merge_reduction(nr_new::NetworkReductionData)
    return !isempty(nr_new.merged_bus_pairs) &&
           isempty(nr_new.removed_buses) &&
           isempty(nr_new.series_branch_map) &&
           isempty(nr_new.removed_arc_to_surviving_bus) &&
           isempty(nr_new.added_admittance_map) &&
           isempty(nr_new.added_arc_impedance_map)
end

# Old bus index -> new compacted index, sending every removed bus to its survivor's new
# index. For a pure merge every bus maps to a surviving index (no drops).
function _build_merge_remap(
    old_lookup::Dict{Int, Int},
    new_lookup::Dict{Int, Int},
    merged_bus_pairs::Dict{Int, Int},
)
    remap = zeros(Int, length(old_lookup))
    for (bus_no, old_ix) in old_lookup
        survivor = get(merged_bus_pairs, bus_no, bus_no)
        remap[old_ix] = new_lookup[survivor]
    end
    return remap
end

# Relabel a square bus×bus sparse matrix through `remap` and sum collisions in one
# `sparse()` pass. Columns/rows mapping to 0 are dropped. Equivalent to summing merged
# rows/columns then slicing out removed buses, without per-entry structural inserts.
function _remap_and_reduce(
    M::SparseArrays.SparseMatrixCSC{T, Int},
    remap::Vector{Int},
    n_new::Int,
) where {T}
    rows = SparseArrays.rowvals(M)
    vals = SparseArrays.nonzeros(M)
    nnz = length(vals)
    I = Vector{Int}(undef, nnz)
    J = Vector{Int}(undef, nnz)
    V = Vector{T}(undef, nnz)
    n = 0
    for col in 1:size(M, 2)
        new_col = remap[col]
        new_col == 0 && continue
        for k in SparseArrays.nzrange(M, col)
            new_row = remap[rows[k]]
            new_row == 0 && continue
            n += 1
            I[n] = new_row
            J[n] = new_col
            V[n] = vals[k]
        end
    end
    return SparseArrays.sparse(
        resize!(I, n),
        resize!(J, n),
        resize!(V, n),
        n_new,
        n_new,
    )
end

# A composite arc is stamped into the complex Ybus but has no adjacency entry of its own, so its
# endpoints' adjacency is re-derived from `data` after the bus slice. `_repair_merged_adjacency!`
# writes only where the entry is zero, so calling it for both endpoints cannot conflict on
# orientation and is idempotent.
function _repair_composite_arc_adjacency!(
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int},
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int},
    bus_lookup::Dict{Int, Int},
    composite_arcs,
)
    for arc in composite_arcs
        for bus in arc
            haskey(bus_lookup, bus) || continue
            _repair_merged_adjacency!(adjacency_data, data, bus_lookup[bus])
        end
    end
    return
end

# Every arc the reduction still exposes must have both endpoints on the surviving bus axis.
#
# An arc key left behind on an eliminated bus is not detectable at the point of use: the arc axis
# and the branch maps stay internally consistent, so the reduction reports success and the failure
# surfaces later, as a bare `KeyError` from whichever consumer first resolves arc endpoints against
# the bus lookup — possibly several reductions downstream of the one that caused it. Raising here
# names the reduction that produced the state.
function _validate_surviving_arc_keys(nr::NetworkReductionData, bus_ax)
    surviving = Set(bus_ax)
    for arc in get_arc_axis(nr)
        for bus in arc
            bus in surviving || error(
                "Network reduction left arc $arc referencing bus $bus, which is not on the \
reduced bus axis. Every arc the reduction retains must resolve to surviving buses.",
            )
        end
    end
    return
end

function _apply_reduction(ybus::Ybus, nr_new::NetworkReductionData)
    # These quantities are modified and used to construct the new Ybus
    data = get_data(ybus)
    adjacency_data = ybus.adjacency_data
    bus_lookup = get_bus_lookup(ybus)
    nr = get_network_reduction_data(ybus)

    # A pure-merge reduction (only ZIBR today) folds removed buses into survivors purely by
    # index relabeling, so its merge and bus-removal slice are fused into one O(nnz) rebuild
    # below; any other reduction merges the square Ybus in place via _merge_ybus_buses!. In
    # either case the arc-admittance bus columns are merged in _merge_arc_admittance_bus_columns.
    fast_merge = _is_pure_merge_reduction(nr_new)
    yft_merged, ytf_merged = ybus.arc_admittance_from_to, ybus.arc_admittance_to_from
    if !isempty(nr_new.merged_bus_pairs)
        fast_merge ||
            _merge_ybus_buses!(data, adjacency_data, bus_lookup, nr_new.merged_bus_pairs)
        yft_merged, ytf_merged = _merge_arc_admittance_bus_columns(
            ybus.arc_admittance_from_to,
            ybus.arc_admittance_to_from,
            bus_lookup,
            nr_new.merged_bus_pairs,
        )
    end
    bus_numbers_to_remove = _apply_bus_reductions!(nr, nr_new)
    # Add additional entries to the ybus corresponding to the equivalent composite arcs
    composite_entries = _composite_entries(nr_new)
    new_y_ft, new_y_tf = _add_series_branches_to_ybus!(
        ybus.data,
        get_bus_lookup(ybus),
        yft_merged,
        ytf_merged,
        composite_entries,
        nr,
    )
    _modify_removed_arc_connections!(
        ybus.data,
        get_bus_lookup(ybus),
        nr,
        nr_new.removed_arc_to_surviving_bus,
        nr_new.reductions,
    )
    _remove_arcs_from_branch_maps!(nr, nr_new)
    if !isempty(nr_new.merged_bus_pairs)
        _remap_merged_bus_in_branch_maps!(nr, nr_new.merged_bus_pairs)
    end
    _apply_added_components!(nr, nr_new, data, bus_lookup)
    _apply_composite_branch_maps!(nr, nr_new)
    add_reduction!(nr.reductions, nr_new.reductions)
    union!(nr.irreducible_buses, nr_new.irreducible_buses)

    # Remake bus axes, lookup, and data matrices without removed buses:
    bus_ax = setdiff(get_bus_axis(ybus), bus_numbers_to_remove)
    bus_lookup = make_ax_ref(bus_ax)
    bus_ix = [get_bus_lookup(ybus)[x] for x in bus_ax]
    if fast_merge
        # Fuse the merge (sum removed bus into survivor) and the bus-removal slice into one
        # relabel-and-sum pass, then re-derive survivor adjacency from the complex Ybus data
        # via the same `_repair_merged_adjacencies!` the in-place merge uses.
        remap =
            _build_merge_remap(get_bus_lookup(ybus), bus_lookup, nr_new.merged_bus_pairs)
        n_new = length(bus_ax)
        data = _remap_and_reduce(data, remap, n_new)
        adjacency_data = _remap_and_reduce(adjacency_data, remap, n_new)
        map!(sign, adjacency_data.nzval, adjacency_data.nzval)
        SparseArrays.dropzeros!(adjacency_data)
        _repair_merged_adjacencies!(
            adjacency_data,
            data,
            bus_lookup,
            nr_new.merged_bus_pairs,
        )
    else
        adjacency_data = adjacency_data[bus_ix, bus_ix]
        data = data[bus_ix, bus_ix]
        _repair_composite_arc_adjacency!(
            adjacency_data,
            data,
            bus_lookup,
            (arc for (arc, _) in composite_entries),
        )
    end

    subnetwork_axes, arc_subnetwork_axis = _make_subnetwork_axes(
        ybus,
        bus_numbers_to_remove,
        nr_new.removed_arcs,
        union(
            Set(keys(nr_new.added_arc_impedance_map)),
            Set(arc for (arc, _) in composite_entries),
        ),
    )

    arc_admittance_from_to, arc_admittance_to_from = _resolve_arc_admittance(
        new_y_ft,
        new_y_tf,
        ybus.arc_admittance_from_to,
        ybus.arc_admittance_to_from,
        nr_new.removed_arcs,
        bus_ax,
        bus_lookup,
        bus_ix,
        nr,
        nr_new.merged_bus_pairs,
    )
    nr.boundary_bus_to_removed_arcs = nr_new.boundary_bus_to_removed_arcs
    if !isempty(nr_new.merged_bus_pairs)
        nr.merged_bus_pairs = nr_new.merged_bus_pairs
    end
    _validate_surviving_arc_keys(nr, bus_ax)
    return Ybus(
        data,
        adjacency_data,
        (bus_ax, bus_ax),
        (bus_lookup, bus_lookup),
        subnetwork_axes,
        arc_subnetwork_axis,
        BranchCatalog(nr),
        arc_admittance_from_to,
        arc_admittance_to_from,
    )
end

function _apply_bus_reductions!(nr::NetworkReductionData, nr_new::NetworkReductionData)
    bus_numbers_to_remove = Vector{Int}()
    for (k, v) in nr_new.reverse_bus_search_map
        _update_bus_maps!(nr.reverse_bus_search_map, nr.bus_reduction_map, k, v)
        push!(bus_numbers_to_remove, k)
    end
    for x in nr_new.removed_buses
        push!(nr.removed_buses, x)
        push!(bus_numbers_to_remove, x)
        delete!(nr.bus_reduction_map, x)
    end
    return bus_numbers_to_remove
end

function _remap_merged_bus_in_branch_maps!(
    nr::NetworkReductionData,
    merged_bus_pairs::Dict{Int, Int},
)
    # All four maps use a two-phase collect-then-apply loop. The collect phase pops every
    # entry whose arc touches a removed bus and records the resolved new arc alongside the
    # value. The apply phase re-inserts with map-specific collision handling. Using two
    # phases avoids visiting entries that were just inserted during the apply phase.

    # --- direct_branch_map: collision → parallel-group promotion ---
    arcs_to_insert = Pair{Tuple{Int, Int}, PSY.ACTransmission}[]
    for arc in collect(keys(nr.direct_branch_map))
        new_from = get(merged_bus_pairs, arc[1], arc[1])
        new_to = get(merged_bus_pairs, arc[2], arc[2])
        (new_from == arc[1] && new_to == arc[2]) && continue
        val = pop!(nr.direct_branch_map, arc)
        new_arc = (new_from, new_to)
        if new_arc[1] == new_arc[2]
            @debug "Bus merge collapsed direct branch $(get_name(val)) (arc $arc) into a self-loop; dropping."
            continue
        end
        push!(arcs_to_insert, new_arc => val)
    end
    for (new_arc, val) in arcs_to_insert
        reverse_new_arc = (new_arc[2], new_arc[1])
        if haskey(nr.direct_branch_map, new_arc)
            existing = pop!(nr.direct_branch_map, new_arc)
            @debug "Bus merge collision on direct arc $new_arc: promoting $(get_name(existing)) and $(get_name(val)) to a parallel group."
            if haskey(nr.parallel_branch_map, new_arc)
                _push_parallel_branch!(nr.parallel_branch_map, new_arc, existing)
                _push_parallel_branch!(nr.parallel_branch_map, new_arc, val)
            else
                nr.parallel_branch_map[new_arc] =
                    _make_parallel_branch_pair(existing, val, new_arc)
            end
        elseif haskey(nr.parallel_branch_map, new_arc)
            @debug "Bus merge collision on direct arc $new_arc: adding $(get_name(val)) to existing parallel group."
            _push_parallel_branch!(nr.parallel_branch_map, new_arc, val)
        elseif haskey(nr.direct_branch_map, reverse_new_arc)
            # The remapped arc is anti-parallel to an existing direct entry 
            # Normalize to the already-established key so the pair is stored as a single parallel group.
            existing = pop!(nr.direct_branch_map, reverse_new_arc)
            @debug "Bus merge created anti-parallel collision: remapped arc $new_arc conflicts with existing $reverse_new_arc; promoting $(get_name(existing)) and $(get_name(val)) to a parallel group under $reverse_new_arc."
            if haskey(nr.parallel_branch_map, reverse_new_arc)
                _push_parallel_branch!(nr.parallel_branch_map, reverse_new_arc, existing)
                _push_parallel_branch!(nr.parallel_branch_map, reverse_new_arc, val)
            else
                nr.parallel_branch_map[reverse_new_arc] =
                    _make_parallel_branch_pair(existing, val, reverse_new_arc)
            end
        elseif haskey(nr.parallel_branch_map, reverse_new_arc)
            @debug "Bus merge created anti-parallel collision: remapped arc $new_arc conflicts with existing parallel group at $reverse_new_arc; adding $(get_name(val)) to that group."
            _push_parallel_branch!(nr.parallel_branch_map, reverse_new_arc, val)
        else
            nr.direct_branch_map[new_arc] = val
        end
    end

    # --- parallel_branch_map: collision → merge both groups into one ---
    parallel_to_insert = Pair{Tuple{Int, Int}, AbstractBranchesParallel}[]
    for arc in collect(keys(nr.parallel_branch_map))
        new_from = get(merged_bus_pairs, arc[1], arc[1])
        new_to = get(merged_bus_pairs, arc[2], arc[2])
        (new_from == arc[1] && new_to == arc[2]) && continue
        val = pop!(nr.parallel_branch_map, arc)
        new_arc = (new_from, new_to)
        if new_arc[1] == new_arc[2]
            @debug "Bus merge collapsed parallel group at arc $arc into a self-loop; dropping $(length(val)) branch(es)."
            continue
        end
        push!(parallel_to_insert, new_arc => val)
    end
    for (new_arc, val) in parallel_to_insert
        # A re-keyed group keeps its `arc_key`, but `get_arc_tuple(bp, nr)` resolves that through
        # the bus map this remap just changed — so any cached two-port is now in a stale frame.
        # Membership changes below invalidate via `add_branch!`; this covers the move-only paths.
        invalidate_equivalent_ybus!(val)
        reverse_new_arc = (new_arc[2], new_arc[1])
        if haskey(nr.parallel_branch_map, new_arc)
            @debug "Bus merge collision on parallel arc $new_arc: merging incoming group ($(length(val)) branch(es)) into existing group."
            # Merge: push all branches from the incoming group into the existing group.
            for br in val
                _push_parallel_branch!(nr.parallel_branch_map, new_arc, br)
            end
        elseif haskey(nr.direct_branch_map, new_arc)
            @debug "Bus merge collision on parallel arc $new_arc: promoting existing direct branch into the incoming parallel group."
            # Promote: move the single direct branch into the incoming group.
            existing = pop!(nr.direct_branch_map, new_arc)
            nr.parallel_branch_map[new_arc] = val
            _push_parallel_branch!(nr.parallel_branch_map, new_arc, existing)
        elseif haskey(nr.parallel_branch_map, reverse_new_arc)
            @debug "Bus merge created anti-parallel collision on parallel arc $new_arc ↔ $reverse_new_arc: merging incoming group into existing group at $reverse_new_arc."
            for br in val
                _push_parallel_branch!(nr.parallel_branch_map, reverse_new_arc, br)
            end
        elseif haskey(nr.direct_branch_map, reverse_new_arc)
            @debug "Bus merge created anti-parallel collision on parallel arc $new_arc ↔ $reverse_new_arc: absorbing existing direct branch into incoming group, stored under $reverse_new_arc."
            existing = pop!(nr.direct_branch_map, reverse_new_arc)
            # Put the key-oriented branch first so the group's reference orientation is the key.
            nr.parallel_branch_map[reverse_new_arc] =
                _make_parallel_branch_pair(existing, first(val), reverse_new_arc)
            for br in Iterators.drop(val, 1)
                _push_parallel_branch!(nr.parallel_branch_map, reverse_new_arc, br)
            end
        else
            nr.parallel_branch_map[new_arc] = val
        end
    end

    # Rebuild reverse maps for the two forward maps that were modified above.
    # `ThreeWindingTransformerCircuit`s live in `direct_branch_map` and are handled by the
    # direct-map remap above. ZIR excludes transformer arcs themselves from bus merging (see
    # `_is_transformer`), but a non-winding zero-impedance branch between two real
    # terminal buses of the same transformer can merge them, remapping both winding arcs onto
    # one (merged, star) arc. That collision promotes the windings into a parallel group,
    # whose equivalent correctly sums the star-leg Pi-models (the windings really are in
    # parallel between the merged bus and the star point). series_branch_map is always empty
    # when ZIR runs (D2 populates it afterwards), so its reverse map does not need rebuilding
    # here.
    _remake_reverse_direct_branch_map!(nr)
    _remake_reverse_parallel_branch_map!(nr)
    return
end

function _remove_arcs_from_branch_maps!(
    nr::NetworkReductionData,
    nr_new::NetworkReductionData,
)
    remake_reverse_direct_branch_map = false
    remake_reverse_parallel_branch_map = false
    remake_reverse_series_branch_map = false
    for x in nr_new.removed_arcs
        push!(nr.removed_arcs, x)
        if haskey(nr.direct_branch_map, x)
            remake_reverse_direct_branch_map = true
            delete!(nr.direct_branch_map, x)
        elseif haskey(nr.parallel_branch_map, x)
            remake_reverse_parallel_branch_map = true
            delete!(nr.parallel_branch_map, x)
        elseif haskey(nr.series_branch_map, x)
            remake_reverse_series_branch_map = true
            delete!(nr.series_branch_map, x)
        end
    end
    remake_reverse_direct_branch_map && _remake_reverse_direct_branch_map!(nr)
    remake_reverse_parallel_branch_map && _remake_reverse_parallel_branch_map!(nr)
    remake_reverse_series_branch_map && _remake_reverse_series_branch_map!(nr)
    return
end

function _apply_added_components!(
    nr::NetworkReductionData,
    nr_new::NetworkReductionData,
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int},
    bus_lookup::Dict{Int, Int},
)
    if !isempty(nr_new.added_arc_impedance_map) && !isempty(nr.added_arc_impedance_map) ||
       !isempty(nr_new.added_admittance_map) && !isempty(nr.added_admittance_map)
        error(
            "Only the final applied reduction can add new arc impedances and/or fixed admittances to the Ybus (e.g. Ward Reduction)",
        )
    end
    nr.added_arc_impedance_map = nr_new.added_arc_impedance_map
    nr.added_admittance_map = nr_new.added_admittance_map
    for (bus_no, admittance) in nr.added_admittance_map
        Y = PSY.get_Y(admittance)
        data[bus_lookup[bus_no], bus_lookup[bus_no]] += Y
    end
    for (bus_tuple, admittance) in nr.added_arc_impedance_map
        bus_from, bus_to = bus_tuple
        Y11, Y12, Y21, Y22 = ybus_branch_entries(admittance)
        data[bus_lookup[bus_from], bus_lookup[bus_from]] += Y11
        data[bus_lookup[bus_from], bus_lookup[bus_to]] += Y12
        data[bus_lookup[bus_to], bus_lookup[bus_from]] += Y21
        data[bus_lookup[bus_to], bus_lookup[bus_to]] += Y22
    end
    return
end

# A degree-two reduction produces a lone chain in `series_branch_map` and a group of sibling
# chains in `parallel_branch_map`, so both have to be filed here.
function _apply_composite_branch_maps!(
    nr::NetworkReductionData,
    nr_new::NetworkReductionData,
)
    if isempty(nr.series_branch_map)
        nr.series_branch_map = nr_new.series_branch_map
        nr.reverse_series_branch_map = nr_new.reverse_series_branch_map
    elseif !isempty(nr_new.series_branch_map) && !isempty(nr.series_branch_map)
        error(
            "Cannot compose series branch maps; should not apply multiple reductions that generate series branch maps",
        )
    end
    for (arc, group) in nr_new.parallel_branch_map
        existing_arc = _existing_arc_key(
            nr.direct_branch_map,
            nr.parallel_branch_map,
            nr.series_branch_map,
            arc,
        )
        isnothing(existing_arc) || error(
            "Composite arc $arc from a degree-two reduction collides with the existing \
arc $existing_arc. A bus pair must keep one forward-map entry, or `get_arc_axis` collapses \
the two arcs into one and `_get_entry` answers with only one of them.",
        )
        nr.parallel_branch_map[arc] = group
        _register_composite_members!(nr.reverse_parallel_branch_map, arc, group)
    end
    return
end

"""
Updates both existing bus maps (forward and reverse) for a new reduction of bus b1 into bus b2.
Resolves each bus to its current root before merging so that the entire group previously
associated with b1's root (not just b1 itself) is correctly transferred to b2's root.
"""
function _update_bus_maps!(
    reverse_bus_search_map::Dict{Int, Int},
    bus_reduction_map::Dict{Int, Set{Int}},
    b1_number::Int,
    b2_number::Int,
)
    b1_root = get(reverse_bus_search_map, b1_number, b1_number)
    b2_root = get(reverse_bus_search_map, b2_number, b2_number)
    b1_root == b2_root && return

    # Collect b1_root plus every bus already reduced under it
    s_moving = union(get(bus_reduction_map, b1_root, Set{Int}()), Set{Int}((b1_root,)))
    b1_number != b1_root && push!(s_moving, b1_number)

    # Merge into b2_root's group
    bus_reduction_map[b2_root] =
        union(get(bus_reduction_map, b2_root, Set{Int}()), s_moving)
    delete!(bus_reduction_map, b1_root)

    for x in s_moving
        reverse_bus_search_map[x] = b2_root
    end
    return
end

function _make_subnetwork_axes(
    ybus::Ybus,
    bus_numbers_to_remove::Vector{Int},
    arcs_to_remove::Set{Tuple{Int, Int}},
    arcs_to_add::Set{Tuple{Int, Int}},
)
    subnetwork_axes = deepcopy(ybus.subnetwork_axes)
    arc_subnetwork_axis = deepcopy(ybus.arc_subnetwork_axis)
    subnetwork_key_removed = Set{Int}()
    for k in keys(subnetwork_axes)
        if k in bus_numbers_to_remove
            push!(subnetwork_key_removed, k)
        end
    end
    reverse_bus_search_map = get_reverse_bus_search_map(get_network_reduction_data(ybus))
    for k in subnetwork_key_removed
        axis_1, axis_2 = subnetwork_axes[k]
        surviving_buses = setdiff(axis_1, bus_numbers_to_remove)
        # An island losing every bus is dropped by the empty-subnetwork sweep below; re-keying
        # it would only swap one dead key for another.
        isempty(surviving_buses) && continue
        # The bus the old representative was folded into inherits the role, so the reference
        # bus stays electrically the same bus. Only a removal that merges nothing (no reverse
        # map entry) falls back to an order-independent pick.
        merge_target = get(reverse_bus_search_map, k, k)
        if merge_target in surviving_buses
            new_ref_bus = merge_target
        else
            new_ref_bus = minimum(surviving_buses)
        end
        delete!(subnetwork_axes, k)
        subnetwork_axes[new_ref_bus] = (axis_1, axis_2)
        # If a reference bus key is reduced, change the arc subnetwork axis key as well:
        arc_subnetwork_axis[new_ref_bus] = pop!(arc_subnetwork_axis, k)
        @warn "Original reference bus $k removed during reduction; reassigning the subnetwork reference bus to $new_ref_bus."
    end
    empty_subnetwork_keys = Set{Int}()
    for (k, values) in subnetwork_axes
        new_values = setdiff(values[1], bus_numbers_to_remove)
        subnetwork_axes[k] = (new_values, new_values)
        isempty(new_values) && push!(empty_subnetwork_keys, k)
    end
    for k in empty_subnetwork_keys
        @warn "Subnetwork with reference bus $k has no remaining buses after reduction and will be removed from the Ybus."
        delete!(subnetwork_axes, k)
        delete!(arc_subnetwork_axis, k)
    end
    for (k, values) in arc_subnetwork_axis
        subnetwork_buses = Set(subnetwork_axes[k][1])
        local_arcs_to_add =
            Set(
                arc for arc in arcs_to_add if
                arc[1] in subnetwork_buses && arc[2] in subnetwork_buses
            )
        arc_subnetwork_axis[k] = union(setdiff(values, arcs_to_remove), local_arcs_to_add)
    end
    return subnetwork_axes, arc_subnetwork_axis
end

function _modify_removed_arc_connections!(
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int64},
    bus_lookup::Dict{Int, Int},
    nrd_old::NetworkReductionData,
    removed_arc_to_surviving_bus::Dict{Tuple{Int, Int}, Int},
    reductions::ReductionContainer,
)
    for (arc, bus) in removed_arc_to_surviving_bus
        arc_entry = _get_entry(arc, nrd_old)
        y11, _, _, y22 = ybus_branch_entries(arc_entry, nrd_old)
        if arc[1] == bus
            data[bus_lookup[arc[1]], bus_lookup[arc[1]]] -= y11
        elseif arc[2] == bus
            data[bus_lookup[arc[2]], bus_lookup[arc[2]]] -= y22
        else
            error("Bad data in removed_arc_to_surviving_bus map")
        end
    end
    return
end

function _get_entry(arc::Tuple{Int, Int}, nrd::NetworkReductionData)
    if haskey(nrd.direct_branch_map, arc)
        return nrd.direct_branch_map[arc]
    elseif haskey(nrd.parallel_branch_map, arc)
        return nrd.parallel_branch_map[arc]
    elseif haskey(nrd.series_branch_map, arc)
        return nrd.series_branch_map[arc]
    else
        error("Key $arc not found in any of the maps")
    end
end

const COMPOSITE_ENTRIES = Vector{Tuple{Tuple{Int, Int}, AbstractReductionAggregate}}

# One iterable over every composite arc a degree-two reduction produced, whether it holds a
# single chain or a parallel group of sibling chains. `DegreeTwoReduction` is the only reduction
# that populates `nr_new.parallel_branch_map` — the others build a `NetworkReductionData` that
# leaves it empty — so everything found there is a composite arc to stamp.
function _composite_entries(nr_new::NetworkReductionData)
    entries = COMPOSITE_ENTRIES()
    for (arc, entry) in nr_new.series_branch_map
        push!(entries, (arc, entry))
    end
    for (arc, entry) in nr_new.parallel_branch_map
        push!(entries, (arc, entry))
    end
    # At most one composite arc per unordered bus pair: `get_degree2_reduction` groups sibling
    # chains by `minmax` of their endpoints. A second composite arc on the same pair would be
    # stamped onto the same two Ybus entries and filed under a second forward-map key that
    # `get_arc_axis` cannot distinguish from the first. Checked rather than assumed because it is
    # invisible at both of those call sites.
    seen = Set{Tuple{Int, Int}}()
    for (arc, _) in entries
        pair = minmax(arc[1], arc[2])
        pair in seen &&
            error(
                "Degree-two reduction produced more than one composite arc on pair $pair.",
            )
        push!(seen, pair)
    end
    return entries
end

function _add_series_branches_to_ybus!(
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int64},
    bus_lookup::Dict{Int, Int},
    yft::Nothing,
    ytf::Nothing,
    composite_entries::COMPOSITE_ENTRIES,
    nrd::NetworkReductionData,
)
    for (equivalent_arc, entry) in composite_entries
        _stamp_composite_arc!(data, bus_lookup, equivalent_arc, entry, nrd)
    end
    return yft, ytf
end

function _add_series_branches_to_ybus!(
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int64},
    bus_lookup::Dict{Int, Int},
    yft::ArcAdmittanceMatrix,
    ytf::ArcAdmittanceMatrix,
    composite_entries::COMPOSITE_ENTRIES,
    nrd::NetworkReductionData,
)
    arc_lookup = get_arc_lookup(yft)
    arc_axis = get_arc_axis(yft)
    I_yft, J_yft, V_yft = SparseArrays.findnz(yft.data)
    I_ytf, J_ytf, V_ytf = SparseArrays.findnz(ytf.data)
    row_ix = size(yft)[1] + 1
    n_buses = size(yft)[2]
    for (equivalent_arc, entry) in composite_entries
        equivalent, equivalent_arc_indices =
            _stamp_composite_arc!(data, bus_lookup, equivalent_arc, entry, nrd)
        from_ix, to_ix = equivalent_arc_indices
        push!(arc_axis, equivalent_arc)
        push!(I_yft, row_ix, row_ix)
        push!(J_yft, from_ix, to_ix)
        push!(V_yft, equivalent[1], equivalent[2])
        push!(I_ytf, row_ix, row_ix)
        push!(J_ytf, to_ix, from_ix)
        push!(V_ytf, equivalent[4], equivalent[3])
        row_ix += 1
    end
    yft_data = SparseArrays.sparse(I_yft, J_yft, V_yft, row_ix - 1, n_buses)
    ytf_data = SparseArrays.sparse(I_ytf, J_ytf, V_ytf, row_ix - 1, n_buses)

    catalog = BranchCatalog(nrd)
    arc_admittance_from_to = ArcAdmittanceMatrix(
        yft_data,
        (arc_axis, get_bus_axis(yft)),
        (arc_lookup, get_bus_lookup(yft)),
        catalog,
        :FromTo,
    )
    arc_admittance_to_from = ArcAdmittanceMatrix(
        ytf_data,
        (arc_axis, get_bus_axis(ytf)),
        (arc_lookup, get_bus_lookup(ytf)),
        catalog,
        :ToFrom,
    )
    return arc_admittance_from_to, arc_admittance_to_from
end

# Entries the composite arc's members already contributed to the unreduced Ybus, in the
# composite arc's frame. A chain's endpoints are not directly connected in the unreduced Ybus,
# so a chain contributes only diagonals.
#
# There is deliberately no `PSY.ACTransmission` arm. A composite entry is always an aggregate, and
# a group a degree-two reduction produced holds only chains, so a physical member would mean the
# stamp is running against a group it was not built for — where backing out only the members'
# diagonals is the wrong correction. A `MethodError` says that; a blanket arm would return a
# plausible wrong number.
function _composite_raw_two_port(entry::BranchesSeries, nr::NetworkReductionData)
    chain = _build_chain_ybus(entry, nr)
    return (chain[1, 1], zero(YBUS_ELTYPE), zero(YBUS_ELTYPE), chain[end, end])
end

function _composite_raw_two_port(entry::AbstractBranchesParallel, nr::NetworkReductionData)
    reference = get_arc_tuple(entry, nr)
    r11 = r12 = r21 = r22 = zero(YBUS_ELTYPE)
    for member in entry
        (m11, m12, m21, m22) = _composite_raw_two_port(member, nr)
        if get_arc_tuple(member, nr) != reference
            r11 += m22
            r12 += m21
            r21 += m12
            r22 += m11
        else
            r11 += m11
            r12 += m12
            r21 += m21
            r22 += m22
        end
    end
    return (r11, r12, r21, r22)
end

# Install the composite arc's equivalent two-port and back out what its members already
# contributed. Every entry accumulates, so an off-diagonal a sibling group's members share with
# anything else on the pair composes rather than overwrites.
function _apply_composite_arc_ybus!(
    ybus_full::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int64},
    equivalent::NTuple{4, YBUS_ELTYPE},
    raw::NTuple{4, YBUS_ELTYPE},
    equivalent_arc_indices::Tuple{Int, Int},
)
    i, j = equivalent_arc_indices
    ybus_full[i, i] += equivalent[1] - raw[1]
    ybus_full[i, j] += equivalent[2] - raw[2]
    ybus_full[j, i] += equivalent[3] - raw[3]
    ybus_full[j, j] += equivalent[4] - raw[4]
    return
end

# Stamp one composite arc's equivalent two-port into the bus Ybus and return it, so the caller
# can also record it as an arc-admittance row.
function _stamp_composite_arc!(
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int64},
    bus_lookup::Dict{Int, Int},
    equivalent_arc::Tuple{Int, Int},
    entry::AbstractReductionAggregate,
    nr::NetworkReductionData,
)
    equivalent_arc_indices =
        (bus_lookup[equivalent_arc[1]], bus_lookup[equivalent_arc[2]])
    equivalent = YBUS_ELTYPE.(ybus_branch_entries(entry, nr))
    raw = _composite_raw_two_port(entry, nr)
    _apply_composite_arc_ybus!(data, equivalent, raw, equivalent_arc_indices)
    return equivalent, equivalent_arc_indices
end

function _build_chain_ybus(series_chain::BranchesSeries, nr::NetworkReductionData)
    segment_orientations = get_segment_orientations(series_chain)
    fb = Vector{Int}()
    tb = Vector{Int}()
    y11 = Vector{YBUS_ELTYPE}()
    y12 = Vector{YBUS_ELTYPE}()
    y21 = Vector{YBUS_ELTYPE}()
    y22 = Vector{YBUS_ELTYPE}()
    for (ix, segment) in enumerate(series_chain)
        add_segment_to_ybus!(
            segment,
            y11,
            y12,
            y21,
            y22,
            fb,
            tb,
            ix,
            segment_orientations[ix],
            nr,
        )
    end
    return Matrix(
        SparseArrays.sparse(
            [fb; fb; tb; tb],  # row indices
            [fb; tb; fb; tb],  # column indices
            [y11; y12; y21; y22],  # values
        ),
    )
end

"""
    add_segment_to_ybus!(
        segment::PSY.ACTransmission
        y11::Vector{YBUS_ELTYPE},
        y12::Vector{YBUS_ELTYPE},
        y21::Vector{YBUS_ELTYPE},
        y22::Vector{YBUS_ELTYPE},
        fb::Vector{Int},
        tb::Vector{Int},
        ix::Int,
        segment_orientation::Symbol
    )

Add a branch segment to Y-bus vectors during series chain reduction.

Adds the Y-bus entries for a single segment (branch or transformer winding) to the
admittance vectors, handling the proper orientation. Used when building equivalent
Y-bus entries for series chains of degree-two buses.

# Arguments
- `segment::Union{PSY.ACTransmission, Tuple{PSY.ThreeWindingTransformer, Int}}`: Branch segment to add
- `y11::Vector{YBUS_ELTYPE}`: Vector for from-bus self admittances
- `y12::Vector{YBUS_ELTYPE}`: Vector for from-to mutual admittances
- `y21::Vector{YBUS_ELTYPE}`: Vector for to-from mutual admittances
- `y22::Vector{YBUS_ELTYPE}`: Vector for to-bus self admittances
- `fb::Vector{Int}`: Vector for from-bus indices
- `tb::Vector{Int}`: Vector for to-bus indices
- `ix::Int`: Index position for the segment
- `segment_orientation::Symbol`: `:FromTo` or `:ToFrom` orientation

# Implementation Details
- Computes Pi-model entries using `ybus_branch_entries()`
- Handles orientation by swapping entries for `:ToFrom`
- Sets bus indices to consecutive values (ix, ix+1) for chain building
- Used in degree-two network reduction algorithms

# See Also
- [`DegreeTwoReduction`](@ref): Degree-two bus elimination
- [`ybus_branch_entries`](@ref): Pi-model computation
"""
function add_segment_to_ybus!(
    segment::PSY.ACTransmission,
    y11::Vector{YBUS_ELTYPE},
    y12::Vector{YBUS_ELTYPE},
    y21::Vector{YBUS_ELTYPE},
    y22::Vector{YBUS_ELTYPE},
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    segment_orientation::Symbol,
    nr::NetworkReductionData,
)
    (Y11, Y12, Y21, Y22) = ybus_branch_entries(segment)
    push!(fb, ix)
    push!(tb, ix + 1)
    if segment_orientation == :FromTo
        push!(y11, Y11)
        push!(y12, Y12)
        push!(y21, Y21)
        push!(y22, Y22)
    elseif segment_orientation == :ToFrom
        push!(y11, Y22)
        push!(y12, Y21)
        push!(y21, Y12)
        push!(y22, Y11)
    else
        error("Invalid segment orientation $(segment_orientation)")
    end
end

"""
    add_segment_to_ybus!(
        segment::AbstractBranchesParallel,
        y11::Vector{YBUS_ELTYPE},
        y12::Vector{YBUS_ELTYPE},
        y21::Vector{YBUS_ELTYPE},
        y22::Vector{YBUS_ELTYPE},
        fb::Vector{Int},
        tb::Vector{Int},
        ix::Int,
        segment_orientation::Symbol
    )

Add multiple parallel branches as a single segment to Y-bus vectors.

Handles the case where a segment in a series chain consists of multiple parallel
branches between the same pair of buses. Each branch in the set is added to the
same Y-bus position, effectively combining their admittances.

# Arguments
- `segment::AbstractBranchesParallel`: Set of parallel AC transmission branches
- `y11::Vector{YBUS_ELTYPE}`: Vector for from-bus self admittances
- `y12::Vector{YBUS_ELTYPE}`: Vector for from-to mutual admittances
- `y21::Vector{YBUS_ELTYPE}`: Vector for to-from mutual admittances
- `y22::Vector{YBUS_ELTYPE}`: Vector for to-bus self admittances
- `fb::Vector{Int}`: Vector for from-bus indices
- `tb::Vector{Int}`: Vector for to-bus indices
- `ix::Int`: Index position for the segment
- `segment_orientation::Symbol`: `:FromTo` or `:ToFrom` orientation

# Implementation Details
- Iterates through all branches in the parallel set
- Calls single-branch `add_segment_to_ybus!()` for each branch
- Y-bus entries are accumulated at the same index position
- Results in equivalent admittance of parallel combination

# See Also
- [`add_segment_to_ybus!`](@ref): Single branch variant
- [`DegreeTwoReduction`](@ref): Series chain elimination
"""
# A chain reached as another chain's segment enters as its own two-port, which is both what a
# composite arc contributes to Ybus and what `_composite_raw_two_port` backs out for it. Resolving
# it through `nr` keeps those two agreeing; the one-argument form has no method for an aggregate
# and would fall through to the single-branch path.
function add_segment_to_ybus!(
    segment::AbstractReductionAggregate,
    y11::Vector{YBUS_ELTYPE},
    y12::Vector{YBUS_ELTYPE},
    y21::Vector{YBUS_ELTYPE},
    y22::Vector{YBUS_ELTYPE},
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    segment_orientation::Symbol,
    nr::NetworkReductionData,
)
    # For a parallel group this is the orientation-correct equivalent block rather than a sum of
    # members under one shared orientation, which mis-handles an anti-parallel asymmetric member.
    (Y11, Y12, Y21, Y22) = ybus_branch_entries(segment, nr)
    push!(fb, ix)
    push!(tb, ix + 1)
    if segment_orientation == :FromTo
        push!(y11, Y11)
        push!(y12, Y12)
        push!(y21, Y21)
        push!(y22, Y22)
    elseif segment_orientation == :ToFrom
        push!(y11, Y22)
        push!(y12, Y21)
        push!(y21, Y12)
        push!(y22, Y11)
    else
        error("Invalid segment orientation $(segment_orientation)")
    end
    return
end

function _reduce_internal_nodes(Y::Matrix{YBUS_ELTYPE})
    dim_Y = size(Y)[1]
    keep_ix = [1, dim_Y]
    eliminate_ix = collect(2:(dim_Y - 1))
    Y_kk = Y[keep_ix, keep_ix]
    Y_ee = Y[eliminate_ix, eliminate_ix]
    Y_ke = Y[keep_ix, eliminate_ix]
    Y_ek = Y[eliminate_ix, keep_ix]
    Y_reduced = Y_kk - Y_ke * (Y_ee \ Y_ek)
    return Y_reduced
end

function _remake_reverse_direct_branch_map!(nr::NetworkReductionData)
    reverse_direct_branch_map = Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    for (k, v) in nr.direct_branch_map
        reverse_direct_branch_map[v] = k
    end
    nr.reverse_direct_branch_map = reverse_direct_branch_map
    return
end
function _remake_reverse_parallel_branch_map!(nr::NetworkReductionData)
    reverse_parallel_branch_map = Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    for (arc, entry) in nr.parallel_branch_map
        _register_composite_members!(reverse_parallel_branch_map, arc, entry)
    end
    nr.reverse_parallel_branch_map = reverse_parallel_branch_map
    return
end
function _remake_reverse_series_branch_map!(nr::NetworkReductionData)
    reverse_series_branch_map = Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    for (arc, entry) in nr.series_branch_map
        _register_composite_members!(reverse_series_branch_map, arc, entry)
    end
    nr.reverse_series_branch_map = reverse_series_branch_map
    return
end

"""
    validate_connectivity(M::Ybus) -> Bool

Validate that the Y-bus represents a fully connected electrical network.

Checks network connectivity by counting the number of electrical islands (subnetworks)
in the Y-bus matrix. A fully connected network should have exactly one subnetwork.
Multiple subnetworks indicate electrical isolation between parts of the system.

# Arguments
- `M::Ybus`: Y-bus matrix to validate

# Returns
- `Bool`: `true` if network is fully connected (single subnetwork), `false` otherwise

# Examples
```julia
ybus = Ybus(system)
if validate_connectivity(ybus)
    println("Network is fully connected")
else
    println("Network has isolated islands")
    islands = find_subnetworks(ybus)
    println("Number of islands: ", length(islands))
end
```

# Implementation Details
- Uses `find_subnetworks()` to identify electrical islands
- Single subnetwork indicates full electrical connectivity
- Multiple subnetworks may require separate power flow solutions

# See Also
- [`find_subnetworks`](@ref): Identify electrical islands
- [`validate_connectivity`](@ref): System-level connectivity validation
"""
function validate_connectivity(M::Ybus)
    sub_nets = find_subnetworks(M)
    return length(sub_nets) == 1
end

"""
    find_subnetworks(M::Ybus) -> Dict{Int, Set{Int}}

Identify electrical islands (subnetworks) in the Y-bus matrix.

Analyzes the network topology to find groups of buses that are electrically connected
to each other but isolated from other groups. Each subnetwork represents an electrical
island that requires its own reference bus and can be solved independently.

# Arguments
- `M::Ybus`: Y-bus matrix to analyze

# Returns
- `Dict{Int, Set{Int}}`: Dictionary mapping reference bus numbers to sets of bus numbers in each subnetwork

# Examples
```julia
ybus = Ybus(system)
subnetworks = find_subnetworks(ybus)
for (ref_bus, buses) in subnetworks
    println("Island ", ref_bus, ": ", sort(collect(buses)))
end

if length(subnetworks) > 1
    @warn "Network has ", length(subnetworks), " electrical islands"
end
```

# Implementation Details
- Uses adjacency matrix analysis to find connected components
- Each subnetwork gets assigned a reference bus for voltage angle reference
- Isolated buses or groups require separate power flow analysis
- Critical for power flow initialization and solution

# See Also
- [`validate_connectivity`](@ref): Check for full connectivity
- [`depth_first_search`](@ref): Graph traversal algorithm
- [`iterative_union_find`](@ref): Alternative connectivity algorithm
"""
function find_subnetworks(M::Ybus)
    bus_numbers = M.axes[2]
    return find_subnetworks(M.adjacency_data, bus_numbers)
end

function get_reduction(ybus::Ybus, sys::PSY.System, reduction::DegreeTwoReduction)
    A = AdjacencyMatrix(ybus)
    return get_reduction(A, sys, reduction)
end

"""
    _validate_study_buses(ybus::Ybus, study_buses::Vector{Int}) -> Union{Int, Nothing}

Validate that Ward reduction study buses are compatible with the network islands.

Checks that all study buses exist in the reduced/unreduced bus maps, that they lie in a
single synchronously connected subnetwork, and that any partially reduced subnetwork
includes its slack bus. Returns the reference-bus key of the matching subnetwork when
validation succeeds.

# Errors
- Throws `IS.DataFormatError` if any study bus is not present in the system.
- Throws `IS.DataFormatError` if study buses span multiple subnetworks.
- Throws `IS.DataFormatError` if a partially reduced subnetwork excludes its slack bus.
"""
function _validate_study_buses(ybus::Ybus, study_buses::Vector{Int})
    nrd = get_network_reduction_data(ybus)
    valid_bus_numbers =
        union(Set(keys(nrd.bus_reduction_map)), Set(keys(nrd.reverse_bus_search_map)))

    for bus_number in study_buses
        bus_number ∉ valid_bus_numbers &&
            throw(IS.DataFormatError("Study bus $bus_number not found in system"))
    end

    study_bus_set = Set(study_buses)
    slack_bus_numbers = get_ref_bus(ybus)

    for (ref_bus_key, axes) in ybus.subnetwork_axes
        subnetwork_bus_set = Set(axes[1])
        all_in_subnetwork = issubset(study_bus_set, subnetwork_bus_set)
        no_study_buses_in_subnetwork = isempty(intersect(study_bus_set, subnetwork_bus_set))
        if !(all_in_subnetwork || no_study_buses_in_subnetwork)
            throw(
                IS.DataFormatError(
                    "All study_buses must occur in a single synchronously connected system.",
                ),
            )
        end

        if !all_in_subnetwork
            continue
        end

        for sb in slack_bus_numbers
            if sb in subnetwork_bus_set && sb ∉ study_bus_set
                throw(
                    IS.DataFormatError(
                        "Slack bus $sb must be included in the study buses for an area that is partially reduced",
                    ),
                )
            end
        end

        return ref_bus_key
    end
    error("Unable to identify subnetwork for provided study buses")
    return nothing
end

function get_reduction(ybus::Ybus, ::PSY.System, reduction::WardReduction)
    study_buses = get_study_buses(reduction)
    ref_bus_key = _validate_study_buses(ybus, study_buses)
    subnetwork_bus_axis = ybus.subnetwork_axes[ref_bus_key][1]
    bus_lookup = get_bus_lookup(ybus)
    bus_axis = get_bus_axis(ybus)
    A = IncidenceMatrix(ybus)
    arc_axis = get_arc_axis(A)
    boundary_buses = Set{Int}()
    removed_arcs = Set{Tuple{Int, Int}}()
    removed_buses = setdiff(Set(bus_axis), Set(subnetwork_bus_axis))
    removed_arc_to_surviving_bus = Dict{Tuple{Int, Int}, Int}()
    study_bus_set = Set(study_buses)
    for arc in arc_axis
        #Determine boundary buses:
        if (arc[1] ∈ study_bus_set) && (arc[2] ∉ study_bus_set)
            push!(boundary_buses, arc[1])
            removed_arc_to_surviving_bus[arc] = arc[1]
        elseif (arc[1] ∉ study_bus_set) && (arc[2] ∈ study_bus_set)
            push!(boundary_buses, arc[2])
            removed_arc_to_surviving_bus[arc] = arc[2]
        end
        #Determine arcs outside of study area
        if !(arc[1] ∈ study_bus_set && arc[2] ∈ study_bus_set)
            push!(removed_arcs, arc)
        end
    end

    boundary_bus_to_removed_arcs = Dict{Int, Set{Tuple{Int, Int}}}()
    for (removed_arc, boundary_bus) in removed_arc_to_surviving_bus
        set = get!(boundary_bus_to_removed_arcs, boundary_bus, Set{Tuple{Int, Int}}())
        push!(set, removed_arc)
    end

    if Set(subnetwork_bus_axis) == Set(study_buses)
        # `removed_buses` holds the buses of every *other* island. With none, Ward has
        # nothing to eliminate anywhere: the study area is the whole island, and there is no
        # other island to drop.
        if isempty(removed_buses)
            throw(
                IS.DataFormatError(
                    "WardReduction study_buses cover the entire network \
                    ($(length(study_buses)) buses, one island), so the reduction would \
                    remove nothing. Narrow study_buses to the area to retain, or drop the \
                    WardReduction.",
                ),
            )
        end
        @warn "The study buses comprise an entire island; ward reduction will not modify \
               this island, and the other islands will be eliminated."
        return NetworkReductionData(;
            removed_arcs = removed_arcs,
            removed_buses = removed_buses,
            reductions = ReductionContainer(; ward_reduction = reduction),
        )
    end

    bus_reduction_map,
    reverse_bus_search_map,
    added_arc_impedance_map,
    added_admittance_map = get_ward_reduction(
        ybus.data,
        bus_lookup,
        bus_axis,
        arc_axis,
        boundary_buses,
        Set(get_ref_bus(ybus)),
        study_buses,
        subnetwork_bus_axis,
    )

    for arc_tuple in keys(added_arc_impedance_map)
        if ybus.data[bus_lookup[arc_tuple[1]], bus_lookup[arc_tuple[2]]] != 0.0
            @warn "Equivalent arc impedance computed during Ward reduction is in parallel with existing system arc.\\
                    Indexing into PTDF/LODF with branch names may give unexpected results for arc $arc_tuple"
        end
    end
    return NetworkReductionData(;
        bus_reduction_map = bus_reduction_map,
        reverse_bus_search_map = reverse_bus_search_map,
        removed_arcs = removed_arcs,
        removed_buses = removed_buses,
        added_arc_impedance_map = added_arc_impedance_map,
        added_admittance_map = added_admittance_map,
        removed_arc_to_surviving_bus = removed_arc_to_surviving_bus,
        boundary_bus_to_removed_arcs = boundary_bus_to_removed_arcs,
        reductions = ReductionContainer(; ward_reduction = reduction),
    )
end
