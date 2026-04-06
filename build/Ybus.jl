"""
    Ybus{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{ComplexF32}

Nodal admittance matrix (Y-bus) representing the electrical admittance relationships between
buses in a power system. This N×N sparse complex matrix encodes the network topology and
electrical parameters needed for power flow calculations and network analysis.

# Fields
- `data::SparseArrays.SparseMatrixCSC{ComplexF32, Int}`: Sparse Y-bus matrix with complex admittance values
- `adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int}`: Network connectivity information
- `axes::Ax`: Tuple of bus axis vectors for indexing (bus_numbers, bus_numbers)
- `lookup::L`: Tuple of lookup dictionaries mapping bus numbers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`: Bus axes for each electrical island/subnetwork
- `arc_subnetwork_axis::Dict{Int, Vector{Tuple{Int, Int}}}`: Arc axes for each subnetwork
- `network_reduction_data::NetworkReductionData`: Metadata from network reduction operations
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
struct Ybus{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{ComplexF32}
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int}
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    arc_subnetwork_axis::Dict{Int, Vector{Tuple{Int, Int}}}
    network_reduction_data::NetworkReductionData
    arc_admittance_from_to::Union{ArcAdmittanceMatrix, Nothing}
    arc_admittance_to_from::Union{ArcAdmittanceMatrix, Nothing}
end

get_axes(M::Ybus) = M.axes
get_lookup(M::Ybus) = M.lookup
get_ref_bus(M::Ybus) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::Ybus) = [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::Ybus) = M.network_reduction_data
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
    return ybus.network_reduction_data
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
ybus = Ybus(system)
reduction = RadialReduction(irreducible_buses=[101, 205])
reduction_data = get_reduction(ybus, system, reduction)
```

# See Also
- [`RadialReduction`](@ref): Radial reduction configuration
- [`get_reduction`](@ref): Other reduction methods
"""
function get_reduction(
    ybus::Ybus,
    sys::PSY.System,
    reduction::RadialReduction,
)
    A = IncidenceMatrix(ybus)
    return get_reduction(A, sys, reduction)
end

function _push_parallel_branch!(
    parallel_branch_map::Dict,
    arc_tuple::Tuple{Int, Int},
    br::T,
) where {T <: PSY.ACTransmission}
    if get_branch_type(parallel_branch_map[arc_tuple]) == T
        add_branch!(parallel_branch_map[arc_tuple], br)
    else
        @warn "Mismatch in parallel device types for arc $(arc_tuple). This could indicate issues in the network data."
        parallel_branch_map[arc_tuple] =
            BranchesParallel([parallel_branch_map[arc_tuple].branches..., br])
    end
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
- If arc already has a direct branch, converts to parallel mapping
- If arc already has parallel branches, adds to existing set
- Otherwise creates new direct mapping
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
    if haskey(parallel_branch_map, arc_tuple) && typeof(br) ∉ SKIP_PARALLEL_REDUCTION_TYPES
        _push_parallel_branch!(parallel_branch_map, arc_tuple, br)
        reverse_parallel_branch_map[br] = arc_tuple
    elseif haskey(direct_branch_map, arc_tuple) &&
           typeof(direct_branch_map[arc_tuple]) ∉ SKIP_PARALLEL_REDUCTION_TYPES &&
           typeof(br) ∉ SKIP_PARALLEL_REDUCTION_TYPES
        corresponding_branch = direct_branch_map[arc_tuple]
        delete!(direct_branch_map, arc_tuple)
        delete!(reverse_direct_branch_map, corresponding_branch)
        parallel_branch_map[arc_tuple] = BranchesParallel([corresponding_branch, br])
        reverse_parallel_branch_map[corresponding_branch] = arc_tuple
        reverse_parallel_branch_map[br] = arc_tuple
    else
        direct_branch_map[arc_tuple] = br
        reverse_direct_branch_map[br] = arc_tuple
    end
    return
end

"""
    add_to_branch_maps!(
        nr::NetworkReductionData,
        primary_star_arc::PSY.Arc,
        secondary_star_arc::PSY.Arc,
        tertiary_star_arc::PSY.Arc,
        br::PSY.ThreeWindingTransformer
    )

Add a three-winding transformer to the transformer mapping in NetworkReductionData.

Three-winding transformers are modeled using a star (wye) configuration with three arcs
connecting to a virtual star bus. Each available winding is mapped separately.

# Arguments
- `nr::NetworkReductionData`: Network reduction data to modify
- `primary_star_arc::PSY.Arc`: Arc for primary winding
- `secondary_star_arc::PSY.Arc`: Arc for secondary winding
- `tertiary_star_arc::PSY.Arc`: Arc for tertiary winding
- `br::PSY.ThreeWindingTransformer`: Three-winding transformer to add

# Implementation Details
- Only adds arcs for available windings (checked via PSY.get_available_*)
- Maintains transformer3W_map and reverse_transformer3W_map
- Each winding is numbered (1=primary, 2=secondary, 3=tertiary)
"""
function add_to_branch_maps!(
    nr::NetworkReductionData,
    primary_star_arc::PSY.Arc,
    secondary_star_arc::PSY.Arc,
    tertiary_star_arc::PSY.Arc,
    br::PSY.ThreeWindingTransformer,
)
    transformer3W_map = get_transformer3W_map(nr)
    reverse_transformer3W_map = get_reverse_transformer3W_map(nr)
    if PSY.get_available_primary(br)
        primary_star_arc_tuple = get_arc_tuple(primary_star_arc, nr)
        winding = ThreeWindingTransformerWinding(br, 1)
        transformer3W_map[primary_star_arc_tuple] = winding
        reverse_transformer3W_map[winding] = primary_star_arc_tuple
    end
    if PSY.get_available_secondary(br)
        secondary_star_arc_tuple = get_arc_tuple(secondary_star_arc, nr)
        winding = ThreeWindingTransformerWinding(br, 2)
        transformer3W_map[secondary_star_arc_tuple] = winding
        reverse_transformer3W_map[winding] = secondary_star_arc_tuple
    end
    if PSY.get_available_tertiary(br)
        tertiary_star_arc_tuple = get_arc_tuple(tertiary_star_arc, nr)
        winding = ThreeWindingTransformerWinding(br, 3)
        transformer3W_map[tertiary_star_arc_tuple] = winding
        reverse_transformer3W_map[winding] = tertiary_star_arc_tuple
    end
    return
end

"""
    add_branch_entries_to_ybus!(
        y11::Vector{ComplexF32},
        y12::Vector{ComplexF32},
        y21::Vector{ComplexF32},
        y22::Vector{ComplexF32},
        branch_ix::Int,
        br::PSY.ACTransmission
    )

Add Y-bus matrix entries for an AC transmission branch to the admittance vectors.

This function calculates the 2×2 Y-bus entries for a branch using `ybus_branch_entries()`
and stores them in the provided vectors at the specified index. The entries represent
the Pi-model admittances: Y11 (from-bus self), Y12 (from-to mutual), Y21 (to-from mutual),
and Y22 (to-bus self).

# Arguments
- `y11::Vector{ComplexF32}`: Vector for from-bus self admittances
- `y12::Vector{ComplexF32}`: Vector for from-to mutual admittances
- `y21::Vector{ComplexF32}`: Vector for to-from mutual admittances
- `y22::Vector{ComplexF32}`: Vector for to-bus self admittances
- `branch_ix::Int`: Index where to store the branch entries
- `br::PSY.ACTransmission`: AC transmission branch

# Implementation Details
- Calls `ybus_branch_entries()` to compute Pi-model parameters
- Stores results directly in the provided vectors
- Used during Y-bus matrix assembly process
"""
function add_branch_entries_to_ybus!(
    y11::Vector{ComplexF32},
    y12::Vector{ComplexF32},
    y21::Vector{ComplexF32},
    y22::Vector{ComplexF32},
    branch_ix::Int,
    br::PSY.ACTransmission,
)
    Y11, Y12, Y21, Y22 = ybus_branch_entries(br)
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
        adj::SparseArrays.SparseMatrixCSC{Int8, Int},
        br::PSY.ACTransmission
    )

Update indexing structures when adding an AC transmission branch to the Y-bus.

This function handles the bookkeeping required when adding a branch: updates network
reduction mappings, sets adjacency matrix entries, and records from/to bus indices
for the branch in the Y-bus construction vectors.

# Arguments
- `num_bus::Dict{Int, Int}`: Mapping from bus numbers to matrix indices
- `branch_ix::Int`: Branch index in the vectors
- `nr::NetworkReductionData`: Network reduction data to update
- `fb::Vector{Int}`: Vector of from-bus indices
- `tb::Vector{Int}`: Vector of to-bus indices
- `adj::SparseArrays.SparseMatrixCSC{Int8, Int}`: Adjacency matrix
- `br::PSY.ACTransmission`: AC transmission branch to add

# Implementation Details
- Calls `add_to_branch_maps!()` to update reduction mappings
- Updates adjacency matrix with branch connectivity
- Records bus indices in from/to vectors for sparse matrix construction
"""
function add_branch_entries_to_indexing_maps!(
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    nr::NetworkReductionData,
    fb::Vector{Int},
    tb::Vector{Int},
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    br::PSY.ACTransmission,
)
    arc = PSY.get_arc(br)
    add_to_branch_maps!(nr, arc, br)
    bus_from_no, bus_to_no = get_bus_indices(arc, num_bus, nr)
    adj[bus_from_no, bus_to_no] = 1
    adj[bus_to_no, bus_from_no] = -1
    fb[branch_ix] = bus_from_no
    tb[branch_ix] = bus_to_no
    return
end

_get_shunt(br::PSY.ACTransmission, node::Symbol) =
    PSY.get_g(br)[node] + 1im * PSY.get_b(br)[node]
_get_shunt(::PSY.DiscreteControlledACBranch, ::Symbol) = zero(ComplexF32)

"""Ybus entries for a `Line` or a `DiscreteControlledACBranch`."""
function ybus_branch_entries(br::PSY.ACTransmission)
    Y_l = (1 / (PSY.get_r(br) + PSY.get_x(br) * 1im))
    Y11 = Y_l + _get_shunt(br, :from)
    if !isfinite(Y11) || !isfinite(Y_l)
        error(
            "Data in $(PSY.get_name(br)) is incorrect. r = $(PSY.get_r(br)), x = $(PSY.get_x(br))",
        )
    end
    Y12 = -Y_l
    Y21 = Y12
    Y22 = Y_l + _get_shunt(br, :to)
    return (Y11, Y12, Y21, Y22)
end

function ybus_branch_entries(parallel_br::BranchesParallel)
    arc = get_arc_tuple(first(parallel_br))
    Y11 = Y12 = Y21 = Y22 = zero(ComplexF32)
    for br in parallel_br
        # All branches in BranchesParallel are have the same orientation when constructed in add_to_branch_maps!
        (y11, y12, y21, y22) = ybus_branch_entries(br)
        Y11 += y11
        Y12 += y12
        Y21 += y21
        Y22 += y22
    end
    return (Y11, Y12, Y21, Y22)
end

function ybus_branch_entries(br::BranchesSeries)
    ybus_chain = _build_chain_ybus(br)
    ybus_reduced = _reduce_internal_nodes(ybus_chain)
    return ybus_reduced[1, 1], ybus_reduced[1, 2], ybus_reduced[2, 1], ybus_reduced[2, 2]
end

_get_tap(::PSY.Transformer2W) = one(ComplexF32)
_get_tap(br::PSY.TwoWindingTransformer) = PSY.get_tap(br)

"""Ybus entries for a `Transformer2W`, `TapTransformer`, or `PhaseShiftingTransformer`."""
function ybus_branch_entries(br::PSY.TwoWindingTransformer)
    Y_t = 1 / (PSY.get_r(br) + PSY.get_x(br) * 1im)
    tap = _get_tap(br) * exp(PSY.get_α(br) * 1im)
    c_tap = _get_tap(br) * exp(-1 * PSY.get_α(br) * 1im)
    y_shunt = PSY.get_primary_shunt(br)
    Y11 = Y_t / abs2(tap)
    if !isfinite(Y11) || !isfinite(Y_t) || !isfinite(y_shunt * c_tap)
        error(
            "Data in $(summary(br)) gives a non-finite Ybus entry; check input data.",
        )
    end
    Y12 = -Y_t / c_tap
    Y21 = -Y_t / tap
    Y22 = Y_t
    return (Y11 + y_shunt, Y12, Y21, Y22)
end

"""Ybus branch entries for an arc in the wye model of a `ThreeWindingTransformer`."""
function ybus_branch_entries(tp::ThreeWindingTransformerWinding)
    br = get_transformer(tp)
    winding_number = get_winding_number(tp)
    if winding_number == 1
        Y_t = 1 / (PSY.get_r_primary(br) + PSY.get_x_primary(br) * 1im)
        tap = PSY.get_primary_turns_ratio(br) * exp(PSY.get_α_primary(br) * 1im)
        c_tap = PSY.get_primary_turns_ratio(br) * exp(-1 * PSY.get_α_primary(br) * 1im)
        Y11 = Y_t / abs2(tap)
        y_shunt = PSY.get_g(br) + im * PSY.get_b(br)
        if !isfinite(Y11) || !isfinite(y_shunt)
            error(
                "Data in $(PSY.get_name(br)) is incorrect.
                r_p = $(PSY.get_r_primary(br)), x_p = $(PSY.get_x_primary(br)), primary_turns_ratio = $(PSY.get_primary_turns_ratio(br))",
            )
        end
        # primary bus alone gets the shunt term
        Y11 += y_shunt
    elseif winding_number == 2
        Y_t = 1 / (PSY.get_r_secondary(br) + PSY.get_x_secondary(br) * 1im)
        tap = PSY.get_secondary_turns_ratio(br) * exp(PSY.get_α_secondary(br) * 1im)
        c_tap = PSY.get_secondary_turns_ratio(br) * exp(-1 * PSY.get_α_secondary(br) * 1im)
        Y11 = Y_t / abs2(tap)
        if !isfinite(Y11)
            error(
                "Data in $(PSY.get_name(br)) is incorrect.
                r_s = $(PSY.get_r_secondary(br)), x_s = $(PSY.get_x_secondary(br)), secondary_turns_ratio = $(PSY.get_secondary_turns_ratio(br))",
            )
        end
    elseif winding_number == 3
        Y_t = 1 / (PSY.get_r_tertiary(br) + PSY.get_x_tertiary(br) * 1im)
        tap = PSY.get_tertiary_turns_ratio(br) * exp(PSY.get_α_tertiary(br) * 1im)
        c_tap = PSY.get_tertiary_turns_ratio(br) * exp(-1 * PSY.get_α_tertiary(br) * 1im)
        Y11 = Y_t / abs2(tap)
        if !isfinite(Y11)
            error(
                "Data in $(PSY.get_name(br)) is incorrect.
                r_t = $(PSY.get_r_tertiary(br)), x_t = $(PSY.get_x_tertiary(br)), tertiary_turns_ratio = $(PSY.get_tertiary_turns_ratio(br))",
            )
        end
    end
    Y12 = (-Y_t / c_tap)
    Y21 = (-Y_t / tap)
    Y22 = Y_t
    return (Y11, Y12, Y21, Y22)
end

"""Handles ybus entries for most 2-node AC branches. The types handled here are:
`Line`, `DiscreteControlledACBranch`, `Transformer2W`, `TapTransformer`, and `PhaseShiftingTransformer`.
"""
function _ybus!(
    y11::Vector{ComplexF32},
    y12::Vector{ComplexF32},
    y21::Vector{ComplexF32},
    y22::Vector{ComplexF32},
    br::PSY.ACTransmission,
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    nr::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
)
    add_branch_entries_to_indexing_maps!(num_bus, branch_ix, nr, fb, tb, adj, br)
    add_branch_entries_to_ybus!(y11, y12, y21, y22, branch_ix, br)
    return
end

function _ybus!(
    y11::Vector{ComplexF32},
    y12::Vector{ComplexF32},
    y21::Vector{ComplexF32},
    y22::Vector{ComplexF32},
    br::PSY.DynamicBranch,
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    nr::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
)
    _ybus!(
        y11,
        y12,
        y21,
        y22,
        br.branch,
        num_bus,
        branch_ix,
        fb,
        tb,
        nr,
        adj,
    )
    return
end

function _ybus!(
    y11::Vector{ComplexF32},
    y12::Vector{ComplexF32},
    y21::Vector{ComplexF32},
    y22::Vector{ComplexF32},
    br::PSY.ThreeWindingTransformer,
    num_bus::Dict{Int, Int},
    offset_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    nr::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
)
    primary_star_arc = PSY.get_primary_star_arc(br)
    secondary_star_arc = PSY.get_secondary_star_arc(br)
    tertiary_star_arc = PSY.get_tertiary_star_arc(br)
    add_to_branch_maps!(nr, primary_star_arc, secondary_star_arc, tertiary_star_arc, br)
    primary_available = PSY.get_available_primary(br)
    secondary_available = PSY.get_available_secondary(br)
    tertiary_available = PSY.get_available_tertiary(br)
    n_entries = 0
    if primary_available
        primary_ix, star_ix = get_bus_indices(primary_star_arc, num_bus, nr)
        adj[primary_ix, star_ix] = 1
        adj[star_ix, primary_ix] = -1
        fb[offset_ix + ix + n_entries] = primary_ix
        tb[offset_ix + ix + n_entries] = star_ix
        (Y11, Y12, Y21, Y22) = ybus_branch_entries(ThreeWindingTransformerWinding(br, 1))
        y11[offset_ix + ix + n_entries] = Y11
        y12[offset_ix + ix + n_entries] = Y12
        y21[offset_ix + ix + n_entries] = Y21
        y22[offset_ix + ix + n_entries] = Y22
        n_entries += 1
    end
    if secondary_available
        secondary_ix, star_ix = get_bus_indices(secondary_star_arc, num_bus, nr)
        adj[secondary_ix, star_ix] = 1
        adj[star_ix, secondary_ix] = -1
        fb[offset_ix + ix + n_entries] = secondary_ix
        tb[offset_ix + ix + n_entries] = star_ix
        (Y11, Y12, Y21, Y22) = ybus_branch_entries(ThreeWindingTransformerWinding(br, 2))
        y11[offset_ix + ix + n_entries] = Y11
        y12[offset_ix + ix + n_entries] = Y12
        y21[offset_ix + ix + n_entries] = Y21
        y22[offset_ix + ix + n_entries] = Y22
        n_entries += 1
    end
    if tertiary_available
        tertiary_ix, star_ix = get_bus_indices(tertiary_star_arc, num_bus, nr)
        adj[tertiary_ix, star_ix] = 1
        adj[star_ix, tertiary_ix] = -1
        fb[offset_ix + ix + n_entries] = tertiary_ix
        tb[offset_ix + ix + n_entries] = star_ix
        (Y11, Y12, Y21, Y22) = ybus_branch_entries(ThreeWindingTransformerWinding(br, 3))
        y11[offset_ix + ix + n_entries] = Y11
        y12[offset_ix + ix + n_entries] = Y12
        y21[offset_ix + ix + n_entries] = Y21
        y22[offset_ix + ix + n_entries] = Y22
        n_entries += 1
    end
    return n_entries
end

function _ybus!(
    ysh::Vector{ComplexF32},
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
        error(
            "Data in $(PSY.get_name(fa)) is incorrect. Y = $(Y)",
        )
    end
    ysh[fa_ix] = Y
    return
end

#Note - PSSE does not include switched admittances in ymatrix
function _ybus!(
    ysh::Vector{ComplexF32},
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
    ysh::Vector{ComplexF32},
    fa::PSY.StandardLoad,
    num_bus::Dict{Int, Int},
    fa_ix::Int,
    sb::Vector{Int},
    nr::NetworkReductionData,
)
    bus_no = get_bus_index(fa, num_bus, nr)
    Y = PSY.get_impedance_active_power(fa) - im * PSY.get_impedance_reactive_power(fa)
    if !isfinite(Y)
        error(
            "Data in $(PSY.get_name(fa)) is incorrect. Y = $(Y)",
        )
    end
    sb[fa_ix] = bus_no
    ysh[fa_ix] = Y
    return
end

function _buildybus!(
    network_reduction_data::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    branches,
    transformer_3w::Vector{PSY.ThreeWindingTransformer},
    num_bus::Dict{Int, Int},
    fixed_admittances::Vector{PSY.FixedAdmittance},
    switched_admittances::Vector{PSY.SwitchedAdmittance},
    standard_loads::Vector{PSY.StandardLoad},
)
    branch_entries_transformer_3w = 0
    for br in transformer_3w
        branch_entries_transformer_3w += count([
            PSY.get_available_primary(br),
            PSY.get_available_secondary(br),
            PSY.get_available_tertiary(br),
        ])
    end
    branchcount = length(branches) + branch_entries_transformer_3w
    branchcount_no_3w = length(branches)
    fa_count = length(fixed_admittances)
    sa_count = length(switched_admittances)
    sl_count = length(standard_loads)
    fb = zeros(Int, branchcount)
    tb = zeros(Int, branchcount)
    sb = zeros(Int, fa_count + sa_count + sl_count)

    y11 = zeros(ComplexF32, branchcount)
    y12 = zeros(ComplexF32, branchcount)
    y21 = zeros(ComplexF32, branchcount)
    y22 = zeros(ComplexF32, branchcount)
    ysh = zeros(ComplexF32, fa_count + sa_count + sl_count)

    for (ix, b) in enumerate(branches)
        if PSY.get_name(b) == "init"
            throw(DataFormatError("The data in Branch is invalid"))
        end
        _ybus!(y11, y12, y21, y22, b, num_bus, ix, fb, tb, network_reduction_data, adj)
    end

    ix = 1
    for b in transformer_3w
        if PSY.get_name(b) == "init"
            throw(DataFormatError("The data in Transformer3W is invalid"))
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
            adj,
        )
        ix += n_entries
    end
    for (ix, fa) in enumerate([fixed_admittances; switched_admittances; standard_loads])
        _ybus!(ysh, fa, num_bus, ix, sb, network_reduction_data)
    end
    return (
        y11,
        y12,
        y21,
        y22,
        ysh,
        fb,
        tb,
        sb,
    )
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
function Ybus(
    sys::PSY.System;
    make_arc_admittance_matrices::Bool = false,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    include_constant_impedance_loads = true,
    subnetwork_algorithm = iterative_union_find,
)
    units_base = PSY.get_units_base(sys)
    if units_base != "SYSTEM_BASE"
        @warn "Setting the system unit base from $units_base to SYSTEM_BASE for matrix construction"
        PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    end
    ref_bus_numbers = Set{Int}()
    nr = NetworkReductionData()
    bus_reduction_map = get_bus_reduction_map(nr)
    reverse_bus_search_map = get_reverse_bus_search_map(nr)

    #Checking for isolated buses; building bus map.
    for b in PSY.get_components(PSY.ACBus, sys)
        !PSY.get_available(b) && continue
        if PSY.get_bustype(b) != ACBusTypes.ISOLATED
            bus_reduction_map[PSY.get_number(b)] = Set{Int}()
            if PSY.get_bustype(b) == ACBusTypes.REF
                push!(ref_bus_numbers, PSY.get_number(b))
            end
        else
            @debug "Found available isolated bus $(PSY.get_name(b)) with number $(PSY.get_number(b)). This is excluded from the Ybus build."
            push!(nr.removed_buses, PSY.get_number(b))
        end
    end

    #Building map for removed Breaker/Switches
    breaker_switches = Vector{PSY.DiscreteControlledACBranch}()
    for br in PSY.get_components(PSY.DiscreteControlledACBranch, sys)
        !PSY.get_available(br) && continue
        r = PSY.get_r(br)
        x = PSY.get_x(br)
        status = PSY.get_branch_status(br)
        if status == PSY.DiscreteControlledBranchStatus.CLOSED
            if r == 0.0 && x < ZERO_IMPEDANCE_LINE_REACTANCE_THRESHOLD
                from_bus_number = PSY.get_number(PSY.get_from(PSY.get_arc(br)))
                to_bus_number = PSY.get_number(PSY.get_to(PSY.get_arc(br)))
                if haskey(reverse_bus_search_map, from_bus_number)
                    reduced_from_bus_number = reverse_bus_search_map[from_bus_number]
                    push!(
                        get!(bus_reduction_map, reduced_from_bus_number, Set{Int}),
                        to_bus_number,
                    )
                    delete!(bus_reduction_map, to_bus_number)
                    reverse_bus_search_map[to_bus_number] = reduced_from_bus_number
                else
                    s1 = get(bus_reduction_map, from_bus_number, Set{Int}())
                    s2 = union(
                        get(bus_reduction_map, to_bus_number, Set{Int}(to_bus_number)),
                        to_bus_number,
                    )
                    bus_reduction_map[from_bus_number] = union(s1, s2)
                    delete!(bus_reduction_map, to_bus_number)
                    for x in s2
                        reverse_bus_search_map[x] = from_bus_number
                    end
                end
                push!(nr.removed_arcs, (from_bus_number, to_bus_number))
            else
                push!(breaker_switches, br)
            end
        end
    end

    bus_ax = sort!([x for x in keys(bus_reduction_map)])
    axes = (bus_ax, bus_ax)
    bus_lookup = Dict{Int, Int}()
    lookup = (bus_lookup, bus_lookup)
    busnumber = length(bus_ax)
    for (ix, b) in enumerate(bus_ax)
        bus_lookup[b] = ix
    end
    adj = SparseArrays.spdiagm(ones(Int8, busnumber))
    branches = collect(
        PSY.get_components(
            x ->
                PSY.get_available(x) &&
                    typeof(x) ∉ [
                        PSY.Transformer3W,
                        PSY.PhaseShiftingTransformer3W,
                        PSY.DiscreteControlledACBranch,
                    ],
            PSY.ACTransmission,
            sys,
        ),
    )
    branches = vcat(branches, breaker_switches)
    transformer_3W =
        collect(
            PSY.get_components(
                x -> PSY.get_available(x),
                PSY.ThreeWindingTransformer,
                sys,
            ),
        )
    fixed_admittances = collect(
        PSY.get_components(
            x ->
                PSY.get_available(x) &&
                    PSY.get_bustype(PSY.get_bus(x)) != ACBusTypes.ISOLATED,
            PSY.FixedAdmittance,
            sys,
        ),
    )
    switched_admittances =
        collect(
            PSY.get_components(
                x ->
                    PSY.get_available(x) &&
                        PSY.get_bustype(PSY.get_bus(x)) != ACBusTypes.ISOLATED,
                PSY.SwitchedAdmittance,
                sys,
            ),
        )
    standard_loads = if include_constant_impedance_loads
        collect(
            PSY.get_components(
                x ->
                    PSY.get_available(x) &&
                        PSY.get_bustype(PSY.get_bus(x)) != ACBusTypes.ISOLATED,
                PSY.StandardLoad,
                sys,
            ),
        )
    else
        PSY.StandardLoad[]
    end
    y11, y12, y21, y22, ysh, fb, tb, sb =
        _buildybus!(
            nr,
            adj,
            branches,
            transformer_3W,
            bus_lookup,
            fixed_admittances,
            switched_admittances,
            standard_loads,
        )
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
        yft_data = SparseArrays.sparse(
            rows_ix_nnz,
            [fb; tb],
            [y11; y12],
            arc_count,
            busnumber,
        )
        ytf_data = SparseArrays.sparse(
            rows_ix_nnz,
            [tb; fb],
            [y22; y21],
            arc_count,
            busnumber,
        )
        arc_admittance_from_to = ArcAdmittanceMatrix(
            yft_data,
            (arc_axis, bus_ax),
            (arc_lookup, bus_lookup),
            nr,
            :FromTo,
        )
        arc_admittance_to_from = ArcAdmittanceMatrix(
            ytf_data,
            (arc_axis, bus_ax),
            (arc_lookup, bus_lookup),
            nr,
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
        )
        if length(subnetworks) > 1
            @warn "More than one island found; Network is not connected"
        end
    else
        subnetworks = Dict{Int, Set{Int}}(bus_lookup[1] => Set(bus_ax))
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
        nr,
        arc_admittance_from_to,
        arc_admittance_to_from,
    )

    for nr in network_reductions
        ybus = build_reduced_ybus(ybus, sys, nr)
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
    subnetwork_axes = sizehint!(
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
    return _apply_reduction(ybus, network_reduction_data)
end

function _apply_reduction(ybus::Ybus, nr_new::NetworkReductionData)
    # These quantities are modified and used to construct the new Ybus
    data = get_data(ybus)
    adjacency_data = ybus.adjacency_data
    bus_lookup = get_bus_lookup(ybus)
    nr = get_network_reduction_data(ybus)

    bus_numbers_to_remove = _apply_bus_reductions!(nr, nr_new)

    # Add additional entries to the ybus corresponding to the equivalent series arcs
    new_y_ft, new_y_tf = _add_series_branches_to_ybus!(
        ybus.data,
        get_bus_lookup(ybus),
        ybus.arc_admittance_from_to,
        ybus.arc_admittance_to_from,
        nr_new.series_branch_map,
        nr,
    )
    _modify_radial_connections!(
        ybus.data,
        get_bus_lookup(ybus),
        nr,
        nr_new.radial_arc_to_surviving_bus,
        nr_new.reductions,
    )
    _remove_arcs_from_branch_maps!(nr, nr_new)
    _apply_added_components!(nr, nr_new, data, bus_lookup)
    _apply_series_branch_maps!(nr, nr_new)
    add_reduction!(nr.reductions, nr_new.reductions)
    union!(nr.irreducible_buses, nr_new.irreducible_buses)

    # Remake bus axes, lookup, and data matrices without removed buses:
    bus_ax = setdiff(get_bus_axis(ybus), bus_numbers_to_remove)
    bus_lookup = make_ax_ref(bus_ax)
    bus_ix = [get_bus_lookup(ybus)[x] for x in bus_ax]
    adjacency_data = adjacency_data[bus_ix, bus_ix]
    data = data[bus_ix, bus_ix]

    subnetwork_axes, arc_subnetwork_axis =
        _make_subnetwork_axes(ybus, bus_numbers_to_remove, nr_new.removed_arcs)

    if new_y_ft !== nothing
        arc_ax = setdiff(get_arc_axis(new_y_ft), nr_new.removed_arcs)
        arc_remove_ixs = indexin(nr_new.removed_arcs, get_arc_axis(new_y_ft))
        arc_keep_ixs = setdiff(collect(1:length(get_arc_axis(new_y_ft))), arc_remove_ixs)
        arc_lookup = make_ax_ref(arc_ax)
        yft_data = new_y_ft.data[arc_keep_ixs, bus_ix]
        ytf_data = new_y_tf.data[arc_keep_ixs, bus_ix]

        arc_admittance_from_to = ArcAdmittanceMatrix(
            yft_data,
            (arc_ax, bus_ax),
            (arc_lookup, bus_lookup),
            nr,
            :FromTo,
        )
        arc_admittance_to_from = ArcAdmittanceMatrix(
            ytf_data,
            (arc_ax, bus_ax),
            (arc_lookup, bus_lookup),
            nr,
            :ToFrom,
        )
    else
        arc_admittance_from_to = ybus.arc_admittance_from_to
        arc_admittance_to_from = ybus.arc_admittance_to_from
    end

    return Ybus(
        data,
        adjacency_data,
        (bus_ax, bus_ax),
        (bus_lookup, bus_lookup),
        subnetwork_axes,
        arc_subnetwork_axis,
        nr,
        arc_admittance_from_to,
        arc_admittance_to_from,
    )
end

function _apply_bus_reductions!(nr::NetworkReductionData, nr_new::NetworkReductionData)
    bus_numbers_to_remove = Vector{Int}()
    for (k, v) in nr_new.reverse_bus_search_map
        nr.reverse_bus_search_map[k] = v
        push!(bus_numbers_to_remove, k)
    end
    for x in nr_new.removed_buses
        push!(nr.removed_buses, x)
        push!(bus_numbers_to_remove, x)
        delete!(nr.bus_reduction_map, x)
    end
    for (k, v) in nr_new.bus_reduction_map
        if haskey(nr.bus_reduction_map, k)
            union!(nr.bus_reduction_map[k], nr_new.bus_reduction_map[k])
            for x in v
                delete!(nr.bus_reduction_map, x)
            end
        else
            error("Bus $k was previously reduced")
        end
    end
    return bus_numbers_to_remove
end

function _remove_arcs_from_branch_maps!(
    nr::NetworkReductionData,
    nr_new::NetworkReductionData,
)
    remake_reverse_direct_branch_map = false
    remake_reverse_parallel_branch_map = false
    remake_reverse_series_branch_map = false
    remake_reverse_transformer3W_map = false
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
        elseif haskey(nr.transformer3W_map, x)
            remake_reverse_transformer3W_map = true
            delete!(nr.transformer3W_map, x)
        end
    end
    remake_reverse_direct_branch_map && _remake_reverse_direct_branch_map!(nr)
    remake_reverse_parallel_branch_map && _remake_reverse_parallel_branch_map!(nr)
    remake_reverse_series_branch_map && _remake_reverse_series_branch_map!(nr)
    remake_reverse_transformer3W_map && _remake_reverse_transformer3W_map!(nr)
    return
end

function _apply_added_components!(
    nr::NetworkReductionData,
    nr_new::NetworkReductionData,
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int},
    bus_lookup::Dict{Int, Int},
)
    if !isempty(nr_new.added_branch_map) && !isempty(nr.added_branch_map) ||
       !isempty(nr_new.added_admittance_map) && !isempty(nr.added_admittance_map)
        error(
            "Only the final applied reduction can add new branches and/or admittances to the Ybus (e.g. Ward Reduction)",
        )
    end
    nr.added_branch_map = nr_new.added_branch_map
    nr.added_admittance_map = nr_new.added_admittance_map
    for (bus_no, admittance) in nr.added_admittance_map
        data[bus_lookup[bus_no], bus_lookup[bus_no]] += admittance
    end
    for (bus_tuple, admittance) in nr.added_branch_map
        bus_from, bus_to = bus_tuple
        data[bus_lookup[bus_from], bus_lookup[bus_to]] += admittance
        data[bus_lookup[bus_to], bus_lookup[bus_from]] += admittance
    end
    return
end

function _apply_series_branch_maps!(nr::NetworkReductionData, nr_new::NetworkReductionData)
    if isempty(nr.series_branch_map)
        nr.series_branch_map = nr_new.series_branch_map
        nr.reverse_series_branch_map = nr_new.reverse_series_branch_map
    elseif !isempty(nr_new.series_branch_map) && !isempty(nr.series_branch_map)
        error(
            "Cannot compose series branch maps; should not apply multiple reductions that generate series branch maps",
        )
    end
    return
end

function _make_subnetwork_axes(
    ybus::Ybus,
    bus_numbers_to_remove::Vector{Int},
    arcs_to_remove::Set{Tuple{Int, Int}},
)
    subnetwork_axes = deepcopy(ybus.subnetwork_axes)
    arc_subnetwork_axis = deepcopy(ybus.arc_subnetwork_axis)
    for (k, values) in subnetwork_axes
        if k in bus_numbers_to_remove
            @warn "Reference bus removed during reduction; assigning arbitrary reference bus."
            axis_1, axis_2 = pop!(subnetwork_axes, k)
            new_ref_bus = pop!(axis_1)
            pop!(axis_2)
            subnetwork_axes[new_ref_bus] = (axis_1, axis_2)
            # If a reference bus key is reduced, change the arc subnetwork axis key as well:
            arc_subnetwork_axis[new_ref_bus] = pop!(arc_subnetwork_axis, k)
        end
    end
    for (k, values) in subnetwork_axes
        new_values = setdiff(values[1], bus_numbers_to_remove)
        subnetwork_axes[k] = (new_values, new_values)
    end
    for (k, values) in arc_subnetwork_axis
        arc_subnetwork_axis[k] = setdiff(values, arcs_to_remove)
    end
    return subnetwork_axes, arc_subnetwork_axis
end

function _modify_radial_connections!(
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int64},
    bus_lookup::Dict{Int, Int},
    nrd_old::NetworkReductionData,
    radial_arc_to_surviving_bus::Dict{Tuple{Int, Int}, Int},
    reductions::ReductionContainer,
)
    !has_radial_reduction(reductions) && return
    for (arc, bus) in radial_arc_to_surviving_bus
        arc_entry = _get_entry(arc, nrd_old)
        y11, _, _, y22 = ybus_branch_entries(arc_entry)
        if arc[1] == bus
            data[bus_lookup[arc[1]], bus_lookup[arc[1]]] -= y11
        elseif arc[2] == bus
            data[bus_lookup[arc[2]], bus_lookup[arc[2]]] -= y22
        else
            error("Bad data in radial_arc_to_surviving_bus map")
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
    elseif haskey(nrd.transformer3W_map, arc)
        return nrd.transformer3W_map[arc]
    else
        error("Key $arc not found in any of the maps")
    end
end

function _add_series_branches_to_ybus!(
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int64},
    bus_lookup::Dict{Int, Int},
    yft::Nothing,
    ytf::Nothing,
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries},
    nrd,
)
    for (equivalent_arc, series_map_entry) in series_branch_map
        ordered_bus_numbers, segment_orientations =
            _get_chain_data(equivalent_arc, series_map_entry, nrd)
        ordered_bus_indices = [bus_lookup[x] for x in ordered_bus_numbers]
        equivalent_arc_indices = (ordered_bus_indices[1], ordered_bus_indices[end])
        ybus_isolated_d2_chain = _build_chain_ybus(series_map_entry)
        ybus_boundary_isolated_d2_chain = _reduce_internal_nodes(ybus_isolated_d2_chain)
        _apply_d2_chain_ybus!(
            data,
            ybus_isolated_d2_chain,
            ybus_boundary_isolated_d2_chain,
            equivalent_arc_indices,
        )
    end
    return yft, ytf
end

function _add_series_branches_to_ybus!(
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int64},
    bus_lookup::Dict{Int, Int},
    yft::ArcAdmittanceMatrix,
    ytf::ArcAdmittanceMatrix,
    series_branch_map::Dict{Tuple{Int, Int}, BranchesSeries},
    nrd::NetworkReductionData,
)
    arc_lookup = get_arc_lookup(yft)
    arc_axis = get_arc_axis(yft)
    I_yft, J_yft, V_yft = SparseArrays.findnz(yft.data)
    I_ytf, J_ytf, V_ytf = SparseArrays.findnz(ytf.data)
    row_ix = size(yft)[1] + 1
    n_buses = size(yft)[2]
    for (equivalent_arc, series_map_entry) in series_branch_map
        ordered_bus_numbers, segment_orientations =
            _get_chain_data(equivalent_arc, series_map_entry, nrd)
        ordered_bus_indices = [bus_lookup[x] for x in ordered_bus_numbers]
        equivalent_arc_indices = (ordered_bus_indices[1], ordered_bus_indices[end])
        ybus_isolated_d2_chain = _build_chain_ybus(series_map_entry)
        ybus_boundary_isolated_d2_chain = _reduce_internal_nodes(ybus_isolated_d2_chain)
        _apply_d2_chain_ybus!(
            data,
            ybus_isolated_d2_chain,
            ybus_boundary_isolated_d2_chain,
            equivalent_arc_indices,
        )

        push!(arc_axis, equivalent_arc)
        push!(I_yft, row_ix)
        push!(I_yft, row_ix)
        push!(J_yft, equivalent_arc_indices[1])
        push!(J_yft, equivalent_arc_indices[2])
        push!(V_yft, ybus_boundary_isolated_d2_chain[1, 1])
        push!(V_yft, ybus_boundary_isolated_d2_chain[1, 2])
        push!(I_ytf, row_ix)
        push!(I_ytf, row_ix)
        push!(J_ytf, equivalent_arc_indices[2])
        push!(J_ytf, equivalent_arc_indices[1])
        push!(V_ytf, ybus_boundary_isolated_d2_chain[2, 2])
        push!(V_ytf, ybus_boundary_isolated_d2_chain[2, 1])
        row_ix += 1
    end
    yft_data = SparseArrays.sparse(I_yft, J_yft, V_yft, row_ix - 1, n_buses)
    ytf_data = SparseArrays.sparse(I_ytf, J_ytf, V_ytf, row_ix - 1, n_buses)

    arc_admittance_from_to = ArcAdmittanceMatrix(
        yft_data,
        (arc_axis, get_bus_axis(yft)),
        (arc_lookup, get_bus_lookup(yft)),
        nrd,
        :FromTo,
    )
    arc_admittance_to_from = ArcAdmittanceMatrix(
        ytf_data,
        (arc_axis, get_bus_axis(ytf)),
        (arc_lookup, get_bus_lookup(ytf)),
        nrd,
        :ToFrom,
    )
    return arc_admittance_from_to, arc_admittance_to_from
end

function _apply_d2_chain_ybus!(
    ybus_full::SparseArrays.SparseMatrixCSC{ComplexF32, Int64},
    ybus_chain::Matrix{ComplexF32},
    ybus_chain_reduced::Matrix{ComplexF32},
    equivalent_arc_indices::Tuple{Int, Int},
)
    equivalent_from_index, equivalent_to_index = equivalent_arc_indices
    from_bus_entry_difference = ybus_chain_reduced[1, 1] - ybus_chain[1, 1]
    from_to_bus_entry = ybus_chain_reduced[1, 2]
    to_from_bus_entry = ybus_chain_reduced[2, 1]
    to_bus_entry_difference = ybus_chain_reduced[end, end] - ybus_chain[end, end]

    ybus_full[equivalent_from_index, equivalent_from_index] += from_bus_entry_difference
    ybus_full[equivalent_from_index, equivalent_to_index] = from_to_bus_entry
    ybus_full[equivalent_to_index, equivalent_from_index] = to_from_bus_entry
    ybus_full[equivalent_to_index, equivalent_to_index] += to_bus_entry_difference
    return
end

function _build_chain_ybus(
    series_chain::BranchesSeries,
)
    segment_orientations = series_chain.segment_orientations
    fb = Vector{Int}()
    tb = Vector{Int}()
    y11 = Vector{ComplexF32}()
    y12 = Vector{ComplexF32}()
    y21 = Vector{ComplexF32}()
    y22 = Vector{ComplexF32}()
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
        y11::Vector{ComplexF32},
        y12::Vector{ComplexF32},
        y21::Vector{ComplexF32},
        y22::Vector{ComplexF32},
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
- `y11::Vector{ComplexF32}`: Vector for from-bus self admittances
- `y12::Vector{ComplexF32}`: Vector for from-to mutual admittances
- `y21::Vector{ComplexF32}`: Vector for to-from mutual admittances
- `y22::Vector{ComplexF32}`: Vector for to-bus self admittances
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
    y11::Vector{ComplexF32},
    y12::Vector{ComplexF32},
    y21::Vector{ComplexF32},
    y22::Vector{ComplexF32},
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    segment_orientation::Symbol,
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
        segment::BranchesParallel,
        y11::Vector{ComplexF32},
        y12::Vector{ComplexF32},
        y21::Vector{ComplexF32},
        y22::Vector{ComplexF32},
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
- `segment::BranchesParallel`: Set of parallel AC transmission branches
- `y11::Vector{ComplexF32}`: Vector for from-bus self admittances
- `y12::Vector{ComplexF32}`: Vector for from-to mutual admittances
- `y21::Vector{ComplexF32}`: Vector for to-from mutual admittances
- `y22::Vector{ComplexF32}`: Vector for to-bus self admittances
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
function add_segment_to_ybus!(
    segment::BranchesParallel,
    y11::Vector{ComplexF32},
    y12::Vector{ComplexF32},
    y21::Vector{ComplexF32},
    y22::Vector{ComplexF32},
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    segment_orientation::Symbol,
)
    for branch in segment
        add_segment_to_ybus!(branch, y11, y12, y21, y22, fb, tb, ix, segment_orientation)
    end
    return
end

function _get_chain_data(
    equivalent_arc::Tuple{Int, Int},
    series_map_entry::BranchesSeries,
    nrd::NetworkReductionData,
)
    equivalent_arc[1] == equivalent_arc[2] && @warn("trying to reduce a loop")
    ordered_bus_numbers = [equivalent_arc[1]]
    segment_orientation = Vector{Symbol}()
    for segment in series_map_entry
        arc_tuple = get_arc_tuple(segment, nrd)
        if arc_tuple[1] in ordered_bus_numbers
            push!(ordered_bus_numbers, arc_tuple[2])
            push!(segment_orientation, :FromTo)
        elseif arc_tuple[2] in ordered_bus_numbers
            push!(ordered_bus_numbers, arc_tuple[1])
            push!(segment_orientation, :ToFrom)
        else
            error("Found disconnected series chain")
        end
    end
    @assert ordered_bus_numbers[end] == equivalent_arc[2]
    return ordered_bus_numbers, segment_orientation
end

function _reduce_internal_nodes(Y::Matrix{ComplexF32})
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
    for (k, v) in nr.parallel_branch_map
        for x in v
            reverse_parallel_branch_map[x] = k
        end
    end
    nr.reverse_parallel_branch_map = reverse_parallel_branch_map
    return
end
function _remake_reverse_series_branch_map!(nr::NetworkReductionData)
    reverse_series_branch_map = Dict{PSY.ACTransmission, Tuple{Int, Int}}()
    for (k, v) in nr.series_branch_map
        for x in v
            reverse_series_branch_map[x] = k
        end
    end
    nr.reverse_series_branch_map = reverse_series_branch_map
    return
end

function _remake_reverse_transformer3W_map!(nr::NetworkReductionData)
    reverse_transformer3W_map =
        Dict{ThreeWindingTransformerWinding, Tuple{Int, Int}}()
    for (k, v) in nr.transformer3W_map
        reverse_transformer3W_map[v] = k
    end
    nr.reverse_transformer3W_map = reverse_transformer3W_map
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

function get_reduction(
    ybus::Ybus,
    sys::PSY.System,
    reduction::DegreeTwoReduction,
)
    A = AdjacencyMatrix(ybus)
    return get_reduction(A, sys, reduction)
end

function _validate_study_buses(
    ybus::Ybus,
    study_buses::Vector{Int},
)
    nrd = get_network_reduction_data(ybus)
    valid_bus_numbers =
        union(Set(keys(nrd.bus_reduction_map)), Set(keys(nrd.reverse_bus_search_map)))
    for b in study_buses
        b ∉ valid_bus_numbers &&
            throw(IS.DataFormatError("Study bus $b not found in system"))
    end
    slack_bus_numbers = get_ref_bus(ybus)
    subnetwork_axes = ybus.subnetwork_axes
    for axes in values(subnetwork_axes)
        subnetwork_bus_ax = axes[1]
        all_in = all(x -> x in Set(subnetwork_bus_ax), study_buses)
        none_in = all(x -> !(x in Set(subnetwork_bus_ax)), study_buses)
        if Set(subnetwork_bus_ax) == Set(study_buses)
            @warn "The study buses comprise an entire island; ward reduction will not modify this island and other islands will be eliminated"
        end
        if !(all_in || none_in)
            throw(
                IS.DataFormatError(
                    "All study_buses must occur in a single synchronously connected system.",
                ),
            )
        end
        for sb in slack_bus_numbers
            if sb in subnetwork_bus_ax && sb ∉ study_buses && !(none_in)
                throw(
                    IS.DataFormatError(
                        "Slack bus $sb must be included in the study buses for an area that is partially reduced",
                    ),
                )
            end
        end
    end

    return
end

function get_reduction(
    ybus::Ybus,
    ::PSY.System,
    reduction::WardReduction,
)
    study_buses = get_study_buses(reduction)
    _validate_study_buses(ybus, study_buses)
    bus_lookup = get_bus_lookup(ybus)
    bus_axis = get_bus_axis(ybus)
    A = IncidenceMatrix(ybus)
    boundary_buses = Set{Int}()
    removed_arcs = Set{Tuple{Int, Int}}()
    for arc in get_arc_axis(A)
        #Deterimine boundary buses:
        if (arc[1] ∈ study_buses) && (arc[2] ∉ study_buses)
            push!(boundary_buses, arc[1])
        elseif (arc[1] ∉ study_buses) && (arc[2] ∈ study_buses)
            push!(boundary_buses, arc[2])
        end
        #Determine arcs outside of study area
        if !(arc[1] ∈ study_buses && arc[2] ∈ study_buses)
            push!(removed_arcs, arc)
        end
    end
    bus_reduction_map, reverse_bus_search_map, added_branch_map, added_admittance_map =
        get_ward_reduction(
            ybus.data,
            bus_lookup,
            bus_axis,
            boundary_buses,
            Set(get_ref_bus(ybus)),
            study_buses,
        )

    for arc_tuple in keys(added_branch_map)
        if ybus.data[bus_lookup[arc_tuple[1]], bus_lookup[arc_tuple[2]]] != 0.0
            @warn "Equivalent branch computed during Ward reduction is in parallel with existing system branch.\\
                    Indexing into PTDF/LODF with branch names may give unexpected results for arc $arc_tuple"
        end
    end

    return NetworkReductionData(;
        bus_reduction_map = bus_reduction_map,
        reverse_bus_search_map = reverse_bus_search_map,
        removed_arcs = removed_arcs,
        added_branch_map = added_branch_map,
        added_admittance_map = added_admittance_map,
        reductions = ReductionContainer(; ward_reduction = reduction),
    )
end
