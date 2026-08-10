"""
    _build_bus_to_valid_idx(n_buses, valid_ix) -> Vector{Int}

Build the inverse of `valid_ix`: a length-`n_buses` vector where entry `b`
is the position of bus `b` inside `valid_ix`, or 0 if `b` is a reference
bus. Used by the Virtual\\* row-computation hot path so it can iterate the
nonzeros of a sparse `BA` column directly (O(nnz_col)) instead of scanning
the full bus axis (O(n_buses)) and bisecting the CSC for each entry.
"""
function _build_bus_to_valid_idx(n_buses::Int, valid_ix::Vector{Int})
    bus_to_valid_idx = zeros(Int, n_buses)
    @inbounds for (i, b) in enumerate(valid_ix)
        bus_to_valid_idx[b] = i
    end
    return bus_to_valid_idx
end

function get_bus_index(bus_no::Int, bus_lookup::Dict{Int, Int}, nr::NetworkReductionData)
    if haskey(nr.reverse_bus_search_map, bus_no)
        return bus_lookup[nr.reverse_bus_search_map[bus_no]]
    else
        return bus_lookup[bus_no]
    end
end

function get_bus_index(
    dev::PSY.Component,
    bus_lookup::Dict{Int, Int},
    nr::NetworkReductionData,
)
    bus_number = PSY.get_number(PSY.get_bus(dev))
    return get_bus_index(bus_number, bus_lookup, nr)
end

function get_bus_indices(arc::PSY.Arc, bus_lookup::Dict{Int, Int}, nr::NetworkReductionData)
    check_arc_validity(arc, IS.get_name(arc))
    fr_bus_number, to_bus_number = get_arc_tuple(arc, nr)
    return bus_lookup[fr_bus_number], bus_lookup[to_bus_number]
end

function check_arc_validity(arc::PSY.Arc, name::String)
    if PSY.get_bustype(PSY.get_from(arc)) == ACBusTypes.ISOLATED
        throw(
            IS.ConflictingInputsError(
                "Branch or arc $(name) is set available and connected to isolated bus " *
                "$(IS.get_name(PSY.get_from(arc)))",
            ),
        )
    end
    if PSY.get_bustype(PSY.get_to(arc)) == ACBusTypes.ISOLATED
        throw(
            IS.ConflictingInputsError(
                "Branch or arc $(name) is set available and connected to isolated bus " *
                "$(IS.get_name(PSY.get_to(arc)))",
            ),
        )
    end
    return
end

function get_arc_tuple(arc::PSY.Arc, nr::NetworkReductionData)
    reverse_bus_search_map = get_reverse_bus_search_map(nr)
    arc_tuple_original = get_arc_tuple(arc)
    return (
        get(reverse_bus_search_map, arc_tuple_original[1], arc_tuple_original[1]),
        get(reverse_bus_search_map, arc_tuple_original[2], arc_tuple_original[2]),
    )
end

function get_arc_tuple(br::PSY.ACTransmission, nr::NetworkReductionData)
    get_arc_tuple(PSY.get_arc(br), nr)
end

# Canonical orientation is the stored `arc_key` (set at construction), remapped through `nr`;
# anti-parallel members may exist post-merge, so do not rely on member order.
function get_arc_tuple(br::AbstractReductionAggregate, nr::NetworkReductionData)
    reverse_bus_search_map = get_reverse_bus_search_map(nr)
    return (
        get(reverse_bus_search_map, br.arc_key[1], br.arc_key[1]),
        get(reverse_bus_search_map, br.arc_key[2], br.arc_key[2]),
    )
end

function get_arc_tuple(br::AbstractReductionAggregate)
    return br.arc_key
end

function get_arc_tuple(br::PSY.ACTransmission)
    return get_arc_tuple(PSY.get_arc(br))
end

get_arc_tuple(arc::PSY.Arc) =
    (PSY.get_number(PSY.get_from(arc)), PSY.get_number(PSY.get_to(arc)))

# Available shunt components whose bus survived the reduction (a shunt on an eliminated bus
# is already folded into its parent bus's diagonal).
function _get_retained_shunts(
    ::Type{T},
    sys::PSY.System,
    reverse_bus_search_map::Dict{Int, Int},
) where {T <: PSY.StaticInjection}
    collection = Vector{T}()
    for sa in PSY.get_components(PSY.get_available, T, sys)
        if !haskey(reverse_bus_search_map, PSY.get_number(PSY.get_bus(sa)))
            push!(collection, sa)
        end
    end
    return collection
end

get_switched_admittances(sys::PSY.System, reverse_bus_search_map::Dict{Int, Int}) =
    _get_retained_shunts(PSY.SwitchedAdmittance, sys, reverse_bus_search_map)

get_fixed_admittances(sys::PSY.System, reverse_bus_search_map::Dict{Int, Int}) =
    _get_retained_shunts(PSY.FixedAdmittance, sys, reverse_bus_search_map)

function _add_branch_to_lookup!(
    branch_lookup::Dict{String, Int},
    ::Dict{String, Vector{String}},
    branch_type::Vector{DataType},
    branch::PSY.ACTransmission,
    branch_number::Int,
)
    branch_lookup[PSY.get_name(branch)] = branch_number
    push!(branch_type, typeof(branch))
    return
end

function _add_branch_to_lookup!(
    branch_lookup::Dict{String, Int},
    transformer_3w_lookup::Dict{String, Vector{String}},
    branch_type::Vector{DataType},
    branch::PSY.ThreeWindingTransformer,
    branch_number::Int,
)
    tr3w_name = PSY.get_name(branch)
    transformer_3w_lookup[tr3w_name] = Vector{String}(undef, 3)
    for (i, side) in enumerate(["primary", "secondary", "tertiary"])
        side_name = "$(tr3w_name)__$side"
        branch_lookup[side_name] = branch_number - 3 + i
        transformer_3w_lookup[tr3w_name][i] = side_name
        push!(branch_type, typeof(branch))
    end
    return
end

"""
Gets the indices  of the reference (slack) buses.
NOTE:
- the indices  corresponds to the columns of zeros belonging to the PTDF matrix.
- BA and ABA matrix miss the columns related to the reference buses.
"""
function find_slack_positions(buses)
    return find_slack_positions(buses, make_ax_ref(buses))
end

function find_slack_positions(buses, bus_lookup::Dict{Int, Int})::Set{Int}
    slack_position = sort([
        bus_lookup[PSY.get_number(n)] for
        n in buses if PSY.get_bustype(n) == ACBusTypes.REF
    ])
    if length(slack_position) == 0
        error("Slack bus not identified in the Bus/buses list, can't build NetworkMatrix")
    end
    return Set{Int}(slack_position)
end

"""
Validates that the user bus input is consistent with the ybus axes and the prior reductions.
Is used to check `irreducible_buses` for `Radial` and `DegreeTwo` reductions and `study_buses` for `WardReduction`.
"""
function validate_buses(A::PowerNetworkMatrix, buses::Set{Int})
    reverse_bus_search_map = get_network_reduction_data(A).reverse_bus_search_map
    for bus_no in buses
        reduced_bus_no = get(reverse_bus_search_map, bus_no, bus_no)
        if reduced_bus_no ∉ get_bus_axis(A)
            if bus_no == reduced_bus_no
                error(
                    "Invalid bus entry found: Bus $bus_no. Check your input data; this bus was not found in the admittance matrix.",
                )
            else
                error(
                    "Invalid bus entry found: Bus $bus_no. Check your input data; this bus was mapped to bus $reduced_bus_no in a prior reductions and not found in the admittance matrix.",
                )
            end
        end
    end
    return
end

"""
Convert the user input for irreducible_buses to a set of indices based on the Ybus lookup and the prior reductions.
"""
function get_irreducible_indices(A::AdjacencyMatrix, irreducible_buses::Vector{Int})
    reverse_bus_search_map = A.network_reduction_data.reverse_bus_search_map
    irreducible_indices = zeros(Int, length(irreducible_buses))
    for (ix, bus_no) in enumerate(irreducible_buses)
        reduced_bus_no = get(reverse_bus_search_map, bus_no, bus_no)
        irreducible_indices[ix] = A.lookup[1][reduced_bus_no]
    end
    return irreducible_indices
end

"""
Evaluates the ABA matrix given the System's Incidence matrix (A), BA matrix and
reference bus positions.

# Arguments
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence matrix.
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`
        BA matrix.

NOTE:
- evaluates A with "calculate_A_matrix", or extract A.data (if A::IncidenceMatrix)
- evaluates BA with "calculate_BA_matrix", or extract BA.data (if A::BA_Matrix)
"""
function calculate_ABA_matrix(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int},
)
    tmp = BA * A
    valid_ix = setdiff(1:size(tmp, 1), ref_bus_positions)
    return tmp[valid_ix, valid_ix]
end

"""
Return a sparse matrix given a dense one by dropping elements whose absolute
value is below a certain tolerance.

Uses optimized `droptol!` for better performance compared to element-wise iteration.

# Arguments
- `dense_array::Matrix{Float64}`:
        input matrix (e.g., PTDF matrix).
- `tol::Float64`:
        tolerance.
"""
function sparsify(dense_array::Matrix{Float64}, tol::Float64)
    sparse_array = SparseArrays.sparse(dense_array)
    SparseArrays.droptol!(sparse_array, tol)
    return sparse_array
end

"""
Return a sparse vector given a dense one by dropping elements whose absolute
value is at or below a certain tolerance (keeps `abs(x) > tol`).

# Arguments
- `dense_array::Vector{Float64}`:
        input vector (e.g., PTDF row from VirtualPTDF).
- `tol::Float64`:
        tolerance.
"""
function sparsify(dense_array::Vector{Float64}, tol::Float64)
    # Count-then-fill: allocate the SparseVector at exactly the survivor count
    # instead of materializing every nonzero (`sparsevec`) and then compacting
    # (`droptol!`). On the VirtualPTDF/LODF row path an almost-dense row collapses
    # to a sparse one, so this avoids an O(n) over-allocation per cached row.
    nnz = 0
    @inbounds for v in dense_array
        abs(v) > tol && (nnz += 1)
    end
    nzind = Vector{Int}(undef, nnz)
    nzval = Vector{Float64}(undef, nnz)
    k = 0
    @inbounds for i in eachindex(dense_array)
        v = dense_array[i]
        if abs(v) > tol
            k += 1
            nzind[k] = i
            nzval[k] = v
        end
    end
    return SparseArrays.SparseVector(length(dense_array), nzind, nzval)
end

"""
    _get_equivalent_physical_branch_parameters(equivalent_ybus::AbstractMatrix{<:Complex})
    _get_equivalent_physical_branch_parameters(
        equivalent_ybus::AbstractMatrix{<:Complex},
        segment::AbstractReductionAggregate,
    )

Recover π-model parameters (PowerModels data format) from a 2x2 two-port representing the Ybus
contribution of an `AbstractBranchesParallel` (homogeneous or mixed) or a `BranchesSeries`.
Callers pass ComplexF64; the `YBUS_ELTYPE` (ComplexF32) `equivalent_ybus` cache field is for
Ybus assembly only.

A single π-model exists only when `abs(y_12) == abs(y_21)` — otherwise the recovered phase
shift has a real part, which no π-model can express. That happens exactly when a group mixes
phase-shift angles with impedance angles. Such a group is still exactly representable, as
*several* parallel π branches: see [`equivalent_partitions`](@ref) and
[`arc_equivalent_branches`](@ref). When a `segment` is provided its name is appended to the
error raised in the impossible case; that lookup runs only on the error path.

Note the recovered shift is `imag(log(y_21 / y_12)) / 2`, which for a lossless group is
`atan(Σbₘ sin αₘ / Σbₘ cos αₘ)` — a nonlinear function of the member angles, *not* the
susceptance-weighted average the DC model uses (that is only its small-angle limit). Never
write a test oracle here by reusing a DC α.
"""
function _phase_shift_error_message()
    return "Equivalent parameters for the series or parallel reduction of branches results \
in a real part of the phase shift angle. This can occur when a lossy phase-shifting \
circuit is in parallel with other branches. Such a group has no single π equivalent; use \
`arc_equivalent_branches` (or `equivalent_partitions`) for the exact multi-branch \
representation."
end

function _phase_shift_tap_shift(y_12::Complex, y_21::Complex)
    isapprox(y_12, y_21) && return 1.0, 0.0
    ratio = log(y_21 / y_12) / 2
    isapprox(0.0, real(ratio); atol = 1e-6) || error(_phase_shift_error_message())
    return 1.0, imag(ratio)
end

function _phase_shift_tap_shift(
    y_12::Complex,
    y_21::Complex,
    segment::AbstractReductionAggregate,
)
    isapprox(y_12, y_21) && return 1.0, 0.0
    ratio = log(y_21 / y_12) / 2
    isapprox(0.0, real(ratio); atol = 1e-6) ||
        error("$(_phase_shift_error_message()) Offending group: $(get_name(segment)).")
    return 1.0, imag(ratio)
end

function _build_equivalent_branch(equivalent_ybus::AbstractMatrix{<:Complex}, tap, shift)
    # Index explicitly: destructuring a 2x2 iterates COLUMN-major, so `a, b, c, d = M` binds
    # `b` to M[2,1] (= Y21), not M[1,2]. The recovery genuinely wants Y21 here — that is what
    # makes the returned shift carry the same sign as PSY's stored α — but spelling it out
    # keeps the next reader from "fixing" the order.
    y_11 = equivalent_ybus[1, 1]
    y_21 = equivalent_ybus[2, 1]
    y_22 = equivalent_ybus[2, 2]
    y_l = y_21 * -1 * exp(1 * shift * im)
    z_12 = 1 / y_l
    r = real(z_12)
    x = imag(z_12)
    g_from = real(y_11 - y_l)
    b_from = imag(y_11 - y_l)
    g_to = real(y_22 - y_l)
    b_to = imag(y_22 - y_l)
    return EquivalentBranch(r, x, g_from, b_from, g_to, b_to, tap, shift)
end

function _get_equivalent_physical_branch_parameters(
    equivalent_ybus::AbstractMatrix{<:Complex},
)
    tap, shift = _phase_shift_tap_shift(equivalent_ybus[2, 1], equivalent_ybus[1, 2])
    return _build_equivalent_branch(equivalent_ybus, tap, shift)
end

function _get_equivalent_physical_branch_parameters(
    equivalent_ybus::AbstractMatrix{<:Complex},
    segment::AbstractReductionAggregate,
)
    # Column-major again: the first off-diagonal read is M[2,1]. See `_build_equivalent_branch`.
    tap, shift = _phase_shift_tap_shift(
        equivalent_ybus[2, 1],
        equivalent_ybus[1, 2],
        segment,
    )
    return _build_equivalent_branch(equivalent_ybus, tap, shift)
end

# `get_equivalent_physical_branch_parameters` / `populate_equivalent_ybus!` live here (rather
# than in BranchesParallel.jl / BranchesSeries.jl) because they take a `NetworkReductionData`,
# which is defined in a file included after those two.
# One method for both aggregate types — `ybus_branch_entries` already dispatches. The assignment
# is the single *declared* ComplexF32 -> ComplexF64 conversion point in this path: series chains
# arrive at Ybus storage precision, parallel groups at ComplexF64. See `CACHED_TWO_PORT`.
function populate_equivalent_ybus!(
    segment::AbstractReductionAggregate,
    nr::NetworkReductionData,
)
    segment.equivalent_ybus = ybus_branch_entries(segment, nr)
    segment.equivalent_ybus_populated = true
    return
end

# Reads the cached two-port, populating it on first use. The cache is ComplexF64 (see
# `CACHED_TWO_PORT`): the old ComplexF32 field cost ~7e-8 relative on the recovered shift and
# left the `real(ratio)` representability test only ~8 Float32 eps wide.
function get_equivalent_physical_branch_parameters(
    segment::AbstractReductionAggregate,
    nr::NetworkReductionData,
)
    if !segment.equivalent_ybus_populated
        populate_equivalent_ybus!(segment, nr)
    end
    Y11, Y12, Y21, Y22 = segment.equivalent_ybus
    return _get_equivalent_physical_branch_parameters(
        ComplexF64[Y11 Y12; Y21 Y22],
        segment,
    )
end

# α of one parallel-group member expressed in the group's arc frame: anti-parallel members
# (post-ZIR merge) enter with a negated angle, mirroring ybus_branch_entries(bp, nr).
function _oriented_member_phase_shift(
    br::PSY.ACTransmission,
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
)
    α = get_series_phase_shift(br)
    if get_arc_tuple(br, nr) != get_arc_tuple(bp, nr)
        return -α
    end
    return α
end

"""
    get_series_phase_shift(bp::AbstractBranchesParallel, nr) -> Float64

Susceptance-weighted equivalent DC phase shift of a parallel group,
`Σ bₘ·αₘ / Σ bₘ`, in the group's arc frame. Exact for the DC model even when members are
lossy (unlike the single-π extraction in `get_equivalent_physical_branch_parameters`).
"""
function get_series_phase_shift(bp::AbstractBranchesParallel, nr::NetworkReductionData)
    b_total = 0.0
    b_alpha = 0.0
    for br in bp
        b = get_series_susceptance(br, PSY.SU)
        b_total += b
        b_alpha += b * _oriented_member_phase_shift(br, bp, nr)
    end
    return b_alpha / b_total
end

# Single traversal of the reduction maps, shared by every arc-keyed accessor below. Probe order
# is load-bearing: a direct branch must win over a composite arc on the same key. `reversed` is
# true only for a group found under the opposite orientation — the direct and added maps are
# probed forward only, matching the pre-consolidation behavior pinned in
# `test_arc_resolution_characterization.jl`.
function _resolve_arc_entry(nr::NetworkReductionData, arc::Tuple{Int, Int})
    direct = get(get_direct_branch_map(nr), arc, nothing)
    isnothing(direct) || return (direct, false)
    rev = (arc[2], arc[1])
    for map in (get_series_branch_map(nr), get_parallel_branch_map(nr))
        forward = get(map, arc, nothing)
        isnothing(forward) || return (forward, false)
        reversed = get(map, rev, nothing)
        isnothing(reversed) || return (reversed, true)
    end
    added = get(get_added_arc_impedance_map(nr), arc, nothing)
    isnothing(added) || return (added, false)
    return error("Arc $(arc) not found in any network reduction map.")
end

_segment_phase_shift(seg::PSY.ACTransmission, ::NetworkReductionData) =
    get_series_phase_shift(seg)
# An aggregate would otherwise match the blanket method and call the single-branch
# `get_series_phase_shift`, which has no method for it.
_segment_phase_shift(seg::AbstractReductionAggregate, nr::NetworkReductionData) =
    get_series_phase_shift(seg, nr)

"""
    get_series_phase_shift(bs::BranchesSeries, nr) -> Float64

Equivalent DC phase shift of a series chain: segment angles add along the chain
(`:ToFrom` segments negated). Exact for the DC model.
"""
function get_series_phase_shift(bs::BranchesSeries, nr::NetworkReductionData)
    total = 0.0
    for (ix, seg) in enumerate(bs)
        α = _segment_phase_shift(seg, nr)
        if bs.segment_orientations[ix] == :ToFrom
            α = -α
        end
        total += α
    end
    return total
end

"""
    arc_dc_phase_shift(nr::NetworkReductionData, arc::Tuple{Int, Int}) -> Float64

Equivalent DC phase-shift angle α of the retained `arc`, resolved through whichever
reduction map owns it and oriented to match `arc` (a reverse-keyed hit negates). Total on
every mapped arc -- added Ward-equivalent arcs shift by 0.0. Throws if `arc` is in no map.
This is the α that `BA_Matrix` deliberately excludes from its susceptances
(`_arc_component_susceptance`); the DC solver applies it as the injection
[`arc_dc_shift_injection`](@ref).
"""
function arc_dc_phase_shift(nr::NetworkReductionData, arc::Tuple{Int, Int})
    entry, reversed = _resolve_arc_entry(nr, arc)
    # An added Ward arc is a `GenericArcImpedance`, so the blanket
    # `get_series_phase_shift(::PSY.ACTransmission) = 0.0` already reports it as unshifted.
    α = _segment_phase_shift(entry, nr)
    if reversed
        return -α
    end
    return α
end

"""
    arc_dc_shift_injection(nr::NetworkReductionData, arc::Tuple{Int, Int}) -> Float64

`b_eq·α_eq` for the retained `arc` in system base -- the magnitude of the DC phase-shift
injection pair (`+b·α` at the from bus, `−b·α` at the to bus) and of the arc-flow offset
(`f = b·Δθ − b·α`). Zero for every non-shifted arc. `b_eq` matches `BA_Matrix`'s value on
every shifted arc (both use `get_series_susceptance` there).
"""
function arc_dc_shift_injection(nr::NetworkReductionData, arc::Tuple{Int, Int})
    α = arc_dc_phase_shift(nr, arc)
    iszero(α) && return 0.0
    return _arc_dc_susceptance(nr, arc) * α
end

# b of the map entry owning `arc` (orientation-symmetric, so no reverse negation). Only
# reached for shifted arcs, which are always direct/parallel/series -- never Ward-added.
# Susceptance is orientation-symmetric, so a reverse hit needs no sign change. An added Ward arc
# is rejected rather than answered: it carries no series element the DC shift injection can use,
# and this is only ever reached for a shifted arc, which is never Ward-added.
_dc_entry_susceptance(br::PSY.ACTransmission) = get_series_susceptance(br, PSY.SU)
_dc_entry_susceptance(br::PSY.GenericArcImpedance) = error(
    "Arc backed by added Ward-equivalent impedance $(get_name(br)) has no series " *
    "susceptance for the DC phase-shift injection.",
)

function _arc_dc_susceptance(nr::NetworkReductionData, arc::Tuple{Int, Int})
    entry, _ = _resolve_arc_entry(nr, arc)
    return _dc_entry_susceptance(entry)
end

"""
    compute_parallel_circulating_flow(bp, nr, branch) -> Float64

DC circulating-flow component of one member of a parallel group, `bₘ·(α_eq − αₘ)` in the
group's arc frame (per unit, system base). The member's total DC flow is
`compute_parallel_multiplier(bp, branch)·f_arc + compute_parallel_circulating_flow(bp, nr, branch)`;
the circulating components sum to zero over the group. Member resolved by object identity;
a non-member is an error.
"""
function compute_parallel_circulating_flow(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    branch::PSY.ACTransmission,
)
    if !any(br === branch for br in bp)
        error(
            "Branch $(get_name(branch)) is not a member of parallel group " *
            "$(get_name(bp)).",
        )
    end
    α_eq = get_series_phase_shift(bp, nr)
    b = get_series_susceptance(branch, PSY.SU)
    return b * (α_eq - _oriented_member_phase_shift(branch, bp, nr))
end

# Combined series impedance of a reduction aggregate from member r/x alone (no shunts, no
# tap, no α) -- the loss-estimate equivalent that exists even when a lossy shifted group has
# no single-π representation. Orientation-symmetric.
function _dc_series_impedance(br::PSY.ACTransmission)
    return complex(PSY.get_r(br, PSY.SU), PSY.get_x(br, PSY.SU))
end

function _dc_series_impedance(t::PSY.TwoWindingTransformer)
    return _dc_series_impedance(PSY.get_circuit(t))
end

function _dc_series_impedance(c::PSY.TransformerCircuit)
    return complex(PSY.get_r(c, PSY.SU), PSY.get_x(c, PSY.SU))
end

function _dc_series_impedance(tw::ThreeWindingTransformerCircuit)
    return _dc_series_impedance(tw.circuit)
end

function _dc_series_impedance(bp::AbstractBranchesParallel)
    return inv(sum(inv(_dc_series_impedance(br)) for br in bp))
end

function _dc_series_impedance(bs::BranchesSeries)
    return sum(_dc_series_impedance(seg) for seg in bs)
end

# Members whose series impedances share a phase angle have `|Y12| == |Y21|` no matter how
# their alphas differ, because `phase(yₘ/tₘ) == -phase(zₘ)` for a real tap: the two sums
# `Σ(yₘ/tₘ)e^{±jαₘ}` become complex conjugates of each other times a common phase factor.
# Bucketing on this angle is therefore what makes each partition exactly single-π
# representable. Tolerance is loose enough to bucket members whose r/x agree to 9 digits.
const PARTITION_ANGLE_ATOL = 1e-9

function _member_impedance_angle(br)
    z = _dc_series_impedance(br)
    if iszero(z)
        # An r == x == 0 member gets a substitute reactance during Ybus assembly, so bucket it
        # as purely reactive rather than letting `angle(0) == 0` group it with resistive members.
        return π / 2
    end
    return angle(z)
end

function _partition_members_by_impedance_angle(bp::AbstractBranchesParallel)
    buckets = Vector{PSY.ACTransmission}[]
    angles = Float64[]
    for br in bp
        θ = _member_impedance_angle(br)
        ix = findfirst(a -> isapprox(a, θ; atol = PARTITION_ANGLE_ATOL), angles)
        if isnothing(ix)
            push!(angles, θ)
            push!(buckets, PSY.ACTransmission[br])
        else
            push!(buckets[ix], br)
        end
    end
    return buckets
end

# A single π-model for a two-port requires |Y12| == |Y21|; anything else needs a phase shift
# with a real part, which no π-model can express.
function _is_single_pi_representable(Y12::Complex, Y21::Complex)
    return isapprox(abs(Y12), abs(Y21); rtol = 1e-9, atol = 1e-12)
end

# Accumulated two-port of an arbitrary member subset, expressed in `reference`'s frame. A bus
# merge can fold an anti-parallel branch into a group, so a member keyed the other way has its
# 2x2 swapped. Members are resolved with `nr` because an aggregate member (a series chain)
# needs it to build its own two-port. This is the single orientation-handling loop;
# `ybus_branch_entries(bp, nr)` delegates here so the convention lives in exactly one place.
function _subset_two_port(
    members,
    reference::Tuple{Int, Int},
    nr::NetworkReductionData,
)
    Y11 = Y12 = Y21 = Y22 = zero(ComplexF64)
    for br in members
        (y11, y12, y21, y22) = ybus_branch_entries(br, nr)
        if get_arc_tuple(br, nr) != reference
            Y11 += y22
            Y12 += y21
            Y21 += y12
            Y22 += y11
        else
            Y11 += y11
            Y12 += y12
            Y21 += y21
            Y22 += y22
        end
    end
    return (Y11, Y12, Y21, Y22)
end

"""
    has_single_pi_equivalent(bp::AbstractBranchesParallel, nr) -> Bool

Whether the whole group collapses to one π-model. False exactly when the group mixes
phase-shift angles *and* impedance angles, which makes `|Y12| != |Y21|` — a phase shift with a
real part, which no π-model can express. Use
[`equivalent_partitions`](@ref) for the total representation.
"""
function has_single_pi_equivalent(bp::AbstractBranchesParallel, nr::NetworkReductionData)
    (_, Y12, Y21, _) = ybus_branch_entries(bp, nr)
    return _is_single_pi_representable(Y12, Y21)
end

"""
    equivalent_partitions(bp::AbstractBranchesParallel, nr) -> Vector{ParallelEquivalent}

Exact representation of a parallel group as one or more parallel π branches, in the group's arc
frame. Returns a single element whenever the group is single-π representable (the common case,
including a lossless phase shifter beside a lossless line, and any group with uniform α);
otherwise one element per impedance-angle partition. The partitions' π-models sum back to
`ybus_branch_entries(bp, nr)` exactly, which is the invariant the tests pin.

This is the total counterpart of `get_equivalent_physical_branch_parameters`, which can only
return a single π and therefore throws on lossy shifted groups (PNM issue #231).
"""
function equivalent_partitions(bp::AbstractBranchesParallel, nr::NetworkReductionData)
    reference = get_arc_tuple(bp, nr)
    Y11, Y12, Y21, Y22 = _subset_two_port(bp, reference, nr)
    # Whole-group representability is checked BEFORE partitioning: a group with uniform α but
    # mixed impedance angles has several angle buckets yet still collapses to one π.
    if _is_single_pi_representable(Y12, Y21)
        member_groups = [Vector{PSY.ACTransmission}(bp.branches)]
    else
        member_groups = _partition_members_by_impedance_angle(bp)
    end
    return [
        ParallelEquivalent(members, _partition_equivalent(members, reference, nr, bp))
        for members in member_groups
    ]
end

# Bucketing by impedance angle guarantees representability in exact arithmetic, but the two
# predicates carry different tolerances, so re-assert per bucket rather than emit a wrong π.
function _partition_equivalent(
    members::Vector{PSY.ACTransmission},
    reference::Tuple{Int, Int},
    nr::NetworkReductionData,
    bp::AbstractBranchesParallel,
)
    Y11, Y12, Y21, Y22 = _subset_two_port(members, reference, nr)
    _is_single_pi_representable(Y12, Y21) || error(
        "Partitioning group $(get_name(bp)) by impedance angle left a slice with no \
single-π equivalent (|Y12| = $(abs(Y12)), |Y21| = $(abs(Y21))). Members: \
$(join(get_name.(members), ", ")).",
    )
    return _get_equivalent_physical_branch_parameters(ComplexF64[Y11 Y12; Y21 Y22], bp)
end

"""
    get_partition_rating(pe::ParallelEquivalent) -> Union{Nothing, Float64}

Summed rating of the partition's members, or `nothing` when any member's rating is unknown —
matching `get_equivalent_rating(::AbstractBranchesParallel)` semantics for a subset.
"""
function get_partition_rating(pe::ParallelEquivalent)
    return _aggregate_known_ratings(sum, get_equivalent_rating, get_members(pe))
end

_segment_has_single_pi(::PSY.ACTransmission, ::NetworkReductionData) = true
# An aggregate would otherwise match the blanket method and answer `true` unconditionally.
_segment_has_single_pi(seg::AbstractReductionAggregate, nr::NetworkReductionData) =
    has_single_pi_equivalent(seg, nr)

"""
    has_single_pi_equivalent(bs::BranchesSeries, nr) -> Bool

Whether the chain collapses to one π-model. A cascade is representable exactly when every
segment is: the segments' asymmetry ratios multiply, so one non-representable segment makes the
whole chain non-representable. Unlike a parallel group, a chain has no multi-branch remedy —
parallel branches cannot express a cascade.
"""
function has_single_pi_equivalent(bs::BranchesSeries, nr::NetworkReductionData)
    for seg in bs
        _segment_has_single_pi(seg, nr) || return false
    end
    return true
end

_arc_equivalents(br::PSY.ACTransmission, ::NetworkReductionData) = [equivalent_branch(br)]

_arc_equivalents(bp::AbstractBranchesParallel, nr::NetworkReductionData) =
    get_equivalent.(equivalent_partitions(bp, nr))

function _arc_equivalents(bs::BranchesSeries, nr::NetworkReductionData)
    for seg in bs
        _segment_has_single_pi(seg, nr) && continue
        error(
            "Series chain $(get_name(bs)) contains segment $(get_name(seg)), a parallel \
group with no single-π equivalent (it mixes phase-shift angles with impedance angles). A \
cascade cannot be split into parallel branches, so this chain has no π representation at \
all. Either exclude that group's buses from the degree-two reduction so the intermediate \
bus is retained, or consume this arc through the DC API (`arc_dc_phase_shift`, \
`arc_dc_resistance`), which is exact for it.",
        )
    end
    return [get_equivalent_physical_branch_parameters(bs, nr)]
end

"""
    arc_equivalent_branches(nr::NetworkReductionData, arc::Tuple{Int, Int}) -> Vector{EquivalentBranch}

Every π branch needed to represent the retained `arc` exactly, oriented `from -> to` to match
`arc`. One element for a direct branch, a Ward-added impedance, a single-π-representable group,
or a representable chain; more than one only for a parallel group that mixes phase-shift angles
with impedance angles (PNM issue #231). Prefer this over [`arc_equivalent_branch`](@ref) in any
consumer that can emit several branches between one bus pair — PowerModels keys branches by
index, so it can.

Throws if `arc` is in no reduction map, or if the arc is a series chain containing a
non-representable parallel segment (which has no π representation in any count).
"""
function arc_equivalent_branches(nr::NetworkReductionData, arc::Tuple{Int, Int})
    entry, reversed = _resolve_arc_entry(nr, arc)
    equivalents = _arc_equivalents(entry, nr)
    if reversed
        return _reverse_equivalent_branch.(equivalents)
    end
    return equivalents
end

# Non-shifting aggregates keep the exact single-π value (bit-identical to
# arc_equivalent_branch); shifting aggregates take the member-impedance combination, which
# is total where the single-π extraction throws (lossy shifted groups).
function _dc_equivalent_resistance(
    group::AbstractReductionAggregate,
    nr::NetworkReductionData,
)
    if !_is_phase_shifting(group)
        return get_equivalent_r(get_equivalent_physical_branch_parameters(group, nr))
    end
    return real(_dc_series_impedance(group))
end

"""
    arc_dc_resistance(nr::NetworkReductionData, arc::Tuple{Int, Int}) -> Float64

Equivalent series resistance of the retained `arc` for DC loss estimation (`r·P²`),
system base. Total on every mapped arc -- including lossy shifted parallel groups, where
[`arc_equivalent_branch`](@ref) has no single-π equivalent and throws.
"""
# Resistance is orientation-symmetric. Single branches (direct and added-Ward alike) take their
# own equivalent; aggregates go through the shifted-group-aware combination.
_dc_entry_resistance(br::PSY.ACTransmission, ::NetworkReductionData) =
    get_equivalent_r(equivalent_branch(br))
_dc_entry_resistance(group::AbstractReductionAggregate, nr::NetworkReductionData) =
    _dc_equivalent_resistance(group, nr)

"""
    get_equivalent_available(seg::AbstractReductionAggregate) -> Bool

Availability of a reduction aggregate: every member must be available for the equivalent arc to
be. Parallel groups and series chains share the rule, so both dispatch here.
"""
function get_equivalent_available(seg::AbstractReductionAggregate)
    return all(PSY.get_available(branch) for branch in seg)
end

PSY.get_available(seg::AbstractReductionAggregate) = get_equivalent_available(seg)

function arc_dc_resistance(nr::NetworkReductionData, arc::Tuple{Int, Int})
    entry, _ = _resolve_arc_entry(nr, arc)
    return _dc_entry_resistance(entry, nr)
end

# Recurses through PNM's own `has_time_series`, not PSY's, so a member that is itself a PNM
# wrapper (a nested group, or a `ThreeWindingTransformerCircuit`, which PSY cannot answer for)
# resolves correctly.
function has_time_series(
    branch::AbstractReductionAggregate,
    ts_type::Type{T},
    ts_name::String,
) where {T <: PSY.TimeSeriesData}
    return any(b -> has_time_series(b, ts_type, ts_name), branch)
end

function has_time_series(
    branch::PSY.ACTransmission,
    ts_type::Type{T},
    ts_name::String,
) where {T <: PSY.TimeSeriesData}
    if PSY.has_time_series(branch, ts_type, ts_name)
        return true
    end
    return false
end

function get_device_with_time_series(
    branch::BranchesSeries,
    ts_type::Type{T},
    ts_name::String,
) where {T <: PSY.TimeSeriesData}
    for b in branch
        if has_time_series(b, ts_type, ts_name)
            return get_device_with_time_series(b, ts_type, ts_name)
        end
    end
    return nothing
end

function get_device_with_time_series(
    branch::AbstractBranchesParallel,
    ts_type::Type{T},
    ts_name::String,
) where {T <: PSY.TimeSeriesData}
    for b in branch
        if has_time_series(b, ts_type, ts_name)
            return b
        end
    end
    return nothing
end

function get_device_with_time_series(
    branch::PSY.ACTransmission,
    ts_type::Type{T},
    ts_name::String,
) where {T <: PSY.TimeSeriesData}
    if has_time_series(branch, ts_type, ts_name)
        return branch
    end
    return nothing
end

"""
    _resolve_branch_arc(nr::NetworkReductionData, component::PSY.ACTransmission)
        -> Tuple{Symbol, Union{Tuple{Int, Int}, Nothing}}

Classify a branch component by looking up which reverse map it belongs to in the
`NetworkReductionData`. Returns `(tag, arc_tuple)` where `tag` is one of:
- `:direct`       -- branch is the sole branch on its arc (a `ThreeWindingTransformerCircuit` is one such branch, on its star-point arc)
- `:parallel`     -- branch is one of several parallel branches on its arc
- `:series`       -- branch is part of a series chain on its arc
- `:not_found`    -- branch is not in any map (e.g., eliminated by radial reduction)

The second element is the arc tuple `(from_bus, to_bus)`, or `nothing` when `:not_found`.
"""
function _resolve_branch_arc(
    nr::NetworkReductionData,
    component::PSY.ACTransmission,
)::Tuple{Symbol, Union{Tuple{Int, Int}, Nothing}}
    if haskey(nr.reverse_direct_branch_map, component)
        return (:direct, nr.reverse_direct_branch_map[component])
    elseif haskey(nr.reverse_parallel_branch_map, component)
        return (:parallel, nr.reverse_parallel_branch_map[component])
    elseif haskey(nr.reverse_series_branch_map, component)
        return (:series, nr.reverse_series_branch_map[component])
    else
        return (:not_found, nothing)
    end
end

"""
    _assert_not_phase_shifting(component::PSY.ACTransmission)

Throws `ErrorException` when `component` is phase shifting; no-op otherwise. Phase shifting
is a per-circuit data property surfaced by `_is_phase_shifting`, not a distinct type.
"""
function _assert_not_phase_shifting(component::PSY.ACTransmission)
    if _is_phase_shifting(component)
        error(
            "Contingencies on phase-shifting transformers are not supported. " *
            "Component: $(get_name(component)).",
        )
    end
    return nothing
end

"""
    _segment_susceptance_after_outage(segment, tripped_set) -> Float64

Compute the remaining susceptance of a series chain segment after removing
tripped components. Dispatches on segment type to handle both single branches
and parallel groups within a series chain.

Returns 0.0 if the segment (or all branches in a parallel group) is fully tripped.
"""
function _segment_susceptance_after_outage(
    segment::PSY.ACTransmission,
    tripped_set::Set{<:PSY.ACTransmission},
)::Float64
    if segment ∈ tripped_set
        return 0.0
    end
    return get_series_susceptance(segment, PSY.SU)
end

function _segment_susceptance_after_outage(
    segment::AbstractBranchesParallel,
    tripped_set::Set{<:PSY.ACTransmission},
)::Float64
    b_remaining = 0.0
    for branch in segment.branches
        if branch ∉ tripped_set
            b_remaining += get_series_susceptance(branch, PSY.SU)
        end
    end
    return b_remaining
end

"""
    _compute_series_outage_delta_b(series_chain::BranchesSeries, component::PSY.ACTransmission) -> Float64

Compute the change in equivalent arc susceptance when `component` is tripped
from `series_chain`. Delegates to the vector version.
"""
function _compute_series_outage_delta_b(
    series_chain::BranchesSeries,
    component::PSY.ACTransmission,
)::Float64
    return _compute_series_outage_delta_b(series_chain, [component])
end

"""
    _compute_series_outage_delta_b(series_chain::BranchesSeries, tripped::Vector{<:PSY.ACTransmission}) -> Float64

Compute the change in equivalent arc susceptance when multiple components are
simultaneously tripped from a series chain.

For a series chain with segments of susceptance b₁, b₂, ..., bₙ, the equivalent
susceptance is: b_eq = 1 / (1/b₁ + 1/b₂ + ... + 1/bₙ).

Segments can be individual branches or `BranchesParallel` groups. When a tripped
component is inside a parallel group, only that branch's susceptance is removed
from the group — the rest of the parallel group remains in the series chain.

Returns Δb = b_new - b_old (always negative for outages).
If all segments are fully tripped, returns -b_eq (full arc outage).
"""
function _compute_series_outage_delta_b(
    series_chain::BranchesSeries,
    tripped::Vector{<:PSY.ACTransmission},
)::Float64
    b_old = get_series_susceptance(series_chain, PSY.SU)
    tripped_set = Set{PSY.ACTransmission}(tripped)
    remaining_inv_sum = 0.0
    for segment in series_chain
        b_seg = _segment_susceptance_after_outage(segment, tripped_set)
        if iszero(b_seg)
            return -b_old
        end
        remaining_inv_sum += 1.0 / b_seg
    end
    b_new = 1.0 / remaining_inv_sum
    return b_new - b_old
end
