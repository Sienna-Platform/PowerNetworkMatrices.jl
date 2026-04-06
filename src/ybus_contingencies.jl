"""
    YbusModification

A lightweight representation of a sparse modification (delta) to a Ybus matrix.

Used for contingency analysis, this struct captures the exact set of changes
to apply to a base Ybus matrix when one or more components are outaged or
have their impedance changed. The delta is stored as a sparse matrix with
the same dimensions and indexing as the base Ybus.

The design follows the same component classification pattern as VirtualMODF's
`ArcModification`/`ContingencySpec`, but stores the full complex Pi-model
delta (4 entries per arc) as a sparse matrix instead of scalar susceptance changes.

# Fields
$(TYPEDFIELDS)
"""
struct YbusModification
    "Sparse delta matrix (ΔY), same dimensions and indexing as the base Ybus"
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int}
    "Names of the modified components"
    component_names::Vector{String}
    "Types of the modified components"
    component_types::Vector{DataType}
    "Whether this modification islands the network (disconnects it into additional components)"
    is_islanding::Bool
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, mod::YbusModification)
    n_components = length(mod.component_names)
    n_nz = SparseArrays.nnz(mod.data)
    island_str = mod.is_islanding ? ", ISLANDING" : ""
    print(
        io,
        "YbusModification: $n_components component(s), $n_nz nonzeros in ΔY$island_str",
    )
    return
end

# --- Accumulation helpers ---

function _accumulate_arc_delta!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    fb_ix::Int,
    tb_ix::Int,
    delta_y11::YBUS_ELTYPE,
    delta_y12::YBUS_ELTYPE,
    delta_y21::YBUS_ELTYPE,
    delta_y22::YBUS_ELTYPE,
)
    push!(I, fb_ix, fb_ix, tb_ix, tb_ix)
    push!(J, fb_ix, tb_ix, fb_ix, tb_ix)
    push!(V, delta_y11, delta_y12, delta_y21, delta_y22)
    return
end

function _accumulate_branch_outage!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    br::Union{PSY.ACTransmission, BranchesSeries, ThreeWindingTransformerWinding},
    fb_ix::Int,
    tb_ix::Int,
)
    Y11, Y12, Y21, Y22 = ybus_branch_entries(br)
    _accumulate_arc_delta!(
        I, J, V, fb_ix, tb_ix,
        YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
        YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22),
    )
    return
end

function _accumulate_shunt_outage!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    component::PSY.FixedAdmittance,
    bus_lookup::Dict{Int, Int},
    nr::NetworkReductionData,
)
    bus_ix = get_bus_index(component, bus_lookup, nr)
    Y = PSY.get_Y(component)
    push!(I, bus_ix)
    push!(J, bus_ix)
    push!(V, YBUS_ELTYPE(-Y))
    return
end

function _accumulate_shunt_outage!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    component::PSY.StandardLoad,
    bus_lookup::Dict{Int, Int},
    nr::NetworkReductionData,
)
    bus_ix = get_bus_index(component, bus_lookup, nr)
    Y =
        PSY.get_impedance_active_power(component) -
        im * PSY.get_impedance_reactive_power(component)
    push!(I, bus_ix)
    push!(J, bus_ix)
    push!(V, YBUS_ELTYPE(-Y))
    return
end

function _accumulate_shunt_outage!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    component::PSY.SwitchedAdmittance,
    bus_lookup::Dict{Int, Int},
    nr::NetworkReductionData,
)
    bus_ix = get_bus_index(component, bus_lookup, nr)
    Y = PSY.get_Y(component)
    push!(I, bus_ix)
    push!(J, bus_ix)
    push!(V, YBUS_ELTYPE(-Y))
    return
end

# --- Island detection ---

"""
Check whether severing the given arcs disconnects the network.
Uses a lightweight union-find on a modified copy of the adjacency matrix.
Returns `true` if the modified network has more connected components than the base.
"""
function _check_islanding(
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int},
    bus_lookup::Dict{Int, Int},
    severed_arcs::Set{Tuple{Int, Int}},
    base_num_components::Int,
)::Bool
    isempty(severed_arcs) && return false
    adj = copy(adjacency_data)
    for (fb, tb) in severed_arcs
        fb_ix = bus_lookup[fb]
        tb_ix = bus_lookup[tb]
        adj[fb_ix, tb_ix] = 0
        adj[tb_ix, fb_ix] = 0
    end
    SparseArrays.dropzeros!(adj)
    n = size(adj, 1)
    n <= 1 && return false
    rows = SparseArrays.rowvals(adj)
    uf = collect(1:n)
    for ix in 1:n
        for j in SparseArrays.nzrange(adj, ix)
            union_sets!(uf, ix, rows[j])
        end
    end
    for i in 1:n
        uf[i] = get_representative(uf, i)
    end
    num_components = length(unique(uf))
    return num_components > base_num_components
end

# --- Series chain helpers ---

function _is_segment_fully_tripped(
    segment::PSY.ACTransmission,
    tripped::Vector{<:PSY.ACTransmission},
)
    return segment ∈ tripped
end

function _is_segment_fully_tripped(
    segment::BranchesParallel,
    tripped::Vector{<:PSY.ACTransmission},
)
    return all(b ∈ tripped for b in segment.branches)
end

"""
Build a modified chain Ybus with tripped branches removed from their parallel groups.
Returns `nothing` if the chain is broken (any segment fully tripped).
"""
function _build_modified_chain_ybus(
    series_chain::BranchesSeries,
    tripped::Vector{<:PSY.ACTransmission},
)
    segment_orientations = series_chain.segment_orientations
    n_segments = length(series_chain)
    fb = Vector{Int}()
    tb = Vector{Int}()
    y11 = Vector{YBUS_ELTYPE}()
    y12 = Vector{YBUS_ELTYPE}()
    y21 = Vector{YBUS_ELTYPE}()
    y22 = Vector{YBUS_ELTYPE}()
    sizehint!(fb, n_segments)
    sizehint!(tb, n_segments)
    sizehint!(y11, n_segments)
    sizehint!(y12, n_segments)
    sizehint!(y21, n_segments)
    sizehint!(y22, n_segments)
    for (ix, segment) in enumerate(series_chain)
        if segment isa BranchesParallel
            if _is_segment_fully_tripped(segment, tripped)
                return nothing
            end
            remaining = [b for b in segment if b ∉ tripped]
            if length(remaining) == length(segment)
                add_segment_to_ybus!(
                    segment,
                    y11, y12, y21, y22, fb, tb,
                    ix, segment_orientations[ix],
                )
            elseif length(remaining) == 1
                add_segment_to_ybus!(
                    remaining[1],
                    y11, y12, y21, y22, fb, tb,
                    ix, segment_orientations[ix],
                )
            else
                modified_parallel = BranchesParallel(remaining)
                add_segment_to_ybus!(
                    modified_parallel,
                    y11, y12, y21, y22, fb, tb,
                    ix, segment_orientations[ix],
                )
            end
        else
            if _is_segment_fully_tripped(segment, tripped)
                return nothing
            end
            add_segment_to_ybus!(
                segment,
                y11, y12, y21, y22, fb, tb,
                ix, segment_orientations[ix],
            )
        end
    end
    return Matrix(
        SparseArrays.sparse(
            [fb; fb; tb; tb],
            [fb; tb; fb; tb],
            [y11; y12; y21; y22],
        ),
    )
end

"""
Compute the Y-bus delta for a series chain outage.
Analogous to VirtualMODF's `_compute_series_outage_delta_b` but returns
full Pi-model entries instead of scalar susceptance.
Returns a tuple of (delta_entries, chain_broken::Bool).
"""
function _compute_series_outage_delta_ybus(
    series_chain::BranchesSeries,
    tripped::Vector{<:PSY.ACTransmission},
)
    Y11_old, Y12_old, Y21_old, Y22_old = ybus_branch_entries(series_chain)

    modified_chain_ybus = _build_modified_chain_ybus(series_chain, tripped)

    if isnothing(modified_chain_ybus)
        return (
            YBUS_ELTYPE(-Y11_old),
            YBUS_ELTYPE(-Y12_old),
            YBUS_ELTYPE(-Y21_old),
            YBUS_ELTYPE(-Y22_old),
        ),
        true
    end

    reduced = _reduce_internal_nodes(modified_chain_ybus)
    return (
        YBUS_ELTYPE(reduced[1, 1] - Y11_old),
        YBUS_ELTYPE(reduced[1, 2] - Y12_old),
        YBUS_ELTYPE(reduced[2, 1] - Y21_old),
        YBUS_ELTYPE(reduced[2, 2] - Y22_old),
    ),
    false
end

# --- Name helper ---

_get_modification_name(c::PSY.Component) = PSY.get_name(c)
_get_modification_name(c::ThreeWindingTransformerWinding) = get_name(c)

# --- Component classification (mirrors VirtualMODF._classify_outage_component!) ---

function _classify_ybus_outage_3w_winding!(
    component::ThreeWindingTransformerWinding,
    nr::NetworkReductionData,
    bus_lookup::Dict{Int, Int},
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
)
    if haskey(nr.reverse_transformer3W_map, component)
        arc_tuple = nr.reverse_transformer3W_map[component]
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]
        _accumulate_branch_outage!(I, J, V, component, fb_ix, tb_ix)
    else
        @warn "ThreeWindingTransformerWinding $(get_name(component)) not found in " *
              "transformer3W map. Skipping."
    end
    return
end

function _classify_ybus_outage_component!(
    ::PSY.PhaseShiftingTransformer,
    ::NetworkReductionData,
    ::Dict{Int, Int},
    ::Vector{Int},
    ::Vector{Int},
    ::Vector{YBUS_ELTYPE},
    ::Dict{Tuple{Int, Int}, Vector{PSY.ACTransmission}},
)
    error(
        "Contingencies on PhaseShiftingTransformer are not supported.",
    )
end

"""
Classify a branch outage by its location in the NetworkReductionData maps.
Direct and parallel branches are accumulated immediately; series chain
components are deferred for grouped computation.
"""
function _classify_ybus_outage_component!(
    component::PSY.ACTransmission,
    nr::NetworkReductionData,
    bus_lookup::Dict{Int, Int},
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    series_components_by_arc::Dict{Tuple{Int, Int}, Vector{PSY.ACTransmission}},
)
    if haskey(nr.reverse_direct_branch_map, component)
        arc_tuple = nr.reverse_direct_branch_map[component]
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]
        _accumulate_branch_outage!(I, J, V, component, fb_ix, tb_ix)

    elseif haskey(nr.reverse_parallel_branch_map, component)
        # Individual branch contribution only, not the full parallel set
        arc_tuple = nr.reverse_parallel_branch_map[component]
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]
        _accumulate_branch_outage!(I, J, V, component, fb_ix, tb_ix)

    elseif haskey(nr.reverse_series_branch_map, component)
        arc_tuple = nr.reverse_series_branch_map[component]
        if !haskey(series_components_by_arc, arc_tuple)
            series_components_by_arc[arc_tuple] = PSY.ACTransmission[]
        end
        push!(series_components_by_arc[arc_tuple], component)

    else
        @warn "Branch $(PSY.get_name(component)) not found in any reduction map. " *
              "The component may have been eliminated by a radial reduction. " *
              "Skipping."
    end
    return
end

# --- Arc lookup helper ---

"""
Find the arc tuple for a branch in the network reduction maps.
Checks direct, parallel, and series maps in order.
"""
function _find_branch_arc(
    nr::NetworkReductionData,
    branch::PSY.ACTransmission,
)
    if haskey(nr.reverse_direct_branch_map, branch)
        return nr.reverse_direct_branch_map[branch]
    elseif haskey(nr.reverse_parallel_branch_map, branch)
        return nr.reverse_parallel_branch_map[branch]
    elseif haskey(nr.reverse_series_branch_map, branch)
        return nr.reverse_series_branch_map[branch]
    end
    error("Branch $(PSY.get_name(branch)) not found in any reduction map.")
end

# --- Constructors ---

"""
    $(TYPEDSIGNATURES)

Compute a lightweight ΔYbus from a set of outaged components.

Uses a two-pass approach mirroring VirtualMODF's `_register_outage!`:
1. Classify components (direct/parallel branches accumulated immediately,
   series chain components grouped by arc)
2. Compute series chain deltas with all tripped components per chain

# Arguments
- `ybus::Ybus`: Base Ybus matrix
- `components::Vector{<:PSY.Component}`: Components to outage

# See Also
- [`YbusModification(::Ybus, ::PSY.System, ::PSY.Contingency)`](@ref): Construct from contingency
- [`YbusModification(::Ybus, ::PSY.ACTransmission, ::PSY.ACTransmission)`](@ref): Construct from impedance change
"""
function YbusModification(
    ybus::Ybus,
    components::Vector{<:PSY.Component},
)
    bus_lookup = get_bus_lookup(ybus)
    nr = get_network_reduction_data(ybus)
    n = length(bus_lookup)

    component_names = Vector{String}(undef, length(components))
    component_types = Vector{DataType}(undef, length(components))

    n_components = length(components)
    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{YBUS_ELTYPE}()
    sizehint!(I, 4 * n_components)
    sizehint!(J, 4 * n_components)
    sizehint!(V, 4 * n_components)
    series_components_by_arc = Dict{Tuple{Int, Int}, Vector{PSY.ACTransmission}}()
    severed_arcs = Set{Tuple{Int, Int}}()
    parallel_tripped_by_arc = Dict{Tuple{Int, Int}, Int}()

    for (idx, component) in enumerate(components)
        component_names[idx] = _get_modification_name(component)
        component_types[idx] = typeof(component)
        if component isa ThreeWindingTransformerWinding
            _classify_ybus_outage_3w_winding!(
                component, nr, bus_lookup, I, J, V,
            )
            if haskey(nr.reverse_transformer3W_map, component)
                push!(severed_arcs, nr.reverse_transformer3W_map[component])
            end
        elseif component isa PSY.ACTransmission
            _classify_ybus_outage_component!(
                component, nr, bus_lookup, I, J, V, series_components_by_arc,
            )
            if haskey(nr.reverse_direct_branch_map, component)
                push!(severed_arcs, nr.reverse_direct_branch_map[component])
            elseif haskey(nr.reverse_parallel_branch_map, component)
                arc_tuple = nr.reverse_parallel_branch_map[component]
                parallel_tripped_by_arc[arc_tuple] =
                    get(parallel_tripped_by_arc, arc_tuple, 0) + 1
            end
        elseif component isa PSY.FixedAdmittance
            _accumulate_shunt_outage!(I, J, V, component, bus_lookup, nr)
        elseif component isa PSY.StandardLoad
            _accumulate_shunt_outage!(I, J, V, component, bus_lookup, nr)
        elseif component isa PSY.SwitchedAdmittance
            _accumulate_shunt_outage!(I, J, V, component, bus_lookup, nr)
        else
            @warn "Component $(_get_modification_name(component)) ($(typeof(component))) " *
                  "does not affect the Ybus. Skipping."
        end
    end

    for (arc_tuple, tripped_count) in parallel_tripped_by_arc
        total_branches = length(nr.parallel_branch_map[arc_tuple].branches)
        if tripped_count >= total_branches
            push!(severed_arcs, arc_tuple)
        end
    end

    for (arc_tuple, tripped) in series_components_by_arc
        series_chain = nr.series_branch_map[arc_tuple]
        (delta_y11, delta_y12, delta_y21, delta_y22), chain_broken =
            _compute_series_outage_delta_ybus(series_chain, tripped)
        if chain_broken
            push!(severed_arcs, arc_tuple)
        end
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]
        _accumulate_arc_delta!(
            I, J, V, fb_ix, tb_ix,
            delta_y11, delta_y12, delta_y21, delta_y22,
        )
    end

    data = SparseArrays.sparse(I, J, V, n, n)
    is_islanding = _check_islanding(
        ybus.adjacency_data,
        bus_lookup,
        severed_arcs,
        length(ybus.subnetwork_axes),
    )
    return YbusModification(data, component_names, component_types, is_islanding)
end

"""
    $(TYPEDSIGNATURES)

Compute a ΔYbus from a PowerSystems Contingency supplemental attribute.

Retrieves the components associated with the contingency and computes
the modification representing their removal from the network.

# Arguments
- `ybus::Ybus`: Base Ybus matrix
- `sys::PSY.System`: System containing the contingency associations
- `contingency::PSY.Contingency`: Contingency attribute to resolve

# See Also
- [`YbusModification(::Ybus, ::Vector{<:PSY.Component})`](@ref): Construct from components
"""
function YbusModification(
    ybus::Ybus,
    sys::PSY.System,
    contingency::PSY.Contingency,
)
    associated = collect(
        PSY.get_associated_components(
            sys,
            contingency;
            component_type = PSY.ACTransmission,
        ),
    )

    shunt_components = PSY.Component[]
    for fa in PSY.get_associated_components(
        sys,
        contingency;
        component_type = PSY.FixedAdmittance,
    )
        push!(shunt_components, fa)
    end
    for sl in PSY.get_associated_components(
        sys,
        contingency;
        component_type = PSY.StandardLoad,
    )
        push!(shunt_components, sl)
    end
    for sa in PSY.get_associated_components(
        sys,
        contingency;
        component_type = PSY.SwitchedAdmittance,
    )
        push!(shunt_components, sa)
    end

    all_components = PSY.Component[associated; shunt_components]

    if isempty(all_components)
        error(
            "Contingency has no associated components that affect the Ybus.",
        )
    end

    return YbusModification(ybus, all_components)
end

"""
    $(TYPEDSIGNATURES)

Compute a ΔYbus for a branch impedance change (ΔY = new_contribution - old_contribution).

Handles the three reduction map cases:
- Direct: delta from individual branch parameter change
- Parallel: delta from individual branch change within the parallel set
- Series: recomputes the series chain equivalent with the modified branch

# Arguments
- `ybus::Ybus`: Base Ybus matrix
- `old_branch::T`: Branch with original parameters
- `new_branch::T`: Branch with updated parameters

# See Also
- [`YbusModification(::Ybus, ::Vector{<:PSY.Component})`](@ref): Construct from outaged components
"""
function YbusModification(
    ybus::Ybus,
    old_branch::T,
    new_branch::T,
) where {T <: PSY.ACTransmission}
    bus_lookup = get_bus_lookup(ybus)
    nr = get_network_reduction_data(ybus)
    n = length(bus_lookup)

    I = Vector{Int}()
    J = Vector{Int}()
    V = Vector{YBUS_ELTYPE}()
    sizehint!(I, 4)
    sizehint!(J, 4)
    sizehint!(V, 4)

    if haskey(nr.reverse_series_branch_map, old_branch)
        arc_tuple = nr.reverse_series_branch_map[old_branch]
        series_chain = nr.series_branch_map[arc_tuple]
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]

        Y11_old, Y12_old, Y21_old, Y22_old = ybus_branch_entries(series_chain)
        modified_chain_ybus = _build_impedance_changed_chain_ybus(
            series_chain, old_branch, new_branch,
        )
        reduced = _reduce_internal_nodes(modified_chain_ybus)

        _accumulate_arc_delta!(
            I, J, V, fb_ix, tb_ix,
            YBUS_ELTYPE(reduced[1, 1] - Y11_old),
            YBUS_ELTYPE(reduced[1, 2] - Y12_old),
            YBUS_ELTYPE(reduced[2, 1] - Y21_old),
            YBUS_ELTYPE(reduced[2, 2] - Y22_old),
        )
    else
        # Direct and parallel branches use the same delta computation
        arc_tuple = _find_branch_arc(nr, old_branch)
        fb_ix = bus_lookup[arc_tuple[1]]
        tb_ix = bus_lookup[arc_tuple[2]]
        Y11_old, Y12_old, Y21_old, Y22_old = ybus_branch_entries(old_branch)
        Y11_new, Y12_new, Y21_new, Y22_new = ybus_branch_entries(new_branch)
        _accumulate_arc_delta!(
            I, J, V, fb_ix, tb_ix,
            YBUS_ELTYPE(Y11_new - Y11_old), YBUS_ELTYPE(Y12_new - Y12_old),
            YBUS_ELTYPE(Y21_new - Y21_old), YBUS_ELTYPE(Y22_new - Y22_old),
        )
    end

    data = SparseArrays.sparse(I, J, V, n, n)
    return YbusModification(
        data,
        [PSY.get_name(old_branch)],
        [typeof(old_branch)],
        false,
    )
end

"""
Build a Ybus for a series chain with one branch replaced by a new version (for impedance changes).
"""
function _build_impedance_changed_chain_ybus(
    series_chain::BranchesSeries,
    old_branch::PSY.ACTransmission,
    new_branch::PSY.ACTransmission,
)
    segment_orientations = series_chain.segment_orientations
    n_segments = length(series_chain)
    fb = Vector{Int}()
    tb = Vector{Int}()
    y11 = Vector{YBUS_ELTYPE}()
    y12 = Vector{YBUS_ELTYPE}()
    y21 = Vector{YBUS_ELTYPE}()
    y22 = Vector{YBUS_ELTYPE}()
    sizehint!(fb, n_segments)
    sizehint!(tb, n_segments)
    sizehint!(y11, n_segments)
    sizehint!(y12, n_segments)
    sizehint!(y21, n_segments)
    sizehint!(y22, n_segments)
    for (ix, segment) in enumerate(series_chain)
        if segment === old_branch
            add_segment_to_ybus!(
                new_branch,
                y11, y12, y21, y22, fb, tb,
                ix, segment_orientations[ix],
            )
        elseif segment isa BranchesParallel
            found_in_parallel = any(b === old_branch for b in segment)
            if found_in_parallel
                modified_branches = PSY.ACTransmission[
                    b === old_branch ? new_branch : b for b in segment
                ]
                modified_parallel = BranchesParallel(modified_branches)
                add_segment_to_ybus!(
                    modified_parallel,
                    y11, y12, y21, y22, fb, tb,
                    ix, segment_orientations[ix],
                )
            else
                add_segment_to_ybus!(
                    segment,
                    y11, y12, y21, y22, fb, tb,
                    ix, segment_orientations[ix],
                )
            end
        else
            add_segment_to_ybus!(
                segment,
                y11, y12, y21, y22, fb, tb,
                ix, segment_orientations[ix],
            )
        end
    end
    return Matrix(
        SparseArrays.sparse(
            [fb; fb; tb; tb],
            [fb; tb; fb; tb],
            [y11; y12; y21; y22],
        ),
    )
end

# --- Apply function ---

"""
    $(TYPEDSIGNATURES)

Return the modified Ybus sparse data matrix with the contingency applied.

This is the lightweight application path: a single sparse matrix addition.
The result shares the same indexing (`get_bus_lookup(ybus)`) as the original Ybus.

# Arguments
- `ybus::Ybus`: Base Ybus matrix
- `modification::YbusModification`: Delta to apply
"""
function apply_ybus_modification(
    ybus::Ybus,
    modification::YbusModification,
)
    return ybus.data + modification.data
end

# --- Combining modifications ---

"""
    $(TYPEDSIGNATURES)

Combine two `YbusModification`s by adding their sparse deltas.

Both modifications must have the same matrix dimensions and bus indexing.
This method validates only dimensional compatibility; callers are responsible
for ensuring both deltas use the same underlying bus lookup/indexing.

# TODO: Store the source system UUID in each matrix and validate consistency here.
# See https://github.com/NREL-Sienna/PowerNetworkMatrices.jl/pull/280 for discussion.
"""
function Base.:+(m1::YbusModification, m2::YbusModification)
    IS.@assert_op size(m1.data) == size(m2.data)
    return YbusModification(
        m1.data + m2.data,
        vcat(m1.component_names, m2.component_names),
        vcat(m1.component_types, m2.component_types),
        m1.is_islanding || m2.is_islanding,
    )
end
