"""
    topology_schedule_components(sys::PSY.System)

Collect what scheduled topology changes act on, from which components carry a
`PSY.PlannedOutage` or `PSY.SwitchingSchedule`; the schedules' time series are never read.

Returns a named tuple:
- `components`: every available component carrying either attribute, each once
- `closed_switches`: available switches carrying a `PSY.SwitchingSchedule`, to be included
  in the base network as closed (pass to `Ybus(sys; closed_switches)`)
- `pinned_buses`: endpoint buses of every scheduled AC branch, to keep out of reductions
  (pass to `Ybus(sys; schedule_pinned_buses)`)
"""
function topology_schedule_components(sys::PSY.System)
    seen = IdDict{PSY.Component, Nothing}()
    components = PSY.Component[]
    closed_switches = PSY.DiscreteControlledACBranch[]
    pinned_buses = Set{Int}()
    for attribute in PSY.get_supplemental_attributes(PSY.PlannedOutage, sys)
        _collect_scheduled!(seen, components, pinned_buses, sys, attribute)
    end
    for attribute in PSY.get_supplemental_attributes(PSY.SwitchingSchedule, sys)
        for switch in _collect_scheduled!(seen, components, pinned_buses, sys, attribute)
            _push_scheduled_switch!(closed_switches, switch)
        end
    end
    return (
        components = components,
        closed_switches = unique(closed_switches),
        pinned_buses = pinned_buses,
    )
end

# Returns the available components associated with `attribute`; records each new one.
function _collect_scheduled!(seen, components, pinned_buses, sys, attribute)
    associated = PSY.Component[]
    for component in PSY.get_associated_components(sys, attribute)
        PSY.get_available(component) || continue
        # Returned even when already seen: a switch carrying both a `PlannedOutage` and a
        # `SwitchingSchedule` is first seen through its outage and must still reach
        # `closed_switches`.
        push!(associated, component)
        haskey(seen, component) && continue
        seen[component] = nothing
        push!(components, component)
        _pin_scheduled_component!(pinned_buses, component)
    end
    return associated
end

_pin_scheduled_component!(buses::Set{Int}, branch::PSY.ACTransmission) =
    _accumulate_protected_buses!(buses, branch)
_pin_scheduled_component!(::Set{Int}, ::PSY.Component) = nothing

_push_scheduled_switch!(switches, switch::PSY.DiscreteControlledACBranch) =
    push!(switches, switch)
_push_scheduled_switch!(_, ::PSY.Component) = nothing

"""
    TopologySchedule(vmodf, sys, removed_by_step)

Per-step network topology on top of a factorized base network. `removed_by_step[t]` is the
set of components out of service at step `t`. Steps with the same set share one topology;
topology 0 is the base network. Sets are compared by component identity, so two distinct
components never collide even when they report the same id.

`ts[arc, t]` returns the PTDF row of `arc` at step `t`, referenced to the reference buses
with no distributed slack. Rows are cached in `vmodf`, so they persist across schedules
built on the same `vmodf`.
"""
struct TopologySchedule{M <: VirtualMODF}
    vmodf::M
    sys::PSY.System
    step_topology::Vector{Int}
    removed::Vector{Vector{PSY.Component}}
    modifications::Vector{NetworkModification}
    arcs_out::Vector{Set{Tuple{Int, Int}}}
    islands::Vector{Vector{Vector{Int}}}
    reference_islands::Vector{Set{Int}}
    contingency_components::Dict{Int, Vector{PSY.Component}}
    contingency_lock::ReentrantLock
end

function TopologySchedule(
    vmodf::VirtualMODF,
    sys::PSY.System,
    removed_by_step::AbstractVector{<:AbstractSet{<:PSY.Component}},
)
    keys_by_topology =
        Dict{Base.IdSet{PSY.Component}, Int}(Base.IdSet{PSY.Component}() => 0)
    step_topology = Vector{Int}(undef, length(removed_by_step))
    removed = Vector{Vector{PSY.Component}}()
    modifications = NetworkModification[]
    arcs_out = Set{Tuple{Int, Int}}[]
    for (t, components) in enumerate(removed_by_step)
        key = Base.IdSet{PSY.Component}(components)
        index = get(keys_by_topology, key, nothing)
        if isnothing(index)
            mod = NetworkModification(vmodf, sys, components)
            push!(removed, collect(PSY.Component, components))
            push!(modifications, mod)
            push!(arcs_out, _fully_out_arcs(vmodf, mod))
            index = length(modifications)
            keys_by_topology[key] = index
        end
        step_topology[t] = index
    end
    islands = Vector{Vector{Int}}[]
    reference_islands = Set{Int}[]
    for out in vcat([Set{Tuple{Int, Int}}()], arcs_out)
        labeled, refs = _label_islands(vmodf, out)
        push!(islands, labeled)
        push!(reference_islands, refs)
    end
    return TopologySchedule(
        vmodf,
        sys,
        step_topology,
        removed,
        modifications,
        arcs_out,
        islands,
        reference_islands,
        Dict{Int, Vector{PSY.Component}}(),
        ReentrantLock(),
    )
end

# An arc is out only when every member is opened; a partly-opened parallel group still
# conducts. The member-count rule is the Woodbury kernel's own. A modification built here
# changes an arc by exactly -b_arc when all its members open, so the kernel's susceptance
# test for islanding agrees.
function _fully_out_arcs(vmodf::VirtualMODF, mod::NetworkModification)
    arc_ax = get_arc_axis(vmodf)
    fully_opened = _arcs_fully_opened(get_core(vmodf), mod.arc_modifications)
    return Set{Tuple{Int, Int}}(
        arc_ax[m.arc_index] for
        (m, full) in zip(mod.arc_modifications, fully_opened) if full
    )
end

function _label_islands(vmodf::VirtualMODF, arcs_out::Set{Tuple{Int, Int}})
    bus_ax = get_bus_axis(vmodf)
    bus_lookup = get_bus_lookup(vmodf)
    uf = collect(1:length(bus_ax))
    for arc in get_arc_axis(vmodf)
        arc in arcs_out && continue
        union_sets!(uf, bus_lookup[arc[1]], bus_lookup[arc[2]])
    end
    members = Dict{Int, Vector{Int}}()
    for (i, bus) in enumerate(bus_ax)
        push!(get!(Vector{Int}, members, get_representative(uf, i)), bus)
    end
    islands = sort!(collect(values(members)); by = first)
    island_of_bus = Dict{Int, Int}(
        bus => k for (k, island) in enumerate(islands) for bus in island
    )
    reference_islands = Set{Int}(
        island_of_bus[bus_ax[position]] for position in get_ref_bus_position(vmodf)
    )
    return islands, reference_islands
end

"""
    get_topology_index(ts::TopologySchedule, t::Int) -> Int

Index of the topology in effect at step `t`. Steps with the same set of removed components
share an index; `0` is the base network.
"""
get_topology_index(ts::TopologySchedule, t::Int) = ts.step_topology[t]

"""
    get_network_modification(ts::TopologySchedule, t::Int) -> Union{NetworkModification, Nothing}

The [`NetworkModification`](@ref) that takes the base network to the topology at step `t`.
Returns `nothing` when the base network is in effect.
"""
function get_network_modification(ts::TopologySchedule, t::Int)
    k = get_topology_index(ts, t)
    k == 0 && return nothing
    return ts.modifications[k]
end

# The modification of topology `k`, the empty one for the base network.
function _topology_modification(ts::TopologySchedule, k::Int)
    k == 0 && return _BASE_NETWORK
    return ts.modifications[k]
end

"""
    get_islands(ts::TopologySchedule, t::Int)

Electrical islands at step `t`. Returns `(islands, reference_islands)`: `islands` holds the
bus numbers of each island and covers the retained buses of the reduced network only;
`reference_islands` holds the positions in `islands` of the islands that contain a
reference bus.
"""
function get_islands(ts::TopologySchedule, t::Int)
    k = get_topology_index(ts, t) + 1
    return ts.islands[k], ts.reference_islands[k]
end

"""
    is_arc_out(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int) -> Bool

Whether `arc` carries no flow at step `t`. An arc is out only when every parallel member is
out of service.
"""
function is_arc_out(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int)
    k = get_topology_index(ts, t)
    k == 0 && return false
    return arc in ts.arcs_out[k]
end

"""
    get_in_service_members(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int) -> Vector{PSY.ACTransmission}

The members of the parallel group `arc` that are in service at step `t`: those that
[`is_open_after_outage`](@ref) reports closed with the step's out branches tripped. Returns
an empty vector when `arc` is not a parallel group in the network reduction data.
"""
function get_in_service_members(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int)
    parallel = get_parallel_branch_map(get_network_reduction_data(ts.vmodf))
    haskey(parallel, arc) || return PSY.ACTransmission[]
    return _in_service_members(parallel[arc], _removed_at(ts, t))
end

# The branches out of service at step `t`, compared by identity. Non-branch components are
# dropped: they open no arc.
function _removed_at(ts::TopologySchedule, t::Int)
    removed = Base.IdSet{PSY.ACTransmission}()
    k = get_topology_index(ts, t)
    k == 0 && return removed
    for c in ts.removed[k]
        c isa PSY.ACTransmission && push!(removed, c)
    end
    return removed
end

_in_service_members(
    bp::AbstractBranchesParallel,
    removed::AbstractSet{PSY.ACTransmission},
) =
    PSY.ACTransmission[m for m in bp if !is_open_after_outage(m, removed)]

"""
    get_in_service_entry(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int) -> Union{PSY.ACTransmission, Nothing}

The branch entry conducting on `arc` at step `t`. Returns `nothing` when the arc is out,
and the arc's reduction entry when none of its members is out. Otherwise returns the member
still in service, a parallel group of the members still in service framed on the original
group's arc, or the Ward equivalent filed on the arc when it is all that remains.
"""
function get_in_service_entry(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int)
    is_arc_out(ts, arc, t) && return nothing
    nr = get_network_reduction_data(ts.vmodf)
    entry, _ = _resolve_arc_entry(nr, arc)
    get_topology_index(ts, t) == 0 && return entry
    return _in_service_entry(entry, nr, arc, _removed_at(ts, t))
end

function _in_service_entry(
    entry::PSY.ACTransmission,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
    removed::AbstractSet{PSY.ACTransmission},
)
    is_open_after_outage(entry, removed) || return entry
    return get_added_arc_impedance_map(nr)[arc]
end

function _in_service_entry(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
    removed::AbstractSet{PSY.ACTransmission},
)
    members = _in_service_members(bp, removed)
    length(members) == length(bp) && return bp
    isempty(members) && return get_added_arc_impedance_map(nr)[arc]
    length(members) == 1 && return only(members)
    return _parallel_group(members, get_arc_tuple(bp))
end

function _parallel_group(members::Vector{PSY.ACTransmission}, arc_key::Tuple{Int, Int})
    types = unique(typeof.(members))
    if length(types) == 1
        T = only(types)
        return BranchesParallel{T}(
            convert(Vector{T}, members),
            arc_key,
            EMPTY_TWO_PORT,
            false,
        )
    end
    return MixedBranchesParallel(members, arc_key, EMPTY_TWO_PORT, false)
end

"""
    get_arc_shift_injection(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int) -> Float64

The DC phase-shift injection `b·α` of `arc` at step `t`, in system base and in `arc`'s
frame: the value [`arc_dc_shift_injection`](@ref) returns on a network holding only the
arc's in-service members. Zero when the arc is out.
"""
get_arc_shift_injection(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int) =
    get_arc_shift_injection(ts, arc, get_in_service_entry(ts, arc, t))

"""
    get_arc_shift_injection(ts::TopologySchedule, arc::Tuple{Int, Int}, entry) -> Float64

The DC phase-shift injection `b·α` of `arc` when `entry`, as
[`get_in_service_entry`](@ref) returns it for some step, conducts on it; zero for
`nothing`. Lets a caller that already resolved the entry skip resolving it again.
"""
get_arc_shift_injection(::TopologySchedule, ::Tuple{Int, Int}, ::Nothing) = 0.0

function get_arc_shift_injection(
    ts::TopologySchedule,
    arc::Tuple{Int, Int},
    entry::PSY.ACTransmission,
)
    nr = get_network_reduction_data(ts.vmodf)
    reduction_entry, _ = _resolve_arc_entry(nr, arc)
    entry === reduction_entry && return arc_dc_shift_injection(nr, arc)
    return _entry_shift_injection(entry, nr, arc)
end

# `b·α` of an entry that is not registered on `arc`; its own arc tuple sets the orientation.
# A group's susceptance-weighted α times its summed susceptance is the sum of its members'
# injections.
function _entry_shift_injection(
    entry::PSY.ACTransmission,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
)
    α = _segment_phase_shift(entry, nr)
    iszero(α) && return 0.0
    if get_arc_tuple(entry, nr) == (arc[2], arc[1])
        α = -α
    end
    return _finite_series_susceptance(entry, nr) * α
end

function Base.getindex(ts::TopologySchedule, arc::Tuple{Int, Int}, t::Int)
    mod = get_network_modification(ts, t)
    isnothing(mod) && return get_ptdf_row(ts.vmodf, arc)
    return ts.vmodf[arc, mod]
end

"""
    get_topology_ptdf_rows(
        ts::TopologySchedule,
        arc::Tuple{Int, Int},
        topologies = 0:length(ts.modifications),
    ) -> Vector

The PTDF rows of `arc` on each topology in `topologies` (indices as returned by
[`get_topology_index`](@ref), `0` the base network; by default every topology of the
schedule, `0` through the number of distinct non-base topologies), from one solve of
the arc's base column. `rows[k] == ts[arc, t]`, bit for bit, for every step `t` on
topology `topologies[k]`. Rows are cached in the schedule's
[`VirtualMODF`](@ref) as `ts[arc, t]` caches them; see
[`get_post_modification_ptdf_rows`](@ref).
"""
function get_topology_ptdf_rows(
    ts::TopologySchedule,
    arc::Tuple{Int, Int},
    topologies::AbstractVector{Int} = 0:length(ts.modifications),
)
    mods = NetworkModification[_topology_modification(ts, k) for k in topologies]
    return get_post_modification_ptdf_rows(ts.vmodf, arc, mods)
end

function _post_contingency_modification(ts::TopologySchedule, k::Int, ctg_components)
    already_out = IdDict{PSY.Component, Nothing}()
    if k != 0
        for c in ts.removed[k]
            already_out[c] = nothing
        end
    end
    remaining = [c for c in ctg_components if !haskey(already_out, c)]
    isempty(remaining) && return nothing
    return NetworkModification(
        ts.vmodf,
        ts.sys,
        vcat(collect(keys(already_out)), remaining),
    )
end

"""
    get_post_contingency_network(ts::TopologySchedule, t::Int, contingency_id::Int) -> Union{PostContingencyNetwork, Nothing}

The [`PostContingencyNetwork`](@ref) of the `PSY.UnplannedOutage` with IS id
`contingency_id` on the topology in effect at step `t`. The outage's available associated
components are resolved through the schedule's system on the schedule's first query for
the outage and reused for every later step, and those already out at `t` are dropped.
Returns `nothing` when none remain, since the contingency then changes nothing.
Steps on the same topology return equal networks.
"""
function get_post_contingency_network(ts::TopologySchedule, t::Int, contingency_id::Int)
    k = get_topology_index(ts, t)
    post =
        _post_contingency_modification(ts, k, _contingency_components(ts, contingency_id))
    isnothing(post) && return nothing
    topology = _topology_modification(ts, k)
    return PostContingencyNetwork(
        _Tier2Derived(),
        topology,
        post,
        _tier2_modification(post, topology),
        contingency_id,
    )
end

function _contingency_components(ts::TopologySchedule, contingency_id::Int)
    return lock(ts.contingency_lock) do
        get!(ts.contingency_components, contingency_id) do
            _available_contingency_components(ts.sys, contingency_id)
        end
    end
end

function _available_contingency_components(sys::PSY.System, contingency_id::Int)
    outage = PSY.get_supplemental_attribute(sys, contingency_id)
    outage isa PSY.UnplannedOutage || throw(
        ArgumentError(
            "Supplemental attribute $contingency_id is a $(typeof(outage)), not a \
            PSY.UnplannedOutage.",
        ),
    )
    return PSY.Component[
        c for c in PSY.get_associated_components(sys, outage) if _in_network_candidate(c)
    ]
end

"""
    PostContingencyNetwork(vmodf::VirtualMODF, contingency_id::Int)

The [`PostContingencyNetwork`](@ref) of the contingency registered in `vmodf` under
`contingency_id`, on the base topology: `topology` is the empty modification, and
`post_contingency` and `tier2` are the registered modification. Throws when no contingency
is registered under `contingency_id`.
"""
function PostContingencyNetwork(vmodf::VirtualMODF, contingency_id::Int)
    mod = @lock get_core(vmodf).solver_lock _resolve_modification(vmodf, contingency_id)
    return _base_topology_network(mod, contingency_id)
end

# Δ = post − topology, arc by arc and bus by bus. An arc's group susceptance after the
# contingency comes from `post`, where the components out at K and the contingency's are
# resolved together, so adding Δ to `topology` lands on `post` with no member-share formula.
function _tier2_modification(post::NetworkModification, topology::NetworkModification)
    arc_mods = _modification_difference(
        ArcModification,
        post.arc_modifications,
        topology.arc_modifications,
        m -> m.arc_index,
        _arc_difference,
        i -> ArcModification(i, 0.0),
    )
    shunt_mods = _modification_difference(
        ShuntModification,
        post.shunt_modifications,
        topology.shunt_modifications,
        m -> m.bus_index,
        _shunt_difference,
        i -> ShuntModification(i, zero(YBUS_ELTYPE)),
    )
    return NetworkModification(post.label, arc_mods, shunt_mods, post.is_islanding)
end

function _modification_difference(
    ::Type{T},
    post::Tuple{Vararg{T}},
    topology::Tuple{Vararg{T}},
    index_of,
    difference,
    zero_at,
) where {T}
    by_index = Dict{Int, T}(index_of(m) => m for m in topology)
    diffs = T[]
    for m in post
        i = index_of(m)
        push!(diffs, difference(m, something(get(by_index, i, nothing), zero_at(i))))
        delete!(by_index, i)
    end
    for (i, m) in by_index
        push!(diffs, difference(zero_at(i), m))
    end
    filter!(!_is_zero_change, diffs)
    return sort!(diffs; by = index_of)
end

_arc_difference(a::ArcModification, b::ArcModification) = ArcModification(
    a.arc_index,
    a.delta_b - b.delta_b,
    a.delta_shift_injection - b.delta_shift_injection,
    a.delta_y11 - b.delta_y11,
    a.delta_y12 - b.delta_y12,
    a.delta_y21 - b.delta_y21,
    a.delta_y22 - b.delta_y22,
    a.opened - b.opened,
)

_shunt_difference(a::ShuntModification, b::ShuntModification) =
    ShuntModification(a.bus_index, a.delta_y - b.delta_y)

_is_zero_change(m::ArcModification) =
    iszero(m.delta_b) && iszero(m.delta_shift_injection) && iszero(m.delta_y11) &&
    iszero(m.delta_y12) && iszero(m.delta_y21) && iszero(m.delta_y22) &&
    iszero(m.opened)

_is_zero_change(m::ShuntModification) = iszero(m.delta_y)
