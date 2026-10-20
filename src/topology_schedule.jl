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

"""
    get_post_contingency_modification(ts::TopologySchedule, t::Int, ctg_components) -> Union{NetworkModification, Nothing}

The network after contingency `ctg_components` at step `t`: the components already out at
`t` plus the contingency's remaining components, resolved together. Returns `nothing` when
every contingency component is already out, since the contingency then changes nothing.
"""
function get_post_contingency_modification(ts::TopologySchedule, t::Int, ctg_components)
    k = get_topology_index(ts, t)
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
