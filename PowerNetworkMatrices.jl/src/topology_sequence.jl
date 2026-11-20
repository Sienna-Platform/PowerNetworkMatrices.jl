"""
    TopologySequence(vmodf, sys, removed_by_step)

Per-step network topology on top of a factorized base network. `removed_by_step[t]` is the
set of components out of service at step `t`. Steps with the same set share one topology;
topology 0 is the base network. Sets are compared by component identity, so two distinct
components never collide even when they report the same id.

`ts[arc, t]` returns the PTDF row of `arc` at step `t`, referenced to the reference buses
with no distributed slack. Rows are cached in `vmodf`, so they persist across sequences
built on the same `vmodf`.
"""
struct TopologySequence{M <: VirtualMODF}
    vmodf::M
    sys::PSY.System
    step_topology::Vector{Int}
    removed::Vector{Vector{PSY.Component}}
    removed_branches::Vector{Base.IdSet{PSY.ACTransmission}}
    touched_arcs::Vector{Set{Tuple{Int, Int}}}
    modifications::Vector{NetworkModification}
    arcs_out::Vector{Set{Tuple{Int, Int}}}
    islands::Vector{Vector{Vector{Int}}}
    reference_islands::Vector{Set{Int}}
end

function TopologySequence(
    vmodf::VirtualMODF,
    sys::PSY.System,
    removed_by_step::AbstractVector{<:AbstractSet{<:PSY.Component}},
)
    keys_by_topology =
        Dict{Base.IdSet{PSY.Component}, Int}(Base.IdSet{PSY.Component}() => 0)
    step_topology = Vector{Int}(undef, length(removed_by_step))
    removed = Vector{Vector{PSY.Component}}()
    removed_branches = Base.IdSet{PSY.ACTransmission}[]
    touched_arcs = Set{Tuple{Int, Int}}[]
    modifications = NetworkModification[]
    arcs_out = Set{Tuple{Int, Int}}[]
    for (t, components) in enumerate(removed_by_step)
        key = Base.IdSet{PSY.Component}(components)
        index = get(keys_by_topology, key, nothing)
        if isnothing(index)
            mod = NetworkModification(vmodf, sys, components)
            push!(removed, collect(PSY.Component, components))
            push!(removed_branches, _removed_branches(components))
            push!(touched_arcs, _touched_arcs(vmodf, mod))
            push!(modifications, mod)
            push!(arcs_out, _fully_out_arcs(vmodf, mod))
            index = length(modifications)
            keys_by_topology[key] = index
        end
        step_topology[t] = index
    end
    islands = Vector{Vector{Int}}[]
    reference_islands = Set{Int}[]
    for mod in vcat([_BASE_NETWORK], modifications)
        labeled, refs = _label_islands(vmodf, mod)
        push!(islands, labeled)
        push!(reference_islands, refs)
    end
    return TopologySequence(
        vmodf,
        sys,
        step_topology,
        removed,
        removed_branches,
        touched_arcs,
        modifications,
        arcs_out,
        islands,
        reference_islands,
    )
end

# Non-branch components are dropped: they open no arc.
function _removed_branches(components)
    branches = Base.IdSet{PSY.ACTransmission}()
    for c in components
        c isa PSY.ACTransmission && push!(branches, c)
    end
    return branches
end

# The arcs whose in-service entry can differ from the base entry: those the modification
# changes.
function _touched_arcs(vmodf::VirtualMODF, mod::NetworkModification)
    arc_ax = get_arc_axis(vmodf)
    return Set{Tuple{Int, Int}}(arc_ax[m.arc_index] for m in mod.arc_modifications)
end

# An arc is out only when every member is opened; a partly-opened parallel group still
# conducts. The member-count rule is the Woodbury kernel's own. A modification built here
# changes an arc by exactly -b_arc when all its members open, so the kernel's susceptance
# test for islanding agrees.
function _fully_out_arcs(vmodf::VirtualMODF, mod::NetworkModification)
    arc_ax = get_arc_axis(vmodf)
    fully_opened = _arcs_fully_opened(get_core(vmodf), _arc_modification_vector(mod))
    return Set{Tuple{Int, Int}}(
        arc_ax[m.arc_index] for
        (m, full) in zip(mod.arc_modifications, fully_opened) if full
    )
end

# The island labels are the Woodbury kernel's, so a bus is outside the reference islands here
# exactly when the kernel zeroes its entries.
function _label_islands(vmodf::VirtualMODF, mod::NetworkModification)
    core = get_core(vmodf)
    bus_ax = get_bus_axis(vmodf)
    labels = _post_contingency_bus_labels(
        core.BA,
        core.arc_susceptances,
        _arc_modification_vector(mod),
        length(bus_ax),
    )
    members = Dict{Int, Vector{Int}}()
    for (i, bus) in enumerate(bus_ax)
        push!(get!(Vector{Int}, members, labels[i]), bus)
    end
    islands = sort!(collect(values(members)); by = first)
    island_of_bus = Dict{Int, Int}(
        bus => k for (k, island) in enumerate(islands) for bus in island
    )
    reference_islands = Set{Int}(
        island_of_bus[bus_ax[position]] for position in PMC._ref_bus_positions(core)
    )
    return islands, reference_islands
end

"""
    get_topology_index(ts::TopologySequence, t::Int) -> Int

Index of the topology in effect at step `t`. Steps with the same set of removed components
share an index; `0` is the base network.
"""
get_topology_index(ts::TopologySequence, t::Int) = ts.step_topology[t]

"""
    get_network_modification(ts::TopologySequence, t::Int) -> Union{NetworkModification, Nothing}

The [`NetworkModification`](@ref) that takes the base network to the topology at step `t`.
Returns `nothing` when the base network is in effect.
"""
function get_network_modification(ts::TopologySequence, t::Int)
    k = get_topology_index(ts, t)
    k == 0 && return nothing
    return ts.modifications[k]
end

# The modification of topology `k`, the empty one for the base network.
function _topology_modification(ts::TopologySequence, k::Int)
    k == 0 && return _BASE_NETWORK
    return ts.modifications[k]
end

"""
    get_islands(ts::TopologySequence, t::Int)

Electrical islands at step `t`. Returns `(islands, reference_islands)`: `islands` holds the
bus numbers of each island and covers the retained buses of the reduced network only;
`reference_islands` holds the positions in `islands` of the islands that contain a
reference bus. Both are shared with the stored topology: treat them as read-only and copy
them before mutating.
"""
function get_islands(ts::TopologySequence, t::Int)
    k = get_topology_index(ts, t) + 1
    return ts.islands[k], ts.reference_islands[k]
end

"""
    is_arc_out(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int) -> Bool

Whether `arc` carries no flow at step `t`. An arc is out only when every parallel member is
out of service. A query on the reverse of a series or parallel arc resolves to the arc's
stored key.
"""
is_arc_out(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int) =
    _is_key_out(ts, _stored_arc_key(get_network_reduction_data(ts.vmodf), arc), t)

# `key` is a stored key, as `_stored_arc_key` returns it.
function _is_key_out(ts::TopologySequence, key::Tuple{Int, Int}, t::Int)
    k = get_topology_index(ts, t)
    k == 0 && return false
    return key in ts.arcs_out[k]
end

"""
    get_in_service_members(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int) -> Vector{PSY.ACTransmission}

The members of the parallel group `arc` that are in service at step `t`: those that
[`is_open_after_outage`](@ref) reports closed with the step's out branches tripped. A member
chain that loses part of a parallel segment without opening is rebuilt from its remaining
members, as in [`get_in_service_entry`](@ref). A query on the reverse of the group's arc
resolves to the group. Returns an empty vector when `arc` is not a parallel group in the
network reduction data.
"""
function get_in_service_members(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int)
    nr = get_network_reduction_data(ts.vmodf)
    parallel = get_parallel_branch_map(nr)
    key = _stored_arc_key(nr, arc)
    haskey(parallel, key) || return PSY.ACTransmission[]
    return _in_service_members(parallel[key], _removed_at(ts, t))
end

# The branches out of service at step `t`, compared by identity.
function _removed_at(ts::TopologySequence, t::Int)
    k = get_topology_index(ts, t)
    k == 0 && return Base.IdSet{PSY.ACTransmission}()
    return ts.removed_branches[k]
end

_in_service_members(
    bp::AbstractBranchesParallel,
    removed::AbstractSet{PSY.ACTransmission},
) = PSY.ACTransmission[
    _in_service_member(m, removed) for m in bp if !is_open_after_outage(m, removed)
]

# A member the outage does not reach keeps its identity.
function _in_service_member(
    member::PSY.ACTransmission,
    removed::AbstractSet{PSY.ACTransmission},
)
    _outage_reaches(member, removed) || return member
    return _entry_after_outage(member, removed)
end

"""
    get_in_service_entry(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int) -> Union{PSY.ACTransmission, Nothing}

The branch entry conducting on `arc` at step `t`. Returns `nothing` when the arc is out,
and the arc's reduction entry when none of its members is out. Otherwise returns the member
still in service, a parallel group of the members still in service framed on the original
group's arc, or the Ward equivalent filed on the arc when it is all that remains. A series
chain, or a group member, that loses part of a parallel segment without opening is rebuilt
from its remaining members, as [`NetworkModification`](@ref) rebuilds it. A query on the
reverse of a series or parallel arc resolves to the arc's stored key.
"""
function get_in_service_entry(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int)
    nr = get_network_reduction_data(ts.vmodf)
    key = _stored_arc_key(nr, arc)
    _is_key_out(ts, key, t) && return nothing
    entry = _base_arc_entry(nr, key)
    k = get_topology_index(ts, t)
    k == 0 && return entry
    key in ts.touched_arcs[k] || return entry
    return _in_service_entry(entry, nr, key, ts.removed_branches[k])
end

# The key the reduction maps file `arc`'s entry under, probed in the order of
# `_probe_arc_entry`: a reverse-key query of a series chain or parallel group resolves to
# the chain's or group's own key, and a direct arc keeps its own key even beside an
# anti-parallel twin.
function _stored_arc_key(nr::NetworkReductionData, arc::Tuple{Int, Int})
    haskey(get_direct_branch_map(nr), arc) && return arc
    rev = (arc[2], arc[1])
    series_map = get_series_branch_map(nr)
    haskey(series_map, arc) && return arc
    haskey(series_map, rev) && return rev
    parallel_map = get_parallel_branch_map(nr)
    haskey(parallel_map, arc) && return arc
    haskey(parallel_map, rev) && return rev
    return arc
end

# The reduction entry filed under the stored key `key`.
function _base_arc_entry(nr::NetworkReductionData, key::Tuple{Int, Int})
    direct_map = get_direct_branch_map(nr)
    haskey(direct_map, key) && return direct_map[key]
    series_map = get_series_branch_map(nr)
    haskey(series_map, key) && return series_map[key]
    parallel_map = get_parallel_branch_map(nr)
    haskey(parallel_map, key) && return parallel_map[key]
    added_map = get_added_arc_impedance_map(nr)
    haskey(added_map, key) && return added_map[key]
    error("Arc $(key) not found in any network reduction map.")
end

# Whether the outage trips any leaf of `entry`; an entry it does not reach keeps its
# identity.
_outage_reaches(entry::PSY.ACTransmission, removed::AbstractSet{PSY.ACTransmission}) =
    is_open_after_outage(entry, removed)
_outage_reaches(
    aggregate::AbstractReductionAggregate,
    removed::AbstractSet{PSY.ACTransmission},
) = any(member -> _outage_reaches(member, removed), aggregate)

function _in_service_entry(
    entry::PSY.ACTransmission,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
    removed::AbstractSet{PSY.ACTransmission},
)
    _outage_reaches(entry, removed) || return entry
    is_open_after_outage(entry, removed) && return get_added_arc_impedance_map(nr)[arc]
    return _entry_after_outage(entry, removed)
end

function _in_service_entry(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
    removed::AbstractSet{PSY.ACTransmission},
)
    _outage_reaches(bp, removed) || return bp
    members = _in_service_members(bp, removed)
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
    get_arc_shift_injection(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int) -> Float64

The DC phase-shift injection `b·α` of `arc` at step `t`, in system base and in `arc`'s
frame: the value [`arc_dc_shift_injection`](@ref) returns on a network holding only the
arc's in-service members. Zero when the arc is out.
"""
get_arc_shift_injection(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int) =
    get_arc_shift_injection(ts, arc, get_in_service_entry(ts, arc, t))

"""
    get_arc_shift_injection(ts::TopologySequence, arc::Tuple{Int, Int}, entry) -> Float64

The DC phase-shift injection `b·α` of `arc` when `entry`, as
[`get_in_service_entry`](@ref) returns it for some step, conducts on it; zero for
`nothing`. Lets a caller that already resolved the entry skip resolving it again.
"""
get_arc_shift_injection(::TopologySequence, ::Tuple{Int, Int}, ::Nothing) = 0.0

function get_arc_shift_injection(
    ts::TopologySequence,
    arc::Tuple{Int, Int},
    entry::PSY.ACTransmission,
)
    nr = get_network_reduction_data(ts.vmodf)
    reduction_entry, _ = _resolve_arc_entry(nr, arc)
    entry === reduction_entry && return arc_dc_shift_injection(nr, arc)
    return _entry_shift_injection(entry, nr, arc)
end

# `b·α` of an entry that is not registered on `arc`; its own arc tuple sets the orientation.
function _entry_shift_injection(
    entry::PSY.ACTransmission,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
)
    α = _segment_phase_shift(entry, nr)
    iszero(α) && return 0.0
    return _orient_to_arc(_finite_series_susceptance(entry, nr) * α, entry, nr, arc)
end

# Injections add across members, so a group's is their sum; the members' susceptances can
# cancel, which rules out the group's equivalent angle times its summed susceptance.
function _entry_shift_injection(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
)
    injection = sum(member -> _member_shift_injection(bp, nr, member), bp)
    return _orient_to_arc(injection, bp, nr, arc)
end

function _orient_to_arc(
    injection::Float64,
    entry::PSY.ACTransmission,
    nr::NetworkReductionData,
    arc::Tuple{Int, Int},
)
    get_arc_tuple(entry, nr) == (arc[2], arc[1]) && return -injection
    return injection
end

function Base.getindex(ts::TopologySequence, arc::Tuple{Int, Int}, t::Int)
    mod = get_network_modification(ts, t)
    isnothing(mod) && return get_ptdf_row(ts.vmodf, arc)
    return ts.vmodf[arc, mod]
end

"""
    get_topology_ptdf_rows(
        ts::TopologySequence,
        arc::Tuple{Int, Int},
        topologies = 0:length(ts.modifications),
    ) -> Vector

The PTDF rows of `arc` on each topology in `topologies` (indices as returned by
[`get_topology_index`](@ref), `0` the base network; by default every topology of the
sequence, `0` through the number of distinct non-base topologies), from one solve of
the arc's base column. `rows[k] == ts[arc, t]`, bit for bit, for every step `t` on
topology `topologies[k]`. Rows are cached in the sequence's
[`VirtualMODF`](@ref) as `ts[arc, t]` caches them; see
[`get_post_modification_ptdf_rows`](@ref).
"""
function get_topology_ptdf_rows(
    ts::TopologySequence,
    arc::Tuple{Int, Int},
    topologies::AbstractVector{Int} = 0:length(ts.modifications),
)
    mods = NetworkModification[_topology_modification(ts, k) for k in topologies]
    return get_post_modification_ptdf_rows(ts.vmodf, arc, mods)
end
