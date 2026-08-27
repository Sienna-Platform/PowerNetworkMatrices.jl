# Outage-protection set for VirtualMODF: buses whose loss would make an
# outaged/monitored component non-queryable in the reduced network.

# Add a monitored component's bus(es). Branches contribute both arc endpoints;
# unsupported component types warn and are skipped.
function _accumulate_protected_buses!(buses::Set{Int}, branch::PSY.ACTransmission)
    _add_arc_buses!(buses, branch)
    return
end

function _accumulate_protected_buses!(buses::Set{Int}, bus::PSY.ACBus)
    PSY.get_available(bus) && push!(buses, PSY.get_number(bus))
    return
end

function _accumulate_protected_buses!(buses::Set{Int}, c::PSY.StaticInjection)
    PSY.get_available(c) || return
    bus = PSY.get_bus(c)
    PSY.get_available(bus) && push!(buses, PSY.get_number(bus))
    return
end

function _accumulate_protected_buses!(::Set{Int}, c::PSY.Component)
    @warn "Outage-monitored component $(typeof(c)) ($(PSY.get_name(c))) has no " *
          "reduction-protection rule; its bus will not be added to the protected " *
          "set and may be reduced away." maxlog = 5
    return
end

# Stale UUID -> warn and skip; everything else rethrows. Dispatched to avoid `isa`.
_warn_or_rethrow_missing_component(::ArgumentError, uuid) =
    @warn "Outage monitored component UUID $uuid not found in system; " *
          "cannot protect it from reduction."
_warn_or_rethrow_missing_component(e, uuid) = rethrow()

function _accumulate_monitored_buses!(
    buses::Set{Int},
    sys::PSY.System,
    outage::PSY.Outage,
)
    for uuid in PSY.get_monitored_components(outage)
        local component
        try
            component = IS.get_component(sys, uuid)
        catch e
            _warn_or_rethrow_missing_component(e, uuid)
            continue
        end
        _accumulate_protected_buses!(buses, component)
    end
    return
end

"""
    _collect_protected_buses(sys) -> Set{Int}

Buses to protect so every `PSY.Outage`'s outaged and monitored components
remain queryable as arcs after reduction. Computed from `sys` alone (no `Ybus`
needed) so the result can be folded into `irreducible_buses` *before* the base
`Ybus` (and its auto-applied zero-impedance reduction) is built — otherwise a
monitored/outaged branch endpoint could be merged away by ZIBR before it is
known to need protecting.
"""
function _collect_protected_buses(sys::PSY.System)
    buses = Set{Int}()
    for outage in PSY.get_supplemental_attributes(PSY.Outage, sys)
        for component in PSY.get_associated_components(sys, outage)
            _accumulate_protected_buses!(buses, component)
        end
        _accumulate_monitored_buses!(buses, sys, outage)
    end
    return buses
end

_merge_irreducible(existing, protected::Set{Int}) =
    sort!(collect(union(Set(existing), protected)))

"""
Branches whose outage or monitoring the MODF must be able to represent: an outaged branch
becomes an `ArcModification` on its own arc, and a monitored branch is the row a contingency
query is read from. Both need their arc to survive the reduction.

Outaged *injectors* are excluded. They register with an empty `NetworkModification`, so no
reduction can invalidate them.
"""
function _contingency_relevant_branches(sys::PSY.System)
    branches = Set{PSY.ACTransmission}()
    for outage in PSY.get_supplemental_attributes(PSY.Outage, sys)
        for component in PSY.get_associated_components(sys, outage)
            component isa PSY.ACTransmission && push!(branches, component)
        end
        for uuid in PSY.get_monitored_components(outage)
            local component
            try
                component = IS.get_component(sys, uuid)
            catch e
                _warn_or_rethrow_missing_component(e, uuid)
                continue
            end
            component isa PSY.ACTransmission && push!(branches, component)
        end
    end
    return branches
end

"""
Reject a `WardReduction` whose study area does not contain every branch the MODF needs.

`study_buses` *defines* the network Ward retains, unlike the radial/degree-two irreducible
set, which only exempts buses from elimination. A contingency on a branch outside it cannot
survive: the arc is gone after the reduction, and every query for that contingency resolves
to the base-case row.
"""
function _validate_ward_contingency_coverage(
    reductions::Vector{NetworkReduction},
    sys::PSY.System,
)
    wards = [r for r in reductions if r isa WardReduction]
    isempty(wards) && return
    relevant = _contingency_relevant_branches(sys)
    isempty(relevant) && return
    for ward in wards
        study = Set(get_study_buses(ward))
        outside = String[]
        for branch in relevant
            buses = Set{Int}()
            _add_arc_buses!(buses, branch)
            issubset(buses, study) || push!(outside, PSY.get_name(branch))
        end
        isempty(outside) && continue
        sort!(outside)
        shown = join(first(outside, 5), ", ")
        suffix = length(outside) > 5 ? " (and $(length(outside) - 5) more)" : ""
        throw(
            IS.ConflictingInputsError(
                "WardReduction retains only its study_buses, so $(length(outside)) \
                outage-monitored branch(es) lying outside that area cannot survive it and \
                their contingencies would silently resolve to the base case: $(shown)$(suffix). \
                Extend study_buses to cover them, or drop the WardReduction.",
            ),
        )
    end
    return
end
