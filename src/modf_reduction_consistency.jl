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

# Ward's `study_buses` need explicit augmentation; other reductions read the
# user set via the orchestrator container and pass through.
_augment_ward(reduction::NetworkReduction, ::Set{Int}) = reduction
function _augment_ward(reduction::WardReduction, protected::Set{Int})
    isempty(protected) && return reduction
    return WardReduction(_merge_irreducible(reduction.study_buses, protected))
end

"""Extend any `WardReduction.study_buses` in `reductions` with `protected`."""
function _augment_ward_reductions(
    reductions::Vector{NetworkReduction},
    protected::Set{Int},
)
    return NetworkReduction[_augment_ward(r, protected) for r in reductions]
end
