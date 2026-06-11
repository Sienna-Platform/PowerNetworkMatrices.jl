# Generic topology helpers shared by the network-reduction code. These operate on
# PowerSystems components (arcs, branches) and the system, not on any particular
# network matrix, so they live here rather than in a matrix-specific file. Consumed
# by `AdjacencyMatrix` reductions and by `modf_reduction_consistency.jl`.

function _arc_connecting_two_areas(arc::PSY.Arc)
    from_bus = PSY.get_from(arc)
    to_bus = PSY.get_to(arc)
    area_from = IS.get_uuid(PSY.get_area(from_bus))
    area_to = IS.get_uuid(PSY.get_area(to_bus))
    return area_from != area_to
end

_arc_connecting_two_areas(br::PSY.ACTransmission) =
    _arc_connecting_two_areas(PSY.get_arc(br))

function _arc_connecting_two_areas(br::PSY.ThreeWindingTransformer)
    # For the 3WT all the 4 buses are kept if any of the 3 arcs connect two areas
    arcs = [
        PSY.get_primary_star_arc(br),
        PSY.get_secondary_star_arc(br),
        PSY.get_tertiary_star_arc(br),
    ]
    for arc in arcs
        if _arc_connecting_two_areas(arc)
            return true
        end
    end
    return false
end

_is_not_nodal_branch(::PSY.ACTransmission) = false
_is_not_nodal_branch(::PSY.AreaInterchange) = true

# Push the available endpoint bus numbers of a branch into `buses`. Set-typed so
# callers can't accidentally pass a spec field and mutate it.
function _add_arc_buses!(buses::Set{Int}, arc::PSY.Arc)
    PSY.get_available(PSY.get_from(arc)) &&
        push!(buses, PSY.get_number(PSY.get_from(arc)))
    PSY.get_available(PSY.get_to(arc)) &&
        push!(buses, PSY.get_number(PSY.get_to(arc)))
    return
end

_add_arc_buses!(buses::Set{Int}, br::PSY.ACTransmission) =
    _add_arc_buses!(buses, PSY.get_arc(br))

function _add_arc_buses!(buses::Set{Int}, br::PSY.ThreeWindingTransformer)
    _add_arc_buses!(buses, PSY.get_primary_star_arc(br))
    _add_arc_buses!(buses, PSY.get_secondary_star_arc(br))
    _add_arc_buses!(buses, PSY.get_tertiary_star_arc(br))
    return
end

"""
    _system_derived_irreducible_buses(sys, reduce_reactive_power_injectors) -> Set{Int}

Fresh `Set{Int}` of buses the system requires kept: `StaticInjection` hosts,
`TwoTerminalHVDC` endpoints, cross-area `ACTransmission` endpoints (when any
`AreaInterchange` exists), and branch-typed `TransmissionInterface` contributors.

A `StaticInjection` host is kept when the injector supports active power. When
`reduce_reactive_power_injectors` is `false`, reactive-power injectors also keep
their host bus; when `true` (the default), buses whose injectors only support
reactive power become reduction candidates. The non-injection protections are
unconditional.
"""
function _system_derived_irreducible_buses(
    sys::PSY.System,
    reduce_reactive_power_injectors::Bool,
)
    buses = Set{Int}()
    for c in PSY.get_components(PSY.StaticInjection, sys)
        PSY.get_available(c) || continue
        # An injector keeps its bus if it supports active power, or (when reactive
        # injectors are not being reduced) reactive power.
        forces_keep = PSY.supports_active_power(c) ||
                      (!reduce_reactive_power_injectors && PSY.supports_reactive_power(c))
        forces_keep || continue
        bus = PSY.get_bus(c)
        PSY.get_available(bus) || continue
        push!(buses, PSY.get_number(bus))
    end
    for tw_hvdc in PSY.get_components(PSY.TwoTerminalHVDC, sys)
        _add_arc_buses!(buses, PSY.get_arc(tw_hvdc))
    end
    if PSY.has_components(sys, PSY.AreaInterchange)
        for br in PSY.get_components(PSY.ACTransmission, sys)
            (PSY.get_available(br) && _arc_connecting_two_areas(br)) || continue
            _add_arc_buses!(buses, br)
        end
    end
    if PSY.has_components(sys, PSY.TransmissionInterface)
        for interface in PSY.get_components(PSY.TransmissionInterface, sys)
            for br in PSY.get_contributing_devices(sys, interface)
                (_is_not_nodal_branch(br) || !PSY.get_available(br)) && continue
                _add_arc_buses!(buses, br)
            end
        end
    end
    return buses
end
