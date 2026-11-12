# Bus rules that map a component to the buses a network reduction must keep. One rule set
# serves `get_irreducible_bus_numbers` and the MODF outage protection.
_add_component_buses!(buses::Set{Int}, bus::PSY.ACBus) = _add_available_bus!(buses, bus)

_add_component_buses!(buses::Set{Int}, branch::PSY.ACTransmission) =
    _add_arc_buses!(buses, branch)

_add_component_buses!(buses::Set{Int}, hvdc::PSY.TwoTerminalHVDC) =
    _add_arc_buses!(buses, PSY.get_arc(hvdc))

function _add_component_buses!(buses::Set{Int}, injector::PSY.StaticInjection)
    PSY.get_available(injector) || return
    _add_available_bus!(buses, PSY.get_bus(injector))
    return
end

function _add_component_buses!(::Set{Int}, component::PSY.Component)
    throw(
        ArgumentError(
            "Component $(PSY.get_name(component)) of type $(typeof(component)) has no \
            irreducible-bus rule. Select buses, AC branches, two-terminal HVDC lines, or \
            static injectors.",
        ),
    )
end

function _add_available_bus!(buses::Set{Int}, bus::PSY.ACBus)
    PSY.get_available(bus) && push!(buses, PSY.get_number(bus))
    return
end

const _IrreducibleBusRuleComponent =
    Union{PSY.ACBus, PSY.ACTransmission, PSY.TwoTerminalHVDC, PSY.StaticInjection}

"""
    get_irreducible_bus_numbers(sys::PSY.System, selector::IS.ComponentSelector) -> Set{Int}

Return the numbers of the buses that a network reduction must keep for the available
components that `selector` selects in `sys`.

The bus rule for each component type:

  - `ACBus`: the bus.
  - `ACTransmission`: the from and to buses of the arc. A `ThreeWindingTransformer` gives the
    buses of each winding arc.
  - `TwoTerminalHVDC`: the from and to buses of the arc.
  - `StaticInjection`: the bus of the injector.

Unavailable buses are not in the result. A component of another type gives an
`ArgumentError`.

# Examples

```julia
get_irreducible_bus_numbers(sys, make_selector(Line, "1-2"))
get_irreducible_bus_numbers(sys, make_selector(make_selector(ACBus, "bus3"), make_selector(Line)))
```
"""
function get_irreducible_bus_numbers(sys::PSY.System, selector::IS.ComponentSelector)
    buses = Set{Int}()
    for component in PSY.get_available_components(selector, sys)
        _add_component_buses!(buses, component)
    end
    return buses
end
