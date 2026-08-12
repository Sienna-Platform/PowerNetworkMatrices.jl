"""
    IterativeTopologyReduction <: NetworkReduction

Applies [`RadialReduction`](@ref) and [`DegreeTwoReduction`](@ref) in alternation until no bus is
eliminated, so the reduced network retains no reducible bus of degree one or two.

The two primitives generate work for each other: peeling a leaf drops its parent's degree, and
collapsing sibling chains or absorbing a chain onto an existing arc merges two of an endpoint's
edges into one. Alternating to convergence is therefore strictly stronger than any fixed sequence.

Buses protected from reduction are unchanged from the primitives: the reference buses, any bus in
`Ybus(sys; irreducible_buses=...)`, and the system-derived set each primitive computes for itself.
The two primitives protect different things, though, and a bus surviving a round is not the same
guarantee both ways: `RadialReduction` has no injector-awareness of its own (only reference buses
and the caller's `irreducible_buses` are exempt from it), so an unpinned injector host that becomes
a leaf is peeled like any other bus — its injection is remapped onto its single surviving parent
via `reverse_bus_search_map`, not dropped, but it no longer has a distinct bus of its own. Only
`DegreeTwoReduction`'s system-derived set keeps an injector host protected throughout the loop; a
bus that needs to remain electrically distinct must also be passed to `irreducible_buses`.

# Fields
- `radial::RadialReduction = RadialReduction()`: the radial step's spec.
- `degree_two::DegreeTwoReduction = DegreeTwoReduction()`: the degree-two step's spec. AC
  consumers must construct this with `reduce_reactive_power_injectors = false`.
- `max_rounds::Int = 20`: bound on *productive* rounds (the terminal no-op round that confirms
  convergence doesn't count against it). Convergence is monotone because each round eliminates at
  least one bus or stops, so reaching this bound means either the topology needed more rounds than
  allotted, `max_rounds` was set too low, or a primitive reduction has a defect.
"""
@kwdef struct IterativeTopologyReduction <: NetworkReduction
    radial::RadialReduction = RadialReduction()
    degree_two::DegreeTwoReduction = DegreeTwoReduction()
    max_rounds::Int = 20
end

get_radial_reduction(r::IterativeTopologyReduction) = r.radial
get_degree_two_reduction(r::IterativeTopologyReduction) = r.degree_two
get_max_rounds(r::IterativeTopologyReduction) = r.max_rounds
