"""
    IterativeTopologyReduction <: NetworkReduction

Applies [`RadialReduction`](@ref) and [`DegreeTwoReduction`](@ref) in alternation until no bus is
eliminated, so the reduced network retains no reducible bus of degree one or two.

The two primitives generate work for each other: peeling a leaf drops its parent's degree, and
collapsing sibling chains or absorbing a chain onto an existing arc merges two of an endpoint's
edges into one. Alternating to convergence is therefore strictly stronger than any fixed sequence.

Buses protected from reduction are unchanged from the primitives: the reference buses, any bus in
`Ybus(sys; irreducible_buses=...)`, and the system-derived set each primitive computes for itself.

# Fields
- `radial::RadialReduction = RadialReduction()`: the radial step's spec.
- `degree_two::DegreeTwoReduction = DegreeTwoReduction()`: the degree-two step's spec. AC
  consumers must construct this with `reduce_reactive_power_injectors = false`.
- `max_rounds::Int = 20`: safety bound. Convergence is monotone because each round eliminates at
  least one bus or stops, so reaching this bound indicates a defect and raises an error rather
  than looping.
"""
@kwdef struct IterativeTopologyReduction <: NetworkReduction
    radial::RadialReduction = RadialReduction()
    degree_two::DegreeTwoReduction = DegreeTwoReduction()
    max_rounds::Int = 20
end

get_radial_reduction(r::IterativeTopologyReduction) = r.radial
get_degree_two_reduction(r::IterativeTopologyReduction) = r.degree_two
get_max_rounds(r::IterativeTopologyReduction) = r.max_rounds
