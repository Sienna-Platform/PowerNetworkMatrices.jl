"""
    EquivalentBranch

Represents the equivalent parameters of a network branch for power flow calculations.

# Fields
- `equivalent_r::Float64`: Equivalent series resistance (p.u.)
- `equivalent_x::Float64`: Equivalent series reactance (p.u.)
- `equivalent_g_from::Float64`: Equivalent shunt conductance at the "from" bus (p.u.)
- `equivalent_b_from::Float64`: Equivalent shunt susceptance at the "from" bus (p.u.)
- `equivalent_g_to::Float64`: Equivalent shunt conductance at the "to" bus (p.u.)
- `equivalent_b_to::Float64`: Equivalent shunt susceptance at the "to" bus (p.u.)
- `equivalent_tap::Float64`: Equivalent transformer tap ratio
- `equivalent_shift::Float64`: Equivalent phase shift angle (radians)
"""
mutable struct EquivalentBranch
    equivalent_r::Float64
    equivalent_x::Float64
    equivalent_g_from::Float64
    equivalent_b_from::Float64
    equivalent_g_to::Float64
    equivalent_b_to::Float64
    equivalent_tap::Float64
    equivalent_shift::Float64
end

get_equivalent_r(eb::EquivalentBranch) = eb.equivalent_r
get_equivalent_x(eb::EquivalentBranch) = eb.equivalent_x
get_equivalent_g_from(eb::EquivalentBranch) = eb.equivalent_g_from
get_equivalent_b_from(eb::EquivalentBranch) = eb.equivalent_b_from
get_equivalent_g_to(eb::EquivalentBranch) = eb.equivalent_g_to
get_equivalent_b_to(eb::EquivalentBranch) = eb.equivalent_b_to
get_equivalent_tap(eb::EquivalentBranch) = eb.equivalent_tap
get_equivalent_shift(eb::EquivalentBranch) = eb.equivalent_shift
