"""
    ZeroImpedanceBranchReduction <: NetworkReduction

Merges buses connected by zero-impedance non-transformer branches. Always
applied as the first step of `Ybus(sys; ...)` to avoid singular admittances;
pass via the `zero_impedance_reduction` kwarg to override parameters — putting
one in `network_reductions` is rejected.

An arc is treated as zero-impedance when *any individual* non-transformer branch
on it has resistance within `resistance_tolerance` (`abs(r) <= resistance_tolerance`)
and series admittance `abs(y) >= susceptance_threshold`, matching PSS(e). Examined
per branch (not via the summed off-diagonal), so a zero-impedance jumper in
parallel with a normal line still merges the buses. The from-bus survives unless
the to-bus is in the user-supplied irreducible set, in which case the sides flip.

# Fields
- `susceptance_threshold::Float64 = ZERO_IMPEDANCE_BRANCH_YBUS_SUSCEPTANCE_THRESHOLD`
- `minimum_retained_impedance::Float64 = ZERO_IMPEDANCE_X_EPSILON`: substitute
  reactance for branches with `r == x == 0` during Ybus assembly.
- `resistance_tolerance::Float64 = 0.0`: a branch counts as zero-impedance when
  `abs(r) <= resistance_tolerance` (default `0.0` ⇒ exact `r == 0`, unchanged
  behavior). Raise it to also merge near-zero-impedance lines that carry a tiny
  but nonzero resistance (PSS(e)-style magnitude threshold; their high `abs(y)`
  still has to clear `susceptance_threshold`).
"""
@kwdef struct ZeroImpedanceBranchReduction <: NetworkReduction
    susceptance_threshold::Float64 = ZERO_IMPEDANCE_BRANCH_YBUS_SUSCEPTANCE_THRESHOLD
    minimum_retained_impedance::Float64 = ZERO_IMPEDANCE_X_EPSILON
    resistance_tolerance::Float64 = 0.0
end

get_susceptance_threshold(z::ZeroImpedanceBranchReduction) = z.susceptance_threshold
get_minimum_retained_impedance(z::ZeroImpedanceBranchReduction) =
    z.minimum_retained_impedance
get_resistance_tolerance(z::ZeroImpedanceBranchReduction) = z.resistance_tolerance
