"""
    ZeroImpedanceBranchReduction <: NetworkReduction

Merges buses connected by zero-impedance non-transformer branches. Always
applied as the first step of `Ybus(sys; ...)` to avoid singular admittances;
pass via the `zero_impedance_reduction` kwarg to override parameters — putting
one in `network_reductions` is rejected.

An arc is treated as zero-impedance when either *any individual* non-transformer
branch on it has series admittance `abs(y) >= susceptance_threshold` (PSS(e)'s
per-branch L2 rule, which catches a near-short hidden by anti-parallel
cancellation in the summed entry) or the combined off-diagonal `abs(Y[i,j]) >=
susceptance_threshold` (the numerically robust measure of the buses' actual
coupling). The from-bus survives unless the to-bus is in the user-supplied
irreducible set, in which case the sides flip.

# Fields
- `susceptance_threshold::Float64 = ZERO_IMPEDANCE_BRANCH_YBUS_SUSCEPTANCE_THRESHOLD`
- `minimum_retained_impedance::Float64 = ZERO_IMPEDANCE_X_EPSILON`: substitute
  reactance for branches with `r == x == 0` during Ybus assembly.
"""
@kwdef struct ZeroImpedanceBranchReduction <: NetworkReduction
    susceptance_threshold::Float64 = ZERO_IMPEDANCE_BRANCH_YBUS_SUSCEPTANCE_THRESHOLD
    minimum_retained_impedance::Float64 = ZERO_IMPEDANCE_X_EPSILON
end

get_susceptance_threshold(z::ZeroImpedanceBranchReduction) = z.susceptance_threshold
get_minimum_retained_impedance(z::ZeroImpedanceBranchReduction) =
    z.minimum_retained_impedance
