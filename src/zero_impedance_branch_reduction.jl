"""
    ZeroImpedanceBranchReduction <: NetworkReduction

Merges buses connected by zero-impedance non-transformer branches. Always
applied as the first step of `Ybus(sys; ...)` to avoid singular admittances.
`Ybus` takes it through the `zero_impedance_reduction` kwarg and rejects one in
`network_reductions`; downstream wrappers accept it in their reduction vectors
and route it with [`split_zero_impedance_reduction`](@ref).

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

_is_zero_impedance_reduction(::NetworkReduction) = false
_is_zero_impedance_reduction(::ZeroImpedanceBranchReduction) = true

"""
    split_zero_impedance_reduction(reductions::Vector{NetworkReduction})

Return `(others, zero_impedance)`: the reductions `Ybus` accepts in `network_reductions`, in
their original order, and the single `ZeroImpedanceBranchReduction` it takes through its own
kwarg — defaulted when none is supplied. Errors on more than one.
"""
function split_zero_impedance_reduction(reductions::Vector{NetworkReduction})
    zero_impedance = ZeroImpedanceBranchReduction[
        r for r in reductions if _is_zero_impedance_reduction(r)
    ]
    if length(zero_impedance) > 1
        throw(
            IS.ConflictingInputsError(
                "Received $(length(zero_impedance)) ZeroImpedanceBranchReduction \
                entries; only one may be passed.",
            ),
        )
    end
    others = filter(!_is_zero_impedance_reduction, reductions)
    if isempty(zero_impedance)
        return others, ZeroImpedanceBranchReduction()
    end
    return others, only(zero_impedance)
end
