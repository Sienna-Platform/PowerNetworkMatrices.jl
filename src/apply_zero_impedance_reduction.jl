# ZIBR `get_reduction` and helpers. Separated from the spec file so the
# struct can precede `ReductionContainer` in the include order while this
# dispatch (which needs `Ybus`) follows it.

_is_transformer(::PSY.TwoWindingTransformer) = true
_is_transformer(::PSY.ACTransmission) = false
_any_transformer(parallel_br::AbstractBranchesParallel) =
    any(_is_transformer(br) for br in parallel_br)

# Series admittance `Y_l = 1 / (r + x im)` from a branch's `(r, x)`, with the
# `r == x == 0` -> `min_x_eps` substitution applied during Ybus assembly so the
# branch is judged against the admittance it contributed. Computed directly from
# `(r, x)` rather than via `ybus_branch_entries`, which would re-emit that
# function's `r == x == 0` warning (assembly already emitted it once) and rebuild
# the full 2x2.
function _series_admittance(r::Float64, x::Float64, min_x_eps::Float64)
    if iszero(r) && iszero(x)
        x = min_x_eps
    end
    return inv(complex(r, x))
end

# True for a branch with near-zero resistance (`abs(r) <= resistance_tolerance`, default `0.0` ⇒
# exact `r == 0`) whose series admittance reaches the threshold (`|y| >= susceptance_threshold`).
# Reads `(r, x)` once and bails on a too-large resistance before touching `x`.
function _is_zero_impedance_branch(
    br,
    susceptance_threshold::Float64,
    min_x_eps::Float64,
    resistance_tolerance::Float64,
)
    r = PSY.get_r(br, PSY.SU)
    abs(r) <= resistance_tolerance || return false
    x = PSY.get_x(br, PSY.SU)
    return abs(_series_admittance(r, x, min_x_eps)) >= susceptance_threshold
end

# An arc is zero-impedance iff some individual non-transformer branch on it qualifies; the
# parallel combination is never considered. So an `r ≈ 0` jumper in parallel with a normal
# line merges the buses (the jumper qualifies on its own), while branches that only
# collectively exceed the threshold do not merge.
function _is_zero_impedance_arc(
    br::PSY.ACTransmission,
    susceptance_threshold::Float64,
    min_x_eps::Float64,
    resistance_tolerance::Float64,
)
    _is_transformer(br) && return false
    return _is_zero_impedance_branch(
        br, susceptance_threshold, min_x_eps, resistance_tolerance)
end

function _is_zero_impedance_arc(
    parallel_br::AbstractBranchesParallel,
    susceptance_threshold::Float64,
    min_x_eps::Float64,
    resistance_tolerance::Float64,
)
    # Transformer-bearing arcs are excluded from zero-impedance bus merging.
    _any_transformer(parallel_br) && return false
    return any(
        _is_zero_impedance_branch(br, susceptance_threshold, min_x_eps, resistance_tolerance)
        for br in parallel_br
    )
end

function get_reduction(
    ybus::Ybus,
    sys::PSY.System,
    reduction::ZeroImpedanceBranchReduction,
)
    nr = NetworkReductionData()
    nrd = get_network_reduction_data(ybus)
    user_irreducible = get_user_irreducible_buses(get_reductions(nrd))
    susceptance_threshold = get_susceptance_threshold(reduction)
    # Match the substitute reactance used for `r == x == 0` branches during assembly,
    # so a branch's merge eligibility is judged against the admittance it contributed.
    min_x_eps = get_minimum_retained_impedance(reduction)
    resistance_tolerance = get_resistance_tolerance(reduction)
    # ZIBR is the first reduction applied, so only the direct and parallel branch maps
    # are populated; series/3W arcs do not exist yet (3W arcs are transformers anyway).
    # Iterating the maps gives the merge granularity directly (one entry per arc) and
    # lets us examine each branch, matching PSS(e), instead of the summed Ybus entry.
    for branch_map in (get_direct_branch_map(nrd), get_parallel_branch_map(nrd))
        for (arc_key, br) in branch_map
            _is_zero_impedance_arc(
                br, susceptance_threshold, min_x_eps, resistance_tolerance) || continue
            from_no, to_no = arc_key
            from_irred = from_no ∈ user_irreducible
            to_irred = to_no ∈ user_irreducible
            if from_irred && to_irred
                @warn "Zero-impedance branch between two irreducible buses $from_no and $to_no; skipping merge."
                continue
            elseif to_irred
                # Flip so the irreducible bus survives.
                from_no, to_no = to_no, from_no
            end

            _update_bus_maps!(
                nr.reverse_bus_search_map,
                nr.bus_reduction_map,
                to_no,
                from_no,
            )
            push!(nr.removed_arcs, arc_key)
        end
    end
    nr.merged_bus_pairs = copy(nr.reverse_bus_search_map)
    return nr
end
