# ZIBR `get_reduction` and helpers. Separated from the spec file so the
# struct can precede `ReductionContainer` in the include order while this
# dispatch (which needs `Ybus`) follows it.

_is_transformer(::PSY.TwoWindingTransformer) = true
_is_transformer(::PSY.ACTransmission) = false
_any_transformer(parallel_br::AbstractBranchesParallel) =
    any(_is_transformer(br) for br in parallel_br)

# Series admittance `Y_l = 1 / (r + x im)` of a non-transformer branch, with the
# `r == x == 0` -> `min_x_eps` substitution applied during Ybus assembly so the
# branch is judged against the admittance it contributed. Computed directly from
# `(r, x)` rather than via `ybus_branch_entries`, which would re-emit that
# function's `r == x == 0` warning (assembly already emitted it once) and rebuild
# the full 2x2.
function _series_admittance(br, min_x_eps::Float64)
    r = PSY.get_r(br)
    x = PSY.get_x(br)
    if r == 0.0 && x == 0.0
        x = min_x_eps
    end
    return inv(complex(r, x))
end

# An arc is treated as zero-impedance when *either* some individual non-transformer
# branch on it clears the susceptance threshold (PSS(e)'s per-branch L2 rule: a
# near-short such as 1e4im directly ties the two buses even if an anti-parallel
# member cancels it in the summed entry), *or* the combined off-diagonal entry
# clears it (the numerically robust measure of the actual bus-to-bus coupling).
# The union is a superset of either test alone; the two terms only differ for
# parallel groups (for a single branch the combined entry equals the branch).
function _is_zero_impedance_arc(
    br::PSY.ACTransmission,
    susceptance_threshold::Float64,
    min_x_eps::Float64,
)
    _is_transformer(br) && return false
    return abs(_series_admittance(br, min_x_eps)) >= susceptance_threshold
end

function _is_zero_impedance_arc(
    parallel_br::AbstractBranchesParallel,
    susceptance_threshold::Float64,
    min_x_eps::Float64,
)
    # Transformer-bearing arcs are excluded from zero-impedance bus merging.
    _any_transformer(parallel_br) && return false
    any_branch = any(
        abs(_series_admittance(br, min_x_eps)) >= susceptance_threshold
        for br in parallel_br
    )
    # Non-transformer members are symmetric, so the group's Ybus off-diagonal is
    # `-sum(Y_l)` regardless of member orientation. Summing `Y_l` directly (with the
    # same `min_x_eps`) keeps the combined-term decision consistent with how the
    # matrix was assembled, which `ybus_branch_entries(group, nr)` does not since it
    # ignores `min_x_eps`.
    combined =
        abs(sum(_series_admittance(br, min_x_eps) for br in parallel_br)) >=
        susceptance_threshold
    return any_branch || combined
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
    # ZIBR is the first reduction applied, so only the direct and parallel branch maps
    # are populated; series/3W arcs do not exist yet (3W arcs are transformers anyway).
    # Iterating the maps gives the merge granularity directly (one entry per arc) and
    # lets us examine each branch, matching PSS(e), instead of the summed Ybus entry.
    for branch_map in (get_direct_branch_map(nrd), get_parallel_branch_map(nrd))
        for (arc_key, br) in branch_map
            _is_zero_impedance_arc(br, susceptance_threshold, min_x_eps) || continue
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
