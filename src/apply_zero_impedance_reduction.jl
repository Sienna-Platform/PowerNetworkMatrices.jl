# ZIBR `get_reduction` and helpers. Separated from the spec file so the
# struct can precede `ReductionContainer` in the include order while this
# dispatch (which needs `Ybus`) follows it.

_is_transformer(::PSY.TwoWindingTransformer) = true
# Three-winding transformer windings are one-to-one arcs in `direct_branch_map`; they are
# transformer arcs and must be excluded from zero-impedance bus merging like any transformer.
_is_transformer(::ThreeWindingTransformerCircuit) = true
_is_transformer(::PSY.ACTransmission) = false
# Without this, aggregates hit the blanket `false`; recursive for nesting.
_is_transformer(seg::AbstractReductionAggregate) = any(_is_transformer, seg)

# Series admittance `Y_l = 1 / (r + x im)` from a branch's `(r, x)`, with the
# `r == x == 0` -> `min_x_eps` substitution applied during Ybus assembly so the
# branch is judged against the admittance it contributed. Computed directly from
# `(r, x)` rather than via `ybus_branch_entries`, which would re-emit that
# function's `r == x == 0` warning (assembly already emitted it once) and rebuild
# the full 2x2.
function _series_admittance(r::Float64, x::Float64, min_x_eps::Float64)
    return inv(complex(r, _retained_x(r, x, min_x_eps)))
end

# True for a branch with near-zero resistance (`abs(r) <= resistance_tolerance`, default `0.0` ⇒
# exact `r == 0`) whose series admittance reaches the threshold (`|y| >= susceptance_threshold`).
# Reads `(r, x)` once and bails on a too-large resistance before touching `x`.
function _is_zero_impedance_branch(
    br::PSY.ACTransmission,
    susceptance_threshold::Float64,
    min_x_eps::Float64,
    resistance_tolerance::Float64,
)
    r = PSY.get_r(br, u"SU")
    abs(r) <= resistance_tolerance || return false
    x = PSY.get_x(br, u"SU")
    return abs(_series_admittance(r, x, min_x_eps)) >= susceptance_threshold
end

# An aggregate has no `(r, x)` of its own, and merging the endpoints of a composite arc would
# discard its interior. ZIBR runs before every other reduction, so nothing routes one here.
_is_zero_impedance_branch(
    seg::AbstractReductionAggregate,
    ::Float64,
    ::Float64,
    ::Float64,
) = error(
    "Zero-impedance eligibility is a per-branch property, but $(get_name(seg)) is a " *
    "reduction aggregate; judge its members individually.",
)

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
    _is_transformer(parallel_br) && return false
    return any(
        _is_zero_impedance_branch(
            br,
            susceptance_threshold,
            min_x_eps,
            resistance_tolerance,
        )
        for br in parallel_br
    )
end

# The power flow types a merged bus from its survivor alone, so a merge keeps the end
# that controls the most: REF over PV over PQ.
_bus_type_rank(bustype::Union{Nothing, ACBusTypes.Value}) =
    if bustype == ACBusTypes.REF || bustype == ACBusTypes.SLACK
        3
    elseif bustype == ACBusTypes.PV
        2
    else
        1
    end

# A skip caused by schedule pins is expected: the scheduled branch must stay an arc so it can
# be switched. Anything else means user-supplied pins blocked a merge, which deserves a warning.
function _log_skipped_zero_impedance_merge(
    from_no::Int,
    to_no::Int,
    schedule_pinned::Set{Int},
)
    if from_no in schedule_pinned && to_no in schedule_pinned
        @debug "Zero-impedance branch between schedule-pinned buses $from_no and $to_no kept as an arc."
    else
        @warn "Zero-impedance branch between two irreducible buses $from_no and $to_no; skipping merge."
    end
    return
end

function get_reduction(ybus::Ybus, sys::PSY.System, reduction::ZeroImpedanceBranchReduction)
    nr = NetworkReductionData()
    nrd = get_network_reduction_data(ybus)
    exempt = get_reduction_exempt_buses(get_reductions(nrd))
    schedule_pinned = get_schedule_pinned_buses(get_reductions(nrd))
    bus_type_rank = Dict{Int, Int}(
        PSY.get_number(b) => _bus_type_rank(PSY.get_bustype(b)) for
        b in PSY.get_components(PSY.ACBus, sys)
    )
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
            from_no = get(nr.reverse_bus_search_map, arc_key[1], arc_key[1])
            to_no = get(nr.reverse_bus_search_map, arc_key[2], arc_key[2])
            from_irred = from_no ∈ exempt
            to_irred = to_no ∈ exempt
            if from_no == to_no
                # Both endpoints already resolve into the same merged group: the arc is a
                # self-loop, not a skipped merge, so it still has to be dropped.
                push!(nr.removed_arcs, arc_key)
                continue
            elseif from_irred && to_irred
                _log_skipped_zero_impedance_merge(from_no, to_no, schedule_pinned)
                continue
            elseif to_irred ||
                   (!from_irred && bus_type_rank[to_no] > bus_type_rank[from_no])
                # Flip so the irreducible bus, else the stronger bus type, survives. Both
                # ends are merged-group roots, so a chain of merges ends on its strongest bus.
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
