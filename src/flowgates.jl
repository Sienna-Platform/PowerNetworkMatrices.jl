"""
Result of [`flowgates`](@ref) for one ranked contingency.

A `NamedTuple` with fields:
- `contingency_branch::PSY.ACTransmission`: the outaged branch.
- `outage::PSY.FixedForcedOutage`: a fresh outage object for the contingency branch with its
  `monitored_components` set populated from `monitored`; attach it with
  `PSY.add_supplemental_attribute!(sys, contingency_branch, outage)` if desired.
- `monitored::Vector{PSY.ACTransmission}`: most-impacted branches, most impacted first.
- `impacts::Vector{Float64}`: signed rating-weighted impacts aligned with `monitored`
  (`LODF · rating_contingency / rating_monitored`, i.e. flow shift as a fraction of the
  monitored branch's rating with the outaged branch assumed at its rating).
- `score::Float64`: importance score in `[0, 1]`.
- `breadth::Float64`, `magnitude::Float64`: raw ranking features over the impacts.
"""
const FlowgateResult = @NamedTuple{
    contingency_branch::PSY.ACTransmission,
    outage::PSY.FixedForcedOutage,
    monitored::Vector{PSY.ACTransmission},
    impacts::Vector{Float64},
    score::Float64,
    breadth::Float64,
    magnitude::Float64,
}

"""
$(TYPEDSIGNATURES)
The outaged (contingency) branch of the flowgate `fg`.
"""
get_flowgate_contingency_branch(fg::FlowgateResult) = fg.contingency_branch

"""
$(TYPEDSIGNATURES)
The `FixedForcedOutage` built for the contingency branch of `fg`.
"""
get_flowgate_outage(fg::FlowgateResult) = fg.outage

"""
$(TYPEDSIGNATURES)
The monitored branches of `fg`, most impacted first.
"""
get_flowgate_monitored(fg::FlowgateResult) = fg.monitored

"""
$(TYPEDSIGNATURES)
The signed rating-weighted impacts of `fg`, aligned with `get_flowgate_monitored(fg)`.
"""
get_flowgate_impacts(fg::FlowgateResult) = fg.impacts

"""
$(TYPEDSIGNATURES)
The importance score of `fg` in `[0, 1]`.
"""
get_flowgate_score(fg::FlowgateResult) = fg.score

"""
$(TYPEDSIGNATURES)
The raw breadth feature of `fg`.
"""
get_flowgate_breadth(fg::FlowgateResult) = fg.breadth

"""
$(TYPEDSIGNATURES)
The raw magnitude feature of `fg`.
"""
get_flowgate_magnitude(fg::FlowgateResult) = fg.magnitude

# Per-contingency accumulators, indexed by contingency (row of the transposed parent).
# Buffer matrices are (top_n × n): column i is contingency i's bounded top-n monitored set.
# buf_min[i] tracks the smallest |value| in contingency i's full buffer (0.0 sentinel while
# the buffer still has empty slots), enabling O(1) rejection of sub-minimum candidates.
struct FlowgateAccumulator
    max_off::Vector{Float64}   # largest off-diagonal |LODF| (drives the meshed test)
    l1::Vector{Float64}        # Σ|v| over |v| > monitor_tol
    l2sq::Vector{Float64}      # Σ v²  over |v| > monitor_tol
    cnt::Vector{Int}           # count of |v| > monitor_tol
    buf_val::Matrix{Float64}   # signed v of buffered monitored entries
    buf_idx::Matrix{Int}       # monitored arc column index of buffered entries
    buf_cnt::Vector{Int}       # number of filled buffer slots per contingency
    buf_min::Vector{Float64}   # smallest |v| in each full buffer; 0.0 while not full
    top_n::Int
    monitor_tol::Float64
end

function FlowgateAccumulator(n::Int, top_n::Int, monitor_tol::Float64)
    return FlowgateAccumulator(
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Float64, n),
        zeros(Int, n),
        zeros(Float64, top_n, n),
        zeros(Int, top_n, n),
        zeros(Int, n),
        zeros(Float64, n),    # 0.0: buffer not yet full; fast-path rejects av <= 0
        top_n,
        monitor_tol,
    )
end

# Offer one candidate to contingency i's bounded top-n buffer. Strict `>` on replacement
# keeps the first-encountered entry on |v| ties, making the monitored order deterministic.
# buf_min[i] caches the current minimum |v|; when full and av <= buf_min[i] we reject in
# O(1); on replacement we rescan O(top_n) to refresh buf_min[i].
@inline function _offer!(acc::FlowgateAccumulator, i::Int, av::Float64, v::Float64, j::Int)
    k = acc.top_n
    c = acc.buf_cnt[i]
    if c < k
        c += 1
        @inbounds acc.buf_val[c, i] = v
        @inbounds acc.buf_idx[c, i] = j
        acc.buf_cnt[i] = c
        # Update buf_min only once the buffer is full.
        if c == k
            minval = abs(acc.buf_val[1, i])
            for p in 2:k
                candidate = abs(acc.buf_val[p, i])
                if candidate < minval
                    minval = candidate
                end
            end
            acc.buf_min[i] = minval
        end
        return nothing
    end
    # Buffer full: O(1) fast-path rejection.
    @inbounds minval = acc.buf_min[i]
    if av <= minval
        return nothing
    end
    # Find the slot holding the current minimum (first occurrence for tie determinism).
    minpos = 1
    @inbounds for p in 1:k
        if abs(acc.buf_val[p, i]) == minval
            minpos = p
            break
        end
    end
    @inbounds acc.buf_val[minpos, i] = v
    @inbounds acc.buf_idx[minpos, i] = j
    # Rescan to refresh buf_min after replacement.
    newmin = abs(acc.buf_val[1, i])
    for p in 2:k
        candidate = abs(acc.buf_val[p, i])
        if candidate < newmin
            newmin = candidate
        end
    end
    acc.buf_min[i] = newmin
    return nothing
end

# Fold one off-diagonal entry into the state. max_off tracks raw |LODF| (the
# rating-independent radial test); breadth/magnitude/monitored use the rating-weighted
# impact |LODF|·rating_c/rating_m. All ratings are validated finite and positive upstream,
# so impact is always finite and the monitor_tol check is the only filter. rc and inv_rm
# are pre-hoisted by the caller.
@inline function _update!(
    acc::FlowgateAccumulator,
    i::Int,
    j::Int,
    v::Float64,
    rc::Float64,
    inv_rm::Float64,
)
    av = abs(v)
    @inbounds if av > acc.max_off[i]
        acc.max_off[i] = av
    end
    impact = av * rc * inv_rm
    if impact > acc.monitor_tol
        @inbounds acc.l1[i] += impact
        @inbounds acc.l2sq[i] += impact * impact
        @inbounds acc.cnt[i] += 1
        signed_impact = v * rc * inv_rm
        _offer!(acc, i, impact, signed_impact, j)
    end
    return nothing
end

# Dense parent: column-major traversal for locality. Function barrier — the LODF `data`
# field is abstractly typed, so dispatch here recovers concrete-type performance.
# inv_rm is hoisted outside the i-loop (column-invariant); the diagonal is skipped via
# split inner loops rather than a branch inside the hot loop.
function _accumulate!(
    acc::FlowgateAccumulator,
    P::Matrix{Float64},
    rating::Vector{Float64},
    inv_rating::Vector{Float64},
)
    IS.@assert_op size(P, 1) == size(P, 2)
    IS.@assert_op size(P, 2) == length(acc.max_off)
    n = size(P, 2)
    @inbounds for j in 1:n
        inv_rm = inv_rating[j]
        for i in 1:(j - 1)
            _update!(acc, i, j, P[i, j], rating[i], inv_rm)
        end
        for i in (j + 1):n
            _update!(acc, i, j, P[i, j], rating[i], inv_rm)
        end
    end
    return nothing
end

# Sparse parent: iterate stored nonzeros. The structural diagonal is stored, so the
# i == j skip still applies; sparsified off-diagonals are simply absent.
# inv_rm is hoisted to the column level for the same aliasing reason as the dense path.
function _accumulate!(
    acc::FlowgateAccumulator,
    P::SparseArrays.SparseMatrixCSC{Float64, Int},
    rating::Vector{Float64},
    inv_rating::Vector{Float64},
)
    IS.@assert_op size(P, 1) == size(P, 2)
    IS.@assert_op size(P, 2) == length(acc.max_off)
    rows = SparseArrays.rowvals(P)
    vals = SparseArrays.nonzeros(P)
    n = size(P, 2)
    @inbounds for j in 1:n
        inv_rm = inv_rating[j]
        for k in SparseArrays.nzrange(P, j)
            i = rows[k]
            if i == j
                continue
            end
            _update!(acc, i, j, vals[k], rating[i], inv_rm)
        end
    end
    return nothing
end

# Rating (MVA, system-base pu) of a reduced-network branch, or nothing when undefined.
# Lines/MonitoredLines (mandatory rating) and GenericArcImpedance (max_flow) resolve directly;
# an unrated transformer is approximated from its base power inside get_equivalent_rating /
# get_impedance_averaged_rating (see _base_power_fallback_rating), so the fallback applies even
# to a transformer equivalenced into a parallel/series group. Parallel groups split flow by
# susceptance, so the DC-consistent equivalent is the impedance-averaged rating; series chains
# take the weakest link. The two equivalents are undefined only when the group is degenerate
# (zero or non-finite series susceptance).
_rating_value(b::PSY.ACTransmission) = get_equivalent_rating(b)
_rating_value(b::AbstractBranchesParallel) = get_impedance_averaged_rating(b)

# Rating (MVA) for an arc; nothing if the arc has no branch or the branch has no rating.
function _arc_rating(nr::NetworkReductionData, arc::Tuple{Int, Int})
    branch = get_arc_branch(nr, arc)
    isnothing(branch) && return nothing
    return _rating_value(branch)
end

# Resolve all arcs once, returning branches and ratings both aligned with get_arc_axis(lodf).
# Resolving each arc once avoids repeated map lookups in the results loop and _extract_monitored.
# Every arc on the axis must resolve to a branch with a finite, positive rating (transformers
# fall back to a base-power approximation above), so the returned vectors are fully concrete and
# every rating divide downstream is well defined. A branch whose rating is still undefined —
# a degenerate parallel/series equivalent — is a hard error rather than a silent skip.
function _build_arc_branches_and_ratings(lodf::LODF)
    nr = get_network_reduction_data(lodf)
    arcs = get_arc_axis(lodf)
    n = length(arcs)
    branches = Vector{PSY.ACTransmission}(undef, n)
    rating = Vector{Float64}(undef, n)
    for (k, arc) in enumerate(arcs)
        b = get_arc_branch(nr, arc)
        isnothing(b) &&
            error("flowgates: arc $arc on the LODF axis resolves to no branch")
        r = _rating_value(b)
        isnothing(r) &&
            error(
                "flowgates: arc $arc has no derivable rating (a degenerate parallel/series equivalent); every branch must be rated for flowgate analysis",
            )
        (isfinite(r) && r > 0.0) ||
            error("flowgates: arc $arc has a non-positive or non-finite rating ($r)")
        branches[k] = b
        rating[k] = r
    end
    return branches, rating
end

# Kept for backward-compat with tests that call _build_arc_ratings directly.
function _build_arc_ratings(lodf::LODF)
    _, rating = _build_arc_branches_and_ratings(lodf)
    return rating
end

# Min-max to [0, 1]. A single contingency or a constant feature (hi == lo) is degenerate;
# define it as 1.0 so the score stays well defined.
function _minmax_normalize(x::Vector{Float64})
    out = Vector{Float64}(undef, length(x))
    lo = minimum(x)
    hi = maximum(x)
    if hi <= lo
        fill!(out, 1.0)
        return out
    end
    span = hi - lo
    for k in eachindex(x)
        @inbounds out[k] = (x[k] - lo) / span
    end
    return out
end

# Materialize contingency i's monitored set as PSY branches, ranked by |impact| descending.
# MergeSort keeps the buffer's first-encountered order on ties for determinism.
# branches is pre-resolved and concrete (aligned with arcs), so no map lookups occur here.
function _extract_monitored(
    acc::FlowgateAccumulator,
    i::Int,
    branches::Vector{PSY.ACTransmission},
)
    c = acc.buf_cnt[i]
    col_view = view(acc.buf_val, 1:c, i)
    order = sortperm(col_view; by = abs, rev = true, alg = Base.Sort.MergeSort)
    monitored = Vector{PSY.ACTransmission}(undef, c)
    impacts = Vector{Float64}(undef, c)
    for (q, p) in enumerate(order)
        @inbounds idx = acc.buf_idx[p, i]
        @inbounds monitored[q] = branches[idx]
        @inbounds impacts[q] = acc.buf_val[p, i]   # signed impact
    end
    return monitored, impacts
end

"""
$(TYPEDSIGNATURES)

Derive ranked flowgates from a built `LODF`. For each meshed, rated contingency branch,
returns the branches most affected by its outage, ranked by **rating-weighted impact**:
the outaged branch is assumed loaded to its rating, so the flow shift onto a monitored
branch `m` is `LODF[m, c] · rating_c`, reported as a fraction of `m`'s own rating
(`impact = LODF[m, c] · rating_c / rating_m`). Results are sorted by score, most important
first, and returned as [`FlowgateResult`](@ref) NamedTuples.

Pure and `System`-free: ratings and branch components come from the LODF's
`network_reduction_data`. The structural radial/meshed test uses the raw LODF, so it is
unaffected by ratings; however, both the radial test and the breadth/magnitude computation
run on the stored matrix — for a tol-sparsified LODF, entries below the build tolerance are
absent, so `radial_tol` is only meaningful when it is ≥ the build tol and breadth/magnitude
likewise reflect only surviving entries. Restricted to `LODF` (dense or sparse);
`VirtualLODF` is unsupported.
Every arc needs a finite, positive rating. `Line`/`MonitoredLine` ratings are mandatory in
PowerSystems, but transformer ratings are optional (`Union{Nothing, Float64}`, or `0.0` for an
unset three-winding winding); an unrated transformer is approximated from its base power
(`rating ≈ base_power / system_base` pu, assuming 1.0 pu voltage) and emits a warning. A
parallel/series equivalent whose rating is still undefined (a degenerate group) is a hard
error. Contingencies whose entire monitored set falls at or below `monitor_tol` (impacts too
tiny to matter) are excluded from the results.

On a reduced network, an arc may resolve to an aggregated equivalent (parallel/series/
three-winding/Ward branch) rather than a single `System` component; attaching its `outage`
via `add_supplemental_attribute!` only applies to arcs backed by a real branch.

# Keywords
- `top_n::Int = 5`: maximum branches in each monitored set.
- `radial_tol::Float64 = 1e-5`: a branch is meshed if its largest off-diagonal raw `|LODF|`
  exceeds this; radial branches are excluded.
- `monitor_tol::Float64 = 0.0`: branches whose impact is at or below this are not monitored
  and do not count toward breadth/magnitude. The default drops exact zeros.
- `breadth_weight::Float64 = 0.5`: convex weight between breadth and magnitude in the score.
- `effective_count::Bool = true`: `true` uses the participation ratio of the impacts for
  breadth; `false` uses the literal count of monitored branches.
"""
function flowgates(
    lodf::LODF;
    top_n::Int = 5,
    radial_tol::Float64 = 1e-5,
    monitor_tol::Float64 = 0.0,
    breadth_weight::Float64 = 0.5,
    effective_count::Bool = true,
)
    IS.@assert_op top_n >= 1
    IS.@assert_op monitor_tol >= 0.0
    IS.@assert_op radial_tol >= 0.0
    IS.@assert_op breadth_weight >= 0.0
    IS.@assert_op breadth_weight <= 1.0

    arcs = get_arc_axis(lodf)
    n = length(arcs)
    branches, rating = _build_arc_branches_and_ratings(lodf)
    inv_rating = inv.(rating)   # every rating validated finite and positive above

    parent_matrix = parent(get_lodf_data(lodf))
    acc = FlowgateAccumulator(n, top_n, monitor_tol)
    _accumulate!(acc, parent_matrix, rating, inv_rating)

    # A contingency is kept when it is meshed (raw max off-diagonal |LODF| above radial_tol)
    # and has at least one monitored branch above monitor_tol; an empty monitored set carries
    # no information.
    meshed = Int[]
    for i in 1:n
        if acc.max_off[i] > radial_tol && !iszero(acc.buf_cnt[i])
            push!(meshed, i)
        end
    end
    if isempty(meshed)
        return FlowgateResult[]
    end

    breadth = Vector{Float64}(undef, length(meshed))
    magnitude = Vector{Float64}(undef, length(meshed))
    for (m, i) in enumerate(meshed)
        l2sq = acc.l2sq[i]
        if iszero(l2sq)
            breadth[m] = 0.0
            magnitude[m] = 0.0
        else
            if effective_count
                breadth[m] = (acc.l1[i] * acc.l1[i]) / l2sq
            else
                breadth[m] = Float64(acc.cnt[i])
            end
            magnitude[m] = sqrt(l2sq)
        end
    end

    breadth_norm = _minmax_normalize(breadth)
    magnitude_norm = _minmax_normalize(magnitude)

    # Compute scores into a concrete Float64 vector to avoid dynamic dispatch through
    # closures over boxed NamedTuples during the final sort.
    scores = Vector{Float64}(undef, length(meshed))
    for (m, _) in enumerate(meshed)
        scores[m] =
            breadth_weight * breadth_norm[m] + (1.0 - breadth_weight) * magnitude_norm[m]
    end

    # Sort by score descending; MergeSort is stable so ascending meshed index breaks ties.
    order = sortperm(scores; rev = true, alg = Base.Sort.MergeSort)

    results = Vector{FlowgateResult}(undef, length(meshed))
    for (q, m) in enumerate(order)
        i = meshed[m]
        monitored, impacts = _extract_monitored(acc, i, branches)
        contingency_branch = branches[i]
        outage =
            PSY.FixedForcedOutage(; outage_status = 1.0, monitored_components = monitored)
        results[q] = FlowgateResult((;
            contingency_branch = contingency_branch,
            outage = outage,
            monitored = monitored,
            impacts = impacts,
            score = scores[m],
            breadth = breadth[m],
            magnitude = magnitude[m],
        ))
    end

    return results
end
