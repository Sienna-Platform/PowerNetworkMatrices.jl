"""
    Winding

The circuit of a three-winding transformer. `PRIMARY` (1) is the circuit from the primary
terminal to the star bus, `SECONDARY` (2) from the secondary terminal, and `TERTIARY` (3)
from the tertiary terminal. A branch with two terminals has no winding.
"""
@enum Winding PRIMARY = 1 SECONDARY = 2 TERTIARY = 3

function _branch_key(rows::Dict{Tuple{Int, Int}, <:Any}, id::Int, missing_text::String)
    haskey(rows, (id, 0)) && return (id, 0)
    if any(w -> haskey(rows, (id, w)), 1:3)
        error(
            "The id $id is a three-winding transformer. Pass a Winding: " *
            "PRIMARY, SECONDARY or TERTIARY.",
        )
    end
    error(missing_text)
end

"""
    BranchArcTable

Columns of the `branch_map` group of a factor file: one row per node of the reduction tree
of each arc, in depth-first preorder. `n_arc` is the number of arcs. Read it with
`read_branch_map(path)`. Split arc flows with it in [`branch_flows`](@ref).

# Columns
- `kind`: 1 to 5 for a branch, 6 to 8 for winding 1 to 3 of a three-winding transformer,
  9 for a parallel group, 10 for a series chain, 11 for a Ward equivalent. Kinds 1 to 8
  are leaf rows.
- `id`: the id of the branch of a leaf row. It is 0 for kinds 9 to 11.
- `parent`: above 0, the row of the parent aggregate, always below the own row. Below 0, the
  row is a top node of arc `-parent`.
- `sign`: +1 or -1, the orientation of the row against its parent.
- `b`: the DC susceptance of a leaf or Ward row, in system base. It is 0.0 for kinds 9 and 10.
- `alpha`: the phase shift of a leaf row, in radians. It is 0.0 for kinds 9 to 11.
"""
struct BranchArcTable
    kind::Vector{Int}
    id::Vector{Int}
    parent::Vector{Int}
    sign::Vector{Int}
    b::Vector{Float64}
    alpha::Vector{Float64}
    n_arc::Int
end

"Return the `kind` column of `t`: the node kind of each row."
get_branch_kind(t::BranchArcTable) = t.kind

"Return the `id` column of `t`: the id of the branch of each row, or 0."
get_branch_id(t::BranchArcTable) = t.id

"Return the `parent` column of `t`: the parent row, or minus the arc index for a top node."
get_branch_parent(t::BranchArcTable) = t.parent

"Return the `sign` column of `t`: the orientation of each row against its parent."
get_branch_sign(t::BranchArcTable) = t.sign

"Return the `b` column of `t`: the DC susceptance of each leaf or Ward row, in system base."
get_branch_susceptance(t::BranchArcTable) = t.b

"Return the `alpha` column of `t`: the phase shift of each leaf row, in radians."
get_branch_alpha(t::BranchArcTable) = t.alpha

"Return the number of arcs of `t`."
get_branch_arc_count(t::BranchArcTable) = t.n_arc

# A magnitude outside this range makes the series and parallel sums overflow or vanish.
_in_susceptance_range(b::Float64) = 1 / MAX_ARC_SUSCEPTANCE <= abs(b) <= MAX_ARC_SUSCEPTANCE

function _check_branch_map(t::BranchArcTable)
    (; kind, id, parent, sign, b, alpha, n_arc) = t
    _check_same_length(
        "branch_map",
        "kind" => kind, "id" => id, "parent" => parent,
        "sign" => sign, "b" => b, "alpha" => alpha,
    )
    n = length(kind)
    has_child = falses(n)
    has_top = falses(n_arc)
    leaf_keys = Set{Tuple{Int, Int}}()
    for r in 1:n
        k = kind[r]
        p = parent[r]
        fail(what) = error("Factor file table branch_map row $r (kind $k): $what.")
        1 <= k <= 11 || fail("kind is not in 1:11")
        iszero(p) && fail("parent is 0")
        p >= -n_arc || fail("parent $p is not an arc of the $n_arc arcs")
        p < r || fail("parent $p is not below the row")
        if p > 0
            (kind[p] == 9 || kind[p] == 10) ||
                fail("parent row $p has kind $(kind[p]), not an aggregate")
            has_child[p] = true
        else
            has_top[-p] = true
        end
        (k == 11 && p > 0) && fail("a Ward equivalent must be a top node")
        (sign[r] == 1 || sign[r] == -1) || fail("sign is not 1 or -1")
        if k <= 8
            id[r] > 0 || fail("id is not positive")
            # Exact zero is BA's "no DC coupling", for example a purely resistive branch. BA
            # uses component values inside a series chain, which are never zero.
            in_chain = p > 0 && kind[p] == 10
            ((iszero(b[r]) && !in_chain) || _in_susceptance_range(b[r])) ||
                fail(
                    "b is $(b[r]); expected 0 outside a series chain or a magnitude from $(1 / MAX_ARC_SUSCEPTANCE) to $MAX_ARC_SUSCEPTANCE",
                )
            isfinite(alpha[r]) || fail("alpha is $(alpha[r]); expected a finite value")
            key = (id[r], max(k - 5, 0))
            key in leaf_keys && fail("id and winding $key repeat")
            push!(leaf_keys, key)
        elseif k == 11
            iszero(id[r]) || fail("id is not 0")
            # A Ward row is a top node, so no flow inverts its b: only the upper bound applies.
            (isfinite(b[r]) && abs(b[r]) <= MAX_ARC_SUSCEPTANCE) ||
                fail(
                    "b is $(b[r]); expected a finite magnitude of at most $MAX_ARC_SUSCEPTANCE",
                )
            iszero(alpha[r]) || fail("alpha is not 0")
        else
            iszero(id[r]) || fail("id is not 0")
            iszero(b[r]) || fail("b is not 0")
            iszero(alpha[r]) || fail("alpha is not 0")
        end
    end
    for r in 1:n
        (kind[r] == 9 || kind[r] == 10) && !has_child[r] &&
            error("Factor file table branch_map row $r (kind $(kind[r])): no child.")
    end
    for k in 1:n_arc
        has_top[k] || error("Factor file table branch_map: arc $k has no top node.")
    end
    return
end

function _write_branch_map!(f::HDF5.File, table::BranchArcTable)
    g = HDF5.create_group(f, "branch_map")
    g["kind"] = table.kind
    g["id"] = table.id
    g["parent"] = table.parent
    g["sign"] = table.sign
    g["b"] = table.b
    g["alpha"] = table.alpha
    return
end

"""
    read_branch_map(path::AbstractString) -> BranchArcTable

Read the group `branch_map` of a factor file. The function does not factorize the network.
It throws an error if the format version is not 2, the group is missing, or a column breaks
a rule of the table.
"""
function read_branch_map(path::AbstractString)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        arc_from = _read(f, "arc_from", Vector{Int}, budget)
        arc_to = _read(f, "arc_to", Vector{Int}, budget)
        _check_same_length("arcs", "arc_from" => arc_from, "arc_to" => arc_to)
        g = _group(f, "branch_map")
        kind = _read(g, "kind", Vector{Int}, budget)
        id = _read(g, "id", Vector{Int}, budget)
        parent = _read(g, "parent", Vector{Int}, budget)
        sign = _read(g, "sign", Vector{Int}, budget)
        b = _read(g, "b", Vector{Float64}, budget)
        alpha = _read(g, "alpha", Vector{Float64}, budget)
        n_arc = length(arc_from)
        table = BranchArcTable(kind, id, parent, sign, b, alpha, n_arc)
        _check_branch_map(table)
        return table
    end
end

"""
    branch_flows(table::BranchArcTable, arc_flows::Vector{Float64}, outaged_rows::Vector{Int} = Int[]) -> Vector{Float64}

Split the arc flows into one flow for each row of `table`. A flow is in system base per
unit, positive from the first to the second bus of the frame of the row. A row in
`outaged_rows` carries no flow, and neither do the rows below an outaged parallel group.
A parallel group without a live member carries no flow. A series chain with an outaged
segment carries no flow, and its live segments carry none. A live parallel group in such a
chain still splits a circulating flow between its members when a member has a phase shift.
The function throws an error if `arc_flows` does not have one value per arc, a row of
`outaged_rows` is out of range, or a series sum is zero. A row with `b = 0` has no DC
coupling and carries no flow. The function also throws an error if a parallel or arc sum is
zero while the flow to split is not zero.
"""
function branch_flows(
    table::BranchArcTable,
    arc_flows::Vector{Float64},
    outaged_rows::Vector{Int} = Int[],
)
    n = length(table.kind)
    n_arc = table.n_arc
    length(arc_flows) == n_arc ||
        error("branch_flows: arc_flows has $(length(arc_flows)) values; expected $n_arc.")
    dead = falses(n)
    for r in outaged_rows
        1 <= r <= n ||
            error("branch_flows: outaged row $r is not in 1:$n.")
        dead[r] = true
    end
    outaged = copy(dead)
    kind = table.kind
    parent = table.parent
    sign = table.sign
    b = copy(table.b)
    alpha = copy(table.alpha)
    sb = zeros(n)
    ss = zeros(n)
    sr = zeros(n)
    sa = zeros(n)
    live_children = zeros(Int, n)
    dead_children = zeros(Int, n)
    arc_sb = zeros(n_arc)
    arc_ss = zeros(n_arc)
    for r in n:-1:1
        if dead[r]
            # An outaged row carries no flow, so no sum is checked.
        elseif kind[r] == 9
            if iszero(live_children[r])
                dead[r] = true
            elseif iszero(sb[r])
                # No member has DC coupling, so the group has none.
                b[r] = 0.0
                alpha[r] = 0.0
            else
                b[r] = sb[r]
                alpha[r] = ss[r] / sb[r]
            end
        elseif kind[r] == 10
            if !iszero(dead_children[r])
                dead[r] = true
            else
                iszero(sr[r]) &&
                    error("branch_flows: row $r (series chain) has a zero impedance sum.")
                b[r] = 1 / sr[r]
                alpha[r] = sa[r]
            end
        end
        p = parent[r]
        if p > 0 && kind[p] == 10
            if dead[r]
                dead_children[p] += 1
            else
                sr[p] += 1 / b[r]
                sa[p] += sign[r] * alpha[r]
            end
        elseif !dead[r]
            if p < 0
                arc_sb[-p] += b[r]
                arc_ss[-p] += b[r] * sign[r] * alpha[r]
            else
                live_children[p] += 1
                sb[p] += b[r]
                ss[p] += b[r] * sign[r] * alpha[r]
            end
        end
    end
    flows = zeros(n)
    inherited = falses(n)
    for r in 1:n
        p = parent[r]
        # A group that is dead only because no member is live does not stop a loop flow
        # in a live group below it.
        if p > 0 && (inherited[p] || (outaged[p] && kind[p] == 9))
            dead[r] = true
            inherited[r] = true
        end
        dead[r] && continue
        if p < 0
            flows[r] = _split_flow(
                r,
                sign[r],
                b[r],
                alpha[r],
                arc_sb[-p],
                arc_flows[-p] + arc_ss[-p],
            )
        elseif kind[p] == 9
            flows[r] = _split_flow(r, sign[r], b[r], alpha[r], sb[p], flows[p] + ss[p])
        else
            flows[r] = sign[r] * flows[p]
        end
        isfinite(flows[r]) || error(
            "branch_flows: the flow of row $r (kind $(kind[r])) is $(flows[r]); " *
            "an arc flow or a susceptance is out of range.",
        )
    end
    return flows
end

# A zero susceptance sum is BA's "no DC coupling": the rows carry no flow, and the flow to
# split must be zero.
function _split_flow(
    r::Int,
    sign::Int,
    b::Float64,
    alpha::Float64,
    b_sum::Float64,
    flow::Float64,
)
    if iszero(b_sum)
        iszero(flow) || error(
            "branch_flows: row $r has no DC coupling (zero susceptance sum) but its parent carries the flow $flow.",
        )
        return 0.0
    end
    return sign * (b / b_sum) * flow - b * alpha
end
