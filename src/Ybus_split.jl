const TransformerRecord = NamedTuple{
    (:from, :to, :turns, :angle, :Y, :shunt),
    Tuple{Int, Int, Float64, Float64, ComplexF64, ComplexF64},
}

struct YbusSplit{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{ComplexF64}
    y_series::SparseArrays.SparseMatrixCSC{ComplexF64, Int}
    y_shunt::Vector{ComplexF64}
    y_lambda::SparseArrays.SparseMatrixCSC{ComplexF64, Int}
    adjacency_data::SparseArrays.SparseMatrixCSC{Int8, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    arc_subnetwork_axis::Dict{Int, Vector{Tuple{Int, Int}}}
    network_reduction_data::NetworkReductionData
    # need record of transformer data: (from, to, turns, angle, Y, shunt) for each.
    # store separately (6 arrays), or together (one array of 6-tuples)?
    transformers::Vector{TransformerRecord}
end

get_y_series(M::YbusSplit) = M.y_series
get_y_shunt(M::YbusSplit) = M.y_shunt
get_axes(M::YbusSplit) = M.axes
get_lookup(M::YbusSplit) = M.lookup
get_ref_bus(M::YbusSplit) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::YbusSplit) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::YbusSplit) = M.network_reduction_data
get_bus_axis(M::YbusSplit) = M.axes[1]
get_bus_lookup(M::YbusSplit) = M.lookup[1]

# --- Branch entry routines (series-only + separate shunts) ---

"""Series-only Ybus entries for a `Line` or `DiscreteControlledACBranch`."""
function ybus_split_branch_entries(br::PSY.ACTransmission)
    Y_l = (1 / (PSY.get_r(br) + PSY.get_x(br) * 1im))
    Y11 = Y_l
    if !isfinite(Y11) || !isfinite(Y_l)
        error(
            "Data in $(PSY.get_name(br)) is incorrect. r = $(PSY.get_r(br)), x = $(PSY.get_x(br))",
        )
    end
    Y12 = -Y_l
    Y21 = Y12
    Y22 = Y_l
    return (Y11, Y12, Y21, Y22, _get_shunt(br, :from), _get_shunt(br, :to))
end

# this covers phase shifting, tap transformers, and plain 2W (fixed winding number)
"""Structurally nonzero but numerically zero entries for two-winding transformers."""
function ybus_split_branch_entries(::PSY.TwoWindingTransformer)
    z = zero(ComplexF64)
    return (z, z, z, z, z, z)
end

function ybus_split_branch_entries(parallel_br::BranchesParallel)
    Y11 = Y12 = Y21 = Y22 = Yshunt_from = Yshunt_to = zero(ComplexF64)
    for br in parallel_br
        # All branches in BranchesParallel have the same orientation when constructed in add_to_branch_maps!
        (y11, y12, y21, y22, yshunt_from, yshunt_to) = ybus_split_branch_entries(br)
        Y11 += y11
        Y12 += y12
        Y21 += y21
        Y22 += y22
        Yshunt_from += yshunt_from
        Yshunt_to += yshunt_to
    end
    return (Y11, Y12, Y21, Y22, Yshunt_from, Yshunt_to)
end

function ybus_split_branch_entries(::BranchesSeries)
    error(
        "Equivalent Circuit Formulation techniques aren't compatible with degree 2 reductions.",
    )
end

# --- Storing branch entries into COO vectors ---

function add_branch_entries_to_ybus_split!(
    y11::Vector{ComplexF64},
    y12::Vector{ComplexF64},
    y21::Vector{ComplexF64},
    y22::Vector{ComplexF64},
    branch_ix::Int,
    y_shunt_from::Vector{ComplexF64},
    y_shunt_to::Vector{ComplexF64},
    br::PSY.ACTransmission,
)
    Y11, Y12, Y21, Y22, Yshunt_from, Yshunt_to = ybus_split_branch_entries(br)
    y11[branch_ix] = Y11
    y12[branch_ix] = Y12
    y21[branch_ix] = Y21
    y22[branch_ix] = Y22
    y_shunt_from[branch_ix] = Yshunt_from
    y_shunt_to[branch_ix] = Yshunt_to
    return
end

# --- Per-branch dispatch (indexing + admittance) ---

function _add_to_transformer_records!(
    transformer_records::Vector{TransformerRecord},
    comp::PSY.TwoWindingTransformer,
    from::Int,
    to::Int,
)
    push!(
        transformer_records,
        (
            from = from,
            to = to,
            turns = _get_tap(comp),
            angle = PSY.get_α(comp),
            Y = 1 / (PSY.get_r(comp) + PSY.get_x(comp) * 1im),
            shunt = PSY.get_primary_shunt(comp),
        ),
    )
    return
end

_add_to_transformer_records!(
    ::Vector{TransformerRecord},
    ::PSY.ACTransmission,
    ::Int,
    ::Int,
) = nothing

function _ybus_split!(
    y11::Vector{ComplexF64},
    y12::Vector{ComplexF64},
    y21::Vector{ComplexF64},
    y22::Vector{ComplexF64},
    y_shunt_from::Vector{ComplexF64},
    y_shunt_to::Vector{ComplexF64},
    br::PSY.ACTransmission,
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    nr::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    transformer_records::Vector{TransformerRecord},
)
    add_branch_entries_to_indexing_maps!(num_bus, branch_ix, nr, fb, tb, adj, br)
    add_branch_entries_to_ybus_split!(
        y11, y12, y21, y22, branch_ix, y_shunt_from, y_shunt_to, br,
    )
    _add_to_transformer_records!(transformer_records, br, fb[branch_ix], tb[branch_ix])
    return
end

function _ybus_split!(
    y11::Vector{ComplexF64},
    y12::Vector{ComplexF64},
    y21::Vector{ComplexF64},
    y22::Vector{ComplexF64},
    y_shunt_from::Vector{ComplexF64},
    y_shunt_to::Vector{ComplexF64},
    br::PSY.DynamicBranch,
    num_bus::Dict{Int, Int},
    branch_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    nr::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    transformer_records::Vector{TransformerRecord},
)
    _ybus_split!(
        y11, y12, y21, y22, y_shunt_from, y_shunt_to,
        br.branch, num_bus, branch_ix, fb, tb, nr, adj,
        transformer_records,
    )
    return
end

function _ybus_split!(
    br::PSY.ThreeWindingTransformer,
    num_bus::Dict{Int, Int},
    offset_ix::Int,
    fb::Vector{Int},
    tb::Vector{Int},
    ix::Int,
    nr::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    transformer_records::Vector{TransformerRecord},
)
    primary_star_arc = PSY.get_primary_star_arc(br)
    secondary_star_arc = PSY.get_secondary_star_arc(br)
    tertiary_star_arc = PSY.get_tertiary_star_arc(br)
    add_to_branch_maps!(nr, primary_star_arc, secondary_star_arc, tertiary_star_arc, br)
    # All admittance values are zero for 3W transformers in the split formulation;
    # only the structural entries (fb/tb) and adjacency need to be set.
    n_entries = 0
    all_windings_available =
        PSY.get_available_primary(br) && PSY.get_available_secondary(br) &&
        PSY.get_available_tertiary(br)
    all_windings_available || error(
        "YbusSplit doesn't support 3WT's with unavailable windings. Found: $(PSY.get_name(br))",
    )
    # primary
    primary_ix, star_ix = get_bus_indices(primary_star_arc, num_bus, nr)
    adj[primary_ix, star_ix] = 1
    adj[star_ix, primary_ix] = -1
    entry_ix = offset_ix + ix + n_entries
    fb[entry_ix] = primary_ix
    tb[entry_ix] = star_ix
    n_entries += 1
    push!(
        transformer_records,
        (
            from = primary_ix,
            to = star_ix,
            turns = PSY.get_primary_turns_ratio(br),
            angle = PSY.get_α_primary(br),
            Y = 1 / (PSY.get_r_primary(br) + PSY.get_x_primary(br) * 1im),
            shunt = PSY.get_g(br) + im * PSY.get_b(br),
        ),
    )
    # secondary
    secondary_ix, star_ix = get_bus_indices(secondary_star_arc, num_bus, nr)
    adj[secondary_ix, star_ix] = 1
    adj[star_ix, secondary_ix] = -1
    entry_ix = offset_ix + ix + n_entries
    fb[entry_ix] = secondary_ix
    tb[entry_ix] = star_ix
    n_entries += 1
    push!(
        transformer_records,
        (
            from = secondary_ix,
            to = star_ix,
            turns = PSY.get_secondary_turns_ratio(br),
            angle = PSY.get_α_secondary(br),
            Y = 1 / (PSY.get_r_secondary(br) + PSY.get_x_secondary(br) * 1im),
            shunt = zero(ComplexF64),
        ),
    )
    # tertiary
    tertiary_ix, star_ix = get_bus_indices(tertiary_star_arc, num_bus, nr)
    adj[tertiary_ix, star_ix] = 1
    adj[star_ix, tertiary_ix] = -1
    entry_ix = offset_ix + ix + n_entries
    fb[entry_ix] = tertiary_ix
    tb[entry_ix] = star_ix
    n_entries += 1
    push!(
        transformer_records,
        (
            from = tertiary_ix,
            to = star_ix,
            turns = PSY.get_tertiary_turns_ratio(br),
            angle = PSY.get_α_tertiary(br),
            Y = 1 / (PSY.get_r_tertiary(br) + PSY.get_x_tertiary(br) * 1im),
            shunt = zero(ComplexF64),
        ),
    )
    return n_entries
end

# --- Shunt device handlers ---

function _ybus_split_shunt!(
    ysh::Vector{ComplexF64},
    fa::PSY.FixedAdmittance,
    num_bus::Dict{Int, Int},
    fa_ix::Int,
    sb::Vector{Int},
    nr::NetworkReductionData,
)
    bus_no = get_bus_index(fa, num_bus, nr)
    Y = PSY.get_Y(fa)
    sb[fa_ix] = bus_no
    if !isfinite(Y)
        error(
            "Data in $(PSY.get_name(fa)) is incorrect. Y = $(Y)",
        )
    end
    ysh[fa_ix] = Y
    return
end

function _ybus_split_shunt!(
    ysh::Vector{ComplexF64},
    fa::PSY.SwitchedAdmittance,
    num_bus::Dict{Int, Int},
    fa_ix::Int,
    sb::Vector{Int},
    nr::NetworkReductionData,
)
    bus_no = get_bus_index(fa, num_bus, nr)
    sb[fa_ix] = bus_no
    ysh[fa_ix] = 0.0
    return
end

# --- Core assembly ---

function _buildybus_split!(
    network_reduction_data::NetworkReductionData,
    adj::SparseArrays.SparseMatrixCSC{Int8, Int},
    branches,
    transformer_3w::Vector{PSY.ThreeWindingTransformer},
    num_bus::Dict{Int, Int},
    fixed_admittances::Vector{PSY.FixedAdmittance},
    switched_admittances::Vector{PSY.SwitchedAdmittance},
    transformer_records::Vector{TransformerRecord},
)
    branch_entries_transformer_3w = 0
    for br in transformer_3w
        branch_entries_transformer_3w += count([
            PSY.get_available_primary(br),
            PSY.get_available_secondary(br),
            PSY.get_available_tertiary(br),
        ])
    end
    branchcount = length(branches) + branch_entries_transformer_3w
    branchcount_no_3w = length(branches)
    fa_count = length(fixed_admittances)
    sa_count = length(switched_admittances)
    fb = zeros(Int, branchcount)
    tb = zeros(Int, branchcount)
    sb = zeros(Int, fa_count + sa_count)

    y11 = zeros(ComplexF64, branchcount)
    y12 = zeros(ComplexF64, branchcount)
    y21 = zeros(ComplexF64, branchcount)
    y22 = zeros(ComplexF64, branchcount)
    y_shunt_from = zeros(ComplexF64, branchcount)
    y_shunt_to = zeros(ComplexF64, branchcount)
    ysh = zeros(ComplexF64, fa_count + sa_count)

    for (ix, b) in enumerate(branches)
        if PSY.get_name(b) == "init"
            throw(DataFormatError("The data in Branch is invalid"))
        end
        _ybus_split!(
            y11, y12, y21, y22, y_shunt_from, y_shunt_to,
            b, num_bus, ix, fb, tb, network_reduction_data, adj,
            transformer_records,
        )
    end

    ix = 1
    for b in transformer_3w
        if PSY.get_name(b) == "init"
            throw(DataFormatError("The data in Transformer3W is invalid"))
        end
        n_entries = _ybus_split!(b, num_bus, branchcount_no_3w, fb, tb, ix,
            network_reduction_data, adj, transformer_records,
        )
        ix += n_entries
    end
    for (ix, fa) in enumerate([fixed_admittances; switched_admittances])
        _ybus_split_shunt!(ysh, fa, num_bus, ix, sb, network_reduction_data)
    end
    return (
        y11,
        y12,
        y21,
        y22,
        y_shunt_from,
        y_shunt_to,
        ysh,
        fb,
        tb,
        sb,
    )
end

# --- Constructor ---

function YbusSplit(
    sys::PSY.System;
    subnetwork_algorithm = iterative_union_find,
    network_reductions = NetworkReduction[],
)
    units_base = PSY.get_units_base(sys)
    if units_base != "SYSTEM_BASE"
        @warn "Setting the system unit base from $units_base to SYSTEM_BASE for matrix construction"
        PSY.set_units_base_system!(sys, "SYSTEM_BASE")
    end

    if !isempty(network_reductions)
        error("YbusSplit doesn't support network reductions.")
    end

    # Prototype guards: error on unsupported component types
    if !isempty(
        collect(
            PSY.get_available_components(PSY.DiscreteControlledACBranch, sys),
        ),
    )
        error("YbusSplit does not support DiscreteControlledACBranch components.")
    end
    if !isempty(
        collect(
            PSY.get_available_components(PSY.StandardLoad, sys),
        ),
    )
        error("YbusSplit does not support StandardLoad components.")
    end

    ref_bus_numbers = Set{Int}()
    nr = NetworkReductionData()
    bus_reduction_map = get_bus_reduction_map(nr)

    for b in PSY.get_available_components(PSY.ACBus, sys)
        if PSY.get_bustype(b) == ACBusTypes.ISOLATED
            @debug "Found available isolated bus $(PSY.get_name(b)) with number $(PSY.get_number(b)). This is excluded from the YbusSplit build."
            push!(nr.removed_buses, PSY.get_number(b))
            continue
        end
        bus_reduction_map[PSY.get_number(b)] = Set{Int}()
        if PSY.get_bustype(b) == ACBusTypes.REF
            push!(ref_bus_numbers, PSY.get_number(b))
        end
    end

    bus_ax = sort!(collect(keys(bus_reduction_map)))
    axes = (bus_ax, bus_ax)
    bus_lookup = Dict{Int, Int}()
    lookup = (bus_lookup, bus_lookup)
    busnumber = length(bus_ax)
    for (ix, b) in enumerate(bus_ax)
        bus_lookup[b] = ix
    end
    adj = SparseArrays.spdiagm(ones(Int8, busnumber))
    branches = collect(
        PSY.get_components(
            x ->
                PSY.get_available(x) &&
                    typeof(x) ∉ [
                        PSY.Transformer3W,
                        PSY.PhaseShiftingTransformer3W,
                    ],
            PSY.ACTransmission,
            sys,
        ),
    )
    transformer_3W =
        collect(
            PSY.get_available_components(PSY.ThreeWindingTransformer, sys),
        )
    fixed_admittances = collect(
        PSY.get_components(
            x ->
                PSY.get_available(x) &&
                    PSY.get_bustype(PSY.get_bus(x)) != ACBusTypes.ISOLATED,
            PSY.FixedAdmittance,
            sys,
        ),
    )
    switched_admittances =
        collect(
            PSY.get_components(
                x ->
                    PSY.get_available(x) &&
                        PSY.get_bustype(PSY.get_bus(x)) != ACBusTypes.ISOLATED,
                PSY.SwitchedAdmittance,
                sys,
            ),
        )
    n_2W_transformers = length(
        collect(
            PSY.get_available_components(PSY.TwoWindingTransformer, sys),
        ),
    )
    transformer_records = Vector{TransformerRecord}()
    sizehint!(transformer_records, n_2W_transformers + 3 * length(transformer_3W))
    y11, y12, y21, y22, y_shunt_from, y_shunt_to, ysh, fb, tb, sb =
        _buildybus_split!(
            nr,
            adj,
            branches,
            transformer_3W,
            bus_lookup,
            fixed_admittances,
            switched_admittances,
            transformer_records,
        )

    y_series = SparseArrays.sparse(
        [fb; fb; tb; tb],
        [fb; tb; fb; tb],
        [y11; y12; y21; y22],
        busnumber,
        busnumber,
    )
    function has_structural_diagonal(A::SparseArrays.SparseMatrixCSC)
        rv = SparseArrays.rowvals(A)
        for j in 1:size(A, 2)
            r = SparseArrays.nzrange(A, j)
            idx = searchsortedfirst(rv, j, first(r), last(r), Base.Order.Forward)
            (idx > last(r) || rv[idx] != j) && return false
        end
        return true
    end
    @assert has_structural_diagonal(y_series) "YbusSplit requires structurally nonzero diagonal entries for connectivity detection. Found zero diagonal entry at bus index $(findfirst(x -> x == 0, diag(y_series)))"

    I, J, V = SparseArrays.findnz(y_series)
    y_lambda = SparseArrays.sparse(
        I, J, zeros(ComplexF64, length(V)),
    )

    # Accumulate per-bus shunt vector: line charging from branches + device shunts.
    y_shunt = zeros(ComplexF64, busnumber)
    for i in eachindex(fb)
        y_shunt[fb[i]] += y_shunt_from[i]
        y_shunt[tb[i]] += y_shunt_to[i]
    end
    for i in eachindex(sb)
        y_shunt[sb[i]] += ysh[i]
    end

    if length(bus_lookup) > 1
        # Use the adjacency matrix for connectivity detection.
        # y_series + spdiagm(y_shunt) would drop structural zeros (transformer
        # entries that are zero at lambda != 0), incorrectly splitting the network.
        subnetworks = assign_reference_buses!(
            find_subnetworks(
                adj,
                bus_ax;
                subnetwork_algorithm = subnetwork_algorithm,
            ),
            ref_bus_numbers,
        )
        if length(subnetworks) > 1
            @warn "More than one island found; Network is not connected"
        end
    else
        subnetworks = Dict{Int, Set{Int}}(bus_lookup[1] => Set(bus_ax))
    end
    subnetwork_axes = _make_bus_subnetwork_axes(subnetworks)
    arc_subnetwork_axis = _make_arc_subnetwork_axis(subnetworks, nr)
    return YbusSplit(
        y_series,
        y_shunt,
        y_lambda,
        adj,
        axes,
        lookup,
        subnetwork_axes,
        arc_subnetwork_axis,
        nr,
        transformer_records,
    )
end

function generic_transfomer_ybus_entries(turns::Number, angle::Number, Y::Complex)
    tap = turns * exp(1im * angle)
    Y11 = Y / abs2(tap)
    Y12 = -Y / conj(tap)
    Y21 = -Y / tap
    Y22 = Y
    return (Y11, Y12, Y21, Y22)
end

function update_y_lambda!(M::YbusSplit, lambda::Number, gamma::Number)
    SparseArrays.nonzeros(M.y_lambda) .= 0.0
    # transmission piece: (1 + λ γ) * Y_series + (1 - λ) * Diagonal(Y_shunt)
    # λ = 0 gives original, Y_series + Diagonal(Y_shunt)
    # λ = 1 gives (1 + γ) * Y_series, shunts removed and all series entries shorted.
    @assert SparseArrays.rowvals(M.y_lambda) == SparseArrays.rowvals(M.y_series)
    @assert M.y_lambda.colptr == M.y_series.colptr
    SparseArrays.nonzeros(M.y_lambda) .+=
        SparseArrays.nonzeros(M.y_series) .* (1 + lambda * gamma)
    for j in 1:size(M.y_lambda, 2)
        M.y_lambda[j, j] += M.y_shunt[j] * (1 - lambda)
    end
    # transformer piece: 
    for record in M.transformers
        from = record.from
        to = record.to
        # λ = 0 gives original
        # λ = 1 gives turns = 1, angle = 0, admittance = Y (1 + γ) ~ infinity, shunt = 0.
        turns = record.turns + lambda * (1 - record.turns)
        angle = (1 - lambda) * record.angle
        Y = (1 + lambda * gamma) * record.Y
        Y11, Y12, Y21, Y22 = generic_transfomer_ybus_entries(turns, angle, Y)
        M.y_lambda[from, from] += Y11 + record.shunt * (1 - lambda)
        M.y_lambda[from, to] += Y12
        M.y_lambda[to, from] += Y21
        M.y_lambda[to, to] += Y22
    end
end
