const FACTOR_FILE_FORMAT_VERSION = 1

# The names table maps a type name to a PSY type through this fixed table. The reader never
# looks a file string up in a module, so a file cannot name an arbitrary binding.
const FACTOR_FILE_BRANCH_TYPES = Dict{String, DataType}(
    string(nameof(T)) => T for T in (
        PSY.Line,
        PSY.TwoWindingTransformer,
        PSY.ThreeWindingTransformer,
        PSY.DiscreteControlledACBranch,
        PSY.GenericArcImpedance,
        PSY.SeriesCompensator,
    )
)

function _branch_type(name::String, path::String)
    haskey(FACTOR_FILE_BRANCH_TYPES, name) ||
        error("Factor file entry $path holds the unknown branch type \"$name\".")
    return FACTOR_FILE_BRANCH_TYPES[name]
end

function _branch_type_name(::Type{T}) where {T}
    name = string(nameof(T))
    haskey(FACTOR_FILE_BRANCH_TYPES, name) ||
        error("Branch type $T has no entry in FACTOR_FILE_BRANCH_TYPES.")
    return name
end

# Every value in a factor file is untrusted. `_typed` admits the one type the layout
# prescribes for an entry and errors with the entry name on anything else.
_typed(x::T, ::Type{T}, ::String) where {T} = x
function _typed(x, ::Type{T}, name::String) where {T}
    return error("Factor file entry $name holds a $(typeof(x)); expected $T.")
end

_expected_rank(::Type{<:Vector}) = 1
_expected_rank(::Type) = 0
_element_type(::Type{Vector{E}}) where {E} = E
_element_type(::Type{T}) where {T} = T

# Layout and filters exist for datasets only. Attributes have neither.
_check_storage(::HDF5.Attribute, ::String) = nothing
function _check_storage(ds::HDF5.Dataset, name::String)
    props = HDF5.get_create_properties(ds)
    props.layout in (:contiguous, :compact, :chunked) ||
        error("Factor file entry $name has the layout $(props.layout).")
    iszero(HDF5.API.h5p_get_nfilters(props)) ||
        error("Factor file entry $name has a filter.")
    iszero(HDF5.API.h5p_get_external_count(props)) ||
        error("Factor file entry $name names an external file.")
    return
end

# One budget per file read. Every object charges its declared bytes against the file size,
# so the sum of all allocations stays below the file size.
mutable struct ReadBudget
    remaining::Int
end
ReadBudget(filename::AbstractString) = ReadBudget(filesize(filename))

# The class and the element size of the stored type. The check runs before any call that
# builds a Julia type from the HDF5 type, because that call can allocate the declared size.
function _stored_type(obj::Union{HDF5.Dataset, HDF5.Attribute})
    dt = HDF5.datatype(obj)
    try
        class = Int(HDF5.API.h5t_get_class(dt))
        variable = class == Int(HDF5.API.H5T_STRING) && HDF5.API.h5t_is_variable_str(dt)
        return (class, Int(HDF5.API.h5t_get_size(dt)), variable)
    finally
        close(dt)
    end
end

# The writer produces these classes. HDF5.jl stores a `Bool` as a bitfield.
_expected_class(::Type{Int}) = Int(HDF5.API.H5T_INTEGER)
_expected_class(::Type{Float64}) = Int(HDF5.API.H5T_FLOAT)
_expected_class(::Type{Bool}) = Int(HDF5.API.H5T_BITFIELD)

function _check_stored_type(
    ::Type{E},
    name::String,
    stored::Tuple{Int, Int, Bool},
) where {E}
    class, elsize, _ = stored
    (class == _expected_class(E) && elsize == sizeof(E)) ||
        error(
            "Factor file entry $name has the HDF5 class $class and $elsize bytes " *
            "per element; expected a $E of $(sizeof(E)) bytes.",
        )
    return
end

# A string is a fixed-length string. A variable-length string points to heap objects that
# the byte budget cannot bound.
const MAX_FIXED_STRING_BYTES = 65536
function _check_stored_type(::Type{String}, name::String, stored::Tuple{Int, Int, Bool})
    class, elsize, variable = stored
    variable && error("Factor file entry $name is a variable-length string.")
    class == Int(HDF5.API.H5T_STRING) ||
        error("Factor file entry $name has the HDF5 class $class; expected a string.")
    (1 <= elsize <= MAX_FIXED_STRING_BYTES) ||
        error("Factor file entry $name has a string of $elsize bytes per element.")
    return
end

# `n <= remaining` and the division keep `n * elsize` from overflowing.
function _charge!(budget::ReadBudget, name::String, n::Int, elsize::Int)
    (n <= budget.remaining && elsize <= budget.remaining ÷ max(n, 1)) ||
        error(
            "Factor file entry $name declares $n elements of $elsize bytes; " *
            "the remaining read budget is $(budget.remaining) bytes.",
        )
    budget.remaining -= n * elsize
    return
end

_check_eltype(::Type{String}, ::Union{HDF5.Dataset, HDF5.Attribute}, ::String) = nothing
function _check_eltype(
    ::Type{E},
    obj::Union{HDF5.Dataset, HDF5.Attribute},
    name::String,
) where {E}
    eltype(obj) == E || error("Factor file entry $name holds $(eltype(obj)); expected $E.")
    return
end

# Each check reads metadata only. No check touches the data. The function returns the
# element size.
function _check_before_read(
    obj::Union{HDF5.Dataset, HDF5.Attribute},
    name::String,
    ::Type{T},
    budget::ReadBudget,
) where {T}
    E = _element_type(T)
    _check_storage(obj, name)
    stored = _stored_type(obj)
    _check_stored_type(E, name, stored)
    elsize = stored[2]
    dims = size(HDF5.dataspace(obj))
    length(dims) == _expected_rank(T) ||
        error(
            "Factor file entry $name has rank $(length(dims)); expected $(_expected_rank(T)).",
        )
    _charge!(budget, name, prod(dims), elsize)
    _check_eltype(E, obj, name)
    return elsize
end

function _check_before_read(obj, name::String, ::Type, ::ReadBudget)
    return error(
        "Factor file entry $name is a $(typeof(obj)); expected a dataset or an attribute.",
    )
end

function _read_raw!(buf::Vector{UInt8}, ds::HDF5.Dataset)
    dt = HDF5.datatype(ds)
    try
        HDF5.API.h5d_read(
            ds,
            dt,
            HDF5.API.H5S_ALL,
            HDF5.API.H5S_ALL,
            HDF5.API.H5P_DEFAULT,
            buf,
        )
    finally
        close(dt)
    end
    return buf
end

function _read_raw!(buf::Vector{UInt8}, attr::HDF5.Attribute)
    dt = HDF5.datatype(attr)
    try
        HDF5.API.h5a_read(attr, dt, buf)
    finally
        close(dt)
    end
    return buf
end

# A fixed-length string ends at its last non-null byte.
function _fixed_string(buf::Vector{UInt8}, i::Int, width::Int)
    first_byte = (i - 1) * width + 1
    padded = view(buf, first_byte:(first_byte + width - 1))
    return String(padded[1:something(findlast(!iszero, padded), 0)])
end

function _read_strings(obj::Union{HDF5.Dataset, HDF5.Attribute}, n::Int, width::Int)
    buf = _read_raw!(zeros(UInt8, n * width), obj)
    return String[_fixed_string(buf, i, width) for i in 1:n]
end

function _read_data(
    obj::Union{HDF5.Dataset, HDF5.Attribute},
    ::Type{String},
    ::String,
    width::Int,
)
    return only(_read_strings(obj, 1, width))
end

function _read_data(
    obj::Union{HDF5.Dataset, HDF5.Attribute},
    ::Type{Vector{String}},
    ::String,
    width::Int,
)
    return _read_strings(obj, only(size(HDF5.dataspace(obj))), width)
end

function _read_data(
    obj::Union{HDF5.Dataset, HDF5.Attribute},
    ::Type{T},
    name::String,
    ::Int,
) where {T}
    return _typed(read(obj), T, name)
end

_entry_path(::Union{HDF5.File, HDF5.Attributes}, name::String) = name
_entry_path(g::HDF5.Group, name::String) = "$(HDF5.name(g))/$name"

function _read(
    g::Union{HDF5.File, HDF5.Group, HDF5.Attributes},
    name::String,
    ::Type{T},
    budget::ReadBudget,
) where {T}
    path = _entry_path(g, name)
    haskey(g, name) || error("Factor file has no entry $path.")
    obj = g[name]
    elsize = _check_before_read(obj, path, T, budget)
    return _read_data(obj, T, path, elsize)
end

function _check_hard_links(g::Union{HDF5.File, HDF5.Group})
    HDF5.API.h5l_iterate(g, HDF5.API.H5_INDEX_NAME, HDF5.API.H5_ITER_INC) do _, name, info
        unsafe_load(info).linktype == HDF5.API.H5L_TYPE_HARD ||
            error("Factor file link $(unsafe_string(name)) is not a hard link.")
        return false
    end
    return
end

# Columns of one table must agree in length, or `zip` would drop rows in silence.
function _check_same_length(table::String, columns::Pair{String, <:AbstractVector}...)
    lengths = map(column -> first(column) => length(last(column)), columns)
    allequal(map(last, lengths)) ||
        error("Factor file table $table has columns of unequal length: $lengths.")
    return
end

# The reader strips trailing null bytes, so a null byte cannot round trip.
function _string_width(values::AbstractVector{String}, name::String)
    any(v -> occursin('\0', v), values) &&
        error("Entry $name holds a string with a null byte.")
    width = max(1, maximum(ncodeunits, values; init = 0))
    width <= MAX_FIXED_STRING_BYTES ||
        error(
            "Entry $name holds a string of $width bytes; the limit is $MAX_FIXED_STRING_BYTES.",
        )
    return width
end

function _fixed_string_type(width::Int)
    dt = HDF5.Datatype(HDF5.API.h5t_copy(HDF5.API.H5T_C_S1))
    HDF5.API.h5t_set_size(dt, width)
    HDF5.API.h5t_set_strpad(dt, HDF5.API.H5T_STR_NULLPAD)
    HDF5.API.h5t_set_cset(dt, HDF5.API.H5T_CSET_UTF8)
    return dt
end

function _string_buffer(values::AbstractVector{String}, width::Int)
    buf = zeros(UInt8, width * length(values))
    for (i, v) in enumerate(values)
        copyto!(buf, (i - 1) * width + 1, codeunits(v), 1, ncodeunits(v))
    end
    return buf
end

# Writes `values` as fixed-length strings as wide as the longest value.
function _write_strings!(
    parent::Union{HDF5.File, HDF5.Group},
    name::String,
    values::Vector{String},
)
    width = _string_width(values, name)
    dt = _fixed_string_type(width)
    try
        ds = HDF5.create_dataset(parent, name, dt, HDF5.dataspace((length(values),)))
        try
            HDF5.API.h5d_write(
                ds,
                dt,
                HDF5.API.H5S_ALL,
                HDF5.API.H5S_ALL,
                HDF5.API.H5P_DEFAULT,
                _string_buffer(values, width),
            )
        finally
            close(ds)
        end
    finally
        close(dt)
    end
    return
end

function _write_string_attribute!(a::HDF5.Attributes, name::String, value::String)
    values = [value]
    width = _string_width(values, name)
    dt = _fixed_string_type(width)
    try
        attr = HDF5.create_attribute(a.parent, name, dt, HDF5.dataspace(()))
        try
            HDF5.API.h5a_write(attr, dt, _string_buffer(values, width))
        finally
            close(attr)
        end
    finally
        close(dt)
    end
    return
end

function _write_tol!(a::HDF5.Attributes, tol::Float64)
    _write_string_attribute!(a, "tol_kind", "absolute")
    a["tol_value"] = tol
    return
end

# A numeric `data_precision` is stored as a Float64. `:auto` leaves the attribute absent.
function _write_data_precision!(a::HDF5.Attributes, dp::Float64)
    a["auto_data_precision"] = dp
    return
end
_write_data_precision!(::HDF5.Attributes, ::Symbol) = nothing

function _write_tol!(a::HDF5.Attributes, tol::AutoTolerance)
    _write_string_attribute!(a, "tol_kind", "auto")
    _write_data_precision!(a, tol.data_precision)
    a["auto_safety"] = tol.safety
    a["auto_quantile"] = tol.quantile
    return
end

function _read_data_precision(a::HDF5.Attributes, budget::ReadBudget)
    if haskey(a, "auto_data_precision")
        return _read(a, "auto_data_precision", Float64, budget)
    end
    return :auto
end

function _read_tol(a::HDF5.Attributes, budget::ReadBudget)
    kind = _read(a, "tol_kind", String, budget)
    if kind == "absolute"
        tol = _read(a, "tol_value", Float64, budget)
        (isfinite(tol) && tol >= 0) ||
            error("Factor file entry tol_value is $tol; expected a finite value >= 0.")
        return tol
    elseif kind == "auto"
        return AutoTolerance(;
            data_precision = _read_data_precision(a, budget),
            safety = _read(a, "auto_safety", Float64, budget),
            quantile = _read(a, "auto_quantile", Float64, budget),
        )
    end
    return error("Unknown tolerance kind \"$kind\" in factor file.")
end

# The attribute holds the two halves of the UUID as Int64 values. A matrix without a UUID
# has no attribute.
_write_system_uuid!(::HDF5.Attributes, ::Nothing) = nothing
function _write_system_uuid!(a::HDF5.Attributes, uuid::Base.UUID)
    halves = (UInt64(uuid.value >> 64), UInt64(uuid.value & typemax(UInt64)))
    a["system_uuid"] = Int64[reinterpret(Int64, h) for h in halves]
    return
end

function _restore_system_uuid(a::HDF5.Attributes, budget::ReadBudget)
    if !haskey(a, "system_uuid")
        return nothing
    end
    halves = _read(a, "system_uuid", Vector{Int}, budget)
    length(halves) == 2 ||
        error("Factor file entry system_uuid has $(length(halves)) values; expected 2.")
    hi, lo = reinterpret(UInt64, halves)
    return Base.UUID((UInt128(hi) << 64) | UInt128(lo))
end

function _check_dist_slack(weights::Vector{Float64}, bus_ax::Vector{Int})
    isempty(weights) && return
    _check_same_length("dist_slack", "bus_axis" => bus_ax, "dist_slack" => weights)
    for (i, w) in enumerate(weights)
        isfinite(w) || error(
            "Factor file entry dist_slack has the value $w at position $i; " *
            "the value must be finite.",
        )
        w >= 0 || error(
            "Factor file entry dist_slack has the weight $w at position $i; " *
            "the weight must not be negative.",
        )
    end
    total = sum(weights)
    (isfinite(total) && total > 0) || error(
        "Factor file entry dist_slack has the sum $total; the sum must be finite and positive.",
    )
    return
end

"""
    BranchArcTable

Columns of the `branch_map` group of a factor file: one row per node of the reduction tree
of each arc, in depth-first preorder. `n_arc` is the number of arcs. See the factor file
guide for the meaning of each column.
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

"Return the `id` column of `t`: the `IS.get_id` of the branch of each row, or 0."
get_branch_id(t::BranchArcTable) = t.id

"Return the `parent` column of `t`: the parent row, or minus the arc index for a top node."
get_branch_parent(t::BranchArcTable) = t.parent

"Return the `sign` column of `t`: the orientation of each row against its parent."
get_branch_sign(t::BranchArcTable) = t.sign

"Return the `b` column of `t`: the susceptance of each leaf or Ward row, in system base."
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
            _in_susceptance_range(b[r]) ||
                fail(
                    "b is $(b[r]); expected a magnitude from $(1 / MAX_ARC_SUSCEPTANCE) to $MAX_ARC_SUSCEPTANCE",
                )
            isfinite(alpha[r]) || fail("alpha is $(alpha[r]); expected a finite value")
            key = (id[r], max(k - 5, 0))
            key in leaf_keys && fail("id and winding $key repeat")
            push!(leaf_keys, key)
        elseif k == 11
            iszero(id[r]) || fail("id is not 0")
            # A Ward row is a top node, so no flow inverts its b: only the upper bound applies.
            (isfinite(b[r]) && !iszero(b[r]) && abs(b[r]) <= MAX_ARC_SUSCEPTANCE) ||
                fail(
                    "b is $(b[r]); expected a finite nonzero magnitude of at most $MAX_ARC_SUSCEPTANCE",
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

_branch_map_kind(::PSY.Line) = 1
_branch_map_kind(::PSY.TwoWindingTransformer) = 2
_branch_map_kind(::PSY.DiscreteControlledACBranch) = 3
_branch_map_kind(::PSY.GenericArcImpedance) = 4
_branch_map_kind(::PSY.SeriesCompensator) = 5
_branch_map_kind(w::ThreeWindingTransformerCircuit) = 5 + get_winding_number(w)
_branch_map_kind(::AbstractBranchesParallel) = 9
_branch_map_kind(::BranchesSeries) = 10

_branch_map_id(x::PSY.ACTransmission) = IS.get_id(x)
_branch_map_id(w::ThreeWindingTransformerCircuit) = IS.get_id(get_transformer(w))
_branch_map_id(::AbstractReductionAggregate) = 0

function _frame_sign(frame::Tuple{Int, Int}, own::Tuple{Int, Int})
    own == frame && return 1
    own == (frame[2], frame[1]) && return -1
    return error("Arc $own is not arc $frame in either orientation.")
end

function _push_row!(
    rows::BranchArcTable,
    kind::Int,
    id::Int,
    parent::Int,
    sign::Int,
    b::Float64,
    alpha::Float64,
)
    push!(rows.kind, kind)
    push!(rows.id, id)
    push!(rows.parent, parent)
    push!(rows.sign, sign)
    push!(rows.b, b)
    push!(rows.alpha, alpha)
    return length(rows.kind)
end

function _push_node!(
    rows::BranchArcTable,
    leaf::PSY.ACTransmission,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
)
    _push_row!(
        rows, _branch_map_kind(leaf), _branch_map_id(leaf), parent, sign,
        get_effective_series_susceptance(leaf, nr), _segment_phase_shift(leaf, nr),
    )
    return
end

function _push_node!(
    rows::BranchArcTable,
    bp::AbstractBranchesParallel,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
)
    row = _push_row!(rows, 9, 0, parent, sign, 0.0, 0.0)
    frame = get_arc_tuple(bp, nr)
    for member in bp
        _push_node!(rows, member, row, _frame_sign(frame, get_arc_tuple(member, nr)), nr)
    end
    return
end

function _push_node!(
    rows::BranchArcTable,
    bs::BranchesSeries,
    parent::Int,
    sign::Int,
    nr::NetworkReductionData,
)
    row = _push_row!(rows, 10, 0, parent, sign, 0.0, 0.0)
    orientations = get_segment_orientations(bs)
    length(orientations) == length(bs) || error(
        "Series chain on arc $(get_arc_tuple(bs, nr)) has $(length(bs)) segments " *
        "and $(length(orientations)) orientations.",
    )
    for (segment, orientation) in zip(bs, orientations)
        if orientation == :FromTo
            seg_sign = 1
        elseif orientation == :ToFrom
            seg_sign = -1
        else
            error(
                "Series chain segment orientation $orientation is not :FromTo or :ToFrom.",
            )
        end
        _push_node!(rows, segment, row, seg_sign, nr)
    end
    return
end

function _push_ward!(
    rows::BranchArcTable,
    ward::PSY.GenericArcImpedance,
    parent::Int,
    nr::NetworkReductionData,
)
    _push_row!(rows, 11, 0, parent, 1, get_effective_series_susceptance(ward, nr), 0.0)
    return
end

function _build_branch_arc_table(nr::NetworkReductionData, arc_ax::Vector{Tuple{Int, Int}})
    table = BranchArcTable(Int[], Int[], Int[], Int[], Float64[], Float64[], length(arc_ax))
    added = get_added_arc_impedance_map(nr)
    for (k, arc) in enumerate(arc_ax)
        entry, _ = _resolve_arc_entry(nr, arc)
        has_ward = haskey(added, arc)
        if has_ward && added[arc] === entry
            _push_ward!(table, entry, -k, nr)
        else
            _push_node!(table, entry, -k, _frame_sign(arc, get_arc_tuple(entry, nr)), nr)
            has_ward && _push_ward!(table, added[arc], -k, nr)
        end
    end
    _check_branch_map(table)
    return table
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

const _REDUCTION_COLUMNS = (
    :applied,
    :removed_arc_from, :removed_arc_to, :removed_arc_code,
    :composite_from, :composite_to,
    :survivor_from, :survivor_to, :survivor_bus,
    :boundary_bus, :boundary_from, :boundary_to,
    :removed_buses,
    :merged_from, :merged_to,
    :irreducible_buses,
)

const _ReductionRecords = NamedTuple{_REDUCTION_COLUMNS, NTuple{16, Vector{Int}}}

function _reduction_records(nr::NetworkReductionData)::_ReductionRecords
    removed = sort!(collect(keys(nr.removed_arc_reduction)))
    survivors = sort!(collect(keys(nr.removed_arc_to_surviving_bus)))
    boundary = sort!([
        (bus, arc) for (bus, arcs) in nr.boundary_bus_to_removed_arcs for arc in arcs
    ])
    merged = sort!(collect(nr.merged_bus_pairs))
    return (
        applied = copy(nr.reduction_codes),
        removed_arc_from = first.(removed),
        removed_arc_to = last.(removed),
        removed_arc_code = [nr.removed_arc_reduction[a] for a in removed],
        composite_from = [first(get(nr.removed_arc_composite, a, (0, 0))) for a in removed],
        composite_to = [last(get(nr.removed_arc_composite, a, (0, 0))) for a in removed],
        survivor_from = first.(survivors),
        survivor_to = last.(survivors),
        survivor_bus = [nr.removed_arc_to_surviving_bus[a] for a in survivors],
        boundary_bus = first.(boundary),
        boundary_from = [first(last(x)) for x in boundary],
        boundary_to = [last(last(x)) for x in boundary],
        removed_buses = sort!(collect(nr.removed_buses)),
        merged_from = first.(merged),
        merged_to = last.(merged),
        irreducible_buses = sort!(collect(nr.irreducible_buses)),
    )
end

function _check_unique(path::String, values::Vector{T}) where {T}
    seen = Set{T}()
    for (i, v) in enumerate(values)
        v in seen &&
            error(
                "Factor file entry $path has the value $v more than once, at row $i.",
            )
        push!(seen, v)
    end
    return
end

function _check_known_bus(name::String, bus::Int, i::Int, known::Set{Int})
    bus in known || error(
        "Factor file entry reduction/$name holds bus $bus at row $i, which is not on the bus " *
        "axis, in the reduced-bus map or in removed_buses.",
    )
    return
end

function _check_known_buses(name::String, buses::Vector{Int}, known::Set{Int})
    for (i, bus) in enumerate(buses)
        _check_known_bus(name, bus, i, known)
    end
    return
end

# One validator serves the writer and the reader.
function _check_reduction_records(
    rec::_ReductionRecords,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
)
    _check_same_length(
        "reduction removed_arc",
        "removed_arc_from" => rec.removed_arc_from,
        "removed_arc_to" => rec.removed_arc_to,
        "removed_arc_code" => rec.removed_arc_code,
        "composite_from" => rec.composite_from,
        "composite_to" => rec.composite_to,
    )
    _check_same_length(
        "reduction survivor",
        "survivor_from" => rec.survivor_from,
        "survivor_to" => rec.survivor_to,
        "survivor_bus" => rec.survivor_bus,
    )
    _check_same_length(
        "reduction boundary",
        "boundary_bus" => rec.boundary_bus,
        "boundary_from" => rec.boundary_from,
        "boundary_to" => rec.boundary_to,
    )
    _check_same_length(
        "reduction merged",
        "merged_from" => rec.merged_from,
        "merged_to" => rec.merged_to,
    )
    for code in rec.applied
        1 <= code <= 4 ||
            error("Factor file entry reduction/applied holds code $code; expected 1 to 4.")
    end
    _check_unique("reduction/applied", rec.applied)
    iszero(count(==(REDUCTION_CODE_WARD), rec.applied)) ||
        last(rec.applied) == REDUCTION_CODE_WARD ||
        error("Factor file entry reduction/applied has the Ward reduction before the end.")
    for (i, code) in enumerate(rec.removed_arc_code)
        code in rec.applied || error(
            "Factor file entry reduction/removed_arc_code row $i holds code $code, " *
            "which is not in applied.",
        )
    end
    removed_buses = Set(rec.removed_buses)
    _check_unique("reduction/removed_buses", rec.removed_buses)
    bus_set = Set(bus_ax)
    for bus in rec.removed_buses
        bus in bus_set && error(
            "Factor file entry reduction/removed_buses holds bus $bus, which is on the bus axis.",
        )
    end
    known = union(bus_set, keys(reduced), removed_buses)
    for root in values(reduced)
        root in bus_set || root in removed_buses ||
            error(
                "Factor file entry reduced_bus_to holds bus $root, which is not on the bus axis or in removed_buses.",
            )
    end
    removed_arcs = collect(zip(rec.removed_arc_from, rec.removed_arc_to))
    _check_unique("reduction/removed_arc", removed_arcs)
    _check_known_buses("removed_arc_from", rec.removed_arc_from, known)
    _check_known_buses("removed_arc_to", rec.removed_arc_to, known)
    removed_set = Set(removed_arcs)
    arc_set = Set(arc_ax)
    for (i, arc) in enumerate(removed_arcs)
        (arc in arc_set || (arc[2], arc[1]) in arc_set) && error(
            "Factor file entry reduction/removed_arc row $i holds arc $arc, which is on the arc axis.",
        )
    end
    code_of_arc = Dict(zip(removed_arcs, rec.removed_arc_code))
    for (i, code) in enumerate(rec.removed_arc_code)
        pair = (rec.composite_from[i], rec.composite_to[i])
        if code == REDUCTION_CODE_DEGREE_TWO
            (iszero(pair[1]) || iszero(pair[2])) && error(
                "Factor file entry reduction/composite row $i is empty for a degree-two arc.",
            )
            _check_known_bus("composite_from", pair[1], i, known)
            _check_known_bus("composite_to", pair[2], i, known)
            any(key -> key in arc_set || key in removed_set, (pair, (pair[2], pair[1]))) ||
                error(
                    "Factor file entry reduction/composite row $i holds arc $pair, which " *
                    "is neither on the arc axis nor a removed arc.",
                )
            # One degree-two step exists, so a composite arc is never a degree-two member.
            # This also stops the reason lookup from following a loop.
            for key in (pair, (pair[2], pair[1]))
                get(code_of_arc, key, 0) == REDUCTION_CODE_DEGREE_TWO && error(
                    "Factor file entry reduction/composite row $i holds the degree-two arc $key.",
                )
            end
        else
            pair == (0, 0) || error(
                "Factor file entry reduction/composite row $i holds an arc, but code $code is not the degree-two code.",
            )
        end
    end
    survivor_arcs = collect(zip(rec.survivor_from, rec.survivor_to))
    _check_unique("reduction/survivor", survivor_arcs)
    _check_known_buses("survivor_bus", rec.survivor_bus, known)
    for arc in survivor_arcs
        arc in removed_set || error(
            "Factor file entry reduction/survivor holds arc $arc, which is not a removed arc.",
        )
    end
    boundary = collect(zip(rec.boundary_bus, rec.boundary_from, rec.boundary_to))
    _check_unique("reduction/boundary", boundary)
    _check_known_buses("boundary_bus", rec.boundary_bus, known)
    for (bus, from, to) in boundary
        (from, to) in removed_set || error(
            "Factor file entry reduction/boundary holds arc $((from, to)), which is not a removed arc.",
        )
    end
    _check_unique("reduction/merged_from", rec.merged_from)
    for bus in rec.merged_from
        bus in bus_set && error(
            "Factor file entry reduction/merged_from holds bus $bus, which is on the bus axis.",
        )
    end
    _check_known_buses("merged_from", rec.merged_from, known)
    _check_known_buses("merged_to", rec.merged_to, known)
    merged_from_set = Set(rec.merged_from)
    for (i, bus) in enumerate(rec.merged_to)
        bus in merged_from_set && error(
            "Factor file entry reduction/merged_to holds bus $bus at row $i, which is also in merged_from.",
        )
    end
    _check_unique("reduction/irreducible_buses", rec.irreducible_buses)
    _check_known_buses("irreducible_buses", rec.irreducible_buses, known)
    return
end

function _write_reduction!(
    f::HDF5.File,
    nr::NetworkReductionData,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
)
    rec = _reduction_records(nr)
    _check_reduction_records(rec, bus_ax, arc_ax, get_reverse_bus_search_map(nr))
    g = HDF5.create_group(f, "reduction")
    for name in _REDUCTION_COLUMNS
        g[string(name)] = rec[name]
    end
    return
end

function _read_reduction(
    f::HDF5.File,
    bus_ax::Vector{Int},
    arc_ax::Vector{Tuple{Int, Int}},
    reduced::Dict{Int, Int},
    budget::ReadBudget,
)
    g = _group(f, "reduction")
    rec = NamedTuple{_REDUCTION_COLUMNS}(
        map(name -> _read(g, string(name), Vector{Int}, budget), _REDUCTION_COLUMNS),
    )
    _check_reduction_records(rec, bus_ax, arc_ax, reduced)
    bus_reduction_map = Dict{Int, Set{Int}}(b => Set{Int}() for b in bus_ax)
    for (from, root) in reduced
        push!(get!(bus_reduction_map, root, Set{Int}()), from)
    end
    for bus in rec.removed_buses
        delete!(bus_reduction_map, bus)
    end
    boundary = Dict{Int, Set{Tuple{Int, Int}}}()
    for (bus, from, to) in zip(rec.boundary_bus, rec.boundary_from, rec.boundary_to)
        push!(get!(boundary, bus, Set{Tuple{Int, Int}}()), (from, to))
    end
    removed_arcs = collect(zip(rec.removed_arc_from, rec.removed_arc_to))
    return NetworkReductionData(;
        irreducible_buses = Set(rec.irreducible_buses),
        bus_reduction_map = bus_reduction_map,
        reverse_bus_search_map = reduced,
        removed_buses = Set(rec.removed_buses),
        removed_arcs = Set(removed_arcs),
        merged_bus_pairs = Dict{Int, Int}(zip(rec.merged_from, rec.merged_to)),
        removed_arc_to_surviving_bus = Dict{Tuple{Int, Int}, Int}(
            zip(zip(rec.survivor_from, rec.survivor_to), rec.survivor_bus),
        ),
        boundary_bus_to_removed_arcs = boundary,
        removed_arc_reduction = Dict{Tuple{Int, Int}, Int}(
            zip(removed_arcs, rec.removed_arc_code),
        ),
        removed_arc_composite = Dict{Tuple{Int, Int}, Tuple{Int, Int}}(
            a => (rec.composite_from[i], rec.composite_to[i]) for
            (i, a) in enumerate(removed_arcs) if
            rec.removed_arc_code[i] == REDUCTION_CODE_DEGREE_TWO
        ),
        reduction_codes = copy(rec.applied),
    )
end

function _write_factor_core(
    f::HDF5.File,
    core::VirtualFactorCore,
    dist_slack::Vector{Float64},
)
    holds_components(core) || error(
        "Cannot write a matrix that was restored from a factor file: it has no " *
        "reduction maps. Write the factor file from the matrix built on the system.",
    )
    nr = get_network_reduction_data(core)
    arc_ax, bus_ax = core.axes
    reduced = get_reverse_bus_search_map(nr)
    signed_susceptances = core.arc_susceptances .* core.arc_susceptance_signs
    ref_positions = get_ref_bus_position(core)
    _validate_dc_arrays(
        signed_susceptances,
        Set(ref_positions),
        core.axes,
        core.subnetwork_axes,
        core.arc_live_members,
        reduced,
    )
    _check_dist_slack(dist_slack, bus_ax)
    a = HDF5.attributes(f)
    a["format_version"] = FACTOR_FILE_FORMAT_VERSION
    _write_system_uuid!(a, get_system_uuid(core))
    _write_tol!(a, get_tol_spec(core))
    f["bus_axis"] = bus_ax
    f["arc_from"] = first.(arc_ax)
    f["arc_to"] = last.(arc_ax)
    f["ref_bus_positions"] = sort(ref_positions)
    f["arc_susceptances"] = signed_susceptances
    f["arc_live_members"] = core.arc_live_members
    f["shift_injection"] = Float64[arc_dc_shift_injection(nr, arc) for arc in arc_ax]
    f["dist_slack"] = dist_slack
    f["reduced_bus_from"] = collect(keys(reduced))
    f["reduced_bus_to"] = collect(values(reduced))

    types = String[]
    names = String[]
    from = Int[]
    to = Int[]
    for (T, entries) in get_name_to_arc_maps(get_branch_catalog(core))
        for (name, arc) in entries
            push!(types, _branch_type_name(T))
            push!(names, name)
            push!(from, arc[1])
            push!(to, arc[2])
        end
    end
    g = HDF5.create_group(f, "names")
    _write_strings!(g, "type", types)
    _write_strings!(g, "name", names)
    g["from"] = from
    g["to"] = to

    _write_branch_map!(f, _build_branch_arc_table(nr, arc_ax))
    _write_reduction!(f, nr, bus_ax, arc_ax)

    g = HDF5.create_group(f, "subnetworks")
    for (ref, (axis_arcs, axis_buses)) in core.subnetwork_axes
        sg = HDF5.create_group(g, string(ref))
        sg["ref"] = ref
        sg["arc_from"] = first.(axis_arcs)
        sg["arc_to"] = last.(axis_arcs)
        sg["bus"] = axis_buses
    end
    return
end

_dist_slack(mat::VirtualPTDF) = get_dist_slack(mat)
_dist_slack(::VirtualMODF) = Float64[]

function _write_factor_file_atomically(body, filename::AbstractString, force::Bool)
    if ispath(filename) && !isfile(filename)
        error("Cannot write $filename: the path exists and is not a regular file.")
    end
    if isfile(filename) && !force
        error("File $filename exists. Pass `force = true` to overwrite it.")
    end
    tmp = tempname(dirname(abspath(filename)); cleanup = false)
    try
        HDF5.h5open(body, tmp, "w")
        _publish_factor_file(tmp, filename, force)
    finally
        rm(tmp; force = true)
    end
    return filename
end

function _publish_factor_file(tmp::AbstractString, filename::AbstractString, force::Bool)
    if force
        Base.Filesystem.rename(tmp, filename)
        return
    end
    try
        hardlink(tmp, filename)
    catch e
        if _is_exists_error(e)
            error("File $filename exists. Pass `force = true` to overwrite it.")
        end
        error(_hardlink_failure_message(filename, e))
    end
    return
end

function _hardlink_failure_message(filename::AbstractString, e::Exception)
    return "Cannot publish $filename with a hard link: $(sprint(showerror, e)). " *
           "The target file system may not support hard links. " *
           "Pass `force = true` to write with a move."
end

_is_exists_error(e::Base.IOError) = e.code == Base.UV_EEXIST
_is_exists_error(::Exception) = false

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename; force = false)

Write the factor data of `mat` to the HDF5 file `filename`. The file holds the arrays that
[`from_hdf5`](@ref) needs to rebuild the matrix without a `System`. The function does not
write cached rows. A `VirtualMODF` writes an empty `dist_slack` dataset. A `VirtualMODF` also
writes its registered contingencies. The function writes a temporary file and moves it to
`filename` when the write succeeds, so a failed write does not change an existing file. If
the file exists and `force` is `false`, the function throws an error and never replaces the
file. If `force` is `true` and `filename` is a symbolic link, the function replaces the link
and does not write through it.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString;
    force::Bool = false,
)
    return _write_factor_file_atomically(filename, force) do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
        _write_registered_outages(f, mat)
    end
end

_outage_key(branch::PSY.ACTransmission) =
    (_branch_type_name(typeof(branch)), PSY.get_name(branch))

# The members that `NetworkModification(mat, branch)` classifies through the reverse maps.
_outage_members(branch::PSY.ACTransmission) = PSY.ACTransmission[branch]

function _outage_members(trf::PSY.ThreeWindingTransformer)
    windings = PSY.ACTransmission[
        ThreeWindingTransformerCircuit(trf, circuit, i) for
        (i, circuit) in enumerate(PSY.get_circuits(trf))
    ]
    return filter(get_equivalent_available, windings)
end

# The live path rejects partial series outages, so every series or direct row removes its arc.
_opens_arc(::Val{:parallel}) = false
_opens_arc(::Val) = true

# The position of the tripped member in its parallel group. Zero on an arc that is not a
# parallel group.
_group_member(::NetworkReductionData, ::Val, ::Tuple{Int, Int}, ::PSY.ACTransmission) = 0
function _group_member(
    nr::NetworkReductionData,
    ::Val{:parallel},
    arc::Tuple{Int, Int},
    branch::PSY.ACTransmission,
)
    bp = nr.parallel_branch_map[arc]
    member = _outaged_group_member(bp, branch)
    return findfirst(m -> m === member, bp.branches)::Int
end

# The number of members of the parallel group on the arc. Zero on an arc that is not a
# parallel group.
_group_size(::NetworkReductionData, ::Val, ::Tuple{Int, Int}) = 0
_group_size(nr::NetworkReductionData, ::Val{:parallel}, arc::Tuple{Int, Int}) =
    length(nr.parallel_branch_map[arc].branches)

# For each arc row of `mod`: `true` when the outage of `branch` removes all admittance of
# that arc, and the position of the tripped member in the parallel group of that arc. An
# outage that trips every member of a group removes the whole arc.
function _arc_row_flags(
    mat::Union{VirtualPTDF, VirtualMODF},
    branch::PSY.ACTransmission,
    mod::NetworkModification,
)
    nr = get_network_reduction_data(mat)
    arc_lookup = get_arc_lookup(mat)
    by_arc = Dict{Int, Tuple{Bool, Int}}()
    tripped = Dict{Int, Set{Int}}()
    sizes = Dict{Int, Int}()
    for member in _outage_members(branch)
        tag, arc = _resolve_branch_arc(nr, member)
        if tag === :not_found
            continue
        end
        ix = arc_lookup[arc]
        position = _group_member(nr, Val(tag), arc, member)
        get!(by_arc, ix) do
            (_opens_arc(Val(tag)), position)
        end
        push!(get!(Set{Int}, tripped, ix), position)
        sizes[ix] = _group_size(nr, Val(tag), arc)
    end
    for (ix, positions) in tripped
        if length(positions) == sizes[ix]
            by_arc[ix] = (true, last(by_arc[ix]))
        end
    end
    flags = [by_arc[m.arc_index] for m in mod.arc_modifications]
    return first.(flags), last.(flags)
end

function _write_row_arrays(
    g::HDF5.Group,
    mods::Vector{NetworkModification},
    n_arc::Int,
)
    starts = Int[]
    lens = Int[]
    arc_index = Int[]
    delta_b = Float64[]
    delta_shift = Float64[]
    opened = Int[]
    for mod in mods
        push!(starts, length(arc_index) + 1)
        push!(lens, length(mod.arc_modifications))
        for m in mod.arc_modifications
            push!(arc_index, m.arc_index)
            push!(delta_b, m.delta_b)
            push!(delta_shift, m.delta_shift_injection)
            push!(opened, m.opened)
        end
    end
    _check_row_values(HDF5.name(g), arc_index, delta_b, delta_shift, opened, n_arc)
    g["start"] = starts
    g["len"] = lens
    g["arc_index"] = arc_index
    g["delta_b"] = delta_b
    g["delta_shift"] = delta_shift
    g["opened"] = opened
    return
end

_is_absorbed(nr::NetworkReductionData, branch::PSY.ACTransmission) =
    _resolve_branch_arc(nr, branch)[1] === :not_found

function _is_absorbed(nr::NetworkReductionData, trf::PSY.ThreeWindingTransformer)
    return all(m -> _resolve_branch_arc(nr, m)[1] === :not_found, _outage_members(trf))
end

_skippable(::UnsupportedOutageError) = true
_skippable(::Any) = false

# An arc with zero susceptance has no DC effect, so its outage has no arc row.
function _drop_zero_susceptance_arcs(mod::NetworkModification)
    kept = ArcModification[m for m in mod.arc_modifications if !iszero(m.delta_b)]
    if length(kept) == length(mod.arc_modifications)
        return mod
    end
    return NetworkModification(
        mod.label,
        kept,
        ShuntModification[m for m in mod.shunt_modifications],
        mod.is_islanding,
    )
end

const _MAX_REJECTION_LINES = 10

function _rejection_summary(rejected::Vector{Pair{String, String}})
    counts = DataStructures.counter(last.(rejected))
    ranked = sort!(collect(counts); by = p -> (-last(p), first(p)))
    lines = ["$(n) x $(msg)" for (msg, n) in first(ranked, _MAX_REJECTION_LINES)]
    if length(ranked) > _MAX_REJECTION_LINES
        push!(lines, "and $(length(ranked) - _MAX_REJECTION_LINES) more")
    end
    return join(lines, "\n")
end

function _write_branch_outages(
    f::HDF5.File,
    mat::Union{VirtualPTDF, VirtualMODF},
    sys::PSY.System,
)
    nr = get_network_reduction_data(mat)
    keys_ = Tuple{String, String}[]
    ids = Int[]
    opens_arc = Bool[]
    members = Int[]
    mods = NetworkModification[]
    absorbed = String[]
    rejected = Pair{String, String}[]
    zero_susceptance = String[]
    partial_zero = String[]
    for branch in PSY.get_components(PSY.ACTransmission, sys)
        if _is_absorbed(nr, branch)
            push!(absorbed, PSY.get_name(branch))
            continue
        end
        mod = try
            NetworkModification(mat, branch)
        catch e
            _skippable(e) || rethrow()
            push!(rejected, PSY.get_name(branch) => sprint(showerror, e))
            continue
        end
        full_mod = mod
        mod = _drop_zero_susceptance_arcs(full_mod)
        if isempty(mod.arc_modifications)
            push!(zero_susceptance, PSY.get_name(branch))
            continue
        end
        if length(mod.arc_modifications) < length(full_mod.arc_modifications)
            push!(partial_zero, PSY.get_name(branch))
        end
        push!(mods, mod)
        push!(keys_, _outage_key(branch))
        push!(ids, IS.get_id(branch))
        flags, positions = _arc_row_flags(mat, branch, mod)
        append!(opens_arc, flags)
        append!(members, positions)
    end
    if !isempty(absorbed)
        @warn "$(length(absorbed)) branches are in no reduction map of the matrix (a network reduction absorbed them, or they are not in service). First: $(first(absorbed))"
    end
    if !isempty(rejected)
        @warn "$(length(rejected)) branches have no outage row because NetworkModification rejected them. First: $(first(first(rejected))).\n$(_rejection_summary(rejected))"
    end
    if !isempty(zero_susceptance)
        @warn "$(length(zero_susceptance)) branches have no outage row because their susceptance is zero, so an outage has no DC effect: $(join(first(zero_susceptance, _MAX_REJECTION_LINES), ", "))"
    end
    if !isempty(partial_zero)
        @warn "$(length(partial_zero)) branches have no row for their zero-susceptance arcs, which have no DC effect. The other arcs keep their rows: $(join(first(partial_zero, _MAX_REJECTION_LINES), ", "))"
    end
    g = HDF5.create_group(f, "branch_outages")
    _write_strings!(g, "type", first.(keys_))
    _write_strings!(g, "name", last.(keys_))
    g["id"] = ids
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods, length(first(get_core(mat).axes)))
    g["opens_arc"] = opens_arc
    g["member"] = members
    return
end

_write_registered_outages(::HDF5.File, ::VirtualPTDF) = nothing

function _write_registered_outages(f::HDF5.File, mat::VirtualMODF)
    registered = @lock get_core(mat).solver_lock copy(get_registered_contingencies(mat))
    if isempty(registered)
        return
    end
    ids = sort!(collect(keys(registered)))
    mods = NetworkModification[registered[id].modification for id in ids]
    g = HDF5.create_group(f, "outages")
    g["id"] = ids
    _write_strings!(g, "label", String[m.label for m in mods])
    g["is_islanding"] = [m.is_islanding for m in mods]
    _write_row_arrays(g, mods, length(first(get_core(mat).axes)))
    return
end

"""
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename, sys::PSY.System; force = false)

Write the factor data of `mat` as the method without `sys` does. Also write the group
`branch_outages` with one row for each branch of `sys` that `NetworkModification(mat, branch)`
accepts, so `from_hdf5(BranchOutageTable, filename)` can build outages by name. A branch that
is in no reduction map of the matrix has no row (a network reduction absorbed it, or it is not
in service). A branch that `NetworkModification` rejects with an `UnsupportedOutageError` has no row.
A branch whose arcs all have zero susceptance has no row either, because its outage has no DC effect.
A branch with some zero-susceptance arcs (a three-winding transformer) has rows for the other arcs only.
The function logs one warning for each case with the count. The warning for rejected branches
also lists the first branch and each distinct message with its count, at most 10 lines. Any other
exception propagates. A `VirtualMODF` also writes the group `outages` with the registered
contingencies. The function writes the file atomically. If `force` is `true` and `filename` is
a symbolic link, the function replaces the link and does not write through it.
"""
function to_hdf5(
    mat::Union{VirtualPTDF, VirtualMODF},
    filename::AbstractString,
    sys::PSY.System;
    force::Bool = false,
)
    _validate_system_uuid(mat, sys)
    return _write_factor_file_atomically(filename, force) do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
        _write_branch_outages(f, mat, sys)
        _write_registered_outages(f, mat)
    end
end

"""
    BranchOutageTable

Branch outages that `to_hdf5(mat, filename, sys)` stored. Read it with
`from_hdf5(BranchOutageTable, filename)`. Build a `NetworkModification` from it with
`NetworkModification(table, type, name)`.

# Fields
- `rows`: for the key `(type, name)`, the islanding flag and the row range.
- `arc_index`, `delta_b`, `delta_shift`, `opened`: the arrays that the row ranges index.
- `opens_arc`: for each arc row, `true` when the outage of that one branch removes all
  admittance of the arc.
- `member`: for each arc row, the position of the tripped member in the parallel group of
  the arc. Zero when the arc is not a parallel group.
- `ids`: the `IS.get_id` of each branch, mapped to its key `(type, name)`. Build a
  modification with `NetworkModification(table, id::Int)`.
- `n_arc`: the number of arcs in the file. Each `arc_index` is at most `n_arc`.
"""
struct BranchOutageTable
    rows::Dict{Tuple{String, String}, Tuple{Bool, UnitRange{Int}}}
    arc_index::Vector{Int}
    delta_b::Vector{Float64}
    delta_shift::Vector{Float64}
    opened::Vector{Int}
    opens_arc::Vector{Bool}
    member::Vector{Int}
    ids::Dict{Int, Tuple{String, String}}
    n_arc::Int
end

"Return the map from the key `(type, name)` to the islanding flag and the row range of `t`."
get_outage_rows(t::BranchOutageTable) = t.rows

"Return the map from `IS.get_id` of a branch to its key `(type, name)` in `t`."
get_outage_ids(t::BranchOutageTable) = t.ids

"Return the `opens_arc` column of `t`: `true` when the outage removes all admittance of the arc."
get_outage_opens_arc(t::BranchOutageTable) = t.opens_arc

"Return the `member` column of `t`: the position of the tripped member in the parallel group."
get_outage_member(t::BranchOutageTable) = t.member

function _check_row_values(
    group::String,
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    n_arc::Int,
)
    for (row, i) in enumerate(arc_index)
        1 <= i <= n_arc || error(
            "Factor file entry $group/arc_index has the value $i at row $row; " *
            "expected a value in 1:$n_arc.",
        )
    end
    for (row, b) in enumerate(delta_b)
        (isfinite(b) && !iszero(b) && abs(b) <= MAX_ARC_SUSCEPTANCE) || error(
            "Factor file entry $group/delta_b has the value $b in arc row $row. The value " *
            "must be finite, not zero, and at most $MAX_ARC_SUSCEPTANCE in magnitude.",
        )
    end
    for (row, x) in enumerate(delta_shift)
        isfinite(x) || error(
            "Factor file entry $group/delta_shift has the value $x at row $row; " *
            "the value must be finite.",
        )
    end
    for (row, x) in enumerate(opened)
        x >= 0 || error(
            "Factor file entry $group/opened has the value $x at row $row; " *
            "the value must not be negative.",
        )
    end
    return
end

function _check_row_ranges(
    group::String,
    starts::Vector{Int},
    lens::Vector{Int},
    n_rows::Int,
)
    for (k, s) in enumerate(starts)
        s >= 1 || error(
            "Factor file entry $group/start has the value $s at row $k; expected 1 or more.",
        )
        s <= n_rows + 1 || error(
            "Factor file entry $group/start has the value $s at row $k; " *
            "the group has $n_rows arc rows.",
        )
    end
    for (k, l) in enumerate(lens)
        l >= 0 || error(
            "Factor file entry $group/len has the value $l at row $k; " *
            "the value must not be negative.",
        )
        l <= n_rows || error(
            "Factor file entry $group/len has the value $l at row $k; " *
            "the group has $n_rows arc rows.",
        )
    end
    return
end

# `extra` holds the arc-row columns that only one group stores. They join the length check.
function _read_row_arrays(
    g::HDF5.Group,
    n_arc::Int,
    budget::ReadBudget,
    extra::Pair{String, <:AbstractVector}...,
)
    starts = _read(g, "start", Vector{Int}, budget)
    lens = _read(g, "len", Vector{Int}, budget)
    arc_index = _read(g, "arc_index", Vector{Int}, budget)
    delta_b = _read(g, "delta_b", Vector{Float64}, budget)
    delta_shift = _read(g, "delta_shift", Vector{Float64}, budget)
    opened = _read(g, "opened", Vector{Int}, budget)
    _check_same_length("rows", "start" => starts, "len" => lens)
    _check_same_length(
        "arc rows",
        "arc_index" => arc_index,
        "delta_b" => delta_b,
        "delta_shift" => delta_shift,
        "opened" => opened,
        extra...,
    )
    _check_row_values(HDF5.name(g), arc_index, delta_b, delta_shift, opened, n_arc)
    n_rows = length(arc_index)
    _check_row_ranges(HDF5.name(g), starts, lens, n_rows)
    ranges = [s:(s + l - 1) for (s, l) in zip(starts, lens)]
    for (k, range) in enumerate(ranges)
        (isempty(range) || checkbounds(Bool, arc_index, range)) ||
            error(
                "Factor file row range $k ($range) is outside the $(length(arc_index)) arc rows.",
            )
    end
    # Overlapping ranges would let a small file allocate one vector per range and row.
    next = 1
    for k in eachindex(ranges)
        starts[k] == next || error(
            "Factor file row range $k starts at $(starts[k]); " *
            "ranges must be contiguous and ordered, so it must start at $next.",
        )
        next += lens[k]
    end
    next == n_rows + 1 || error(
        "Factor file row ranges cover $(next - 1) of the $n_rows arc rows.",
    )
    for (k, range) in enumerate(ranges)
        seen = Set{Int}()
        for row in range
            arc_index[row] in seen && error(
                "Factor file row range $k has the arc_index $(arc_index[row]) more than once, at row $row.",
            )
            push!(seen, arc_index[row])
        end
    end
    return ranges, (; arc_index, delta_b, delta_shift, opened)
end

# An outage on an arc with zero susceptance divides by zero in the Woodbury factors.
function _check_outage_arcs(
    arc_index::Vector{Int},
    arc_susceptances::Vector{Float64},
)
    for (row, i) in enumerate(arc_index)
        iszero(arc_susceptances[i]) && error(
            "Factor file outage row $row is on arc $i, whose arc_susceptances entry is zero.",
        )
    end
    return
end

function _arc_modifications(
    arc_index::Vector{Int},
    delta_b::Vector{Float64},
    delta_shift::Vector{Float64},
    opened::Vector{Int},
    rows::Union{Vector{Int}, UnitRange{Int}},
)
    z = zero(YBUS_ELTYPE)
    return ArcModification[
        ArcModification(arc_index[i], delta_b[i], delta_shift[i], z, z, z, z, opened[i])
        for i in rows
    ]
end

"""
    from_hdf5(::Type{BranchOutageTable}, filename) -> BranchOutageTable

Read the branch outage table from a file that `to_hdf5(mat, filename, sys)` wrote. The
function throws an error if the file has no such table.
"""
function from_hdf5(::Type{BranchOutageTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        if !haskey(f, "branch_outages")
            error(
                "File $filename has no branch outage table. " *
                "The file was written without a system.",
            )
        end
        g = _group(f, "branch_outages")
        arc_susceptances = _read(f, "arc_susceptances", Vector{Float64}, budget)
        n_arc = length(_read(f, "arc_from", Vector{Int}, budget))
        length(arc_susceptances) == n_arc || error(
            "Factor file entry arc_susceptances has $(length(arc_susceptances)) values for $n_arc arcs.",
        )
        opens_arc = _read(g, "opens_arc", Vector{Bool}, budget)
        member = _read(g, "member", Vector{Int}, budget)
        ranges, arrays = _read_row_arrays(
            g,
            n_arc,
            budget,
            "opens_arc" => opens_arc,
            "member" => member,
        )
        _check_outage_arcs(arrays.arc_index, arc_susceptances)
        # A row that does not open its arc removes one group member, and the reader keys it
        # by that member.
        for (row, m) in enumerate(member)
            opens_arc[row] || m >= 1 ||
                error(
                    "Factor file entry branch_outages/member has the value $m at row $row, " *
                    "a partial row; expected 1 or more.",
                )
        end
        types = _read(g, "type", Vector{String}, budget)
        names = _read(g, "name", Vector{String}, budget)
        for t in types
            _branch_type(t, "branch_outages/type")
        end
        ids = _read(g, "id", Vector{Int}, budget)
        is_islanding = _read(g, "is_islanding", Vector{Bool}, budget)
        _check_same_length(
            "branch_outages",
            "start" => ranges,
            "type" => types,
            "name" => names,
            "id" => ids,
            "is_islanding" => is_islanding,
        )
        _check_unique("branch_outages/id", ids)
        _check_unique("branch_outages/(type, name)", collect(zip(types, names)))
        rows = Dict(zip(zip(types, names), zip(is_islanding, ranges)))
        id_keys = Dict(id => (t, n) for (id, t, n) in zip(ids, types, names))
        return BranchOutageTable(
            rows,
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            opens_arc,
            member,
            id_keys,
            n_arc,
        )
    end
end

"""
    NetworkModification(table::BranchOutageTable, type, name)
    NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})

Build the outage of one branch, or of several branches at once, from a
[`BranchOutageTable`](@ref). `type` is a `String`, for example
`"Line"`. The function groups the arc rows of all keys by arc:

  - If a row opens the whole arc, the arc gets the first such row only.
  - Otherwise the arc gets the first row of each tripped group member. Partial outages of
    different members add. Two lines of one chain in a parallel group remove that chain one
    time.

The modification has no Ybus deltas and no shunt modifications. It serves the DC row
kernels only.
"""
function NetworkModification(table::BranchOutageTable, keys::Vector{Tuple{String, String}})
    if isempty(keys)
        error("An outage needs at least one branch key.")
    end
    allunique(keys) || error(
        "Duplicate branch key in outage: $(collect(Set(k for k in keys if count(==(k), keys) > 1))).",
    )
    by_arc = DataStructures.OrderedDict{Int, Vector{Int}}()
    islanding = false
    for key in keys
        if !haskey(table.rows, key)
            error("Branch $(key[1]) $(key[2]) has no row in the outage table.")
        end
        is_islanding, range = table.rows[key]
        islanding |= is_islanding
        for i in range
            push!(get!(by_arc, table.arc_index[i], Int[]), i)
        end
    end
    picked = Int[]
    for rows in values(by_arc)
        append!(picked, _combined_rows(table, rows))
    end
    return NetworkModification(
        join((key[2] for key in keys), "+"),
        _arc_modifications(
            table.arc_index, table.delta_b, table.delta_shift, table.opened, picked,
        ),
        ShuntModification[],
        islanding,
    )
end

# `rows` holds the rows of one arc.
function _combined_rows(table::BranchOutageTable, rows::Vector{Int})
    for i in rows
        if table.opens_arc[i]
            return [i]
        end
    end
    return unique(i -> table.member[i], rows)
end

function NetworkModification(table::BranchOutageTable, type::String, name::String)
    return NetworkModification(table, [(type, name)])
end

"""
    NetworkModification(table::BranchOutageTable, id::Int)

Build the outage of the branch with `IS.get_id(branch) == id`. The function throws an error
if the table has no row for `id`.
"""
function NetworkModification(table::BranchOutageTable, id::Int)
    haskey(table.ids, id) || error("Branch with id $id has no row in the outage table.")
    return NetworkModification(table, [table.ids[id]])
end

function _group(parent::Union{HDF5.File, HDF5.Group}, name::String)
    path = string(rstrip(HDF5.name(parent), '/'), "/", name)
    haskey(parent, name) || error("Factor file has no entry $path.")
    return _checked_group(parent[name], path)
end

function _checked_group(g::Union{HDF5.File, HDF5.Group}, ::String)
    _check_hard_links(g)
    return g
end
function _checked_group(::HDF5.Dataset, name::String)
    return error("Factor file entry $name is a dataset; expected a group.")
end

function _read_registered_outages!(
    f::HDF5.File,
    cache::Dict{Int, ContingencySpec},
    arc_susceptances::Vector{Float64},
    budget::ReadBudget,
)
    g = _group(f, "outages")
    ranges, arrays = _read_row_arrays(g, length(arc_susceptances), budget)
    _check_outage_arcs(arrays.arc_index, arc_susceptances)
    ids = _read(g, "id", Vector{Int}, budget)
    labels = _read(g, "label", Vector{String}, budget)
    is_islanding = _read(g, "is_islanding", Vector{Bool}, budget)
    _check_same_length(
        "outages",
        "start" => ranges,
        "id" => ids,
        "label" => labels,
        "is_islanding" => is_islanding,
    )
    _check_unique("outages/id", ids)
    for (id, label, islanding, range) in zip(ids, labels, is_islanding, ranges)
        mods = _arc_modifications(
            arrays.arc_index,
            arrays.delta_b,
            arrays.delta_shift,
            arrays.opened,
            range,
        )
        cache[id] = ContingencySpec(
            id,
            NetworkModification(label, mods, ShuntModification[], islanding),
        )
    end
    return
end

function _check_format_version(a::HDF5.Attributes, budget::ReadBudget)
    version = _read(a, "format_version", Int, budget)
    if version != FACTOR_FILE_FORMAT_VERSION
        error(
            "Unsupported factor file format version $version; " *
            "this version of PowerNetworkMatrices reads version $FACTOR_FILE_FORMAT_VERSION.",
        )
    end
    return
end

"""
    from_hdf5(::Type{BranchArcTable}, filename) -> BranchArcTable

Read the `branch_map` group from a file that [`to_hdf5`](@ref) wrote. The function throws an
error if the format version is not 1, the group is missing, or a column breaks a rule of
the table.
"""
function from_hdf5(::Type{BranchArcTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
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
`outaged_rows` is out of range, or a parallel or series sum is zero.
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
            else
                iszero(sb[r]) &&
                    error(
                        "branch_flows: row $r (parallel group) has a zero susceptance sum.",
                    )
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
            iszero(arc_sb[-p]) &&
                error("branch_flows: arc $(-p) has a zero susceptance sum.")
            flows[r] =
                sign[r] * (b[r] / arc_sb[-p]) * (arc_flows[-p] + arc_ss[-p]) -
                b[r] * alpha[r]
        elseif kind[p] == 9
            flows[r] =
                sign[r] * (b[r] / sb[p]) * (flows[p] + ss[p]) - b[r] * alpha[r]
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

function _read_factor_core(f::HDF5.File, budget::ReadBudget; linear_solver::String)
    _check_hard_links(f)
    a = HDF5.attributes(f)
    _check_format_version(a, budget)
    bus_ax = _read(f, "bus_axis", Vector{Int}, budget)
    arc_from = _read(f, "arc_from", Vector{Int}, budget)
    arc_to = _read(f, "arc_to", Vector{Int}, budget)
    _check_same_length("arcs", "arc_from" => arc_from, "arc_to" => arc_to)
    arc_ax = collect(zip(arc_from, arc_to))

    reduced_from = _read(f, "reduced_bus_from", Vector{Int}, budget)
    reduced_to = _read(f, "reduced_bus_to", Vector{Int}, budget)
    _check_same_length(
        "reduced buses",
        "reduced_bus_from" => reduced_from,
        "reduced_bus_to" => reduced_to,
    )
    _check_unique("reduced_bus_from", reduced_from)
    bus_set = Set(bus_ax)
    for from in reduced_from
        from in bus_set &&
            error(
                "Factor file entry reduced_bus_from holds bus $from, which is on the bus axis.",
            )
    end
    nrd = _read_reduction(
        f,
        bus_ax,
        arc_ax,
        Dict{Int, Int}(zip(reduced_from, reduced_to)),
        budget,
    )
    name_to_arc = NAME_TO_ARC()
    g = _group(f, "names")
    types = _read(g, "type", Vector{String}, budget)
    names = _read(g, "name", Vector{String}, budget)
    name_from = _read(g, "from", Vector{Int}, budget)
    name_to = _read(g, "to", Vector{Int}, budget)
    _check_same_length(
        "names",
        "type" => types,
        "name" => names,
        "from" => name_from,
        "to" => name_to,
    )
    _check_unique("names/(type, name)", collect(zip(types, names)))
    arc_set = Set(arc_ax)
    for (row, (n, from, to)) in enumerate(zip(names, name_from, name_to))
        (from, to) in arc_set || error(
            "Factor file group names maps $n to ($from, $to) at row $row, which is not an arc.",
        )
    end
    for (t, n, from, to) in zip(types, names, name_from, name_to)
        T = _branch_type(t, "names/type")
        entries = get!(name_to_arc, T) do
            DataStructures.SortedDict{String, ARC_ENTRY}()
        end
        entries[n] = (from, to)
    end

    subnetwork_axes = Dict{Int, Tuple{Vector{Tuple{Int, Int}}, Vector{Int}}}()
    subnetworks = _group(f, "subnetworks")
    length(keys(subnetworks)) <= length(bus_ax) || error(
        "Factor file has $(length(keys(subnetworks))) subnetworks and only $(length(bus_ax)) buses.",
    )
    for k in keys(subnetworks)
        sg = _group(subnetworks, k)
        sub_from = _read(sg, "arc_from", Vector{Int}, budget)
        sub_to = _read(sg, "arc_to", Vector{Int}, budget)
        _check_same_length("subnetwork $k", "arc_from" => sub_from, "arc_to" => sub_to)
        subnetwork_axes[_read(sg, "ref", Int, budget)] =
            (collect(zip(sub_from, sub_to)), _read(sg, "bus", Vector{Int}, budget))
    end

    ref_positions = _read(f, "ref_bus_positions", Vector{Int}, budget)
    _check_unique("ref_bus_positions", ref_positions)
    core = VirtualFactorCore(
        _read(f, "arc_susceptances", Vector{Float64}, budget),
        Set{Int}(ref_positions),
        (arc_ax, bus_ax),
        subnetwork_axes,
        _read(f, "arc_live_members", Vector{Int}, budget),
        BranchCatalog(nrd, name_to_arc);
        linear_solver = linear_solver,
        tol = _read_tol(a, budget),
        system_uuid = _restore_system_uuid(a, budget),
    )
    return core
end

"""
    from_hdf5(::Type{VirtualPTDF}, filename; linear_solver, max_cache_size, persistent_arcs) -> VirtualPTDF

Read a `VirtualPTDF` from a file that [`to_hdf5`](@ref) wrote. The function does not need a
`System`. It factorizes the ABA matrix again with `linear_solver` and resolves the stored
`tol` as a fresh build does. The file holds no cache settings. `max_cache_size` and
`persistent_arcs` have the same meaning as in the `VirtualPTDF` constructor.

The restored matrix has no components and no reduction maps. `NetworkModification(mat, branch)`,
`NetworkModification(mat, arc)` and `NetworkModification(mat, sys, outage)` throw an error.
`get_branch_susceptances_by_arc` and `to_hdf5` throw an error. `mat[name, bus]` does not work.
The function throws an error if the format version of the file is not 1.
"""
function from_hdf5(
    ::Type{VirtualPTDF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        core = _read_factor_core(f, budget; linear_solver = linear_solver)
        bus_ax = core.axes[2]
        weights = _read(f, "dist_slack", Vector{Float64}, budget)
        _check_dist_slack(weights, bus_ax)
        dist_slack =
            Dict{Int, Float64}(
                bus_ax[i] => w for (i, w) in enumerate(weights) if !iszero(w)
            )
        return VirtualPTDF(
            core;
            dist_slack = dist_slack,
            max_cache_size = max_cache_size,
            persistent_arcs = persistent_arcs,
        )
    end
end

"""
    from_hdf5(::Type{VirtualMODF}, filename; linear_solver, max_cache_size) -> VirtualMODF

Read a `VirtualMODF` from a file that [`to_hdf5`](@ref) wrote. If the file has the group
`outages`, the restored matrix has those registered contingencies. Their modifications serve
the DC row kernels only. The restored matrix has the same limits as a restored `VirtualPTDF`:
`NetworkModification` from a branch or outage, `get_branch_susceptances_by_arc` and `to_hdf5`
throw an error.
"""
function from_hdf5(
    ::Type{VirtualMODF},
    filename::AbstractString;
    linear_solver::String = _default_linear_solver(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        core = _read_factor_core(f, budget; linear_solver = linear_solver)
        vmodf = VirtualMODF(core; max_cache_size = max_cache_size)
        if haskey(f, "outages")
            _read_registered_outages!(
                f,
                get_registered_contingencies(vmodf),
                core.arc_susceptances,
                budget,
            )
        end
        return vmodf
    end
end

const NO_ARC_HAS_ARC = 0
const NO_ARC_RADIAL = 1
const NO_ARC_ZERO_IMPEDANCE = 2
const NO_ARC_WARD = 3
const NO_ARC_NOT_IN_NETWORK = 4

# Reason a removed arc has no arc of its own. `_reason_of_arc` handles the degree-two code.
const _REASON_OF_REDUCTION = Dict{Int, Int}(
    REDUCTION_CODE_ZERO_IMPEDANCE => NO_ARC_ZERO_IMPEDANCE,
    REDUCTION_CODE_RADIAL => NO_ARC_RADIAL,
    REDUCTION_CODE_WARD => NO_ARC_WARD,
)

# The key of a removed arc is in the numbering of its step: the original numbers, or the
# numbers after the zero-impedance merge (that reduction runs first).
function _candidate_arcs(nr::NetworkReductionData, from::Int, to::Int)
    zf = get(nr.merged_bus_pairs, from, from)
    zt = get(nr.merged_bus_pairs, to, to)
    return ((from, to), (to, from), (zf, zt), (zt, zf))
end

function _has_removed_arc(nr::NetworkReductionData, from::Int, to::Int)
    candidates = _candidate_arcs(nr, from, to)
    return any(a -> haskey(nr.removed_arc_reduction, a), candidates)
end

function _removed_arc(nr::NetworkReductionData, from::Int, to::Int)
    candidates = _candidate_arcs(nr, from, to)
    for arc in candidates
        haskey(nr.removed_arc_reduction, arc) && return arc
    end
    error("No removed arc matches the buses ($from, $to).")
end

"""
    no_arc_reason(nr, arc_lookup, from, to) -> Int
    no_arc_reason(mat, from, to) -> Int

Tell why a branch between the bus numbers `from` and `to`, as the system document has them,
has or lacks an arc. The function returns one of these codes:

  - `NO_ARC_HAS_ARC` (0): the branch has an arc, alone or inside a parallel group or series chain.
  - `NO_ARC_RADIAL` (1): the radial reduction absorbed the branch. Call [`no_arc_surviving_bus`](@ref) for the bus that kept it.
  - `NO_ARC_ZERO_IMPEDANCE` (2): the zero-impedance reduction merged both ends into one bus.
  - `NO_ARC_WARD` (3): the Ward reduction eliminated the branch.
  - `NO_ARC_NOT_IN_NETWORK` (4): no record names the branch. It is out of service.

`arc_lookup` maps each arc of the arc axis to its index, as `get_arc_lookup` returns. The
function reads only records that a factor file stores, so a matrix that `from_hdf5` restored
gives the same answer as the matrix that was written. The pair of buses identifies the
branch: a branch out of service in parallel with a branch in service has the code of the
latter. The function throws an error if `from` equals `to`, or if a bus is not a bus of the
matrix, a reduced bus or a removed bus.
"""
function no_arc_reason(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    from::Int,
    to::Int,
)
    from == to &&
        error("no_arc_reason: the buses of a branch must differ; got $from twice.")
    _check_query_bus(nr, "from", from)
    _check_query_bus(nr, "to", to)
    if _has_removed_arc(nr, from, to)
        return _reason_of_arc(nr, arc_lookup, _removed_arc(nr, from, to))
    end
    zf = get(nr.merged_bus_pairs, from, from)
    zt = get(nr.merged_bus_pairs, to, to)
    # A branch with impedance between two merged buses is a self-loop: no record names it.
    zf == zt && return NO_ARC_ZERO_IMPEDANCE
    if haskey(arc_lookup, (zf, zt)) || haskey(arc_lookup, (zt, zf))
        return NO_ARC_HAS_ARC
    end
    return NO_ARC_NOT_IN_NETWORK
end

function _check_query_bus(nr::NetworkReductionData, name::String, bus::Int)
    if haskey(nr.bus_reduction_map, bus) || haskey(nr.reverse_bus_search_map, bus) ||
       bus in nr.removed_buses || haskey(nr.merged_bus_pairs, bus)
        return
    end
    error(
        "no_arc_reason: bus $bus ($name) is not a bus of the matrix, a reduced bus or a removed bus.",
    )
end

# A degree-two member has an arc while its composite arc survives. A later reduction can
# remove the composite arc, and then that record gives the reason.
function _reason_of_arc(
    nr::NetworkReductionData,
    arc_lookup::Dict{Tuple{Int, Int}, Int},
    arc::Tuple{Int, Int},
)
    code = nr.removed_arc_reduction[arc]
    code == REDUCTION_CODE_DEGREE_TWO || return _REASON_OF_REDUCTION[code]
    composite = nr.removed_arc_composite[arc]
    reverse_composite = reverse(composite)
    if haskey(arc_lookup, composite) || haskey(arc_lookup, reverse_composite)
        return NO_ARC_HAS_ARC
    end
    for key in (composite, reverse_composite)
        haskey(nr.removed_arc_reduction, key) &&
            return _reason_of_arc(nr, arc_lookup, key)
    end
    error(
        "The degree-two arc $arc has the composite arc $composite, which no record explains.",
    )
end

# The record that last removed the arc: a degree-two member leads to its composite arc when a
# later reduction removed that arc.
function _terminal_arc(nr::NetworkReductionData, arc::Tuple{Int, Int})
    nr.removed_arc_reduction[arc] == REDUCTION_CODE_DEGREE_TWO || return arc
    composite = nr.removed_arc_composite[arc]
    for key in (composite, reverse(composite))
        haskey(nr.removed_arc_reduction, key) && return _terminal_arc(nr, key)
    end
    return arc
end

function no_arc_reason(mat::PowerNetworkMatrix, from::Int, to::Int)
    return no_arc_reason(get_network_reduction_data(mat), get_arc_lookup(mat), from, to)
end

"""
    no_arc_surviving_bus(nr, from, to) -> Int

The bus that kept the branch between `from` and `to` when the radial reduction absorbed it, in
the numbering after the later reductions. A later degree-two reduction can remove that bus.
The function throws an error if the radial reduction did not absorb the branch.
"""
function no_arc_surviving_bus(nr::NetworkReductionData, from::Int, to::Int)
    _has_removed_arc(nr, from, to) ||
        error("No reduction removed a branch between buses ($from, $to).")
    arc = _terminal_arc(nr, _removed_arc(nr, from, to))
    nr.removed_arc_reduction[arc] == REDUCTION_CODE_RADIAL ||
        error("The radial reduction did not remove the branch between buses ($from, $to).")
    # A cascaded stub arc, where the peel removed both ends, has no stored survivor.
    return get_mapped_bus_number(nr, get(nr.removed_arc_to_surviving_bus, arc, first(arc)))
end

function no_arc_surviving_bus(mat::PowerNetworkMatrix, from::Int, to::Int)
    return no_arc_surviving_bus(get_network_reduction_data(mat), from, to)
end
