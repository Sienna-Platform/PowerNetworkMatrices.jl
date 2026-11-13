const FACTOR_FILE_FORMAT_VERSION = 1

# Columns of one table must agree in length, or `zip` would drop rows in silence.
function _check_same_length(table::String, columns::Pair{String, <:AbstractVector}...)
    lengths = map(column -> first(column) => length(last(column)), columns)
    allequal(map(last, lengths)) ||
        error("Factor file table $table has columns of unequal length: $lengths.")
    return
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
function _checked_group(::HDF5.Datatype, name::String)
    return error("Factor file entry $name is a committed datatype; expected a group.")
end

function _check_format_version(a::HDF5.Attributes, budget::ReadBudget)
    version = _read(a, "format_version", Int, budget)
    if version != FACTOR_FILE_FORMAT_VERSION
        error(
            "Unsupported factor file format version $version; " *
            "this version of PowerMatricesCore reads version $FACTOR_FILE_FORMAT_VERSION.",
        )
    end
    return
end

# Every bus that has no row in the ABA matrix is a reference bus.
_ref_bus_positions(core::FactorCore) = setdiff(1:length(core.axes[2]), core.valid_ix)

"""
    write_factor_core(f::HDF5.File, core::FactorCore; shift_injection, dist_slack,
        ref_bus_positions, reverse_bus_map)

Write the root attributes and the datasets that `read_factor_core` needs:
`format_version`, `system_uuid`, the `tol` attributes, `bus_axis`, `arc_from`, `arc_to`,
`ref_bus_positions`, `arc_susceptances`, `arc_live_members`, `shift_injection`, `dist_slack`,
`reduced_bus_from`, `reduced_bus_to`, and the group `subnetworks`. `shift_injection` holds one
value for each arc. `dist_slack` is empty or holds one weight for each bus. The function checks
the arrays before it writes. A subnetwork key that is not on the bus axis resolves through
`reverse_bus_map`, which the function writes as `reduced_bus_from` and `reduced_bus_to`.
"""
function write_factor_core(
    f::HDF5.File,
    core::FactorCore;
    shift_injection::Vector{Float64},
    dist_slack::Vector{Float64} = Float64[],
    ref_bus_positions::Vector{Int} = _ref_bus_positions(core),
    reverse_bus_map::Dict{Int, Int} = Dict{Int, Int}(),
)
    arc_ax, bus_ax = core.axes
    signed_susceptances = core.arc_susceptances .* core.arc_susceptance_signs
    _validate_dc_arrays(
        signed_susceptances,
        Set(ref_bus_positions),
        core.axes,
        core.subnetwork_axes,
        core.arc_live_members,
        reverse_bus_map,
    )
    _check_same_length("arcs", "arc_from" => arc_ax, "shift_injection" => shift_injection)
    _check_dist_slack(dist_slack, bus_ax)
    a = HDF5.attributes(f)
    a["format_version"] = FACTOR_FILE_FORMAT_VERSION
    _write_system_uuid!(a, get_system_uuid(core))
    _write_tol!(a, get_tol_spec(core))
    f["bus_axis"] = bus_ax
    f["arc_from"] = first.(arc_ax)
    f["arc_to"] = last.(arc_ax)
    f["ref_bus_positions"] = sort(ref_bus_positions)
    f["arc_susceptances"] = signed_susceptances
    f["arc_live_members"] = core.arc_live_members
    f["shift_injection"] = shift_injection
    f["dist_slack"] = dist_slack
    f["reduced_bus_from"] = collect(keys(reverse_bus_map))
    f["reduced_bus_to"] = collect(values(reverse_bus_map))

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

"""
    read_factor_arrays(f::HDF5.File, budget::ReadBudget) -> NamedTuple

Read and check the core arrays of a factor file. The result has the fields `axes`,
`arc_susceptances`, `ref_bus_positions`, `subnetwork_axes`, `arc_live_members`,
`reverse_bus_map`, `tol`, and `system_uuid`, in the form that the arc-susceptance `FactorCore`
constructor takes. The function checks the format version first.
"""
function read_factor_arrays(f::HDF5.File, budget::ReadBudget)
    _check_hard_links(f)
    a = HDF5.attributes(f)
    _check_format_version(a, budget)
    bus_ax = _read(f, "bus_axis", Vector{Int}, budget)
    arc_from = _read(f, "arc_from", Vector{Int}, budget)
    arc_to = _read(f, "arc_to", Vector{Int}, budget)
    _check_same_length("arcs", "arc_from" => arc_from, "arc_to" => arc_to)
    arc_ax = collect(zip(arc_from, arc_to))

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
    return (
        axes = (arc_ax, bus_ax),
        arc_susceptances = _read(f, "arc_susceptances", Vector{Float64}, budget),
        ref_bus_positions = Set{Int}(ref_positions),
        subnetwork_axes = subnetwork_axes,
        arc_live_members = _read(f, "arc_live_members", Vector{Int}, budget),
        reverse_bus_map = Dict{Int, Int}(zip(reduced_from, reduced_to)),
        tol = _read_tol(a, budget),
        system_uuid = _restore_system_uuid(a, budget),
    )
end

"""
    read_factor_core(f::HDF5.File, budget = ReadBudget(HDF5.filename(f));
        network_data = nothing, linear_solver) -> FactorCore
    read_factor_core(path::AbstractString; network_data = nothing, linear_solver) -> FactorCore

Read the core arrays of a factor file and build a `FactorCore` from them. The function
factorizes the ABA matrix again with `linear_solver`, a solver name or a `LinearSolverType`
value, and checks the factorization. The core
holds `network_data` and does not read it. A subnetwork key that is not on the bus axis
resolves through the map in `reduced_bus_from` and `reduced_bus_to`.
"""
function read_factor_core(
    f::HDF5.File,
    budget::ReadBudget = ReadBudget(HDF5.filename(f));
    network_data = nothing,
    linear_solver::Union{String, LinearSolverType} = _default_linear_solver(),
)
    return _factor_core(
        read_factor_arrays(f, budget);
        network_data = network_data,
        linear_solver = linear_solver,
    )
end

function _factor_core(
    arrays::NamedTuple;
    network_data = nothing,
    linear_solver::Union{String, LinearSolverType} = _default_linear_solver(),
)
    return FactorCore(
        arrays.arc_susceptances,
        arrays.ref_bus_positions,
        arrays.axes,
        arrays.subnetwork_axes,
        arrays.arc_live_members;
        network_data = network_data,
        reverse_bus_map = arrays.reverse_bus_map,
        linear_solver = linear_solver,
        tol = arrays.tol,
        system_uuid = arrays.system_uuid,
    )
end

function read_factor_core(
    path::AbstractString;
    network_data = nothing,
    linear_solver::Union{String, LinearSolverType} = _default_linear_solver(),
)
    return HDF5.h5open(path, "r") do f
        read_factor_core(
            f,
            ReadBudget(path);
            network_data = network_data,
            linear_solver = linear_solver,
        )
    end
end

"""
    read_shift_injection(f::HDF5.File, core::FactorCore[, budget]) -> Vector{Float64}

Read the DC phase shift injection of each arc of `core`. The function checks that the dataset
has one finite value for each arc. The default `budget` is a new `ReadBudget` for the file.
One budget for all reads of a file bounds their total bytes.
"""
function read_shift_injection(
    f::HDF5.File,
    core::FactorCore,
    budget::ReadBudget = ReadBudget(HDF5.filename(f)),
)
    shift = _read(f, "shift_injection", Vector{Float64}, budget)
    _check_same_length(
        "arcs",
        "arc_from" => get_arc_axis(core),
        "shift_injection" => shift,
    )
    for (i, s) in enumerate(shift)
        isfinite(s) || error(
            "Factor file entry shift_injection has the value $s at position $i; " *
            "the value must be finite.",
        )
    end
    return shift
end

"""
    read_dist_slack(f::HDF5.File, core::FactorCore[, budget]) -> Vector{Float64}

Read the distributed slack weights of the buses of `core`. The result is empty if the file
holds no weights. Otherwise it has one non-negative weight for each bus and a positive sum.
The default `budget` is a new `ReadBudget` for the file. One budget for all reads of a file
bounds their total bytes.
"""
function read_dist_slack(
    f::HDF5.File,
    core::FactorCore,
    budget::ReadBudget = ReadBudget(HDF5.filename(f)),
)
    weights = _read(f, "dist_slack", Vector{Float64}, budget)
    _check_dist_slack(weights, get_bus_axis(core))
    return weights
end

function _read_branch_names(
    f::HDF5.File,
    arc_ax::Vector{Tuple{Int, Int}},
    budget::ReadBudget,
)
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
    return Dict{Tuple{String, String}, Tuple{Int, Int}}(
        (t, n) => (from, to) for
        (t, n, from, to) in zip(types, names, name_from, name_to)
    )
end

"""
    read_branch_names(path::AbstractString) -> Dict{Tuple{String, String}, Tuple{Int, Int}}

Read the group `names` of a factor file: the arc `(from, to)` of each entry of the group,
under the key `(type, name)`. Every arc is on the arc axis, and `get_arc_lookup(core)` gives
its index. Several entries can have the same arc. The function does not factorize the
network. It throws an error if the file has no such group. The `type` of a key is a `String`
that the function does not interpret.
"""
function read_branch_names(path::AbstractString)
    return HDF5.h5open(path, "r") do f
        budget = ReadBudget(path)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        arc_from = _read(f, "arc_from", Vector{Int}, budget)
        arc_to = _read(f, "arc_to", Vector{Int}, budget)
        _check_same_length("arcs", "arc_from" => arc_from, "arc_to" => arc_to)
        return _read_branch_names(f, collect(zip(arc_from, arc_to)), budget)
    end
end

"""
    FactorFile

The handle type of an open factor file. The functions that read or write a group of a factor
file take a `FactorFile`.
"""
const FactorFile = HDF5.File

"""
    open_factor_file(body, filename::AbstractString)

Open the factor file `filename` for reading, run `body(f)` with the handle `f`, and close the
file. Return the value of `body`. The function does not check the content of the file.
"""
function open_factor_file(body, filename::AbstractString)
    return HDF5.h5open(body, filename, "r")
end

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
    write_branch_names!(f::FactorFile, types, names, from, to)

Write the group `names` of the factor file `f`. Row `i` maps the branch `(types[i], names[i])`
to the arc `(from[i], to[i])`. The four vectors have the same length. The function throws an
error if a key `(type, name)` repeats.
"""
function write_branch_names!(
    f::FactorFile,
    types::Vector{String},
    names::Vector{String},
    from::Vector{Int},
    to::Vector{Int},
)
    _check_same_length(
        "names",
        "type" => types,
        "name" => names,
        "from" => from,
        "to" => to,
    )
    _check_unique("names/(type, name)", collect(zip(types, names)))
    g = HDF5.create_group(f, "names")
    _write_strings!(g, "type", types)
    _write_strings!(g, "name", names)
    g["from"] = from
    g["to"] = to
    return
end
