module HDF5Ext

import HDF5
import PowerNetworkMatrices as PNM
import PowerNetworkMatrices: to_hdf5, from_hdf5
using PowerNetworkMatrices:
    ARC_ENTRY,
    ArcModification,
    AutoTolerance,
    BranchArcTable,
    BranchCatalog,
    BranchOutageTable,
    BranchReasonTable,
    ContingencySpec,
    MAX_CACHE_SIZE_MiB,
    NAME_TO_ARC,
    NetworkModification,
    NetworkReductionData,
    REDUCTION_CODE_DEGREE_TWO,
    ShuntModification,
    ThreeWindingTransformerCircuit,
    UnsupportedOutageError,
    VirtualFactorCore,
    VirtualMODF,
    VirtualPTDF,
    _REDUCTION_COLUMNS,
    _arc_modifications,
    _build_branch_arc_table,
    _check_branch_map,
    _check_branch_reason,
    _check_dist_slack,
    _check_outage_arcs,
    _check_reduction_records,
    _check_row_ranges,
    _check_row_values,
    _check_same_length,
    _check_unique,
    _default_linear_solver,
    _outaged_group_member,
    _reduction_records,
    _resolve_branch_arc,
    _validate_dc_arrays,
    arc_dc_shift_injection,
    get_arc_lookup,
    get_branch_catalog,
    get_branch_susceptances_by_arc,
    get_core,
    get_dist_slack,
    get_equivalent_available,
    get_name_to_arc_maps,
    get_network_reduction_data,
    get_ref_bus_position,
    get_registered_contingencies,
    get_reverse_bus_search_map,
    get_system_uuid,
    get_tol_spec,
    holds_components

const PSY = PNM.PSY
const IS = PNM.IS
const DataStructures = PNM.DataStructures

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
    to_hdf5(mat::Union{VirtualPTDF, VirtualMODF}, filename, sys::PSY.System; force = false, outages = true)

Write the factor data of `mat` as the method without `sys` does. Also write the group
`branch_reason` with the [`BranchReasonTable`](@ref) of every AC branch of `sys`. If `outages`
is `true`, also write the group
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
    outages::Bool = true,
)
    return _write_factor_file_atomically(filename, force) do f
        _write_factor_core(f, get_core(mat), _dist_slack(mat))
        _write_branch_reason!(f, BranchReasonTable(mat, sys))
        if outages
            _write_branch_outages(f, mat, sys)
        end
        _write_registered_outages(f, mat)
    end
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

function _write_branch_reason!(f::HDF5.File, t::BranchReasonTable)
    rows = sort!(collect(t.rows); by = first)
    id = [first(first(r)) for r in rows]
    winding = [last(first(r)) for r in rows]
    reason = [first(last(r)) for r in rows]
    surviving_bus = [last(last(r)) for r in rows]
    _check_branch_reason(id, winding, reason, surviving_bus)
    g = HDF5.create_group(f, "branch_reason")
    g["id"] = id
    g["winding"] = winding
    g["reason"] = reason
    g["surviving_bus"] = surviving_bus
    return
end

"""
    from_hdf5(::Type{BranchReasonTable}, filename) -> BranchReasonTable

Read the `branch_reason` group that `to_hdf5(mat, filename, sys)` wrote. The function throws an
error if the file has no such group, because a file written without a `System` cannot name
its branches. It also throws an error if the format version is not 1 or a column breaks a
rule of the table.
"""
function from_hdf5(::Type{BranchReasonTable}, filename::AbstractString)
    return HDF5.h5open(filename, "r") do f
        budget = ReadBudget(filename)
        _check_hard_links(f)
        _check_format_version(HDF5.attributes(f), budget)
        haskey(f, "branch_reason") || error(
            "Factor file $filename has no group branch_reason: it was written without a " *
            "System. Write it with `to_hdf5(mat, filename, sys)`.",
        )
        g = _group(f, "branch_reason")
        id = _read(g, "id", Vector{Int}, budget)
        winding = _read(g, "winding", Vector{Int}, budget)
        reason = _read(g, "reason", Vector{Int}, budget)
        surviving_bus = _read(g, "surviving_bus", Vector{Int}, budget)
        _check_branch_reason(id, winding, reason, surviving_bus)
        return BranchReasonTable(
            Dict(zip(zip(id, winding), zip(reason, surviving_bus))),
        )
    end
end

end
