"""
The Virtual Line Outage Distribution Factor (VirtualLODF) structure gathers
the rows of the LODF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualLODF is initialized with no row stored.

The VirtualLODF struct is indexed using branch names.

# Arguments
- `K::KLU.KLUFactorization{Float64, Int}`:
        LU factorization matrices of the ABA matrix, evaluated by means of KLU.
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix.
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence matrix.
- `inv_PTDF_A_diag::Vector{Float64}`:
        Vector contiaining the element-wise reciprocal of the diagonal elements
        coming from multuiplying the PTDF matrix with th Incidence matrix
- `ref_bus_positions::Set{Int}`:
        Vector containing the indexes of the rows of the transposed BA matrix
        corresponding to the reference buses.
- `dist_slack::Vector{Float64}`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `axes<:NTuple{2, Dict}`:
        Tuple containing two vectors showing the branch names.
- `lookup<:NTuple{2, Dict}`:
        Tuple containing two dictionaries, mapping the branches names
        the enumerated row indexes indexes.
- `valid_ix::Vector{Int}`:
        Vector containing the row/columns indices of matrices related the buses
        which are not slack ones.
- `temp_data::Vector{Float64}`:
        Temporary vector for internal use.
- `cache::RowCache`:
        Cache were LODF rows are stored.
- `subnetworks::Dict{Int, Set{Int}}`:
        Dictionary containing the subsets of buses defining the different subnetwork of the system.
- `tol::Base.RefValue{Float64}`:
        Tolerance related to scarification and values to drop.
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
"""
struct VirtualLODF{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{Float64}
    K::KLU.KLUFactorization{Float64, Int}
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    A::SparseArrays.SparseMatrixCSC{Int8, Int}
    inv_PTDF_A_diag::Vector{Float64}
    dist_slack::Vector{Float64}
    axes::Ax
    lookup::L
    valid_ix::Vector{Int}
    temp_data::Vector{Float64}
    cache::RowCache
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    network_reduction_data::NetworkReductionData
    work_ba_col::Vector{Float64}
end

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualLODF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

function _get_PTDF_A_diag(
    K::KLU.KLUFactorization{Float64, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    ref_bus_positions::Set{Int},
)
    n_branches = size(BA, 2)
    n_buses = size(BA, 1)
    diag_ = zeros(n_branches)

    # Pre-compute valid indices (non-reference buses)
    valid_ix = setdiff(1:n_buses, ref_bus_positions)
    n_valid = length(valid_ix)

    # Pre-allocate work arrays for efficiency
    ba_col = zeros(n_valid)
    ptdf_row = zeros(n_buses)

    # For each branch, compute PTDF row and dot with incidence column
    for i in 1:n_branches
        # Extract BA column for valid indices
        fill!(ba_col, 0.0)
        for idx in 1:n_valid
            bus_idx = valid_ix[idx]
            ba_col[idx] = BA[bus_idx, i]
        end

        # Solve for PTDF row: ptdf_row_valid = ABA^(-1) * ba_col
        ptdf_row_valid = K \ ba_col

        # Map back to full bus indices
        fill!(ptdf_row, 0.0)
        for idx in 1:n_valid
            ptdf_row[valid_ix[idx]] = ptdf_row_valid[idx]
        end

        # Compute diagonal element: sum of PTDF[i,j] * A[i,j] for all buses j
        for j in 1:n_buses
            diag_[i] += ptdf_row[j] * A[i, j]
        end
    end
    return diag_
end

"""
Builds the Virtual LODF matrix from a system. The return is a VirtualLODF
struct with an empty cache.

# Arguments
- `sys::PSY.System`:
        PSY system for which the matrix is constructed

# Keyword Arguments
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `kwargs...`:
        other keyword arguments used by VirtualPTDF
"""
function VirtualLODF(
    sys::PSY.System;
    dist_slack::Vector{Float64} = Float64[],
    tol::Float64 = eps(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    if length(dist_slack) != 0
        @info "Distributed bus"
    end
    Ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    ref_bus_positions = get_ref_bus_position(Ymatrix)
    A = IncidenceMatrix(Ymatrix)
    arc_ax = get_arc_axis(A)
    axes = (arc_ax, arc_ax)
    arc_ax_ref = make_ax_ref(arc_ax)
    look_up = (arc_ax_ref, arc_ax_ref)
    subnetwork_axes = make_arc_arc_subnetwork_axes(A)
    BA = BA_Matrix(Ymatrix)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    K = klu(ABA)
    bus_ax = get_bus_axis(A)

    temp_data = zeros(length(bus_ax))
    valid_ix = setdiff(1:length(bus_ax), ref_bus_positions)
    PTDF_diag = _get_PTDF_A_diag(
        K,
        BA.data,
        A.data,
        Set(ref_bus_positions),
    )
    PTDF_diag[PTDF_diag .> 1 - LODF_ENTRY_TOLERANCE] .= 0.0

    if isempty(persistent_arcs)
        empty_cache =
            RowCache(max_cache_size * MiB, Set{Int}(), length(bus_ax) * sizeof(Float64))
    else
        init_persistent_dict = Set{Int}(A.lookup[1][k] for k in persistent_arcs)
        empty_cache =
            RowCache(
                max_cache_size * MiB,
                init_persistent_dict,
                length(bus_ax) * sizeof(Float64),
            )
    end

    # Pre-allocate work array for BA column extraction
    work_ba_col = zeros(length(valid_ix))

    return VirtualLODF(
        K,
        BA.data,
        A.data,
        1.0 ./ (1.0 .- PTDF_diag),
        dist_slack,
        axes,
        look_up,
        valid_ix,
        temp_data,
        empty_cache,
        subnetwork_axes,
        Ref(tol),
        Ymatrix.network_reduction_data,
        work_ba_col,
    )
end

# Overload Base functions

"""
Checks if the any of the fields of VirtualLODF is empty.
"""
function Base.isempty(vlodf::VirtualLODF)
    for name in fieldnames(typeof(vlodf))
        # note: impossible to define empty KLU field
        if name != :K && isempty(getfield(vlodf, name))
            @debug "Field " * string(name) * " is empty."
            return true
        end
    end
    return false
end

"""
Shows the size of the whole LODF matrix, not the number of rows stored.
"""
Base.size(vlodf::VirtualLODF) = (size(vlodf.BA, 2), size(vlodf.BA, 2))

"""
Gives the cartesian indexes of the LODF matrix.
"""
Base.eachindex(vlodf::VirtualLODF) = CartesianIndices(size(vlodf))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualLODF) = "VirtualLODF"
end

function _getindex(
    vlodf::VirtualLODF,
    row::Int,
    column::Union{Int, Colon},
)
    # check if value is in the cache
    if haskey(vlodf.cache, row)
        return vlodf.cache.temp_cache[row][column]
    else
        # evaluate the value for the LODF column
        # Use pre-allocated work array instead of collect() to reduce allocations
        @inbounds for i in eachindex(vlodf.valid_ix)
            vlodf.work_ba_col[i] = vlodf.BA[vlodf.valid_ix[i], row]
        end
        lin_solve = KLU.solve!(vlodf.K, vlodf.work_ba_col)

        # get full lodf row
        @inbounds for i in eachindex(vlodf.valid_ix)
            vlodf.temp_data[vlodf.valid_ix[i]] = lin_solve[i]
        end

        # now get the LODF row
        lodf_row = (vlodf.A * vlodf.temp_data) .* vlodf.inv_PTDF_A_diag
        lodf_row[row] = -1.0

        if get_tol(vlodf) > eps()
            vlodf.cache[row] = sparsify(lodf_row, get_tol(vlodf))
        else
            vlodf.cache[row] = copy(lodf_row)
        end
        return vlodf.cache[row][column]
    end
end

"""
Gets the value of the element of the LODF matrix given the row and column indices
corresponding to the selected and outage branch respectively. If `column` is a Colon then
the entire row is returned.

# Arguments
- `vlodf::VirtualLODF`:
        VirtualLODF struct where to evaluate and store the row values.
- `row`:
        selected line name
- `column`:
        outage line name. If `Colon` then get the values of the whole row.
"""
function Base.getindex(vlodf::VirtualLODF, row, column)
    row_, column_ = to_index(vlodf, row, column)
    return _getindex(vlodf, row_, column_)
end

# Define for ambiguity resolution
function Base.getindex(vlodf::VirtualLODF, row::Integer, column::Integer)
    return _getindex(vlodf, row, column)
end

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualLODF, _, idx...) = error("Operation not supported by VirtualLODF")

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualLODF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualLODF")

get_lodf_data(mat::VirtualLODF) = mat.cache.temp_cache

function get_arc_axis(mat::VirtualLODF)
    return mat.axes[1]
end

""" Gets the tolerance used for sparsifying the rows of the VirtualLODF matrix"""
function get_tol(mat::VirtualLODF)
    return mat.tol[]
end
