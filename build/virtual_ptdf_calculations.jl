"""
The Virtual Power Transfer Distribution Factor (VirtualPTDF) structure gathers
the rows of the PTDF matrix as they are evaluated on-the-go. These rows are
evaluated independently, cached in the structure and do not require the
computation of the whole matrix (therefore significantly reducing the
computational requirements).

The VirtualPTDF is initialized with no row stored.

The VirtualPTDF is indexed using branch names and bus numbers as for the PTDF
matrix.

# Arguments
- `K::Union{KLU.KLUFactorization{Float64, Int}, AppleAccelerate.AAFactorization{Float64}}`:
        LU factorization matrices of the ABA matrix, evaluated by means of KLU or AppleAccelerate
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        BA matrix
- `ref_bus_positions::Set{Int}`:
        Vector containing the indexes of the columns of the BA matrix corresponding
        to the reference buses
- `dist_slack::Vector{Float64}`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `axes<:NTuple{2, Dict}`:
        Tuple containing two vectors: the first one showing the branches names,
        the second showing the buses numbers. There is no link between the
        order of the vector of the branches names and the way the PTDF rows are
        stored in the cache.
- `lookup<:NTuple{2, Dict}`:
        Tuple containing two dictionaries, mapping the branches
        and buses with their enumerated indexes. The branch indexes refer to
        the key of the cache dictionary. The bus indexes refer to the position
        of the elements in the PTDF row stored.
- `temp_data::Vector{Float64}`:
        Temporary vector for internal use.
- `valid_ix::Vector{Int}`:
        Vector containing the row/columns indices of matrices related the buses
        which are not slack ones.
- `cache::RowCache`:
        Cache were PTDF rows are stored.
- `subnetworks::Dict{Int, Set{Int}}`:
        Dictionary containing the subsets of buses defining the different subnetwork of the system.
- `tol::Base.RefValue{Float64}`:
        Tolerance related to scarification and values to drop.
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
"""
struct VirtualPTDF{Ax, L <: NTuple{2, Dict}, K <: LinearAlgebra.Factorization} <:
       PowerNetworkMatrix{Float64}
    K::K
    BA::SparseArrays.SparseMatrixCSC{Float64, Int}
    dist_slack::Vector{Float64}
    dist_slack_normalized::Vector{Float64}
    axes::Ax
    lookup::L
    temp_data::Vector{Float64}
    valid_ix::Vector{Int}
    cache::RowCache
    subnetwork_axes::Dict{Int, Ax}
    tol::Base.RefValue{Float64}
    network_reduction_data::NetworkReductionData
    work_ba_col::Vector{Float64}
end

get_axes(M::VirtualPTDF) = M.axes
get_lookup(M::VirtualPTDF) = M.lookup
get_ref_bus(M::VirtualPTDF) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::VirtualPTDF) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::VirtualPTDF) = M.network_reduction_data
get_bus_lookup(M::VirtualPTDF) = M.lookup[2]
get_arc_lookup(M::VirtualPTDF) = M.lookup[1]

function Base.show(io::IO, ::MIME{Symbol("text/plain")}, array::VirtualPTDF)
    summary(io, array)
    isempty(array) && return
    println(io, ":")
    Base.print_array(io, array)
    return
end

"""
Builds the Virtual PTDF matrix from a system. The return is a VirtualPTDF
struct with an empty cache.

# Arguments
- `sys::PSY.System`:
        PSY system for which the matrix is constructed

# Keyword Arguments
- `dist_slack::Vector{Float64} = Float64[]`:
        Vector of weights to be used as distributed slack bus.
        The distributed slack vector has to be the same length as the number of buses.
- `linear_solver::String = "KLU"`:
        Linear solver to use for factorization. Options: "KLU", "AppleAccelerate"
- `tol::Float64 = eps()`:
        Tolerance related to sparsification and values to drop.
- `max_cache_size::Int`:
        max cache size in MiB (initialized as MAX_CACHE_SIZE_MiB).
- `persistent_lines::Vector{String}`:
        line to be evaluated as soon as the VirtualPTDF is created (initialized as empty vector of strings).
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `kwargs...`:
        other keyword arguments used by VirtualPTDF
"""
function VirtualPTDF(
    sys::PSY.System;
    dist_slack::Dict{Int, Float64} = Dict{Int, Float64}(),
    linear_solver::String = "KLU",
    tol::Float64 = eps(),
    max_cache_size::Int = MAX_CACHE_SIZE_MiB,
    persistent_arcs::Vector{Tuple{Int, Int}} = Vector{Tuple{Int, Int}}(),
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    validate_linear_solver(linear_solver)
    Ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    ref_bus_positions = get_ref_bus_position(Ymatrix)
    A = IncidenceMatrix(Ymatrix)
    if !(isempty(dist_slack))
        dist_slack_vector = redistribute_dist_slack(dist_slack, A, A.network_reduction_data)
    else
        dist_slack_vector = Float64[]
    end
    BA = BA_Matrix(Ymatrix)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    bus_ax = get_bus_axis(A)
    axes = A.axes
    look_up = A.lookup
    subnetwork_axes = A.subnetwork_axes
    if length(subnetwork_axes) > 1
        @info "Network is not connected, using subnetworks"
    end
    temp_data = zeros(length(axes[2]))

    if isempty(persistent_arcs)
        empty_cache =
            RowCache(max_cache_size * MiB, Set{Int}(), length(bus_ax) * sizeof(Float64))
    else
        init_persistent_dict = Set{Int}(look_up[1][k] for k in persistent_arcs)
        empty_cache =
            RowCache(
                max_cache_size * MiB,
                init_persistent_dict,
                length(bus_ax) * sizeof(Float64),
            )
    end

    # Create factorization based on solver choice
    if linear_solver == "KLU"
        K = klu(ABA)
    elseif linear_solver == "AppleAccelerate"
        if !_has_apple_accelerate_ext()
            error(
                "AppleAccelerate extension is not available. This solver is only available on macOS. Install AppleAccelerate: using Pkg; Pkg.add(\"AppleAccelerate\")",
            )
        end
        # This will use the AAFactorization type from the extension
        K = _create_apple_accelerate_factorization(ABA)
    else
        error("Unsupported linear solver: $linear_solver")
    end

    # Pre-compute normalized distributed slack for efficiency
    if !isempty(dist_slack_vector)
        dist_slack_normalized = dist_slack_vector / sum(dist_slack_vector)
    else
        dist_slack_normalized = Float64[]
    end

    # Pre-allocate work array for BA column extraction
    valid_ix = setdiff(1:length(temp_data), ref_bus_positions)
    work_ba_col = zeros(length(valid_ix))

    return VirtualPTDF(
        K,
        BA.data,
        dist_slack_vector,
        dist_slack_normalized,
        axes,
        look_up,
        temp_data,
        valid_ix,
        empty_cache,
        subnetwork_axes,
        Ref(tol),
        Ymatrix.network_reduction_data,
        work_ba_col,
    )
end

# Overload Base functions

"""
Checks if the any of the fields of VirtualPTDF is empty.
"""
function Base.isempty(vptdf::VirtualPTDF)
    for name in fieldnames(typeof(vptdf))
        if name == :dist_slack && !isempty(getfield(vptdf, name))
            @debug "Field dist_slack has default value: " *
                   string(getfield(vptdf, name)) * "."
            return false
        elseif (name in [:cache, :radial_branches]) && !isempty(getfield(vptdf, name))
            @debug "Field " * string(name) * " not defined."
            return false
        end
    end
    return true
end

"""
Gives the size of the whole PTDF matrix, not the number of rows stored.
"""
Base.size(vptdf::VirtualPTDF) = size(vptdf.BA)

"""
Gives the cartesian indexes of the PTDF matrix (same as the BA one).
"""
Base.eachindex(vptdf::VirtualPTDF) = CartesianIndices(size(vptdf.BA))

if isdefined(Base, :print_array) # 0.7 and later
    Base.print_array(io::IO, X::VirtualPTDF) = "VirtualPTDF"
end

# Helper function to solve with different factorization types
function _solve_factorization(K::KLU.KLUFactorization{Float64, Int}, b::Vector{Float64})
    return KLU.solve!(K, b)
end

# Generic fallback for other factorization types (will be extended by extensions)
function _solve_factorization(K::LinearAlgebra.Factorization, b::Vector{Float64})
    return K \ b
end

function _getindex(
    vptdf::VirtualPTDF,
    row::Int,
    column::Union{Int, Colon},
)
    # check if value is in the cache
    if haskey(vptdf.cache, row)
        return vptdf.cache.temp_cache[row][column]
    else
        # evaluate the value for the PTDF column
        valid_ix = vptdf.valid_ix
        # Use pre-allocated work array instead of collect() to reduce allocations
        @inbounds for i in eachindex(valid_ix)
            vptdf.work_ba_col[i] = vptdf.BA[valid_ix[i], row]
        end
        lin_solve = _solve_factorization(vptdf.K, vptdf.work_ba_col)
        buscount = size(vptdf, 1)
        ref_bus_positions = get_ref_bus_position(vptdf)
        if !isempty(vptdf.dist_slack) && length(ref_bus_positions) != 1
            error(
                "Distributed slack is not supported for systems with multiple reference buses.",
            )
        elseif isempty(vptdf.dist_slack) && length(ref_bus_positions) < buscount
            @inbounds for i in eachindex(valid_ix)
                vptdf.temp_data[valid_ix[i]] = lin_solve[i]
            end
            vptdf.cache[row] = copy(vptdf.temp_data)
        elseif length(vptdf.dist_slack) == buscount
            @inbounds for i in eachindex(valid_ix)
                vptdf.temp_data[valid_ix[i]] = lin_solve[i]
            end
            # Use pre-computed normalized slack array for efficiency
            adjustment = dot(vptdf.temp_data, vptdf.dist_slack_normalized)
            vptdf.cache[row] = vptdf.temp_data .- adjustment
        else
            error("Distributed bus specification doesn't match the number of buses.")
        end

        if get_tol(vptdf) > eps()
            vptdf.cache[row] = sparsify(vptdf.cache[row], get_tol(vptdf))
        end

        return vptdf.cache[row][column]
    end
end

function Base.getindex(vptdf::VirtualPTDF, branch_name::String, bus)
    multiplier, arc = get_branch_multiplier(vptdf, branch_name)
    row_, column_ = to_index(vptdf, arc, bus)
    return _getindex(vptdf, row_, column_) * multiplier
end

"""
Gets the value of the element of the PTDF matrix given the row and column indices
corresponding to the branch and buses one respectively. If `column` is a Colon then
the entire row is returned.

# Arguments
- `vptdf::VirtualPTDF`:
        VirtualPTDF struct where to evaluate and store the row values.
- `row`:
        Branch index.
- `column`:
        Bus index. If Colon then get the values of the whole row.
"""
function Base.getindex(vptdf::VirtualPTDF, row, column)
    row_, column_ = to_index(vptdf, row, column)
    return _getindex(vptdf, row_, column_)
end

# Define for ambiguity resolution
function Base.getindex(vptdf::VirtualPTDF, row::Integer, column::Integer)
    return _getindex(vptdf, row, column)
end

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualPTDF, _, idx...) = error("Operation not supported by VirtualPTDF")

"""
!!! STILL TO IMPLEMENT !!!
"""
Base.setindex!(::VirtualPTDF, _, ::CartesianIndex) =
    error("Operation not supported by VirtualPTDF")

get_ptdf_data(mat::VirtualPTDF) = mat.cache.temp_cache

function get_arc_axis(ptdf::VirtualPTDF)
    return ptdf.axes[1]
end

function get_bus_axis(ptdf::VirtualPTDF)
    return ptdf.axes[2]
end

""" Gets the tolerance used for sparsifying the rows of the VirtualPTDF matrix"""
function get_tol(vptdf::VirtualPTDF)
    return vptdf.tol[]
end
