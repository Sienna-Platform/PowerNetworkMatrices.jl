"""
Structure containing the BA matrix and related network topology data.

The BA matrix represents the branch-bus incidence matrix weighted by branch susceptances,
computed as the product of the incidence matrix A and the susceptance matrix B.

# Fields
- `data::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        The transposed BA matrix data. Each row corresponds to a bus and each column
        corresponds to a branch, with values representing weighted branch susceptances
- `axes::Ax`:
        Tuple containing two vectors: bus numbers (rows) and branch identifiers (columns)
- `lookup::L <: NTuple{2, Dict}`:
        Tuple of dictionaries providing fast lookup from bus/branch names to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`:
        Mapping from reference bus numbers to their corresponding subnetwork axes
- `network_reduction_data::NetworkReductionData`:
        Container for network reduction information applied during matrix construction

# Notes
- The matrix is stored in transposed form for computational efficiency
- Reference buses are identified through `subnetwork_axes` keys
- Supports various network reduction techniques for computational efficiency
"""
struct BA_Matrix{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{Float64}
    data::SparseArrays.SparseMatrixCSC{Float64, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    network_reduction_data::NetworkReductionData
end

get_axes(M::BA_Matrix) = M.axes
get_lookup(M::BA_Matrix) = M.lookup
get_ref_bus(M::BA_Matrix) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::BA_Matrix) = [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::BA_Matrix) = M.network_reduction_data
get_bus_axis(M::BA_Matrix) = M.axes[1]
get_bus_lookup(M::BA_Matrix) = M.lookup[1]
get_arc_axis(M::BA_Matrix) = M.axes[2]
get_arc_lookup(M::BA_Matrix) = M.lookup[2]
stores_transpose(::BA_Matrix) = true

"""
    BA_Matrix(sys::PSY.System; network_reductions::Vector{NetworkReduction} = Vector{NetworkReduction}(), kwargs...)

Construct a BA_Matrix from a PowerSystems.System by first building the underlying Ybus matrix
and then computing the branch-bus incidence matrix weighted by branch susceptances.

# Arguments
- `sys::PSY.System`: The power system from which to construct the BA matrix

# Keyword Arguments
- `network_reductions::Vector{NetworkReduction} = Vector{NetworkReduction}()`:
        Vector of network reduction algorithms to apply before matrix construction
- `include_constant_impedance_loads::Bool=true`:
        Whether to include constant impedance loads as shunt admittances in the network model
- `subnetwork_algorithm=iterative_union_find`:
        Algorithm used for identifying electrical islands and connected components
- Additional keyword arguments are passed to the underlying `Ybus` constructor

# Returns
- `BA_Matrix`: The constructed BA matrix structure containing the transposed branch-bus incidence
              matrix weighted by susceptances, along with network topology information

# Notes
- This constructor creates a `Ybus` matrix internally and then converts it to a `BA_Matrix`
- Network reductions can significantly improve computational efficiency for large systems
- The resulting matrix supports DC power flow calculations and sensitivity analysis
"""
function BA_Matrix(sys::PSY.System;
    network_reductions::Vector{NetworkReduction} = Vector{NetworkReduction}(),
    kwargs...,
)
    return BA_Matrix(
        Ybus(
            sys;
            network_reductions = network_reductions,
            kwargs...,
        ),
    )
end

"""
    BA_Matrix(ybus::Ybus)

Construct a BA_Matrix from a Ybus matrix.

# Arguments
- `ybus::Ybus`: The Ybus matrix from which to construct the BA matrix

# Returns
- `BA_Matrix`: The constructed BA matrix structure containing the transposed BA matrix
"""
function BA_Matrix(ybus::Ybus)
    nr = ybus.network_reduction_data
    bus_ax = get_bus_axis(ybus)
    bus_lookup = get_bus_lookup(ybus)
    arc_ax = get_arc_axis(nr)
    n_isolated_buses = length(get_isolated_buses(ybus))
    n_entries = length(arc_ax) * 2 + n_isolated_buses
    BA_I = Vector{Int}(undef, n_entries)
    BA_J = Vector{Int}(undef, n_entries)
    BA_V = Vector{Float64}(undef, n_entries)
    nr_data = get_network_reduction_data(ybus)
    for (ix_arc, arc) in enumerate(arc_ax)
        ix_from_bus = get_bus_index(arc[1], bus_lookup, nr)
        ix_to_bus = get_bus_index(arc[2], bus_lookup, nr)
        # Get series susceptance from components, not the equivalent ybus for reductions of degree two nodes
        # This results in reduced error relative to the DC power flow result without reductions
        if is_arc_in_series_map(nr_data, arc)
            b = get_series_susceptance(get_mapped_series_branch(nr_data, arc))
        else
            Yt = -1 * ybus.data[ix_from_bus, ix_to_bus]
            Zt = 1 / Yt
            # TODO - should we consider phase shift?
            b = 1 / imag(Zt)
        end
        BA_I[2 * ix_arc - 1] = ix_from_bus
        BA_J[2 * ix_arc - 1] = ix_arc
        BA_V[2 * ix_arc - 1] = b
        BA_I[2 * ix_arc] = ix_to_bus
        BA_J[2 * ix_arc] = ix_arc
        BA_V[2 * ix_arc] = -1 * b
    end
    BA_I[(end - n_isolated_buses + 1):end] =
        [get_bus_index(x, bus_lookup, nr) for x in get_isolated_buses(ybus)]
    BA_J[(end - n_isolated_buses + 1):end] = ones(n_isolated_buses)
    BA_V[(end - n_isolated_buses + 1):end] = zeros(n_isolated_buses)
    data = SparseArrays.sparse(BA_I, BA_J, BA_V)
    SparseArrays.dropzeros!(data)
    axes = (bus_ax, arc_ax)
    lookup = (make_ax_ref(bus_ax), make_ax_ref(arc_ax))
    subnetwork_axes = make_bus_arc_subnetwork_axes(ybus)
    return BA_Matrix(data, axes, lookup, subnetwork_axes, ybus.network_reduction_data)
end

function get_series_susceptance(segment::PSY.ACTransmission)
    return PSY.get_series_susceptance(segment)
end

"""
Structure containing the ABA matrix and related power system analysis data.

The ABA matrix represents the bus susceptance matrix computed as A^T * B * A, where A is the
incidence matrix and B is the branch susceptance matrix. This matrix is fundamental for DC
power flow analysis, sensitivity calculations, and linear power system studies.

# Fields
- `data::SparseArrays.SparseMatrixCSC{Float64, Int}`:
        The ABA matrix data representing the bus susceptance matrix. This square matrix has
        dimensions equal to the number of buses excluding reference buses
- `axes::Ax`:
        Tuple containing identical bus number vectors for rows and columns, excluding reference buses
- `lookup::L <: NTuple{2, Dict}`:
        Tuple of identical dictionaries providing fast lookup from bus numbers to matrix indices
- `subnetwork_axes::Dict{Int, Ax}`:
        Mapping from reference bus numbers to their corresponding subnetwork axes
- `ref_bus_position::Vector{Int}`:
        Vector containing the original indices of reference buses before matrix reduction
- `K::F <: Union{Nothing, KLU.KLUFactorization{Float64, Int}}`:
        Optional KLU factorization object for efficient linear system solving. Nothing if unfactorized
- `network_reduction_data::NetworkReductionData`:
        Container for network reduction information applied during matrix construction

# Mathematical Properties
- **Matrix Form**: ABA = A^T * B * A (bus susceptance matrix)
- **Dimensions**: (n_buses - n_ref) × (n_buses - n_ref)
- **Symmetry**: Positive definite symmetric matrix (for connected networks)
- **Sparsity**: Inherits sparsity pattern from network topology

# Notes
- Reference buses are excluded from the matrix to ensure invertibility
- Factorization enables efficient solving of linear systems Ax = b
- Used primarily for DC power flow analysis and power system sensitivity studies
- Supports various network reduction techniques for computational efficiency
"""
struct ABA_Matrix{
    Ax,
    L <: NTuple{2, Dict},
    F <: Union{Nothing, KLU.KLUFactorization{Float64, Int}},
} <: PowerNetworkMatrix{Float64}
    data::SparseArrays.SparseMatrixCSC{Float64, Int}
    axes::Ax
    lookup::L
    subnetwork_axes::Dict{Int, Ax}
    ref_bus_position::Vector{Int}
    K::F
    network_reduction_data::NetworkReductionData
end

get_axes(M::ABA_Matrix) = M.axes
get_lookup(M::ABA_Matrix) = M.lookup
get_ref_bus(M::ABA_Matrix) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::ABA_Matrix) = M.ref_bus_position
get_network_reduction_data(M::ABA_Matrix) = M.network_reduction_data
get_bus_axis(M::ABA_Matrix) = M.axes[1]
get_bus_lookup(M::ABA_Matrix) = M.lookup[1]

"""
    ABA_Matrix(sys::PSY.System; factorize::Bool = false, network_reductions::Vector{NetworkReduction} = NetworkReduction[], kwargs...)

Construct an ABA_Matrix from a PowerSystems.System by computing A^T * B * A where A is the
incidence matrix and B is the branch susceptance matrix. The resulting matrix is fundamental
for DC power flow analysis and power system sensitivity studies.

# Arguments
- `sys::PSY.System`: The power system from which to construct the ABA matrix

# Keyword Arguments
- `factorize::Bool = false`:
        Whether to perform KLU factorization during construction for efficient linear system solving
- `network_reductions::Vector{NetworkReduction} = NetworkReduction[]`:
        Vector of network reduction algorithms to apply before matrix construction
- `include_constant_impedance_loads::Bool=true`: 
        Whether to include constant impedance loads as shunt admittances in the network model
- `subnetwork_algorithm=iterative_union_find`:
        Algorithm used for identifying electrical islands and connected components
- Additional keyword arguments are passed to the underlying `Ybus` constructor

# Returns
- `ABA_Matrix`: The constructed ABA matrix structure containing:
  - Bus susceptance matrix data (excluding reference buses)
  - Network topology information and reference bus positions
  - Optional KLU factorization for efficient solving

# Mathematical Process
1. **Ybus Construction**: Creates admittance matrix from system data
2. **Incidence Matrix**: Computes bus-branch incidence matrix A
3. **BA Matrix**: Forms branch susceptance weighted incidence matrix
4. **ABA Computation**: Calculates A^T * B * A (bus susceptance matrix)
5. **Reference Bus Removal**: Excludes reference buses for invertibility
6. **Optional Factorization**: Performs KLU decomposition if requested

# Notes
- Reference buses are automatically detected and excluded from the final matrix
- Factorization significantly improves performance for repeated linear system solves
- Network reductions can dramatically improve computational efficiency for large systems
- The resulting matrix supports PTDF, LODF, and other power system analysis calculations
"""
function ABA_Matrix(sys::PSY.System;
    factorize::Bool = false,
    network_reductions::Vector{NetworkReduction} = NetworkReduction[],
    kwargs...,
)
    ymatrix = Ybus(
        sys;
        network_reductions = network_reductions,
        kwargs...,
    )
    ref_bus_positions = get_ref_bus_position(ymatrix)
    A = IncidenceMatrix(ymatrix)
    BA = BA_Matrix(ymatrix)
    ABA = calculate_ABA_matrix(A.data, BA.data, Set(ref_bus_positions))
    axes, subnetwork_axes = _remake_axes_without_ref(ymatrix.axes, ymatrix.subnetwork_axes)
    bus_ax_ref = make_ax_ref(axes[1])
    lookup = (bus_ax_ref, bus_ax_ref)
    if factorize
        K = klu(ABA)
    else
        K = nothing
    end
    return ABA_Matrix(
        ABA,
        axes,
        lookup,
        subnetwork_axes,
        ref_bus_positions,
        K,
        ymatrix.network_reduction_data,
    )
end

function _remake_axes_without_ref(
    axes::Tuple{Vector{Int}, Vector{Int}},
    subnetwork_axes::Dict{Int, Tuple{Vector{Int}, Vector{Int}}},
)
    ref_bus_numbers = collect(keys(subnetwork_axes))
    bus_ax_ = setdiff(axes[1], Set(ref_bus_numbers))
    axes_ref = (bus_ax_, bus_ax_)
    subnetwork_axes_ref = deepcopy(subnetwork_axes)
    for (k, v) in subnetwork_axes_ref
        setdiff!(v[1], [k])
        setdiff!(v[2], [k])
    end
    return axes_ref, subnetwork_axes_ref
end

"""
Evaluates the LU factorization matrices of the ABA matrix, using KLU.

# Arguments
- `ABA::ABA_Matrix{Ax, L, Nothing} where {Ax, L <: NTuple{2, Dict}}`:
        container for the ABA matrix, with ABA.K == nothing (LU matrices in K not evaluated)
"""
function factorize(ABA::ABA_Matrix{Ax, L, Nothing}) where {Ax, L <: NTuple{2, Dict}}
    ABA_lu = ABA_Matrix(
        deepcopy(ABA.data),
        deepcopy(ABA.axes),
        deepcopy(ABA.lookup),
        deepcopy(ABA.subnetwork_axes),
        deepcopy(ABA.ref_bus_position),
        klu(ABA.data),
        deepcopy(ABA.network_reduction_data),
    )
    return ABA_lu
end

"""
    is_factorized(ABA::ABA_Matrix)

Check if an ABA_Matrix has been factorized (i.e., contains LU factorization matrices).

# Arguments
- `ABA::ABA_Matrix`: The ABA matrix to check

# Returns
- `Bool`: true if the matrix has been factorized, false otherwise
"""
is_factorized(ABA::ABA_Matrix{Ax, L, Nothing}) where {Ax, L <: NTuple{2, Dict}} = false
is_factorized(
    ABA::ABA_Matrix{Ax, L, KLU.KLUFactorization{Float64, Int}},
) where {Ax, L <: NTuple{2, Dict}} = true

# get_index functions: BA_Matrix stores the transposed matrix, thus get index
# must export values according to [branch, bus] indexing.
function Base.getindex(A::BA_Matrix, line, bus)
    i, j = to_index(A, bus, line)
    return A.data[i, j]
end

function Base.getindex(
    A::BA_Matrix,
    line_number::Union{Int, Colon},
    bus_number::Union{Int, Colon},
)
    return A.data[bus_number, line_number]
end

_is_ref_bus(bus::Int, ref_bus_numbers::Set{Int}) = (bus in ref_bus_numbers)
_is_ref_bus(bus::PSY.ACBus, ref_bus_numbers::Set{Int}) =
    (PSY.get_number(bus) in ref_bus_numbers)
_is_ref_bus(::Colon, ::Set{Int}) = false
# get_index functions: ABA_Matrix stores a square matrix whose number of rows
# and column is equal to the number of the system's buses minus the slack ones,
# NOTE: bus_1, bus_2 are bus numbers (or PSY.ACBus objects) not row and column indices!
function Base.getindex(A::ABA_Matrix, bus_1, bus_2)
    ref_bus_numbers = Set(get_ref_bus(A))
    if _is_ref_bus(bus_1, ref_bus_numbers) || _is_ref_bus(bus_2, ref_bus_numbers)
        err_msg = string(
            " Rows and columns related to slack buses are not defined for the ABA matrix. \n",
            "Indices must be referred to any bus number that is not a slack one. \n",
            "For the current Systems the reference slack buses are: ",
            collect(ref_bus_numbers), ". \n")
        error(err_msg)
    else
        i, j = to_index(A, bus_1, bus_2)
        return A.data[i, j]
    end
    return
end
