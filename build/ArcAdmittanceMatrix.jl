"""
Arc admittance matrix

# Arguments
- `data::SparseArrays.SparseMatrixCSC{ComplexF32, Int}`:
        The arc admittance matrix in the from-to direction
- `axes<:NTuple{2, Dict}`:
        Tuple containing two vectors (the first one showing the arc tuples,
        the second showing the buses numbers).
- `lookup<:NTuple{2, Dict}`:
        Tuple containing two dictionaries, the first mapping the arc tuples
        and the second the buses with their enumerated indexes.
- `network_reduction::NetworkReduction`:
        Structure containing the details of the network reduction applied when computing the matrix
- `direction::Symbol`:
        Direction of admittance (:FromTo or :ToFrom)
"""
struct ArcAdmittanceMatrix{Ax, L <: NTuple{2, Dict}} <: PowerNetworkMatrix{ComplexF32}
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int}
    axes::Ax
    lookup::L
    network_reduction_data::NetworkReductionData
    direction::Symbol
end

# functions to get stored data
get_axes(M::ArcAdmittanceMatrix) = M.axes
get_lookup(M::ArcAdmittanceMatrix) = M.lookup
get_ref_bus(M::ArcAdmittanceMatrix) = sort!(collect(keys(M.subnetwork_axes)))
get_ref_bus_position(M::ArcAdmittanceMatrix) =
    [get_bus_lookup(M)[x] for x in keys(M.subnetwork_axes)]
get_network_reduction_data(M::ArcAdmittanceMatrix) = M.network_reduction_data
get_arc_axis(M::ArcAdmittanceMatrix) = M.axes[1]
get_arc_lookup(M::ArcAdmittanceMatrix) = M.lookup[1]
get_bus_axis(M::ArcAdmittanceMatrix) = M.axes[2]
get_bus_lookup(M::ArcAdmittanceMatrix) = M.lookup[2]
get_direction(M::ArcAdmittanceMatrix) = M.direction
