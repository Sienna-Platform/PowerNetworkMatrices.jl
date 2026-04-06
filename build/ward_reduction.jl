"""
    WardReduction <: NetworkReduction
Perform Ward reduction to create an equivalent network representation.

Ward reduction is a network reduction technique that eliminates external buses while preserving 
the electrical characteristics seen from the study buses. External buses are mapped to boundary 
buses based on impedance criteria, and equivalent admittances are computed.

# Fields
- `study_buses::Vector{Int}`: List of internal bus numbers that are retained in the equivalent representation.

# Examples
```julia
# Create ward reduction with internal bus numbers 101, 102, and 103
reduction = WardReduction([101, 102, 103])

# Apply to system
ybus = Ybus(system; network_reductions=[reduction])
```
"""
struct WardReduction <: NetworkReduction
    study_buses::Vector{Int}
end
get_study_buses(nr::WardReduction) = nr.study_buses

"""
    get_ward_reduction(data, bus_lookup, bus_axis, boundary_buses, ref_bus_numbers, study_buses)

Perform Ward reduction to create an equivalent network representation.

Ward reduction is a network reduction technique that eliminates external buses while preserving 
the electrical characteristics seen from the study buses. External buses are mapped to boundary 
buses based on impedance criteria, and equivalent admittances are computed.

# Arguments
- `data::SparseArrays.SparseMatrixCSC{ComplexF32, Int}`: Admittance matrix of the system
- `bus_lookup::Dict{Int, Int}`: Dictionary mapping bus numbers to matrix indices
- `bus_axis::Vector{Int}`: Vector of all bus numbers in the system
- `boundary_buses::Set{Int}`: Set of boundary bus numbers between study and external areas
- `ref_bus_numbers::Set{Int}`: Set of reference bus numbers
- `study_buses::Vector{Int}`: Vector of study bus numbers to retain

# Returns
- `Tuple`: Contains bus reduction map, reverse bus search map, added branch map, and added admittance map
"""
function get_ward_reduction(
    data::SparseArrays.SparseMatrixCSC{ComplexF32, Int},
    bus_lookup::Dict{Int, Int},
    bus_axis::Vector{Int},
    boundary_buses::Set{Int},
    ref_bus_numbers::Set{Int},
    study_buses::Vector{Int},
)
    all_buses = bus_axis
    external_buses = setdiff(all_buses, study_buses)
    boundary_buses = unique(boundary_buses)
    n_external = length(external_buses)
    n_boundary = length(boundary_buses)
    n_buses = length(bus_axis)

    bus_reduction_map_index = Dict{Int, Set{Int}}(k => Set{Int}() for k in study_buses)

    added_branch_map = Dict{Tuple{Int, Int}, Complex{Float32}}()
    added_admittance_map = Dict{Int, Complex{Float32}}()
    if isempty(boundary_buses)
        first_ref_study_bus = findfirst(x -> x ∈ ref_bus_numbers, study_buses)
        @error "no boundary buses found; cannot make bus_reduction_map based on impedance based criteria. mapping all external buses to the first reference bus ($first_ref_study_bus)"
        bus_reduction_map_index[first_ref_study_bus] = Set(external_buses)
    else
        # Optimized: Only compute Z rows for external buses instead of full Z matrix
        # This reduces complexity from O(n³) to O(n_external * n²)
        K = klu(data)
        boundary_bus_indices = [bus_lookup[x] for x in boundary_buses]
        boundary_bus_numbers = collect(boundary_buses)
        e = zeros(ComplexF64, n_buses)

        for b in external_buses
            row_index = bus_lookup[b]
            # Compute only the row we need: Z[row_index, :] = e_row^T * Y^(-1)
            # This is equivalent to solving Y^T * z = e_row, but since Y is symmetric, Y * z = e_row
            fill!(e, zero(ComplexF64))
            e[row_index] = one(ComplexF64)
            Z_row = K \ e  # Single row solve instead of full inverse

            Z_row_boundary = abs.(Z_row[boundary_bus_indices])
            closest_boundary_bus = boundary_bus_numbers[argmin(Z_row_boundary)]
            push!(bus_reduction_map_index[closest_boundary_bus], b)
        end
    end
    reverse_bus_search_map =
        _make_reverse_bus_search_map(bus_reduction_map_index, length(all_buses))

    #Populate matrices for computing external equivalent
    y_ee = SparseArrays.spzeros(ComplexF32, n_external, n_external)
    for (ix, i) in enumerate(external_buses)
        for (jx, j) in enumerate(external_buses)
            y_ee[ix, jx] = data[bus_lookup[i], bus_lookup[j]]
        end
    end
    y_be = SparseArrays.spzeros(ComplexF32, n_boundary, n_external)
    for (ix, i) in enumerate(boundary_buses)
        for (jx, j) in enumerate(external_buses)
            y_be[ix, jx] = data[bus_lookup[i], bus_lookup[j]]
        end
    end
    y_eb = SparseArrays.spzeros(ComplexF32, n_external, n_boundary)
    for (ix, i) in enumerate(external_buses)
        for (jx, j) in enumerate(boundary_buses)
            y_eb[ix, jx] = data[bus_lookup[i], bus_lookup[j]]
        end
    end

    # Eq. (2.16) from  https://core.ac.uk/download/pdf/79564835.pdf
    y_eq = y_be * KLU.solve!(klu(y_ee), Matrix{Complex{Float64}}(y_eb))

    #Loop upper diagonal of Yeq
    for ix in 1:length(boundary_buses)
        for jx in ix:length(boundary_buses)
            bus_ix = boundary_buses[ix]
            bus_jx = boundary_buses[jx]
            if y_eq[ix, jx] != 0.0
                if ix == jx
                    added_admittance_map[bus_ix] = y_eq[ix, jx]
                else
                    #check if the arc of virtual line is already existing so we don't add an additional arc
                    if (bus_ix, bus_jx) ∈ bus_axis
                        arc_key = (bus_ix, bus_jx)
                    else
                        arc_key = (bus_jx, bus_ix)
                    end
                    added_branch_map[arc_key] = y_eq[ix, jx]
                end
            end
        end
    end
    return bus_reduction_map_index,
    reverse_bus_search_map,
    added_branch_map,
    added_admittance_map
end
