"""
A [`NetworkReduction`](@ref) that eliminates external buses while preserving the
electrical behavior within the study area, using Ward equivalencing.

# Fields
- `study_buses::Vector{Int}`: bus numbers to retain in the reduced network
"""
struct WardReduction <: NetworkReduction
    study_buses::Vector{Int}
end
get_study_buses(nr::WardReduction) = nr.study_buses

"""
    get_ward_reduction(data, bus_lookup, bus_axis, arc_axis, boundary_buses, ref_bus_numbers, study_buses, subnetwork_bus_axis)

Perform Ward reduction to create an equivalent network representation.

Ward reduction is a network reduction technique that eliminates external buses while preserving 
the electrical characteristics seen from the study buses. External buses are mapped to boundary 
buses based on impedance criteria, and equivalent admittances are computed.

# Arguments
- `data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int}`: Admittance matrix of the system
- `bus_lookup::Dict{Int, Int}`: Dictionary mapping bus numbers to matrix indices
- `bus_axis::Vector{Int}`: Vector of all bus numbers in the system
- `arc_axis::Vector{Tuple{Int, Int}}`: Vector of all arc tuples in the system
- `boundary_buses::Set{Int}`: Set of boundary bus numbers between study and external areas
- `ref_bus_numbers::Set{Int}`: Set of reference bus numbers
- `study_buses::Vector{Int}`: Vector of study bus numbers to retain
- `subnetwork_bus_axis::Vector{Int}`: Bus numbers in the selected subnetwork to reduce

# Returns
- `Tuple`: Contains bus reduction map, reverse bus search map, added branch map, and added admittance map
"""
function get_ward_reduction(
    data::SparseArrays.SparseMatrixCSC{YBUS_ELTYPE, Int},
    bus_lookup::Dict{Int, Int},
    bus_axis::Vector{Int},
    arc_axis::Vector{Tuple{Int, Int}},
    boundary_buses::Set{Int},
    ref_bus_numbers::Set{Int},
    study_buses::Vector{Int},
    subnetwork_bus_axis::Vector{Int},
)
    # Restrict Ward operations to the study subnetwork.
    all_buses = subnetwork_bus_axis
    subnetwork_bus_indices = [bus_lookup[x] for x in all_buses]
    subnetwork_bus_lookup = Dict(bus => ix for (ix, bus) in enumerate(all_buses))
    subnetwork_data = data[subnetwork_bus_indices, subnetwork_bus_indices]
    println(typeof(data))
    println(typeof(subnetwork_data))
    boundary_buses = collect(intersect(boundary_buses, Set(all_buses)))

    external_buses = setdiff(all_buses, study_buses)
    n_buses = length(all_buses)

    bus_reduction_map_index = Dict{Int, Set{Int}}(k => Set{Int}() for k in study_buses)

    added_arc_impedance_map = Dict{Tuple{Int, Int}, PSY.GenericArcImpedance}()
    added_admittance_map = Dict{Int, PSY.FixedAdmittance}()
    if isempty(boundary_buses)
        first_ref_study_bus = findfirst(x -> x ∈ ref_bus_numbers, study_buses)
        @error "no boundary buses found; cannot make bus_reduction_map based on impedance based criteria. mapping all external buses to the first reference bus ($first_ref_study_bus)"
        bus_reduction_map_index[first_ref_study_bus] = Set(external_buses)
    else
        # Optimized: Only compute Z rows for external buses instead of full Z matrix
        # This reduces complexity from O(n³) to O(n_external * n²)
        K = klu(subnetwork_data)
        boundary_bus_indices = [subnetwork_bus_lookup[x] for x in boundary_buses]
        boundary_bus_numbers = collect(boundary_buses)
        e = zeros(ComplexF64, n_buses)

        for b in external_buses
            row_index = subnetwork_bus_lookup[b]
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
    external_bus_indices = [subnetwork_bus_lookup[x] for x in external_buses]
    boundary_bus_indices = [subnetwork_bus_lookup[x] for x in boundary_buses]
    y_ee = subnetwork_data[external_bus_indices, external_bus_indices]
    y_be = subnetwork_data[
        boundary_bus_indices,
        external_bus_indices,
    ]
    y_eb = subnetwork_data[
        external_bus_indices,
        boundary_bus_indices,
    ]


    # Eq. (2.16) from  https://core.ac.uk/download/pdf/79564835.pdf
    y_eq =
        y_be * KLU.solve!(klu(y_ee), Matrix{Complex{Float64}}(y_eb))
    #Loop upper diagonal of Yeq
    for ix in 1:length(boundary_buses)
        for jx in ix:length(boundary_buses)
            bus_ix = boundary_buses[ix]
            bus_jx = boundary_buses[jx]
            if y_eq[ix, jx] != 0.0
                if ix != jx
                    arc_impedance = 1.0 / y_eq[ix, jx]
                    generic_arc_impedance = PSY.GenericArcImpedance(;
                        name = "",
                        available = true,
                        active_power_flow = 0.0,
                        reactive_power_flow = 0.0,
                        max_flow = 1e6,
                        arc = PSY.Arc(nothing),
                        r = real(arc_impedance),
                        x = imag(arc_impedance),
                    )
                    Y11, Y12, _, Y22 = ybus_branch_entries(generic_arc_impedance)
                    @assert isapprox(-1.0 * Y12, y_eq[ix, jx])
                    #check if the arc of virtual line is already existing so we don't add an additional arc
                    if (bus_ix, bus_jx) ∈ arc_axis
                        arc_key = (bus_ix, bus_jx)
                        y_eq[ix, ix] -= Y11
                        y_eq[jx, jx] -= Y22
                    else
                        arc_key = (bus_jx, bus_ix)
                        y_eq[ix, ix] -= Y22
                        y_eq[jx, jx] -= Y11
                    end
                    added_arc_impedance_map[arc_key] = generic_arc_impedance
                end
            end
        end
    end
    for ix in 1:length(boundary_buses)
        bus_ix = boundary_buses[ix]
        if y_eq[ix, ix] != 0.0
            added_admittance_map[bus_ix] = PSY.FixedAdmittance(;
                name = "",
                available = true,
                bus = PSY.ACBus(nothing),
                Y = y_eq[ix, ix],
            )
        end
    end
    return bus_reduction_map_index,
    reverse_bus_search_map,
    added_arc_impedance_map,
    added_admittance_map
end
