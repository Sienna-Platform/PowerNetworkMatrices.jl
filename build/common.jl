function _add_to_collection!(
    collection_br::Vector{PSY.ACTransmission},
    branch::PSY.ACTransmission,
)
    push!(collection_br, branch)
    return
end

function _add_to_collection!(
    collection_tr3w::Vector{PSY.ThreeWindingTransformer},
    transformer_tr3w::PSY.ThreeWindingTransformer,
)
    push!(collection_tr3w, transformer_tr3w)
    return
end

function get_bus_index(bus_no::Int, bus_lookup::Dict{Int, Int}, nr::NetworkReductionData)
    if haskey(nr.reverse_bus_search_map, bus_no)
        return bus_lookup[nr.reverse_bus_search_map[bus_no]]
    else
        return bus_lookup[bus_no]
    end
end

function get_bus_index(
    dev::PSY.Component,
    bus_lookup::Dict{Int, Int},
    nr::NetworkReductionData,
)
    bus_number = PSY.get_number(PSY.get_bus(dev))
    return get_bus_index(bus_number, bus_lookup, nr)
end

function get_bus_indices(
    arc::PSY.Arc,
    bus_lookup::Dict{Int, Int},
    nr::NetworkReductionData,
)
    check_arc_validity(arc, IS.get_name(arc))
    reverse_bus_search_map = get_reverse_bus_search_map(nr)
    fr_bus_number = PSY.get_number(PSY.get_from(arc))
    if haskey(reverse_bus_search_map, fr_bus_number)
        fr_bus_number_reduced = reverse_bus_search_map[fr_bus_number]
    else
        fr_bus_number_reduced = fr_bus_number
    end
    fr_bus_ix = bus_lookup[fr_bus_number_reduced]

    to_bus_number = PSY.get_number(PSY.get_to(arc))
    if haskey(reverse_bus_search_map, to_bus_number)
        to_bus_number_reduced = reverse_bus_search_map[to_bus_number]
    else
        to_bus_number_reduced = to_bus_number
    end
    to_bus_ix = bus_lookup[to_bus_number_reduced]
    return fr_bus_ix, to_bus_ix
end

function check_arc_validity(arc::PSY.Arc, name::String)
    if PSY.get_bustype(PSY.get_from(arc)) == ACBusTypes.ISOLATED
        throw(
            IS.ConflictingInputsError(
                "Branch or arc $(name) is set available and connected to isolated bus " *
                "$(IS.get_name(PSY.get_from(arc)))",
            ),
        )
    end
    if PSY.get_bustype(PSY.get_to(arc)) == ACBusTypes.ISOLATED
        throw(
            IS.ConflictingInputsError(
                "Branch or arc $(name) is set available and connected to isolated bus " *
                "$(IS.get_name(PSY.get_to(arc)))",
            ),
        )
    end
    return
end

function get_arc_tuple(arc::PSY.Arc, nr::NetworkReductionData)
    reverse_bus_search_map = get_reverse_bus_search_map(nr)
    arc_tuple_original = get_arc_tuple(arc)
    return (
        get(reverse_bus_search_map, arc_tuple_original[1], arc_tuple_original[1]),
        get(reverse_bus_search_map, arc_tuple_original[2], arc_tuple_original[2]),
    )
end

function get_arc_tuple(
    tr::ThreeWindingTransformerWinding,
    nr::NetworkReductionData,
)
    reverse_bus_search_map = get_reverse_bus_search_map(nr)
    arc_tuple_original = get_arc_tuple(tr)
    return (
        get(reverse_bus_search_map, arc_tuple_original[1], arc_tuple_original[1]),
        get(reverse_bus_search_map, arc_tuple_original[2], arc_tuple_original[2]),
    )
end

function get_arc_tuple(br::PSY.ACTransmission, nr::NetworkReductionData)
    get_arc_tuple(PSY.get_arc(br), nr)
end

# Parallel branches: all oriented in same direction, so just take arc of first.
function get_arc_tuple(br::BranchesParallel, nr::NetworkReductionData)
    get_arc_tuple(PSY.get_arc(first(br)), nr)
end

function get_arc_tuple(br::BranchesParallel)
    return get_arc_tuple(PSY.get_arc(first(br)))
end

function get_arc_tuple(br::PSY.ACTransmission)
    return get_arc_tuple(PSY.get_arc(br))
end

get_arc_tuple(arc::PSY.Arc) =
    (PSY.get_number(PSY.get_from(arc)), PSY.get_number(PSY.get_to(arc)))

function get_switched_admittances(sys::PSY.System, reverse_bus_search_map)
    collection = Vector{PSY.SwitchedAdmittance}()
    for sa in
        collect(PSY.get_components(x -> PSY.get_available(x), PSY.SwitchedAdmittance, sys))
        if !haskey(reverse_bus_search_map, PSY.get_number(PSY.get_bus(sa)))
            push!(collection, sa)
        end
    end
    return collection
end

function get_fixed_admittances(sys::PSY.System, reverse_bus_search_map)
    collection = Vector{PSY.FixedAdmittance}()
    for sa in
        collect(PSY.get_components(x -> PSY.get_available(x), PSY.FixedAdmittance, sys))
        if !haskey(reverse_bus_search_map, PSY.get_number(PSY.get_bus(sa)))
            push!(collection, sa)
        end
    end
    return collection
end

function _add_branch_to_lookup!(
    branch_lookup::Dict{String, Int},
    ::Dict{String, Vector{String}},
    branch_type::Vector{DataType},
    branch::PSY.ACTransmission,
    branch_number::Int,
)
    branch_lookup[PSY.get_name(branch)] = branch_number
    push!(branch_type, typeof(branch))
    return
end

function _add_branch_to_lookup!(
    branch_lookup::Dict{String, Int},
    transformer_3w_lookup::Dict{String, Vector{String}},
    branch_type::Vector{DataType},
    branch::PSY.ThreeWindingTransformer,
    branch_number::Int,
)
    tr3w_name = PSY.get_name(branch)
    transformer_3w_lookup[tr3w_name] = Vector{String}(undef, 3)
    for (i, side) in enumerate(["primary", "secondary", "tertiary"])
        side_name = "$(tr3w_name)__$side"
        branch_lookup[side_name] = branch_number - 3 + i
        transformer_3w_lookup[tr3w_name][i] = side_name
        push!(branch_type, typeof(branch))
    end
    return
end

"""
Gets the indices  of the reference (slack) buses.
NOTE:
- the indices  corresponds to the columns of zeros belonging to the PTDF matrix.
- BA and ABA matrix miss the columns related to the reference buses.
"""
function find_slack_positions(buses)
    return find_slack_positions(buses, make_ax_ref(buses))
end

function find_slack_positions(buses, bus_lookup::Dict{Int, Int})::Set{Int}
    slack_position = sort([
        bus_lookup[PSY.get_number(n)]
        for n in buses if PSY.get_bustype(n) == ACBusTypes.REF
    ])
    if length(slack_position) == 0
        error("Slack bus not identified in the Bus/buses list, can't build NetworkMatrix")
    end
    return Set{Int}(slack_position)
end

"""
Validates if the selected linear solver is supported.
"""
function validate_linear_solver(linear_solver::String)
    if linear_solver ∉ SUPPORTED_LINEAR_SOLVERS
        error(
            "Invalid linear solver. Supported linear solvers are: $(SUPPORTED_LINEAR_SOLVERS)",
        )
    end
    return
end

"""
Validates that the user bus input is consistent with the ybus axes and the prior reductions.
Is used to check `irreducible_buses` for `Radial` and `DegreeTwo` reductions and `study_buses` for `WardReduction`.
"""
function validate_buses(A::AdjacencyMatrix, buses::Vector{Int})
    reverse_bus_search_map = A.network_reduction_data.reverse_bus_search_map
    for bus_no in buses
        reduced_bus_no = get(reverse_bus_search_map, bus_no, bus_no)
        if reduced_bus_no ∉ get_bus_axis(A)
            if bus_no == reduced_bus_no
                error(
                    "Invalid bus entry found: Bus $bus_no. Check your input data; this bus was not found in the admittance matrix.",
                )
            else
                error(
                    "Invalid bus entry found: Bus $bus_no. Check your input data; this bus was mapped to bus $reduced_bus_no in a prior reductions and not found in the admittance matrix.",
                )
            end
        end
    end
    return
end

"""
Convert the user input for irreducible_buses to a set of indices based on the Ybus lookup and the prior reductions.
"""
function get_irreducible_indices(A::AdjacencyMatrix, irreducible_buses::Vector{Int})
    reverse_bus_search_map = A.network_reduction_data.reverse_bus_search_map
    irreducible_indices = zeros(Int, length(irreducible_buses))
    for (ix, bus_no) in enumerate(irreducible_buses)
        reduced_bus_no = get(reverse_bus_search_map, bus_no, bus_no)
        irreducible_indices[ix] = A.lookup[1][reduced_bus_no]
    end
    return irreducible_indices
end

"""
Evaluates the ABA matrix given the System's Incidence matrix (A), BA matrix and
reference bus positions.

# Arguments
- `A::SparseArrays.SparseMatrixCSC{Int8, Int}`:
        Incidence matrix.
- `BA::SparseArrays.SparseMatrixCSC{Float64, Int}`
        BA matrix.

NOTE:
- evaluates A with "calculate_A_matrix", or extract A.data (if A::IncidenceMatrix)
- evaluates BA with "calculate_BA_matrix", or extract BA.data (if A::BA_Matrix)
"""
function calculate_ABA_matrix(
    A::SparseArrays.SparseMatrixCSC{Int8, Int},
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    ref_bus_positions::Set{Int})
    tmp = BA * A
    valid_ix = setdiff(1:size(tmp, 1), ref_bus_positions)
    return tmp[valid_ix, valid_ix]
end

"""
Return a sparse matrix given a dense one by dropping elements whose absolute
value is below a certain tolerance.

Uses optimized `droptol!` for better performance compared to element-wise iteration.

# Arguments
- `dense_array::Matrix{Float64}`:
        input matrix (e.g., PTDF matrix).
- `tol::Float64`:
        tolerance.
"""
function sparsify(dense_array::Matrix{Float64}, tol::Float64)
    sparse_array = SparseArrays.sparse(dense_array)
    SparseArrays.droptol!(sparse_array, tol)
    return sparse_array
end

"""
Return a sparse vector given a dense one by dropping elements whose absolute
value is below a certain tolerance.

Uses optimized `droptol!` for better performance compared to element-wise iteration.

# Arguments
- `dense_array::Vector{Float64}`:
        input vector (e.g., PTDF row from VirtualPTDF).
- `tol::Float64`:
        tolerance.
"""
function sparsify(dense_array::Vector{Float64}, tol::Float64)
    sparse_array = SparseArrays.sparsevec(dense_array)
    SparseArrays.droptol!(sparse_array, tol)
    return sparse_array
end

"""
    get_equivalent_physical_branch_parameters(equivalent_ybus::Matrix{ComplexF32})

Takes as input a 2x2 Matrix{ComplexF32} representing the Ybus contribution of either a
BranchesParallel or BranchesSeries object.
Returns a dictionary of equivalent parameters, matching the PowerModels data format.
"""
function _get_equivalent_physical_branch_parameters(equivalent_ybus::Matrix{ComplexF32})
    y_11, y_12, y_21, y_22 = equivalent_ybus
    if isapprox(y_12, y_21)
        tap = 1.0
        shift = 0.0
    else
        tap = 1.0
        ratio = log(y_21 / y_12) / 2
        if !isapprox(0.0, real(ratio); atol = 1e-6)
            error(
                "Equivalent parameters for the series or parallel reduction of branches results \
          in a real part of the phase shift angle. This indicates invalid data for the branches being reduced \
          possible due to branches in parallel with different phase angles.",
            )
        end
        shift = imag(ratio)
    end
    y_l = y_12 * -1 * exp(1 * shift * im)
    z_12 = 1 / y_l
    r = real(z_12)
    x = imag(z_12)
    g_from = real(y_11 - y_l)
    b_from = imag(y_11 - y_l)
    g_to = real(y_22 - y_l)
    b_to = imag(y_22 - y_l)
    return EquivalentBranch(r, x, g_from, b_from, g_to, b_to, tap, shift)
end
