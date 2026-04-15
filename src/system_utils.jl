"""
    validate_connectivity(sys::PSY.System) -> Bool

Check whether the power system network is fully connected using Depth First Search (DFS).

# Arguments
- `sys::PSY.System`: The power system to validate

# Returns
- `Bool`: `true` if the network is fully connected, `false` otherwise
"""
function validate_connectivity(sys::PSY.System)
    sbn, _ = _find_subnetworks(sys)
    return length(sbn) == 1
end

function _find_subnetworks(sys::PSY.System)
    Ymatrix = Ybus(sys)
    @info "Validating connectivity with depth first search (network traversal)"
    subnetworks = find_subnetworks(Ymatrix)
    return subnetworks, get_ref_bus(Ymatrix)
end

"""
Finds the subnetworks in a system using Depth First Search (DFS). Returns a dictionary keyed
by the reference bus of the subnetworks if they exist
"""
function find_subnetworks(sys::PSY.System)
    sbn, ref_buses = _find_subnetworks(sys)
    return assign_reference_buses!(sbn, Set(ref_buses))
end
