"""
Takes the reference bus numbers and re-assigns the keys in the subnetwork dictionaries to use
the reference bus within each subnetwork.

A network island may hold more than one swing (reference) bus; each is an independent
fixed-complex-voltage anchor. When it does, the island is keyed by its smallest-angle swing
(`ref_angles`, radians as stored), with the smallest bus number as the tie-break for equal
angles. This only selects the representative used as the subnetwork-dict key / matrix
reference-bus role; the other swings remain full reference buses for the power flow. If
`ref_angles` is empty (angles unavailable), the representative falls back to the smallest bus
number.
"""
function assign_reference_buses!(
    subnetworks::Dict{Int, Set{Int}},
    ref_buses::Set{Int},
    ref_angles::Dict{Int, Float64} = Dict{Int, Float64}(),
)
    if isempty(ref_buses)
        @warn "No reference buses found. References buses will be assigned arbitrarily"
        return deepcopy(subnetworks)
    end
    bus_groups = Dict{Int, Set{Int}}()
    for (bus_key, subnetwork_buses) in subnetworks
        ref_bus = intersect(ref_buses, subnetwork_buses)
        if length(ref_bus) == 1
            bus_groups[first(ref_bus)] = pop!(subnetworks, bus_key)
        elseif length(ref_bus) == 0
            bus_groups[bus_key] = pop!(subnetworks, bus_key)
            @warn "No reference bus in the subnetwork associated with bus $bus_key. Reference bus assigned arbitrarily"
        elseif length(ref_bus) > 1
            representative = argmin(r -> (get(ref_angles, r, 0.0), r), ref_bus)
            bus_groups[representative] = pop!(subnetworks, bus_key)
        else
            @assert false
        end
    end
    return bus_groups
end
