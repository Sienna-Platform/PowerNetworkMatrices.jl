# Accumulation helpers used by compute_ybus_delta in network_modification.jl

function _accumulate_arc_delta!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    fb_ix::Int,
    tb_ix::Int,
    delta_y11::YBUS_ELTYPE,
    delta_y12::YBUS_ELTYPE,
    delta_y21::YBUS_ELTYPE,
    delta_y22::YBUS_ELTYPE,
)
    push!(I, fb_ix, fb_ix, tb_ix, tb_ix)
    push!(J, fb_ix, tb_ix, fb_ix, tb_ix)
    push!(V, delta_y11, delta_y12, delta_y21, delta_y22)
    return
end

function _accumulate_branch_outage!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    br::Union{
        PSY.ACTransmission,
        BranchesParallel,
        BranchesSeries,
        ThreeWindingTransformerWinding,
    },
    fb_ix::Int,
    tb_ix::Int,
)
    Y11, Y12, Y21, Y22 = ybus_branch_entries(br)
    _accumulate_arc_delta!(
        I, J, V, fb_ix, tb_ix,
        YBUS_ELTYPE(-Y11), YBUS_ELTYPE(-Y12),
        YBUS_ELTYPE(-Y21), YBUS_ELTYPE(-Y22),
    )
    return
end

"""
Accumulate Ybus delta entries for a partial outage on a parallel branch group.
Greedily matches circuit(s) by susceptance and negates their Pi-model entries.
Supports multi-circuit outages where the total delta_b is the sum of tripped circuits.
"""
function _accumulate_parallel_partial_outage!(
    I::Vector{Int},
    J::Vector{Int},
    V::Vector{YBUS_ELTYPE},
    bp::BranchesParallel,
    fb_ix::Int,
    tb_ix::Int,
    delta_b::Float64,
)
    remaining_delta = delta_b
    for br in bp.branches
        b_circuit = get_series_susceptance(br)
        if isapprox(-b_circuit, remaining_delta; atol = YBUS_DELTA_TOL, rtol = 0)
            _accumulate_branch_outage!(I, J, V, br, fb_ix, tb_ix)
            return
        elseif -b_circuit > remaining_delta
            _accumulate_branch_outage!(I, J, V, br, fb_ix, tb_ix)
            remaining_delta += b_circuit
        end
    end
    # Check that all delta_b was consumed by matched circuits.
    if abs(remaining_delta) < YBUS_DELTA_TOL
        return
    end
    error(
        "Could not resolve partial parallel outage to individual circuit Pi-models. " *
        "Arc delta_b=$(delta_b), unmatched remainder=$(remaining_delta).",
    )
end
