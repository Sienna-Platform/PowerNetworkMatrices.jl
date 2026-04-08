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

