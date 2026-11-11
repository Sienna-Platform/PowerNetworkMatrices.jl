# Solves and the scratch slot go through `with_solver`, which holds the core's `solver_lock`.
"""
    _compute_woodbury_factors(mat, modifications[, labeler]) -> WoodburyFactors

Woodbury factors `Z[:,j] = B⁻¹ν_j` for each modified arc, on a `VirtualPTDF`, a `VirtualMODF`
or a `VirtualFactorCore` (solving on that core's factorization).
"""
_compute_woodbury_factors(
    mat::Union{VirtualPTDF, VirtualMODF},
    modifications::Tuple{Vararg{ArcModification}},
) = _compute_woodbury_factors(get_core(mat), modifications)

function _apply_woodbury_correction(
    vptdf::VirtualPTDF,
    monitored_idx::Int,
    wf::WoodburyFactors,
)
    row = _apply_woodbury_correction(get_core(vptdf), monitored_idx, wf)
    return _apply_dist_slack!(row, vptdf, wf)
end

_apply_woodbury_correction(
    vmodf::VirtualMODF,
    monitored_idx::Int,
    wf::WoodburyFactors,
) = _apply_woodbury_correction(get_core(vmodf), monitored_idx, wf)

function _apply_dist_slack!(row::Vector{Float64}, vptdf::VirtualPTDF, wf::WoodburyFactors)
    _use_dist_slack(vptdf) || return row
    if wf.is_islanding
        error(
            "Distributed slack is not supported for a modification that islands the network.",
        )
    end
    # A distributed slack withdraws each injection in proportion to the slack weights, which
    # shifts every entry of a sensitivity row by the row's weighted mean.
    row .-= LinearAlgebra.dot(row, get_dist_slack_normalized(vptdf))
    return row
end
