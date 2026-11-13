# Solves and the scratch slot go through `with_solver`, which holds the core's `solver_lock`.
"""
    _compute_woodbury_factors(mat, modifications) -> WoodburyFactors

Woodbury factors `Z[:,j] = B⁻¹ν_j` for each modified arc. `mat` is a `VirtualPTDF` or a
`VirtualMODF`. The method solves on the factorization of the core of `mat`.
"""
_compute_woodbury_factors(
    mat::Union{VirtualPTDF, VirtualMODF},
    modifications::Vector{ArcModification},
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
    return apply_dist_slack!(row, get_core(vptdf), get_dist_slack_normalized(vptdf), wf)
end
