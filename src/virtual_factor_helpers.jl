"""
    _extract_branch_susceptances_by_arc(BA, arc_ax, nr_data) -> Vector{Vector{Float64}}

Extract per-branch susceptances for each arc. For arcs with a single branch,
returns a one-element vector equal to the arc susceptance. For arcs with
parallel branches (double circuits), returns one entry per branch. For arcs
with series-reduced branches (D2 reduction), returns one entry per segment.

This enables single-branch contingencies on parallel and series-reduced arcs.
"""
function _extract_branch_susceptances_by_arc(
    BA::SparseArrays.SparseMatrixCSC{Float64, Int},
    arc_ax::Vector{Tuple{Int, Int}},
    nr_data::NetworkReductionData,
)::Vector{Vector{Float64}}
    n_arcs = size(BA, 2)
    nzv = SparseArrays.nonzeros(BA)
    result = Vector{Vector{Float64}}(undef, n_arcs)

    for j in 1:n_arcs
        arc = arc_ax[j]

        if haskey(nr_data.parallel_branch_map, arc)
            bp = nr_data.parallel_branch_map[arc]
            result[j] = Float64[
                _finite_series_susceptance(branch, nr_data) for branch in bp.branches
            ]
        elseif haskey(nr_data.series_branch_map, arc)
            bs = nr_data.series_branch_map[arc]
            result[j] = Float64[
                _finite_series_susceptance(segment, nr_data) for segment in bs
            ]
        else
            result[j] = [_ba_column_susceptance(BA, nzv, j)]
        end
    end

    return result
end

# Shared by VirtualPTDF and VirtualLODF: one row per arc, each `n_bus` long.
function _persistent_row_cache(
    max_cache_size::Int,
    core,
    persistent_arcs::Vector{Tuple{Int, Int}},
)
    return RowCache(
        max_cache_size * MiB,
        Set{Int}(core.lookup[1][k] for k in persistent_arcs),
        length(core.axes[2]) * sizeof(Float64),
        length(core.axes[1]),
    )
end
