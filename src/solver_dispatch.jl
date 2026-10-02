# Shared dispatch for "run a solve under whatever solver we have." Lives
# outside the Virtual{PTDF, LODF, MODF} files so all three matrices share
# the same `with_solver` seam and the same KLU/AppleAccelerate factory.
#
# One factor + one cache per Virtual matrix. The per-cache `solver_lock`
# held here serializes solves on that cache and its single scratch slot,
# for both the KLU and `AAFactorCache` backends. Distinct caches need no
# process-wide lock (see `_LIBKLU_LOCK` in `KLUWrapper.jl`, Windows only).

"""
    with_solver(f, K, work_ba_col, temp_data, solver_lock) -> result

Acquire `solver_lock`, then invoke `f(K, work_ba_col[1], temp_data[1])`.
Generic over the solver cache (KLU or Apple Accelerate); serializes through
`solver_lock`; the per-cache scratch slot at index 1 is the only slot —
`work_ba_col` and `temp_data` are single-element vectors, kept as
`Vector{Vector{Float64}}` because `_solve_factorization` is typed on
`Vector{Float64}` and the two buffers have different lengths
(`n_buses` vs. `n_buses - n_ref_buses`).
"""
function with_solver(
    f::F,
    K::KT,
    work_ba_col::Vector{Vector{Float64}},
    temp_data::Vector{Vector{Float64}},
    solver_lock::ReentrantLock,
) where {F, KT}
    return @lock solver_lock f(K, work_ba_col[1], temp_data[1])
end
