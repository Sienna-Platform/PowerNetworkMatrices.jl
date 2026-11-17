# Without this workload, PowerNetworkMatrices compiles the Apple Accelerate kernels on first
# use. The KLU kernels need no workload. PowerNetworkMatrices builds each `FactorCore` with its
# own `network_data` type, so only the leaf kernels compiled here are reused downstream.
@setup_workload begin
    arc_ax = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    bus_ax = [1, 2, 3, 4, 5]
    b = [10.0, 5.0, 8.0, 4.0, 6.0, 7.0]
    subnetwork_axes = Dict(1 => (copy(arc_ax), copy(bus_ax)))
    @compile_workload begin
        if _has_apple_accelerate_backend()
            for tol in (DEFAULT_AUTO_TOLERANCE, 0.0)
                core = FactorCore(
                    b,
                    Set([1]),
                    (arc_ax, bus_ax),
                    subnetwork_axes,
                    ones(Int, length(arc_ax));
                    network_data = nothing,
                    linear_solver = AppleAccelerateLUSolver(),
                    tol = tol,
                )
                row = apply_cutoff(get_cutoff(core), ptdf_row(core, 1))
                cache = RowCache(10^6, Set{Int}(), length(row), length(arc_ax))
                cache[1] = row
                cache[1]
                mod = NetworkModification("outage", [ArcModification(1, -b[1])])
                apply_woodbury_correction(core, 2, compute_woodbury_factors(core, mod))
                # The full PTDF matrix solves every column at once.
                out = zeros(length(bus_ax), length(arc_ax))
                AccelerateWrapper.solve_sparse!(
                    aa_factorize(calculate_ABA_matrix(core.A, core.BA, Set([1]))),
                    core.BA[core.valid_ix, :],
                    view(out, core.valid_ix, :),
                )
            end
        end
    end
end
