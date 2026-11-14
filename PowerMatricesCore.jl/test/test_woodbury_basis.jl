import LinearAlgebra

# Largest entry of `B̃ + U D Uᵀ − B_post` on the live buses, with `B̃` the Schur complement of the
# base Laplacian onto them.
function _five_bus_identity_error(core, basis, out, live)
    n_bus = size(core.BA, 1)
    dead = setdiff(1:n_bus, live)
    B = _five_bus_laplacian(core, Int[])
    B_tilde = B[live, live]
    if !isempty(dead)
        B_tilde -= B[live, dead] * (B[dead, dead] \ B[dead, live])
    end
    U = zeros(n_bus, length(basis))
    for j in 1:length(basis)
        U[basis.p[j], j] += basis.scale[j]
        U[basis.q[j], j] -= basis.scale[j]
    end
    B_post = _five_bus_laplacian(core, out)
    return maximum(abs, B_tilde + U[live, :] * basis.D * U[live, :]' - B_post[live, live])
end

@testset "a singular dead block has no Kron reduction" begin
    # Dead node 3 joined to terminals 1 and 2 by susceptances that cancel.
    @test PMC._kron_component([1, 2], [3], [(1, 3, 2.0), (2, 3, -2.0)]) === nothing
    R = PMC._kron_component([1, 2], [3], [(1, 3, 2.0), (2, 3, 2.0)])
    @test R ≈ fill(1.0, 1, 1)
end

@testset "an empty label vector makes every bus live" begin
    @test PMC._live_buses(Int[], [1], 3) == [true, true, true]
    @test PMC._live_buses([1, 1, 3], [1], 3) == [true, true, false]
end

# (open arcs, minimal rank, arc form, buses live after the outage)
const _FIVE_BUS_CASES = [
    ([1], 1, true, [1, 2, 3, 4, 5]),
    ([1, 6], 2, true, [1, 2, 3, 4, 5]),
    ([4, 6], 1, false, [1, 2, 3, 5]),
    ([2, 3, 5], 2, false, [1, 2, 4, 5]),
]

@testset "the five-bus basis has minimal rank and reproduces the outage" begin
    core = _five_bus_core()
    n_bus = size(core.BA, 1)
    for (out, rank, arc_form, live) in _FIVE_BUS_CASES
        mods = _five_bus_outage(out)
        labels = PMC._post_contingency_bus_labels(core.BA, core.arc_susceptances, mods,
            n_bus)
        basis = PMC._woodbury_basis(core, PMC.ArcModification[], Int[], mods, labels)
        @test length(basis) == rank
        @test basis.arc_form == arc_form
        @test findall(PMC._live_buses(labels, [1], n_bus)) == live
        @test _five_bus_identity_error(core, basis, out, live) < 1e-12
    end
end

@testset "five-bus rows over the minimal-rank basis match a dense post-outage PTDF" begin
    core = _five_bus_core()
    n_arc = size(core.BA, 2)
    for (out, rank, arc_form, live) in _FIVE_BUS_CASES
        wf = PMC._compute_woodbury_factors(core, _five_bus_outage(out))
        @test PMC._woodbury_rank(wf) == rank
        @test wf.basis.arc_form == arc_form
        @test wf.is_islanding == (length(live) < 5)
        expected = _five_bus_dense_ptdf(core, out, live)
        for e in 1:n_arc
            @test PMC.apply_woodbury_correction(core, e, wf) ≈ expected[e, :] atol = 1e-12
            from_bus = first(PMC._arc_ends(core, e))
            @test PMC._outside_reference_islands(core, wf, e) == !(from_bus in live)
        end
    end
end

@testset "five-bus Woodbury flows read zero outside the reference islands" begin
    core = _five_bus_core()
    n_bus, n_arc = size(core.BA)
    out, live = [4, 6], [1, 2, 3, 5]
    wf = PMC._compute_woodbury_factors(core, _five_bus_outage(out))
    p = [b in live ? 0.1 * b : 0.0 for b in 1:n_bus]
    θ = PMC.solve_bus_angles!(zeros(n_bus), core, p)
    flows = PMC.arc_flows!(zeros(n_arc), core, θ, collect(1:n_arc), wf)
    expected = _five_bus_dense_ptdf(core, out, live) * p
    @test flows ≈ expected atol = 1e-12
    @test flows[4] === 0.0 && flows[6] === 0.0
end

@testset "five-bus arc-form factors equal the per-arc assembly bit for bit" begin
    core = _five_bus_core()
    for out in ([1], [1, 6])
        mods = _five_bus_outage(out)
        basis_factors = PMC._compute_woodbury_factors(core, mods)
        per_arc_factors = PMC.with_solver(core.K, core.work_ba_col, core.temp_data,
            core.solver_lock) do K, w, t
            PMC._woodbury_factors_from_Z(
                PMC._base_woodbury_columns(core, K, w, length(t), mods),
                PMC._arc_form_basis(core, mods), core.BA, core.arc_susceptance_signs,
                core.arc_susceptances, mods, PMC._arcs_fully_opened(core, mods))
        end
        @test basis_factors.basis.arc_form
        @test isequal(basis_factors.Z, per_arc_factors.Z)
        @test isequal(basis_factors.W_inv, per_arc_factors.W_inv)
    end
end

@testset "a change cannot revive a bus dead on the starting network" begin
    core = _five_bus_core()
    n_bus = size(core.BA, 1)
    start = _five_bus_outage([4, 6])
    start_labels = PMC._post_contingency_bus_labels(core.BA, core.arc_susceptances, start,
        n_bus)
    @test_throws ArgumentError PMC._woodbury_basis(core, start, start_labels,
        _five_bus_outage([1]), Int[])
end

@testset "reference positions are the gaps of the valid buses" begin
    core = _five_bus_core()
    @test [PMC._reference_position(core.valid_ix, 1)] == PMC._ref_bus_positions(core)
    for refs in ([1], [3], [5], [2, 4], [1, 5], [1, 2, 3])
        valid = setdiff(1:5, refs)
        @test [PMC._reference_position(valid, j) for j in eachindex(refs)] == refs
    end
end
