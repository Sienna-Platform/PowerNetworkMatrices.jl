import LinearAlgebra

# Signed DC Laplacian of the network `mods` takes the base network to.
function _signed_laplacian(core, mods)
    n_bus = size(core.BA, 1)
    B = zeros(n_bus, n_bus)
    for e in 1:size(core.BA, 2)
        b =
            core.arc_susceptance_signs[e] *
            PNM._post_modification_susceptance(core.arc_susceptances, e, mods)
        f, t = PMC._arc_ends(core, e)
        B[f, f] += b
        B[t, t] += b
        B[f, t] -= b
        B[t, f] -= b
    end
    return B
end

# Largest entry of `B̃ + U D Uᵀ − B_post` on the buses live after the change, where `B̃` is the
# Schur complement of the starting network's live block onto them and `post` takes the base
# network to the network after the change.
function _basis_identity_error(core, basis, start, start_labels, post, post_labels)
    n_bus = size(core.BA, 1)
    refs = PNM.get_ref_bus_position(core)
    live_start = PMC._live_buses(start_labels, refs, n_bus)
    live_post = PMC._live_buses(post_labels, refs, n_bus)
    L = findall(live_post)
    X = findall(live_start .& .!live_post)
    B_start = _signed_laplacian(core, start)
    B_tilde = B_start[L, L]
    if !isempty(X)
        B_tilde -= B_start[L, X] * (B_start[X, X] \ B_start[X, L])
    end
    U = zeros(n_bus, length(basis))
    for j in 1:length(basis)
        U[basis.p[j], j] += basis.scale[j]
        U[basis.q[j], j] -= basis.scale[j]
    end
    B_post = _signed_laplacian(core, post)
    return maximum(abs, B_tilde + U[L, :] * basis.D * U[L, :]' - B_post[L, L])
end

# Arc-form columns are `ν_e = BA[:, e] / arc_sus[e]` in the change's order and `D` is the
# diagonal of signed susceptance changes.
function _test_arc_form_columns(core, basis, change)
    kept = [m for m in change if !iszero(m.delta_b)]
    @test length(basis) == length(kept)
    for (j, m) in enumerate(kept)
        e = m.arc_index
        column = zeros(size(core.BA, 1))
        column[basis.p[j]] += basis.scale[j]
        column[basis.q[j]] -= basis.scale[j]
        @test column ≈ Vector(core.BA[:, e]) / core.arc_susceptances[e]
        @test basis.scale[j] == core.arc_susceptance_signs[e]
    end
    signed_delta_b = [core.arc_susceptance_signs[m.arc_index] * m.delta_b for m in kept]
    @test basis.D == LinearAlgebra.diagm(signed_delta_b)
end

# The base solves `(p, q) -> B⁻¹(e_min − e_max)` of `vmodf`'s tier-2 column cache, read under
# the core's `solver_lock` as every `Tier2Cache` function requires.
function _tier2_column(vmodf)
    core = PNM.get_core(vmodf)
    cache = PNM.get_tier2_cache(vmodf)
    return (p, q) -> @lock core.solver_lock PMC._cached_column!(cache, core, p, q)
end

# (components out, minimal rank, arc form)
const _CORRIDOR_RANKS = [
    (["s15"], 1, true),
    (["L24"], 1, true),
    (["L56", "s15", "s67"], 1, false),
    (["s15", "s83", "s92"], 2, false),
    (["s15", "L56", "s67", "L78", "s83", "L79", "s92"], 2, false),
    (["L12", "L23", "L13"], 2, false),
    (["L78", "s83"], 1, false),
]

@testset "the basis has the minimal rank of the change and reproduces it" begin
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    for (names, rank, arc_form) in _CORRIDOR_RANKS
        mods = PNM._arc_modification_vector(
            NetworkModification(vmodf, sys, _out_set(sys, names)))
        labels = PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances, mods,
            size(core.BA, 1))
        basis = PMC._woodbury_basis(core, PNM.ArcModification[], Int[], mods, labels)
        @test length(basis) == rank
        @test basis.arc_form == arc_form
        @test length(basis) <= length(mods)
        @test _basis_identity_error(core, basis, PNM.ArcModification[], Int[], mods,
            labels) < 1e-10
        arc_form && _test_arc_form_columns(core, basis, mods)
    end
end

# (components out on the starting network, components the change adds, minimal rank, arc form)
const _CORRIDOR_TIER2 = [
    (["s15"], ["L24"], 1, true),
    (["L56", "s15", "s67"], ["L78", "s83"], 1, false),
]

@testset "the basis reproduces a change on a starting network" begin
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    n_bus = size(core.BA, 1)
    for (start_names, change_names, rank, arc_form) in _CORRIDOR_TIER2
        topology = NetworkModification(vmodf, sys, _out_set(sys, start_names))
        post = NetworkModification(vmodf, sys, _out_set(sys, [start_names; change_names]))
        change = PNM._arc_modification_vector(PNM._tier2_modification(post, topology))
        start = PNM._arc_modification_vector(topology)
        start_labels = PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances,
            start, n_bus)
        post_labels = PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances,
            PNM._arc_modification_vector(post), n_bus)
        basis = PMC._woodbury_basis(core, start, start_labels, change, post_labels)
        @test length(basis) == rank
        @test basis.arc_form == arc_form
        @test _basis_identity_error(core, basis, start, start_labels,
            PNM._arc_modification_vector(post), post_labels) < 1e-10
        arc_form && _test_arc_form_columns(core, basis, change)
    end
end

@testset "arc-form factors equal the per-arc assembly bit for bit" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    vptdf = VirtualPTDF(sys)
    core = PNM.get_core(vptdf)
    for names in (["Line3"], ["Line5", "Line9"])
        mods = PNM._arc_modification_vector(
            NetworkModification(vptdf, sys, _out_set(sys, names)))
        basis_factors = PNM._compute_woodbury_factors(core, mods)
        per_arc_factors = PNM.with_solver(core.K, core.work_ba_col, core.temp_data,
            core.solver_lock) do K, w, t
            PNM._woodbury_factors_from_Z(
                PMC._base_woodbury_columns(core, K, w, length(t), mods),
                PMC._arc_form_basis(core, mods), core.BA, core.arc_susceptance_signs,
                core.arc_susceptances, mods, PNM._arcs_fully_opened(core, mods))
        end
        @test basis_factors.basis.arc_form
        @test _bitwise_equal(basis_factors.Z, per_arc_factors.Z) &&
              _bitwise_equal(basis_factors.W_inv, per_arc_factors.W_inv)
    end
end

@testset "minimal-rank rows match rebuilt systems" begin
    for (names, rank, _) in _CORRIDOR_RANKS
        sys = _node_breaker_corridor()
        vmodf = VirtualMODF(VirtualPTDF(sys), sys)
        islands, n_islands =
            _checked_reference_islands(vmodf, sys, _node_breaker_corridor, names)
        ts = TopologySequence(vmodf, sys, [_out_set(sys, names)])
        arcs = PNM.get_arc_axis(vmodf)
        rows = Dict(a => get_topology_ptdf_rows(ts, a, [1])[1] for a in arcs)
        wf = PNM.get_woodbury_cache(vmodf)[PNM._topology_modification(ts, 1)]
        @test PNM._woodbury_rank(wf) == rank
        err =
            _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
                builder = _node_breaker_corridor,
                islands = n_islands > 1 ? islands : nothing)
        @test err < 1e-10
    end
end

@testset "minimal-rank rows honor distributed slack" begin
    names = ["L12", "L23", "L13"]
    sys = _node_breaker_corridor()
    buses = PSY.get_number.(PSY.get_components(PSY.ACBus, sys))
    dist = Dict(b => 1.0 / length(buses) for b in buses)
    vptdf = VirtualPTDF(sys; dist_slack = dist)
    wf = compute_woodbury_factors(
        vptdf,
        NetworkModification(vptdf, sys, _out_set(sys, names)),
    )
    @test PNM._woodbury_rank(wf) == 2
    @test !wf.basis.arc_form
    err = _max_error_against_rebuild(a -> apply_woodbury_correction(vptdf, a, wf),
        PNM.get_arc_axis(vptdf), PNM.get_bus_axis(vptdf), names;
        builder = _node_breaker_corridor, dist_slack = dist)
    @test err < 1e-10
end

@testset "a monitored arc outside the reference islands has the zero row" begin
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    ts = TopologySequence(vmodf, sys, [_out_set(sys, ["s15", "s83", "s92"])])
    for name in ("L56", "s67", "L78", "L79")
        @test all(iszero,
            get_topology_ptdf_rows(ts, _arc_numbers(_transmission(sys, name)), [1])[1])
    end
end

@testset "workers and populate_cache give a reduced contingency's rows" begin
    names = ["s15", "L56", "s67", "L78", "s83", "L79", "s92"]
    sys, vmodf, ids = _two_tier_fixture(_node_breaker_corridor, [names])
    ctg = PNM.get_registered_contingencies(vmodf)[ids[names]]
    arcs = PNM.get_arc_axis(vmodf)
    reference = Dict(a => vmodf[a, ctg] for a in arcs)
    _, vmodf2, ids2 = _two_tier_fixture(_node_breaker_corridor, [names])
    populate_cache(vmodf2, [ids2[names]]; monitored = arcs)
    worker = PNM.worker_core(vmodf2)
    for a in arcs
        @test _bitwise_equal(
            vmodf2[a, PNM.get_registered_contingencies(vmodf2)[ids2[names]]], reference[a])
        @test _bitwise_equal(
            get_post_modification_ptdf_row(vmodf2, a, ids2[names], worker), reference[a])
    end
end

@testset "base-network modification and contingency rows match the row plan" begin
    names = ["s15", "s83", "s92"]
    sys, vmodf, ids = _two_tier_fixture(_node_breaker_corridor, [names])
    core = PNM.get_core(vmodf)
    ctg = PNM.get_registered_contingencies(vmodf)[ids[names]]
    net = PostContingencyNetwork(vmodf, ids[names])
    mod = NetworkModification(vmodf, sys, _out_set(sys, ["L56", "s15", "s67"]))
    arcs = PNM.get_arc_axis(vmodf)
    function planned_row(a, modification)
        m = PNM.get_arc_lookup(vmodf)[a]
        wf = PNM.get_woodbury_cache(vmodf)[modification]
        plan = PNM._base_topology_row_plan(
            core,
            m,
            PNM._arc_modification_vector(modification),
            wf,
        )
        return PNM.apply_cutoff(PNM.get_cutoff(vmodf),
            only(PNM._rows_from_plans(core, m, (plan,))))
    end
    @test !PNM._get_woodbury_factors(vmodf, mod).basis.arc_form
    for a in arcs
        @test _bitwise_equal(vmodf[a, mod], planned_row(a, mod))
        @test _bitwise_equal(vmodf[a, ctg], planned_row(a, ctg.modification))
        @test _bitwise_equal(
            only(get_post_contingency_ptdf_rows(vmodf, a, [net])), vmodf[a, ctg])
    end
    PNM.clear_caches!(vmodf)
    for a in arcs
        @test _bitwise_equal(
            get_post_modification_ptdf_rows(vmodf, a, [mod])[1], planned_row(a, mod))
        @test _bitwise_equal(
            only(get_post_contingency_ptdf_rows(vmodf, a, [net])),
            planned_row(a, ctg.modification))
    end
end

# (components out at K, contingency, minimal tier-2 rank, or 0 with no tier-2 factors)
const _CORRIDOR_TIER2_ROWS = [
    (["L56", "s15", "s67"], ["L78", "s83"], 1),
    (["L56", "s15", "s67"], ["L79", "s92"], 1),
    (["L56", "s15", "s67"], ["L78", "s83", "L79", "s92"], 1),
    (["L24"], ["s15", "L56", "s67", "L78", "s83", "L79", "s92"], 2),
    # The contingency lies outside K's reference islands: K's rows, no tier-2 factors.
    (["s15", "s83", "s92"], ["L78"], 0),
]

@testset "minimal-rank tier-2 rows match rebuilt systems" begin
    sys, vmodf, ids =
        _two_tier_fixture(_node_breaker_corridor, [c[2] for c in _CORRIDOR_TIER2_ROWS])
    topologies = unique([c[1] for c in _CORRIDOR_TIER2_ROWS])
    ts = TopologySequence(vmodf, sys, [_out_set(sys, k) for k in topologies])
    arcs = PNM.get_arc_axis(vmodf)
    for (k, c, rank) in _CORRIDOR_TIER2_ROWS
        net = get_post_contingency_network(ts, findfirst(==(k), topologies), ids[c])
        rows = Dict(
            a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs
        )
        tier2_factors = PNM.get_tier2_factor_cache(vmodf)
        if haskey(tier2_factors, net)
            @test PNM._woodbury_rank(tier2_factors[net]) == rank
        else
            @test rank == 0
        end
        names = union(k, c)
        islands, n = _checked_reference_islands(vmodf, sys, _node_breaker_corridor, names)
        if n == 1
            islands = nothing
        end
        @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
            builder = _node_breaker_corridor, islands = islands) < 1e-10
    end
end

@testset "a tier-2 column shared by two topologies is solved once" begin
    c = ["L79", "s92"]
    sys, vmodf, ids = _two_tier_fixture(_node_breaker_corridor, [c])
    ts = TopologySequence(vmodf, sys,
        [_out_set(sys, ["L56", "s15", "s67"]), _out_set(sys, ["L24"])])
    nets = [get_post_contingency_network(ts, t, ids[c]) for t in 1:2]
    get_post_contingency_ptdf_rows(vmodf, (1, 2), nets)
    stats = cache_statistics(vmodf)
    @test stats.columns.misses == 1
    @test stats.columns.hits >= 1
    clear_caches!(vmodf)
    @test cache_statistics(vmodf).columns ==
          (; entries = 0, bytes = 0, hits = 0, misses = 0)
end

@testset "the column cache is bounded and evicts the least recently used column" begin
    sys, vmodf0, ids = _two_tier_fixture(_node_breaker_corridor, [["L79", "s92"]])
    n_bus = size(vmodf0)[2]
    vmodf = VirtualMODF(PNM.get_core(vmodf0), sys;
        max_column_cache_bytes = 2 * n_bus * sizeof(Float64))
    column = _tier2_column(vmodf)
    for (p, q) in ((1, 2), (1, 3), (1, 2), (2, 3))
        column(p, q)
    end
    @test Set(keys(PMC.get_column_cache(PNM.get_tier2_cache(vmodf)))) ==
          Set([(1, 2), (2, 3)])
    stats = cache_statistics(vmodf).columns
    @test stats.entries == 2
    @test stats.bytes == 2 * n_bus * sizeof(Float64)
    @test (stats.hits, stats.misses) == (1, 3)
end

@testset "a rank-0 tier-2 change that kills a bus zeroes its entries" begin
    # On K bus 8 hangs from bus 7 by L78 alone; opening L78 kills it with no column.
    k, c = ["s83", "s92"], ["L78"]
    sys, vmodf, ids = _two_tier_fixture(_node_breaker_corridor, [c])
    ts = TopologySequence(vmodf, sys, [_out_set(sys, k)])
    net = get_post_contingency_network(ts, 1, ids[c])
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(
        a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs
    )
    tier2 = PNM.get_tier2_factor_cache(vmodf)[net]
    @test PNM._woodbury_rank(tier2) == 0
    @test tier2.is_islanding
    bus8 = PNM.get_bus_lookup(vmodf)[8]
    @test all(a -> rows[a][bus8] == 0.0, arcs)
    names = union(k, c)
    islands, n = _checked_reference_islands(vmodf, sys, _node_breaker_corridor, names)
    @test n > 1
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
        builder = _node_breaker_corridor, islands = islands) < 1e-10
end

@testset "retaining networks keeps factors and columns, drops rows and tier-2 factors" begin
    c = ["L79", "s92"]
    sys, vmodf, ids = _two_tier_fixture(_node_breaker_corridor, [c])
    ts = TopologySequence(vmodf, sys, [_out_set(sys, ["L24"])])
    net = get_post_contingency_network(ts, 1, ids[c])
    get_post_contingency_ptdf_rows(vmodf, (1, 2), [net])
    get_ptdf_row(vmodf, (1, 2))
    retain_modifications!(vmodf, [net])
    stats = cache_statistics(vmodf)
    @test haskey(PNM.get_woodbury_cache(vmodf), net.topology)
    @test stats.tier2_factors.entries == 0
    @test isempty(PNM.get_tier2_row_caches(vmodf))
    @test keys(PNM.get_row_caches(vmodf)) == Set([PNM._BASE_NETWORK])
    @test stats.columns.entries >= 1
end

# L78 (x = -0.01) and s83 (x = 0.01) cancel at bus 8, so outaging both leaves a dead block
# with zero susceptance.
_compensated_corridor() = _node_breaker_corridor(; reactances = Dict("L78" => -0.01))

@testset "a series-compensated dead section falls back to arc form" begin
    c = ["L78", "s83"]
    sys = _compensated_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    ts = TopologySequence(vmodf, sys, [_out_set(sys, c)])
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => get_topology_ptdf_rows(ts, a, [1])[1] for a in arcs)
    wf = PNM.get_woodbury_cache(vmodf)[PNM._topology_modification(ts, 1)]
    @test wf.basis.arc_form
    @test PNM._woodbury_rank(wf) == 2
    islands, _ = _checked_reference_islands(vmodf, sys, _compensated_corridor, c)
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), c;
        builder = _compensated_corridor, islands = islands) < 1e-10

    k = ["L56", "s15", "s67"]
    sys, vmodf, ids = _two_tier_fixture(_compensated_corridor, [c])
    ts = TopologySequence(vmodf, sys, [_out_set(sys, k)])
    net = get_post_contingency_network(ts, 1, ids[c])
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    @test PNM.get_tier2_factor_cache(vmodf)[net].basis.arc_form
    names = union(k, c)
    islands, _ = _checked_reference_islands(vmodf, sys, _compensated_corridor, names)
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
        builder = _compensated_corridor, islands = islands) < 1e-10
end

# `basis` with `D = -K⁻¹`, so that `S = I + D K` is zero.
function _singular_basis(basis, Z)
    r = length(basis)
    K = [basis.scale[i] * (Z[basis.p[i], j] - Z[basis.q[i], j]) for i in 1:r, j in 1:r]
    return PNM.WoodburyBasis(basis.p, basis.q, basis.scale, -inv(K), false)
end

@testset "a singular S on the base network falls back to the arc-form factors" begin
    names = ["s15", "s83", "s92"]
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    mods = PNM._arc_modification_vector(
        NetworkModification(vmodf, sys, _out_set(sys, names)))
    labels = PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances, mods,
        size(core.BA, 1))
    basis = PMC._woodbury_basis(core, PNM.ArcModification[], Int[], mods, labels)
    @test !basis.arc_form
    Z, arc_Z = PNM.with_solver(core.K, core.work_ba_col, core.temp_data,
        core.solver_lock) do K, w, t
        (PMC._basis_columns(core, K, w, length(t), basis, mods),
            PMC._base_woodbury_columns(core, K, w, length(t), mods))
    end
    arc_out = PNM._arcs_fully_opened(core, mods)
    refs = PNM.get_ref_bus_position(core)
    fallback_calls = Ref(0)
    fallback = function ()
        fallback_calls[] += 1
        return PMC._arc_form_factors(arc_Z, PMC._arc_form_basis(core, mods), core.BA,
            core.arc_susceptance_signs, core.arc_susceptances, mods, arc_out, labels,
            refs)
    end
    wf = PMC._woodbury_factors_from_basis(Z, _singular_basis(basis, Z), core.BA,
        core.arc_susceptance_signs, core.arc_susceptances, mods, arc_out, labels, refs,
        fallback)
    @test fallback_calls[] == 1
    @test wf.basis.arc_form
    @test wf.is_islanding
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(
        a => PNM.apply_woodbury_correction(core, i, wf) for (i, a) in enumerate(arcs)
    )
    islands, _ = _checked_reference_islands(vmodf, sys, _node_breaker_corridor, names)
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
        builder = _node_breaker_corridor, islands = islands) < 1e-10
end

@testset "a singular S on a topology falls back to the tier-2 arc-form factors" begin
    k, c = ["L56", "s15", "s67"], ["L78", "s83"]
    sys, vmodf, ids = _two_tier_fixture(_node_breaker_corridor, [c])
    core = PNM.get_core(vmodf)
    ts = TopologySequence(vmodf, sys, [_out_set(sys, k)])
    net = get_post_contingency_network(ts, 1, ids[c])
    topology_wf = PNM._get_woodbury_factors(vmodf, net.topology)
    change = PNM._arc_modification_vector(net.tier2)
    post = PNM._arc_modification_vector(net.post_contingency)
    labels = PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances, post,
        size(core.BA, 1))
    basis = PMC._woodbury_basis(core, PNM._arc_modification_vector(net.topology),
        topology_wf.bus_island_labels, change, labels)
    @test !basis.arc_form
    column = _tier2_column(vmodf)
    Z = PMC._tier2_columns(column, topology_wf, basis, change)
    arc_out = Bool[PMC._opens_every_member_at(core, post, m.arc_index) for m in change]
    refs = PNM.get_ref_bus_position(core)
    fallback_calls = Ref(0)
    fallback = function ()
        fallback_calls[] += 1
        return PMC._tier2_arc_form_factors(column, core, topology_wf, change, arc_out,
            labels)
    end
    wf = PMC._woodbury_factors_from_basis(Z, _singular_basis(basis, Z), core.BA,
        core.arc_susceptance_signs, core.arc_susceptances, change, arc_out, labels,
        refs,
        fallback)
    @test fallback_calls[] == 1
    @test wf.basis.arc_form
    PNM.get_tier2_factor_cache(vmodf)[net] = wf
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    @test PNM.get_tier2_factor_cache(vmodf)[net] === wf
    names = union(k, c)
    islands, _ = _checked_reference_islands(vmodf, sys, _node_breaker_corridor, names)
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
        builder = _node_breaker_corridor, islands = islands) < 1e-10

    # A modification that leaves its arc unchanged gets no column and no factor entry.
    l24 = PNM.get_arc_lookup(vmodf)[_arc_numbers(_transmission(sys, "L24"))]
    padded = [change[1]; PNM.ArcModification(l24, 0.0); change[2:end]]
    padded_out = [arc_out[1]; false; arc_out[2:end]]
    padded_Z = PMC._tier2_columns(column, topology_wf, PMC._arc_form_basis(core, padded),
        padded)
    arc_Z =
        PMC._tier2_columns(column, topology_wf, PMC._arc_form_basis(core, change), change)
    @test padded_Z == arc_Z
    padded_wf =
        PMC._tier2_arc_form_factors(column, core, topology_wf, padded, padded_out, labels)
    @test _bitwise_equal(padded_wf.Z, wf.Z) && _bitwise_equal(padded_wf.W_inv, wf.W_inv)
    @test padded_wf.arc_indices == wf.arc_indices
end

# The starting network with the corridor's tap buses 5-9 dead, its labels and the network
# after `change_names` on top.
function _dead_corridor_start(vmodf, sys, change_names)
    core = PNM.get_core(vmodf)
    n_bus = size(core.BA, 1)
    k = ["s15", "s83", "s92"]
    topology = NetworkModification(vmodf, sys, _out_set(sys, k))
    post = NetworkModification(vmodf, sys, _out_set(sys, [k; change_names]))
    change = PNM._arc_modification_vector(PNM._tier2_modification(post, topology))
    start = PNM._arc_modification_vector(topology)
    start_labels =
        PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances, start, n_bus)
    post_labels = PNM._post_contingency_bus_labels(core.BA, core.arc_susceptances,
        PNM._arc_modification_vector(post), n_bus)
    return start, start_labels, change, post_labels
end

@testset "changed arcs dead on the starting network leave the basis" begin
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    start, start_labels, change, post_labels = _dead_corridor_start(vmodf, sys, ["L78"])
    basis = PMC._woodbury_basis(core, start, start_labels, change, post_labels)
    @test length(basis) == 0
    @test !basis.arc_form

    start, start_labels, change, post_labels =
        _dead_corridor_start(vmodf, sys, ["L78", "L24"])
    basis = PMC._woodbury_basis(core, start, start_labels, change, post_labels)
    @test basis.arc_form
    l24 = PNM.get_arc_lookup(vmodf)[_arc_numbers(_transmission(sys, "L24"))]
    _test_arc_form_columns(core, basis, [m for m in change if m.arc_index == l24])
end

@testset "arc-form factors with a dead bus carry the island labels" begin
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    start, _, change, post_labels = _dead_corridor_start(vmodf, sys, ["L78"])
    topology_wf = PNM._compute_woodbury_factors(core, start)
    basis = PMC._arc_form_basis(core, change)
    arc_out = fill(true, length(change))
    wf = PMC._woodbury_factors_from_basis(
        PMC._tier2_columns(_tier2_column(vmodf), topology_wf, basis, change), basis,
        core.BA,
        core.arc_susceptance_signs, core.arc_susceptances, change, arc_out, post_labels,
        PNM.get_ref_bus_position(core), () -> error("no fallback expected"))
    @test wf.is_islanding
    @test wf.bus_island_labels == post_labels
end

@testset "a change that revives a dead bus is rejected" begin
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    start, start_labels, _, _ = _dead_corridor_start(vmodf, sys, String[])
    err = try
        PMC._woodbury_basis(core, start, start_labels, PNM.ArcModification[], Int[])
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("bus 5", sprint(showerror, err))
end

@testset "Woodbury flows read zero outside the reference islands" begin
    names = ["s15", "s83", "s92"]
    sys = _node_breaker_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    core = PNM.get_core(vmodf)
    mod = NetworkModification(vmodf, sys, _out_set(sys, names))
    wf = compute_woodbury_factors(vmodf, mod)
    @test wf.is_islanding
    n_bus, n_arc = size(core.BA)
    live = PMC._live_buses(wf.bus_island_labels, PNM.get_ref_bus_position(core), n_bus)
    p = collect(range(-1.0, 1.0; length = n_bus)) .* live
    θ = PNM.solve_bus_angles!(zeros(n_bus), vmodf, p)
    arcs = collect(1:n_arc)
    flows = PNM.arc_flows!(zeros(n_arc), vmodf, θ, arcs, wf)
    for a in arcs
        row = PNM.apply_woodbury_correction(core, a, wf)
        @test flows[a] ≈ LinearAlgebra.dot(row, p) atol = 1e-12
    end
    arc_lookup = PNM.get_arc_lookup(vmodf)
    for name in ("L56", "s67", "L78", "L79")
        @test flows[arc_lookup[_arc_numbers(_transmission(sys, name))]] === 0.0
    end
end

@testset "rows on two subnetworks match rebuilt systems" begin
    c = ["s1013", "s1412"]
    sys = _two_subnetwork_corridor()
    vmodf = VirtualMODF(VirtualPTDF(sys), sys)
    ts = TopologySequence(vmodf, sys, [_out_set(sys, c)])
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => get_topology_ptdf_rows(ts, a, [1])[1] for a in arcs)
    wf = PNM.get_woodbury_cache(vmodf)[PNM._topology_modification(ts, 1)]
    @test PNM._woodbury_rank(wf) == 1
    @test !wf.basis.arc_form
    islands, _ = _checked_reference_islands(vmodf, sys, _two_subnetwork_corridor, c)
    @test length(islands) == 2
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), c;
        builder = _two_subnetwork_corridor, islands = islands) < 1e-10

    k = ["L56", "s15", "s67"]
    sys, vmodf, ids = _two_tier_fixture(_two_subnetwork_corridor, [c])
    ts = TopologySequence(vmodf, sys, [_out_set(sys, k)])
    net = get_post_contingency_network(ts, 1, ids[c])
    arcs = PNM.get_arc_axis(vmodf)
    rows = Dict(a => only(get_post_contingency_ptdf_rows(vmodf, a, [net])) for a in arcs)
    tier2 = PNM.get_tier2_factor_cache(vmodf)[net]
    @test PNM._woodbury_rank(tier2) == 1
    @test !tier2.basis.arc_form
    names = union(k, c)
    islands, _ = _checked_reference_islands(vmodf, sys, _two_subnetwork_corridor, names)
    @test _max_error_against_rebuild(a -> rows[a], arcs, PNM.get_bus_axis(vmodf), names;
        builder = _two_subnetwork_corridor, islands = islands) < 1e-10
end
