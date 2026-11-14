const _Z32 = zero(ComplexF32)

# The five-bus network after the arcs `out` open, as a modification from the base network.
_five_bus_modification(label, out) = PMC.NetworkModification(label, _five_bus_outage(out))

# The network after `c` on the topology that opens `k`, with its tier-2 modification derived.
function _five_bus_network(k, c, id = 1)
    topology = isempty(k) ? PMC._BASE_NETWORK : _five_bus_modification("K", k)
    post = _five_bus_modification("K ∪ c", union(k, c))
    return PMC.PostContingencyNetwork(
        PMC._Tier2Derived(),
        topology,
        post,
        PMC._tier2_modification(post, topology),
        id,
    )
end

# Buses connected to reference bus 1 once the arcs `out` open.
function _five_bus_live(core, out)
    live = Set([1])
    grew = true
    while grew
        grew = false
        for e in 1:size(core.BA, 2)
            e in out && continue
            f, t = PMC._arc_ends(core, e)
            if (f in live) != (t in live)
                push!(live, f, t)
                grew = true
            end
        end
    end
    return sort!(collect(live))
end

# Empty tier-1 caches and a `Tier2Cache` for `core`.
function _five_bus_caches(max_column_cache_bytes = 2^20)
    return (
        PMC.Tier2Cache(max_column_cache_bytes),
        Dict{PMC.NetworkModification, PMC.WoodburyFactors}(),
        Dict{PMC.NetworkModification, PMC.RowCache{PMC.RowCacheValue}}(),
    )
end

function _five_bus_rows(core, caches, m, nets)
    cache, woodbury, rows = caches
    return @lock core.solver_lock PMC._post_contingency_ptdf_rows!(
        cache, core, woodbury, rows, 2^20, m, nets)
end

@testset "a tier-2 modification adds back to the post-contingency modification" begin
    # Arc 1 partly out on K and fully out after c; arc 3 out on K only; arc 2 out after c.
    topology = PMC.NetworkModification(
        "K",
        [
            PMC.ArcModification(1, -4.0, 0.0, _Z32, _Z32, _Z32, _Z32, 1),
            PMC.ArcModification(3, -8.0, 0.0, _Z32, _Z32, _Z32, _Z32, 1),
        ],
        [PMC.ShuntModification(2, ComplexF32(0.5))],
        false,
    )
    post = PMC.NetworkModification(
        "K ∪ c",
        [
            PMC.ArcModification(1, -10.0, 0.0, _Z32, _Z32, _Z32, _Z32, 2),
            PMC.ArcModification(2, -5.0, 0.0, _Z32, _Z32, _Z32, _Z32, 1),
        ],
        [PMC.ShuntModification(2, ComplexF32(0.5))],
        true,
    )
    tier2 = PMC._tier2_modification(post, topology)
    @test tier2.label == "K ∪ c"
    @test tier2.is_islanding
    @test [m.arc_index for m in tier2.arc_modifications] == [1, 2, 3]
    @test [m.delta_b for m in tier2.arc_modifications] == [-6.0, -5.0, 8.0]
    @test [m.opened for m in tier2.arc_modifications] == [1, 1, -1]
    @test isempty(tier2.shunt_modifications)
    merged = PMC._merge_arc_modifications(
        vcat(collect(topology.arc_modifications), collect(tier2.arc_modifications)),
    )
    @test [(m.arc_index, m.delta_b, m.opened) for m in merged if !PMC._is_zero_change(m)] ==
          [(m.arc_index, m.delta_b, m.opened) for m in post.arc_modifications]
end

@testset "post-contingency networks key by topology and tier-2 modification" begin
    net = _five_bus_network([6], [1], 1)
    relabeled = _five_bus_network([6], [1], 2)
    @test net == relabeled
    @test hash(net) == hash(relabeled)
    @test net.key_hash == hash((net.topology, net.tier2))
    @test net != _five_bus_network([6], [2])
    @test !PMC._on_base_topology(net)
    base = _five_bus_network(Int[], [1])
    @test PMC._on_base_topology(base)
    @test base.tier2 == base.post_contingency
    @test_throws MethodError PMC.PostContingencyNetwork(
        net.topology, net.post_contingency, net.tier2, 1)
end

# (arcs out on K, contingency arcs)
const _FIVE_BUS_TIER2_CASES = [
    ([6], [1]),
    ([6], [2]),
    ([1], [6]),
    ([6], [4]),
    ([2], [3, 5]),
    ([1, 6], [2, 3]),
]

@testset "five-bus two-tier rows match a dense post-outage PTDF" begin
    core = _five_bus_core()
    n_arc = size(core.BA, 2)
    caches = _five_bus_caches()
    nets = [_five_bus_network(k, c) for (k, c) in _FIVE_BUS_TIER2_CASES]
    for m in 1:n_arc
        rows = _five_bus_rows(core, caches, m, nets)
        for ((k, c), row) in zip(_FIVE_BUS_TIER2_CASES, rows)
            out = union(k, c)
            expected = _five_bus_dense_ptdf(core, out, _five_bus_live(core, out))
            @test row ≈ expected[m, :] atol = 1e-12
        end
    end
    cache, woodbury, _ = caches
    @test length(PMC.get_tier2_factor_cache(cache)) == length(nets)
    @test Set(keys(woodbury)) == Set(n.topology for n in nets)
    @test length(PMC.get_tier2_row_caches(cache)) == length(nets)
end

@testset "five-bus base-topology rows are the modification's rows bit for bit" begin
    core = _five_bus_core()
    caches = _five_bus_caches()
    net = _five_bus_network(Int[], [2, 3])
    _, woodbury, row_caches = caches
    for m in 1:size(core.BA, 2)
        row = only(_five_bus_rows(core, caches, m, [net]))
        plan = PMC._modification_row_plan(
            Dict{PMC.NetworkModification, PMC.WoodburyFactors}(), core, m, net.tier2)
        @test isequal(row, only(PMC._rows_from_plans(core, m, (plan,))))
    end
    @test isempty(PMC.get_tier2_factor_cache(caches[1]))
    @test keys(woodbury) == Set([net.tier2])
    @test keys(row_caches) == Set([net.tier2])
end

@testset "the five-bus column cache is bounded and evicts the least recently used column" begin
    core = _five_bus_core()
    n_bus = size(core.BA, 1)
    cache = PMC.Tier2Cache(2 * n_bus * sizeof(Float64))
    B = _five_bus_laplacian(core, Int[])
    X = zeros(n_bus, n_bus)
    X[2:end, 2:end] = inv(B[2:end, 2:end])
    for (p, q) in ((2, 3), (4, 2), (3, 2), (3, 5))
        column = @lock core.solver_lock PMC._cached_column!(cache, core, p, q)
        a, b = minmax(p, q)
        @test column ≈ X[:, a] .- X[:, b] atol = 1e-12
    end
    @test Set(keys(PMC.get_column_cache(cache))) == Set([(2, 3), (3, 5)])
    @test collect(PMC.get_column_cache_order(cache)) == [(2, 3), (3, 5)]
    stats =
        PMC.cache_statistics(cache, Dict{PMC.NetworkModification, PMC.WoodburyFactors}(),
            Dict{PMC.NetworkModification, PMC.RowCache{PMC.RowCacheValue}}())
    @test stats.columns == (; entries = 2, bytes = 2 * n_bus * sizeof(Float64), hits = 1,
        misses = 3)
    tiny = PMC.Tier2Cache(n_bus * sizeof(Float64) - 1)
    @lock core.solver_lock PMC._cached_column!(tiny, core, 2, 3)
    @test isempty(PMC.get_column_cache(tiny))
    @test PMC.get_column_cache_bytes(tiny)[] == 0
end

@testset "retaining keeps the columns and drops tier-2 entries; clearing empties all" begin
    core = _five_bus_core()
    caches = _five_bus_caches()
    cache, woodbury, row_caches = caches
    nets = [_five_bus_network([6], [1]), _five_bus_network([1], [6]),
        _five_bus_network(Int[], [2])]
    _five_bus_rows(core, caches, 3, nets)
    stats = PMC.cache_statistics(cache, woodbury, row_caches)
    @test stats.tier1_factors.entries == 3
    @test stats.tier2_factors.entries == 2
    @test stats.rows.entries == 3
    @test stats.rows.bytes == 3 * size(core.BA, 1) * sizeof(Float64)
    @test stats.columns.entries > 0
    @test stats.columns.misses == stats.columns.entries
    columns = stats.columns
    PMC.retain_modifications!(cache, nets)
    @test isempty(PMC.get_tier2_factor_cache(cache))
    @test isempty(PMC.get_tier2_row_caches(cache))
    @test PMC.cache_statistics(cache, woodbury, row_caches).columns == columns
    # The second build solves no column the first one cached.
    _five_bus_rows(core, caches, 3, nets[1:2])
    again = PMC.cache_statistics(cache, woodbury, row_caches).columns
    @test again.misses == columns.misses
    @test again.hits > columns.hits
    PMC.clear_caches!(cache)
    @test PMC.cache_statistics(cache, woodbury, row_caches).columns ==
          (; entries = 0, bytes = 0, hits = 0, misses = 0)
    @test isempty(PMC.get_tier2_factor_cache(cache))
    @test isempty(PMC.get_column_cache_order(cache))
end
