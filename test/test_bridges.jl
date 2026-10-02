function _components_without(n::Int, edges::Vector{Tuple{Int, Int}}, skip::Int)
    uf = collect(1:n)
    for (k, (u, v)) in enumerate(edges)
        k == skip && continue
        PNM.union_sets!(uf, u, v)
    end
    return [PNM.get_representative(uf, i) for i in 1:n]
end

@testset "find_bridges agrees with removing each edge" begin
    rng = Random.Xoshiro(17)
    for trial in 1:200
        n = rand(rng, 1:30)
        m = rand(rng, 0:(2 * n))
        edges = [(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:m]
        # Repeat some edges so parallel pairs are exercised.
        for _ in 1:rand(rng, 0:3)
            isempty(edges) || push!(edges, rand(rng, edges))
        end
        b = PNM.find_bridges(n, edges)
        full = _components_without(n, edges, 0)
        for (e, (u, v)) in enumerate(edges)
            label = _components_without(n, edges, e)
            splits = label[u] != label[v]
            @test PNM.is_bridge(b, e) == splits
            if splits
                far = Set(PNM.bridge_far_side(b, e))
                c = b.far_end[e]
                @test far == Set(i for i in 1:n if label[i] == label[c])
                @test all(full[i] == full[u] for i in far)
            else
                @test_throws ErrorException PNM.bridge_far_side(b, e)
            end
        end
    end
end

@testset "find_bridges: parallel edges and a deep chain" begin
    b = PNM.find_bridges(3, [(1, 2), (2, 1), (2, 3)])
    @test b.is_bridge == [false, false, true]
    @test collect(PNM.bridge_far_side(b, 3)) == [3]
    n = 200_000
    chain = [(i, i + 1) for i in 1:(n - 1)]
    b = PNM.find_bridges(n, chain)
    @test all(b.is_bridge)
    @test length(PNM.bridge_far_side(b, 1)) == n - 1
end
