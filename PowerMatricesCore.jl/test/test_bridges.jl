import Random
import SparseArrays

function _components_without(n::Int, edges::Vector{Tuple{Int, Int}}, skip::Int)
    uf = collect(1:n)
    for (k, (u, v)) in enumerate(edges)
        k == skip && continue
        PMC.union_sets!(uf, u, v)
    end
    return [PMC.get_representative(uf, i) for i in 1:n]
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
        b = PMC.find_bridges(n, edges)
        full = _components_without(n, edges, 0)
        for (e, (u, v)) in enumerate(edges)
            label = _components_without(n, edges, e)
            splits = label[u] != label[v]
            @test PMC.is_bridge(b, e) == splits
            if splits
                far = Set(PMC.bridge_far_side(b, e))
                c = b.far_end[e]
                @test far == Set(i for i in 1:n if label[i] == label[c])
                @test all(full[i] == full[u] for i in far)
            else
                @test_throws ErrorException PMC.bridge_far_side(b, e)
            end
        end
    end
end

@testset "find_bridges: parallel edges and a deep chain" begin
    b = PMC.find_bridges(3, [(1, 2), (2, 1), (2, 3)])
    @test b.is_bridge == [false, false, true]
    @test collect(PMC.bridge_far_side(b, 3)) == [3]
    n = 200_000
    chain = [(i, i + 1) for i in 1:(n - 1)]
    b = PMC.find_bridges(n, chain)
    @test all(b.is_bridge)
    @test length(PMC.bridge_far_side(b, 1)) == n - 1
end

function _first_seen(labels::Vector{Int})
    d = Dict{Int, Int}()
    return [get!(d, l, length(d) + 1) for l in labels]
end

@testset "BridgeLabels partitions like the union-find" begin
    rng = Random.Xoshiro(29)
    for trial in 1:100
        n = rand(rng, 2:25)
        edges = [(rand(rng, 1:n), rand(rng, 1:n)) for _ in 1:rand(rng, 1:(2 * n))]
        edges = [e for e in edges if e[1] != e[2]]
        isempty(edges) && continue
        m = length(edges)
        rows = vcat(first.(edges), last.(edges))
        cols = vcat(1:m, 1:m)
        BA = SparseArrays.sparse(rows, cols, vcat(ones(m), -ones(m)), n, m)
        sus = ones(m)
        bl = PMC.BridgeLabels(BA)
        cases =
            Tuple{Vararg{PMC.ArcModification}}[(PMC.ArcModification(e, -1.0),) for e in 1:m]
        push!(cases, (PMC.ArcModification(1, -0.5),))
        m > 1 && push!(cases, (PMC.ArcModification(1, -1.0), PMC.ArcModification(m, -1.0)))
        for mods in cases
            uf = PMC._post_contingency_bus_labels(BA, sus, mods, n)
            @test _first_seen(bl(BA, sus, mods, n)) == _first_seen(uf)
        end
    end
end
