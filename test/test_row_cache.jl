@testset "Test RowCache functions" begin

    # dummy rows
    key = 1
    row = ones((24,))

    # define cache struct
    cache = PNM.RowCache(
        5 * length(row) * sizeof(Float64),
        Set([1, 2, 3]),
        length(row) * sizeof(Float64),
        100,
    )

    # test: Base.isempty
    @test isempty(cache) == true

    # test: Base.haskey and Base.getindex
    cache[key] = row
    @test haskey(cache, 1) == true

    # test: Base.length (number of rows stored)
    @test length(cache) == 1

    # test: purge_one!
    PNM.purge_one!(cache)
    @test length(cache) == 1
    @test haskey(cache, 5) == false

    # test: Base.setindex, check_cache_size!
    for i in 1:5
        cache[i] = row
    end
    cache[6] = row
    @test length(cache) == 5

    # test: Base.empty!
    empty!(cache)
    @test length(cache) == 0
end

@testset "RowCache: pinned rows count against capacity" begin
    row_size = 8
    cache = PNM.RowCache(10 * row_size, Set{Int}(), row_size, 100)
    @test cache.max_num_keys == 10
    row = [1.0]

    # The constructor reserves one evictable slot (`length(persistent_rows) + 1`),
    # so pinning is capped at `max_num_keys - 1`.
    for k in 1:9
        PNM.set_persistent_row!(cache, k, row)
    end
    @test length(cache) == 9
    @test_throws ErrorException PNM.set_persistent_row!(cache, 10, row)

    # Pinning the last evictable row would leave `check_cache_size!` nothing to
    # purge, so it is rejected too.
    cache[10] = row
    @test_throws ErrorException PNM.pin_row!(cache, 10)

    # Lazy inserts stay within capacity by evicting the one unpinned row.
    cache[11] = row
    @test length(cache) == cache.max_num_keys
end

@testset "RowCache: a row written around the LRU order is reported as such" begin
    row_size = 8
    cache = PNM.RowCache(3 * row_size, Set{Int}(), row_size, 100)
    PNM.set_persistent_row!(cache, 1, [1.0])
    PNM.set_persistent_row!(cache, 2, [1.0])
    cache.temp_cache[99] = [1.0]
    @test_throws r"RowCache API" setindex!(cache, [1.0], 3)
end

@testset "RowCache: empty! clears the persistent keys" begin
    row_size = 8
    cache = PNM.RowCache(10 * row_size, Set([1, 2]), row_size, 100)
    PNM.set_persistent_row!(cache, 3, [1.0])
    empty!(cache)
    @test isempty(cache.persistent_cache_keys)
    @test isempty(cache.temp_cache)
    @test isempty(cache.access_order)
end

@testset "_cached_row returns stored storage that survives eviction" begin
    row_bytes = 3 * sizeof(Float64)
    cache = PNM.RowCache(2 * row_bytes, Set{Int}(), row_bytes, 100)
    lk = ReentrantLock()
    cutoff = PNM.AbsoluteCutoff(eps(Float64))
    r1 = PNM._cached_row(() -> [1.0, 2.0, 3.0], cache, lk, 1, cutoff)
    @test r1 === cache[1]
    @test PNM._cached_row(() -> error("must not recompute a hit"), cache, lk, 1, cutoff) ===
          r1
    for k in 2:4
        PNM._cached_row(() -> fill(Float64(k), 3), cache, lk, k, cutoff)
    end
    @test !haskey(cache, 1)
    @test r1 == [1.0, 2.0, 3.0]
end

@testset "A RowCache allocates storage only for the rows it holds" begin
    row_bytes = 73 * sizeof(Float64)
    budget = 100 * 2^20
    n_rows = 10^6
    @test Base.summarysize(PNM.RowCache(budget, Set{Int}(), row_bytes, n_rows)) <
          10 * row_bytes
    cache = PNM.RowCache(budget, Set{Int}(), row_bytes, n_rows)
    @test cache.max_num_keys == floor(Int, budget / row_bytes)
    for k in 1:(cache.max_num_keys)
        cache[k] = fill(Float64(k), 73)
    end
    @test length(cache) == cache.max_num_keys
    cache[cache.max_num_keys + 1] = zeros(73)
    @test length(cache) == cache.max_num_keys
    @test !haskey(cache, 1)
end

@testset "A RowCache holds at most one key per matrix row" begin
    row_bytes = 5 * sizeof(Float64)
    budget = 100 * 2^20
    cache = PNM.RowCache(budget, Set{Int}(), row_bytes, 7)
    @test cache.max_num_keys == 8
    for k in 1:7
        cache[k] = fill(Float64(k), 5)
    end
    @test length(cache) == 7
    @test all(haskey(cache, k) for k in 1:7)

    # `populate_cache` pins every row one at a time.
    pinned = PNM.RowCache(budget, Set{Int}(), row_bytes, 7)
    for k in 1:7
        PNM.set_persistent_row!(pinned, k, fill(Float64(k), 5))
    end
    @test length(pinned) == 7
    @test pinned.persistent_cache_keys == Set(1:7)

    # A byte budget smaller than the matrix still bounds the cache.
    @test PNM.RowCache(3 * row_bytes, Set{Int}(), row_bytes, 7).max_num_keys == 3

    # Pinning every row keeps one evictable slot.
    @test PNM.RowCache(budget, Set(1:7), row_bytes, 7).max_num_keys == 8
end
