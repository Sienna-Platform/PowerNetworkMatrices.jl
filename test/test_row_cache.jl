@testset "Test RowCache functions" begin

    # dummy rows
    key = 1
    row = ones((24,))

    # define cache struct
    cache = PNM.RowCache(
        5 * length(row) * sizeof(Float64),
        Set([1, 2, 3]),
        length(row) * sizeof(Float64),
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
    cache = PNM.RowCache(10 * row_size, Set{Int}(), row_size)
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

@testset "RowCache: empty! clears the persistent keys" begin
    row_size = 8
    cache = PNM.RowCache(10 * row_size, Set([1, 2]), row_size)
    PNM.set_persistent_row!(cache, 3, [1.0])
    empty!(cache)
    @test isempty(cache.persistent_cache_keys)
    @test isempty(cache.temp_cache)
    @test isempty(cache.access_order)
end
