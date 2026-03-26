@testset "Test _update_bus_maps! -- b2 already reduced" begin
    reverse_bus_search_map = Dict(30 => 20, 31 => 20, 11 => 10)
    bus_reduction_map = Dict(20 => Set([30, 31]), 10 => Set([11]))

    PNM._update_bus_maps!(reverse_bus_search_map, bus_reduction_map, 10, 30)

    @test bus_reduction_map[20] == Set([10, 11, 30, 31])
    @test !haskey(bus_reduction_map, 10)
    @test reverse_bus_search_map[10] == 20
    @test reverse_bus_search_map[11] == 20
    @test reverse_bus_search_map[30] == 20
    @test reverse_bus_search_map[31] == 20
end

@testset "Test _update_bus_maps! -- b2 not reduced" begin
    reverse_bus_search_map = Dict{Int, Int}(51 => 50, 41 => 40, 42 => 40)
    bus_reduction_map = Dict(50 => Set([51]), 40 => Set([41, 42]))

    PNM._update_bus_maps!(reverse_bus_search_map, bus_reduction_map, 40, 50)

    @test bus_reduction_map[50] == Set([40, 41, 42, 51])
    @test !haskey(bus_reduction_map, 40)
    @test reverse_bus_search_map[40] == 50
    @test reverse_bus_search_map[41] == 50
    @test reverse_bus_search_map[42] == 50
    @test reverse_bus_search_map[51] == 50
end

