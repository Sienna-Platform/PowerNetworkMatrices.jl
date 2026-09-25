@testset "Test parallel branch naming" begin
    # An aggregate is named for its arc, not its members. The `double_circuit` suffix is
    # retained -- it is what marks the row as a total across the members -- and only the
    # stem changed: the longest common prefix of the member names was not injective and
    # moved when membership did.
    l1 = Line(nothing)
    set_name!(l1, "A33-1")
    l2 = Line(nothing)
    set_name!(l2, "A33-2")
    bp = PNM.BranchesParallel{PSY.Line}(PSY.Line[l1, l2], (7, 9), PNM.EMPTY_TWO_PORT, false)
    @test PNM.get_name(bp) == "7_9_double_circuit"

    # Member names no longer participate, so the two cases the old scheme distinguished --
    # shared prefix vs none -- now give the same name for the same arc.
    set_name!(l1, "B1")
    set_name!(l2, "C2")
    bp = PNM.BranchesParallel{PSY.Line}(PSY.Line[l1, l2], (7, 9), PNM.EMPTY_TWO_PORT, false)
    @test PNM.get_name(bp) == "7_9_double_circuit"

    # Distinct arcs give distinct names, which the prefix scheme could not guarantee:
    # `La`/`Lb` and `Lc`/`Ld` both collapsed to `Ldouble_circuit`.
    other =
        PNM.BranchesParallel{PSY.Line}(PSY.Line[l1, l2], (3, 4), PNM.EMPTY_TWO_PORT, false)
    @test PNM.get_name(other) == "3_4_double_circuit"
    @test PNM.get_name(other) != PNM.get_name(bp)

    chain = PNM.BranchesSeries((11, 12))
    @test PNM.get_name(chain) == "series_11_12"
end
