#=
Aggregates that contain aggregates.

Two shapes exist and are built deliberately by `get_degree2_reduction`:

  - a parallel group as a chain segment  -- `BranchesSeries` holding a `BranchesParallel`
  - sibling chains in parallel           -- `BranchesParallel{BranchesSeries}`

`test_2d_reduction.jl` covers the reduction producing them. These tests cover what the
catalog then does with them: resolving an entry to component types, and folding a filter
over one. Both have to descend, and both got the nesting wrong by unwrapping a fixed number
of levels -- which files a group of chains under `BranchesSeries`, a PNM wrapper type that
no `name_to_arc_map` consumer ever asks for, taking the arc out of the index entirely.
=#

# Ring core over buses 1-4 plus two three-segment chains between buses 1 and 3, so the
# composite arc carries a `BranchesParallel{BranchesSeries}`. Lines are named `L_<from>_<to>`.
function _nested_aggregate_catalog()
    sys = build_two_parallel_degree_two_chains()
    ybus = Ybus(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
    return sys, get_network_reduction_data(ybus), PNM.get_branch_catalog(ybus)
end

# The composite arc's orientation depends on interior bus numbering, so match either way.
_composite_arc(nrd) = only(
    k for (k, v) in PNM.get_parallel_branch_map(nrd)
    if all(m isa PNM.BranchesSeries for m in v)
)

@testset "Nested aggregates resolve to leaf components and leaf types" begin
    _, nrd, _ = _nested_aggregate_catalog()
    group = PNM.get_parallel_branch_map(nrd)[_composite_arc(nrd)]

    # The walk descends group -> chain -> line. Unwrapping one level yields the two
    # `BranchesSeries` members instead of any component.
    leaves = PNM._get_segment_components(group)
    @test all(l isa PSY.Line for l in leaves)
    @test Set(PSY.get_name(l) for l in leaves) ==
          Set(["L_1_10", "L_10_11", "L_11_3", "L_1_20", "L_20_21", "L_21_3"])

    # Bucket keys are PSY component types, never a PNM aggregate wrapper.
    @test PNM._get_concrete_types(group) == [PSY.Line]
    for chain in group
        @test PNM._get_concrete_types(chain) == [PSY.Line]
    end
end

@testset "The composite arc is reachable under its members' component type" begin
    _, nrd, catalog = _nested_aggregate_catalog()
    composite = _composite_arc(nrd)

    line_entries = PNM.get_name_to_arc_map(catalog, PSY.Line)
    arcs = Set(arc for (arc, _) in values(line_entries))

    # `PSY.Line` is the only type any DeviceModel would ask for, so the composite arc has to
    # be reachable there...
    @test composite in arcs
    # ...and must not be filed under a PNM wrapper type instead.
    @test !haskey(PNM.get_name_to_arc_maps(catalog), PNM.BranchesSeries)

    # Every absorbed member redirects to the entry carrying its flow.
    redirects = PNM.get_component_to_reduction_name_map(catalog, PSY.Line)
    entry_names = Set(keys(line_entries))
    for name in ("L_1_10", "L_10_11", "L_11_3", "L_1_20", "L_20_21", "L_21_3")
        @test haskey(redirects, name)
        @test redirects[name] in entry_names
    end
end

@testset "Catalog closure holds over nested aggregates" begin
    _, nrd, catalog = _nested_aggregate_catalog()

    # A degree-two reduction folds; it does not absorb. So every arc must still be reachable
    # by component type.
    @test PNM._validate_catalog_closure(nrd, PNM.get_name_to_arc_maps(catalog)) === nothing

    # Same check reached through the opt-in construction path.
    @test PNM.BranchCatalog(nrd; validate = true) isa PNM.BranchCatalog

    # Refused on a filtered catalog: the invariant does not hold there by design, so
    # answering "valid" would report a guarantee the check cannot give.
    @test_throws ArgumentError PNM.BranchCatalog(nrd, (T, c) -> true; validate = true)
end

@testset "Filters see PSY components, never aggregate wrappers" begin
    _, nrd, _ = _nested_aggregate_catalog()

    # A filter is written against PSY components. Handing it a `BranchesSeries` -- which has
    # no `PSY.get_arc` -- throws inside the caller's own filter. This is the shape of a real
    # template filter.
    seen = DataType[]
    by_voltage = function (T, component)
        push!(seen, T)
        return PSY.get_base_voltage(PSY.get_from(PSY.get_arc(component))) >= 100.0
    end

    catalog = PNM.BranchCatalog(nrd, by_voltage)
    @test !isempty(seen)
    @test all(T -> T <: PSY.ACTransmission, seen)
    @test !any(T -> T <: PNM.AbstractReductionAggregate, seen)

    # Every line here is 230 kV, so the filter keeps the composite arc.
    arcs = Set(arc for (arc, _) in values(PNM.get_name_to_arc_map(catalog, PSY.Line)))
    @test _composite_arc(nrd) in arcs

    # A filter excluding everything drops the arc. That is a filter decision, not a lost
    # entry, so closure does not apply to a filtered catalog.
    empty_catalog = PNM.BranchCatalog(nrd, (T, c) -> false)
    @test isempty(PNM.get_name_to_arc_map(empty_catalog, PSY.Line))
end

@testset "Nested filters fold per level, not over flattened leaves" begin
    sys, nrd, _ = _nested_aggregate_catalog()
    lines = Dict(PSY.get_name(l) => l for l in PSY.get_components(PSY.Line, sys))
    keep(names) = (T, c) -> PSY.get_name(c) in names

    # --- parallel inside series ---------------------------------------------------------
    # Hand-built rather than reduced from a system: this shape needs a chain segment carrying
    # two branches on one bus pair, which no fixture here produces.
    group = PNM.BranchesParallel(PSY.Line[lines["L_1_10"], lines["L_1_20"]])
    chain = PNM.BranchesSeries((1, 3))
    PNM.add_branch!(chain, group, :FromTo)
    PNM.add_branch!(chain, lines["L_11_3"], :FromTo)

    # `any` within the parallel segment, `all` across segments: one qualifying member still
    # carries that segment, so the chain qualifies. Folding `all` over flattened leaves
    # would reject this, because `L_1_20` fails.
    @test PNM._entry_matches(chain, keep(Set(["L_1_10", "L_11_3"])))

    # No member of the parallel segment qualifies, so that segment is not carried.
    @test !PNM._entry_matches(chain, keep(Set(["L_11_3"])))

    # A failing plain segment fails the chain however well the group does.
    @test !PNM._entry_matches(chain, keep(Set(["L_1_10", "L_1_20"])))

    # --- series inside parallel ---------------------------------------------------------
    real_group = PNM.get_parallel_branch_map(nrd)[_composite_arc(nrd)]

    # `all` within each chain, `any` across them. One complete chain carries the arc...
    @test PNM._entry_matches(real_group, keep(Set(["L_1_10", "L_10_11", "L_11_3"])))

    # ...but one leaf from each chain does not: neither path is complete. Folding `any` over
    # flattened leaves would wrongly accept this.
    @test !PNM._entry_matches(real_group, keep(Set(["L_1_10", "L_1_20"])))
end
