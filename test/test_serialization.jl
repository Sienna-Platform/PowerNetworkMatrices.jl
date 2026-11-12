function _by_winding(f, w::Int, x, id::Int, tail...)
    if iszero(w)
        return f(x, id, tail...)
    end
    return f(x, id, PMC.Winding(w), tail...)
end

function _roundtrip_ptdf(sys; kwargs...)
    vptdf = VirtualPTDF(sys; kwargs...)
    path = joinpath(mktempdir(), "ptdf.h5")
    to_hdf5(vptdf, path)
    return vptdf, from_hdf5(VirtualPTDF, path), path
end

function _modf_roundtrip(vmodf)
    path = joinpath(mktempdir(), "modf.h5")
    to_hdf5(vmodf, path)
    return from_hdf5(VirtualMODF, path)
end

function _outage_roundtrip(vmodf, sys)
    path = joinpath(mktempdir(), "outages.h5")
    to_hdf5(vmodf, path, sys)
    return from_hdf5(VirtualMODF, path), from_hdf5(BranchOutageTable, path)
end

function _test_ptdf_rows_equal(a, b, arcs)
    for arc in arcs
        @test isapprox(
            collect(PNM.get_ptdf_row(a, arc)),
            collect(PNM.get_ptdf_row(b, arc));
            atol = 1e-12,
        )
    end
    return
end

function _test_same_core(a, b)
    @test PNM.get_tol(a) == PNM.get_tol(b)
    @test PNM.get_cutoff(PNM.get_core(a)) == PNM.get_cutoff(PNM.get_core(b))
    @test PNM.get_tol_spec(PNM.get_core(a)) == PNM.get_tol_spec(PNM.get_core(b))
    @test PNM.get_bus_axis(a) == PNM.get_bus_axis(b)
    @test PNM.get_arc_axis(a) == PNM.get_arc_axis(b)
    @test PNM.get_ref_bus_position(a) == PNM.get_ref_bus_position(b)
    @test PNM.get_system_uuid(a) == PNM.get_system_uuid(b)
    @test PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(a))) ==
          PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(b)))
    @test PNM.get_core(a).arc_live_members == PNM.get_core(b).arc_live_members
    @test PNM.get_core(a).BA == PNM.get_core(b).BA
    @test PNM.get_core(a).A == PNM.get_core(b).A
    return
end

# Every monitored row of `expected` on `vmodf` equals the row of `actual` on `restored`.
function _test_modf_rows_equal(vmodf, restored, expected, actual = expected)
    @test actual.is_islanding == expected.is_islanding
    for monitored in 1:length(PNM.get_arc_axis(vmodf))
        @test isapprox(
            collect(vmodf[monitored, expected]),
            collect(restored[monitored, actual]);
            atol = 1e-9,
        )
    end
    return
end

function _test_arc_rows_equal(expected, actual)
    @test length(actual.arc_modifications) == length(expected.arc_modifications)
    for (a, e) in zip(actual.arc_modifications, expected.arc_modifications)
        @test a.arc_index == e.arc_index
        @test a.delta_b ≈ e.delta_b
        @test isapprox(a.delta_shift_injection, e.delta_shift_injection; atol = 1e-12)
    end
    return
end

function _test_post_outage_rows_equal(vmodf, restored, expected, actual)
    @test actual.is_islanding == expected.is_islanding
    for monitored in 1:length(PNM.get_arc_axis(vmodf))
        @test isapprox(
            collect(vmodf[monitored, expected]),
            collect(restored[monitored, actual]);
            atol = 1e-10,
        )
    end
    return
end

function _add_outage!(sys, components...)
    outage = PSY.FixedForcedOutage(; outage_status = 1.0)
    for component in components
        PSY.add_supplemental_attribute!(sys, component, outage)
    end
    return outage
end

_store(g, name, value) = (g[name] = value)
_store(g, name, value::Vector{String}) = PNM._write_strings!(g, name, value)

_flows_by_leaf_key(fref) = Dict(
    (PNM._branch_map_id(l), max(PNM._branch_map_kind(l) - 5, 0)) => f for (l, f) in fref
)

# Branch flows from the file, the leaf-level reference flows, and the arc flows.
function _leaf_flow_oracle(sys, vptdf, path, table)
    arc_ax = PNM.get_arc_axis(vptdf)
    bus_ax = PNM.get_bus_axis(vptdf)
    restored = from_hdf5(VirtualPTDF, path)
    shift = HDF5.h5open(f -> read(f["shift_injection"]), path, "r")
    rng = MersenneTwister(7)
    inj = randn(rng, length(bus_ax))
    inj .-= sum(inj) / length(inj)
    bix = Dict(b => i for (i, b) in enumerate(bus_ax))
    q = copy(inj)
    for (k, a) in enumerate(arc_ax)
        q[bix[a[1]]] += shift[k]
        q[bix[a[2]]] -= shift[k]
    end
    arc_flows = Float64[
        sum(Vector(restored[a, :]) .* q) - shift[k] for (k, a) in enumerate(arc_ax)
    ]
    fref = leaf_dc_flows(sys, Dict{Int, Float64}(b => inj[i] for (b, i) in bix))
    return branch_flows(table, arc_flows), _flows_by_leaf_key(fref), arc_flows
end

function _corrupted_copy(good, group, edits::Pair...)
    path = joinpath(mktempdir(), "bad.h5")
    cp(good, path)
    for (name, value) in edits
        _write_bad_dataset(path, group, string(name), value)
    end
    return path
end

function _write_bad_dataset(path, group, name, value)
    HDF5.h5open(path, "r+") do f
        g = f
        if !isempty(group)
            g = f[group]
        end
        HDF5.delete_object(g, name)
        _store(g, name, value)
    end
    return
end

const RTS_REDUCTIONS = NetworkReduction[RadialReduction(), DegreeTwoReduction()]
const RTS_SYS = PSB.build_system(PSB.PSITestSystems, "test_RTS_GMLC_sys")
const RTS_VMODF = VirtualMODF(RTS_SYS; network_reductions = RTS_REDUCTIONS)
const RTS_RESTORED, RTS_TABLE = _outage_roundtrip(RTS_VMODF, RTS_SYS)
const RTS_NR = PNM.get_network_reduction_data(RTS_VMODF)

@testset "factor file" begin
    @testset "all-relations fixture" begin
        sys = build_branch_map_system()
        @test length(PSY.get_components(PSY.ACBus, sys)) == 15
        reds = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vptdf = VirtualPTDF(sys; network_reductions = reds, tol = 0.0)
        @test length(PNM.get_bus_axis(vptdf)) == 6
        @test length(PNM.get_arc_axis(vptdf)) == 14
        _, restored, _ = _roundtrip_ptdf(
            sys; network_reductions = reds, tol = 0.0)
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        p = Dict{Int, Float64}(
            b => 0.1 * b for b in PSY.get_number.(PSY.get_components(PSY.ACBus, sys)))
        flows = leaf_dc_flows(sys, p)
        @test length(flows) == 29
        @test all(isfinite, values(flows))
        stub = PSY.get_component(PSY.Line, sys, "L_40_41")
        @test isapprox(flows[stub], -p[41]; atol = 1e-12)
        feeder = PSY.get_component(PSY.Line, sys, "L_4_40")
        @test isapprox(flows[feeder], -(p[40] + p[41]); atol = 1e-12)
    end

    @testset "_check_branch_map" begin
        # Row 1 leaf on arc 1, row 2 parallel group on arc 2, rows 3 and 4 its members,
        # row 5 a Ward equivalent on arc 1.
        valid() = (
            kind = [1, 9, 1, 6, 11],
            id = [7, 0, 8, 9, 0],
            parent = [-1, -2, 2, 2, -1],
            sign = [1, 1, 1, -1, 1],
            b = [10.0, 0.0, 5.0, -4.0, 2.0],
            alpha = [0.1, 0.0, 0.0, 0.2, 0.0],
        )
        check(t, n = 2) = PNM._check_branch_map(
            BranchArcTable(t.kind, t.id, t.parent, t.sign, t.b, t.alpha, n))
        function bad(field, row, value)
            t = valid()
            v = copy(getfield(t, field))
            v[row] = value
            return merge(t, NamedTuple{(field,)}((v,)))
        end
        @test isnothing(check(valid()))
        t = valid()
        @test_throws "unequal length" check(merge(t, (b = t.b[1:4],)))
        @test_throws "kind is not in 1:11" check(bad(:kind, 2, 12))
        @test_throws "kind is not in 1:11" check(bad(:kind, 1, 0))
        @test_throws "parent is 0" check(bad(:parent, 3, 0))
        @test_throws "is not an arc of the" check(bad(:parent, 1, -3))
        @test_throws "is not below the row" check(bad(:parent, 3, 3))
        @test_throws "not an aggregate" check(bad(:parent, 3, 1))
        @test_throws "is not below the row" check(bad(:parent, 2, 2))
        @test_throws "must be a top node" check(bad(:parent, 5, 2))
        @test_throws "sign is not 1 or -1" check(bad(:sign, 3, 0))
        @test_throws "id is not positive" check(bad(:id, 1, 0))
        @test isnothing(check(bad(:b, 1, 0.0)))
        @test_throws "b is NaN; expected 0 outside a series chain" check(bad(:b, 1, NaN))
        @test_throws "b is 1.0e308; expected 0 outside" check(bad(:b, 1, 1e308))
        @test_throws "b is 1.0e-320; expected 0 outside" check(bad(:b, 1, 1e-320))
        # A zero inside a series chain would make the chain sum infinite.
        chain = bad(:kind, 2, 10)
        @test isnothing(check(chain))
        @test_throws "b is 0.0; expected 0 outside a series chain" check(
            merge(chain, (b = [10.0, 0.0, 0.0, -4.0, 2.0],)),
        )
        @test_throws "alpha is Inf; expected a finite value" check(bad(:alpha, 1, Inf))
        @test_throws "alpha is NaN; expected a finite value" check(bad(:alpha, 1, NaN))
        @test_throws "id is not 0" check(bad(:id, 5, 3))
        @test isnothing(check(bad(:b, 5, 0.0)))
        @test_throws "b is 1.0e308; expected a finite magnitude" check(
            bad(:b, 5, 1e308),
        )
        @test isnothing(check(bad(:b, 5, 7.14e-13)))
        @test_throws "alpha is not 0" check(bad(:alpha, 5, 0.1))
        @test_throws "id is not 0" check(bad(:id, 2, 3))
        @test_throws "b is not 0" check(bad(:b, 2, 1.0))
        @test_throws "alpha is not 0" check(bad(:alpha, 2, 0.1))
        @test_throws "id and winding (7, 0) repeat" check(bad(:id, 3, 7))
        orphan = merge(valid(), (parent = [-1, -2, -1, -1, -1],))
        @test_throws "no child" check(orphan)
        @test_throws "no top node" check(valid(), 3)
    end

    @testset "branch_map writer" begin
        _add_leaf_key!(keys, br::PSY.ACTransmission) =
            (keys[(IS.get_id(br), 0)] = nameof(typeof(br)))
        _add_leaf_key!(keys, ::PSY.ThreeWindingTransformer) = nothing
        function leaf_keys(sys)
            keys = Dict{Tuple{Int, Int}, Symbol}()
            for br in PSY.get_components(PSY.ACTransmission, sys)
                _add_leaf_key!(keys, br)
            end
            for t in PSY.get_components(PSY.ThreeWindingTransformer, sys)
                for w in 1:3
                    keys[(IS.get_id(t), w)] = :ThreeWindingTransformer
                end
            end
            return keys
        end
        kind_type = Dict(
            1 => :Line, 2 => :TwoWindingTransformer, 3 => :DiscreteControlledACBranch,
            4 => :GenericArcImpedance, 5 => :SeriesCompensator,
            6 => :ThreeWindingTransformer, 7 => :ThreeWindingTransformer,
            8 => :ThreeWindingTransformer,
        )
        cases = [
            (
                build_branch_map_system(),
                PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()],
                1,
            ),
            (
                PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions"),
                PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()],
                0,
            ),
        ]
        for (sys, reds, n_stubs) in cases
            vptdf = VirtualPTDF(sys; network_reductions = reds, tol = 0.0)
            path = joinpath(mktempdir(), "ptdf.h5")
            to_hdf5(vptdf, path)
            g = HDF5.h5open(
                f -> Dict(
                    k => read(f["branch_map"][k]) for k in
                    ("kind", "id", "parent", "sign", "b", "alpha")
                ), path, "r")
            n_arc = length(PNM.get_arc_axis(vptdf))
            @test all(length(v) == length(g["kind"]) for v in values(g))
            @test HDF5.h5open(f -> read(f["arc_from"]), path, "r") ==
                  first.(PNM.get_arc_axis(vptdf))
            @test sort(unique(-g["parent"][g["parent"] .< 0])) == collect(1:n_arc)
            known = leaf_keys(sys)
            leaf = findall(k -> k <= 8, g["kind"])
            seen = [(g["id"][r], max(g["kind"][r] - 5, 0)) for r in leaf]
            @test allunique(seen)
            for (r, key) in zip(leaf, seen)
                @test known[key] == kind_type[g["kind"][r]]
            end
            catalog = PNM.get_branch_catalog(PNM.get_core(vptdf))
            n_leaves = sum(
                length(PNM.get_arc_leaves(catalog, arc)) for arc in PNM.get_arc_axis(vptdf)
            )
            @test length(leaf) == n_leaves
            @test length(leaf) < length(known)
            stubs = [
                br for br in PSY.get_components(PSY.Line, sys) if
                _arc_bus_pair(PSY.get_arc(br)) == (40, 41)
            ]
            @test length(stubs) == n_stubs
            for br in stubs
                @test !(IS.get_id(br) in g["id"])
            end
        end
    end

    @testset "branch_map reader" begin
        sys = build_branch_map_system()
        vptdf = VirtualPTDF(
            sys;
            network_reductions = PNM.NetworkReduction[
                RadialReduction(),
                DegreeTwoReduction(),
            ],
            tol = 0.0,
        )
        good = joinpath(mktempdir(), "good.h5")
        to_hdf5(vptdf, good)
        cols = (:kind, :id, :parent, :sign, :b, :alpha)
        written = HDF5.h5open(
            f -> Dict(c => read(f["branch_map"][string(c)]) for c in cols), good, "r")
        table = from_hdf5(BranchArcTable, good)
        @test PNM.get_branch_kind(table) == written[:kind]
        @test PNM.get_branch_id(table) == written[:id]
        @test PNM.get_branch_parent(table) == written[:parent]
        @test PNM.get_branch_sign(table) == written[:sign]
        @test PNM.get_branch_susceptance(table) == written[:b]
        @test PNM.get_branch_alpha(table) == written[:alpha]
        @test PNM.get_branch_arc_count(table) == length(PNM.get_arc_axis(vptdf))

        at(name, row, value) = name => setindex!(copy(written[name]), value, row)
        corrupt(edits...) = _corrupted_copy(good, "branch_map", edits...)
        last_row = length(written[:kind])
        leaf = findfirst(==(1), written[:kind])
        top = findfirst(<(0), written[:parent])
        @test_throws "unequal length" from_hdf5(
            BranchArcTable, corrupt(:b => written[:b][1:(end - 1)]))
        @test_throws "parent is 0" from_hdf5(BranchArcTable, corrupt(at(:parent, top, 0)))

        # Rules of _check_branch_map that need several columns edited at once.
        corrupt_many(row, edits...) = corrupt((at(n, row, v) for (n, v) in edits)...)
        leaf_rows = findall(<=(8), written[:kind])
        top_leaf = first(r for r in leaf_rows if written[:parent][r] < 0)
        member = first(r for r in leaf_rows if written[:parent][r] > 0)
        second_leaf = first(r for r in leaf_rows if r != leaf && written[:kind][r] <= 5)
        @test written[:kind][leaf] <= 5
        @test_throws "not an aggregate" from_hdf5(
            BranchArcTable, corrupt(at(:parent, last_row, leaf)))
        @test_throws "must be a top node" from_hdf5(
            BranchArcTable, corrupt_many(member, :kind => 11, :id => 0, :alpha => 0.0))
        @test_throws "id and winding" from_hdf5(
            BranchArcTable, corrupt(at(:id, second_leaf, written[:id][leaf])))
        @test_throws "id is not 0" from_hdf5(
            BranchArcTable, corrupt(at(:kind, top_leaf, 11)))
        @test_throws "(kind 11): b is NaN; expected a finite magnitude" from_hdf5(
            BranchArcTable,
            corrupt_many(top_leaf, :kind => 11, :id => 0, :alpha => 0.0, :b => NaN))
        @test_throws "alpha is not 0" from_hdf5(
            BranchArcTable,
            corrupt_many(top_leaf, :kind => 11, :id => 0, :alpha => 0.5))
        @test_throws "no child" from_hdf5(
            BranchArcTable,
            corrupt_many(
                last_row, :kind => 9, :id => 0, :b => 0.0, :alpha => 0.0,
                :parent => -1))

        corrupt_root(name, value) = _corrupted_copy(good, "", name => value)
        arc_from = HDF5.h5open(f -> read(f["arc_from"]), good, "r")
        arc_to = HDF5.h5open(f -> read(f["arc_to"]), good, "r")
        @test_throws "table arcs has columns of unequal length" from_hdf5(
            BranchArcTable, corrupt_root("arc_to", arc_to[1:(end - 1)]))
        extra_arc = corrupt_root("arc_to", [arc_to; 1])
        _write_bad_dataset(extra_arc, "", "arc_from", [arc_from; 2])
        @test_throws "no top node" from_hdf5(BranchArcTable, extra_arc)

        missing_group = joinpath(mktempdir(), "nogroup.h5")
        cp(good, missing_group)
        HDF5.h5open(f -> HDF5.delete_object(f, "branch_map"), missing_group, "r+")
        @test_throws "Factor file has no entry /branch_map" from_hdf5(
            BranchArcTable,
            missing_group,
        )

        @test_throws "HDF5 class" from_hdf5(
            BranchArcTable, corrupt(:kind => Float64.(written[:kind])))
        @test_throws "HDF5 class" from_hdf5(
            BranchArcTable, corrupt(:alpha => Int.(zeros(last_row))))

        bad_version = joinpath(mktempdir(), "version.h5")
        cp(good, bad_version)
        HDF5.h5open(bad_version, "r+") do f
            HDF5.delete_attribute(f, "format_version")
            HDF5.write_attribute(f, "format_version", PMC.FACTOR_FILE_FORMAT_VERSION + 1)
        end
        @test_throws "reads version $(PMC.FACTOR_FILE_FORMAT_VERSION)" from_hdf5(
            BranchArcTable,
            bad_version,
        )
    end

    @testset "branch_flows" begin
        cases = [
            build_branch_map_system(),
            PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions"),
        ]
        reds = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        for sys in cases
            vptdf = VirtualPTDF(sys; network_reductions = reds, tol = 0.0)
            arc_ax = PNM.get_arc_axis(vptdf)
            path = joinpath(mktempdir(), "ptdf.h5")
            to_hdf5(vptdf, path)
            table = from_hdf5(BranchArcTable, path)
            flows, by_key, arc_flows = _leaf_flow_oracle(sys, vptdf, path, table)
            kind = PNM.get_branch_kind(table)
            id = PNM.get_branch_id(table)
            parent = PNM.get_branch_parent(table)
            sign = PNM.get_branch_sign(table)
            @test length(flows) == length(kind)
            for r in findall(<=(8), kind)
                @test flows[r] ≈ by_key[(id[r], max(kind[r] - 5, 0))] atol = 1e-12
            end
            for r in findall(==(9), kind)
                kids = findall(==(r), parent)
                @test sum(sign[c] * flows[c] for c in kids) ≈ flows[r] atol = 1e-12
            end
            for k in eachindex(arc_ax)
                tops = findall(==(-k), parent)
                @test sum(sign[c] * flows[c] for c in tops) ≈ arc_flows[k] atol = 1e-12
            end
        end

        tiny(kind, parent, b) = BranchArcTable(
            kind, collect(1:length(kind)) .* (kind .<= 8), parent,
            ones(Int, length(kind)), b, zeros(length(kind)), 1)
        par = tiny([9, 1, 1], [-1, 1, 1], [0.0, 1.0, 3.0])
        @test branch_flows(par, [4.0]) ≈ [4.0, 1.0, 3.0]
        @test branch_flows(par, [4.0], [2]) ≈ [4.0, 0.0, 4.0]
        @test branch_flows(par, [4.0], [2, 3]) == zeros(3)
        nest = BranchArcTable(
            [9, 10, 1, 9, 1, 2, 1], [0, 0, 3, 0, 5, 6, 7], [-1, 1, 2, 2, 4, 4, 1],
            ones(Int, 7), [0.0, 0.0, 5.0, 0.0, 2.0, 4.0, 3.0], zeros(7), 1)
        @test branch_flows(nest, [0.0], [1]) == zeros(7)
        ser = tiny([10, 1, 1], [-1, 1, 1], [0.0, 1.0, 3.0])
        @test branch_flows(ser, [2.0]) ≈ [2.0, 2.0, 2.0]
        @test branch_flows(ser, [2.0], [3]) == zeros(3)
        @test_throws "arc_flows has 2 values; expected 1" branch_flows(par, [1.0, 2.0])
        @test_throws "outaged row 4" branch_flows(par, [1.0], [4])
        @test_throws "outaged row 0" branch_flows(par, [1.0], [0])
        @test_throws "row 1 has no DC coupling" branch_flows(
            tiny([9, 1, 1], [-1, 1, 1], [0.0, 1.0, -1.0]), [1.0])
        @test_throws "row 1 (series chain)" branch_flows(
            tiny([10, 1, 1], [-1, 1, 1], [0.0, 1.0, -1.0]), [1.0])
        @test_throws "row 1 has no DC coupling" branch_flows(
            tiny([1, 1], [-1, -1], [1.0, -1.0]), [1.0])
        @test_throws "the flow of row 1 (kind 1) is Inf" branch_flows(
            tiny([1], [-1], [1.0]), [Inf])
        @test_throws "the flow of row 1 (kind 1) is NaN" branch_flows(
            tiny([1], [-1], [1.0]), [NaN])
        # An outaged aggregate is skipped before its sum is checked.
        cancel = tiny([9, 1, 1], [-1, 1, 1], [0.0, 1.0, -1.0])
        @test branch_flows(cancel, [1.0], [1]) == zeros(3)
        buried = tiny([10, 9, 1, 1, 1], [-1, 1, 2, 2, 1], [0.0, 0.0, 1.0, -1.0, 2.0])
        @test branch_flows(buried, [1.0], [2]) == zeros(5)
        twin = tiny([1, 1], [-1, -1], [1.0, 2.0])
        @test branch_flows(twin, [1.0], [1, 2]) == zeros(2)
    end

    @testset "Ward, zero-impedance merge and resistive member" begin
        radial_d2 = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        function table_of(sys, reductions)
            vptdf = VirtualPTDF(sys; network_reductions = reductions, tol = 0.0)
            path = joinpath(mktempdir(), "ptdf.h5")
            to_hdf5(vptdf, path)
            return vptdf, path, from_hdf5(BranchArcTable, path)
        end
        # Branch flows against the leaf-level solve, in the numbering after the merge.
        function oracle_error(sys, vptdf, path, table)
            flows, by_key, _ = _leaf_flow_oracle(sys, vptdf, path, table)
            kind = PNM.get_branch_kind(table)
            id = PNM.get_branch_id(table)
            leaves = findall(<=(8), kind)
            return maximum(
                abs(flows[r] - by_key[(id[r], max(kind[r] - 5, 0))]) for r in leaves
            ),
            length(leaves)
        end

        @testset "Ward equivalent beside a branch" begin
            sys = PSB.build_system(
                PSB.PSSEParsingTestSystems,
                "psse_14_network_reduction_test_system",
            )
            _, _, table = table_of(
                sys,
                PNM.NetworkReduction[WardReduction([101, 114, 110, 111, 109])],
            )
            kind = PNM.get_branch_kind(table)
            parent = PNM.get_branch_parent(table)
            ward_rows = findall(==(11), kind)
            @test !isempty(ward_rows)
            tops_of(arc) = findall(==(-arc), parent)
            shared = [r for r in ward_rows if length(tops_of(-parent[r])) == 2]
            @test !isempty(shared)
            for r in shared
                @test sort(kind[tops_of(-parent[r])]) ==
                      sort([11, first(kind[tops_of(-parent[r])])])
                @test any(<=(8), kind[tops_of(-parent[r])])
            end
        end

        @testset "zero-impedance merge" begin
            plain = build_branch_map_system()
            vptdf, _, plain_table = table_of(plain, radial_d2)
            merged = build_branch_map_system()
            line = first(
                l for l in PSY.get_components(PSY.Line, merged) if
                _arc_bus_pair(PSY.get_arc(l)) == (2, 3)
            )
            PSY.set_r!(line, 0.0u"SU")
            PSY.set_x!(line, 0.0u"SU")
            vptdf, path, table = table_of(merged, radial_d2)
            @test PNM.get_reduction_codes(PNM.get_network_reduction_data(vptdf))[1] ==
                  PNM.REDUCTION_CODE_ZERO_IMPEDANCE
            @test count(==(-1), PNM.get_branch_sign(table)) >
                  count(==(-1), PNM.get_branch_sign(plain_table))
            err, n_leaf = oracle_error(merged, vptdf, path, table)
            @test n_leaf > 20
            @test err < 1e-12
        end

        @testset "purely resistive parallel member" begin
            sys = build_branch_map_system()
            member = PSY.get_component(PSY.Line, sys, "L_1_2_b")
            PSY.set_r!(member, 0.01u"SU")
            PSY.set_x!(member, 0.0u"SU")
            _, _, table = table_of(sys, radial_d2)
            row = findfirst(==(IS.get_id(member)), PNM.get_branch_id(table))
            # BA gives the member no DC coupling, so it carries no flow.
            @test iszero(PNM.get_branch_susceptance(table)[row])
            @test PNM.get_branch_kind(table)[PNM.get_branch_parent(table)[row]] == 9
            flows = branch_flows(table, ones(PNM.get_branch_arc_count(table)))
            @test iszero(flows[row])
        end
    end

    @testset "branch_outages id" begin
        sys = build_branch_map_system()
        reds = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vptdf = VirtualPTDF(sys; network_reductions = reds, tol = 0.0)
        path = joinpath(mktempdir(), "ptdf.h5")
        to_hdf5(vptdf, path, sys)
        table = from_hdf5(BranchOutageTable, path)
        bmap = from_hdf5(BranchArcTable, path)

        @testset "round trip" begin
            n_checked = 0
            for br in PSY.get_components(PSY.ACTransmission, sys)
                key = PNM._outage_key(br)
                haskey(PNM.get_outage_rows(table), key) || continue
                @test PNM.get_outage_ids(table)[IS.get_id(br)] == key
                @test NetworkModification(table, IS.get_id(br)) ==
                      NetworkModification(table, key...)
                n_checked += 1
            end
            @test n_checked > 10
            @test length(PNM.get_outage_ids(table)) == length(PNM.get_outage_rows(table))
            @test_throws "has no row in the outage table" NetworkModification(table, -1)
        end

        @testset "corrupt id" begin
            col = HDF5.h5open(f -> read(f["branch_outages"]["id"]), path, "r")
            dup = joinpath(mktempdir(), "dup.h5")
            cp(path, dup)
            bad = copy(col)
            bad[2] = bad[1]
            _write_bad_dataset(dup, "branch_outages", "id", bad)
            @test_throws "more than once" from_hdf5(BranchOutageTable, dup)
            short = joinpath(mktempdir(), "short.h5")
            cp(path, short)
            _write_bad_dataset(short, "branch_outages", "id", col[1:(end - 1)])
            @test_throws "unequal length" from_hdf5(BranchOutageTable, short)
            missing_id = joinpath(mktempdir(), "missing.h5")
            cp(path, missing_id)
            HDF5.h5open(f -> HDF5.delete_object(f["branch_outages"], "id"),
                missing_id, "r+")
            @test_throws "branch_outages/id" from_hdf5(BranchOutageTable, missing_id)
        end

        children(parent_row) = findall(==(parent_row), get_branch_parent(bmap))
        top_group(arc) =
            first(
                r for r in findall(==(-arc), get_branch_parent(bmap)) if
                get_branch_kind(bmap)[r] == 9
            )

        @testset "member child b equals -delta_b" begin
            n_leaf = 0
            n_chain = 0
            for i in eachindex(PNM.get_outage_member(table))
                PNM.get_outage_opens_arc(table)[i] && continue
                kids = children(top_group(table.arc_index[i]))
                child = kids[PNM.get_outage_member(table)[i]]
                if get_branch_kind(bmap)[child] <= 8
                    @test get_branch_susceptance(bmap)[child] ≈ -table.delta_b[i]
                    n_leaf += 1
                elseif get_branch_kind(bmap)[child] == 10
                    segs = children(child)
                    if all(s -> get_branch_kind(bmap)[s] <= 8, segs)
                        @test 1 / sum(1 / get_branch_susceptance(bmap)[s] for s in segs) ≈
                              -table.delta_b[i]
                        n_chain += 1
                    end
                end
            end
            @test n_leaf > 0
            @test n_chain > 0
        end

        @testset "post-outage branch_flows against a leaf-level solve" begin
            arc_ax = PNM.get_arc_axis(vptdf)
            bus_ax = PNM.get_bus_axis(vptdf)
            rng = MersenneTwister(11)
            inj = randn(rng, length(bus_ax))
            inj .-= sum(inj) / length(inj)
            p = Dict{Int, Float64}(b => inj[i] for (i, b) in enumerate(bus_ax))

            function dead_rows(id)
                dead = Int[]
                _, range = PNM.get_outage_rows(table)[PNM.get_outage_ids(table)[id]]
                for i in range
                    tops = findall(==(-table.arc_index[i]), get_branch_parent(bmap))
                    if PNM.get_outage_opens_arc(table)[i]
                        append!(dead, r for r in tops if get_branch_kind(bmap)[r] != 11)
                    else
                        push!(
                            dead,
                            children(top_group(table.arc_index[i]))[PNM.get_outage_member(
                                table,
                            )[i]],
                        )
                    end
                end
                return dead
            end

            function post_outage_arc_flows(by_key, dead)
                n = length(get_branch_kind(bmap))
                flow = zeros(n)
                for r in n:-1:1
                    r in dead && continue
                    if get_branch_kind(bmap)[r] <= 8
                        flow[r] = get(
                            by_key,
                            (get_branch_id(bmap)[r], max(get_branch_kind(bmap)[r] - 5, 0)),
                            0.0,
                        )
                    elseif get_branch_kind(bmap)[r] == 9
                        flow[r] =
                            sum(get_branch_sign(bmap)[c] * flow[c] for c in children(r))
                    elseif get_branch_kind(bmap)[r] == 10
                        c = first(children(r))
                        flow[r] = get_branch_sign(bmap)[c] * flow[c]
                    end
                end
                return Float64[
                    sum(
                        get_branch_sign(bmap)[r] * flow[r] for
                        r in findall(==(-k), get_branch_parent(bmap))
                    ) for k in eachindex(arc_ax)
                ]
            end

            matched = 0
            # The last case leaves the top parallel group without a live member.
            for names in (("L_1_2_b",), ("L_3_20",), ("L_1_10",), ("L_1_10", "L_3_20"))
                local sys_out = build_branch_map_system()
                dead = Int[]
                for name in names
                    br = PSY.get_component(PSY.Line, sys_out, name)
                    append!(dead, dead_rows(IS.get_id(br)))
                    PSY.set_available!(br, false)
                end
                @test !isempty(dead)
                fref = leaf_dc_flows(sys_out, p)
                by_key = _flows_by_leaf_key(fref)
                f_post = post_outage_arc_flows(by_key, dead)
                flows = branch_flows(bmap, f_post, dead)
                for r in findall(<=(8), get_branch_kind(bmap))
                    expected = get(
                        by_key,
                        (get_branch_id(bmap)[r], max(get_branch_kind(bmap)[r] - 5, 0)),
                        0.0,
                    )
                    @test flows[r] ≈ expected atol = 1e-10
                    matched += 1
                end
                @test iszero(flows[first(dead)])
            end
            @test matched >= 20
        end
    end

    @testset "branch_map writer rejects bad input" begin
        @test PNM._frame_sign((1, 2), (1, 2)) == 1
        @test PNM._frame_sign((1, 2), (2, 1)) == -1
        @test_throws "either orientation" PNM._frame_sign((1, 2), (1, 3))
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vptdf = VirtualPTDF(
            sys;
            network_reductions = PNM.NetworkReduction[
                RadialReduction(),
                DegreeTwoReduction(),
            ],
            tol = 0.0,
        )
        nr = PNM.get_network_reduction_data(PNM.get_branch_catalog(PNM.get_core(vptdf)))
        bs = first(values(PNM.get_series_branch_map(nr)))
        rows = BranchArcTable(Int[], Int[], Int[], Int[], Float64[], Float64[], 1)
        good = copy(bs.segment_orientations)
        try
            push!(bs.segment_orientations, :FromTo)
            @test_throws "orientations" PNM._push_node!(
                rows,
                bs,
                -1,
                1,
                nr,
                PNM._ba_arc_susceptance,
            )
            bs.segment_orientations[1] = :Sideways
            pop!(bs.segment_orientations)
            @test_throws "not :FromTo or :ToFrom" PNM._push_node!(
                rows,
                bs,
                -1,
                1,
                nr,
                PNM._ba_arc_susceptance,
            )
        finally
            empty!(bs.segment_orientations)
            append!(bs.segment_orientations, good)
        end
    end

    @testset "reduction records" begin
        function reduction_check(sys, reductions)
            vptdf = VirtualPTDF(sys; network_reductions = reductions, tol = 0.0)
            path = joinpath(mktempdir(), "ptdf.h5")
            to_hdf5(vptdf, path, sys; outages = false)
            restored = from_hdf5(VirtualPTDF, path)
            live_reductions = BranchReductionTable(vptdf, sys)
            read_reductions = from_hdf5(BranchReductionTable, path)
            @test read_reductions.rows == live_reductions.rows
            table = from_hdf5(BranchArcTable, path)
            has_row = Set(
                (PNM.get_branch_id(table)[r], max(PNM.get_branch_kind(table)[r] - 5, 0))
                for r in eachindex(PNM.get_branch_kind(table)) if
                PNM.get_branch_kind(table)[r] <= 8
            )
            nr = PNM.get_network_reduction_data(vptdf)
            nr_restored = PNM.get_network_reduction_data(restored)
            for field in (
                :removed_buses, :removed_arcs, :merged_bus_pairs,
                :removed_arc_to_surviving_bus, :boundary_bus_to_removed_arcs,
                :irreducible_buses, :bus_reduction_map, :reverse_bus_search_map,
                :removed_arc_reduction, :removed_arc_composite, :reduction_codes,
            )
                @test getfield(nr_restored, field) == getfield(nr, field)
            end
            code_counts = Dict{Int, Int}()
            for key in keys(leaf_bus_pairs(sys))
                live = _by_winding(branch_reduction, key[2], read_reductions, key[1])
                @test iszero(live) == (key in has_row)
                code_counts[live] = get(code_counts, live, 0) + 1
                if live == BRANCH_REDUCTION_RADIAL
                    @test _by_winding(
                        branch_surviving_bus,
                        key[2],
                        read_reductions,
                        key[1],
                    ) in
                          union(PNM.get_bus_axis(restored), nr_restored.removed_buses)
                end
            end
            return code_counts, path
        end

        radial_d2 = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        code_counts, _ = reduction_check(build_branch_map_system(), radial_d2)
        @test code_counts[BRANCH_REDUCTION_RADIAL] > 0
        @test code_counts[BRANCH_REDUCTION_HAS_ARC] > 0
        # A zero-impedance merge renames buses before the radial and degree-two steps.
        sys = build_branch_map_system()
        for line in PSY.get_components(PSY.Line, sys)
            if _arc_bus_pair(PSY.get_arc(line)) == (2, 3)
                PSY.set_r!(line, 0.0u"SU")
                PSY.set_x!(line, 0.0u"SU")
            end
        end
        code_counts, _ = reduction_check(sys, radial_d2)
        @test code_counts[BRANCH_REDUCTION_ZERO_IMPEDANCE] == 1
        @test code_counts[BRANCH_REDUCTION_RADIAL] > 0
        @test PNM.get_reduction_codes(
            PNM.get_network_reduction_data(
                VirtualPTDF(sys; network_reductions = radial_d2, tol = 0.0),
            ),
        ) == [1, 2, 3]
        code_counts, _ = reduction_check(
            PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions"),
            radial_d2,
        )
        @test code_counts[BRANCH_REDUCTION_RADIAL] > 0

        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        line = PSY.get_component(PSY.Line, sys, "Line3")
        PSY.set_r!(line, 0.0u"SU")
        PSY.set_x!(line, 0.0u"SU")
        PSY.set_available!(PSY.get_component(PSY.Line, sys, "Line10"), false)
        code_counts, path = reduction_check(sys, PNM.NetworkReduction[])
        @test code_counts[BRANCH_REDUCTION_ZERO_IMPEDANCE] == 1
        @test code_counts[BRANCH_REDUCTION_NOT_IN_NETWORK] == 1
        line10 = IS.get_id(PSY.get_component(PSY.Line, sys, "Line10"))
        @test branch_reduction(from_hdf5(BranchReductionTable, path), line10) ==
              BRANCH_REDUCTION_NOT_IN_NETWORK
        # Two zero-impedance sides of a triangle make the third side a self-loop.
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        for line in PSY.get_components(PSY.Line, sys)
            pair = _arc_bus_pair(PSY.get_arc(line))
            if pair == (6, 12) || pair == (12, 13)
                PSY.set_r!(line, 0.0u"SU")
                PSY.set_x!(line, 0.0u"SU")
            end
        end
        code_counts, _ = reduction_check(sys, PNM.NetworkReduction[])
        @test code_counts[BRANCH_REDUCTION_ZERO_IMPEDANCE] == 3

        sys = PSB.build_system(
            PSB.PSSEParsingTestSystems,
            "psse_14_network_reduction_test_system",
        )
        code_counts, _ = reduction_check(
            sys,
            PNM.NetworkReduction[WardReduction([101, 114, 110, 111, 109])],
        )
        @test code_counts[BRANCH_REDUCTION_WARD] > 0
        @test code_counts[BRANCH_REDUCTION_HAS_ARC] > 0
        # Later reductions can remove the composite arc of a degree-two chain.
        ward = WardReduction([101, 114, 110, 111, 109])
        for reductions in (
            PNM.NetworkReduction[DegreeTwoReduction(), ward],
            PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction(), ward],
        )
            code_counts, _ = reduction_check(sys, reductions)
            @test code_counts[BRANCH_REDUCTION_WARD] > 0
        end
        code_counts, _ = reduction_check(
            PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions"),
            PNM.NetworkReduction[DegreeTwoReduction(), RadialReduction()],
        )
        @test code_counts[BRANCH_REDUCTION_RADIAL] > 0
        # An out-of-service branch beside a radial stub is not in the network.
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        for line in PSY.get_components(PSY.Line, sys)
            if _arc_bus_pair(PSY.get_arc(line)) == (6, 12)
                PSY.set_available!(line, false)
            end
        end
        code_counts, _ = reduction_check(sys, PNM.NetworkReduction[RadialReduction()])
        @test code_counts[BRANCH_REDUCTION_NOT_IN_NETWORK] == 1
    end

    @testset "reduction group corrupt" begin
        sys = build_branch_map_system()
        vptdf = VirtualPTDF(
            sys;
            network_reductions = PNM.NetworkReduction[
                RadialReduction(),
                DegreeTwoReduction(),
            ],
            tol = 0.0,
        )
        good = joinpath(mktempdir(), "good.h5")
        to_hdf5(vptdf, good)
        rec = HDF5.h5open(
            f -> Dict(c => read(f["reduction"][string(c)]) for c in PNM._REDUCTION_COLUMNS),
            good, "r")
        bus_ax = PNM.get_bus_axis(vptdf)
        corrupt(edits...) = _corrupted_copy(good, "reduction", edits...)
        bad(name, value, msg) =
            @test_throws msg from_hdf5(VirtualPTDF, corrupt(name => value))
        @test !isempty(rec[:removed_arc_from])
        @test !isempty(rec[:removed_buses])
        n = length(rec[:removed_arc_from])
        bad(:removed_arc_to, rec[:removed_arc_to][1:(end - 1)], "unequal length")
        bad(:survivor_bus, Int[], "unequal length")
        bad(:merged_from, [1], "unequal length")
        bad(:boundary_bus, [1], "unequal length")
        bad(:applied, [9], "code 9")
        bad(:applied, [2, 2], "applied has the value 2 more than once, at row 2")
        bad(:applied, [4, 2], "applied has the Ward reduction before the end")
        bad(:removed_arc_code, fill(1, n), "not in applied")
        bad(
            :removed_arc_from,
            fill(rec[:removed_arc_from][1], n),
            "removed_arc has the value (1, 10) more than once, at row 4",
        )
        bad(
            :removed_arc_from,
            vcat(-5, rec[:removed_arc_from][2:end]),
            "removed_arc_from holds bus -5 at row 1",
        )
        bad(:removed_buses, [first(bus_ax)], "on the bus axis")
        bad(
            :removed_buses,
            fill(rec[:removed_buses][1], 2),
            "removed_buses has the value $(rec[:removed_buses][1]) more than once, at row 2",
        )
        bad(:irreducible_buses, [-7], "irreducible_buses holds")
        # Rules that need several columns edited at once.
        @test_throws "not a removed arc" from_hdf5(
            VirtualPTDF,
            corrupt(
                :survivor_from => [-1],
                :survivor_to => [-1],
                :survivor_bus => [first(bus_ax)],
            ),
        )
        @test_throws "not a removed arc" from_hdf5(
            VirtualPTDF,
            corrupt(
                :boundary_bus => [first(bus_ax)],
                :boundary_from => [-1],
                :boundary_to => [-1],
            ),
        )
        @test_throws "reduction/survivor_bus holds bus -7 at row 1" from_hdf5(
            VirtualPTDF,
            corrupt(
                :survivor_from => rec[:removed_arc_from][1:1],
                :survivor_to => rec[:removed_arc_to][1:1],
                :survivor_bus => [-7],
            ),
        )
        @test_throws "merged_from holds bus $(first(bus_ax)), which is on the bus axis" from_hdf5(
            VirtualPTDF,
            corrupt(:merged_from => [first(bus_ax)], :merged_to => [first(bus_ax)]),
        )
        @test_throws "merged_from has the value -9 more than once, at row 2" from_hdf5(
            VirtualPTDF,
            corrupt(
                :merged_from => [-9, -9],
                :merged_to => [first(bus_ax), first(bus_ax)],
            ),
        )
        r1 = (rec[:removed_arc_from][1], rec[:removed_arc_to][1])
        @test_throws "survivor has the value (1, 10) more than once, at row 2" from_hdf5(
            VirtualPTDF,
            corrupt(
                :survivor_from => [r1[1], r1[1]],
                :survivor_to => [r1[2], r1[2]],
                :survivor_bus => [first(bus_ax), first(bus_ax)],
            ),
        )
        @test_throws "boundary has the value (1, 1, 10) more than once, at row 2" from_hdf5(
            VirtualPTDF,
            corrupt(
                :boundary_bus => [first(bus_ax), first(bus_ax)],
                :boundary_from => [r1[1], r1[1]],
                :boundary_to => [r1[2], r1[2]],
            ),
        )
        @test_throws "irreducible_buses has the value 1 more than once, at row 2" from_hdf5(
            VirtualPTDF,
            corrupt(:irreducible_buses => [first(bus_ax), first(bus_ax)]),
        )
        bad(:removed_arc_to, vcat(-5, rec[:removed_arc_to][2:end]), "removed_arc_to holds")
        @test_throws "boundary_bus holds" from_hdf5(
            VirtualPTDF,
            corrupt(
                :boundary_bus => [-7],
                :boundary_from => [r1[1]],
                :boundary_to => [r1[2]],
            ),
        )
        @test_throws "merged_to holds" from_hdf5(
            VirtualPTDF,
            corrupt(:merged_from => rec[:removed_buses][1:1], :merged_to => [-8]),
        )
        @test_throws "merged_from holds" from_hdf5(
            VirtualPTDF,
            corrupt(:merged_from => [-9], :merged_to => [first(bus_ax)]),
        )
        d2_row = findfirst(==(PNM.REDUCTION_CODE_DEGREE_TWO), rec[:removed_arc_code])
        radial_row = findfirst(==(PNM.REDUCTION_CODE_RADIAL), rec[:removed_arc_code])
        bad(
            :composite_from,
            replace(rec[:composite_from], rec[:composite_from][d2_row] => 0),
            "empty for a degree-two arc",
        )
        bad(
            :composite_to,
            replace(rec[:composite_to], rec[:composite_to][d2_row] => -3),
            "composite_to holds bus -3 at row $d2_row",
        )
        bad(
            :composite_from,
            replace(rec[:composite_from], rec[:composite_from][d2_row] => -3),
            "composite_from holds bus -3 at row $d2_row",
        )
        axis_arc = first(PNM.get_arc_axis(vptdf))
        @test_throws "removed_arc row $(n + 1) holds arc $axis_arc, which is on the arc axis" from_hdf5(
            VirtualPTDF,
            corrupt(
                :removed_arc_from => vcat(rec[:removed_arc_from], axis_arc[1]),
                :removed_arc_to => vcat(rec[:removed_arc_to], axis_arc[2]),
                :removed_arc_code =>
                    vcat(rec[:removed_arc_code], PNM.REDUCTION_CODE_RADIAL),
                :composite_from => vcat(rec[:composite_from], 0),
                :composite_to => vcat(rec[:composite_to], 0),
            ),
        )
        @test_throws "removed_arc row $(n + 1) holds arc $(reverse(axis_arc)), which is on the arc axis" from_hdf5(
            VirtualPTDF,
            corrupt(
                :removed_arc_from => vcat(rec[:removed_arc_from], axis_arc[2]),
                :removed_arc_to => vcat(rec[:removed_arc_to], axis_arc[1]),
                :removed_arc_code =>
                    vcat(rec[:removed_arc_code], PNM.REDUCTION_CODE_RADIAL),
                :composite_from => vcat(rec[:composite_from], 0),
                :composite_to => vcat(rec[:composite_to], 0),
            ),
        )
        @test length(rec[:removed_buses]) >= 2
        @test_throws "merged_to holds bus $(rec[:removed_buses][2]) at row 1, which is also in merged_from" from_hdf5(
            VirtualPTDF,
            corrupt(
                :merged_from => rec[:removed_buses][1:2],
                :merged_to => [rec[:removed_buses][2], first(bus_ax)],
            ),
        )
        radial_from = copy(rec[:composite_from])
        radial_from[radial_row] = 5
        bad(:composite_from, radial_from, "not the degree-two code")
        bad(:composite_to, rec[:composite_to][1:(end - 1)], "unequal length")
        # A composite arc that is a degree-two row would loop the reduction lookup.
        d2_arc = (rec[:removed_arc_from][d2_row], rec[:removed_arc_to][d2_row])
        function with_composite(row, arc)
            from_col = copy(rec[:composite_from])
            to_col = copy(rec[:composite_to])
            from_col[row] = arc[1]
            to_col[row] = arc[2]
            return (:composite_from => from_col, :composite_to => to_col)
        end
        @test_throws "holds the degree-two arc" from_hdf5(
            VirtualPTDF,
            corrupt(with_composite(d2_row, d2_arc)...),
        )
        d2_rows = findall(==(PNM.REDUCTION_CODE_DEGREE_TWO), rec[:removed_arc_code])
        @test length(d2_rows) >= 2
        other = (rec[:removed_arc_from][d2_rows[2]], rec[:removed_arc_to][d2_rows[2]])
        @test_throws "holds the degree-two arc" from_hdf5(
            VirtualPTDF,
            corrupt(with_composite(d2_row, other)...),
        )
        # A composite arc must be on the arc axis or removed, in either orientation.
        known_arcs = Set(PNM.get_arc_axis(vptdf))
        union!(known_arcs, zip(rec[:removed_arc_from], rec[:removed_arc_to]))
        unknown = first(
            (a, b) for a in bus_ax for b in bus_ax if
            a != b && !((a, b) in known_arcs) && !((b, a) in known_arcs)
        )
        @test_throws "neither on the arc axis nor a removed arc" from_hdf5(
            VirtualPTDF,
            corrupt(with_composite(d2_row, unknown)...),
        )
        # Error paths of the no-arc queries.
        nr = PNM.get_network_reduction_data(from_hdf5(VirtualPTDF, good))
        @test_throws "No reduction removed" PNM._bus_pair_surviving_bus(nr, -1, -2)
        @test_throws "did not remove" PNM._bus_pair_surviving_bus(nr, d2_arc...)
        nr.removed_arc_composite[d2_arc] = (-1, -2)
        @test_throws "no record explains" PNM._bus_pair_reduction(
            nr,
            Dict{Tuple{Int, Int}, Int}(),
            d2_arc...,
        )
        reduced_to = HDF5.h5open(f -> read(f["reduced_bus_to"]), good, "r")
        root_path = joinpath(mktempdir(), "root.h5")
        cp(good, root_path)
        _write_bad_dataset(root_path, "", "reduced_bus_to", vcat(-77, reduced_to[2:end]))
        @test_throws "reduced_bus_to holds bus -77" from_hdf5(VirtualPTDF, root_path)
        missing_group = joinpath(mktempdir(), "missing.h5")
        cp(good, missing_group)
        HDF5.h5open(f -> HDF5.delete_object(f, "reduction"), missing_group, "r+")
        @test_throws "Factor file has no entry /reduction" from_hdf5(
            VirtualPTDF,
            missing_group,
        )
    end

    @testset "VirtualPTDF c_sys14" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf, restored, _ = _roundtrip_ptdf(sys)
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        @test isempty(PNM.get_dist_slack(restored))
        arcs = PNM.get_arc_axis(vptdf)[1:2]
        path = joinpath(mktempdir(), "persistent.h5")
        to_hdf5(vptdf, path)
        persistent = from_hdf5(VirtualPTDF, path; persistent_arcs = arcs)
        @test persistent.cache.persistent_cache_keys ==
              Set(PNM.get_arc_lookup(persistent)[a] for a in arcs)
    end

    @testset "VirtualPTDF with dist_slack" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        buses = sort!([PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)])
        ds = Dict{Int, Float64}(buses[2] => 0.3, buses[5] => 0.7)
        vptdf, restored, _ = _roundtrip_ptdf(sys; dist_slack = ds)
        @test !isempty(PNM.get_dist_slack(restored))
        @test PNM.get_dist_slack(restored) == PNM.get_dist_slack(vptdf)
        @test PNM.get_dist_slack_normalized(restored) ≈
              PNM.get_dist_slack_normalized(vptdf)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
    end

    @testset "VirtualPTDF with an explicit AutoTolerance" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        spec = AutoTolerance(; data_precision = 1e-8, safety = 2.0, quantile = 0.9)
        vptdf, restored, _ = _roundtrip_ptdf(sys; tol = spec)
        @test PNM.get_tol_spec(PNM.get_core(restored)) == spec
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
    end

    @testset "VirtualMODF RTS-GMLC with reductions" begin
        @test PNM.get_reverse_bus_search_map(
            PNM.get_network_reduction_data(RTS_RESTORED),
        ) ==
              PNM.get_reverse_bus_search_map(RTS_NR)
        _test_same_core(RTS_VMODF, RTS_RESTORED)
        _test_ptdf_rows_equal(
            VirtualPTDF(PNM.get_core(RTS_VMODF)),
            VirtualPTDF(PNM.get_core(RTS_RESTORED)),
            PNM.get_arc_axis(RTS_VMODF),
        )
        for map in (
            RTS_NR.reverse_direct_branch_map,
            RTS_NR.reverse_parallel_branch_map,
            RTS_NR.reverse_series_branch_map,
        )
            mod = NetworkModification(RTS_VMODF, first(keys(map)))
            _test_modf_rows_equal(RTS_VMODF, RTS_RESTORED, mod)
        end
    end

    @testset "case10 radial and series reductions with 3WT outage" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf = VirtualMODF(sys; network_reductions = RTS_REDUCTIONS)
        restored = _modf_roundtrip(vmodf)
        _test_same_core(vmodf, restored)
        _test_ptdf_rows_equal(
            VirtualPTDF(PNM.get_core(vmodf)),
            VirtualPTDF(PNM.get_core(restored)),
            PNM.get_arc_axis(vmodf),
        )
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        _test_modf_rows_equal(vmodf, restored, NetworkModification(vmodf, trf))
    end

    @testset "ACTIVSg2000 large system" begin
        sys = PSB.build_system(PSB.MatpowerTestSystems, "matpower_ACTIVSg2000_sys")
        vptdf, restored, _ = _roundtrip_ptdf(sys; network_reductions = RTS_REDUCTIONS)
        _test_same_core(vptdf, restored)
        arcs = PNM.get_arc_axis(vptdf)
        _test_ptdf_rows_equal(
            vptdf,
            restored,
            arcs[round.(Int, range(1, length(arcs); length = 5))],
        )
        vmodf =
            VirtualMODF(PNM.get_core(vptdf), sys; automatically_register_outages = false)
        restored_modf = _modf_roundtrip(vmodf)
        nr = PNM.get_network_reduction_data(vmodf)
        mod = NetworkModification(vmodf, first(keys(nr.reverse_direct_branch_map)))
        monitored = length(arcs) ÷ 2
        @test isapprox(
            collect(vmodf[monitored, mod]),
            collect(restored_modf[monitored, mod]);
            atol = 1e-10,
        )
    end

    @testset "phase shifter" begin
        sys = _mk_line_pst_parallel_system()
        vptdf, restored, path = _roundtrip_ptdf(sys)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        shift = HDF5.h5open(path, "r") do f
            return read(f["shift_injection"])
        end
        @test any(!iszero, shift)
    end

    @testset "islanding outage" begin
        sys = PSB.build_system(
            PSB.PSSEParsingTestSystems,
            "psse_14_network_reduction_test_system",
        )
        vmodf =
            VirtualMODF(sys; linear_solver = "KLU", automatically_register_outages = false)
        restored = _modf_roundtrip(vmodf)
        bridges = [
            e for
            e in eachindex(vmodf.PTDF_A_diag) if abs(vmodf.PTDF_A_diag[e] - 1.0) < 1e-6
        ]
        @test !isempty(bridges)
        e = first(bridges)
        mod = NetworkModification(
            "bridge",
            [ArcModification(e, -vmodf.arc_susceptances[e])],
            ShuntModification[],
            true,
        )
        monitored = first(i for i in eachindex(vmodf.PTDF_A_diag) if i != e)
        base_row = PNM.get_ptdf_row(
            VirtualPTDF(PNM.get_core(vmodf)),
            PNM.get_arc_axis(vmodf)[monitored],
        )
        @test !isapprox(collect(vmodf[monitored, mod]), collect(base_row); atol = 1e-9)
        _test_modf_rows_equal(vmodf, restored, mod)
    end

    @testset "an opened arc cannot be written" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf = VirtualPTDF(sys)
        arc = PNM.get_arc_axis(vptdf)[3]
        PNM.get_core(vptdf).arc_live_members[3] = 0
        @test_throws "The arc $arc has arc_live_members 0, below 1. A FactorCore cannot hold an opened arc" to_hdf5(
            vptdf, joinpath(mktempdir(), "open.h5"),
        )
    end

    @testset "multi-island round trip" begin
        sys = build_multi_island_composite_arc_system()
        vptdf, back, _ = _roundtrip_ptdf(sys)
        @test length(PNM.get_core(back).subnetwork_axes) == 2
        _test_ptdf_rows_equal(vptdf, back, PNM.get_arc_axis(vptdf))
    end

    @testset "errors" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf, _, path = _roundtrip_ptdf(sys)
        @test_throws "exists. Pass `force = true`" to_hdf5(vptdf, path)
        to_hdf5(vptdf, path; force = true)
        HDF5.h5open(path, "r+") do f
            HDF5.delete_attribute(f, "tol_kind")
            HDF5.write_attribute(f, "tol_kind", "other")
        end
        @test_throws "Unknown tolerance kind" from_hdf5(VirtualPTDF, path)
        HDF5.h5open(path, "r+") do f
            HDF5.delete_attribute(f, "format_version")
            HDF5.write_attribute(f, "format_version", PMC.FACTOR_FILE_FORMAT_VERSION + 1)
        end
        @test_throws "reads version $(PMC.FACTOR_FILE_FORMAT_VERSION)" from_hdf5(
            VirtualPTDF,
            path,
        )
        @test_throws "reads version $(PMC.FACTOR_FILE_FORMAT_VERSION)" from_hdf5(
            VirtualMODF,
            path,
        )

        to_hdf5(vptdf, path; force = true)
        HDF5.h5open(path, "r+") do f
            HDF5.delete_object(f, "names")
        end
        @test_throws "Factor file has no entry /names" from_hdf5(VirtualPTDF, path)

        # Each case rewrites the file, corrupts one entry and expects an error naming it.
        function _replace!(g, name, value)
            HDF5.delete_object(g, name)
            g[name] = value
            return
        end
        corrupt = (
            (f -> _replace!(f, "bus_axis", [1.0]), "entry bus_axis has the HDF5 class"),
            (
                f -> _replace!(f, "arc_to", Int[]),
                "table arcs has columns of unequal length",
            ),
            (
                f -> _replace!(f["subnetworks"][first(keys(f["subnetworks"]))], "ref", "1"),
                "/ref has the HDF5 class",
            ),
        )
        for (mutate, message) in corrupt
            to_hdf5(vptdf, path; force = true)
            HDF5.h5open(mutate, path, "r+")
            @test_throws message from_hdf5(VirtualPTDF, path)
        end

        to_hdf5(vptdf, path; force = true)
        n_names = HDF5.h5open(f -> length(read(f["names"]["name"])), path, "r")
        HDF5.h5open(path, "r+") do f
            HDF5.delete_object(f["names"], "type")
            PNM._write_strings!(f["names"], "type", fill("eval", n_names))
        end
        @test_throws "unknown branch type" from_hdf5(VirtualPTDF, path)
    end
end

# Every monitored row of `mod` from the core functions equals the row of `expected` on `vmodf`.
function _test_core_modf_rows(vmodf, core, mod, expected = mod)
    wf = PMC.compute_woodbury_factors(core, mod)
    @test wf.is_islanding == expected.is_islanding
    cutoff = PMC.get_cutoff(core)
    for monitored in 1:length(PMC.get_arc_axis(core))
        row = PMC.apply_cutoff(cutoff, PMC.apply_woodbury_correction(core, monitored, wf))
        @test isapprox(collect(vmodf[monitored, expected]), collect(row); atol = 1e-9)
    end
    return
end

# The reads and the rows below use PowerMatricesCore only: no System, no reduction data and
# no PNM matrix.
@testset "factor file read by PowerMatricesCore alone" begin
    @testset "PTDF rows with distributed slack and a cutoff" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        buses = sort!([PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)])
        ds = Dict{Int, Float64}(buses[2] => 0.3, buses[5] => 0.7)
        vptdf, _, path = _roundtrip_ptdf(sys; dist_slack = ds, tol = 1e-3)
        core = PMC.read_factor_core(path)
        @test isnothing(PMC.get_network_data(core))
        @test PMC.get_arc_axis(core) == PNM.get_arc_axis(vptdf)
        @test PMC.get_bus_axis(core) == PNM.get_bus_axis(vptdf)
        cutoff = PMC.get_cutoff(core)
        @test cutoff == PMC.AbsoluteCutoff(1e-3)
        weights = HDF5.h5open(path, "r") do f
            return PMC.read_dist_slack(f, core, PMC.ReadBudget(path))
        end
        @test weights == PNM.get_dist_slack(vptdf)
        normalized = weights ./ sum(weights)
        dropped = 0
        for (e, arc) in enumerate(PMC.get_arc_axis(core))
            row = PMC.apply_dist_slack!(PMC.ptdf_row(core, e), core, normalized)
            stored = PMC.apply_cutoff(cutoff, row)
            dropped += count(iszero, collect(stored)) - count(iszero, row)
            @test isapprox(
                collect(stored),
                collect(PNM.get_ptdf_row(vptdf, arc));
                atol = 1e-12,
            )
        end
        @test dropped > 0
    end

    @testset "post-outage rows on RTS-GMLC with reductions" begin
        path = joinpath(mktempdir(), "rts.h5")
        to_hdf5(RTS_VMODF, path)
        core = PMC.read_factor_core(path)
        @test isnothing(PMC.get_network_data(core))
        for map in (
            RTS_NR.reverse_direct_branch_map,
            RTS_NR.reverse_parallel_branch_map,
            RTS_NR.reverse_series_branch_map,
        )
            mod = NetworkModification(RTS_VMODF, first(keys(map)))
            _test_core_modf_rows(RTS_VMODF, core, mod)
        end
    end

    @testset "branch reductions" begin
        sys = PSB.build_system(
            PSB.PSITestSystems,
            "case10_radial_series_reductions",
        )
        vmodf =
            VirtualMODF(sys; network_reductions = [RadialReduction(), DegreeTwoReduction()])
        path = joinpath(mktempdir(), "reductions.h5")
        to_hdf5(vmodf, path, sys)
        read_reductions = PMC.read_branch_reductions(path)
        @test read_reductions.rows == BranchReductionTable(vmodf, sys).rows
        radial = [
            k for (k, v) in read_reductions.rows if first(v) == PMC.BRANCH_REDUCTION_RADIAL
        ]
        @test !isempty(radial)
        bus_ax = PNM.get_bus_axis(vmodf)
        for (id, winding) in radial
            @test _by_winding(PMC.branch_surviving_bus, winding, read_reductions, id) in
                  bus_ax
        end
    end

    @testset "branch outage table on RTS-GMLC with reductions" begin
        path = joinpath(mktempdir(), "rts_outages.h5")
        to_hdf5(RTS_VMODF, path, RTS_SYS)
        core = PMC.read_factor_core(path)
        table = PMC.read_branch_outages(path)
        rows = PMC.get_outage_rows(table)
        @test rows == PNM.get_outage_rows(RTS_TABLE)
        @test !isempty(rows)
        ids = Dict(key => id for (id, key) in PMC.get_outage_ids(table))
        for (type, name) in keys(rows)
            branch = PSY.get_component(PNM._branch_type(type, "type"), RTS_SYS, name)
            mod = PMC.NetworkModification(table, type, name)
            @test IS.get_id(branch) == ids[(type, name)]
            @test mod == PMC.NetworkModification(table, ids[(type, name)])
            _test_core_modf_rows(
                RTS_VMODF,
                core,
                mod,
                NetworkModification(RTS_VMODF, branch),
            )
        end
        bare = joinpath(mktempdir(), "bare.h5")
        to_hdf5(RTS_VMODF, bare)
        @test_throws "no branch outage table" PMC.read_branch_outages(bare)
    end

    @testset "branch names on RTS-GMLC with reductions" begin
        path = joinpath(mktempdir(), "rts_names.h5")
        to_hdf5(RTS_VMODF, path)
        core = PMC.read_factor_core(path)
        names = PMC.read_branch_names(path)
        catalog = PNM.get_name_to_arc_maps(PNM.get_branch_catalog(PNM.get_core(RTS_VMODF)))
        expected = Dict(
            (PNM._branch_type_name(T), name) => arc for (T, entries) in catalog for
            (name, arc) in entries
        )
        @test !isempty(names)
        @test names == expected
        arc_ax = PMC.get_arc_axis(core)
        lookup = PMC.get_arc_lookup(core)
        @test all(arc -> arc_ax[lookup[arc]] == arc, values(names))
        @test Set(values(names)) == Set(arc_ax)
    end

    @testset "branch flows from the branch map" begin
        reds = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        for sys in (
            build_branch_map_system(),
            PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions"),
        )
            vptdf = VirtualPTDF(sys; network_reductions = reds, tol = 0.0)
            path = joinpath(mktempdir(), "ptdf.h5")
            to_hdf5(vptdf, path)
            core = PMC.read_factor_core(path)
            table = PMC.read_branch_map(path)
            shift = HDF5.h5open(f -> PMC.read_shift_injection(f, core), path, "r")
            arc_ax = PMC.get_arc_axis(core)
            bix = PMC.get_bus_lookup(core)
            inj = randn(MersenneTwister(7), length(bix))
            inj .-= sum(inj) / length(inj)
            q = copy(inj)
            for (k, a) in enumerate(arc_ax)
                q[bix[a[1]]] += shift[k]
                q[bix[a[2]]] -= shift[k]
            end
            arc_flows = Float64[
                sum(PMC.ptdf_row(core, k) .* q) - shift[k] for k in eachindex(arc_ax)
            ]
            flows = PMC.branch_flows(table, arc_flows)
            by_key = _flows_by_leaf_key(
                leaf_dc_flows(sys, Dict{Int, Float64}(b => inj[i] for (b, i) in bix)),
            )
            kind = PMC.get_branch_kind(table)
            id = PMC.get_branch_id(table)
            leaves = findall(<=(8), kind)
            @test !isempty(leaves)
            for r in leaves
                @test flows[r] ≈ by_key[(id[r], max(kind[r] - 5, 0))] atol = 1e-12
            end
        end
    end

    @testset "branch factors by id" begin
        reds = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        for (name, sys) in (
            ("branch_map", build_branch_map_system()),
            (
                "case10",
                PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions"),
            ),
        )
            vptdf = VirtualPTDF(sys; network_reductions = reds, tol = 0.0)
            path = joinpath(mktempdir(), "ptdf.h5")
            to_hdf5(vptdf, path, sys)
            x = PMC.read_branch_factors(path)
            ref = only(PMC.get_ref_bus(PMC.get_core(x)))
            buses = PMC.get_bus_axis(x)
            # The flows for the unit injection at each bus, from the full network.
            function reference_flows(system)
                base = _flows_by_leaf_key(leaf_dc_flows(system, Dict{Int, Float64}()))
                moved = Dict{Int, Dict{Tuple{Int, Int}, Float64}}()
                for b in buses
                    if b != ref
                        moved[b] = _flows_by_leaf_key(
                            leaf_dc_flows(system, Dict(b => 1.0, ref => -1.0)),
                        )
                    end
                end
                return base, moved
            end
            function compare_rows(row_of, base, moved, leaves)
                compared = Tuple{Int, Int}[]
                for key in leaves
                    if haskey(base, key)
                        row = row_of(key)
                        for b in buses
                            expected = 0.0
                            if b != ref
                                expected = moved[b][key] - base[key]
                            end
                            @test row[PMC.bus_column(x, b)] ≈ expected atol = 1e-9
                        end
                        push!(compared, key)
                    end
                end
                return compared
            end
            base, moved = reference_flows(sys)
            leaves = Tuple{Int, Int}[
                key for key in keys(base) if _by_winding(PMC.has_branch, key[2], x, key[1])
            ]
            compared = compare_rows(
                key -> _by_winding(PMC.branch_ptdf_row, key[2], x, key[1]),
                base,
                moved,
                leaves,
            )
            @test length(compared) > 0
            if name == "case10"
                @test any(key -> key[2] > 0, compared)
            end

            table = PMC.read_branch_outages(path)
            picked = Tuple{Int, String, String}[]
            for id in sort!(collect(keys(PMC.get_outage_ids(table))))
                type, branch_name = PMC.get_outage_ids(table)[id]
                if isempty(picked) && PMC.has_branch(x, id) &&
                   !PMC.NetworkModification(table, id).is_islanding
                    push!(picked, (id, type, branch_name))
                end
            end
            id, type, branch_name = only(picked)
            ctg = PMC.branch_contingency(x, [id])
            @test !PMC.is_islanding(ctg)
            sys_out = deepcopy(sys)
            PSY.set_available!(
                PSY.get_component(PNM._branch_type(type, "type"), sys_out, branch_name),
                false,
            )
            base_out, moved_out = reference_flows(sys_out)
            @test iszero(PMC.branch_ptdf_row(x, id, ctg))
            compared_out = compare_rows(
                key -> _by_winding(PMC.branch_ptdf_row, key[2], x, key[1], ctg),
                base_out,
                moved_out,
                leaves,
            )
            @test length(compared_out) > 0
            if name == "case10"
                @test any(key -> key[2] > 0, compared_out)
            end
        end
    end

    @testset "three-winding transformer contingency by id" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        reds = PNM.NetworkReduction[RadialReduction(), DegreeTwoReduction()]
        vmodf = VirtualMODF(sys; network_reductions = reds, tol = 0.0)
        path = joinpath(mktempdir(), "modf.h5")
        to_hdf5(vmodf, path, sys)
        x = PMC.read_branch_factors(path)
        table = PMC.read_branch_map(path)
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        id = IS.get_id(trf)
        mod = NetworkModification(vmodf, trf)
        ctg = PMC.branch_contingency(x, [id])
        @test PMC.is_islanding(ctg) == mod.is_islanding
        outaged_rows = Int[
            r for r in eachindex(PMC.get_branch_kind(table)) if
            6 <= PMC.get_branch_kind(table)[r] <= 8 && PMC.get_branch_id(table)[r] == id
        ]
        @test !isempty(outaged_rows)
        arc_count = length(PMC.get_arc_axis(PMC.get_core(x)))
        rows = [collect(vmodf[k, mod]) for k in 1:arc_count]
        buses = PMC.get_bus_axis(x)
        zero_flows = PMC.branch_flows(table, zeros(arc_count), outaged_rows)
        kind = PMC.get_branch_kind(table)
        branch_id = PMC.get_branch_id(table)
        compared = 0
        for r in findall(<=(8), kind)
            winding = max(kind[r] - 5, 0)
            row = _by_winding(PMC.branch_ptdf_row, winding, x, branch_id[r], ctg)
            if branch_id[r] == id
                @test iszero(row)
            end
            for (j, b) in enumerate(buses)
                arc_values = Float64[rows[k][j] for k in 1:arc_count]
                flows = PMC.branch_flows(table, arc_values, outaged_rows)
                expected = flows[r] - zero_flows[r]
                @test row[PMC.bus_column(x, b)] ≈ expected atol = 1e-9
            end
            compared += 1
        end
        @test compared > 0
    end

    @testset "registered outages" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        lines = collect(PSY.get_components(PSY.Line, sys))
        _add_outage!(sys, lines[1])
        _add_outage!(sys, lines[2])
        vmodf = VirtualMODF(sys)
        original = PNM.get_registered_contingencies(vmodf)
        @test length(original) == 2
        path = joinpath(mktempdir(), "registered.h5")
        to_hdf5(vmodf, path)
        core = PMC.read_factor_core(path)
        specs = PMC.read_registered_outages(path)
        @test sort!(collect(keys(specs))) == sort!(collect(keys(original)))
        for (id, spec) in original
            @test specs[id].modification.label == spec.modification.label
            _test_core_modf_rows(vmodf, core, specs[id].modification, spec.modification)
        end
        _, _, ptdf_path = _roundtrip_ptdf(sys)
        @test isempty(PMC.read_registered_outages(ptdf_path))
    end

    @testset "islanding outage" begin
        sys = PSB.build_system(
            PSB.PSSEParsingTestSystems,
            "psse_14_network_reduction_test_system",
        )
        vmodf =
            VirtualMODF(sys; linear_solver = "KLU", automatically_register_outages = false)
        path = joinpath(mktempdir(), "island.h5")
        to_hdf5(vmodf, path)
        core = PMC.read_factor_core(path; linear_solver = "KLU")
        diag = PMC.get_PTDF_A_diag(core)
        e = first(i for i in eachindex(diag) if abs(diag[i] - 1.0) < 1e-6)
        mod = NetworkModification(
            "bridge",
            [ArcModification(e, -vmodf.arc_susceptances[e])],
            ShuntModification[],
            true,
        )
        _test_core_modf_rows(vmodf, core, mod)
    end
end

@testset "restored matrix guards" begin
    @test PNM.holds_components(PNM.get_branch_catalog(RTS_VMODF))
    @test !PNM.holds_components(PNM.get_branch_catalog(RTS_RESTORED))
    msg = "restored from a factor file"
    branch = first(keys(RTS_NR.reverse_direct_branch_map))
    @test_throws msg NetworkModification(RTS_RESTORED, branch)
    sys10 = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
    trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys10))
    @test_throws msg NetworkModification(RTS_RESTORED, trf)
    outage = _add_outage!(RTS_SYS, branch)
    @test_throws msg NetworkModification(RTS_RESTORED, RTS_SYS, outage)
    @test_throws msg PNM.get_branch_susceptances_by_arc(PNM.get_core(RTS_RESTORED))
    dir = mktempdir()
    path = joinpath(dir, "again.h5")
    @test_throws msg to_hdf5(RTS_RESTORED, path)
    @test !ispath(path)
    @test isempty(readdir(dir))
end

@testset "rejection summary" begin
    rejected = Pair{String, String}[]
    for i in 1:12, j in 1:i
        push!(rejected, "b$i$j" => "msg$(lpad(i, 2, '0'))")
    end
    lines = split(PNM._rejection_summary(rejected), "\n")
    @test length(lines) == PNM._MAX_REJECTION_LINES + 1
    @test lines[1] == "12 x msg12"
    @test lines[2] == "11 x msg11"
    @test lines[end - 1] == "3 x msg03"
    @test lines[end] == "and 2 more"
    tie = PNM._rejection_summary(["a" => "zeta", "b" => "alpha"])
    @test tie == "1 x alpha\n1 x zeta"
end

@testset "factor file outages" begin
    @testset "RTS-GMLC outage by name" begin
        for map in (
            RTS_NR.reverse_direct_branch_map,
            RTS_NR.reverse_series_branch_map,
            RTS_NR.reverse_parallel_branch_map,
        )
            branch = first(keys(map))
            key = PNM._outage_key(branch)
            @test NetworkModification(RTS_TABLE, key[1], key[2]) ==
                  NetworkModification(RTS_TABLE, [key])
            _test_modf_rows_equal(
                RTS_VMODF,
                RTS_RESTORED,
                NetworkModification(RTS_VMODF, branch),
                NetworkModification(RTS_TABLE, key...),
            )
        end
        checked = 0
        for branch in keys(RTS_NR.reverse_series_branch_map)
            key = PNM._outage_key(branch)
            haskey(PNM.get_outage_rows(RTS_TABLE), key) || continue
            @test all(
                PNM.get_outage_opens_arc(RTS_TABLE)[PNM.get_outage_rows(RTS_TABLE)[key][2]],
            )
            checked += 1
        end
        @test checked > 0
        @test_throws "has no row in the outage table" NetworkModification(
            RTS_TABLE,
            "Line",
            "no such line",
        )
        @test_throws "needs at least one branch key" NetworkModification(
            RTS_TABLE,
            Tuple{String, String}[],
        )
        dup = first(keys(PNM.get_outage_rows(RTS_TABLE)))
        @test_throws "Duplicate branch key" NetworkModification(RTS_TABLE, [dup, dup])
    end

    @testset "case10 3WT outage and skipped series chain" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf = VirtualMODF(sys; network_reductions = RTS_REDUCTIONS)
        path = joinpath(mktempdir(), "case10.h5")
        @test_logs (
            :warn,
            r"3 branches are in no reduction map of the matrix",
        ) (
            :warn,
            r"2 branches have no outage row because NetworkModification rejected them",
        ) match_mode = :any to_hdf5(vmodf, path, sys)
        restored = from_hdf5(VirtualMODF, path)
        table = from_hdf5(BranchOutageTable, path)
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        key = PNM._outage_key(trf)
        _test_modf_rows_equal(
            vmodf,
            restored,
            NetworkModification(vmodf, trf),
            NetworkModification(table, key...),
        )
        nr = PNM.get_network_reduction_data(vmodf)
        chain = [
            line for line in PSY.get_components(PSY.Line, sys) if
            haskey(nr.reverse_series_branch_map, line)
        ]
        @test length(chain) == 2
        for branch in chain
            @test !haskey(PNM.get_outage_rows(table), PNM._outage_key(branch))
        end
    end

    @testset "writer warnings and absorbed 3WT" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf = VirtualMODF(sys; network_reductions = RTS_REDUCTIONS)
        nr = PNM.get_network_reduction_data(vmodf)
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        @test !PNM._is_absorbed(nr, trf)
        PSY.set_available!(trf, false)
        @test PNM._is_absorbed(nr, trf)
        path = joinpath(mktempdir(), "absorbed.h5")
        to_hdf5(vmodf, path, sys)
        table = from_hdf5(BranchOutageTable, path)
        @test !haskey(PNM.get_outage_rows(table), PNM._outage_key(trf))
        # The warning gives the count, the first branch and each distinct message with a count.
        @test_logs (:warn, r"in no reduction map") (
            :warn,
            r"2 branches have no outage row because NetworkModification rejected them, or because the outage leaves a series chain partly in service\. First: .*\n2 x The branch outage table has no row for a branch that leaves its series chain partly in service\.",
        ) match_mode = :any to_hdf5(vmodf, joinpath(mktempdir(), "w.h5"), sys)
    end

    @testset "zero-susceptance branch has no row" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        line = first(PSY.get_components(PSY.Line, sys))
        PSY.set_r!(line, 0.01u"SU")
        PSY.set_x!(line, 0.0u"SU")
        vptdf = VirtualPTDF(sys)
        path = joinpath(mktempdir(), "zero_b.h5")
        @test_logs (:warn, Regex("susceptance is zero.*$(PSY.get_name(line))")) match_mode =
            :any to_hdf5(vptdf, path, sys)
        table = from_hdf5(BranchOutageTable, path)
        @test !haskey(PNM.get_outage_rows(table), PNM._outage_key(line))
        for other in PSY.get_components(PSY.Line, sys)
            other === line && continue
            key = PNM._outage_key(other)
            @test haskey(PNM.get_outage_rows(table), key)
            _test_arc_rows_equal(
                NetworkModification(vptdf, other),
                NetworkModification(table, [key]),
            )
        end
    end

    @testset "mixed zero-susceptance arcs drop only the zero arc rows" begin
        # A matrix with a zero-susceptance 3WT winding fails the core validation (the star
        # bus is cut off), so the mixed case is tested on the helper the writer calls.
        z = zero(PNM.YBUS_ELTYPE)
        mixed = NetworkModification(
            "m",
            [
                ArcModification(1, -1.0, 0.0, z, z, z, z, 1),
                ArcModification(2, 0.0, 0.0, z, z, z, z, 1),
                ArcModification(3, -2.0, 0.0, z, z, z, z, 1),
            ],
        )
        kept = PNM._drop_zero_susceptance_arcs(mixed)
        @test [m.arc_index for m in kept.arc_modifications] == [1, 3]
        @test PNM._drop_zero_susceptance_arcs(kept) === kept
        all_zero = NetworkModification("z", [ArcModification(2, 0.0, 0.0, z, z, z, z, 1)])
        @test isempty(PNM._drop_zero_susceptance_arcs(all_zero).arc_modifications)
    end

    @testset "VirtualMODF registers a partial series outage" begin
        reductions = NetworkReduction[DegreeTwoReduction()]
        sys = build_antiparallel_chain_segment_system()
        vmodf = VirtualMODF(sys; network_reductions = reductions)
        # The outage comes after the reduction, so the reduction folded P1 into a chain.
        p1 = PSY.get_component(PSY.Line, sys, "L_10_3")
        outage = _add_outage!(sys, p1)
        @test_logs (:info, r"Registered 1 contingencies") PNM._register_all_outages!(
            vmodf,
            sys,
        )
        mod = NetworkModification(vmodf, sys, outage)
        rebuilt_sys = build_antiparallel_chain_segment_system()
        PSY.set_available!(PSY.get_component(PSY.Line, rebuilt_sys, "L_10_3"), false)
        rebuilt = VirtualPTDF(rebuilt_sys; network_reductions = reductions)
        arcs = PNM.get_arc_axis(vmodf)
        @test arcs == PNM.get_arc_axis(rebuilt)
        monitored = first(i for i in eachindex(arcs) if arcs[i] == (1, 3))
        @test isapprox(
            collect(vmodf[monitored, mod]),
            collect(PNM.get_ptdf_row(rebuilt, arcs[monitored]));
            atol = 1e-10,
        )
    end

    # Disabled until the SeriesCompensator `bypassed` schema reaches PowerSystems psy6.
    #=
    @testset "SeriesCompensator system round trip" begin
        sys, _ = compensated_sys5(-0.01)
        vptdf, restored, _ = _roundtrip_ptdf(sys)
        _test_same_core(vptdf, restored)
        _test_ptdf_rows_equal(vptdf, restored, PNM.get_arc_axis(vptdf))
        vmodf = VirtualMODF(sys)
        restored_modf, table = _outage_roundtrip(vmodf, sys)
        _test_same_core(vmodf, restored_modf)
        line = get_component(Line, sys, COMPENSATED_LINE)
        compensator = first(PSY.get_components(PSY.SeriesCompensator, sys))
        for branch in (line, compensator)
            key = PNM._outage_key(branch)
            @test haskey(PNM.get_outage_rows(table), key)
            _test_modf_rows_equal(
                vmodf,
                restored_modf,
                NetworkModification(vmodf, branch),
                NetworkModification(table, key...),
            )
        end
    end
    =#

    @testset "case10 branch outage keys are unique" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf = VirtualMODF(sys; network_reductions = RTS_REDUCTIONS)
        path = joinpath(mktempdir(), "case10_keys.h5")
        to_hdf5(vmodf, path, sys)
        types, names = HDF5.h5open(path, "r") do f
            read(f["branch_outages/type"]), read(f["branch_outages/name"])
        end
        keys_ = collect(zip(types, names))
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        @test allunique(keys_)
        @test count(==(PNM._outage_key(trf)), keys_) == 1
    end

    @testset "writer skips only UnsupportedOutageError" begin
        @test PNM._skippable(PNM.UnsupportedOutageError("x"))
        @test !PNM._skippable(ErrorException("x"))
        @test !PNM._skippable(MethodError(sin, ()))
        @test !PNM._skippable(InterruptException())
    end

    @testset "wrong system" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        other = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        vptdf = VirtualPTDF(sys)
        wrong = joinpath(mktempdir(), "wrong.h5")
        @test_throws "System UUID mismatch" to_hdf5(vptdf, wrong, other)
        @test !ispath(wrong)
    end

    @testset "directory target is never removed" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        vptdf = VirtualPTDF(sys)
        dir = mktempdir()
        keep = joinpath(dir, "keep.txt")
        write(keep, "x")
        for force in (false, true)
            @test_throws "not a regular file" to_hdf5(vptdf, dir; force = force)
            @test isfile(keep)
        end
    end

    @testset "multi-branch outage" begin
        by_arc = Dict{Tuple{Int, Int}, Vector{PSY.ACTransmission}}()
        for (branch, arc) in RTS_NR.reverse_series_branch_map
            push!(get!(by_arc, arc, PSY.ACTransmission[]), branch)
        end
        chain = first(v for v in values(by_arc) if length(v) >= 2)
        direct = first(keys(RTS_NR.reverse_direct_branch_map))
        for pair in (chain[1:2], [chain[1], direct])
            sys_outage = _add_outage!(RTS_SYS, pair...)
            oracle = NetworkModification(RTS_VMODF, RTS_SYS, sys_outage)
            actual = NetworkModification(RTS_TABLE, [PNM._outage_key(b) for b in pair])
            @test length(actual.arc_modifications) == length(oracle.arc_modifications)
            _test_modf_rows_equal(RTS_VMODF, RTS_RESTORED, oracle, actual)
        end
        chain_mod =
            NetworkModification(RTS_TABLE, [PNM._outage_key(b) for b in chain[1:2]])
        @test length(chain_mod.arc_modifications) == 1
    end

    @testset "series chain with a parallel segment" begin
        sys = build_antiparallel_chain_segment_system()
        vmodf =
            VirtualMODF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
        nr = PNM.get_network_reduction_data(vmodf)
        p1 = PSY.get_component(PSY.Line, sys, "L_10_3")
        p2 = PSY.get_component(PSY.Line, sys, "L_3_10")
        l1 = PSY.get_component(PSY.Line, sys, "L_1_10")
        for branch in (p1, p2, l1)
            @test PNM._resolve_branch_arc(nr, branch) == (:series, (1, 3))
        end
        segments = collect(nr.series_branch_map[(1, 3)])
        @test length(segments) == 2
        @test l1 in segments
        @test !(p1 in segments)
        @test !(p2 in segments)
        # A partial series outage keeps the arc in service, so the writer stores no row for P1 or P2.
        @test iszero(only(NetworkModification(vmodf, p1).arc_modifications).opened)
        @test iszero(only(NetworkModification(vmodf, p2).arc_modifications).opened)

        restored, table = _outage_roundtrip(vmodf, sys)
        key = PNM._outage_key
        @test !haskey(PNM.get_outage_rows(table), key(p1))
        @test !haskey(PNM.get_outage_rows(table), key(p2))
        @test PNM.get_outage_opens_arc(table)[only(PNM.get_outage_rows(table)[key(l1)][2])]
        oracle = NetworkModification(vmodf, sys, _add_outage!(sys, l1))
        actual = NetworkModification(table, [key(l1)])
        _test_arc_rows_equal(oracle, actual)
        _test_post_outage_rows_equal(vmodf, restored, oracle, actual)
    end

    @testset "combine rule" begin
        # Arc 1: a partial row and a row that opens it. Arc 2: two partial rows of parallel
        # members.
        rows = Dict(
            ("Line", "a1") => (false, 1:1),
            ("Line", "a2") => (false, 2:2),
            ("Line", "p1") => (false, 3:3),
            ("Line", "p2") => (false, 4:4),
        )
        table = BranchOutageTable(
            rows,
            [1, 1, 2, 2],
            [-0.5, -4.0, -1.0, -2.0],
            zeros(4),
            ones(Int, 4),
            [false, true, false, false],
            [1, 0, 1, 2],
            Dict(i => k for (i, k) in enumerate(keys(rows))),
            2,
        )
        opened = NetworkModification(table, [("Line", "a1"), ("Line", "a2")])
        @test only(opened.arc_modifications).delta_b == -4.0
        parallel = NetworkModification(table, [("Line", "p1"), ("Line", "p2")])
        @test only(parallel.arc_modifications).delta_b == -3.0
        # Two rows of one group member remove that member one time.
        same_member = BranchOutageTable(
            Dict(("Line", "c1") => (false, 1:1), ("Line", "c2") => (false, 2:2)),
            [1, 1],
            [-1.0, -1.0],
            zeros(2),
            ones(Int, 2),
            [false, false],
            [1, 1],
            Dict(1 => ("Line", "c1"), 2 => ("Line", "c2")),
            1,
        )
        chain_mod = NetworkModification(same_member, [("Line", "c1"), ("Line", "c2")])
        @test only(chain_mod.arc_modifications).delta_b == -1.0
    end

    @testset "two lines of one chain in a parallel group" begin
        sys = build_two_parallel_degree_two_chains()
        reductions =
            NetworkReduction[DegreeTwoReduction(; reduce_reactive_power_injectors = false)]
        vmodf = VirtualMODF(sys; network_reductions = reductions)
        nr = PNM.get_network_reduction_data(vmodf)
        a = PSY.get_component(PSY.Line, sys, "L_1_10")
        b = PSY.get_component(PSY.Line, sys, "L_10_11")
        @test PNM._resolve_branch_arc(nr, a) == (:parallel, (1, 3))
        @test PNM._resolve_branch_arc(nr, b) == (:parallel, (1, 3))
        bp = PNM.get_parallel_branch_map(nr)[(1, 3)]
        b_chain = PNM._finite_series_susceptance(PNM._outaged_group_member(bp, a), nr)

        oracle = NetworkModification(vmodf, sys, _add_outage!(sys, a, b))
        @test only(oracle.arc_modifications).delta_b ≈ -b_chain
        @test oracle == NetworkModification(vmodf, a)

        restored, table = _outage_roundtrip(vmodf, sys)
        actual = NetworkModification(table, PNM._outage_key.([a, b]))
        @test only(actual.arc_modifications).delta_b ≈ -b_chain
        _test_arc_rows_equal(oracle, actual)
        _test_post_outage_rows_equal(vmodf, restored, oracle, actual)
    end

    @testset "case10 3WT circuit and a line on one chain" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        vmodf =
            VirtualMODF(sys; network_reductions = NetworkReduction[DegreeTwoReduction()])
        nr = PNM.get_network_reduction_data(vmodf)
        trf = first(PSY.get_components(PSY.ThreeWindingTransformer, sys))
        line = PSY.get_component(PSY.Line, sys, "RADIAL1-RADIAL2-i_1")
        winding = PNM.ThreeWindingTransformerCircuit(trf, PSY.get_circuits(trf)[2], 2)
        @test PNM._resolve_branch_arc(nr, winding) == (:series, (110, 1001))
        @test PNM._resolve_branch_arc(nr, line) == (:series, (110, 1001))
        @test line in collect(nr.series_branch_map[(110, 1001)])
        restored, table = _outage_roundtrip(vmodf, sys)
        oracle = NetworkModification(vmodf, sys, _add_outage!(sys, trf, line))
        actual = NetworkModification(table, PNM._outage_key.([trf, line]))
        _test_arc_rows_equal(oracle, actual)
        _test_post_outage_rows_equal(vmodf, restored, oracle, actual)
    end

    @testset "registered outages round trip" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        lines = collect(PSY.get_components(PSY.Line, sys))
        _add_outage!(sys, lines[1])
        _add_outage!(sys, lines[2])
        vmodf = VirtualMODF(sys)
        restored, _ = _outage_roundtrip(vmodf, sys)
        original = PNM.get_registered_contingencies(vmodf)
        copies = PNM.get_registered_contingencies(restored)
        @test length(original) == 2
        @test sort(collect(keys(copies))) == sort(collect(keys(original)))
        for (id, spec) in original
            @test copies[id].modification.label == spec.modification.label
            _test_modf_rows_equal(
                vmodf,
                restored,
                spec.modification,
                copies[id].modification,
            )
        end
        bare = _modf_roundtrip(vmodf)
        bare_copies = PNM.get_registered_contingencies(bare)
        @test sort(collect(keys(bare_copies))) == sort(collect(keys(original)))
        for (id, spec) in original
            @test bare_copies[id].modification.label == spec.modification.label
            _test_modf_rows_equal(
                vmodf,
                bare,
                spec.modification,
                bare_copies[id].modification,
            )
        end
    end

    @testset "failed write keeps the old file" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        other = PSB.build_system(PSB.PSITestSystems, "c_sys5")
        vptdf = VirtualPTDF(sys)
        dir = mktempdir()
        path = joinpath(dir, "keep.h5")
        to_hdf5(vptdf, path)
        listing = readdir(dir)
        @test_throws "System UUID mismatch" to_hdf5(vptdf, path, other; force = true)
        _test_same_core(vptdf, from_hdf5(VirtualPTDF, path))
        @test readdir(dir) == listing
        @test_throws "write failed" PNM._write_factor_file_atomically(path, true) do f
            f["partial"] = [1, 2, 3]
            error("write failed")
        end
        _test_same_core(vptdf, from_hdf5(VirtualPTDF, path))
        @test readdir(dir) == listing
        @test_throws "exists. Pass `force = true`" to_hdf5(vptdf, path)
    end

    @testset "no overwrite when a file appears during the write" begin
        dir = mktempdir()
        path = joinpath(dir, "race.h5")
        @test_throws "exists. Pass `force = true`" PNM._write_factor_file_atomically(
            path,
            false,
        ) do f
            f["x"] = [1]
            write(path, "racing")
        end
        @test read(path, String) == "racing"
        @test readdir(dir) == ["race.h5"]
    end

    @testset "phase shifter outage" begin
        sys = _mk_line_pst_parallel_system()
        vmodf = VirtualMODF(sys)
        restored, table = _outage_roundtrip(vmodf, sys)
        pst = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")
        key = PNM._outage_key(pst)
        @test any(!iszero, table.delta_shift[PNM.get_outage_rows(table)[key][2]])
        _test_modf_rows_equal(
            vmodf,
            restored,
            NetworkModification(vmodf, pst),
            NetworkModification(table, key...),
        )
    end

    @testset "file without system" begin
        sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
        _, _, path = _roundtrip_ptdf(sys)
        @test_throws "has no branch outage table" from_hdf5(BranchOutageTable, path)
    end
end

@testset "factor file rejects bad values" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    _add_outage!(sys, first(PSY.get_components(PSY.Line, sys)))
    vmodf = VirtualMODF(sys)
    n_arc = length(PNM.get_arc_axis(vmodf))

    function fresh_path()
        path = joinpath(mktempdir(), "bad.h5")
        to_hdf5(vmodf, path, sys)
        return path
    end

    for (group, type) in (("branch_outages", BranchOutageTable), ("outages", VirtualMODF))
        for bad_index in (0, n_arc + 1)
            path = fresh_path()
            good = HDF5.h5open(f -> read(f[group]["arc_index"]), path, "r")
            bad = copy(good)
            bad[1] = bad_index
            _write_bad_dataset(path, group, "arc_index", bad)
            @test_throws "$group/arc_index has the value $bad_index at row 1" from_hdf5(
                type,
                path,
            )
        end
        for bad_b in (0.0, NaN)
            path = fresh_path()
            good = HDF5.h5open(f -> read(f[group]["delta_b"]), path, "r")
            bad = copy(good)
            bad[1] = bad_b
            _write_bad_dataset(path, group, "delta_b", bad)
            @test_throws "$group/delta_b has the value $bad_b in arc row 1" from_hdf5(
                type,
                path,
            )
        end
    end

    path = fresh_path()
    _write_bad_dataset(path, "branch_outages", "opens_arc", Bool[])
    @test_throws "columns of unequal length" from_hdf5(BranchOutageTable, path)

    path = fresh_path()
    _write_bad_dataset(path, "branch_outages", "member", Int[])
    @test_throws "columns of unequal length" from_hdf5(BranchOutageTable, path)

    path = fresh_path()
    n_rows = HDF5.h5open(f -> length(read(f["branch_outages"]["member"])), path, "r")
    _write_bad_dataset(path, "branch_outages", "opens_arc", fill(false, n_rows))
    _write_bad_dataset(path, "branch_outages", "member", zeros(Int, n_rows))
    @test_throws "branch_outages/member has the value 0 at row 1" from_hdf5(
        BranchOutageTable,
        path,
    )

    vptdf = VirtualPTDF(sys)
    n_bus = length(PNM.get_bus_axis(vptdf))
    for len in (n_bus - 1, n_bus + 1)
        path = joinpath(mktempdir(), "slack.h5")
        to_hdf5(vptdf, path)
        _write_bad_dataset(path, "", "dist_slack", fill(1.0, len))
        @test_throws "dist_slack has columns of unequal length" from_hdf5(VirtualPTDF, path)
    end

    @testset "absolute tol must be finite and not negative" begin
        vptdf5 = VirtualPTDF(sys)
        for bad in (Inf, NaN, -1.0)
            path = joinpath(mktempdir(), "tol.h5")
            to_hdf5(vptdf5, path)
            HDF5.h5open(path, "r+") do f
                a = HDF5.attributes(f)
                HDF5.delete_attribute(f, "tol_kind")
                a["tol_kind"] = "absolute"
                haskey(a, "tol_value") && HDF5.delete_attribute(f, "tol_value")
                a["tol_value"] = bad
            end
            @test_throws "tol_value is $bad; expected a finite value >= 0" from_hdf5(
                VirtualPTDF,
                path,
            )
        end
    end

    @testset "dist_slack sum must be finite" begin
        path = joinpath(mktempdir(), "slack.h5")
        to_hdf5(VirtualPTDF(sys), path)
        _write_bad_dataset(path, "", "dist_slack", [1e308, 1e308, 0.0, 0.0, 0.0])
        @test_throws "the sum must be finite and positive" from_hdf5(VirtualPTDF, path)
    end

    @testset "repeated keys and bad row ranges" begin
        path = fresh_path()
        HDF5.h5open(path, "r+") do f
            g = f["branch_outages"]
            t = read(g["type"])
            nm = read(g["name"])
            t[2] = t[1]
            nm[2] = nm[1]
            HDF5.delete_object(g, "type")
            HDF5.delete_object(g, "name")
            PNM._write_strings!(g, "type", t)
            PNM._write_strings!(g, "name", nm)
        end
        @test_throws "more than once" from_hdf5(BranchOutageTable, path)

        for (name, index, value, message) in (
            ("len", 1, -3, "len has the value -3 at row 1; the value must not be negative"),
            ("start", 1, 0, "start has the value 0 at row 1; expected 1 or more"),
            ("len", 1, typemax(Int), "len has the value $(typemax(Int)) at row 1"),
        )
            path = fresh_path()
            col = HDF5.h5open(f -> read(f["branch_outages"][name]), path, "r")
            col[index] = value
            _write_bad_dataset(path, "branch_outages", name, col)
            @test_throws message from_hdf5(BranchOutageTable, path)
        end
    end

    @testset "outage rows: ranges, repeats, size, zero arcs, version, type" begin
        function write_outages!(
            path; start, len, arc_index, delta_b, id = collect(1:length(start)),
        )
            n = length(arc_index)
            HDF5.h5open(path, "r+") do f
                HDF5.delete_object(f, "outages")
                g = HDF5.create_group(f, "outages")
                g["start"] = start
                g["len"] = len
                g["arc_index"] = arc_index
                g["delta_b"] = delta_b
                g["delta_shift"] = zeros(n)
                g["opened"] = ones(Int, n)
                g["id"] = id
                PNM._write_strings!(g, "label", ["o$i" for i in 1:length(start)])
                g["is_islanding"] = fill(false, length(start))
            end
            return
        end

        path = fresh_path()
        write_outages!(path; start = [1, 2], len = [1, 1], arc_index = [1, 2],
            delta_b = [-1.0, -1.0], id = [7, 7])
        @test_throws "outages/id has the value 7 more than once" from_hdf5(
            VirtualMODF,
            path,
        )

        path = fresh_path()
        write_outages!(path; start = [1, 1], len = [2, 1], arc_index = [1, 2, 3],
            delta_b = [-1.0, -1.0, -1.0])
        @test_throws "must be contiguous and ordered" from_hdf5(VirtualMODF, path)
        path = fresh_path()
        write_outages!(path; start = [1, 3], len = [1, 1], arc_index = [1, 2, 3],
            delta_b = [-1.0, -1.0, -1.0])
        @test_throws "must be contiguous and ordered" from_hdf5(VirtualMODF, path)
        path = fresh_path()
        write_outages!(path; start = [1, 2], len = [1, 1], arc_index = [1, 2, 3],
            delta_b = [-1.0, -1.0, -1.0])
        @test_throws "cover 2 of the 3" from_hdf5(VirtualMODF, path)

        for (name, index, value) in (("start", 2, 1), ("len", 1, 0))
            path = fresh_path()
            col = HDF5.h5open(f -> read(f["branch_outages"][name]), path, "r")
            col[index] = value
            _write_bad_dataset(path, "branch_outages", name, col)
            @test_throws "contiguous and ordered" from_hdf5(BranchOutageTable, path)
        end

        path = fresh_path()
        write_outages!(path; start = [1], len = [2], arc_index = [1, 1],
            delta_b = [-1.0, 1.0])
        @test_throws "has the arc_index 1 more than once" from_hdf5(VirtualMODF, path)

        path = fresh_path()
        write_outages!(path; start = [1], len = [1], arc_index = [1], delta_b = [2e12])
        @test_throws "delta_b has the value 2.0e12 in arc row 1" from_hdf5(
            VirtualMODF,
            path,
        )

        for type in (BranchOutageTable, VirtualMODF)
            path = fresh_path()
            HDF5.h5open(path, "r+") do f
                b = read(f["arc_susceptances"])
                b[1] = 0.0
                HDF5.delete_object(f, "arc_susceptances")
                f["arc_susceptances"] = b
            end
            if type === VirtualMODF
                write_outages!(path; start = [1], len = [1], arc_index = [1],
                    delta_b = [-1.0])
            else
                ai = HDF5.h5open(f -> read(f["branch_outages"]["arc_index"]), path, "r")
                ai[1] = 1
                _write_bad_dataset(path, "branch_outages", "arc_index", ai)
            end
            @test_throws "arc_susceptances entry is zero" from_hdf5(type, path)
        end

        path = fresh_path()
        HDF5.h5open(path, "r+") do f
            HDF5.delete_attribute(f, "format_version")
            HDF5.write_attribute(f, "format_version", PMC.FACTOR_FILE_FORMAT_VERSION + 1)
        end
        @test_throws "Unsupported factor file format version" from_hdf5(
            BranchOutageTable,
            path,
        )

        path = fresh_path()
        types = HDF5.h5open(f -> read(f["branch_outages"]["type"]), path, "r")
        types[1] = "eval"
        _write_bad_dataset(path, "branch_outages", "type", types)
        @test_throws "unknown branch type" from_hdf5(BranchOutageTable, path)
    end

    @testset "kernel rejects out-of-range arc index" begin
        mod = NetworkModification(
            "bad",
            [ArcModification(n_arc + 1, 1.0, 0.0, 0, 0, 0, 0, 1)],
        )
        @test_throws ArgumentError get_post_modification_ptdf_row(VirtualPTDF(sys), 1, mod)
    end

    @testset "dataset where a group belongs" begin
        H = HDF5
        key = H.h5open(f -> first(keys(f["subnetworks"])), fresh_path(), "r")
        for (parent_path, name, entry) in (
            ("", "names", "/names"),
            ("", "subnetworks", "/subnetworks"),
            ("subnetworks", key, "/subnetworks/$key"),
            ("", "outages", "/outages"),
        )
            path = fresh_path()
            _write_bad_dataset(path, parent_path, name, [1, 2])
            @test_throws "entry $entry is a dataset; expected a group" from_hdf5(
                VirtualMODF,
                path,
            )
        end
        path = fresh_path()
        _write_bad_dataset(path, "", "branch_outages", [1, 2])
        @test_throws "entry /branch_outages is a dataset; expected a group" from_hdf5(
            BranchOutageTable,
            path,
        )
    end
end

@testset "factor file checks every object before it reads" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys14")
    vptdf = VirtualPTDF(sys)
    H = HDF5

    function fresh_path()
        path = joinpath(mktempdir(), "check.h5")
        to_hdf5(vptdf, path)
        return path
    end

    function mutate!(f, path)
        H.h5open(f, path, "r+")
        return
    end

    @testset "external link" begin
        path = fresh_path()
        other = joinpath(mktempdir(), "other.h5")
        H.h5open(g -> g["arc_from"] = [1, 2], other, "w")
        mutate!(path) do f
            H.delete_object(f, "arc_from")
            H.create_external(f, "arc_from", other, "arc_from")
        end
        @test_throws "link arc_from is not a hard link" from_hdf5(VirtualPTDF, path)
    end

    @testset "external file list" begin
        path = fresh_path()
        n = length(PNM.get_arc_axis(vptdf))
        other = joinpath(mktempdir(), "data.bin")
        write(other, zeros(n))
        mutate!(path) do f
            H.delete_object(f, "arc_susceptances")
            H.create_dataset(
                f,
                "arc_susceptances",
                Float64,
                (n,);
                external = (other, 0, n * 8),
            )
        end
        @test_throws "arc_susceptances names an external file" from_hdf5(VirtualPTDF, path)
    end

    @testset "virtual layout" begin
        path = fresh_path()
        n = length(PNM.get_arc_axis(vptdf))
        mutate!(path) do f
            H.delete_object(f, "arc_susceptances")
            space = H.dataspace((n,))
            dcpl = H.create_property(H.API.H5P_DATASET_CREATE)
            H.API.h5p_set_virtual(dcpl, space, ".", "arc_from", space)
            H.API.h5d_create(
                f,
                "arc_susceptances",
                H.datatype(Float64),
                space,
                H.API.H5P_DEFAULT,
                dcpl,
                H.API.H5P_DEFAULT,
            )
        end
        @test_throws "arc_susceptances has the layout virtual" from_hdf5(VirtualPTDF, path)
    end

    @testset "soft link" begin
        path = fresh_path()
        mutate!(path) do f
            H.API.h5l_create_soft(
                "bus_axis",
                f,
                "alias",
                H.API.H5P_DEFAULT,
                H.API.H5P_DEFAULT,
            )
        end
        @test_throws "link alias is not a hard link" from_hdf5(VirtualPTDF, path)
    end

    @testset "declared size of 2^40 elements" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_object(f, "arc_from")
            H.create_dataset(f, "arc_from", Int, (2^40,); chunk = (1024,))
        end
        H.h5open(path, "r") do f
            @test_throws "declares 1099511627776 elements" PNM._check_before_read(
                f["arc_from"],
                "arc_from",
                Vector{Int},
                PNM.ReadBudget(filesize(path)),
            )
        end
    end

    function fixed_type(width)
        dt = H.Datatype(H.API.h5t_copy(H.API.H5T_C_S1))
        H.API.h5t_set_size(dt, width)
        return dt
    end

    @testset "declared element size beyond the file size" begin
        path = fresh_path()
        mutate!(path) do f
            H.create_dataset(f, "wide", fixed_type(2^26), H.dataspace((16,)); chunk = (1,))
        end
        H.h5open(path, "r") do f
            @test_throws "string of 67108864 bytes per element" PNM._check_before_read(
                f["wide"],
                "wide",
                Vector{String},
                PNM.ReadBudget(filesize(path)),
            )
        end
    end

    @testset "string element beyond the string cap" begin
        @test_throws "string of 1073741824 bytes" PNM._check_stored_type(
            String,
            "system_uuid",
            (Int(H.API.H5T_STRING), 2^30, false),
        )
    end

    @testset "number entry declared as a wide string" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_object(f, "bus_axis")
            H.create_dataset(
                f,
                "bus_axis",
                fixed_type(2^30),
                H.dataspace((1,));
                chunk = (1,),
            )
        end
        H.h5open(path, "r") do f
            @test_throws "bus_axis has the HDF5 class 3" PNM._check_before_read(
                f["bus_axis"],
                "bus_axis",
                Vector{Int},
                PNM.ReadBudget(filesize(path)),
            )
        end
    end

    @testset "number entry of the wrong size" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_object(f, "bus_axis")
            f["bus_axis"] = Int32[1, 2, 3]
        end
        @test_throws "bus_axis has the HDF5 class 0 and 4 bytes" from_hdf5(
            VirtualPTDF,
            path,
        )
    end

    @testset "variable-length string column" begin
        path = fresh_path()
        mutate!(path) do f
            g = f["names"]
            n = length(H.read(g["from"]))
            H.delete_object(g, "name")
            g["name"] = fill("x", n)
        end
        @test_throws "name is a variable-length string" from_hdf5(
            VirtualPTDF,
            path,
        )
    end

    @testset "string column beyond the string cap" begin
        path = fresh_path()
        mutate!(path) do f
            g = f["names"]
            n = length(H.read(g["from"]))
            H.delete_object(g, "name")
            H.create_dataset(
                g,
                "name",
                fixed_type(PNM.MAX_FIXED_STRING_BYTES + 1),
                H.dataspace((n,));
                chunk = (1,),
            )
        end
        @test_throws "string of 65537 bytes per element" from_hdf5(VirtualPTDF, path)
    end

    @testset "writer rejects a string beyond the string cap" begin
        path = fresh_path()
        H.h5open(path, "w") do f
            @test_throws "the limit is 65536" PNM._write_strings!(
                f,
                "x",
                [repeat("a", PNM.MAX_FIXED_STRING_BYTES + 1)],
            )
        end
    end

    @testset "writer rejects a string with a null byte" begin
        H.h5open(joinpath(mktempdir(), "nul.h5"), "w") do f
            @test_throws "holds a string with a null byte" PNM._write_strings!(
                f,
                "x",
                ["a\0"],
            )
        end
    end

    @testset "group in place of a dataset" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_object(f, "arc_from")
            H.create_group(f, "arc_from")
        end
        @test_throws "entry arc_from is a HDF5.Group" from_hdf5(VirtualPTDF, path)
    end

    @testset "fixed-length strings round trip exactly" begin
        path = fresh_path()
        names = ["bus one", "héllo wörld ✓", "", "a"]
        H.h5open(path, "w") do f
            PNM._write_strings!(f, "s", names)
            PNM._write_string_attribute!(H.attributes(f), "a", "naïve uuid")
            PNM._write_strings!(f, "empty", String[])
        end
        H.h5open(path, "r") do f
            budget = PNM.ReadBudget(filesize(path))
            @test PNM._read(f, "s", Vector{String}, budget) == names
            @test PNM._read(f, "empty", Vector{String}, budget) == String[]
            @test PNM._read(H.attributes(f), "a", String, budget) == "naïve uuid"
            dt = H.datatype(f["s"])
            @test !H.API.h5t_is_variable_str(dt)
            close(dt)
        end
    end

    @testset "system_uuid round trip" begin
        uuid = Base.UUID(0xffeeddccbbaa99887766554433221100)
        path = fresh_path()
        mutate!(path) do f
            H.delete_attribute(f, "system_uuid")
            PNM._write_system_uuid!(H.attributes(f), uuid)
        end
        @test H.h5open(f -> H.read_attribute(f, "system_uuid"), path, "r") ==
              reinterpret(Int64, [0xffeeddccbbaa9988, 0x7766554433221100])
        @test PNM.get_system_uuid(from_hdf5(VirtualPTDF, path)) == uuid
    end

    @testset "no system_uuid" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_attribute(f, "system_uuid")
        end
        @test isnothing(PNM.get_system_uuid(from_hdf5(VirtualPTDF, path)))
    end

    @testset "malformed system_uuid" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_attribute(f, "system_uuid")
            H.attributes(f)["system_uuid"] = Int64[1, 2, 3]
        end
        @test_throws "system_uuid has 3 values; expected 2" from_hdf5(VirtualPTDF, path)
        mutate!(path) do f
            H.delete_attribute(f, "system_uuid")
            PNM._write_string_attribute!(H.attributes(f), "system_uuid", "not-a-uuid")
        end
        @test_throws "system_uuid has the HDF5 class 3 and 10 bytes" from_hdf5(
            VirtualPTDF,
            path,
        )
    end

    @testset "declared bytes of all objects share one budget" begin
        path = fresh_path()
        n = (filesize(path) ÷ 8) * 6 ÷ 10
        mutate!(path) do f
            for i in 1:2
                H.create_dataset(f, "big$i", Int, (n,); chunk = (min(n, 1024),))
            end
        end
        H.h5open(path, "r") do f
            budget = PNM.ReadBudget(filesize(path))
            @test PNM._check_before_read(f["big1"], "big1", Vector{Int}, budget) == 8
            @test_throws "remaining read budget" PNM._check_before_read(
                f["big2"],
                "big2",
                Vector{Int},
                budget,
            )
        end
        budget = PNM.ReadBudget(100)
        @test_throws "declares" PNM._charge!(budget, "x", typemax(Int) ÷ 2, 4)
        @test_throws "declares 16 elements of 67108864 bytes" PNM._charge!(
            budget,
            "x",
            16,
            2^26,
        )
    end

    @testset "hard link failure message" begin
        msg = PNM._hardlink_failure_message("a.h5", ErrorException("boom"))
        @test occursin("boom", msg)
        @test occursin("hard links", msg)
        @test occursin("force = true", msg)
    end

    @testset "compound dataset" begin
        path = fresh_path()
        mutate!(path) do f
            H.delete_object(f, "arc_from")
            f["arc_from"] = [(a = 1, b = 2.0)]
        end
        @test_throws "entry arc_from has the HDF5 class 6" from_hdf5(VirtualPTDF, path)
    end

    @testset "filtered dataset" begin
        path = fresh_path()
        n = length(PNM.get_arc_axis(vptdf))
        mutate!(path) do f
            H.delete_object(f, "arc_susceptances")
            ds = H.create_dataset(
                f,
                "arc_susceptances",
                Float64,
                (n,);
                chunk = (n,),
                deflate = 3,
            )
            ds[:] = ones(n)
        end
        @test_throws "arc_susceptances has a filter" from_hdf5(VirtualPTDF, path)
    end

    @testset "rank 2 bus axis" begin
        path = fresh_path()
        _write_bad_dataset(path, "", "bus_axis", [1 2; 3 4])
        @test_throws "bus_axis has rank 2" from_hdf5(VirtualPTDF, path)
    end

    @testset "more subnetworks than buses" begin
        path = fresh_path()
        n_bus = length(PNM.get_bus_axis(vptdf))
        mutate!(path) do f
            for i in 1:(n_bus + 1)
                H.create_group(f["subnetworks"], "extra$i")
            end
        end
        @test_throws "subnetworks and only" from_hdf5(VirtualPTDF, path)
    end

    @testset "inconsistent network arrays" begin
        path = fresh_path()
        _write_bad_dataset(path, "", "ref_bus_positions", Int[])
        @test_throws "reference bus positions for" from_hdf5(VirtualPTDF, path)
        path = fresh_path()
        _write_bad_dataset(path, "", "ref_bus_positions", [1, 1])
        @test_throws "more than once" from_hdf5(VirtualPTDF, path)
        n_arc = length(PNM.get_arc_axis(vptdf))
        path = fresh_path()
        _write_bad_dataset(path, "", "arc_live_members", fill(-1, n_arc))
        @test_throws "below 1" from_hdf5(VirtualPTDF, path)
        path = fresh_path()
        _write_bad_dataset(path, "", "arc_susceptances", [1e308; fill(1.0, n_arc - 1)])
        @test_throws "at position 1 is 1.0e308" from_hdf5(VirtualPTDF, path)
        path = fresh_path()
        arc_from = H.h5open(f -> read(f["arc_from"]), path, "r")
        arc_to = H.h5open(f -> read(f["arc_to"]), path, "r")
        arc_to[1] = arc_from[1]
        _write_bad_dataset(path, "", "arc_to", arc_to)
        # The names check runs first, so point the entries on the old arc at the new one.
        name_to = H.h5open(f -> read(f["names"]["to"]), path, "r")
        name_from = H.h5open(f -> read(f["names"]["from"]), path, "r")
        for i in eachindex(name_to)
            (name_from[i], name_to[i]) == (arc_from[1], PNM.get_arc_axis(vptdf)[1][2]) &&
                (name_to[i] = arc_from[1])
        end
        _write_bad_dataset(path, "names", "to", name_to)
        @test_throws "is a self-loop" from_hdf5(VirtualPTDF, path)
        path = fresh_path()
        bus_ax = PNM.get_bus_axis(vptdf)
        _write_bad_dataset(path, "", "bus_axis", [bus_ax[1]; bus_ax[1]; bus_ax[3:end]])
        @test_throws "bus axis has the entry" from_hdf5(VirtualPTDF, path)
        path = fresh_path()
        names = H.h5open(f -> read(f["names"]["name"]), path, "r")
        types = H.h5open(f -> read(f["names"]["type"]), path, "r")
        names[2] = names[1]
        types[2] = types[1]
        _write_bad_dataset(path, "names", "name", names)
        _write_bad_dataset(path, "names", "type", types)
        @test_throws "more than once" from_hdf5(VirtualPTDF, path)
    end

    @testset "subnetwork key off the bus axis" begin
        bus_ax = PNM.get_bus_axis(vptdf)
        ref = only(keys(PNM.get_core(vptdf).subnetwork_axes))
        other = first(b for b in bus_ax if b != ref)
        function off_axis_path(target)
            path = fresh_path()
            H.h5open(path, "r+") do f
                H.delete_object(f["subnetworks"][string(ref)], "ref")
                f["subnetworks"][string(ref)]["ref"] = 99999
                if !isnothing(target)
                    H.delete_object(f, "reduced_bus_from")
                    H.delete_object(f, "reduced_bus_to")
                    f["reduced_bus_from"] = [99999]
                    f["reduced_bus_to"] = [target]
                end
            end
            return path
        end
        restored = from_hdf5(VirtualPTDF, off_axis_path(ref))
        @test PNM.get_ref_bus_position(restored) == PNM.get_ref_bus_position(vptdf)
        @test_throws "not the reference bus" from_hdf5(VirtualPTDF, off_axis_path(other))
        @test_throws "no reduction maps it" from_hdf5(VirtualPTDF, off_axis_path(nothing))
    end

    @testset "reduced_bus_from repeats a key" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        reduced = VirtualPTDF(sys; network_reductions = RTS_REDUCTIONS)
        path = joinpath(mktempdir(), "reduced.h5")
        to_hdf5(reduced, path)
        from = H.h5open(f -> read(f["reduced_bus_from"]), path, "r")
        to = H.h5open(f -> read(f["reduced_bus_to"]), path, "r")
        _write_bad_dataset(path, "", "reduced_bus_from", [from; from[1]])
        _write_bad_dataset(path, "", "reduced_bus_to", [to; to[1]])
        @test_throws "reduced_bus_from has the value" from_hdf5(
            VirtualPTDF,
            path,
        )
    end

    @testset "reduced bus key on the bus axis" begin
        sys = PSB.build_system(PSB.PSITestSystems, "case10_radial_series_reductions")
        reduced = VirtualPTDF(sys; network_reductions = RTS_REDUCTIONS)
        function reduced_path()
            path = joinpath(mktempdir(), "reduced.h5")
            to_hdf5(reduced, path)
            return path
        end
        bus_ax = PNM.get_bus_axis(reduced)
        path = reduced_path()
        from = H.h5open(f -> read(f["reduced_bus_from"]), path, "r")
        _write_bad_dataset(path, "", "reduced_bus_from", [bus_ax[1]; from[2:end]])
        @test_throws "reduced_bus_from holds bus $(bus_ax[1]), which is on the bus axis" from_hdf5(
            VirtualPTDF,
            path,
        )
    end

    @testset "name entry off the arc axis" begin
        path = fresh_path()
        from = H.h5open(f -> read(f["names"]["from"]), path, "r")
        _write_bad_dataset(path, "names", "from", [maximum(from) + 1; from[2:end]])
        @test_throws "which is not an arc" from_hdf5(VirtualPTDF, path)
        # The check runs before the core build: a bad solver name is never reached.
        @test_throws "which is not an arc" from_hdf5(
            VirtualPTDF,
            path;
            linear_solver = "NoSuchSolver",
        )
    end

    @testset "dist_slack weights" begin
        bus_ax = PNM.get_bus_axis(vptdf)
        n_bus = length(bus_ax)
        path = fresh_path()
        _write_bad_dataset(path, "", "dist_slack", zeros(n_bus))
        @test_throws "the sum must be finite and positive" from_hdf5(VirtualPTDF, path)
        path = fresh_path()
        _write_bad_dataset(path, "", "dist_slack", [-1.0; fill(1.0, n_bus - 1)])
        @test_throws "must not be negative" from_hdf5(VirtualPTDF, path)
        path = joinpath(mktempdir(), "bad_writer.h5")
        buses = sort!([PSY.get_number(b) for b in PSY.get_components(PSY.ACBus, sys)])
        bad = VirtualPTDF(sys; dist_slack = Dict(buses[2] => -1.0, buses[5] => 2.0))
        @test_throws "must not be negative" to_hdf5(bad, path)
        @test !ispath(path)
    end
end

@testset "outage of a branch beside a Ward equivalent" begin
    ward_buses = [101, 114, 110, 111, 109]
    build() = PSB.build_system(
        PSB.PSSEParsingTestSystems,
        "psse_14_network_reduction_test_system",
    )
    reductions() = PNM.NetworkReduction[WardReduction(ward_buses)]
    sys = build()
    vm = VirtualMODF(sys; network_reductions = reductions(), tol = 0.0)
    vp = VirtualPTDF(sys; network_reductions = reductions(), tol = 0.0)
    nr = PNM.get_network_reduction_data(vm)
    arc_lookup = PNM.get_arc_lookup(vm)
    arc_sus = PNM._get_arc_susceptances(vm)
    added = PNM.get_added_arc_impedance_map(nr)
    shared = [
        arc for arc in keys(added) if haskey(nr.direct_branch_map, arc) &&
        !(nr.direct_branch_map[arc] === added[arc])
    ]
    @test !isempty(shared)
    path = joinpath(mktempdir(), "ward.h5")
    to_hdf5(vm, path, sys)
    table = from_hdf5(BranchOutageTable, path)
    for arc in shared
        branch = nr.direct_branch_map[arc]
        b_ward = PNM._ba_arc_susceptance(added[arc], nr)
        b_arc = arc_sus[arc_lookup[arc]]
        mod = NetworkModification(vm, branch)
        m = only(m for m in mod.arc_modifications if m.arc_index == arc_lookup[arc])
        @test m.delta_b ≈ b_ward - b_arc
        @test !mod.is_islanding
        sys_mod = build()
        for br in PSY.get_components(PSY.ACTransmission, sys_mod)
            if PSY.get_name(br) == PSY.get_name(branch)
                PSY.set_available!(br, false)
            end
        end
        ptdf = PTDF(sys_mod; network_reductions = reductions())
        mod_lookup = PNM.get_arc_lookup(ptdf)
        mod_bus = PNM.get_bus_lookup(ptdf)
        wf = compute_woodbury_factors(vp, mod)
        for (k, a) in enumerate(PNM.get_arc_axis(vm))
            haskey(mod_lookup, a) || continue
            row = apply_woodbury_correction(vp, k, wf)
            for (j, bus) in enumerate(PNM.get_bus_axis(vm))
                @test isapprox(row[j], ptdf[a, bus]; atol = 1e-6)
            end
        end
        rt = NetworkModification(table, PNM._outage_key(branch)...)
        file_m = only(x for x in rt.arc_modifications if x.arc_index == arc_lookup[arc])
        @test file_m.delta_b ≈ m.delta_b
        @test file_m.delta_shift_injection ≈ m.delta_shift_injection
    end
end

@testset "outage beside a Ward equivalent with negative susceptance" begin
    function build_neg(x43, x23; out = false)
        sys, buses = _mk_bus_system(4)
        for (f, t, x) in [(1, 2, 0.2), (2, 3, x23), (1, 3, 0.25), (2, 4, 0.1), (4, 3, x43)]
            arc = PSY.Arc(buses[f], buses[t])
            PSY.add_component!(sys, arc)
            _add_test_line!(sys, "L_$(f)_$(t)", arc, 0.0, x)
        end
        if out
            PSY.set_available!(PSY.get_component(PSY.Line, sys, "L_2_3"), false)
        end
        return sys
    end
    reds() = PNM.NetworkReduction[WardReduction([1, 2, 3])]
    for (x43, x23) in [(0.3, 0.5), (-0.3, 0.5), (0.3, -0.05), (-0.3, -0.05)]
        sys = build_neg(x43, x23)
        vp = VirtualPTDF(sys; network_reductions = reds(), tol = 0.0)
        mod = NetworkModification(vp, PSY.get_component(PSY.Line, sys, "L_2_3"))
        ptdf = PTDF(build_neg(x43, x23; out = true); network_reductions = reds())
        ml = PNM.get_arc_lookup(ptdf)
        wf = compute_woodbury_factors(vp, mod)
        for (k, a) in enumerate(PNM.get_arc_axis(vp))
            key = a
            if !haskey(ml, key)
                key = (a[2], a[1])
            end
            haskey(ml, key) || continue
            s = 1.0
            if key != a
                s = -1.0
            end
            row = apply_woodbury_correction(vp, k, wf)
            for (j, bus) in enumerate(PNM.get_bus_axis(vp))
                @test isapprox(row[j], s * ptdf[key, bus]; atol = 1e-8)
            end
        end
    end
end

@testset "surviving bus of a cascaded stub" begin
    sys, buses = _mk_bus_system(7)
    for (f, t) in [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 6), (4, 7)]
        arc = PSY.Arc(buses[f], buses[t])
        PSY.add_component!(sys, arc)
        _add_test_line!(sys, "L_$(f)_$(t)", arc, 0.0, 0.1)
    end
    vptdf = VirtualPTDF(
        sys;
        network_reductions = PNM.NetworkReduction[
            DegreeTwoReduction(),
            RadialReduction(),
        ],
        tol = 0.0,
    )
    bus_ax = PNM.get_bus_axis(vptdf)
    reductions = BranchReductionTable(vptdf, sys)
    n_radial = 0
    for line in PSY.get_components(PSY.Line, sys)
        id = IS.get_id(line)
        if branch_reduction(reductions, id) == BRANCH_REDUCTION_RADIAL
            @test branch_surviving_bus(reductions, id) in bus_ax
            n_radial += 1
        end
    end
    @test n_radial > 0
end

@testset "outage of every member of a parallel group opens the arc" begin
    sys = build_degenerate_3wt_system()
    vmodf = VirtualMODF(sys; tol = 0.0)
    path = joinpath(mktempdir(), "degenerate.h5")
    to_hdf5(vmodf, path, sys)
    table = from_hdf5(BranchOutageTable, path)
    bmap = from_hdf5(BranchArcTable, path)
    trf = PSY.get_component(PSY.ThreeWindingTransformer, sys, "T3W_degenerate")
    _, range = PNM.get_outage_rows(table)[PNM._outage_key(trf)]
    kind = get_branch_kind(bmap)
    opens = PNM.get_outage_opens_arc(table)
    tops = [findall(==(-table.arc_index[i]), get_branch_parent(bmap)) for i in range]
    group_arcs = [i for (i, t) in zip(range, tops) if any(==(9), kind[t])]
    @test length(group_arcs) == 1
    @test all(opens[group_arcs])
    dead = [r for (i, t) in zip(range, tops) if opens[i] for r in t if kind[r] != 11]
    flows = branch_flows(bmap, ones(get_branch_arc_count(bmap)), dead)
    for r in findall(==(IS.get_id(trf)), get_branch_id(bmap))
        @test iszero(flows[r])
    end
end

@testset "arc-susceptance VirtualFactorCore accepts verify_factorization" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    core = PNM.get_core(VirtualPTDF(sys; tol = 0.0))
    for verify in (false, true)
        rebuilt = PNM.VirtualFactorCore(
            core.arc_susceptances .* core.arc_susceptance_signs,
            Set(PNM.get_ref_bus_position(core)),
            core.axes,
            core.subnetwork_axes,
            core.arc_live_members,
            PNM.get_branch_catalog(core);
            tol = 0.0,
            verify_factorization = verify,
        )
        @test rebuilt.BA == core.BA
        @test rebuilt.A == core.A
    end
end

@testset "branch_reduction by branch id" begin
    sys = PSB.build_system(PSB.PSITestSystems, "c_sys5")
    vptdf = VirtualPTDF(sys; tol = 0.0)
    reductions = BranchReductionTable(vptdf, sys)
    line = first(PSY.get_components(PSY.Line, sys))
    @test branch_reduction(reductions, IS.get_id(line)) == BRANCH_REDUCTION_HAS_ARC
    @test_throws "has no branch with id -1." branch_reduction(reductions, -1)
    @test_throws "did not remove" branch_surviving_bus(reductions, IS.get_id(line))
    path = joinpath(mktempdir(), "nosys.h5")
    to_hdf5(vptdf, path)
    @test_throws "no group branch_reduction" from_hdf5(BranchReductionTable, path)
end

@testset "out-of-service branch beside an in-service one" begin
    sys = PSB.build_system(
        PSB.PSITestSystems,
        "c_sys14";
        force_build = true,
        skip_serialization = true,
    )
    twin = first(PSY.get_components(PSY.Line, sys))
    _add_test_line!(sys, "out_twin", PSY.get_arc(twin), 0.01, 0.1)
    out = PSY.get_component(PSY.Line, sys, "out_twin")
    PSY.set_available!(out, false)
    # Ybus skips an available breaker that is open.
    breaker = PSY.DiscreteControlledACBranch(; input_basis = u"CU",
        name = "open_breaker", available = true, active_power_flow = 0.0,
        reactive_power_flow = 0.0, arc = PSY.get_arc(twin), r = 0.0, x = 0.01,
        rating = 1.0, branch_status = PSY.DiscreteControlledBranchStatus.OPEN,
    )
    PSY.add_component!(sys, breaker)
    vptdf = VirtualPTDF(sys; tol = 0.0)
    path = joinpath(mktempdir(), "twin.h5")
    to_hdf5(vptdf, path, sys; outages = false)
    reductions = from_hdf5(BranchReductionTable, path)
    @test branch_reduction(reductions, IS.get_id(twin)) == BRANCH_REDUCTION_HAS_ARC
    @test branch_reduction(reductions, IS.get_id(out)) == BRANCH_REDUCTION_NOT_IN_NETWORK
    @test branch_reduction(reductions, IS.get_id(breaker)) ==
          BRANCH_REDUCTION_NOT_IN_NETWORK
    @test !HDF5.h5open(f -> haskey(f, "branch_outages"), path, "r")
    @test_throws "no branch outage table" from_hdf5(BranchOutageTable, path)

    to_hdf5(vptdf, path, sys; force = true)
    @test HDF5.h5open(f -> haskey(f, "branch_outages"), path, "r")
    @test from_hdf5(BranchReductionTable, path).rows == reductions.rows

    bad(column, value) = from_hdf5(
        BranchReductionTable,
        _corrupted_copy(
            path,
            "branch_reduction",
            column => fill(value, length(reductions.rows)),
        ),
    )
    @test_throws "winding 4 is not in 0:3" bad("winding", 4)
    @test_throws "code 7 is not in 0:4" bad("code", 7)
    @test_throws "surviving_bus is not positive" bad("code", 1)
    @test_throws "more than once" bad("id", 1)
end

@testset "branch_map stores the BA susceptance" begin
    row_of(table, br) = only(findall(==(IS.get_id(br)), get_branch_id(table)))
    # An arc with no DC coupling carries no flow, and a flow there is an error.
    lone = PNM.BranchArcTable([1], [1], [-1], [1], [0.0], [0.0], 1)
    @test branch_flows(lone, [0.0]) == [0.0]
    @test_throws "no DC coupling" branch_flows(lone, [1.0])

    # A transformer with an impedance correction stores the corrected value.
    sys = PSB.build_system(
        PSB.PSITestSystems,
        "c_sys14";
        force_build = true,
        skip_serialization = true,
    )
    tr = first(PSY.get_components(PSY.TwoWindingTransformer, sys))
    PSY.add_supplemental_attribute!(
        sys,
        tr,
        PSY.ImpedanceCorrectionData(;
            table_number = 1,
            tap_ratio_correction_curve = IS.PiecewiseLinearData([
                (x = 0.5, y = 1.5),
                (x = 1.5, y = 1.5),
            ]),
            transformer_winding = PSY.WindingCategory.TR2W_WINDING,
            transformer_control_mode =
            PSY.ImpedanceCorrectionTransformerControlMode.TAP_RATIO,
        ),
    )
    vptdf = VirtualPTDF(sys; tol = 0.0)
    nr = PNM.get_network_reduction_data(vptdf)
    path = joinpath(mktempdir(), "corrected.h5")
    to_hdf5(vptdf, path)
    table = from_hdf5(BranchArcTable, path)
    b_tr = get_branch_susceptance(table)[row_of(table, tr)]
    @test b_tr ≈ PNM._ba_arc_susceptance(tr, nr)
    @test !(b_tr ≈ PNM._finite_series_susceptance(tr, nr))
end
