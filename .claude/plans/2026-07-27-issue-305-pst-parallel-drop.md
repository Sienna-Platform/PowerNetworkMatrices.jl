# Issue #305: PST Parallel-Branch Drop — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** No branch is ever silently dropped from `NetworkReductionData` when a phase-shifting circuit shares an arc with other branches, for any arrival order and any member count.

**Architecture:** Remove the `_is_phase_shifting` guards in `add_to_branch_maps!` (`src/Ybus.jl:284-289`) so every co-arc branch is filed into the parallel maps. This aligns the assembly path with the ZIR-merge path, which already produces phase-shifter-containing parallel groups (see the passing test at `test/test_ybus_reductions.jl:725`). The single-π-equivalent representation constraint is enforced only where an equivalent is actually requested (`_get_equivalent_physical_branch_parameters`), with an informative error.

**Tech Stack:** Julia, PowerSystems.jl (psy6 winding design), ReTest.

## Context: the issue vs. the new winding design

The issue (written pre-refactor) blames `SKIP_PARALLEL_REDUCTION_TYPES`, which no longer exists. On `jd/transformer_refactor` the same bug survives in data-driven form: `add_to_branch_maps!` refuses to group a branch when either it or the incumbent satisfies `_is_phase_shifting`, and the `else` clause overwrites `direct_branch_map[arc_tuple]`, dropping the incumbent from all NRD maps. Ybus is unaffected (it accumulates per-device); only NRD bookkeeping — and therefore downstream flow reporting — is wrong.

Answers to the open questions in the issue comments (m-bossart, orennia-juan), under this plan:

  - **PST ∥ Line** → `MixedBranchesParallel` (mixed types already warn — unchanged behavior for any mixed group).
  - **PST ∥ PST** (both `TwoWindingTransformer`) → homogeneous `BranchesParallel{TwoWindingTransformer}`. Nothing dropped.
  - **Line + PST + PST** (any order) → one group of three. Order-independent because the first branch on an occupied arc always promotes to a group and every later branch always pushes into it.

Why removing the guards is safe (verified against the code):

  - **Ybus:** accumulated per-device, unaffected (`Ybus.jl` assembly). Group-level `ybus_branch_entries(bp, nr)` already handles asymmetric (tap/shift) members with orientation (regression test at `test_ybus_reductions.jl:725-823`).
  - **BA/PTDF:** `BA_Matrix` detects asymmetric off-diagonals and falls back to `_arc_component_susceptance` (`BA_ABA_matrices.jl:102-108`), which already resolves parallel-map arcs by summing member susceptances (phase-independent `1/(a·x)`). PTDF row apportionment via `compute_parallel_multiplier` is a susceptance-ratio split — correct for *sensitivities* even with a shifted member (α contributes a constant injection, not a sensitivity).
  - **AC flow output (PowerFlows):** `_compute_segment_flows(::PNM.AbstractBranchesParallel, …)` evaluates each member independently through its own 2×2 (`post_processing.jl:544-557`) — exact for shifted members.
  - **DC solve (PowerFlows):** the DC path does not model α injections at all today, so group membership changes nothing.
  - **Contingencies:** tripping a phase-shifting component is already rejected by `_assert_not_phase_shifting` at the top of the generic classification methods (`network_modification.jl:334,438,491,517`); tripping a *non-shifting* member of a group uses only that member's susceptance — correct.
  - **Single-π equivalent (`arc_equivalent_branch`):** the only place with a real representation limit. For **lossless** members the summed 2×2 keeps `|y12| == |y21|` and the existing shift-extraction in `_get_equivalent_physical_branch_parameters` is *exact*. For **lossy** mixed-α groups no single-π equivalent exists and the existing error fires — Task 5 makes it name the offending group. Sole downstream consumer: PowerFlows DCLF loss injection (`dcpf_loss_injection.jl:24`), an opt-in feature; a follow-up PowerFlows issue is filed in Task 8, not fixed here.

## Global Constraints

  - **Never `git commit` or `git push`** — leave all changes UNSTAGED (use `git add -N` only for new files). The user commits.
  - All test runs use `julia --project=test …` from the repo root `/Users/jdlara/cache/psy6/PowerNetworkMatrices.jl`.
  - Compile-check after each source edit: `julia --project=test -e 'using PowerNetworkMatrices'`.
  - Formatter before completion: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`.
  - Julia style: no `isa`/type-check branches (multiple dispatch), no ternaries, `iszero(x)`, explicit `function … end` + `return`, terse comments (only non-obvious WHY).
  - ReTest: never `@test_logs` for warnings; name-filtered runs via `run_tests("<pattern>")`.
  - Never export `KLUWrapper` symbols; exports live only in `src/PowerNetworkMatrices.jl`.

* * *

### Task 1: White-box ordering tests for `add_to_branch_maps!`

**Files:**

  - Modify: `test/test_ybus_reductions.jl` (append after the `_mk_bus_system` helper, `:826-849`)

**Interfaces:**

  - Consumes: `PNM.NetworkReductionData()` (empty NRD; `get_arc_tuple` falls back to raw bus numbers), `PNM.add_to_branch_maps!(nr, arc, br)`, `PNM.get_direct_branch_map`, `PNM.get_parallel_branch_map`, `PNM.get_reverse_direct_branch_map`, `PNM.get_reverse_parallel_branch_map`.

  - Produces: helper `_mk_detached_pst_fixture()` returning `(line, line2, pst1, pst2)` detached components on arc (1, 2), reused by no other task (integration tasks use attached systems because detached circuits have `base_value === nothing`).
  - [ ] **Step 1: Write the failing tests**

Append to `test/test_ybus_reductions.jl`:

```julia
# Detached components suffice for map-filing tests: `add_to_branch_maps!` only reads arc bus
# numbers, never impedances (which require an attached system).
function _mk_detached_pst_fixture()
    b1 = ACBus(;
        number = 1, name = "b1", available = true, bustype = ACBusTypes.REF,
        angle = 0.0, magnitude = 1.0, voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
    b2 = ACBus(;
        number = 2, name = "b2", available = true, bustype = ACBusTypes.PV,
        angle = 0.0, magnitude = 1.0, voltage_limits = (min = 0.9, max = 1.1),
        base_voltage = 230.0,
    )
    function _mk_fixture_line(name)
        return Line(;
            name = name, available = true, active_power_flow = 0.0,
            reactive_power_flow = 0.0, arc = Arc(; from = b1, to = b2),
            r = 0.0, x = 0.1, b = (from = 0.0, to = 0.0), rating = 1.0,
            angle_limits = (min = -1.5, max = 1.5),
        )
    end
    function _mk_fixture_pst(name, α)
        return PSY.TwoWindingTransformer(;
            name = name,
            circuit = PSY.TransformerCircuit(;
                arc = Arc(; from = b1, to = b2), tap = 1.0, α = α,
                available = true, active_power_flow = 0.0, reactive_power_flow = 0.0,
                rating = 1.0, base_power = 100.0, base_voltage_primary = 230.0,
                r = 0.0, x = 0.2,
            ),
            magnetizing_shunt = Complex(0.0, 0.0),
        )
    end
    return (
        _mk_fixture_line("L1"),
        _mk_fixture_line("L2"),
        _mk_fixture_pst("PST1", 0.15),
        _mk_fixture_pst("PST2", 0.10),
    )
end

# Every branch filed on the arc must be reachable in exactly one reverse map, and the arc must
# live in exactly one forward map.
function _assert_arc_maps_complete(nr, branches)
    arc_tuple = (1, 2)
    direct = PNM.get_direct_branch_map(nr)
    parallel = PNM.get_parallel_branch_map(nr)
    if length(branches) == 1
        @test haskey(direct, arc_tuple)
        @test !haskey(parallel, arc_tuple)
        @test PNM.get_reverse_direct_branch_map(nr)[branches[1]] == arc_tuple
    else
        @test !haskey(direct, arc_tuple)
        @test haskey(parallel, arc_tuple)
        @test length(parallel[arc_tuple]) == length(branches)
        for br in branches
            @test PNM.get_reverse_parallel_branch_map(nr)[br] == arc_tuple
        end
    end
end

@testset "issue 305: add_to_branch_maps! never drops a co-arc branch" begin
    (line, line2, pst1, pst2) = _mk_detached_pst_fixture()
    orderings = [
        [line, pst1],          # regular first, shifter second (issue's table)
        [pst1, line],          # shifter first
        [pst1, pst2],          # PST ∥ PST (m-bossart's question)
        [line, pst1, pst2],    # Line+PST+PST (orennia-juan's question)
        [pst1, pst2, line],    # shifters first, regular last
        [line, line2, pst1],   # shifter joins an existing homogeneous group
    ]
    for branches in orderings
        nr = PNM.NetworkReductionData()
        for br in branches
            PNM.add_to_branch_maps!(nr, PSY.get_arc(br), br)
        end
        _assert_arc_maps_complete(nr, branches)
    end
end
```

  - [ ] **Step 2: Run and verify the new testset fails**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("issue 305")'
```

Expected: FAIL — orderings containing a shifter leave one branch missing from the reverse maps (the overwrite bug). If ReTest reports no matching testset, verify registration with `run_tests(dry = true)`.

### Task 2: Remove the phase-shifting guards in `add_to_branch_maps!`

**Files:**

  - Modify: `src/Ybus.jl:282-300` (the branch-classification `if/elseif/else` inside `add_to_branch_maps!`)
  - Test: `test/test_ybus_reductions.jl` (Task 1's testset)

**Interfaces:**

  - Consumes: existing `_push_parallel_branch!` (`Ybus.jl:204`) and `_make_parallel_branch_pair` (`Ybus.jl:187-202`) — both already handle heterogeneous members (Mixed promotion + `@warn`).

  - Produces: `add_to_branch_maps!(nr, arc, br)` files any second co-arc branch into `parallel_branch_map` and any later one into the existing group, regardless of `_is_phase_shifting`.
  - [ ] **Step 1: Replace the classification block**

In `src/Ybus.jl`, replace lines 282-300:

```text
    # A phase shifter is never folded into a parallel-equivalent group: the parallel
    # susceptance model cannot represent a per-branch phase shift.
    if haskey(parallel_branch_map, arc_tuple) && !_is_phase_shifting(br)
        _push_parallel_branch!(parallel_branch_map, arc_tuple, br)
        reverse_parallel_branch_map[br] = arc_tuple
    elseif haskey(direct_branch_map, arc_tuple) &&
           !_is_phase_shifting(direct_branch_map[arc_tuple]) &&
           !_is_phase_shifting(br)
```

with:

```text
    # Phase-shifting members are legitimate group members (the ZIR merge path already
    # produces them): Ybus and per-member flow evaluation are exact; the single-π
    # equivalent constraint is enforced in `_get_equivalent_physical_branch_parameters`.
    if haskey(parallel_branch_map, arc_tuple)
        _push_parallel_branch!(parallel_branch_map, arc_tuple, br)
        reverse_parallel_branch_map[br] = arc_tuple
    elseif haskey(direct_branch_map, arc_tuple)
```

(The promote-to-parallel body and the `else` clause are unchanged.)

  - [ ] **Step 2: Compile-check**

```bash
julia --project=test -e 'using PowerNetworkMatrices'
```

  - [ ] **Step 3: Run Task 1's testset — expect PASS**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("issue 305")'
```

  - [ ] **Step 4: Run the Ybus/reduction suites for regressions**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests(r"Ybus|reduction|Reduction")'
```

Expected: PASS. If an existing test asserted the old drop behavior (a PST-only direct entry on a shared arc), that assertion is the bug this issue reports — update the test, and say so in the final report.

### Task 3: Integration test — Ybus, NRD, and BA with a Line ∥ PST arc

**Files:**

  - Modify: `test/test_ybus_reductions.jl` (append after Task 1's testset)
  - Modify (only if BA fails): `src/BA_ABA_matrices.jl:102-108`

**Interfaces:**

  - Consumes: `Ybus(sys)`, `BA_Matrix(ybus)`, `PNM.ybus_branch_entries(bp, nr)`, `PNM.get_series_susceptance(br, PSY.SU)`, Task 1's fixture pattern (attached this time — impedance reads require `base_value`, which only `add_component!` populates).

  - Produces: an attached-system builder `_mk_line_pst_parallel_system(; pst_r = 0.0)` reused by Tasks 5 and 6. Returns `(sys, line_names = ["L1"], pst_name = "PST")` with arcs: L1 on (1, 2), PST on (1, 2), L2 on (2, 3).
  - [ ] **Step 1: Write the test**

```julia
function _mk_line_pst_parallel_system(; pst_r = 0.0)
    sys, buses = _mk_bus_system(3)
    function _mk_sys_line(name, f, t)
        arc = Arc(; from = buses[f], to = buses[t])
        add_component!(sys, arc)
        add_component!(
            sys,
            Line(;
                name = name, available = true, active_power_flow = 0.0,
                reactive_power_flow = 0.0, arc = arc, r = 0.0, x = 0.1,
                b = (from = 0.0, to = 0.0), rating = 1.0,
                angle_limits = (min = -1.5, max = 1.5),
            ),
        )
    end
    _mk_sys_line("L1", 1, 2)
    _mk_sys_line("L2", 2, 3)
    pst_arc = Arc(; from = buses[1], to = buses[2])
    add_component!(sys, pst_arc)
    add_component!(
        sys,
        PSY.TwoWindingTransformer(;
            name = "PST",
            circuit = PSY.TransformerCircuit(;
                arc = pst_arc, tap = 1.0, α = 0.15, available = true,
                active_power_flow = 0.0, reactive_power_flow = 0.0, rating = 1.0,
                base_power = 100.0, base_voltage_primary = 230.0,
                r = pst_r, x = 0.2,
            ),
            magnetizing_shunt = Complex(0.0, 0.0),
        ),
    )
    return sys
end

@testset "issue 305: Line ∥ PST — Ybus, NRD completeness, BA susceptance" begin
    sys = _mk_line_pst_parallel_system()
    ybus = Ybus(sys)
    nr = ybus.network_reduction_data

    # NRD completeness: both branches on (1, 2) are in the parallel maps.
    parallel = PNM.get_parallel_branch_map(nr)
    @test haskey(parallel, (1, 2))
    @test length(parallel[(1, 2)]) == 2
    @test !haskey(PNM.get_direct_branch_map(nr), (1, 2))
    reverse_parallel = PNM.get_reverse_parallel_branch_map(nr)
    for br in parallel[(1, 2)]
        @test reverse_parallel[br] == (1, 2)
    end

    # Group ybus entries match the accumulated Ybus (independent code path), including
    # the phase-shift asymmetry.
    bl = ybus.lookup[1]
    ip = bl[1]
    iq = bl[2]
    aware = PNM.ybus_branch_entries(parallel[(1, 2)], nr)
    @test !isapprox(aware[2], aware[3])
    @test aware[1] ≈ ybus.data[ip, ip]
    @test aware[2] ≈ ybus.data[ip, iq]
    @test aware[3] ≈ ybus.data[iq, ip]
    @test aware[4] ≈ ybus.data[iq, iq]

    # BA takes the asymmetric-arc fallback: b = sum of member susceptances (α-independent).
    ba = BA_Matrix(ybus)
    b_expected = sum(
        PNM.get_series_susceptance(br, PSY.SU) for br in parallel[(1, 2)]
    )
    arc_ix = findfirst(==((1, 2)), ba.axes[2])
    @test ba.data[arc_ix, bl[1]] ≈ b_expected
end
```

Note on the BA indexing: `BA_Matrix` stores transposed data — if `ba.data[arc_ix, bl[1]]` errors or mismatches, read the axes convention at the top of `src/BA_ABA_matrices.jl` and index accordingly; the assertion target is the (from-bus, arc) entry equal to `b_expected`.

  - [ ] **Step 2: Run**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("issue 305")'
```

Expected: PASS with no source change (both group-entry summation and the BA fallback already support parallel-map arcs). A failure here means `_arc_component_susceptance` or group orientation has a real gap — fix it in the file listed above, minimally, before proceeding.

### Task 4: Group-level `_is_phase_shifting`

The fallback `_is_phase_shifting(::PSY.ACTransmission) = false` (`definitions.jl:24`) silently answers `false` for `AbstractBranchesParallel` and `BranchesSeries` (both subtype `PSY.ACTransmission`). Now that groups can legitimately contain shifters, that is a wrong answer waiting for a caller.

**Files:**

  - Modify: `src/BranchesParallel.jl` (after the `add_branch!` methods, `:47`)
  - Modify: `src/BranchesSeries.jl` (after the struct + iteration definitions; place next to the other trait-style accessors)
  - Test: `test/test_ybus_reductions.jl`

**Interfaces:**

  - Consumes: `_is_phase_shifting` function defined in `definitions.jl` (included at `PowerNetworkMatrices.jl:122`, before both files — adding methods later in include order is fine).

  - Produces: `_is_phase_shifting(bp::AbstractBranchesParallel)::Bool`, `_is_phase_shifting(bs::BranchesSeries)::Bool` — true when any member (recursively, since chain segments can be parallel groups) is phase shifting.
  - [ ] **Step 1: Write the failing test**

```julia
@testset "issue 305: group-level _is_phase_shifting" begin
    (line, line2, pst1, pst2) = _mk_detached_pst_fixture()
    @test PNM._is_phase_shifting(PNM.MixedBranchesParallel([line, pst1]))
    @test !PNM._is_phase_shifting(PNM.BranchesParallel([line, line2]))
    @test PNM._is_phase_shifting(
        PNM.BranchesParallel(PSY.TwoWindingTransformer[pst1, pst2]),
    )
end
```

(If `BranchesParallel([pst1, pst2])` fails to infer the concrete type from the vector literal, use `PNM.BranchesParallel(PSY.TwoWindingTransformer[pst1, pst2])`.)

  - [ ] **Step 2: Run to verify it fails**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("group-level")'
```

Expected: FAIL — the first `@test` returns `false` via the `ACTransmission` fallback.

  - [ ] **Step 3: Implement**

In `src/BranchesParallel.jl` after `add_branch!`:

```julia
# The blanket `_is_phase_shifting(::PSY.ACTransmission) = false` in definitions.jl would
# silently answer for groups; a group shifts when any member does.
function _is_phase_shifting(bp::AbstractBranchesParallel)
    return any(_is_phase_shifting, bp.branches)
end
```

In `src/BranchesSeries.jl` (chain segments can themselves be parallel groups, so this recurses through the method above):

```julia
function _is_phase_shifting(bs::BranchesSeries)
    return any(_is_phase_shifting, bs)
end
```

If `BranchesSeries` iteration yields something other than segment objects, iterate the segments field directly (check the struct at the top of `src/BranchesSeries.jl`).

  - [ ] **Step 4: Compile-check, then run — expect PASS**

```bash
julia --project=test -e 'using PowerNetworkMatrices'
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("group-level")'
```

### Task 5: Equivalent-branch behavior for shifted parallel groups

Lossless mixed-α groups keep `|y12| == |y21|`, so the existing extraction in `_get_equivalent_physical_branch_parameters` (`common.jl:286-312`) is exact and must keep working. Lossy mixed-α groups have no single-π equivalent; the existing error must name the offending group instead of only describing the math.

**Files:**

  - Modify: `src/common.jl:286` (`_get_equivalent_physical_branch_parameters` — add a context argument) and `src/common.jl:329-337` (`get_equivalent_physical_branch_parameters` — pass the group name)
  - Test: `test/test_ybus_reductions.jl`

**Interfaces:**

  - Consumes: `arc_equivalent_branch(nr, arc)` (`BranchAdmittance.jl:252`), `_mk_line_pst_parallel_system(; pst_r)` from Task 3, `PNM.get_equivalent_shift`.

  - Produces: `_get_equivalent_physical_branch_parameters(equivalent_ybus, context::String = "")`; unchanged return type `EquivalentBranch`. The wrapper `get_equivalent_physical_branch_parameters(segment, nr)` passes `"Offending group: <name>."` — callers of the 2-arg form need no change.
  - [ ] **Step 1: Write the tests**

```julia
@testset "issue 305: equivalent branch for shifted parallel groups" begin
    # Lossless members: |y12| == |y21|, the single-π equivalent is exact.
    sys = _mk_line_pst_parallel_system()
    ybus = Ybus(sys)
    nr = ybus.network_reduction_data
    eq = PNM.arc_equivalent_branch(nr, (1, 2))
    @test eq isa PNM.EquivalentBranch
    # The extracted shift is intermediate between the members' angles (0 and 0.15).
    @test 0.0 < abs(PNM.get_equivalent_shift(eq)) < 0.15

    # Lossy members: no single-π equivalent exists; the error must name the group.
    sys_lossy = _mk_line_pst_parallel_system(; pst_r = 0.05)
    ybus_lossy = Ybus(sys_lossy)
    nr_lossy = ybus_lossy.network_reduction_data
    err = try
        PNM.arc_equivalent_branch(nr_lossy, (1, 2))
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("Offending group", err.msg)
end
```

  - [ ] **Step 2: Run to verify the second half fails**

Expected: the lossless half passes already; the lossy half fails on `occursin("Offending group", …)` (current message has no group context). If instead the *lossless* half fails, stop and investigate `_get_equivalent_physical_branch_parameters` numerics before touching the message — that would be a real regression, not a message gap. If the lossy half does not throw at all (the `atol = 1e-6` tolerance can absorb small `r`), raise `pst_r` until it does and note the threshold in the test.

  - [ ] **Step 3: Implement the context argument**

In `src/common.jl`, change the signature and the error:

```text
function _get_equivalent_physical_branch_parameters(
    equivalent_ybus::Matrix{YBUS_ELTYPE},
    context::String = "",
)
```

and inside the existing `!isapprox(0.0, real(ratio); atol = 1e-6)` branch, extend the `error(...)` string with the group identity (keep the existing physics explanation):

```julia
error(
    "Equivalent parameters for the series or parallel reduction of branches results \
in a real part of the phase shift angle. A lossy phase-shifting circuit in parallel \
with other branches has no single-π equivalent. $(context)",
)
```

In `get_equivalent_physical_branch_parameters(segment, nr)` (`common.jl:329-337`), pass the context:

```julia
return _get_equivalent_physical_branch_parameters(
    segment.equivalent_ybus,
    "Offending group: $(get_name(segment)).",
)
```

  - [ ] **Step 4: Compile-check, run testset, then the equivalent-getters suite**

```bash
julia --project=test -e 'using PowerNetworkMatrices'
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("issue 305")'
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("equivalent")'
```

Expected: all PASS (WECC-240 equivalent-parameters testset at `test/test_equivalent_getters.jl:435` guards against collateral damage).

### Task 6: Contingency behavior on a PST-containing group

No source change expected — this pins the intended semantics: tripping a non-shifting member of a shifted group works; tripping the shifting member keeps erroring via `_assert_not_phase_shifting`.

**Files:**

  - Modify: `test/test_ybus_contingencies.jl` (append; follow the file's existing `ContingencySpec` construction pattern — read the top of the file first and reuse its helpers)
  - Test fixture: `_mk_line_pst_parallel_system()` from Task 3 (defined in `test/test_ybus_reductions.jl`; if the contingency test file cannot see it, move the helper to `test/testing_data.jl` and keep a single definition)

**Interfaces:**

  - Consumes: `PNM._classify_branch_modification(nr, arc_lookup, arc_susceptances, branch)` or the public `ContingencySpec` path used by the file's existing tests; `PSY.get_component(Line, sys, "L1")`, `PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")`.

  - Produces: nothing new — regression pins only.
  - [ ] **Step 1: Write the tests**

```julia
@testset "issue 305: contingencies on a Line ∥ PST group" begin
    sys = _mk_line_pst_parallel_system()
    line = PSY.get_component(Line, sys, "L1")
    pst = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")

    # Tripping the non-shifting member classifies as a parallel-arc modification
    # using only that member's susceptance.
    # (Build the ContingencySpec exactly as the sibling testsets in this file do.)
    spec = ContingencySpec([line])            # adapt constructor to the file's pattern
    @test spec isa ContingencySpec             # and assert the parallel classification
    # the way sibling tests assert theirs

    # Tripping the shifting member is rejected loudly.
    @test_throws ErrorException ContingencySpec([pst])
end
```

The exact `ContingencySpec` construction and assertion idiom must be copied from the sibling testsets in `test/test_ybus_contingencies.jl` — the two behavioral targets are fixed: non-shifting member trips classify (no throw, one parallel `ArcModification` carrying `-get_series_susceptance(line, PSY.SU)`), shifting member trips throw `ErrorException` mentioning "phase-shifting".

  - [ ] **Step 2: Run**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("issue 305")'
```

Expected: PASS without source changes. If the Line trip throws, the failure is in `_resolve_branch_arc`/classification for parallel members — diagnose before changing anything (superpowers:systematic-debugging).

### Task 7: Comment/doc audit, formatter, full suite

**Files:**

  - Modify: `src/Ybus.jl` docstring of `add_to_branch_maps!` (`:252-271`) — the "Implementation Details" list should state that all co-arc branches are grouped, including phase-shifting members.

  - Modify: `.claude/CLAUDE.md` — update any stale "phase shifter never grouped" phrasing if present (check the Numerics/orientation sections).
  - Audit (read-only, change only if stale): `src/BA_ABA_matrices.jl:96-101` comment (already says "direct or parallel branch" — likely fine), `src/NetworkReductionData.jl:119` docstring.
  - [ ] **Step 1: Update the `add_to_branch_maps!` docstring** — replace the "If arc already has..." bullets to reflect: parallel-map hit → push into group; direct-map hit → promote both to a group (homogeneous `BranchesParallel{T}` or `MixedBranchesParallel` + `@warn` for mixed types); phase-shifting members are grouped like any other (issue #305).
  - [ ] **Step 2: Run the formatter**

```bash
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'
```

  - [ ] **Step 3: Full PNM suite**

```bash
julia --project=test test/runtests.jl
```

Expected: PASS. Report any failure verbatim; do not skip.

### Task 8: Downstream verification (PowerFlows + POM)

PNM's contract: after a PNM change, run the PF suite and at least the POM network-constructor tests. Heavy runs stay at the controller level (`run_in_background`), never inside a subagent.

  - [ ] **Step 1: Point PowerFlows' test env at this PNM checkout and run its suite in the background**

```bash
cd /Users/jdlara/cache/psy6/PowerFlows.jl
julia --project=test -e 'using Pkg; Pkg.develop(path="../PowerNetworkMatrices.jl"); Pkg.instantiate()'
julia --project=test test/runtests.jl   # run_in_background at controller level
```

Watch specifically: AC post-processing branch-flow testsets (per-member evaluation must now report flows for previously-dropped branches) and any DCLF loss-injection testset (a fixture with a lossy shifted parallel group would now hit the Task 5 error — if one exists, that is the known limitation; report it, do not silence it).

  - [ ] **Step 2: POM network-constructor tests**

```bash
cd /Users/jdlara/cache/psy6/PowerOperationsModels.jl
julia --project=test -e 'using Pkg; Pkg.develop(path="../PowerNetworkMatrices.jl"); Pkg.instantiate()'
julia --project=test -e 'using PowerOperationsModels; include("test/<network constructor test entry>")'
```

(Locate the network-constructor test file names in POM's `test/` before running; run at least those.)

  - [ ] **Step 3: File the PowerFlows follow-up** (do not fix here): `_get_arc_branch_params` (`src/dcpf_loss_injection.jl:13-31`) calls `PNM.arc_equivalent_branch` on every arc; on a system with a *lossy* shifted parallel group, DCLF loss injection now throws the informative PNM error. Correct fix is per-member loss computation via `PNM.ybus_branch_entries(member)` summed over the group. Draft the issue text for the user to file — do not open it without their say-so.

  - [ ] **Step 4 (optional, manual): case6470rte end-to-end validation** — download `pglib_opf_case6470_rte.m` per the issue, run the issue's reproduction script, and confirm the "no branches dropped" branch is taken. Not CI material (external download); offer to the user.

* * *

## Self-review notes

  - **Spec coverage:** issue root cause → Tasks 1-2; m-bossart's PST∥PST → Task 1 ordering 3; orennia-juan's Line+PST+PST and order-dependence → Task 1 orderings 4-6; issue's "Impact: branch flow output" → Task 8 Step 1; issue's proposed patch is *superseded* (it leaves the `parallel_branch_map` guard in place, so Line+PST+PST would still misfile — our removal covers it, proven by ordering 6).
  - **Types:** `_mk_detached_pst_fixture` (Task 1) is consumed by Task 4; `_mk_line_pst_parallel_system(; pst_r)` (Task 3) by Tasks 5-6; `_get_equivalent_physical_branch_parameters(::Matrix{YBUS_ELTYPE}, ::String)` keeps a default so the one other call site (`populate_equivalent_ybus!` path) is untouched.
  - **Known deliberate non-goals:** DC α-injection modeling (PowerFlows doesn't model it today), exact per-member DC flow split for shifted groups in `_distribute_arc_flows` (consistent with the current DC model), PSS/E export of grouped shifters (exporter iterates system components, not NRD maps).
