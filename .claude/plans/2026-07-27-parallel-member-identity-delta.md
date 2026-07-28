# Parallel-Member Identity Resolution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Contingency deltas and flow multipliers for parallel-group members resolve the member by object identity, never by susceptance value or name, so equal-susceptance or same-named members can no longer yield the wrong π-model.

**Architecture:** Thread the tripped component from the classification layer (which always has it) down through a new 4-arg `_compute_arc_ybus_delta`, replacing the susceptance-value scan in `_compute_parallel_partial_ybus_delta` with a direct member-outage delta that also applies the anti-parallel orientation swap the group-level `ybus_branch_entries(bp, nr)` already performs (a second latent defect in the value-based path). The value-based partial-parallel path is deleted, not deprecated — the arc-tuple-only entry point supports full outages only. The same identity-vs-value disease in `compute_parallel_multiplier` (name matching) gets an identity method, with the name method hardened to error on ambiguity.

**Tech Stack:** Julia, PowerSystems.jl (psy6 winding design), ReTest.

## Background (verified against source, branch jd/transformer_refactor @ f1fdca1)

  - `_compute_parallel_partial_ybus_delta` (`src/network_modification.jl:34-47`) scans `bp.branches` for the first member whose `get_series_susceptance(br, PSY.SU)` matches `delta_b` within `atol = 1e-10` (`_is_full_outage`, `:1-3`) and returns that member's negated π-model. Two defects:
    
     1. **Identity:** a `Line` (`x = 0.1`, b = 10) parallel to a `TwoWindingTransformer` (`tap = 1.0, x = 0.1`, b = 10) collide; tripping the line can return the transformer's asymmetric 2×2. Silent, wrong AC contingency Ybus. Real double circuits are identical twins, which is why this never bit — identical members make the wrong pick harmless.
     2. **Orientation:** the group equivalent `ybus_branch_entries(bp, nr)` (`src/Ybus.jl:463-489`) swaps an anti-parallel member's 2×2 into the group's key frame (`Y11↔Y22`, `Y12↔Y21`); the member delta path returns raw member entries with no swap, so tripping an anti-parallel asymmetric member (tap ≠ 1) applies the delta in the wrong frame. Masked for symmetric lines (swap is a no-op).

  - The classification layer always has the component in hand at the parallel/direct arms: `_classify_outage_component!` (`network_modification.jl:340-348`) and `_classify_branch_modification(…, branch::PSY.ACTransmission)` (`:523-531`). Only they reach the partial-parallel delta. Other `_compute_arc_ybus_delta` callers are full-outage (`NetworkModification(mat, arc::Tuple)`, `:142`) or series-arc (`:278`, `:536`) and never reach the member scan.
  - `ThreeWindingTransformerCircuit <: PSY.ACTransmission` can be a parallel member post-bus-merge; the value-based scan is *unreliable* for it (that is why `_direct_arc_ybus_delta` has a dedicated TWTC method, `:62-72`). Identity resolution fixes the parallel-TWTC case for free.
  - Sibling disease: `compute_parallel_multiplier(bp, branch_name::String)` (`src/BranchesParallel.jl:75-90`) sums susceptance over **all** members matching the name — PSY names are unique per concrete type only, so a `Line` "T1" and a transformer "T1" in one `MixedBranchesParallel` double-count. Its PNM caller `get_branch_multiplier` (`src/PowerNetworkMatrix.jl:351-373`) scans `reverse_parallel_branch_map` by name and already holds the matched object. Downstream (PowerFlows `post_processing.jl:1540`, `powersystems_utils.jl:81`) also hold the object but pass its name.

## Global Constraints

  - Commits and pushes are the controller's decision — implementers leave changes uncommitted and never `git add`.
  - All test runs use `julia --project=test …` from `/Users/jdlara/cache/psy6/PowerNetworkMatrices.jl`; ReTest name filter: `julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("<pattern>")'`.
  - Compile-check after each source edit: `julia --project=test -e 'using PowerNetworkMatrices'`.
  - Do NOT run the full-repo formatter script mid-plan (it aborts on a pre-existing `.claude/plans/2026-07-24-*.md` file); use JuliaFormatter directly on touched files with the options from `scripts/formatter/formatter_code.jl`. The full-suite/formatter gate is Task 5.
  - Julia style: no `isa`/type-check branches (dispatch instead; `x === y` identity comparison is fine), no ternaries, `iszero(x)`, explicit `function … end` + explicit `return`, terse comments (non-obvious WHY only).
  - Never export the new internals; exports live only in `src/PowerNetworkMatrices.jl` (nothing new is exported by this plan).
  - Contingency fixtures: `PSY.FixedForcedOutage(; outage_status = 1.0)` if a contingency must merely exist — never fabricate `GeometricDistributionForcedOutage` params. (The tests below use `NetworkModification` directly and need no outage attributes.)

* * *

### Task 1: Failing tests — equal-susceptance member collision

**Files:**

  - Modify: `test/test_ybus_reductions.jl` (extend `_mk_line_pst_parallel_system`, ~line 934)
  - Modify: `test/test_ybus_contingencies.jl` (append testset after "issue 305: contingencies on a Line ∥ PST group")

**Interfaces:**

  - Consumes: `_mk_detached_pst_fixture()` (test/test_ybus_reductions.jl, detached Line/PST components on arc (1, 2)); `PNM.NetworkReductionData()`; `PNM.add_to_branch_maps!(nr, arc, br)`; the "issue 305: contingencies" sibling testset's `NetworkModification` idiom (copy its matrix construction exactly).

  - Produces: `_mk_line_pst_parallel_system(; pst_r = 0.0, pst_x = 0.2)` — new `pst_x` keyword, default preserves all existing callers. Tests reference `PNM._compute_arc_ybus_delta(nr, arc_tuple, delta_b, component)` (4-arg, created in Task 2) and expect the 3-arg form to reject partial parallel outages.
  - [ ] **Step 1: Add the `pst_x` keyword to the fixture**

In `test/test_ybus_reductions.jl`, change the helper signature and the one place it uses `x = 0.2` for the PST circuit:

```text
function _mk_line_pst_parallel_system(; pst_r = 0.0, pst_x = 0.2)
    ...
                r = pst_r,
                x = pst_x,
```

(Keyword only — every existing call site keeps its current behavior.)

  - [ ] **Step 2: Write the failing tests**

Append to `test/test_ybus_contingencies.jl`:

```julia
@testset "parallel member outage resolves by identity, not susceptance value" begin
    # White-box: PST filed FIRST so the old value-scan would hit it before the line.
    # Line x=0.1 and PST (tap=1.0, x=0.1) both have b = 10.0 — deliberate collision.
    (line, _, _, _) = _mk_detached_pst_fixture()
    pst = PSY.TwoWindingTransformer(;
        name = "PSTx01",
        circuit = PSY.TransformerCircuit(;
            arc = PSY.get_arc(line), tap = 1.0, α = 0.15,
            available = true, active_power_flow = 0.0, reactive_power_flow = 0.0,
            rating = 1.0, base_power = 100.0, base_voltage_primary = 230.0,
            r = 0.0, x = 0.1,
        ),
        magnetizing_shunt = Complex(0.0, 0.0),
    )
    nr = PNM.NetworkReductionData()
    PNM.add_to_branch_maps!(nr, PSY.get_arc(pst), pst)
    PNM.add_to_branch_maps!(nr, PSY.get_arc(line), line)
    b_line = PNM.get_series_susceptance(line, PSY.SU)

    # Identity-resolved delta for tripping the line: the negated LINE pi-model
    # (symmetric), never the PST's asymmetric one.
    dy = PNM._compute_arc_ybus_delta(nr, (1, 2), -b_line, line)
    expected = PNM.ybus_branch_entries(line)
    @test dy[1] ≈ -expected[1]
    @test dy[2] ≈ -expected[2]
    @test dy[3] ≈ -expected[3]
    @test dy[4] ≈ -expected[4]
    @test dy[2] ≈ dy[3]

    # The arc-tuple-only entry point cannot know which member tripped: partial
    # parallel outages must be rejected loudly, not value-guessed.
    err = try
        PNM._compute_arc_ybus_delta(nr, (1, 2), -b_line)
        nothing
    catch e
        e
    end
    @test err isa ErrorException
    @test occursin("identity", err.msg) || occursin("component", err.msg)

    # A component that is not a member of the group is a loud error.
    (_, line2, _, _) = _mk_detached_pst_fixture()
    err2 = try
        PNM._compute_arc_ybus_delta(nr, (1, 2), -b_line, line2)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
end

@testset "black-box: tripping the line member of an equal-b Line ∥ PST group" begin
    sys = _mk_line_pst_parallel_system(; pst_x = 0.1)
    line = PSY.get_component(Line, sys, "L1")
    # Copy the matrix + NetworkModification construction from the sibling testset
    # "issue 305: contingencies on a Line ∥ PST group" in this file, verbatim.
    # With the fixture's pst_x = 0.1 the two members collide at b = 10.0.
    # Assertions on the single parallel ArcModification `m`:
    #   m.delta_b == -PNM.get_series_susceptance(line, PSY.SU)
    #   (m.delta_y12 ≈ m.delta_y21)                        # symmetric — the LINE's model
    #   m.delta_y11 ≈ -PNM.ybus_branch_entries(line)[1]
end
```

The second testset's construction idiom is deliberately deferred to the sibling testset in the same file (matrix type and accessor names must match it exactly); the three assertions listed are fixed requirements.

  - [ ] **Step 3: Run and verify the failures are the right ones**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("identity")'
```

Expected: the 4-arg `_compute_arc_ybus_delta` call fails with `MethodError` (method does not exist yet); the `@test_throws`-style checks fail because the 3-arg form currently *returns* the PST's entries instead of erroring. The black-box testset fails on `delta_y12 ≈ delta_y21` **only if** iteration order put the PST first — that testset is an order-independent invariant pin, not the red anchor; do not chase it if it happens to pass pre-fix. If ReTest doesn't find the testsets, check registration with `run_tests(dry = true)`.

### Task 2: Identity-threaded member deltas

**Files:**

  - Modify: `src/network_modification.jl:28-84` (replace `_compute_parallel_partial_ybus_delta`, adjust `_parallel_arc_ybus_delta`), `:102-129` (add 4-arg `_compute_arc_ybus_delta`), `:334-348` and `:517-537` (thread the component at the direct/parallel arms)
  - Test: Task 1's testsets

**Interfaces:**

  - Consumes: `ybus_branch_entries(br)` / `ybus_branch_entries(bp, nr)` (`src/Ybus.jl:438, 463`), `get_arc_tuple(br, nr)` and `get_arc_tuple(bp, nr)` (`src/common.jl:70, 76`), `_negated_pi_model` (`network_modification.jl:6-13`).

  - Produces: `_member_outage_ybus_delta(bp::AbstractBranchesParallel, nr::NetworkReductionData, component::PSY.ACTransmission)::NTuple{4, YBUS_ELTYPE}` and `_compute_arc_ybus_delta(nr::NetworkReductionData, arc_tuple::Tuple{Int, Int}, delta_b::Float64, component::PSY.ACTransmission)::NTuple{4, YBUS_ELTYPE}`. `_compute_parallel_partial_ybus_delta` is deleted.
  - [ ] **Step 1: Replace the value-scan with an identity + orientation-aware member delta**

In `src/network_modification.jl`, delete `_compute_parallel_partial_ybus_delta` (lines 28-47) and add:

```julia
"""
    _member_outage_ybus_delta(bp, nr, component) -> NTuple{4, YBUS_ELTYPE}

π-model Ybus delta for tripping `component`, a specific member of the parallel group `bp`,
resolved by object identity. Anti-parallel members are swapped into the group's key frame,
mirroring `ybus_branch_entries(bp, nr)`.
"""
function _member_outage_ybus_delta(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    component::PSY.ACTransmission,
)::NTuple{4, YBUS_ELTYPE}
    if !any(br === component for br in bp.branches)
        error(
            "Component $(get_name(component)) is not a member of the parallel group " *
            "$(get_name(bp)); cannot compute its outage delta.",
        )
    end
    entries = ybus_branch_entries(component)
    if get_arc_tuple(component, nr) != get_arc_tuple(bp, nr)
        entries = (entries[4], entries[3], entries[2], entries[1])
    end
    return _negated_pi_model(entries)
end
```

  - [ ] **Step 2: Make the arc-tuple-only partial-parallel path loud**

Replace `_parallel_arc_ybus_delta` (lines 74-84) so the partial branch errors instead of value-guessing:

```julia
# Parallel group: full outage negates the equivalent; a partial outage needs the tripped
# member's identity (see `_member_outage_ybus_delta`) — value-matching susceptances picks
# the wrong member when two members share a susceptance.
function _parallel_arc_ybus_delta(
    bp::AbstractBranchesParallel,
    nr::NetworkReductionData,
    delta_b::Float64,
)::NTuple{4, YBUS_ELTYPE}
    if _is_full_outage(delta_b, get_series_susceptance(bp, PSY.SU))
        return _negated_pi_model(ybus_branch_entries(bp, nr))
    end
    error(
        "Partial outage on parallel group $(get_name(bp)) requires the tripped " *
        "component's identity; construct the modification from the branch component " *
        "instead of the arc tuple.",
    )
end
```

  - [ ] **Step 3: Add the component-aware `_compute_arc_ybus_delta`**

After the existing 3-arg method (ends ~line 129), add:

```julia
"""
    _compute_arc_ybus_delta(nr, arc_tuple, delta_b, component) -> NTuple{4, YBUS_ELTYPE}

Component-aware variant: on a parallel-map arc the tripped member is resolved by object
identity instead of susceptance value. Direct and series arcs delegate to the arc-tuple
handlers (the direct-map entry *is* the component; series arcs carry aggregate deltas).
"""
function _compute_arc_ybus_delta(
    nr::NetworkReductionData,
    arc_tuple::Tuple{Int, Int},
    delta_b::Float64,
    component::PSY.ACTransmission,
)::NTuple{4, YBUS_ELTYPE}
    if haskey(nr.parallel_branch_map, arc_tuple)
        return _member_outage_ybus_delta(
            nr.parallel_branch_map[arc_tuple],
            nr,
            component,
        )
    end
    return _compute_arc_ybus_delta(nr, arc_tuple, delta_b)
end
```

  - [ ] **Step 4: Thread the component at the classification call sites**

Exactly four calls change, all in `src/network_modification.jl` — each gains the component as a fourth argument:

```text
:340  dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_arc, component)
:347  dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_circuit, component)
:523  dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_arc, branch)
:530  dy11, dy12, dy21, dy22 = _compute_arc_ybus_delta(nr, arc_tuple, -b_circuit, branch)
```

The series arms (`:278`, `:536`) and the arc-tuple constructor (`:142`) keep the 3-arg form — they carry aggregate deltas or full outages and never resolve a member.

  - [ ] **Step 5: Compile-check, then run Task 1's tests — expect PASS**

```bash
julia --project=test -e 'using PowerNetworkMatrices'
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("identity")'
```

  - [ ] **Step 6: Run the contingency/modification regression suites**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests(r"conting|modification|modf|MODF")'
```

Expected: PASS. RTS/matpower double-circuit fixtures have identical twin members — the identity path returns the same entries the value scan did for them. If a test asserted the *error text* of the deleted scan ("Could not resolve partial parallel outage"), update it to the new message and note it in the report.

### Task 3: Anti-parallel member orientation regression pin

**Files:**

  - Modify: `test/test_ybus_contingencies.jl` (append)

**Interfaces:**

  - Consumes: the ZIB-merge fixture pattern from `test/test_ybus_reductions.jl:725-823` ("anti-parallel asymmetric group" testset) — 3-bus system where a zero-impedance branch merges bus 3 into bus 1, folding a branch in anti-parallel; `NetworkModification` idiom from this file's sibling testsets; `PNM.ybus_branch_entries(bp, nr)`.

  - Produces: nothing new — pins the orientation swap in `_member_outage_ybus_delta`.
  - [ ] **Step 1: Write the test**

```julia
@testset "anti-parallel asymmetric member outage delta is swapped into the key frame" begin
    # 3-bus: L1 on (1,2), ZIB on (1,3) merges bus 3 into 1, transformer T on (2,3)
    # becomes anti-parallel to L1. tap = 1.05 makes T's diagonals asymmetric
    # (Y11 != Y22); α = 0 keeps it non-phase-shifting so contingencies are allowed.
    # Build the system exactly like the "anti-parallel asymmetric group" testset in
    # test/test_ybus_reductions.jl:725-823, with the transformer's α = 0.0.
    # ... (system construction copied from that testset, α = 0.0) ...

    ybus = Ybus(sys)
    nr = ybus.network_reduction_data
    (arc_key, bp) = first(PNM.get_parallel_branch_map(nr))
    t = PSY.get_component(PSY.TwoWindingTransformer, sys, "PST")

    # Copy the sibling testsets' matrix + NetworkModification(mat, t) construction.
    # Oracle: removal delta == (remaining member alone) - (full group), both already
    # oriented in the key frame by ybus_branch_entries(bp, nr) and the surviving
    # member's own frame (L1 seeds the key, so no swap on the survivor).
    l1 = PSY.get_component(Line, sys, "L1")
    group_entries = PNM.ybus_branch_entries(bp, nr)
    remaining_entries = PNM.ybus_branch_entries(l1)
    m = only(mod.arc_modifications)
    @test m.delta_y11 ≈ remaining_entries[1] - group_entries[1]
    @test m.delta_y12 ≈ remaining_entries[2] - group_entries[2]
    @test m.delta_y21 ≈ remaining_entries[3] - group_entries[3]
    @test m.delta_y22 ≈ remaining_entries[4] - group_entries[4]
end
```

The system-construction block and the `NetworkModification` idiom are copies from the two named sibling testsets (the plan defers to them so the file keeps one idiom); the oracle assertions are fixed. Approximate comparisons must account for `YBUS_ELTYPE` being `ComplexF32` — if `≈` fails on precision alone, compare with `atol = 1e-5`.

  - [ ] **Step 2: Run — expect PASS (regression pin)**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("anti-parallel asymmetric member")'
```

This passes because Task 2's `_member_outage_ybus_delta` includes the swap. To confirm the pin bites, temporarily comment out the swap branch and confirm `delta_y11`/`delta_y22` fail, then restore (mention the check in the report; do not commit the mutation).

### Task 4: Identity-based `compute_parallel_multiplier`

**Files:**

  - Modify: `src/BranchesParallel.jl:75-90` (replace the name-matching body; add the identity method)
  - Modify: `src/PowerNetworkMatrix.jl:351-373` (`get_branch_multiplier` passes the object; ambiguous names error)
  - Test: `test/test_ybus_reductions.jl` (append)

**Interfaces:**

  - Consumes: `get_series_susceptance(br, PSY.SU)`, `get_name` (PNM's, handles groups and `ThreeWindingTransformerCircuit`), `_mk_detached_pst_fixture()`.

  - Produces: `compute_parallel_multiplier(bp::AbstractBranchesParallel, branch::PSY.ACTransmission)::Float64` (identity); the existing `compute_parallel_multiplier(bp, branch_name::String)` keeps its signature (PowerFlows calls it with names) but resolves the name to exactly one member and errors on 0 or ≥2 matches instead of double-counting.
  - [ ] **Step 1: Write the failing tests**

Append to `test/test_ybus_reductions.jl`:

```julia
@testset "parallel multiplier resolves members by identity" begin
    (line, _, pst1, _) = _mk_detached_pst_fixture()
    # line: x=0.1 -> b=10; pst1: tap=1.0, x=0.2 -> b=5
    group = PNM.MixedBranchesParallel([line, pst1])
    @test PNM.compute_parallel_multiplier(group, line) ≈ 10.0 / 15.0
    @test PNM.compute_parallel_multiplier(group, pst1) ≈ 5.0 / 15.0

    # Name collision across concrete types: was silently double-counted, now loud.
    pst_same_name = PSY.TwoWindingTransformer(;
        name = PSY.get_name(line),
        circuit = PSY.TransformerCircuit(;
            arc = PSY.get_arc(pst1), tap = 1.0, α = 0.0,
            available = true, active_power_flow = 0.0, reactive_power_flow = 0.0,
            rating = 1.0, base_power = 100.0, base_voltage_primary = 230.0,
            r = 0.0, x = 0.2,
        ),
        magnetizing_shunt = Complex(0.0, 0.0),
    )
    collided = PNM.MixedBranchesParallel([line, pst_same_name])
    err = try
        PNM.compute_parallel_multiplier(collided, PSY.get_name(line))
        nothing
    catch e
        e
    end
    @test err isa ErrorException

    # Unambiguous name still resolves (delegates to the identity method).
    @test PNM.compute_parallel_multiplier(group, PSY.get_name(line)) ≈ 10.0 / 15.0

    # Non-member is a loud error.
    (_, line2, _, _) = _mk_detached_pst_fixture()
    err2 = try
        PNM.compute_parallel_multiplier(group, line2)
        nothing
    catch e
        e
    end
    @test err2 isa ErrorException
end
```

NOTE: the susceptance values above assume `get_series_susceptance` on detached fixture components works via stored `r`/`x` (b = 1/(tap·x)); the detached-circuit `base_value === nothing` caveat applies to explicit-units getters, and `PSY.SU` on these per-unit fixture values resolves to the same numbers. If the detached read errors, attach the components to a `System(100.0)` with the fixture's buses first (mirror `_mk_line_pst_parallel_system`) and keep the same assertions.

  - [ ] **Step 2: Run to verify the right failures**

```bash
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("multiplier resolves")'
```

Expected: identity-method calls fail with `MethodError`; the collision case fails because the current code returns `(10+5)/15 = 1.0` instead of erroring.

  - [ ] **Step 3: Implement**

Replace `compute_parallel_multiplier` in `src/BranchesParallel.jl:75-90` with:

```julia
"""
    compute_parallel_multiplier(parallel_branch_set, branch) -> Float64

Susceptance fraction `b_branch / b_total` of one member of a parallel group. The member is
resolved by object identity; passing a component that is not in the group is an error.
"""
function compute_parallel_multiplier(
    parallel_branch_set::AbstractBranchesParallel,
    branch::PSY.ACTransmission,
)
    b_total = 0.0
    b_branch = 0.0
    found = false
    for br in parallel_branch_set
        # `get_series_susceptance` (see BranchAdmittance.jl) is tap-aware for
        # two-winding transformers and dispatches PNM's three-winding winding wrapper.
        if br === branch
            b_branch = get_series_susceptance(br, PSY.SU)
            found = true
        end
        b_total += get_series_susceptance(br, PSY.SU)
    end
    if !found
        error(
            "Branch $(get_name(branch)) is not a member of parallel group " *
            "$(get_name(parallel_branch_set)).",
        )
    end
    return b_branch / b_total
end

# Name-based lookup kept for callers that only hold a name (PTDF row API, PowerFlows).
# PSY names are unique per concrete type only, so a name may match several members of a
# mixed group; that was silently double-counted before — now it must resolve to exactly one.
function compute_parallel_multiplier(
    parallel_branch_set::AbstractBranchesParallel,
    branch_name::String,
)
    matches = PSY.ACTransmission[]
    for br in parallel_branch_set
        if get_name(br) == branch_name
            push!(matches, br)
        end
    end
    if length(matches) != 1
        error(
            "Branch name $(branch_name) matches $(length(matches)) members of parallel " *
            "group $(get_name(parallel_branch_set)); resolve by component identity.",
        )
    end
    return compute_parallel_multiplier(parallel_branch_set, first(matches))
end
```

In `src/PowerNetworkMatrix.jl` `get_branch_multiplier` (`:361-368`), pass the matched object and make cross-group name ambiguity loud:

```julia
if !isempty(nr.reverse_parallel_branch_map)
    matched = nothing
    matched_arc = (0, 0)
    n_matches = 0
    for (k, v) in nr.reverse_parallel_branch_map
        if branch_name == get_name(k)
            matched = k
            matched_arc = v
            n_matches += 1
        end
    end
    if n_matches > 1
        error(
            "Branch name $(branch_name) matches $(n_matches) parallel-group members; " *
            "name-based lookup is ambiguous.",
        )
    end
    if n_matches == 1
        parallel_branch_set = nr.parallel_branch_map[matched_arc]
        multiplier = compute_parallel_multiplier(parallel_branch_set, matched)
        return multiplier, matched_arc
    end
end
```

(`matched = nothing` here is a local accumulator guarded by `n_matches`, not a returned absence sentinel — the function still exits via the error at `:371` when nothing matched. Note the switch from `PSY.get_name(k)` to PNM's `get_name(k)`: `k` can be a `ThreeWindingTransformerCircuit`, which only PNM's `get_name` handles.)

  - [ ] **Step 4: Compile-check, run the new testset, then the PTDF row-lookup suites**

```bash
julia --project=test -e 'using PowerNetworkMatrices'
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests("multiplier resolves")'
julia --project=test -e 'using PowerNetworkMatrices; include("test/PowerNetworkMatricesTests.jl"); run_tests(r"PTDF|ptdf")'
```

Expected: PASS (real fixtures have unique names within groups, so the name path behaves identically for them).

### Task 5: Formatter, full suite, downstream note

**Files:**

  - Modify: only formatting fallout in files this plan touched.

  - [ ] **Step 1: Format touched files** with JuliaFormatter using the exact options in `scripts/formatter/formatter_code.jl` (per-file; not the full-repo walk).
  - [ ] **Step 2: Full PNM suite**

```bash
julia --project=test test/runtests.jl
```

Expected: PASS. Report counts verbatim.

  - [ ] **Step 3: Record the downstream contract note for the controller's summary** (no code): PowerFlows calls `compute_parallel_multiplier(arc_entry, PNM.get_name(br))` (`post_processing.jl:1540`, `powersystems_utils.jl:81`) while holding `br` — after this plan, same-named mixed-group members make that call error loudly instead of double-counting. PF should migrate to the identity method opportunistically; behavior is unchanged for unique names.

* * *

## Self-review notes

  - **Spec coverage:** identity defect → Tasks 1-2; orientation defect (found during planning) → Task 2 Step 1 + Task 3 pin; the reviewer's sibling name-based finding → Task 4; TWTC-in-parallel unreliability → fixed structurally by identity resolution (no value matching), no dedicated task needed since `_direct_arc_ybus_delta`'s TWTC method still covers the direct case.
  - **Deliberate non-goals:** series-arc partial semantics (`_series_arc_ybus_delta` unchanged); the name-based PTDF row API surface (kept, hardened); PF call-site migration (downstream, noted in Task 5 Step 3).
  - **Type consistency:** `_member_outage_ybus_delta(bp, nr, component)` (Task 2) is what the 4-arg `_compute_arc_ybus_delta` calls; `compute_parallel_multiplier(bp, branch::PSY.ACTransmission)` (Task 4) is what both the hardened name method and `get_branch_multiplier` delegate to. Task 1's tests reference only Task 2 names.
  - **Formatter safety:** incomplete code snippets in this plan use `text` fences; all `julia` fences are complete, parseable expressions.
