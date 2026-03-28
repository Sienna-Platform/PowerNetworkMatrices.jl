# MODF vs LODF Reduction Combination Tests — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify VirtualMODF post-contingency PTDF rows match the LODF-based ground truth for every combination of network reduction (none, radial, degree-two, radial+degree-two) and branch type (direct, parallel, series).

**Architecture:** Single test file with a shared helper function and 4 testsets. Each testset builds matrices with the same `network_reductions` kwarg, picks one representative branch per type from `NetworkReductionData`, and asserts the N-1 identity `post_PTDF[m,:] = pre_PTDF[m,:] + LODF[m,e] * pre_PTDF[e,:]`.

**Tech Stack:** Julia, ReTest, PowerNetworkMatrices, PowerSystems, PowerSystemCaseBuilder

---

### Task 1: Create test file with helper function

**Files:**
- Create: `test/test_modf_lodf_reductions.jl`

- [ ] **Step 1: Write the helper function and the first testset (no reductions, direct branch)**

Create `test/test_modf_lodf_reductions.jl` with the full content below. The helper
`verify_modf_lodf_identity` encapsulates the MODF-vs-LODF comparison logic. The
first testset validates a direct branch outage with no network reductions.

```julia
"""
Helper: verify the N-1 identity for a single contingency.

Builds a ContingencySpec, computes the LODF reference (standard column for full
outage, `get_partial_lodf_row` for partial), and asserts:

    VirtualMODF[m, ctg] ≈ PTDF[m, :] + lodf_col[m] * PTDF[e, :]

for every monitored arc m.
"""
function verify_modf_lodf_identity(
    vmodf::VirtualMODF,
    vlodf::VirtualLODF,
    ptdf::PTDF,
    arc_idx::Int,
    delta_b::Float64;
    atol = 1e-6,
)
    n_arcs = size(vlodf, 1)
    b_arc = vmodf.arc_susceptances[arc_idx]

    # Build and register ContingencySpec
    ctg_uuid = Base.UUID(UInt128(hash((arc_idx, delta_b))))
    ctg = ContingencySpec(
        ctg_uuid,
        "test_arc_$(arc_idx)",
        [ArcModification(arc_idx, delta_b)],
    )
    vmodf.contingency_cache[ctg_uuid] = ctg

    # LODF reference: standard column for full outage, partial otherwise
    if isapprox(delta_b, -b_arc; atol = 1e-12)
        lodf_col = [vlodf[m, arc_idx] for m in 1:n_arcs]
    else
        lodf_col = PNM.get_partial_lodf_row(vlodf, arc_idx, delta_b)
    end

    # Verify identity for every monitored arc
    for m in 1:n_arcs
        modf_row = vmodf[m, ctg]
        expected = ptdf[m, :] .+ lodf_col[m] .* ptdf[arc_idx, :]
        @test isapprox(modf_row, expected; atol = atol)
    end

    # Clean up Woodbury cache so next call starts fresh
    empty!(vmodf.woodbury_cache)
    return
end
```

- [ ] **Step 2: Run tests to verify the helper compiles (no testsets yet)**

Run:
```bash
julia --project=test -e '
    using Pkg; Pkg.develop(path=".")
    include("test/PowerNetworkMatricesTests.jl")
'
```

Expected: No errors (the helper is defined but not called yet — it will be pulled
in by the `test_*.jl` glob in `PowerNetworkMatricesTests.jl`).

- [ ] **Step 3: Commit**

```bash
git add test/test_modf_lodf_reductions.jl
git commit -m "test: add verify_modf_lodf_identity helper for MODF-LODF reduction tests"
```

---

### Task 2: Testset — no reductions (direct + parallel)

**Files:**
- Modify: `test/test_modf_lodf_reductions.jl`

- [ ] **Step 1: Add the "no reductions" testset**

Append the following to `test/test_modf_lodf_reductions.jl`:

```julia
@testset "MODF vs LODF: no reductions" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[]

    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    # --- Direct branch outage ---
    @testset "direct branch" begin
        arc_tuple = first(keys(nrd.direct_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    # --- Parallel single-circuit outage ---
    @testset "parallel single-circuit" begin
        arc_tuple = first(keys(nrd.parallel_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end
```

- [ ] **Step 2: Run the testset**

Run:
```bash
julia --project=test -e '
    using Pkg; Pkg.develop(path=".")
    include("test/PowerNetworkMatricesTests.jl")
    retest("MODF vs LODF: no reductions")
'
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/test_modf_lodf_reductions.jl
git commit -m "test: MODF vs LODF identity with no reductions (direct + parallel)"
```

---

### Task 3: Testset — radial reduction (direct + parallel)

**Files:**
- Modify: `test/test_modf_lodf_reductions.jl`

- [ ] **Step 1: Add the "radial reduction" testset**

Append the following to `test/test_modf_lodf_reductions.jl`:

```julia
@testset "MODF vs LODF: radial reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[RadialReduction()]

    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    # --- Direct branch outage ---
    @testset "direct branch" begin
        arc_tuple = first(keys(nrd.direct_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    # --- Parallel single-circuit outage ---
    @testset "parallel single-circuit" begin
        arc_tuple = first(keys(nrd.parallel_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end
```

- [ ] **Step 2: Run the testset**

Run:
```bash
julia --project=test -e '
    using Pkg; Pkg.develop(path=".")
    include("test/PowerNetworkMatricesTests.jl")
    retest("MODF vs LODF: radial reduction")
'
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/test_modf_lodf_reductions.jl
git commit -m "test: MODF vs LODF identity with radial reduction (direct + parallel)"
```

---

### Task 4: Testset — degree-two reduction (direct + parallel + series)

**Files:**
- Modify: `test/test_modf_lodf_reductions.jl`

- [ ] **Step 1: Add the "degree-two reduction" testset**

Append the following to `test/test_modf_lodf_reductions.jl`:

```julia
@testset "MODF vs LODF: degree-two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[DegreeTwoReduction()]

    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    # --- Direct branch outage ---
    @testset "direct branch" begin
        arc_tuple = first(keys(nrd.direct_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    # --- Parallel single-circuit outage ---
    @testset "parallel single-circuit" begin
        arc_tuple = first(keys(nrd.parallel_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    # --- Series segment outage ---
    @testset "series segment" begin
        arc_tuple = first(keys(nrd.series_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        series_chain = nrd.series_branch_map[arc_tuple]
        segment = first(series_chain)
        delta_b = PNM._compute_series_outage_delta_b(series_chain, segment)
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end
```

- [ ] **Step 2: Run the testset**

Run:
```bash
julia --project=test -e '
    using Pkg; Pkg.develop(path=".")
    include("test/PowerNetworkMatricesTests.jl")
    retest("MODF vs LODF: degree-two reduction")
'
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/test_modf_lodf_reductions.jl
git commit -m "test: MODF vs LODF identity with degree-two reduction (direct + parallel + series)"
```

---

### Task 5: Testset — radial + degree-two reduction (direct + parallel + series)

**Files:**
- Modify: `test/test_modf_lodf_reductions.jl`

- [ ] **Step 1: Add the "radial + degree-two" testset**

Append the following to `test/test_modf_lodf_reductions.jl`:

```julia
@testset "MODF vs LODF: radial + degree-two reduction" begin
    sys = PSB.build_system(PSSEParsingTestSystems, "psse_14_network_reduction_test_system")
    reductions = NetworkReduction[RadialReduction(), DegreeTwoReduction()]

    ptdf = PTDF(sys; network_reductions = reductions)
    vlodf = VirtualLODF(sys; network_reductions = reductions)
    vmodf = VirtualMODF(sys; network_reductions = reductions)
    nrd = get_network_reduction_data(vmodf)

    # --- Direct branch outage ---
    @testset "direct branch" begin
        arc_tuple = first(keys(nrd.direct_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        delta_b = -vmodf.arc_susceptances[arc_idx]
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    # --- Parallel single-circuit outage ---
    @testset "parallel single-circuit" begin
        arc_tuple = first(keys(nrd.parallel_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        parallel = nrd.parallel_branch_map[arc_tuple]
        b_circuit = PSY.get_series_susceptance(first(parallel.branches))
        delta_b = -b_circuit
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end

    # --- Series segment outage ---
    @testset "series segment" begin
        arc_tuple = first(keys(nrd.series_branch_map))
        arc_idx = vmodf.lookup[1][arc_tuple]
        series_chain = nrd.series_branch_map[arc_tuple]
        segment = first(series_chain)
        delta_b = PNM._compute_series_outage_delta_b(series_chain, segment)
        verify_modf_lodf_identity(vmodf, vlodf, ptdf, arc_idx, delta_b)
    end
end
```

- [ ] **Step 2: Run the testset**

Run:
```bash
julia --project=test -e '
    using Pkg; Pkg.develop(path=".")
    include("test/PowerNetworkMatricesTests.jl")
    retest("MODF vs LODF: radial .* degree-two reduction")
'
```

Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/test_modf_lodf_reductions.jl
git commit -m "test: MODF vs LODF identity with radial + degree-two reduction (direct + parallel + series)"
```

---

### Task 6: Run full test suite

**Files:** None (validation only)

- [ ] **Step 1: Run the full test suite to check for regressions**

Run:
```bash
julia --project=test test/runtests.jl
```

Expected: All tests pass, including the 4 new testsets.

- [ ] **Step 2: Run the formatter**

Run:
```bash
julia -e 'include("scripts/formatter/formatter_code.jl")'
```

Expected: No formatting changes (or apply any changes if needed).

- [ ] **Step 3: Final commit (if formatter changed anything)**

```bash
git diff --exit-code || (git add test/test_modf_lodf_reductions.jl && git commit -m "style: format test_modf_lodf_reductions.jl")
```
