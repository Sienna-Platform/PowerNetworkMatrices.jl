# PowerNetworkMatrices.jl - Project Guide

> **Development Guidelines:** Always load [Sienna.md](./Sienna.md) development preferences, style conventions, and best practices for projects using Sienna. Before running tests confirm that the [Sienna.md](./Sienna.md) file has been read.

## Package Role

PowerNetworkMatrices.jl constructs classic power systems network matrices (Ybus, PTDF, LODF) for power flow analysis, sensitivity analysis, and contingency studies. Part of the Sienna-Platform ecosystem, it provides computational building blocks for optimization and analysis packages.

## Design Objectives

### Primary: Performance

Efficient computation of power network matrices for large-scale systems. Supports sparse and virtual (on-demand) matrix implementations for memory efficiency. Network reduction algorithms decrease computational complexity by 30-60%. **All code must be written with performance in mind.**

### Package-Specific Design Principles

  - Support multiple electrical islands (subnetworks) transparently
  - Provide both dense and memory-efficient virtual matrix options
  - Leverage KLU factorization for sparse linear solves

### Source Layout (`src/`)

#### Core

  - **PowerNetworkMatrices.jl**: module entry point and include order
  - **PowerNetworkMatrix.jl**: abstract base type implementing array interface
  - **PowerflowMatrixTypes.jl**: shared matrix type aliases (DC_PTDF_Matrix, DC_ABA_Matrix_Factorized, AC_Ybus_Matrix, etc.)
  - **definitions.jl**: constants (solvers, cache limits, tolerances)
  - **linalg_settings.jl**: linear-algebra backend selection
  - **common.jl**: shared utility functions and getters
  - **system_utils.jl**: PowerSystems integration helpers
  - **serialization.jl**: HDF5 I/O support

#### Network Matrices

  - **Ybus.jl**: nodal admittance matrix (Complex, sparse)
  - **YbusACBranches.jl**: per-branch Ybus Pi-model contributions
  - **ArcAdmittanceMatrix.jl**: arc-level admittance
  - **IncidenceMatrix.jl**: bus-branch connectivity (Int8, sparse)
  - **AdjacencyMatrix.jl**: bus connectivity structure
  - **BA_ABA_matrices.jl**: BA_Matrix and ABA_Matrix (susceptance forms for DC power flow)
  - **ptdf_calculations.jl**: PTDF computation (KLU, Dense, MKL, Apple)
  - **lodf_calculations.jl**: LODF calculation from PTDF
  - **row_cache.jl**: LRU caching for virtual matrices
  - **virtual_ptdf_calculations.jl**: on-demand PTDF row computation
  - **virtual_lodf_calculations.jl**: on-demand LODF row computation
  - **virtual_modf_calculations.jl**: on-demand MODF (modification distribution factors)

#### Network Modification & Contingencies

  - **modf_definitions.jl**: MODF-related types (`ArcModification`, `ShuntModification`, `ContingencySpec`)
  - **network_modification.jl**: `NetworkModification` container and application logic
  - **virtual_ptdf_modification.jl**: apply modifications to virtual PTDF rows
  - **woodbury_kernel.jl**: Woodbury-identity updates for low-rank modifications
  - **ybus_contingencies.jl**: Ybus contingency application helpers

#### Network Reduction

  - **NetworkReduction.jl**: abstract base for reductions
  - **NetworkReductionData.jl**: tracks bus/branch mappings
  - **ReductionContainer.jl**: tracks applied reduction algorithms
  - **radial_reduction.jl**: eliminates dangling buses
  - **degree_two_reduction.jl**: eliminates degree-2 buses
  - **ward_reduction.jl**: preserves study area, reduces external
  - **BranchesParallel.jl**: parallel branch equivalencing
  - **BranchesSeries.jl**: series chain compression
  - **ThreeWindingTransformerWinding.jl**: 3-winding transformer support
  - **EquivalentBranch.jl**: equivalent branch representation

#### Connectivity

  - **connectivity_checks.jl**: electrical island detection
  - **subnetworks.jl**: multi-island handling

### `ext/`

  - **MKLPardisoExt.jl**: MKL-Pardiso sparse solver (Windows/Linux)
  - **AppleAccelerateExt.jl**: native macOS BLAS acceleration

### `test/`

Test files validating against PSS/E and Matpower cases

### `docs/`

Documentation source

## Consumed By

  - **PowerSimulations.jl**: production cost modeling, unit commitment, economic dispatch
  - **PowerFlows.jl**: power flow analysis
  - **PowerSystemsInvestmentsPortfolios.jl**: capacity expansion portfolios

## Dependencies

### Primary

  - **PowerSystems.jl**: power system data structures and components
  - **InfrastructureSystems.jl**: shared utilities for NREL packages

### Computational

  - **KLU.jl**: default sparse LU factorization solver
  - **SparseArrays**: sparse matrix storage (stdlib)
  - **LinearAlgebra**: matrix operations (stdlib)

### Data

  - **HDF5.jl**: matrix serialization
  - **DataStructures.jl**: SortedDict and utilities

### Optional Extensions

  - **MKL + Pardiso**: high-performance sparse factorization
  - **AppleAccelerate**: native macOS dense linear algebra

## Core Abstractions

### Base Type

**`PowerNetworkMatrix{T}`**: Abstract type inheriting from `AbstractArray{T, 2}`. All matrices implement standard Julia array indexing with support for bus numbers and branch identifiers directly. Provides axes, lookup dictionaries, and subnetwork handling.

### Matrix Types

#### Network Model

  - **Ybus**: N_buses × N_buses complex sparse nodal admittance matrix
  - **IncidenceMatrix**: N_branches × N_buses Int8 sparse bus-branch connectivity
  - **AdjacencyMatrix**: N_buses × N_buses Int8 sparse bus connectivity
  - **BA_Matrix**: N_buses × N_branches Float64 sparse susceptance-weighted incidence
  - **ABA_Matrix**: N_buses × N_buses Float64 sparse factorized susceptance matrix
  - **ArcAdmittanceMatrix**: N_arcs × N_buses complex sparse arc admittance

#### Sensitivity Analysis

  - **PTDF**: N_arcs × N_buses power transfer distribution factors (transposed storage)
  - **LODF**: N_arcs × N_arcs line outage distribution factors (diagonal = -1.0)
  - **VirtualPTDF**: on-demand PTDF with LRU row caching for memory efficiency
  - **VirtualLODF**: on-demand LODF with LRU row caching
  - **VirtualMODF**: on-demand modification distribution factors using Woodbury updates over a base PTDF

### Network Reduction

  - **NetworkReduction**: abstract base for reduction strategies
  - **RadialReduction**: eliminates radial (dangling) buses
  - **DegreeTwoReduction**: eliminates degree-two buses
  - **WardReduction**: preserves study area while reducing external network
  - **NetworkReductionData**: tracks all reduction mappings and equivalents

### Network Modification

  - **ArcModification**: describes an arc (branch) change with precomputed Ybus Pi-model deltas
  - **ShuntModification**: describes a shunt admittance change at a bus
  - **ContingencySpec**: named bundle of modifications representing a contingency case
  - **NetworkModification**: container applied to matrices/Ybus to produce post-modification state
  - **apply_ybus_modification / compute_ybus_delta / apply_woodbury_correction**: helpers for applying modifications to Ybus and to factorized PTDF solves

### Key Patterns

  - **Indexing**: `matrix[bus_num, branch_tuple]` auto-maps to internal indices
  - **Subnetworks**: `subnetwork_axes` Dict maps reference buses to island components
  - **Caching**: VirtualPTDF/LODF use LRU cache (default 100 MiB) for row storage
  - **Solvers**: KLU (default), Dense, MKLPardiso, AppleAccelerate via extensions

## Test Patterns

  - **Location**: `test/`
  - **Dev local**: `julia --project=test -e 'using Pkg; Pkg.develop(path=".")'`
  - **Runner**: `julia --project=test test/runtests.jl`
  - **Test data**: uses PowerSystemCaseBuilder.jl for standard IEEE/Matpower cases
  - **Validation**: results compared against PSS/E and Matpower reference implementations

## Code Conventions

**Style Guide:** [Sienna-Platform Style Guide](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/style/)

### Formatter

  - **Tool**: JuliaFormatter
  - **Command**: `julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'`

### Key Rules

  - **Constructors**: use `function Foo()` not `Foo() = ...`
  - **Asserts**: prefer `InfrastructureSystems.@assert_op` over `@assert`
  - **Globals**: UPPER_CASE for constants
  - **Exports**: all exports in main module file
  - **Comments**: complete sentences, describe why not how
  - **Sparse matrices**: use `SparseMatrixCSC` throughout, avoid dense when possible

## Documentation Practices

**Framework:** [Diataxis](https://diataxis.fr/)
**Sienna Guide:** [Documentation Best Practices](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/docs_best_practices/explanation/)

### Docstring Requirements

  - **Scope**: all elements of public interface
  - **Include**: function signatures and arguments list
  - **Automation**: `DocStringExtensions.TYPEDSIGNATURES`
  - **See also**: add links for functions with same name (multiple dispatch)

### Docs Organization

Documentation is organized following the [Diataxis](https://diataxis.fr/) framework under `docs/src/`:

  - **tutorials/**: learning-oriented walkthroughs (PTDF, LODF, VirtualPTDF/LODF, Radial/Degree-Two Reduction, etc.)
  - **how_to_guides/**: task-oriented recipes (`compute_network_matrices.md`, `choose_linear_solver.md`)
  - **explanation/**: conceptual background (DC power-flow approximation, network-reduction theory, computational considerations)
  - **reference/**: API and matrix overview (`public.md` uses `@autodocs` with `Public=true`; `network_matrices_overview.md` summarizes the matrix types)

## Common Tasks

```bash
# Develop locally
julia --project=test -e 'using Pkg; Pkg.develop(path=".")'

# Run tests
julia --project=test test/runtests.jl

# Build documentation
julia --project=docs docs/make.jl

# Format code
julia --project=scripts/formatter -e 'include("scripts/formatter/formatter_code.jl")'

# Check formatting
git diff --exit-code

# Instantiate test environment
julia --project=test -e 'using Pkg; Pkg.instantiate()'
```

## Contribution Workflow

  - **Branch naming**: `feature/description` or `fix/description` (branches in main repo)
  - **Main branch**: `main`

### PR Process

 1. Create a feature branch in the main repo
 2. Make changes following the style guide
 3. Run formatter before committing
 4. Ensure tests pass
 5. Submit pull request

## Package-Specific Troubleshooting

### Subnetwork Errors

  - **Symptom**: Matrix construction fails with disconnected network
  - **Diagnosis**: Check for isolated buses or multiple electrical islands
  - **Solution**: Use `find_subnetworks()` to identify islands, ensure each has reference bus

### Memory Issues

  - **Symptom**: Out of memory for large PTDF/LODF matrices
  - **Solution**: Use `VirtualPTDF`/`VirtualLODF` with row caching instead

## AI Agent Guidance

### Code Generation Priorities

  - Performance matters - use concrete types in hot paths
  - Use sparse matrices (`SparseMatrixCSC`) by default
  - Apply anti-patterns list with judgment (not exhaustively everywhere)
  - Run formatter on all changes
  - Add docstrings to public interface elements
  - Consider type stability in performance-critical functions

### Domain Knowledge

  - PTDF[i,j] represents sensitivity of flow on arc i to injection at bus j
  - LODF[i,j] represents flow redistribution on arc i when arc j trips
  - Ybus diagonal elements are sum of admittances connected to that bus
  - Network reductions preserve electrical equivalence at retained buses
  - Virtual matrices trade computation time for memory efficiency

### When Modifying Code

  - Read existing code patterns before making changes
  - Maintain consistency with existing style
  - Prefer failing fast with clear errors over silent failures
  - Consider impact on subnetwork handling (multiple islands)
  - Test with both single-island and multi-island systems
