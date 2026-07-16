---
name: sienna-documentation
description: >
  Write and structure Sienna Julia package documentation using the Diátaxis framework
  and Documenter.jl. Use when creating or revising docs/src pages, organizing tutorials
  vs how-to vs reference vs explanation, integrating pages into docs/make.jl, reviewing
  docstrings, or editing PowerSystems Model Library / generated model documentation—not
  for the Jekyll Sienna website unless explicitly in the Sienna repo.
---

# Sienna Documentation

Guides writing **documentation pages** (`docs/src/**/*.md`) for packages in the [Sienna Platform](https://github.com/Sienna-Platform) ecosystem. Merges Diátaxis content strategy with Sienna/Documenter.jl conventions.

> **Not for Julia docstrings alone.** For API docstrings in `src/`, follow [InfrastructureSystems docstring guidance](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/docs_best_practices/how-to/write_docstrings_org_api/) and surface them via Documenter `@docs` blocks on reference pages.

## When to use

Use when:

  - Writing or revising a documentation page in `docs/src/`
  - Deciding whether content belongs in a tutorial, how-to, reference, or explanation
  - Adding a page to `docs/make.jl` navigation
  - Reorganizing package docs to Diátaxis structure

Do **not** use when:

  - Editing the public **Sienna website** (Jekyll under the `Sienna/` repo) unless that is the active project
  - Writing **unit tests** (use package test conventions)
  - Generating **PowerGraphics Weave reports** (separate from Documenter)

## Workspace context

This workspace ([`Sienna_repos`](../../)) is a **multi-repo root**, not one git repository. Before propagating patterns across packages:

 1. Check [`cursor-cloud-propagate/repos.manifest.json`](../../cursor-cloud-propagate/repos.manifest.json) for `repos` vs `excludedFromWorkspace`
 2. **SiennaTemplate.jl** — canonical layout; avoid adding substantive docs unless the task says to
 3. **Sienna** (website) — Jekyll + `SiennaDocs/`; different toolchain
 4. **MultiDocumenter.jl**, **PowerSystemsTestData** — excluded from standard propagation

## Step 1: Classify with Diátaxis

Every page belongs to **exactly one** category:

| Category         | Orientation   | User need                     |
|:---------------- |:------------- |:----------------------------- |
| **Tutorial**     | Learning      | "Help me get started"         |
| **How-to guide** | Task          | "Help me accomplish X"        |
| **Reference**    | Information   | "What are the facts about X?" |
| **Explanation**  | Understanding | "Why does X work this way?"   |

### Decision rules

 1. Reader learning with no prior context? → **Tutorial**
 2. Specific goal or problem? → **How-to**
 3. Lookup of API, schema, config? → **Reference**
 4. Rationale, trade-offs, design? → **Explanation**

### Separations

  - **Tutorial vs how-to:** Tutorial holds the hand end-to-end; how-to assumes competence and one task.
  - **Reference vs explanation:** Reference states *what*; explanation discusses *why*.

## Step 2: Review existing docs and code

 1. Read the package `docs/make.jl` `pages` structure ([SiennaTemplate example](../../SiennaTemplate.jl/docs/make.jl))
 2. Read related pages in the same section for tone and depth. Note that existing pages may not satisfactorily follow Diataxis principles yet. Proceed with caution.
 3. Read relevant `src/` for technical accuracy
 4. Link instead of duplicating content covered elsewhere

## Step 3: Write the page

### General principles (all types)

  - Second person ("you") for the reader; first person plural ("we") in tutorials
  - Active voice; short paragraphs (3–5 sentences)
  - Scannable headings; front-load key information
  - **Link, don't repeat**

### Tutorials (`docs/src/tutorials/`)

| Principle         | Guideline                                                                          |
|:----------------- |:---------------------------------------------------------------------------------- |
| Audience          | No prior knowledge of this topic                                                   |
| Voice             | "We will…"                                                                         |
| Structure         | Linear steps with visible results                                                  |
| Iterative Process | Run tutorial and explicitly reference key numeric outputs in reflection statements |
| Explanation       | Minimal—link to explanation pages                                                  |
| Reliability       | Every step must work; test end-to-end                                              |

**Format**: Tutorials are written as .jl scripts and processed to markdown and downloadable Jupyter notebooks with [Literate.jl](https://fredrikekre.github.io/Literate.jl/v2/)

**Naming:**

 1. Outcome-based — *Getting started with PackageX*
 2. Do not use `@id` tags on tutorial titles which causes poor Jupyter notebook formatting

**Template:**

 1. One-sentence outcome
 2. Prerequisites (Julia version, deps, data)
 3. Numbered steps with primary functions hyperlinked in comment preceeding code blocks, key output blocks followed by "Notice", "See", "Double-check", etc. reflection statements: [example](../../PowerSystems.jl/docs/src/tutorials/manipulating_datasets.jl)
 4. Wrap-up and next steps

### How-to guides (`docs/src/how_to_guides/`)

| Principle   | Guideline                                     |
|:----------- |:--------------------------------------------- |
| Audience    | Basic competence with the package             |
| Voice       | Imperative — "To do X, run…"                  |
| Structure   | Steps for one task; branches for alternatives |
| Explanation | None inline—link out                          |

**Naming:** *How to …*

### Reference (`docs/src/reference/`)

| Principle    | Guideline                                   |
|:------------ |:------------------------------------------- |
| Audience     | Developers using the API                    |
| Voice        | Neutral, precise                            |
| Structure    | Mirrors code; `@docs` blocks for docstrings |
| Completeness | Public parameters, methods, types           |

**API pages:** use Documenter:

Preferred for completeness:

````markdown
# Public API

```@autodocs
Modules = [SomeSiennaPackage]
Pages = ["mypage.jl"]
Public = true
Private = false
```
````

For custom organization:

````markdown
```@docs
MyFunction
MyType
```
````

**Naming:** Matches the thing described — *Public API*, *Developer guidelines*

### Generated model reference (PowerSystems.jl Model Library)

PowerSystems has **three separate "generated" documentation paths**. Do not conflate them:

| Path                                    | Source of truth                                         | Where to edit                                                |
|:--------------------------------------- |:------------------------------------------------------- |:------------------------------------------------------------ |
| `src/models/generated/*.jl`             | `src/descriptors/power_system_structs.json`             | JSON (`docstring`, field `comment`), then regenerate structs |
| `docs/src/model_library/generated_*.md` | Same JSON (thin `@autodocs` stubs over the `.jl` files) | JSON + regen structs; stubs recreated by `docs/make.jl`      |
| `docs/src/tutorials/generated_*.md`     | Source `.jl` in `docs/src/tutorials/` (Literate)        | Edit the tutorial `.jl`, not JSON                            |

**Never hand-edit** `src/models/generated/` or `docs/src/model_library/generated_*.md` for docstring or field documentation. Reading them is fine to verify rendered API text.

Regenerate structs from the PowerSystems.jl repo root:

```bash
julia --project=test -e 'using InfrastructureSystems; InfrastructureSystems.generate_structs("./src/descriptors/power_system_structs.json", "./src/models/generated")'
```

Hand-written Model Library pages (e.g. `dynamic_inverter.md`, `reserves.md`) and other Diátaxis pages follow the normal workflow in this skill. For JSON editing details see `julia-style-rules` (Generated code) and `.cursor/rules/generated-model-docs.mdc`.

### Explanation (`docs/src/explanation/`)

| Principle | Guideline                                |
|:--------- |:---------------------------------------- |
| Voice     | Discursive — rationale and trade-offs    |
| Structure | Topic-centered, not step-centered        |
| Content   | No procedural steps or full API listings |

**Naming:** Topic titles — *Design of the simulation loop*

## Step 4: Documenter.jl formatting

Sienna packages use **Documenter.jl**, not Sphinx/MyST.

### Page setup

```markdown
# Page Title

```@meta
CurrentModule = MyPackage
```

Introductory paragraph.

```

- One `#` title per page
- Headings `#` through `####`; do not skip levels; sentence case

### Admonitions

```markdown
!!! note "Optional title"
    Important information.

!!! warning
    Critical constraint.

!!! tip
    Helpful shortcut.
```

### Code blocks

Use example blocks to ensure code is checked errors for during docs compilation:

````markdown
```example
using MyPackage
result = foo(bar)
```
````

Use julia with language identifier only when needed for pseudo code:

````markdown
```julia
using MyPackage
result = somePseudoCodeHere()
```
````

### Cross-references

  - Same package: ```[`MyType`](@ref)``` or `[Other page](@ref)`
  - Other packages: `DocumenterInterLinks` + `@extref` (configure in `docs/make.jl`)
  - All mentions of an exported Sienna type or function should be hyperlinked

## External cross-reference format

  - Do, so user can see correct package: ```[`PowerSimulations.SimulationResults`](@extref)```
  - Don't: ```[`SimulationResults`](@extref PowerSimulations.SimulationResults)```

### Mermaid (when supported)

Use [DocumenterMermaid.jl](https://juliadocs.org/DocumenterMermaid.jl/stable/) for diagrams

### Tables

Standard pipe tables for parameters and options.

## Step 5: Integrate into the docs build

 1. **Save** under the correct `docs/src/` subdirectory
 2. **Add to `docs/make.jl`** `pages` OrderedDict (see [SiennaTemplate.jl/docs/make.jl](../../SiennaTemplate.jl/docs/make.jl))
 3. **Build locally:**

```bash
julia --project=docs docs/make.jl
```

 4. **Deploy:** `deploydocs` in `make.jl` handles GitHub Pages for each repo

### Standard layout (Sienna packages)

```
docs/
  make.jl
  make_tutorials.jl   # optional
  Project.toml
  src/
    index.md
    tutorials/
    how_to_guides/
    explanation/
    reference/ # api/ legacy in some packages
```

## Docstrings

  - Follow [InfrastructureSystems style](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/style/) and [docstring how-to](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/docs_best_practices/how-to/write_docstrings_org_api/)
  - Use `DocStringExtensions` for consistent signatures
  - Expose public API on reference pages with `@docs`; keep internal docs on `reference/internal.md` if used

## Validation checklist

### Content

  - [ ] Exactly one Diátaxis category
  - [ ] Verified against `src/` and tests
  - [ ] No duplication—linked instead

### Structure

  - [ ] Single `#` title
  - [ ] Heading hierarchy valid
  - [ ] Entry added to `docs/make.jl` `pages`

### Format

  - [ ] example code blocks where applicable
  - [ ] `@meta` / `@docs` / admonitions valid for Documenter
  - [ ] Cross-refs resolve after `makedocs`

### Category-specific

  - **Tutorial:** numbered steps, tested end-to-end
  - **How-to:** title "How to …", single task
  - **Reference:** complete public API coverage
  - **Explanation:** why-focused, no step-by-step instructions

## Project documentation stack

| Tool                                                                      | Role                                    |
|:------------------------------------------------------------------------- |:--------------------------------------- |
| [Documenter.jl](https://documenter.juliadocs.org/)                        | Build system (`makedocs`, `deploydocs`) |
| [DocumenterInterLinks.jl](https://juliadocs.org/DocumenterInterLinks.jl/) | Cross-package `@extref`                 |
| [DocStringExtensions.jl](https://docstringextensions.juliadocs.org/)      | Docstring formatting                    |

## References

  - Diátaxis: [https://diataxis.fr/](https://diataxis.fr/)
  - Documenter: [https://documenter.juliadocs.org/](https://documenter.juliadocs.org/)
  - InfrastructureSystems docs best practices: [https://sienna-platform.github.io/InfrastructureSystems.jl/stable/docs_best_practices/explanation/](https://sienna-platform.github.io/InfrastructureSystems.jl/stable/docs_best_practices/explanation/)
  - Local cheatsheet: `references/documenter_cheatsheet.md`

## Provenance

Diátaxis patterns adapted from [mcollina/skills documentation](https://github.com/mcollina/skills) and internal documentation-drafting guidance; Sienna specifics from workspace conventions and SiennaTemplate.jl.
