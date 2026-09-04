---
status: as-built
covers: the docs/ tree of abl-mesh — the one-screen map, the lanes, and the status legend
---

# abl-mesh — documentation map

This repo is the 2025-08 gmsh-based prototype of the terrain-meshing
pipeline that `ablmesh` went on to build. The docs here answer one question
above all others: *what of this is still worth anything, and where did the
rest go?* [`AGENTS.md`](../AGENTS.md) carries the layout and the commands.

## The map, in one screen

| Question | Go to |
| --- | --- |
| What is this repo, and should it be archived? | [`decisions/0001-superseded-by-ablmesh.md`](decisions/0001-superseded-by-ablmesh.md) — the proposal, and what an archive would lose |
| What did the prototype implement against the paper, and what did `ablmesh` keep? | [`lab_book/archeology.md`](lab_book/archeology.md) — dated, with the `ablmesh` file paths that were verified |
| Why do the tests fail, why do three modules not import? | [`gotchas/prototype.md`](gotchas/prototype.md) |
| What did the 2025-08 session think it had built? | [`../PROJECT_OVERVIEW.md`](../PROJECT_OVERVIEW.md) — two write-ups concatenated; the conformity matrix in the first is the useful part |
| How was the pipeline meant to be run? | [`../README.md`](../README.md), [`../examples/precompute_and_load.md`](../examples/precompute_and_load.md), [`../examples/run_raster_ho.md`](../examples/run_raster_ho.md) — 2025-08 text, unchanged |
| What brief generated the code? | [`../.github/instructions/Paper.instructions.md`](../.github/instructions/Paper.instructions.md) |
| The paper itself | `../literature/garallo/main_arxiv.tex`, figures under `figures/` |

## Lanes

| Lane | Holds |
| --- | --- |
| [`decisions/`](decisions/) | ADRs: Context / Decision / Consequences / what would retire it. `0001` is the only one and it is `proposed`. |
| [`gotchas/`](gotchas/README.md) | Dead ends in short form, each ending with **Instead:**. One file, `prototype.md`, because there is one domain here. |
| [`lab_book/`](lab_book/README.md) | Dated, append-only. `archeology.md` is the read of this repo against `ablmesh`. Nothing links into it. |

There is no `validation/` lane. Nothing in this repo is claimed at any
grade against any evidence; the one measurement worth keeping (the
complexity-integral overcount) is a gotcha, not a validated number.

## Status legend

Every file under `docs/` carries a YAML header; `status:` is greppable.

| Value | Meaning |
| --- | --- |
| `as-built` | Describes what the repo does **today**. Names used here exist. |
| `proposed` | A decision Fabien has not ruled on. Nothing follows from it yet. |
| `evidence` | A dated record. Its facts are the day's; nothing in it is re-derived later. |
