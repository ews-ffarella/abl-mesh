---
status: proposed
covers: whether abl-mesh is retired as a repo, and what an archive would lose
---

# ADR 0001 — `abl-mesh` is superseded by `ablmesh` and kept as reference

**Date:** 2026-09-04
**Status:** proposed — Fabien has not ruled that this repo is retired

## Context

`abl-mesh` is two commits from 2025-08-16: a pure-Python prototype of
Gargallo-Peiró, Avila and Folch's topography-adapted meshing pipeline,
generated under a Copilot brief that asked for the paper *"as precisely as
possible"* with gmsh, pyvista and scipy. It builds a high-order local fit of
a DEM, a curvature metric with complexity scaling, a three-zone size field,
and a metric background mesh for gmsh's `setBackgroundMesh`. Volume
extrusion and hybrid optimization exist as sketches under `unused/`.

`ablmesh` (500 commits to 2026-03-30) is the same pipeline rebuilt: the
polynomial fit, the tangent and curvature metrics, the complexity integral
and beta*, the three-zone sizing and the Gargallo optimization functional
all continue there, in C++ behind pybind11, with MMG2D doing the
metric-driven remeshing that gmsh did here. What `ablmesh` did not keep is
the gmsh background-mesh route, the raster-stencil fit with its memmap and
GeoTIFF coefficient stores, the adaptive background samplers, and the
prism-plus-TetGen volume. The dated read with verified file paths is
`docs/lab_book/archeology.md`.

Nothing in `ablmesh/src` imports or copies this package. The only
`abl-mesh` string there is a C++ comment naming `abl-mesh-core`, an
intermediate repo on the Linux host, not this one.

## Decision (proposed)

Treat `abl-mesh` as **reference material for `ablmesh`**, frozen at its
2025-08-16 tree. No pipeline work lands here; documentation and agent config
do. Whether the repo is then archived on GitHub is Fabien's call, and the
rest of this record is what that call weighs.

## What an archive would lose

An archived repo stays readable, so nothing below is destroyed by archiving
alone; it is lost the day the archive is deleted or stops being cloned.

- **The paper's supporting files.** `ablmesh` carries the article text
  (`docs/reference/gargallo.tex`, identical to `literature/garallo/
  main_arxiv.tex` up to trailing whitespace) but not the 33 figures, the
  `references.bib`, the `.bbl`, or `definitions.txt`. This is the only
  checkout that has them.
- **The raster HO approximant** (`raster_topography.py`): a pixel-stencil
  polynomial fit with tiled memmap precompute and multiband GeoTIFF export
  of the coefficient stack. `ablmesh` fits per triangle and reads the DEM
  bilinearly; the out-of-core coefficient store has no successor.
- **The gmsh background-mesh route** (`zone_tensor_mesher.py`, and the two
  earlier meshers under `unused/`): writing a `.msh` with per-node 2x2
  metric components and calling `gmsh.model.mesh.setBackgroundMesh`. That
  call does not exist in gmsh 4.14.0 (its only background API is the
  scalar `field.setAsBackgroundMesh`), so the route never ran end to end;
  what stands is the sampler and metric-writing code in front of it.
  `ablmesh` uses gmsh only for quad recombination.
- **The Copilot brief** (`.github/instructions/Paper.instructions.md`) and
  the session write-up (`PROJECT_OVERVIEW.md`): the record of how the
  prototype was asked for and what it thought it had built.

## Consequences

- The tree is not repaired: the four failing tests, the three modules that
  do not import and the 183 `ruff` findings stay as recorded
  (`docs/gotchas/prototype.md`). A green suite would be work on a snapshot.
- Anyone reaching for a background-mesh gmsh recipe reads
  `zone_tensor_mesher.py` here rather than re-deriving it.
- Nothing here joins the estate ledger on its own; the CFD front's ledger
  status is Fabien's to state (`fabien-context/docs/repos/cfd-front.md`).

## What would retire this record

Fabien ruling either way: *archive it* (then this ADR becomes `accepted` and
the repo is archived by him, never by an agent), or *keep working here*
(then the ADR is `superseded` and rule `abl-mesh-project` § 1 is rewritten).
