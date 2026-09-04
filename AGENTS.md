# AGENTS.md — abl-mesh

The 2025-08 pure-Python, gmsh-based prototype of the terrain-meshing pipeline
behind `ablmesh`. Package `abl-mesh`, import root `src/abl_mesh/`, no console
script. **Reference material, not a build target**: the pipeline continued in
`ablmesh` (MMG2D + CGAL + Ceres, C++/pybind11), which is where any new work
goes. Read [`docs/README.md`](docs/README.md) for the map and
[`docs/decisions/0001-superseded-by-ablmesh.md`](docs/decisions/0001-superseded-by-ablmesh.md)
for what this repo is kept for.

## What this repo is

Two commits, both 2025-08-16, generated under the Copilot brief in
`.github/instructions/Paper.instructions.md`: *"I want to program a CFD
mesher as described in this article (file literature/garallo/main_arxiv.tex)"*.
The article is Gargallo-Peiró, Avila and Folch, *A hybrid meshing framework
adapted to the topography to simulate Atmospheric Boundary Layer flows* —
the reference for the whole CFD meshing chain. Fabien, 2026-09-04
`[FF-0409]`: *"the reference paper was Gargallo about CFD meshing in complex
terrain."*

What the prototype implemented, what it left on the roadmap, and which ideas
survived into `ablmesh` are read out, dated and with file paths, in
[`docs/lab_book/archeology.md`](docs/lab_book/archeology.md).

## Layout

```text
src/abl_mesh/
  raster_topography.py   RasterTopography + RasterHighOrderApproximant: local
                         polynomial LS fit on a DEM pixel stencil -> z, grad,
                         Hessian; tiled memmap precompute; GeoTIFF export/load
  metric_complexity.py   complexity integral C = int sqrt(det M), beta*
  zone_size.py           3-zone scalar size: inner polygon hmin / ramp / hmax
  zone_tensor_mesher.py  ZoneTensorMesher: samples a metric field on a
                         background grid (structured, gradient-adaptive,
                         axis-aligned or oblique anisotropic), writes a .msh
                         with per-node 2x2 metric, hands it to gmsh >= 4.14
                         setBackgroundMesh, lifts the 2D mesh with the HO fit
  delaunay_backends.py   scipy / triangle / meshpy / startinpy adapter
  visualize.py           PVVisualizer (pyvista)
  topography.py          the paper's triangle-based HO approximant (earlier
                         route; examples only)
  surface_mesher.py      Algorithm 1 refinement loop over `triangle`
                         (imports `.metrics`, which now lives in unused/, so
                         it does not import)
  unused/                extruder (prism sweep, pseudo-normals), hybrid
                         (prisms + TetGen fill), optimizer (Gauss-Seidel
                         distortion), metrics (tangent + curvature metric),
                         two earlier gmsh meshers, utils
examples/                drivers; run_pipeline_raster.py is the end-to-end one
scripts/precompute_coeffs.py   memmap precompute + GeoTIFF export CLI
tests/                   4 files, 6 tests, synthetic rasters in tmp_path
literature/garallo/      the paper: main_arxiv.tex, .bbl, references.bib,
                         definitions.txt, packages.txt, figures/ (33 png)
PROJECT_OVERVIEW.md      the 2025-08-16 session write-up, two versions
                         concatenated; the conformity matrix against the
                         paper is in the first
docs/                    the map, one proposed decision, gotchas, lab book
```

## Commands

```bash
uv sync                      # core deps; gmsh is an extra
uv sync --extra gmsh         # adds the gmsh wheel (needs libGLU.so.1 on the host)
uv run pytest -q             # 4 failed, 2 passed on the 2025-08-16 tree
ruff check .                 # 183 findings on that tree; not a gate
ruff format --check .        # clean
uv run python scripts/sync_agent_config.py          # regenerate .claude/
uv run python scripts/sync_agent_config.py --check  # exit 1 on drift
```

No `nox`, no `prek`, no `Makefile` here. `.vscode/tasks.json` names
`pre-commit`, but there is no `.pre-commit-config.yaml` in the tree. Why the
tests fail, and that `tests/data/tiny_dem.tif` named in the README and the
overview never existed (the tests build their own 6x5 raster):
[`docs/gotchas/prototype.md`](docs/gotchas/prototype.md).

## Read before editing

1. The always-on rule `abl-mesh-project` — the five boundaries, and where
   history goes.
2. When touching `.cursor/` or `.claude/`: rule `agent-config-sync`.
3. Before citing a capability of this code, run it: the overview's second
   half claims more than the tree imports
   ([`docs/lab_book/archeology.md`](docs/lab_book/archeology.md)).

## Hard boundaries

The five boundaries this repo enforces — no pipeline work here, keep
`literature/`, never delete `unused/` or the overview, the failing tests are
findings, `.claude/` is generated — are the always-on rule
`abl-mesh-project` (`.cursor/rules/abl-mesh-project.mdc`), not restated here.

## Non-goals

- Reaching paper parity. `ablmesh` does that, in C++.
- A green test suite or a clean `ruff check`. The tree is a dated snapshot;
  its defects are recorded, not repaired.
- Estate conventions on the code (attrs, `StrEnum`, classmethod
  constructors, `ews-logger`). The gap is real and is not being closed on a
  snapshot.

## The hub

Estate-wide context and decisions live in `fabien-context`
(`C:\Users\f.farella\AI` on the Windows host, `/home/user/fabien-context` on
Claude Code on the web). The CFD chain this repo belongs to is
`docs/repos/cfd-front.md` there; the C++ successors are
`users/fabien/cpp-extensions.md`. Route from its `context.md` before asking
where something lives.
