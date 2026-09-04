---
status: evidence
covers: the dated read of the 2025-08-16 tree against the paper and against ablmesh, and the test and lint outcomes of that day
---

# Lab book — archeology

Append-only. Nothing links into an entry; see [README.md](README.md).

## 2026-09-04 — what the prototype is, what it implemented, what `ablmesh` kept

Read on Claude Code on the web, checkouts `/home/user/abl-mesh` (branch
`claude/elegant-dirac-skbbvj`, tree at `829f0fd`) and `/home/user/ablmesh`
(`5c07fab`, 2026-03-30). Every `ablmesh` path below was opened or grepped
that day; nothing is quoted from memory.

### Identity

- Two commits, both 2025-08-16 by Fabien: `7b277b9` *Initial commit for
  abl-mesh project*, `829f0fd` *Ruff*. Remote `ews-ffarella/abl-mesh`.
- Generated under `.github/instructions/Paper.instructions.md`, a Copilot
  instruction file: implement `literature/garallo/main_arxiv.tex` *"as
  precisely as possible"*, class-based with a `verbosity` int, gmsh +
  scipy + pyvista, report every deviation. The code style asked for there
  (Google docstrings, pathlib, absolute imports, `contextlib.suppress`) is
  what the tree shows.
- The paper: Gargallo-Peiró, Avila, Folch, *A hybrid meshing framework
  adapted to the topography to simulate Atmospheric Boundary Layer flows*
  (BSC). Sections: 3 topography modeling and metrics (3.1 HO derivative
  computation, 3.2 metrics), 4 adapted surface mesh, 5 hybrid ABL mesh,
  6 quality and optimization, 7 results (Bolund, Alaiz, Badaia).
- `PROJECT_OVERVIEW.md` is two write-ups concatenated (a second title at
  line 578, same date). The first carries a conformity matrix; the second
  overstates the volume side. Fabien 2026-09-04 `[FF-0409]`: *"the
  reference paper was Gargallo about CFD meshing in complex terrain."*

### Implemented versus roadmap, per the overview's conformity matrix, checked against the tree

| Paper component | Overview says | Tree, 2026-09-04 |
| --- | --- | --- |
| HO approximant (§3.1) | implemented, raster variant | `raster_topography.py`: pixel-stencil monomial LS, degree 3 default, numba kernels optional, tiled memmap precompute, multiband GeoTIFF export/load. `topography.py`: the paper's triangle-neighbourhood variant, used only by examples. Both import. |
| Tangent + curvature metric (§3.2) | partial | curvature sampler lives in `examples/run_pipeline_raster.py` (`metric_sampler_from_ho`, which lacks `import numpy`); tangent metric only in `unused/metrics.py`. |
| Complexity integral, beta* | first half: not automated; second half: done | `metric_complexity.py` implements C and beta*; `zone_tensor_mesher.generate(target_num_nodes=...)` wires it (lines 984-1014). The grid integrator overcounts by `nx/(nx-1)*ny/(ny-1)`. |
| 3-zone size policy | implemented | `zone_size.compute_zone_size_for_points`: inner polygon `hmin`, linear ramp over `transition_width`, `hmax` outside. Imports. |
| Background mesh + gmsh `setBackgroundMesh` (§4) | implemented | `zone_tensor_mesher.py` (1183 lines): structured, gradient-adaptive quadtree, axis-aligned and oblique anisotropic samplers, `.msh` v4 via meshio with `metric_m11/m12/m22`, gmsh >= 4.14. Imports without gmsh; constructing needs it. |
| Adaptive surface refinement loop (Algorithm 1) | not listed | `surface_mesher.py` sketch over `triangle`; does not import (`.metrics` moved). |
| Prism extrusion (§5) | not implemented | `unused/extruder.py` `SafeExtruder`: pseudo-normals, inversion check via tet split, per-node backoff. Imports; nothing calls it. |
| Tet fill + hybrid orchestration (§5) | not implemented | `unused/hybrid.py` `HybridMesher` with TetGen; does not import (`.surface_mesher`). |
| Hybrid optimization (§6) | not implemented | `unused/optimizer.py` `QualityOptimizer`: scaled-Jacobian quality, per-node L-BFGS-B, Gauss-Seidel sweep. Imports; nothing calls it. |
| Solver coupling (Alya) | out of scope | absent. |

Two earlier gmsh meshers sit under `unused/` as the route's history:
`gmsh_surface_mesher.py` (metric reduced to a scalar size per GEO point)
and `gmsh_background_metric_mesher.py` (Delaunay background + metric point
data, the first `setBackgroundMesh` attempt); `gmsh_zone_mesher.py` is the
isotropic 3-zone version that `ZoneTensorMesher` unified.

### What `ablmesh` kept — file paths verified

- **Polynomial fit for z, gradient, Hessian**: `src/cpp/abl_mesh/
  fitting_helpers.hpp` (Vandermonde builders, monomial evaluators, QR / SVD
  / LAPACKE solvers), `surface_mesh_metric_evaluator.cpp`, wrapped by
  `src/ablmesh/preprocessing/surface_mesh_metric_evaluator.py`. The fit is
  per *triangle* of the surface mesh, the paper's route and the one
  `topography.py` sketched, not the raster-cell stencil of
  `raster_topography.py`. The DEM is read bilinearly through
  `preprocessing/raster_provider.py` (xdem / geoutils).
- **Tangent and curvature metrics, complexity integral, beta***: the
  evaluator docstring cites the paper's Eq. 16 and Eqs. 18, 21-23 and
  names `compute_metric_complexity_integral()` in C++;
  `preprocessing/adaptive_terrain_mesher.py:180` passes
  `use_complexity_integral_for_beta_star`. Per-region beta* is new there.
- **Three-zone sizing**: `preprocessing/terrain_mesher_config.py`
  `CellSizingConfig` (`h_objects` / `h_transition` / `h_far_field`, linear
  or geometric blending, plus a turbine override and an `h_min` floor); the
  successor of `zone_size.py`'s `hmin` / ramp / `hmax`.
- **Pseudo-normal extrusion and the Gauss-Seidel optimizer** as ideas:
  pseudo-normals are in `src/cpp/abl_mesh/surface_mesh.cpp` and
  `_deprecated/surface_extruder.py`; node-by-node optimization is the C++
  `VariationalOptimizer` with the Gargallo functional (README § C++
  Modules). `ablmesh`'s live extrusion is pure vertical, not along
  pseudo-normals.
- **The paper**: `ablmesh/docs/reference/gargallo.tex` differs from
  `literature/garallo/main_arxiv.tex` only in trailing whitespace (`diff`:
  458 changed lines, every one a whitespace edit; first at line 125), plus
  a pandoc `gargallo.md`. `ablmesh` has no `literature/` directory, no
  figures beyond one png, no `.bib`. Its evaluator docstring cites
  `literature/garallo/main_arxiv.md`, a path that does not exist in that
  repo.

### What `ablmesh` replaced

- gmsh `setBackgroundMesh` as the anisotropic mesher: replaced by MMG2D
  (mmgpy) over a Medit `.sol` metric, with CGAL smoothing and quality fix.
  gmsh survives for quad recombination only
  (`preprocessing/quad_utils.py:81`). `grep -rn setBackgroundMesh
  ablmesh/src` returns nothing.
- Raster-stencil HO fit with memmap / GeoTIFF coefficient store: no
  successor (`grep -rni "memmap\|coeffs_geotiff" ablmesh/src` returns
  nothing).
- Adaptive background samplers (gradient quadtree, axis-aligned, oblique)
  and the pluggable Delaunay backends: gone; MMG handles the anisotropy.
- Prisms + TetGen fill: replaced by hex-dominant vertical extrusion with a
  graded top fill and warp-based hex splitting; no tetrahedra.

### Is anything here imported or copied by `ablmesh`?

No. `grep -rn "abl_mesh\|abl-mesh" ablmesh/src` hits one C++ comment
(`cpp/cgal_smooth/cgal_bindings.cpp:2`, *"abl-mesh-core"*, the intermediate
repo on the Linux host). `ablmesh`'s C++ directory is named
`src/cpp/abl_mesh/`; that is name lineage, not code.

### Tests, environment and lint, that day

- `uv sync` (`.python-version` 3.12): builds the editable package, 191
  packages resolved. gmsh is an extra; `uv sync --extra gmsh` installs the
  wheel, which then fails `import gmsh` with `libGLU.so.1: cannot open
  shared object file` on this container. `apt-get` was held by another
  worker's OpenFOAM install at the time.
- `uv run pytest -q`: **4 failed, 2 passed** in 2.5 s (six tests in four
  files). Causes, each verified: two `RuntimeError: gmsh python bindings
  required` from `ZoneTensorMesher.__init__` even with `gmsh_init=False`;
  one `AttributeError: 'RasterHighOrderApproximant' object has no attribute
  'load_coeffs_memmap'`; one `assert (0.2457 / 8.0) < 0.001` in
  `test_constant_isotropic_metric_grid`, reproduced exactly by the
  `nx/(nx-1)*ny/(ny-1) = 1.030718` Riemann-sum overcount. The pytest cache
  in the checkout already recorded the same four failures from an earlier
  run that day.
- `tests/data/tiny_dem.tif` does not exist and the tests do not need it;
  they write a 6 x 5 `z = x + 2y` raster into `tmp_path`.
- Imports: 13 of 16 modules import; `surface_mesher`,
  `unused/gmsh_zone_mesher`, `unused/hybrid` do not (relative imports
  split by the `unused/` move).
- `ruff check .`: 183 findings (89 E501, 14 RUF059, 13 F841, 11 PTH110,
  10 B007, 8 SIM105, 6 F821, ...), 3 auto-fixable. `ruff format --check
  .`: 28 files already formatted. Nothing reformatted.
- `README.md` says *"MIT License. See LICENSE file"*; there is no
  `LICENSE` in the tree.

Extracted the same day: the dead ends to `docs/gotchas/prototype.md`, the
supersession to `docs/decisions/0001-superseded-by-ablmesh.md`.
