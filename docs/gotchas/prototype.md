---
status: as-built
covers: the traps in the 2025-08-16 tree — tests, imports, environment, and documentation that outruns the code
---

# Gotchas — the prototype tree

Short form. Each entry ends with what to do instead; the contract is in
[README.md](README.md). None of these is repaired in the tree: rule
`abl-mesh-project` § 4.

### Running `uv run pytest -q` and expecting green

Four of six tests fail on the committed tree, for four different reasons
listed below. The two that pass are the beta* formula and the memmap
precompute plus GeoTIFF round trip.

**Instead:** read the failures as findings. A targeted fix is Fabien's call.

### `ZoneTensorMesher(..., gmsh_init=False)` without the gmsh wheel

The constructor raises `RuntimeError: gmsh python bindings required` even
with `gmsh_init=False`; the import guard fires before the flag is read. Both
`test_anisotropic_sampler.py` tests die there under a bare `uv sync`.

**Instead:** `uv sync --extra gmsh`. On a headless container the wheel then
needs `apt-get install libglu1-mesa libxft2` before it imports — and the
constructor still refuses, see the next entry.

### `gmsh.model.mesh.setBackgroundMesh(path)` on gmsh 4.14

The call every mesher here is built around does not exist. gmsh 4.14.0's
Python API has one background-mesh entry point,
`gmsh.model.mesh.field.setAsBackgroundMesh(tag)`, which takes a scalar
size *field*, not a `.msh` with tensor point data. `ZoneTensorMesher`,
`unused/gmsh_background_metric_mesher.py` and `unused/gmsh_zone_mesher.py`
all `hasattr`-check for it and raise, so the scalar fallback further down
is unreachable and the route never ran end to end.

**Instead:** the tensor metric goes to MMG2D as a Medit `.sol`, which is
what `ablmesh` does (`preprocessing/mmg_utils.py`). In gmsh, a scalar size
field via a `PostView` field is the only sanctioned route.

### `ho.load_coeffs_memmap(...)`

Documented in the README, `examples/precompute_and_load.md` and both halves
of `PROJECT_OVERVIEW.md`, tested by `test_export_load_memmap_geotiff.py`,
and never written: `RasterHighOrderApproximant` has no such attribute. The
overview's own roadmap says *"implement load_coeffs_memmap convenience API
(if not present)"*.

**Instead:** `precompute_all_coeffs(memmap_path=...)` on a fresh approximant
re-reads nothing; only `load_coeffs_geotiff` exists as a loader.

### Trusting `compute_complexity_on_grid` to a tenth of a percent

It sums `sqrt(det M)` over `nx * ny` `linspace` samples and multiplies by
`dx * dy` with `dx = L / (nx - 1)`, so the Riemann sum covers
`nx/(nx-1) * ny/(ny-1)` of the box. At 50 x 100 that is a 3.07 % overcount,
which is exactly the residual `test_constant_isotropic_metric_grid` reports
against its `1e-3` tolerance.

**Instead:** if the number matters, use `dx = L / nx` on cell centres, or
integrate on triangles with `integrate_complexity_from_components_tris`.
`ablmesh` computes the integral in C++ on the surface mesh.

### `import abl_mesh.surface_mesher`, `abl_mesh.unused.hybrid`, `abl_mesh.unused.gmsh_zone_mesher`

Moving half the modules into `unused/` split a relative-import graph:
`surface_mesher` wants `.metrics` (now under `unused/`), `unused/hybrid`
wants `.surface_mesher` and `unused/gmsh_zone_mesher` wants `.zone_size`
(both still at top level). Every example except `run_pipeline_raster.py`
imports the pre-move paths.

**Instead:** the working surface is `raster_topography`,
`metric_complexity`, `zone_size`, `zone_tensor_mesher`, `delaunay_backends`,
`visualize`, and `scripts/precompute_coeffs.py` on top of them.

### `examples/run_pipeline_raster.py` as the one script to run the pipeline

It uses `np.` from line 167 on without importing numpy (`ruff` F821 x 6), so
it dies the moment the metric sampler is built.

**Instead:** the programmatic sequence in `PROJECT_OVERVIEW.md` § *Minimal
sequence*, with `import numpy as np` added.

### `--raster tests/data/tiny_dem.tif`

The README and the overview pass that path to every CLI example. The file
never existed; the tests build a 6 x 5 synthetic raster in `tmp_path`.

**Instead:** `_create_test_raster` in `tests/test_raster_precompute_memmap.py`
writes one anywhere you point it.

### Reading `PROJECT_OVERVIEW.md` as one document

It is two session write-ups concatenated, both dated 2025-08-16. The first
says beta*, volume extrusion and hybrid optimization are not implemented;
the second says beta* is done and the volume pipeline is *"implemented at
algorithmic level"*. The tree matches the first for the volume side (only
sketches under `unused/`, which do not import) and the second for beta*
(`zone_tensor_mesher.py` wires `target_num_nodes`).

**Instead:** the conformity matrix in the first half, checked against the
imports in `docs/lab_book/archeology.md`.
