# abl-mesh

Topography‑aware meshing pipeline for Atmospheric Boundary Layer (ABL) simulations.

This repository implements a production-capable pipeline to build high‑quality surface and
hybrid (prismatic + tetrahedral) volume meshes adapted to complex terrain. The pipeline
implements a high‑order (HO) topography approximant, metric-driven surface adaptation,
memmap- and GeoTIFF-backed precompute of HO coefficients, adaptive and anisotropic background
sampling for metric meshes, metric complexity (β*) sizing to target node budgets, integration
with Gmsh background-mesh meshing (≥ 4.14), and hybrid mesh optimization for prismatic/tetrahedral
ABLs. Visualization helpers (PyVista) and CLI helpers are included.

This README provides a concise but complete developer and user quickstart. For an exhaustive
overview of the project design, algorithms and rationale see PROJECT_OVERVIEW.md.

## Features

- Raster-based HO approximant (polynomial LS) with:
  - on-demand fits via `query_at((x,y)) -> z, grad, hess`
  - parallel tiled memmap precompute for out-of-core use
  - multiband GeoTIFF export/import (one band per coefficient, monomial metadata)
- Metric construction: curvature (Hessian) and tangent (first fundamental form)
- Metric complexity integral and global scaling β* to target node budgets
- Background-mesh generation:
  - structured, gradient-driven adaptive, axis-aligned anisotropic, oblique anisotropic
  - Delaunay fallback and pluggable backends
- ZoneTensorMesher: writes metric background `.msh` and calls `gmsh.model.mesh.setBackgroundMesh`
- Hybrid meshing: prismatic sweep for SBL + tetrahedral fill, with optimization for quality
- PVVisualizer: PyVista helpers for mesh and refinement debug overlays
- CLI helpers:
  - `scripts/precompute_coeffs.py` : memmap precompute and GeoTIFF export
  - `examples/run_pipeline_raster.py` : end-to-end driver exposing adaptivity and precompute options
- Unit tests covering complexity, precompute, GeoTIFF roundtrip, anisotropic samplers

## Quick start — installation (developer)

1. Clone repo

   ```bash
   git clone <repo-url> abl-mesh
   cd abl-mesh
   ```

2. Create Python 3.11+ virtual environment and install core dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev]"        # installs core + dev extras
   ```

   To enable optional features:
   - Visualization: `pip install -e ".[viz]"`
   - Gmsh integration: `pip install -e ".[gmsh]"` (or install gmsh system package; Python bindings >=4.14 recommended)
   - Accelerators (numba/cython): `pip install -e ".[accel]"`

3. Sanity: run unit tests (requires rasterio & shapely for some tests)

   ```bash
   pytest -q
   ```

Quick start — minimal run (tiny DEM)

1. Generate or obtain a tiny DEM GeoTIFF (tests provide tiny generators). Example tests provide utilities.
2. Precompute coefficients in-memory or to memmap (recommended for large rasters):
   - memmap tiled precompute:

     ```bash
     python scripts/precompute_coeffs.py \
       --raster tests/data/tiny_dem.tif \
       --degree 3 \
       --do-precompute \
       --memmap-path /tmp/coeffs_p3.memmap \
       --tile-size 64,64 \
       --n-jobs 4
     ```

   - in-memory precompute + export GeoTIFF:

     ```bash
     python scripts/precompute_coeffs.py \
       --raster tests/data/tiny_dem.tif \
       --do-precompute \
       --export-coeffs /tmp/coeffs_p3.tif
     ```

3. End-to-end mesh run (use precomputed coefficients for speed):

   ```bash
   python examples/run_pipeline_raster.py \
     --raster tests/data/tiny_dem.tif \
     --load-coeffs /tmp/coeffs_p3.tif \
     --hmin 1.0 --hmax 50.0 \
     --bg-nx 200 --bg-ny 200 \
     --bg-adapt-anisotropic --bg-adapt-anisotropic-oblique \
     --anisotropy-threshold 2.0 --bg-adapt-max-levels 2 \
     --write-mesh final_surface.msh \
     --visualize
   ```

   Notes:
   - If using memmap: instead of `--load-coeffs` you can load memmap programmatically via `ho.load_coeffs_memmap(...)` (convenience helper).
   - To drive β* scaling, add `--target-num-nodes 50000` (CLI mapping present in the example script).

## Typical workflows (recommended)

- Precompute on HPC, export coefficients:
  1. Run `scripts/precompute_coeffs.py` on large machine with `--memmap-path` or `--do-precompute` and large `--tile-size`.
  2. Export to GeoTIFF (`--export-coeffs`) to produce a portable coefficients file.
  3. Copy GeoTIFF to desktop / meshing node and run `examples/run_pipeline_raster.py --load-coeffs`.

- Metric complexity (target nodes):
  - Provide `--target-num-nodes` to compute β* (mapped into ZoneTensorMesher.generate). Adjust `complexity_nx`/`complexity_ny` if you prefer a different integration grid.

- Refinement polygons (per-turbine local refinement):
  - Prepare a vector file (Shapefile or GeoJSON) with polygon features and a numeric attribute `hmin` (or `h_min`).
  - Pass `--refinement-shapefile path/to/polygons.shp` to `examples/run_pipeline_raster.py` or programmatically pass `refinement_polygons` to `ZoneTensorMesher.generate()`.

## Commands & CLI reference

- Precompute helper:

  ```text
  scripts/precompute_coeffs.py --raster <raster.tif> [--degree 3] [--do-precompute] [--memmap-path PATH] [--load-memmap PATH] [--export-coeffs PATH] [--tile-size R,C] [--n-jobs N] [--use-tqdm]
  ```

- Full pipeline driver:

  ```text
  examples/run_pipeline_raster.py --raster <raster.tif> --hmin <m> --hmax <m> [--do-precompute] [--memmap-path PATH] [--load-coeffs PATH] [--export-coeffs PATH] [--bg-nx N] [--bg-ny N]
    [--bg-adapt-gradient-threshold T] [--bg-adapt-max-levels L]
    [--bg-adapt-anisotropic] [--bg-adapt-anisotropic-oblique]
    [--anisotropy-threshold RATIO]
    [--target-num-nodes NODES]
    [--refinement-shapefile PATH]
    [--write-mesh final_surface.msh]
    [--visualize] [--visualize-debug]
  ```

## Primary classes & entrypoints (programmatic)

- RasterTopography(raster_path, band=1, verbosity=1)
  - bounds(), sample(xy), xy_to_colrow, colrow_to_xy
- RasterHighOrderApproximant(rtopo, degree=3, precompute=False, ...)
  - query_at(xy) -> (z, grad, hess)
  - precompute_all_coeffs(..., memmap_path=..., tile_size=(r,c), n_jobs=...)
  - export_coeffs_geotiff(path), load_coeffs_geotiff(path), load_coeffs_memmap(path,...)
- ZoneTensorMesher(ho, metric_sampler, bbox, verbosity=1, gmsh_init=True)
  - generate(... see function signature in docs/PROJECT_OVERVIEW.md)
  - visualize_background(), visualize_mesh(), finalize()

## Development & testing

- Run tests:

  ```bash
  pytest -q
  ```

- Packaging / wheels:
  - This repo uses `setuptools` as the build backend (pyproject.toml). If you later add Cython / native extensions, implement `setuptools.Extension` objects in `setup.cfg` or `setup.py` as required.

Performance & extension roadmap

- Short-term (already available): Numba-accelerated kernels for Vandermonde, polynomial eval and sqrt(det).
- Mid-term (recommended): Cython or pybind11/Eigen batched LS solver + efficient Vandermonde/eval to accelerate precompute_all_coeffs.
- Long-term: optional GPU kernels (CuPy/CUDA) for extremely large precompute workloads. Fallback pure-Python paths are kept for portability.

## Troubleshooting & notes

- Gmsh:
  - Ensure Gmsh Python bindings >= 4.14 for `setBackgroundMesh`. If not available the code falls back to inserting scalar sizes as points in the GEO model.
- RasterIO:
  - `rasterio` is required for reading GeoTIFF rasters and for GeoTIFF coefficient export/import.
- Shapely:
  - Oblique anisotropic sampling uses `shapely.ops.split`. If shapely is not installed the code falls back to axis-aligned anisotropic splitting.
- Memory & disk:
  - Memmap precompute writes a (H × W × num_coeffs) memmap; ensure disk space. Use float32 memmap dtype to reduce disk usage.
- Platform notes:
  - On Apple M1/M2 (aarch64) numba builds may be unavailable or slower; prefer Cython/pybind11 compiled packages or use float32 memmap and smaller tiles.

## Contributing

- Use feature branches and open PRs with:
  - tests for new functionality
  - docs updates (PROJECT_OVERVIEW.md, README.md)
  - keep changes small and focused (feature + tests + docs)
- CI recommendation:
  - Add GitHub Actions job that installs rasterio, shapely, scipy, pytest and runs unit tests; skip gmsh-dependent integration tests in CI or run them with a separate integration job.

## License & credits

- MIT License. See LICENSE file.
- Primary authors & affiliations: see main_arxiv.tex frontmatter (A. Gargallo‑Peiró, M. Avila, A. Folch — Barcelona Supercomputing Center).
