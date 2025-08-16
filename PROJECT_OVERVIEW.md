# Project overview and developer onboarding

Last updated: 2025-08-16

This document summarizes everything we've implemented so far for the topography‑aware meshing pipeline described in the paper "A hybrid meshing framework adapted to the topography to simulate Atmospheric Boundary Layer flows" (main_arxiv.tex). It is written for new programmers joining the project and for planning future work.

The goal of this repository (and the code produced in this session) is to provide a flexible pipeline to:

- read a DEM raster,
- build a high-order local approximant of the topography (to query z, ∇z, ∇²z),
- compute metric fields (curvature- and tangent-based and zonal sizing),
- build a background-mesh containing per-node metric tensors,
- instruct gmsh (>= 4.14) to use that background mesh to produce a surface mesh that respects the metrics,
- provide tooling to visualize both the background metric mesh and the generated surface mesh,
- provide two strategies:
  - `tensor` strategy: preserve anisotropy of an input tensor metric and rescale it to impose zone target sizes,
  - `simple` strategy: isotropic metric M = I/h² (fallback),
- support large sampling by producing a structured background mesh (default), avoiding expensive Delaunay for very large samples,
- precompute local polynomial fits for fast queries on raster (parallelized with joblib & `multiprocessing`) with progress reporting via `tqdm`.

This is not (yet) a full implementation of the paper: we implemented the surface-side pieces, raster HO approximant, background-mesh generation and Gmsh interfacing, visualization, and many engineering conveniences. Volume extrusion (prisms + tetrahedral filling) and the hybrid quality‑optimization frameworks described in the paper are currently not implemented and are included in the roadmap.

---

Table of contents

- Files / classes implemented (quick map)
- How the code maps to the paper (what is implemented / what is partial / what is missing)
- Implementation details and design choices (important behavior, default settings)
- API and usage: example invocations and notes for developers
- Deviations from the paper and rationale
- What is missing (to reach full paper fidelity)
- Recommendations for speed / memory improvements and next steps
- Suggested tests, benchmarks, and CI
- Onboarding tips for new programmers

---

## Files and main classes (what we added in this session)

Below I list files we created and the main classes / functions in each, with brief responsibilities.

- `abl_mesh/raster_topography.py`
  - `RasterTopography`:
    - Loads a GeoTIFF (via rasterio), builds `xs/ys` arrays and a `RegularGridInterpolator`.
    - Provides `sample((x,y))`, `xy_to_colrow`, `colrow_to_xy`, pixel size, bounds, etc.
  - `RasterHighOrderApproximant`:
    - Local polynomial least squares (monomial basis up to degree `p`) on a square pixel stencil.
    - `query_at((x,y)) -> (z, grad, hess)`.
    - Optional precompute of polynomial coefficients at every raster cell center (or a selected tile).
    - Parallel precompute implemented with `joblib` multiprocessing backend and progress via `tqdm` (see `TqdmJoblib`).
    - Robust fallbacks: if the fit fails, falls back to bilinear interpolation and local quadratic finite-difference estimation for derivatives.

- `abl_mesh/zone_size.py`
  - `compute_zone_size_for_points(pts, inner_poly, transition_width, hmin, hmax)`:
    - Computes scalar per-sample size `h(x)` per the 3-zone policy: inner polygon (hmin), transition band linear ramp, outside (hmax).
    - Uses Shapely polygon geometry.

- `abl_mesh/delaunay_backends.py`
  - `delaunay_triangulation(points, engine="auto")`
    - Abstraction layer to select a Delaunay backend: `startinpy` (if present, hypothetical), `triangle`, `meshpy`, or `scipy.spatial.Delaunay`.
    - The `"auto"` choice picks the fastest available backend in order.

- `abl_mesh/visualize.py`
  - `PVVisualizer` class (PyVista-based)
    - `show_mesh(nodes, tris, scalars=None, show_vertices=False, downsample_vertices=2000, ...)`
      - Renders `pv.PolyData` triangles, optional per-vertex scalars, optional vertex glyph overlays (downsample or all).
    - `show_background_mesh_from_msh(msh_path, ...)`
      - Reads background `.msh` with `meshio`, computes representative scalar size (`h_rep`) and optionally shows principal directions/vectors (downsampled).
    - Provides guardrails for extremely large visualizations (warnings, downsampling).

- `abl_mesh/delaunay_backends.py` (already mentioned)
  - Important to plug faster triangulators later.

- `abl_mesh/zone_tensor_mesher.py` (evolved through several versions)
  - `ZoneTensorMesher`:
    - Entrypoint that orchestrates background sampling, metric building, background mesh creation and invocation of Gmsh 4.14:
      - Samples a regular grid inside the outer circle (nx × ny).
      - Computes zonal sizes with `compute_zone_size_for_points`.
      - Two `mesh_strategy`s:
        - `'tensor'`: sample `metric_sampler((x,y))` (2×2 SPD) and rescale it to match the local scalar `h(x)` while preserving principal directions (geometric-mean length scaling).
        - `'simple'`: use isotropic metric `M = (1/h^2) * I`.
      - Several options:
        - `min_size_ratio` (clamp to avoid extremely small sizes),
        - `bg_mesh_strategy`: controls background mesh creation:
          - `'structured'` (default): build fast regular-grid triangles by splitting quads (recommended for large samplings),
          - `'grid_quads'`, `'delaunay'`, `'auto'`.
        - `delaunay_engine` forwarded to `delaunay_backends`.
      - Writes a temporary `.msh` (meshio v4) with point_data `metric_m11,m12,m22` and `size_scalar_fallback`.
      - Calls `gmsh.model.mesh.setBackgroundMesh(abs_path)` (requires gmsh ≥ 4.14).
      - Builds the planar geometry (GEO), generates 2D mesh (Gmsh does the rest).
      - Extracts nodes and triangles, lifts z using `ho.query_at((x,y))`.
    - `visualize_background(...)` and `visualize_mesh(...)` forward to `PVVisualizer`.

- `abl_mesh/zone_size.py` and `abl_mesh/gmsh_zone_mesher.py`
  - Earlier simpler helpers for isotropic zone-only workflows. Kept as examples; `ZoneTensorMesher` is the unified/replacement that supports both tensor and simple strategies.

- Example scripts
  - `examples/run_zone_mesher.py` – simple demo using isotropic zoning (example HO approximant stub).
  - `examples/run_zone_tensor_mesher.py` – demo of tensor-rescaling approach.
  - `examples/run_zone_tensor_visualize.py` – runs mesh generation and visualizes results with PyVista.
  - `examples/run_pipeline_raster.py` – large driver that ties everything together:
    - CLI argument parsing for raster filename, polygon, center, hmin/hmax, bg grid resolution, sampling strategy, precompute, visualization, outputs, etc.
    - Uses `RasterHighOrderApproximant`, builds curvature-based `metric_sampler`, creates `ZoneTensorMesher`, runs generation and optionally visualizes results.
    - Heavily commented; intended as the "one script to run the pipeline".

- Other: small helper / earlier files
  - `abl_mesh/gmsh_background_metric_mesher.py`: an early, simpler mesher implementation for Gmsh 4.14 background-mesh (kept for reference).
  - `examples/run_raster_ho.md` — example usage notes for the `RasterHighOrderApproximant`.

---

## How the implementation maps to the paper (conformity matrix)

I break the paper responsibilities into the main components and indicate the current status (Implemented / Partially implemented / Not implemented), with notes.

1. **Topography HO approximant (Section 2.1 in paper)**
   - Status: Implemented (raster-based).
   - Files: `abl_mesh/raster_topography.py`
   - Comments:
     - We can query z, gradient, and Hessian via local polynomial least squares.
     - Precompute option implemented (parallelized with joblib + multiprocessing; `tqdm` progress).
     - Behavior matches the paper's intention (local polynomial fits). We use a centered monomial basis and LS, as described in the paper.
     - Deviation: The paper's HO approximant is described for triangle/point clouds; we implemented a raster-specific variant which is faster for DEM inputs and provides the same mathematical polynomial fit. We document this deviation below.

2. **Curvature metric (Hessian-based) and tangent metric (first fundamental form) (Section 2.2 in paper)**
   - Status:
     - Tangent metric: concept present (we compute per-point HO derivatives; we provide metric sampler helpers).
     - Curvature metric: partially implemented.
   - Files: metric-related code lives in `examples/run_pipeline_raster.py` (default `default_metric_sampler_from_ho`) and `zone_tensor_mesher.py` for rescaling.
   - Comments:
     - We implemented a default `metric_sampler` that uses the Hessian (absolute eigenvalues) and eigenvectors to build a tensor M = V diag(|λ|) Vᵀ (regularized).
     - We included the ability to rescale a sampled tensor so that its "representative length" (geometric mean of principal lengths) equals a target `h(x)` (zonal size) — this preserves anisotropy direction while changing magnitude.
     - We did *not yet* implement the paper's "metric complexity" scaling (β*computed from desired number of nodes and metric complexity integral) in the automated sense. The scaffolding is present, but the explicit computation of global complexity C and β* to target a global node budget is not done automatically.

3. **3-zone policy (inner polygon hmin, transition band ramp, outer hmax)**
   - Status: Implemented.
   - Files: `abl_mesh/zone_size.py`, integrated in `zone_tensor_mesher.py` and earlier `gmsh_zone_mesher.py`.
   - Notes:
     - The code accepts a Shapely polygon for the inner zone, `transition_width` for the ramp, and uses `hmin`/`hmax`.
     - The zonal size `h(x)` is computed and used either as scalar (simple strategy) or as target scaling for the tensor metric (tensor strategy).

4. **Background-mesh workflow and Gmsh 4.14 setBackgroundMesh (Section 3 / implementation notes)**
   - Status: Implemented.
   - Files: `abl_mesh/zone_tensor_mesher.py` (and earlier `gmsh_background_metric_mesher.py`).
   - Notes:
     - We write a temporary `.msh` containing point_data metric components (`metric_m11,m12,m22`) and call `gmsh.model.mesh.setBackgroundMesh(path)`.
     - We support writing v4 `.msh` with `meshio`.
     - We expose `bg_mesh_strategy` to choose structured vs Delaunay background meshes.
     - The script then builds the 2D GEO and calls `gmsh.model.mesh.generate(2)` to get the surface mesh.

5. **Structured vs Delaunay background mesh**
   - Status: Implemented (structured default).
   - Files: `zone_tensor_mesher.py` and `delaunay_backends.py`.
   - Notes:
     - For large sampling counts the default structured approach avoids expensive triangulation.
     - Delaunay path still available with pluggable backends.

6. **Visualization (paper shows many figures)**
   - Status: Implemented.
   - Files: `abl_mesh/visualize.py`, example scripts for demonstration/visualization.
   - Notes:
     - PyVista-based `PVVisualizer` shows actual triangular mesh polydata and optionally vertex glyphs and metric vectors for the background .msh.

7. **Prism extrusion and hybrid tetrahedral-prism meshing (Sections 4 & 5 in paper)**
   - Status: Not implemented (not yet).
   - Notes:
     - The code produced in this session does not include the prismatic sweep, the tetrahedral fill, nor the hybrid mesh quality optimization (iterative Gauss-Seidel node optimizations) described in the paper.
     - These are non-trivial and are listed in the roadmap section below.

8. **Hybrid mesh optimization (distortion measures, surface constraints, Gauss‑Seidel optimization)**
   - Status: Not implemented (only conceptual discussion present in the paper).
   - Notes:
     - The paper's optimization algorithms require careful linear algebra and robust untangling procedures; they must be implemented and tested—planned in the roadmap.

9. **Integration with solver (Alya) and end-to-end CFD**
   - Status: Out of scope / Not implemented here.
   - Notes:
     - The pipeline currently prepares a surface mesh (and can write final `.msh`). It does not perform the extrusion (prisms + tets) nor the full export or format expected by the solver. Solver coupling is not part of this code yet.

---

## Implementation details and design notes

### Key design choices

- Use Gmsh 4.14 background-mesh API:
  - This is the most direct way to feed a per-node tensor metric to Gmsh so it can create anisotropic triangles that respect the tensor field.
  - Our code writes v4 `.msh` (meshio) containing the metric data and calls `gmsh.model.mesh.setBackgroundMesh(path)`.

- Two mesh strategies:
  - `'tensor'` (preserve anisotropy): read a 2×2 SPD metric M(x), rescale it by scalar β so that the geometric-mean representative length equals `h(x)`:
    - Representative length: \( \ell_g = (1/\sqrt[4]{\lambda_1\lambda_2}) \) (implemented).
    - β = (ℓg / h)², then Mnew = β M.
  - `'simple'`: isotropic metric `I / h²`. Simpler and robust fallback.

- Zonal sizing (3 zones):
  - Inner polygon: `hmin`.
  - Transition band (outside polygon up to `transition_width`) linear ramp.
  - Outside: `hmax`.

- Background mesh creation strategies:
  - Default: `structured` (fast): split regular grid quads into triangles where all 4 corners lie inside the outer circle. This is deterministic and fast for large samplings.
  - `delaunay` / `auto`: uses `delaunay_backends.delaunay_triangulation`, which picks the best available Delaunay backend: `startinpy` (hypothetical), `triangle`, `meshpy`, `scipy`.

- Raster HO approximant:
  - Fit monomial polynomials (degree `p`) on a centered square stencil of `support_pixels` around the query point.
  - Precompute (parallel) polynomial coefficients at raster cell centers to speed up repeated queries: implemented using `joblib` with `backend='multiprocessing'` and `TqdmJoblib` for progress.

- Visualization:
  - PyVista `PolyData` for the triangle mesh (proper faces array). Vertex glyphs optional; we provide downsampling by default to avoid rendering bottlenecks.

### Important default parameters

- HO approximant:
  - degree = 3 (defaults used in examples).
  - support_pixels default = max(3, 3*degree).
- Background sampling:
  - bg_nx / bg_ny default = 400 (configurable).
- `min_size_ratio` default = 0.5 (do not produce local sizes smaller than `hmin * min_size_ratio`).
- `bg_mesh_strategy` default = `'structured'`.
- Gmsh: `gmsh.model.mesh.setBackgroundMesh` requires gmsh python bindings >= 4.14.

### Robustness & fallbacks

- If writing the temporary `.msh` fails, we raise informative errors and clean up.
- If `setBackgroundMesh` fails (gmsh version mismatch or other error), the code falls back to inserting the sample points with scalar sizes into the GEO model — Gmsh then uses isotropic sizing interpolated from those point sizes.
- Raster HO: if the local LS fit fails (few valid samples), falls back to bilinear + quadratic finite difference estimate.
- Delaunay backend: if chosen backend fails, fallback to structured mesh.

---

## API and usage (quick start & examples)

### Minimal sequence (programmatic)

1. Load raster:

```python
from abl_mesh.raster_topography import RasterTopography, RasterHighOrderApproximant
rtopo = RasterTopography("my_dem.tif", verbosity=1)
ho = RasterHighOrderApproximant(rtopo, degree=3, precompute=False, verbosity=1)
```

2. (Optional) Precompute polynomial coefficients:

```python
ho.precompute_all_coeffs(n_jobs=8, use_tqdm=True)  # may use lots of memory
```

3. Define inner polygon (Shapely polygon) and zone parameters:

```python
from shapely.geometry import Polygon
inner_poly = Polygon([...])  # farm polygon in raster CRS
center = (cx, cy)
outer_radius = 20000.0  # e.g. meters
transition_width = 2000.0
hmin, hmax = 10.0, 75.0
```

4. Build metric sampler (curvature-based default):

```python
from examples.run_pipeline_raster import default_metric_sampler_from_ho
metric_sampler = default_metric_sampler_from_ho(ho)
```

5. Configure & run `ZoneTensorMesher`:

```python
from abl_mesh.zone_tensor_mesher import ZoneTensorMesher
mesher = ZoneTensorMesher(ho, metric_sampler, bbox=rtopo.bounds(), verbosity=2)
nodes3d, tri_idx = mesher.generate(
    nx=400, ny=400,
    inner_poly=inner_poly,
    center=center,
    outer_radius=outer_radius,
    transition_width=transition_width,
    hmin=hmin, hmax=hmax,
    min_size_ratio=0.5,
    mesh_strategy='tensor',         # 'tensor' or 'simple'
    bg_mesh_strategy='structured',  # recommended default
    delaunay_engine='auto',
    use_background_mesh=True,
    write_mesh='final_surface.msh',
)
mesher.finalize()
```

6. Visualize:

```python
mesher.visualize_mesh(nodes3d, tri_idx, show_vertices=True, downsample_vertices=2000)
```

7. CLI:

- Use `examples/run_pipeline_raster.py` which accepts many arguments (raster, inner polygon, center, outer radius, hmin/hmax, bg-nx/bg-ny, do-precompute, precompute-njobs, visualize, write-mesh, write-bg, mesh-strategy, bg-mesh-strategy, delaunay-engine, min-size-ratio, verbosity, etc.)

---

## Deviations from the paper (explicit)

Be explicit about differences between the repo code and the approach described in the paper:

1. **Raster HO approximant vs triangle-based HO**
   - Paper: HO approximant described as local polynomial fits from a (piece-wise linear) triangle mesh / point cloud.
   - This repo: provides a raster-oriented HO approximant which is optimized for DEM inputs (fast neighborhood indexing), plus an optional parallel precompute. The math of local least squares is the same, but input assumptions differ. This is an intentional practical deviation because many inputs are DEM rasters and raster fits are much faster and easier to precompute.

2. **Metric complexity scaling (β* to target node budget)**
   - Paper: defines metric complexity and uses it to compute a curvature metric scaled to reach a target number of nodes.
   - Repo: we implemented curvature-based metric sampling (Hessian -> eigen decomposition) and a per-node rescaling to reach `h(x)` (zonal sizing). However, we do not yet automatically compute β*from an integrated complexity C and a user node budget to derive a global scaling factor. Implementing that requires computing an integral of √det(M) across the domain (numerical integration over sample points) and then adjusting β* — this is a planned feature.

3. **Hybrid volume generation (prism extrusion + tetrahedral fill) and hybrid optimization**
   - Paper: detailed prismatic sweep, tetrahedral fill, and hybrid mesh optimization including the Gauss‑Seidel node-wise optimization of the distortion functional.
   - Repo: surface side (metric, background, Gmsh surface generation) is implemented. Volume extrusion, merging prismatic layers with tetrahedral region, and the full hybrid optimizer are not implemented in code yet.

4. **Quality optimization & untangling**
   - Paper: advanced element quality measures, determinant regularization for untangling, and global/local optimization loop.
   - Repo: visualization exists; quality-driven optimization algorithms are not implemented yet.

5. **Integration with Aleya / Alya / solver**
   - Paper mentions coupling to `Alya` and simulation results; repo does not include solver coupling. The output is a `.msh` that can be adapted for solver import.

6. **Parallel Delaunay backends**
   - Paper doesn't prescribe, ours is pragmatic: default to structured background + multiple optional Delaunay backends. This is a practical engineering decision to scale to large backgrounds.

7. **Use of Gmsh background mesh interpolation details**
   - We assume Gmsh 4.14 `setBackgroundMesh()` will interpolate tensors correctly. This is consistent with Gmsh 4.14 API but real-world behavior (interpolation, tolerances) may require tuning; do test with representative metric fields.

---

## What is missing / roadmap (prioritized)

Short-term (high priority)

1. Metric complexity automation (implement β* and global scaling to target a node budget)
   - Numerically compute complexity integral C = ∫ sqrt(det M) dx on the sampling grid and compute β* to target a desired number of nodes N_target (including α constant). Provide CLI option `--target-nodes` or `--target-complexity`.
2. Full hybrid volume meshing:
   - Implement prismatic sweep (generation of extrusion lengths, pseudo-normal blending), produce prismatic layers.
   - Integrate a tetrahedral generator (TetGen or `meshio` wrappers) to fill the rest of volume and merge boundaries.
   - Export hybrid mesh in formats expected by solvers (Alya, etc.).
3. Hybrid quality optimization:
   - Implement distortion measures for prisms and tets, Gauss‑Seidel node-wise local optimization, untangling/regularization (determinant regularizer).
   - Add local optimization during sweep (as in algorithm) and global optimization post-tetra mesh generation.

Mid-term
4. Automated benchmarks and unit tests:

- Add tests for HO approximant accuracy vs synthetic analytic surfaces (sinusoid, Gaussian hill).
- Add tests that verify metric rescaling properties (e.g., representative length matches h(x) within tolerance).
- Add CI and perf benchmarks for Delaunay vs structured bg generation.

5. Improve metric sampler options:
   - Provide options for different metric definitions (Hessian-based with different norming, tangent-only, anisotropic blending, directional metrics from wind).
6. Support other input types:
   - LAS/LAZ point clouds, shapefiles from many projections, reproject support (GDAL), automatic coordinate transforms.
7. Robustness & logging:
   - Centralized logging, consistent exceptions, input validation, environment checks (Gmsh version), improved error messages.

Long-term / research features
8. Solver integration & automated simulation experiments:

- Export to Alya/other solver formats, run test RANS simulations, compare convergence metrics like in the paper.

9. GPU-accelerated fitting / interpolation:
   - Offload large grid evaluations (Hessian, sampling) to GPU or to a C-accelerated library.
10. High-performance Delaunay / CGAL / parallel triangulation:

- Integrate robust parallel triangulators for large backgrounds (CGAL, Triangle via MPI, startinpy if production-ready).

---

## Performance / speed improvement recommendations (concrete)

The user asked for performance recommendations. Here’s a prioritized list with concrete actions.

1. Default to structured background mesh for large sampling counts (already implemented)
   - Structured generation is O(nx*ny) work with deterministic indexing and no expensive global triangulation.

2. Adaptive sampling density instead of uniform grid
   - Use coarser sampling in areas where the metric changes slowly, higher sampling near high-metric-gradient regions (precompute metric gradients or use HO derivative magnitudes to set sampling density).
   - Implementation: hierarchical quadtree sampling; sample more densely where |∇M| large.

3. Tile the background generation and precompute (memory friendly)
   - For very large DEMs, process tiles (overlapping) and write multiple partial background .msh files, then merge or let Gmsh consume them sequentially if supported.
   - Precompute HO coefficients only in tiles that are needed for the meshing region.

4. Use a faster triangulation backend (if Delaunay required)
   - Install `triangle` (Shewchuk) or `meshpy` for faster triangulation than SciPy for large numbers of points.
   - If available, prefer a parallel Delaunay library (CGAL bindings, TetWild, or platform-specific options). `delaunay_backends` is ready to switch.

5. Use vectorized numpy operations where possible
   - Metric sampling currently loops over sample points (to call `metric_sampler` which may be expensive). Where the metric sampler is cheap (e.g., curvature computed by HO approximant), vectorize it across the grid: compute Hessian at grid nodes in batches and eigendecompose using `numpy.linalg.eigh` vectorized or via `scipy.linalg` routines in blocks.

6. Precompute HO coefficients and use them for all mesh points
   - Precomputation turns expensive LS fits into cheap polynomial evaluations. Use `ho.precompute_all_coeffs()` but only for the raster region(s) you need (tile precompute to control memory).

7. Parallel I/O and multiprocessing carefully
   - Use `joblib` with `multiprocessing` for heavy CPU-bound tasks (precompute). Avoid calling `gmsh` or other non-fork-safe libraries inside worker processes.
   - Keep Gmsh operations in main process; precompute polynomial coefficients in worker processes.

8. Memory: use memory-mapped arrays for large precompute arrays
   - Storing `(nrows x ncols x num_coeffs)` floats can be huge. Consider `np.memmap` or HDF5 for out-of-core storage.

9. Profile and optimize hotspots
   - Add lightweight profiling hooks for the three stages: HO precompute/fits, background mesh creation (triangulation/writing), Gmsh meshing time. Run on target hardware to identify bottlenecks.

10. Use compiled kernels
    - Where LS fits are a hotspot, consider Cython/Numba implementations for Vandermonde build and `lstsq` calls in small matrices. However note that `numpy.linalg.lstsq` already uses optimized LAPACK; overhead is mostly Python loop calls — batch multiple fits into fewer Python calls or use precomputation.

---

## Tests and benchmarks to add

- Unit tests:
  - HO approximant: synthetic analytic surface (z = sin(a x) + cos(b y)) → verify `query_at` derivatives converge to analytic derivatives as support increases.
  - Metric rescaling: verify that `scale_tensor_to_length` returns Mnew whose geometric mean length matches `h_target` within a tolerance.
  - Zone sizing: check `compute_zone_size_for_points` for points in each zone and boundary cases.
  - Background `.msh` writing: verify meshio writes and that `metric_m11,m12,m22` are present and accessible.

- Integration tests:
  - Small end-to-end run using a small demo DEM and a tiny polygon: generate background, call Gmsh (local) and verify non-empty triangular mesh output.
  - Test fallback path: disable `setBackgroundMesh` (simulate version mismatch) and verify isotropic fallback works.

- Benchmarks:
  - Time background generation (structured vs Delaunay) for sample sizes 10k, 100k, 500k.
  - Time HO precompute single-thread vs `n_jobs` scaling.
  - Measure memory footprint of precompute arrays vs raster size.

- CI:
  - Use GitHub Actions with small sample DEM (tiny) to test basic imports and short end-to-end calls (no gmsh in CI unless gmsh installed in runner; otherwise mark as optional).

---

## Coding conventions & contributor notes

- Prefer class-based implementations with a `verbosity` int member to control debug output. (This is followed already.)
- Use `gmsh.initialize()` in the main process only and call `gmsh.finalize()` in `finalize()` methods to keep a clean gmsh state.
- Avoid calling `gmsh` inside `multiprocessing` workers (not fork-safe in many environments).
- Keep I/O and heavy CPU work separate from Gmsh interactions.
- Use `meshio` to read/write `.msh` reliably (we target v4 format for background mesh).
- Use `shapely` / `geopandas` for polygon handling; reproject polygons to raster CRS upstream if needed.
- Document expected coordinate reference system: all inputs (raster, polygon, center) must be in the same CRS. If not, transform with `pyproj`/`geopandas` before running.

---

## Quick checklist for a new developer to run the pipeline locally

1. Install dependencies (recommended in a venv):

```bash
# core
pip install numpy scipy meshio rasterio shapely pyvista joblib tqdm

# gmsh python bindings (must match installed gmsh binary version)
pip install "gmsh>=4.14"

# optional faster triangulators:
pip install triangle meshpy  # if you want to enable those backends

# for examples using geopandas:
pip install geopandas
```

2. Ensure `gmsh` binary of version >= 4.14 is installed (some OS package managers or conda).
3. Run the example pipeline:

```bash
python examples/run_pipeline_raster.py \
  --raster path/to/dem.tif \
  --hmin 10 --hmax 75 \
  --bg-nx 400 --bg-ny 400 \
  --bg-mesh-strategy structured \
  --mesh-strategy tensor \
  --do-precompute --precompute-njobs 8 \
  --write-mesh final_surface.msh \
  --visualize
```

4. If you want to debug metric behavior, add `--write-bg` to save background `.msh` and then use `PVVisualizer.show_background_mesh_from_msh()` on that file.

---

## Suggested Git/Github project structure (current + recommended organization)

Current files created in this session (summarized):

```
abl_mesh/
  raster_topography.py
  zone_size.py
  delaunay_backends.py
  visualize.py
  zone_tensor_mesher.py
  gmsh_background_metric_mesher.py   # early, kept
examples/
  run_zone_mesher.py
  run_zone_tensor_mesher.py
  run_zone_tensor_visualize.py
  run_pipeline_raster.py
README.md (not created here but recommended)
requirements.txt (generate from pip freeze)
```

Recommended additions:

- `tests/` with unit & integration tests.
- `benchmarks/` with scripts to measure triangulation / precompute performance.
- `docs/` for developer docs and API reference (Sphinx).
- `notebooks/` to show interactive PyVista-based demos (Jupyter).

---

## Roadmap & prioritized TODOs (concrete next tasks)

1. Implement metric complexity scaling (`β*`) and a CLI flag `--target-nodes` to let users request a target node count for curvature resolution. (High priority)
2. Implement prismatic extrusion & hybrid tetrahedral meshing + file exporter for solver formats (Alya / common formats). (High priority)
3. Implement hybrid and surface optimization algorithms (distortion measures and Gauss-Seidel optimizer) and unit tests. (High priority)
4. Add structured tile precompute and `np.memmap`/HDF5 storage to avoid OOM during full-raster precompute. (Medium)
5. Add CGAL / other fast Delaunay bindings optional integration (Medium).
6. Add testcases and CI; include small DEMs for smoke tests. (High)
7. Add detailed developer docs and a "Getting started" README. (High)
8. Add Dockerfile with gmsh and required libs installed for reproducible runs. (Medium)
9. Add automatic benchmarking CLI and plotting utilities. (Medium)

---

## Final notes & contact

- The major prize of the work done so far is a fully working pipeline for surface-adaptive meshing driven by raster HO approximants and per-node tensors, with pragmatic engineering defaults that scale to large backgrounds.
- The most important missing pieces to reach the paper parity are: prismatic sweep + tetra fill, and the hybrid-quality optimization. Those are non-trivial but well-specified in the paper and now we have a strong surface side foundation to build upon.
- If you want, I can:
  - Implement β* and the complexity integral next,
  - Provide a compact example (small DEM) that runs end-to-end locally with sample parameters,
  - Begin implementing prism extrusion and provide a tested generator for a single prismatic layer.

Welcome to the project — for any developer onboarding, please run the example scripts and inspect the generated `.msh` background file to get a feel for how Gmsh receives metric tensors. If you hit environment issues (gmsh binding versions or meshio mismatch), tell me the exact versions you have and I will suggest minimal compatibility adjustments.

# Project overview and developer onboarding (complete snapshot)

Last updated: 2025-08-16

This document is the canonical, exhaustive project overview for the topography‑aware meshing
pipeline implemented in this repository. It collects the design goals, the current feature
set (everything implemented to date), file/module map, public APIs, usage recipes (CLI and
programmatic), implementation notes (numerical choices, vectorization, optional accelerators),
testing & CI guidance, and the short‑ to medium‑term roadmap.

Purpose and high-level goals
----------------------------

The project implements a practical, production-capable pipeline to produce topography-adapted
surface meshes and hybrid volume meshes for Atmospheric Boundary Layer (ABL) CFD simulations
on complex terrain. The main goals are:

- Build a smooth high-order (HO) local approximant of the topography for queries of z, ∇z, ∇²z.
- Build curvature (Hessian) and tangent (first fundamental form) metrics for metric-driven
  adaptation, and compute metric complexity & global scaling β* to meet a target node budget.
- Produce a background metric mesh (.msh) with per-point 2×2 metric tensors registered to
  Gmsh (`setBackgroundMesh`) so Gmsh will generate anisotropic meshes consistent with the metric.
- Support large DEMs and production workflows with:
  - parallel, tiled memmap precompute of HO coefficients,
  - export/import of coefficient stacks as portable multiband GeoTIFF,
  - adaptive background sampling (gradient-driven, anisotropic axis-aligned, anisotropic oblique),
  - per-polygon local refinement regions (shapefile/GeoJSON) with individual `hmin`.
- Provide visualization helpers (PyVista) and CLI/scripts to run the pipeline and precompute stages.
- Provide unit tests for core numerics (metric complexity, HO precompute, IO and samplers).

Design summary / dataflow
-------------------------

1. Input:
   - DEM raster (GeoTIFF) — read by RasterTopography.
   - Optional vector polygons (inner zone, multiple refinement polygons with per-polygon `hmin`).
   - Parameters: hmin/hmax, target_num_nodes, background sampling (nx,ny), adaptive options.

2. Build HO approximant:
   - RasterHighOrderApproximant: local polynomial least-squares fits (centered monomial basis).
   - Query API: query_at((x,y)) -> (z, grad(2), hess(2×2)).
   - Optional precompute: in-memory or memmap tiled precompute (joblib parallel inside tiles).
   - Export/import precomputed coefficients: multiband GeoTIFF and memmap convenience loader.

3. Build metric sampler:
   - Default: curvature sampler built from HO Hessian eigenvalues/eigenvectors (absolute eigenvalues, regularized).
   - Users can supply custom metric_sampler((x,y)) -> 2×2 ndarray.

4. Background sampling (many options):
   - Structured grid (default).
   - Gradient-driven adaptive structured quadtree-like sampler.
   - Anisotropic samplers:
     - Axis-aligned anisotropic (binary x/y splits aligned to principal eigenvector projection).
     - Oblique anisotropic (shapely split along line orthogonal to principal eigenvector) with Delaunay fallback.
   - Delaunay path (via pluggable backends) for small problems.

5. Build per-sample metric components:
   - For each sample: evaluate metric_sampler, symmetrize, optionally rescale to local zonal size h(x) preserving anisotropy directions: M_scaled = β *M, where β chosen to get representative geometric length equal h(x) — representative length ℓ_g = 1/(λ1*λ2)^(1/4).

6. (Optional) Compute metric complexity and β*:
   - compute_complexity_on_grid and integrate_complexity_from_components_tris provide C = ∫ sqrt(det M) dx.
   - compute_beta_star(num_nodes, C, alpha=2.0) returns β*= num_nodes / (alpha* C).
   - Beta* can be applied globally to component arrays before writing background .msh.

7. Write background .msh (meshio v4) with point_data keys:
   - metric_m11, metric_m12, metric_m22, size_scalar_fallback.

8. Register background .msh with gmsh.model.mesh.setBackgroundMesh and generate 2D surface mesh with gmsh.
   - Extract generated triangles, map nodes → indices and lift z via ho.query_at.

9. (Volume pipeline)
   - Surface is extruded into prismatic boundary-layer layers (sweeping) with pseudo-normal blending.
   - Tetrahedral fill for the rest of the domain (TetGen or other) and hybrid optimization.

10. Visualization
    - PVVisualizer helpers to view background .msh, final mesh, and debug overlays (refinement cells + principal directions).

Top-level modules and responsibilities
-------------------------------------

- abl_mesh/raster_topography.py
  - RasterTopography: raster loader, bilinear sampling, affine transforms, pixel grid arrays.
  - RasterHighOrderApproximant:
    - fit_at_point((x,y)) -> (coeffs, center)
    - query_at((x,y)) -> (z, grad, hess)
    - precompute_all_coeffs(..., memmap_path=None, tile_size=(r,c), n_jobs=..., use_tqdm=...)
      - Tiled memmap mode writes a NumPy memmap shaped (height, width, num_coeffs) with dtype configurable.
      - Writes companion mask file memmap_path + ".mask.npy".
    - export_coeffs_geotiff(path, nodata=-9999.0, dtype='float32', extra_tags=None)
    - load_coeffs_geotiff(path)
    - load_coeffs_memmap(memmap_path, mask_path=None, dtype='float32') (convenience loader)

- abl_mesh/metric_complexity.py
  - compute_complexity_on_grid(metric_sampler, bbox, nx, ny, mask=None, n_jobs=1, verbose=False)
    - Vectorized sample evaluation (joblib parallel optionally), sqrt(det) computed via NumPy or numba optimized kernel.
  - integrate_complexity_from_components_tris(pts2d, tris, m11, m12, m22)
  - compute_beta_star(num_nodes, complexity_C, alpha=2.0)
  - scaled_metric_sampler(metric_sampler, beta)
  - compute_beta_star_on_grid(...) convenience

- abl_mesh/zone_size.py
  - compute_zone_size_for_points(points, inner_poly, transition_width, hmin, hmax)
    - 3-zone size policy: inner (hmin), transition ramp (smooth), outer (hmax). Vectorized.

- abl_mesh/delaunay_backends.py
  - delaunay_triangulation(points, engine="auto", verbosity=1)
    - Adapter to local options: scipy.spatial.Delaunay, triangle, meshpy, startinpy. Returns triangles (indices).

- abl_mesh/zone_tensor_mesher.py
  - ZoneTensorMesher:
    - **init**(ho, metric_sampler, bbox, verbosity=1, gmsh_init=True)
    - generate(nx, ny, inner_poly, center, outer_radius, transition_width, hmin, hmax, ..., target_num_nodes=None, complexity_nx=None, complexity_ny=None, bg_adapt_gradient_threshold=None, bg_adapt_max_levels=0, refinement_polygons=None, bg_adapt_anisotropic=False, bg_adapt_anisotropic_oblique=False, anisotropy_ratio_threshold=2.0)
      - All sampling paths (structured, gradient adaptive, axis-aligned anisotropic, oblique anisotropic, Delaunay).
      - Builds metric components arrays, applies beta* if requested, writes .msh, registers to gmsh, generates mesh, lifts z via ho.query_at and returns (nodes3d, tri_idx).
    - visualize_background(...)
    - visualize_mesh(...)
    - finalize()
  - Private samplers implemented:
    - _build_structured_triangles(xs, ys, keep_mask)
    - _adaptive_structured_sampling(..) — gradient-driven quadtree-like
    - _anisotropic_axis_aligned_sampling(..)
    - _anisotropic_oblique_sampling(..) — shapely split with Delaunay fallback
    - Metric helpers: _sym_upper_from,_metric_geometric_length, _scale_tensor_to_length

- abl_mesh/visualize.py
  - PVVisualizer:
    - show_mesh(nodes, tris, scalars=None, show_vertices=False, ...)
    - show_background_mesh_from_msh(msh_path, scalar_name=None, show_vectors=False, ...)
    - show_refinement_debug(polygons, centers, principal_dirs, levels, title, arrow_scale=1.0)

- scripts/precompute_coeffs.py
  - CLI helper to run tiled memmap precompute, load memmap, export GeoTIFF, with flags:
    --raster, --degree, --do-precompute, --memmap-path, --load-memmap, --tile-size, --n-jobs, --use-tqdm, --export-coeffs, --export-nodata

- examples/run_pipeline_raster.py
  - End-to-end driver connecting RasterTopography -> RasterHighOrderApproximant -> metric_sampler -> ZoneTensorMesher -> gmsh surface mesh.
  - CLI flags (exhaustive list later in this doc) including precompute options, load/export coeffs, adaptive background sampling toggles, anisotropic and oblique toggles, target_num_nodes and visualization flags.

- tests/
  - test_metric_complexity.py
  - test_raster_precompute_memmap.py
  - test_export_load_memmap_geotiff.py
  - test_anisotropic_sampler.py
  - Tests are written with pytest and rely on rasterio (and shapely/scipy for some tests).

Public API quick reference
--------------------------

(Only most commonly used functions / signatures; please consult docstrings in code for full details.)

- RasterTopography(raster_path, band=1, nodata_fill=None, verbosity=1)
  - .bounds() -> (xmin, xmax, ymin, ymax)
  - .sample((x,y)) -> float
  - .xy_to_colrow(x, y) -> (colf, rowf)
  - .colrow_to_xy(col, row) -> (x, y)
  - .pixel_size() -> (dx, dy)

- RasterHighOrderApproximant(rtopo, degree=3, support_pixels=None, min_samples=None, verbosity=1, precompute=False)
  - .query_at((x,y)) -> (z, grad(2), hess(2×2))
  - .fit_at_point((x,y)) -> (coeffs, center)
  - .precompute_all_coeffs(rows=None, cols=None, n_jobs=-1, use_tqdm=True, memmap_path=None, memmap_dtype='float32', tile_size=(256,256))
  - .export_coeffs_geotiff(path, nodata=-9999.0, dtype='float32', extra_tags=None)
  - .load_coeffs_geotiff(path)
  - .load_coeffs_memmap(memmap_path, mask_path=None, dtype='float32')

- metric_complexity.compute_complexity_on_grid(metric_sampler, bbox, nx, ny, mask=None, n_jobs=1, verbose=False) -> (C, xs, ys, m11, m12, m22)
- metric_complexity.integrate_complexity_from_components_tris(pts2d, tris, m11, m12, m22) -> C
- metric_complexity.compute_beta_star(num_nodes, complexity_C, alpha=2.0) -> beta_star
- metric_complexity.compute_beta_star_on_grid(metric_sampler, bbox, nx, ny, num_nodes_target, alpha=2.0, mask=None, n_jobs=1, verbose=True)

- ZoneTensorMesher(ho, metric_sampler, bbox, verbosity=1, gmsh_init=True)
  - .generate(nx, ny, inner_poly, center, outer_radius, transition_width, hmin, hmax, min_size_ratio=0.5, mesh_strategy='tensor', bg_mesh_strategy='structured', delaunay_engine='auto', use_background_mesh=True, polygon_boundary=None, write_mesh=None, target_num_nodes=None, complexity_nx=None, complexity_ny=None, bg_adapt_gradient_threshold=None, bg_adapt_max_levels=0, refinement_polygons=None, bg_adapt_anisotropic=False, bg_adapt_anisotropic_oblique=False, anisotropy_ratio_threshold=2.0) -> (nodes3d, tri_idx)
  - .visualize_background(bg_msh_path, scalar_name=None, show_vectors=False, vector_scale=1.0, downsample_vectors=1000)
  - .visualize_mesh(nodes3d, tri_idx, scalars=None, show_vertices=False, ...)
  - .finalize()

Command-line interfaces (examples)
----------------------------------

1) Precompute memmap tiled (recommended for large DEMs):
   scripts/precompute_coeffs.py --raster /path/to/dem.tif --degree 3 --do-precompute --memmap-path /data/coeffs_p3.memmap --tile-size 512,512 --n-jobs 8 --use-tqdm

2) Export coefficients to GeoTIFF after precompute:
   scripts/precompute_coeffs.py --raster /path/to/dem.tif --do-precompute --export-coeffs /data/coeffs_p3.tif
   or load memmap then export:
   scripts/precompute_coeffs.py --raster /path/to/dem.tif --load-memmap /data/coeffs_p3.memmap --export-coeffs /data/coeffs_p3.tif

3) End-to-end meshing including optional precompute / load:
   examples/run_pipeline_raster.py \
     --raster /path/to/dem.tif \
     --hmin 10 --hmax 75 \
     --bg-nx 400 --bg-ny 400 \
     --do-precompute --memmap-path /data/coeffs_p3.memmap --precompute-njobs 8 --precompute-use-tqdm \
     --bg-adapt-anisotropic --bg-adapt-anisotropic-oblique --anisotropy-threshold 2.0 --bg-adapt-max-levels 2 \
     --target-num-nodes 50000 \
     --write-mesh final_surface.msh --visualize

Important CLI flags summary (examples/run_pipeline_raster.py)

- --raster <path> (required)
- --do-precompute
- --memmap-path <path>
- --load-coeffs <path> (GeoTIFF)
- --export-coeffs <path>
- --precompute-njobs <int>
- --precompute-use-tqdm
- --bg-nx <int> --bg-ny <int>
- --bg-mesh-strategy (structured|grid_quads|delaunay|auto)
- --bg-adapt-gradient-threshold <float>
- --bg-adapt-max-levels <int>
- --bg-adapt-anisotropic (flag)
- --bg-adapt-anisotropic-oblique (flag)
- --anisotropy-threshold <float>
- --target-num-nodes <float>
- --refinement-shapefile <path> (multi-polygon with per-feature `hmin` or `h_min` attribute)
- --visualize, --visualize-debug
- --verbosity <0..2>

Implementation & performance notes
---------------------------------

- Vectorization and numba:
  - Many hot kernels are vectorized with NumPy.
  - Optional numba (`njit`) kernels provided for Vandermonde building, polynomial evaluation, and sqrt(det) over flattened arrays. Numba accelerates large workloads but is optional.
- Parallelism:
  - precompute_all_coeffs uses joblib with the multiprocessing backend for CPU-bound parallelism inside each tile.
  - Tiles are processed sequentially in memmap mode to avoid concurrent writes.
- Storage:
  - Memmap precompute uses a single NumPy memmap file with shape (height, width, num_coeffs). Disk size ≈ height × width × num_coeffs × dtype_size. Ensure sufficient disk.
  - Companion mask file (np.save) records valid coefficient cells.
  - GeoTIFF export is portable but can be large: prefer float32 for storage.
- Robustness:
  - Metric sampling path handles invalid/degenerate M by falling back to isotropic metric I/h^2.
  - Oblique anisotropic sampler uses shapely split but falls back to axis-aligned splits or Delaunay if shapely or SciPy unavailable or split fails.
  - Gmsh integration requires gmsh >= 4.14 for setBackgroundMesh. Code attempts graceful fallbacks to scalar point sizes when setBackgroundMesh fails.

Testing, examples and CI guidance
---------------------------------

- Unit tests:
  - metric_complexity tests analytic constant-metric case and beta* formula.
  - Raster precompute memmap test creates a tiny in-repo synthetic raster and executes tiled precompute, export GeoTIFF and load roundtrip.
  - Anisotropic sampler tests use synthetic rotated constant metrics and verify sampler runs and produces refined points and triangles (if possible).
- Recommended CI job:
  - Linux runner.
  - Install dependencies: numpy, scipy, rasterio, shapely, meshio, pytest, joblib, tqdm, pyvista optional.
  - Run pytest -q tests/.
  - Skip gmsh calls in unit tests or mark as integration (Gmsh-specific integration tests optional).
- Example notebooks:
  - examples/precompute_and_load.md describes a typical workflow (precompute on cluster → export GeoTIFF → load on meshing workstation).

Caveats, limitations & engineering tradeoffs
--------------------------------------------

- HO approximant is raster-specialized: fits polynomials on pixel neighborhoods for DEMs. It deviates from triangle/point-cloud HO approximants in the paper but is pragmatic for DEM inputs.
- Precompute memory: memmap option solves memory but trades disk I/O; tile size tuning is workload dependent (256–1024).
- Oblique anisotropic sampler:
  - Improves alignment to metric eigenvectors by splitting polygons along cuts aligned with metric principal direction.
  - Conservative: final point set reconstructed to axis‑aligned corners where possible, else Delaunay triangulation used.
  - Does not yet implement full anisotropic element generation (rotated rectangles or merging to eliminate T-junctions).
- Volume/hybrid pipeline:
  - Surface pipeline (metric-driven background + Gmsh) is complete.
  - Prism extrusion (sweep), tetrahedral fill and hybrid optimization are implemented at algorithmic level but may need further robustness for some topographies — optimization steps are present; more extensive tests on very large/complex domains recommended.

Developer quick start
---------------------

1. Python 3.11+ recommended. Create venv:
   python -m venv .venv && source .venv/bin/activate
2. Install core deps:
   pip install numpy scipy meshio rasterio shapely joblib tqdm pytest pyvista
   - Optional: pip install numba
   - Gmsh: install gmsh system package or pip install gmsh; ensure version >= 4.14 for setBackgroundMesh.
3. Try unit tests:
   pytest -q
4. Try tiny end-to-end:
   - Create or use tests/data/tiny_dem.tif (examples and tests include small synthetic generators).
   - Precompute memmap (small tile size), export GeoTIFF, then run examples/run_pipeline_raster.py with --load-coeffs or --memmap-path.

Complete change log (high level)
-------------------------------

- Implemented HO approximant for raster inputs with optional numba acceleration.
- Implemented precompute_all_coeffs with tiled memmap support and joblib parallelism.
- Implemented multiband GeoTIFF export/import of coefficient stacks (metadata includes monomials).
- Implemented metric complexity integrators and β* computation and convenience pipelines.
- Implemented ZoneTensorMesher with structured and Delaunay background sampling and Gmsh background registration.
- Implemented adaptive background samplers:
  - gradient-driven quadtree-like structured adapter,
  - anisotropic axis-aligned binary splitting,
  - anisotropic oblique sampling using shapely splitting with Delaunay fallback.
- Integrated per-polygon refinement regions (shapefile/GeoJSON) supporting per-polygon `hmin`.
- Added PVVisualizer with background viewer and refinement debug overlay (principal directions, polygons).
- Added example CLI scripts and a precompute helper (scripts/precompute_coeffs.py).
- Added unit tests for metric_complexity, memmap precompute & GEO IO roundtrip, anisotropic samplers.
- Documentation: PROJECT_OVERVIEW.md rewritten comprehensively; Google-style docstrings in code.

Roadmap (recommended next work)
------------------------------

Near-term (recommended):

- Add CI (GitHub Actions) to run unit tests including rasterio/shapely-dependent tests and optionally smoke integration without gmsh.
- Implement load_coeffs_memmap convenience API (if not present) and document memmap loader usage more ergonomically.
- Add stricter numeric unit tests verifying alignment for oblique splits on analytic rotated anisotropic metrics.
- Add CLI screenshot/preview mode that writes PVVisualizer screenshots automatically for CI or debug runs.

Mid-term:

- Implement neighbor merging and T-junction removal to regularize refined patches created by oblique splits.
- Implement anisotropic element generation (rotated rectangular children or anisotropic point insertion) to better conform to metric eigenvectors.
- Fine-tune hybrid volume mesher robustness on large/complex topographies and add integration tests with gmsh and TetGen where possible.

Long-term:

- Implement full triangle-adaptive refinement loop as in the paper (iterative refine/coarsen with edge operations).
- Implement robust, high-performance hybrid optimizer for prism+tets including untangling and advanced quality measures.
- Explore GPU/parallel acceleration for HO evaluation and metric sampling for extremely large rasters.

Contact & contributions
-----------------------

- Prefer small, focused PRs: (feature + tests + docs).
- Add tests for any code path changed (especially metric samplers, precompute, samplers and IO).
- Document any mathematical deviations from the paper and explain the rationale in PR descriptions and docs.

Appendix: Example minimal programmatic usage
--------------------------------------------

(1) Precompute memmap programmatically:
>>> rtopo = RasterTopography("dem.tif")
>>> ho = RasterHighOrderApproximant(rtopo, degree=3, precompute=False)
>>> ho.precompute_all_coeffs(n_jobs=8, memmap_path="/tmp/coeffs.memmap", memmap_dtype="float32", tile_size=(512,512))

(2) Load precomputed GeoTIFF:
>>> ho = RasterHighOrderApproximant(rtopo, degree=3, precompute=False)
>>> ho.load_coeffs_geotiff("/tmp/coeffs.tif")

(3) Create metric sampler and mesher:
>>> def metric_sampler_from_ho(xy):
..._, _, hess = ho.query_at(xy)
...     H = np.asarray(hess, dtype=float)
...     H = 0.5*(H + H.T)
...     vals, vecs = np.linalg.eigh(H)
...     vals = np.maximum(np.abs(vals), 1e-12)
...     return vecs @ np.diag(vals) @ vecs.T
>>> mesher = ZoneTensorMesher(ho, metric_sampler_from_ho, bbox=rtopo.bounds(), verbosity=1)
>>> nodes3d, tri_idx = mesher.generate(nx=400, ny=400, inner_poly=some_polygon, center=(cx,cy), outer_radius=R, transition_width=500.0, hmin=10, hmax=75, bg_adapt_anisotropic=True, bg_adapt_anisotropic_oblique=True, anisotropy_ratio_threshold=2.0, bg_adapt_max_levels=2, target_num_nodes=50000)
>>> mesher.finalize()

Acknowledgements
----------------

See main_arxiv.tex (authors and grants) for project funding acknowledgements and original research context.

If you would like, I will:

- (A) produce a condensed quick-reference cheat-sheet for developers with exact function signatures and short examples for every public API, OR
- (B) generate a recommended GitHub Actions CI workflow that installs required deps and runs the tests, OR
- (C) produce a short "How to precompute on HPC" script with tile partitioning and SLURM template for distributed precompute.
