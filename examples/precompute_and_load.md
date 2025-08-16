```markdown
# Example: Precompute HO coefficients, export to GeoTIFF, load and use in meshing

This walkthrough demonstrates a typical workflow: precompute coefficients (memmap tiled),
export to a GeoTIFF, load coefficients on another machine, and run the mesher.

Requirements:
- Python 3.11+
- rasterio, numpy, scipy, meshio, gmsh (for meshing step), abl_mesh package on PYTHONPATH

1) Precompute (on a big machine) - memmap tiled:
```bash
python scripts/precompute_coeffs.py \
  --raster /path/to/dem.tif \
  --degree 3 \
  --do-precompute \
  --memmap-path /data/coeffs_p3.memmap \
  --tile-size 512,512 \
  --n-jobs 8 \
  --use-tqdm
```
This produces:
- /data/coeffs_p3.memmap  (NumPy memmap binary)
- /data/coeffs_p3.memmap.mask.npy (boolean mask for validity)

2) Optional: Export coefficients to a portable GeoTIFF (smaller machines may prefer this):
```bash
python scripts/precompute_coeffs.py \
  --raster /path/to/dem.tif \
  --load-memmap /data/coeffs_p3.memmap \
  --export-coeffs /data/coeffs_p3.tif \
  --export-nodata -9999
```

3) On the meshing machine: load memmap directly (fast) or load GeoTIFF:
```python
from abl_mesh.raster_topography import RasterTopography, RasterHighOrderApproximant
rtopo = RasterTopography("/path/to/dem.tif", verbosity=1)
ho = RasterHighOrderApproximant(rtopo, degree=3, precompute=False, verbosity=1)

# Option A: load memmap
ho.load_coeffs_memmap("/data/coeffs_p3.memmap", "/data/coeffs_p3.memmap.mask.npy")

# Option B: load GeoTIFF
ho.load_coeffs_geotiff("/data/coeffs_p3.tif")

# Now ho.query_at((x,y)) will use precomputed polynomial coefficients for cell-centered queries.
```

4) Run mesher (example minimal invocation)
```python
from abl_mesh.zone_tensor_mesher import ZoneTensorMesher
# create metric sampler from ho (curvature-based) then
mesher = ZoneTensorMesher(ho, metric_sampler, bbox=rtopo.bounds(), verbosity=1, gmsh_init=True)
nodes3d, tri_idx = mesher.generate(nx=400, ny=400, inner_poly=..., center=..., outer_radius=..., transition_width=500.0, hmin=10, hmax=75, target_num_nodes=50000)
```

Notes:
- For production workflows prefer memmap precompute and memmap load; GeoTIFF is recommended for portability and archival.
- Always verify monomial ordering and metadata when exchanging coefficient files across machines.
```