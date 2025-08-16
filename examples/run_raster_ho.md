```text
Example: using the raster-based high-order approximant

1) Install requirements:
   pip install numpy scipy rasterio meshio pyvista tqdm

2) Minimal usage:

```python
from abl_mesh.raster_topography import RasterTopography, RasterHighOrderApproximant

# path to a DEM GeoTIFF
raster_path = "path/to/dem.tif"

rtopo = RasterTopography(raster_path, verbosity=1)
ho = RasterHighOrderApproximant(rtopo, degree=3, support_pixels=6, verbosity=2, precompute=False)

# Query at arbitrary point (x,y) in raster CRS
x, y = 100234.5, 4532100.2
z, grad, hess = ho.query_at((x, y))
print("z:", z)
print("grad:", grad)
print("hess:", hess)

# If your mesh nodes coincide with raster cell centers and you will query many points,
# consider precomputing all coefficients (may take time and memory):
# ho.precompute_all_coeffs()
```

Notes:
- If your mesh pipeline queries many points located near raster cell centers, enabling precompute
  will significantly speed up the queries (the per-point query becomes a simple polynomial evaluation).
- If your points are arbitrary (not aligned with raster grid), on-the-fly fits are used.
- For very large rasters consider:
  - Using structured background sampling (we implemented that elsewhere) and coarser sampling.
  - Splitting work into tiles and precomputing coefficients only where needed.