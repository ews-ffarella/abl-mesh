"""
Unit tests for RasterHighOrderApproximant precompute (memmap tiled) and GeoTIFF export/load.

The test creates a tiny synthetic raster (5x6) with a simple analytic surface,
runs a small-degree precompute using memmap tiles, exports to GeoTIFF and loads back.
This test requires rasterio and numpy (and optionally joblib).
"""

import os

import numpy as np
import rasterio
from rasterio.transform import Affine

from abl_mesh.raster_topography import RasterHighOrderApproximant, RasterTopography


def _create_test_raster(path, nx=6, ny=5):
    """Create a tiny GeoTIFF with a simple analytic height (z = x + 2*y)."""
    xmin, ymin, dx = 0.0, 0.0, 1.0
    transform = Affine.translation(xmin, ymin) * Affine.scale(dx, dx)
    data = np.empty((ny, nx), dtype=np.float32)
    xs = np.arange(nx) * dx + xmin
    ys = np.arange(ny) * dx + ymin
    for j in range(ny):
        for i in range(nx):
            data[j, i] = xs[i] + 2.0 * ys[j]
    profile = {
        "driver": "GTiff",
        "height": ny,
        "width": nx,
        "count": 1,
        "dtype": "float32",
        "crs": None,
        "transform": transform,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_memmap_precompute_and_export(tmp_path):
    tif = tmp_path / "test_dem.tif"
    _create_test_raster(str(tif), nx=6, ny=5)

    rtopo = RasterTopography(str(tif), verbosity=0)
    ho = RasterHighOrderApproximant(rtopo, degree=2, verbosity=0, precompute=False)

    memmap_file = str(tmp_path / "coeffs.memmap")
    # Run tiled precompute with small tile size to exercise tile logic
    ho.precompute_all_coeffs(
        n_jobs=1, use_tqdm=False, memmap_path=memmap_file, memmap_dtype="float32", tile_size=(2, 3)
    )
    assert ho._precomputed is not None
    assert ho._precompute_mask is not None
    # At least one valid coefficient cell should exist (interior cells)
    assert ho._precompute_mask.sum() >= 1

    # Export to GeoTIFF
    out_tif = str(tmp_path / "coeffs.tif")
    ho.export_coeffs_geotiff(out_tif, nodata=-9999.0, dtype="float32")
    assert os.path.exists(out_tif)

    # Load into a fresh approximant and compare mask counts
    ho2 = RasterHighOrderApproximant(rtopo, degree=2, verbosity=0, precompute=False)
    ho2.load_coeffs_geotiff(out_tif)
    assert ho2._precomputed is not None
    # basic sanity: number of valid cells should be equal
    assert int(ho2._precompute_mask.sum()) == int(ho._precompute_mask.sum())
