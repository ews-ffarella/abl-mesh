"""
Unit test: memmap tiled precompute -> load via convenience loader -> export GeoTIFF -> load GeoTIFF

This test exercises:
 - precompute_all_coeffs(memmap_path=...)
 - RasterHighOrderApproximant.load_coeffs_memmap()
 - RasterHighOrderApproximant.export_coeffs_geotiff()
 - RasterHighOrderApproximant.load_coeffs_geotiff()

Uses a tiny synthetic raster created in the test to remain fast.
"""
import os
import numpy as np
import tempfile
from rasterio.transform import Affine
import rasterio

from abl_mesh.raster_topography import RasterTopography, RasterHighOrderApproximant


def _create_test_raster(path, nx=6, ny=5):
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
        "transform": transform
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
    return path


def test_memmap_precompute_export_import(tmp_path):
    tif = tmp_path / "tiny_dem.tif"
    _create_test_raster(str(tif), nx=6, ny=5)

    rtopo = RasterTopography(str(tif), verbosity=0)
    ho = RasterHighOrderApproximant(rtopo, degree=2, verbosity=0, precompute=False)

    memmap_file = str(tmp_path / "coeffs.memmap")
    # Run tiled precompute with small tile size to exercise tile logic
    ho.precompute_all_coeffs(n_jobs=1, use_tqdm=False, memmap_path=memmap_file, memmap_dtype="float32", tile_size=(2,3))
    assert ho._precomputed is not None
    assert ho._precompute_mask is not None
    assert ho._precompute_mask.sum() >= 1

    # Now load the memmap into a fresh approximant via convenience loader
    ho2 = RasterHighOrderApproximant(rtopo, degree=2, precompute=False, verbosity=0)
    ho2.load_coeffs_memmap(memmap_file, memmap_file + ".mask.npy")
    assert ho2._precomputed is not None
    assert ho2._precompute_mask is not None
    assert int(ho2._precompute_mask.sum()) == int(ho._precompute_mask.sum())

    # Export to GeoTIFF
    out_tif = str(tmp_path / "coeffs_roundtrip.tif")
    ho2.export_coeffs_geotiff(out_tif, nodata=-9999.0, dtype="float32")
    assert os.path.exists(out_tif)

    # Load into a fresh approximant via GeoTIFF loader and compare masks
    ho3 = RasterHighOrderApproximant(rtopo, degree=2, precompute=False, verbosity=0)
    ho3.load_coeffs_geotiff(out_tif)
    assert ho3._precomputed is not None
    assert int(ho3._precompute_mask.sum()) == int(ho._precompute_mask.sum())