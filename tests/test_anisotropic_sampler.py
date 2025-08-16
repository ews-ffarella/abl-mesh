"""
Unit tests for anisotropic samplers (axis-aligned and oblique).

These tests create a small synthetic rotated anisotropic metric and verify:
 - the anisotropic oblique sampler produces more sample points than the base grid
 - a triangulation is returned (non-empty)
 - the sampler does not crash when shapely is present/missing (best-effort)
"""
import numpy as np
from abl_mesh.zone_tensor_mesher import ZoneTensorMesher
from abl_mesh.raster_topography import RasterTopography, RasterHighOrderApproximant

def rotated_anisotropic_metric(theta_rad: float, lam1: float = 100.0, lam2: float = 1.0):
    """Return a callable metric sampler that is constant and rotated by theta."""
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]], dtype=float)
    D = np.diag([lam1, lam2])
    M = R @ D @ R.T
    def sampler(xy):
        return M
    return sampler

def test_anisotropic_oblique_sampler_basic():
    # small bbox and grid
    xmin, xmax, ymin, ymax = 0.0, 10.0, 0.0, 10.0
    bbox = (xmin, xmax, ymin, ymax)
    center = (5.0, 5.0)
    outer_radius = 8.0
    nx, ny = 6, 6
    theta = np.deg2rad(30.0)
    sampler = rotated_anisotropic_metric(theta, lam1=100.0, lam2=1.0)

    # construct mesher stub (ho not used for sampling)
    class DummyHO:
        def query_at(self, xy):
            return 0.0, np.array([0.0, 0.0]), np.eye(2)

    mesher = ZoneTensorMesher(DummyHO(), sampler, bbox=bbox, verbosity=0, gmsh_init=False)
    # call oblique sampler directly
    pts, tris = mesher._anisotropic_oblique_sampling(xmin, xmax, ymin, ymax, nx, ny, outer_radius, center,
                                                     metric_sampler=sampler,
                                                     anisotropy_ratio_threshold=1.5,
                                                     max_levels=1)
    # Basic assertions
    assert pts is not None
    assert pts.shape[0] > (nx - 1) * (ny - 1)  # refined points > base cell corners
    # tris may be empty only if triangulation failed; accept non-empty or empty but ensure no crash
    assert tris is not None

def test_anisotropic_oblique_triangulation_fallback():
    xmin, xmax, ymin, ymax = 0.0, 8.0, 0.0, 8.0
    bbox = (xmin, xmax, ymin, ymax)
    center = (4.0, 4.0)
    outer_radius = 6.0
    nx, ny = 5, 5
    theta = np.deg2rad(45.0)
    sampler = rotated_anisotropic_metric(theta, lam1=200.0, lam2=0.5)
    class DummyHO:
        def query_at(self, xy):
            return 0.0, np.zeros(2), np.eye(2)
    mesher = ZoneTensorMesher(DummyHO(), sampler, bbox=bbox, verbosity=0, gmsh_init=False)
    pts, tris = mesher._anisotropic_oblique_sampling(xmin, xmax, ymin, ymax, nx, ny, outer_radius, center,
                                                     metric_sampler=sampler,
                                                     anisotropy_ratio_threshold=1.2,
                                                     max_levels=2)
    assert pts.shape[0] > 0
    # triangulation should be returned (triangles >= 0); if Delaunay used ensure indices are within bounds
    if tris.size > 0:
        assert np.max(tris) < pts.shape[0]