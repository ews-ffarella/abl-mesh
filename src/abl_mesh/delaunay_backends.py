"""
Small abstraction layer to choose a Delaunay triangulation backend.

Supported backends (attempted in this order when engine='auto'):
 - "startinpy" : (optional) fast triangulation library if installed
 - "triangle"  : Jonathan Shewchuk's Triangle via the 'triangle' python wrapper (if installed)
 - "meshpy"    : meshpy.triangle (if installed)
 - "scipy"     : scipy.spatial.Delaunay (fallback, robust but can be slow for very large point clouds)

API:
 - delaunay_triangulation(points[:,2]) -> (triangles (M,3) int)
   triangles are indices into points

Notes / performance
 - For very large sample grids (hundreds of thousands of points) full Delaunay triangulation can be expensive.
   Consider lowering sampling resolution, building a structured background mesh (regular grid -> quad -> split into triangles),
   or using a specialized library (startinpy / tetwild / CGAL bindings).
 - This wrapper only returns the connectivity (triangles). It does not write files.
"""

from __future__ import annotations

import numpy as np

# Try optional backends
_HAS_STARTINPY = False
_HAS_TRIANGLE = False
_HAS_MESHPY = False
_HAS_SCIPY = False
try:
    _HAS_STARTINPY = True
except Exception:
    _HAS_STARTINPY = False

try:
    import triangle as _triangle  # Jonathan Shewchuk's Triangle python wrapper

    _HAS_TRIANGLE = True
except Exception:
    _HAS_TRIANGLE = False

try:
    from meshpy import triangle as _meshpy_triangle

    _HAS_MESHPY = True
except Exception:
    _HAS_MESHPY = False

try:
    from scipy.spatial import Delaunay as _scipy_delaunay

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def delaunay_triangulation(
    points: np.ndarray, engine: str = "auto", verbosity: int = 1
) -> np.ndarray:
    """
    Compute a 2D Delaunay triangulation of points (N,2).

    Parameters
    ----------
    points : (N,2) ndarray
    engine : "auto" | "startinpy" | "triangle" | "meshpy" | "scipy"
      If "auto", choose the fastest available in order: startinpy, triangle, meshpy, scipy.
    verbosity : int

    Returns
    -------
    triangles : (M,3) int ndarray with indices into points
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("points must be (N,2) array")

    chosen = engine
    if engine == "auto":
        if _HAS_STARTINPY:
            chosen = "startinpy"
        elif _HAS_TRIANGLE:
            chosen = "triangle"
        elif _HAS_MESHPY:
            chosen = "meshpy"
        elif _HAS_SCIPY:
            chosen = "scipy"
        else:
            raise RuntimeError(
                "No Delaunay backend available: install scipy, triangle, meshpy or startinpy"
            )
    if verbosity:
        print(f"[delaunay] using engine: {chosen}")

    if chosen == "startinpy":
        if not _HAS_STARTINPY:
            raise RuntimeError("startinpy not available")
        # startinpy API is hypothetical here; adapt to installed startinpy if present.
        import startinpy as sp

        tri = sp.delaunay(pts)  # assume returns (M,3)
        return np.asarray(tri, dtype=int)

    if chosen == "triangle":
        if not _HAS_TRIANGLE:
            raise RuntimeError("triangle wrapper not available")
        # triangle expects dict with 'vertices'
        A = {"vertices": pts}
        # 'Q' quiet, 'z' zero-based, 'p' PSLG; but we only have pointset, use 'Qz'
        t = _triangle.triangulate(A, "Qz")
        if "triangles" not in t:
            raise RuntimeError("triangle failed to produce triangles")
        return np.asarray(t["triangles"], dtype=int)

    if chosen == "meshpy":
        if not _HAS_MESHPY:
            raise RuntimeError("meshpy not available")
        # meshpy expects a boundary polygon; we create convex hull as boundary to ensure triangulation
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
        info = _meshpy_triangle.MeshInfo()
        info.set_points(pts.tolist())
        info.set_facets([(i, (i + 1) % len(hull.vertices)) for i in range(len(hull.vertices))])
        mesh = _meshpy_triangle.build(info)
        tris = np.array(mesh.elements, dtype=int)
        return tris

    if chosen == "scipy":
        if not _HAS_SCIPY:
            raise RuntimeError("scipy.spatial.Delaunay not available")
        tri = _scipy_delaunay(pts)
        return np.asarray(tri.simplices, dtype=int)

    raise ValueError(f"Unknown engine '{engine}'")
