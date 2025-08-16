"""
ZoneTensorMesher: background-metric builder and Gmsh-driven surface mesher.

This module provides the ZoneTensorMesher class which automates the workflow to:
 - sample a background set of points (structured, adaptively refined, or Delaunay),
 - evaluate/construct per-point 2x2 metric tensors (tensor or isotropic strategies),
 - optionally compute the metric complexity integral and global scaling factor beta*
   to meet a target node budget and apply it to the metric,
 - write a MeshIO v4 background .msh containing metric components as point_data and
   register it with Gmsh using gmsh.model.mesh.setBackgroundMesh(path),
 - generate a 2D surface mesh in Gmsh, extract triangles and nodes, and lift node z
   using a provided high-order topography approximant (ho.query_at),
 - visualization helpers to inspect background metrics and final mesh (via PVVisualizer).

Key features
- Structured sampling (fast, deterministic) and Delaunay path for smaller problems.
- Two adaptive sampling modes:
  - gradient-driven quadtree-like adaptive refinement (scalar-gradient based),
  - anisotropic metric-aligned adaptive refinement:
    - axis-aligned binary splits along coordinate midlines (fast, robust),
    - oblique splits using shapely to cut rectangles with a line aligned to the metric
      principal direction (better alignment, with shapely and Delaunay fallback).
- Per-point metric rescaling to match zonal scalar size h(x) while preserving
  anisotropy directions (tensor strategy) or using isotropic fallback (simple strategy).
- Integration with metric_complexity to compute beta* and scale metric globally.

Usage outline
--------------
1. Instantiate with a high-order approximant (ho) and a metric_sampler callable:
    mesher = ZoneTensorMesher(ho, metric_sampler, bbox, verbosity=1)
2. Call generate(...) with desired options (nx, ny, inner_poly, center, outer_radius, hmin/hmax, etc.)
   The method will write a temporary background .msh, call gmsh to generate a surface mesh and
   return (nodes3d, tri_idx) arrays. Call mesher.finalize() after finishing to clean up gmsh.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Sequence

import numpy as np

# third-party
try:
    import meshio
except Exception as exc:
    meshio = None
    _MESHIO_IMPORT_ERROR = exc
else:
    _MESHIO_IMPORT_ERROR = None

try:
    import gmsh
except Exception as exc:
    gmsh = None
    _GMSH_IMPORT_ERROR = exc
else:
    _GMSH_IMPORT_ERROR = None

# shapely optional (used for oblique splitting)
try:
    from shapely import ops as shapely_ops
    from shapely.geometry import LineString, Point, Polygon

    _HAS_SHAPELY = True
except Exception:
    Polygon = None
    Point = None
    LineString = None
    shapely_ops = None
    _HAS_SHAPELY = False

# optional scipy Delaunay fallback
try:
    from scipy.spatial import Delaunay

    _HAS_SCIPY_DELAUNAY = True
except Exception:
    Delaunay = None
    _HAS_SCIPY_DELAUNAY = False

# local helpers
from abl_mesh.visualize import PVVisualizer
from abl_mesh.zone_size import compute_zone_size_for_points

from . import delaunay_backends, metric_complexity


class ZoneTensorMesher:
    """Zone-aware background-metric mesher and Gmsh surface mesh generator.

    The ZoneTensorMesher orchestrates sampling, metric construction, optional global
    beta* scaling, writing a background .msh and invoking gmsh to generate the surface mesh.

    Args:
        ho: High-order approximant object exposing query_at((x,y)) -> (z, grad, hess).
        metric_sampler: Callable((x,y)) -> 2x2 numpy.ndarray metric (symmetric positive-definite).
        bbox: Tuple (xmin, xmax, ymin, ymax) defining sampling bbox.
        verbosity: Integer verbosity level (0 quiet, 1 info, 2 debug).
        gmsh_init: If True the constructor calls gmsh.initialize(); set False to defer gmsh initialization.

    Attributes:
        ho: the HO approximant.
        metric_sampler: the provided metric sampler callable.
        bbox: bounding box used for sampling.
        verbosity: verbosity level.
    """

    def __init__(
        self,
        ho,
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        bbox: tuple[float, float, float, float],
        verbosity: int = 1,
        gmsh_init: bool = True,
    ):
        if meshio is None:
            raise RuntimeError(f"meshio is required: import error: {_MESHIO_IMPORT_ERROR}")
        if gmsh is None:
            raise RuntimeError(f"gmsh python bindings required: import error: {_GMSH_IMPORT_ERROR}")
        self.ho = ho
        self.metric_sampler = metric_sampler
        self.bbox = tuple(float(x) for x in bbox)
        self.verbosity = int(verbosity)
        self._pv = PVVisualizer(verbosity=max(0, verbosity - 1))

        if gmsh_init:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1 if self.verbosity else 0)

        if not hasattr(gmsh.model.mesh, "setBackgroundMesh"):
            raise RuntimeError(
                "This implementation requires gmsh >= 4.14 with setBackgroundMesh support"
            )

        try:
            self.gmsh_version = gmsh.__version__
        except Exception:
            self.gmsh_version = "unknown"

        if self.verbosity:
            print(f"[ZoneTensorMesher] initialized (gmsh {self.gmsh_version})")

    # ---------------------------------------------------------------------
    # Metric helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _sym_upper_from(M: np.ndarray) -> tuple[float, float, float]:
        """Return the upper-triangle symmetric components (m11, m12, m22)."""
        M = 0.5 * (M + M.T)
        return float(M[0, 0]), float(M[0, 1]), float(M[1, 1])

    @staticmethod
    def _metric_geometric_length(M: np.ndarray) -> float:
        """Representative geometric length for a 2x2 metric M: 1/(lambda1*lambda2)^(1/4)."""
        M = 0.5 * (M + M.T)
        eig = np.linalg.eigvalsh(M)
        eig = np.maximum(eig, 1e-18)
        lg = 1.0 / ((eig[0] * eig[1]) ** 0.25)
        return float(lg)

    @staticmethod
    def _scale_tensor_to_length(M: np.ndarray, target_h: float) -> np.ndarray:
        """Scale SPD tensor M so its geometric representative length equals target_h.

        If M is invalid, returns isotropic metric I / target_h^2.
        """
        M = 0.5 * (M + M.T)
        try:
            eig = np.linalg.eigvalsh(M)
            eig = np.maximum(eig, 1e-16)
            lg = 1.0 / ((eig[0] * eig[1]) ** 0.25)
        except Exception:
            return np.eye(2) / (target_h * target_h)
        if lg <= 0:
            return np.eye(2) / (target_h * target_h)
        beta = (lg / float(target_h)) ** 2
        Mnew = float(beta) * M
        Mnew = 0.5 * (Mnew + Mnew.T)
        return Mnew

    # ---------------------------------------------------------------------
    # Structured background building
    # ---------------------------------------------------------------------
    def _build_structured_triangles(self, xs: np.ndarray, ys: np.ndarray, keep_mask: np.ndarray):
        """Create triangles from regular grid nodes defined by xs (nx) and ys (ny).

        keep_mask is a flattened boolean array of length nx*ny or a (ny,nx) mask.
        Returns:
            pts2d: (K,2) array of kept point coordinates
            tris: (T,3) connectivity array (0-based indices into pts2d)
        """
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        nx = xs.size
        ny = ys.size
        mask = np.asarray(keep_mask)
        if mask.size == nx * ny:
            mask2d = mask.reshape(ny, nx)
        elif mask.shape == (ny, nx):
            mask2d = mask
        else:
            raise ValueError("keep_mask shape mismatch with xs/ys")
        idx_map = -np.ones((ny, nx), dtype=int)
        pts = []
        cnt = 0
        for j in range(ny):
            for i in range(nx):
                if mask2d[j, i]:
                    idx_map[j, i] = cnt
                    pts.append((xs[i], ys[j]))
                    cnt += 1
        if cnt == 0:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)
        tris = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                a = idx_map[j, i]
                b = idx_map[j, i + 1]
                c = idx_map[j + 1, i + 1]
                d = idx_map[j + 1, i]
                if a >= 0 and b >= 0 and c >= 0 and d >= 0:
                    tris.append((a, b, c))
                    tris.append((a, c, d))
        pts2d = np.array(pts, dtype=float)
        tris = np.array(tris, dtype=int) if len(tris) > 0 else np.zeros((0, 3), dtype=int)
        return pts2d, tris

    # ---------------------------------------------------------------------
    # Adaptive samplers
    # ---------------------------------------------------------------------
    def _adaptive_structured_sampling(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        outer_radius: float,
        center: tuple[float, float],
        metric_scalar_fn: Callable[[np.ndarray], np.ndarray],
        grad_threshold: float = 0.5,
        max_levels: int = 2,
    ):
        """Quadtree-like adaptive structured sampling driven by metric_scalar_fn.

        metric_scalar_fn must accept an (N,2) array of points and return an (N,) array of scalars.
        Cells with gradient magnitude above grad_threshold are subdivided up to max_levels.

        Returns:
            pts2d (K,2) sampled points, tris (T,3) triangulation indices (may use Delaunay fallback).
        """
        xmin = float(xmin)
        xmax = float(xmax)
        ymin = float(ymin)
        ymax = float(ymax)
        xs0 = np.linspace(xmin, xmax, nx)
        ys0 = np.linspace(ymin, ymax, ny)
        cell_list = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                x0 = xs0[i]
                x1 = xs0[i + 1]
                y0 = ys0[j]
                y1 = ys0[j + 1]
                cell_list.append((x0, x1, y0, y1, 0))

        for level in range(max_levels):
            candidates = [c for c in cell_list if c[4] == level]
            if not candidates:
                break
            # compute centers for candidate cells
            centers = np.array(
                [((c[0] + c[1]) * 0.5, (c[2] + c[3]) * 0.5) for c in candidates], dtype=float
            )
            vals_centers = metric_scalar_fn(centers)
            new_cells = []
            idx = 0
            refined_any = False
            for c in cell_list:
                x0, x1, y0, y1, lvl = c
                if lvl != level:
                    new_cells.append(c)
                    continue
                center_val = float(vals_centers[idx])
                idx += 1
                # estimate gradient magnitude inside cell using 3x3 stencil at cell-specific coords
                xs_s = np.array([x0, 0.5 * (x0 + x1), x1])
                ys_s = np.array([y0, 0.5 * (y0 + y1), y1])
                xx, yy = np.meshgrid(xs_s, ys_s)
                pts = np.column_stack([xx.ravel(), yy.ravel()])
                sval = metric_scalar_fn(pts).reshape(3, 3)
                gx = 0.5 * (sval[1, 2] - sval[1, 0]) / (x1 - x0 + 1e-12)
                gy = 0.5 * (sval[2, 1] - sval[0, 1]) / (y1 - y0 + 1e-12)
                gm = float(np.hypot(gx, gy))
                if gm > float(grad_threshold):
                    xm = 0.5 * (x0 + x1)
                    ym = 0.5 * (y0 + y1)
                    new_cells.extend(
                        [
                            (x0, xm, y0, ym, lvl + 1),
                            (xm, x1, y0, ym, lvl + 1),
                            (x0, xm, ym, y1, lvl + 1),
                            (xm, x1, ym, y1, lvl + 1),
                        ]
                    )
                    refined_any = True
                else:
                    new_cells.append(c)
            cell_list = new_cells
            if not refined_any:
                break

        # collect unique corners
        corners = {}
        for c in cell_list:
            x0, x1, y0, y1, lvl = c
            for x in (x0, x1):
                for y in (y0, y1):
                    key = (round(float(x), 12), round(float(y), 12))
                    corners[key] = (float(x), float(y))
        pts = np.array(list(corners.values()))
        if pts.size == 0:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)

        # keep points inside outer circle
        cx, cy = center
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        pts = pts[d <= outer_radius + 1e-12]
        if pts.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)

        # try structured reconstruction, else Delaunay
        xs_u = np.unique(np.round(pts[:, 0], 12))
        ys_u = np.unique(np.round(pts[:, 1], 12))
        if xs_u.size * ys_u.size == pts.shape[0]:
            xs_sorted = np.sort(xs_u)
            ys_sorted = np.sort(ys_u)
            mask2d = np.zeros((ys_sorted.size, xs_sorted.size), dtype=bool)
            idx_map = {}
            cnt = 0
            for j, y in enumerate(ys_sorted):
                for i, x in enumerate(xs_sorted):
                    if (round(float(x), 12), round(float(y), 12)) in corners and np.hypot(
                        x - cx, y - cy
                    ) <= outer_radius + 1e-12:
                        mask2d[j, i] = True
                        idx_map[(j, i)] = cnt
                        cnt += 1
            if cnt == 0:
                return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)
            kept_pts = []
            for j in range(ys_sorted.size):
                for i in range(xs_sorted.size):
                    if mask2d[j, i]:
                        kept_pts.append((xs_sorted[i], ys_sorted[j]))
            kept_pts = np.array(kept_pts, dtype=float)
            # build triangles
            tris = []
            for j in range(ys_sorted.size - 1):
                for i in range(xs_sorted.size - 1):
                    if (
                        mask2d[j, i]
                        and mask2d[j, i + 1]
                        and mask2d[j + 1, i + 1]
                        and mask2d[j + 1, i]
                    ):
                        a = idx_map[(j, i)]
                        b = idx_map[(j, i + 1)]
                        c_idx = idx_map[(j + 1, i + 1)]
                        d = idx_map[(j + 1, i)]
                        tris.append((a, b, c_idx))
                        tris.append((a, c_idx, d))
            tris = np.array(tris, dtype=int) if tris else np.zeros((0, 3), dtype=int)
            return kept_pts, tris
        else:
            if _HAS_SCIPY_DELAUNAY:
                tri = Delaunay(pts)
                return pts, tri.simplices
            else:
                # last resort: return points without triangles
                return pts, np.zeros((0, 3), dtype=int)

    def _anisotropic_axis_aligned_sampling(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        outer_radius: float,
        center: tuple[float, float],
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        anisotropy_ratio_threshold: float = 2.0,
        max_levels: int = 2,
    ):
        """Axis-aligned anisotropic binary splitting: split cells along x or y midline.

        Cells whose local anisotropy sqrt(lambda_max / lambda_min) exceed threshold
        are split along x (vertical cut) or y (horizontal cut) depending on principal vector.
        """
        xmin = float(xmin)
        xmax = float(xmax)
        ymin = float(ymin)
        ymax = float(ymax)
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        cells = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                x0 = xs[i]
                x1 = xs[i + 1]
                y0 = ys[j]
                y1 = ys[j + 1]
                cells.append((x0, x1, y0, y1, 0))

        for level in range(max_levels):
            new_cells = []
            refined_any = False
            for x0, x1, y0, y1, lvl in cells:
                if lvl != level:
                    new_cells.append((x0, x1, y0, y1, lvl))
                    continue
                xm = 0.5 * (x0 + x1)
                ym = 0.5 * (y0 + y1)
                if ((xm - center[0]) ** 2 + (ym - center[1]) ** 2) > (outer_radius + 1e-12) ** 2:
                    new_cells.append((x0, x1, y0, y1, lvl))
                    continue
                try:
                    M = np.asarray(metric_sampler((float(xm), float(ym))), dtype=float)
                    M = 0.5 * (M + M.T)
                    eigvals, eigvecs = np.linalg.eigh(M)
                    eigvals = np.maximum(eigvals, 1e-18)
                    ratio = float(np.sqrt(max(eigvals) / min(eigvals)))
                except Exception:
                    new_cells.append((x0, x1, y0, y1, lvl))
                    continue
                if ratio <= anisotropy_ratio_threshold:
                    new_cells.append((x0, x1, y0, y1, lvl))
                    continue
                idx_max = int(np.argmax(eigvals))
                v = eigvecs[:, idx_max]
                abs_vx = abs(float(v[0]))
                abs_vy = abs(float(v[1]))
                if abs_vx >= abs_vy:
                    xm_cut = 0.5 * (x0 + x1)
                    new_cells.append((x0, xm_cut, y0, y1, lvl + 1))
                    new_cells.append((xm_cut, x1, y0, y1, lvl + 1))
                else:
                    ym_cut = 0.5 * (y0 + y1)
                    new_cells.append((x0, x1, y0, ym_cut, lvl + 1))
                    new_cells.append((x0, x1, ym_cut, y1, lvl + 1))
                refined_any = True
            cells = new_cells
            if not refined_any:
                break

        # collect corners and continue like others
        corners = {}
        for c in cells:
            x0, x1, y0, y1, lvl = c
            for x in (x0, x1):
                for y in (y0, y1):
                    key = (round(float(x), 12), round(float(y), 12))
                    corners[key] = (float(x), float(y))
        pts = np.array(list(corners.values()))
        cx, cy = center
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        pts = pts[dists <= outer_radius + 1e-12]
        if pts.size == 0:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)
        xs_u = np.unique(np.round(pts[:, 0], 12))
        ys_u = np.unique(np.round(pts[:, 1], 12))
        if xs_u.size * ys_u.size == pts.shape[0]:
            # reconstruct structured
            xs_sorted = np.sort(xs_u)
            ys_sorted = np.sort(ys_u)
            mask2d = np.zeros((ys_sorted.size, xs_sorted.size), dtype=bool)
            idx_map = {}
            cnt = 0
            for j, y in enumerate(ys_sorted):
                for i, x in enumerate(xs_sorted):
                    key = (round(float(x), 12), round(float(y), 12))
                    if key in corners and np.hypot(x - cx, y - cy) <= outer_radius + 1e-12:
                        mask2d[j, i] = True
                        idx_map[(j, i)] = cnt
                        cnt += 1
            kept_pts = []
            for j in range(ys_sorted.size):
                for i in range(xs_sorted.size):
                    if mask2d[j, i]:
                        kept_pts.append((xs_sorted[i], ys_sorted[j]))
            # triangles
            tris = []
            for j in range(ys_sorted.size - 1):
                for i in range(xs_sorted.size - 1):
                    if (
                        mask2d[j, i]
                        and mask2d[j, i + 1]
                        and mask2d[j + 1, i + 1]
                        and mask2d[j + 1, i]
                    ):
                        a = idx_map[(j, i)]
                        b = idx_map[(j, i + 1)]
                        c_idx = idx_map[(j + 1, i + 1)]
                        d = idx_map[(j + 1, i)]
                        tris.append((a, b, c_idx))
                        tris.append((a, c_idx, d))
            tris = np.array(tris, dtype=int) if tris else np.zeros((0, 3), dtype=int)
            return np.array(kept_pts, dtype=float), tris
        else:
            if _HAS_SCIPY_DELAUNAY:
                tri = Delaunay(pts)
                return pts, tri.simplices
            else:
                return pts, np.zeros((0, 3), dtype=int)

    def _anisotropic_oblique_sampling(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
        nx: int,
        ny: int,
        outer_radius: float,
        center: tuple[float, float],
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        anisotropy_ratio_threshold: float = 2.0,
        max_levels: int = 2,
    ):
        """Oblique anisotropic sampling: split rectangles with cut line aligned to principal metric direction.

        Uses shapely.ops.split if available; otherwise falls back to axis-aligned anisotropic sampler.
        """
        if not _HAS_SHAPELY:
            # fallback to axis-aligned approach when shapely not present
            return self._anisotropic_axis_aligned_sampling(
                xmin,
                xmax,
                ymin,
                ymax,
                nx,
                ny,
                outer_radius,
                center,
                metric_sampler,
                anisotropy_ratio_threshold=anisotropy_ratio_threshold,
                max_levels=max_levels,
            )

        # initial grid
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        cells_polys = []  # will store shapely Polygons with level
        for j in range(ny - 1):
            for i in range(nx - 1):
                x0 = float(xs[i])
                x1 = float(xs[i + 1])
                y0 = float(ys[j])
                y1 = float(ys[j + 1])
                poly = Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
                cells_polys.append((poly, 0))

        for level in range(max_levels):
            new_cells = []
            refined_any = False
            for poly, lvl in cells_polys:
                if lvl != level:
                    new_cells.append((poly, lvl))
                    continue
                xm, ym = poly.representative_point().x, poly.representative_point().y
                # optionally skip outside domain
                if ((xm - center[0]) ** 2 + (ym - center[1]) ** 2) > (outer_radius + 1e-12) ** 2:
                    new_cells.append((poly, lvl))
                    continue
                try:
                    M = np.asarray(metric_sampler((float(xm), float(ym))), dtype=float)
                    M = 0.5 * (M + M.T)
                    eigvals, eigvecs = np.linalg.eigh(M)
                    eigvals = np.maximum(eigvals, 1e-18)
                    ratio = float(np.sqrt(max(eigvals) / min(eigvals)))
                except Exception:
                    new_cells.append((poly, lvl))
                    continue
                if ratio <= anisotropy_ratio_threshold:
                    new_cells.append((poly, lvl))
                    continue
                idx_max = int(np.argmax(eigvals))
                v = eigvecs[:, idx_max]
                vx = float(v[0])
                vy = float(v[1])
                # cut direction: line perpendicular to principal vector to produce child polygons elongated along principal
                nx_cut = -vy
                ny_cut = vx
                # extend as needed
                bound = poly.bounds  # (minx, miny, maxx, maxy)
                diag = np.hypot(bound[2] - bound[0], bound[3] - bound[1])
                L = diag * 3.0
                xm_c = float(xm)
                ym_c = float(ym)
                p1 = (xm_c - nx_cut * L, ym_c - ny_cut * L)
                p2 = (xm_c + nx_cut * L, ym_c + ny_cut * L)
                cut = LineString([p1, p2])
                try:
                    splitted = shapely_ops.split(poly, cut)
                    if len(splitted.geoms) >= 2:
                        # keep all pieces, mark with next level
                        for g in splitted.geoms:
                            # sometimes split produces tiny sliver polygons; skip negligible ones
                            if g.area <= 1e-15:
                                continue
                            new_cells.append((g, lvl + 1))
                        refined_any = True
                    else:
                        # fallback to axis-aligned splitting along longest side
                        minx, miny, maxx, maxy = poly.bounds
                        if (maxx - minx) >= (maxy - miny):
                            xm_cut = 0.5 * (minx + maxx)
                            left = Polygon(
                                [(minx, miny), (xm_cut, miny), (xm_cut, maxy), (minx, maxy)]
                            )
                            right = Polygon(
                                [(xm_cut, miny), (maxx, miny), (maxx, maxy), (xm_cut, maxy)]
                            )
                            new_cells.extend([(left, lvl + 1), (right, lvl + 1)])
                        else:
                            ym_cut = 0.5 * (miny + maxy)
                            low = Polygon(
                                [(minx, miny), (maxx, miny), (maxx, ym_cut), (minx, ym_cut)]
                            )
                            high = Polygon(
                                [(minx, ym_cut), (maxx, ym_cut), (maxx, maxy), (minx, maxy)]
                            )
                            new_cells.extend([(low, lvl + 1), (high, lvl + 1)])
                        refined_any = True
                except Exception:
                    # fallback axis-aligned split
                    minx, miny, maxx, maxy = poly.bounds
                    if (maxx - minx) >= (maxy - miny):
                        xm_cut = 0.5 * (minx + maxx)
                        left = Polygon([(minx, miny), (xm_cut, miny), (xm_cut, maxy), (minx, maxy)])
                        right = Polygon(
                            [(xm_cut, miny), (maxx, miny), (maxx, maxy), (xm_cut, maxy)]
                        )
                        new_cells.extend([(left, lvl + 1), (right, lvl + 1)])
                    else:
                        ym_cut = 0.5 * (miny + maxy)
                        low = Polygon([(minx, miny), (maxx, miny), (maxx, ym_cut), (minx, ym_cut)])
                        high = Polygon([(minx, ym_cut), (maxx, ym_cut), (maxx, maxy), (minx, maxy)])
                        new_cells.extend([(low, lvl + 1), (high, lvl + 1)])
                    refined_any = True
            cells_polys = new_cells
            if not refined_any:
                break

        # collect unique corner coordinates (including split intersections)
        corners = {}
        for poly, lvl in cells_polys:
            if not isinstance(poly, Polygon):
                continue
            for x, y in poly.exterior.coords:
                key = (round(float(x), 12), round(float(y), 12))
                corners[key] = (float(x), float(y))
        pts = np.array(list(corners.values()))
        if pts.size == 0:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)
        cx, cy = center
        dists = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        pts = pts[dists <= outer_radius + 1e-12]
        if pts.shape[0] == 0:
            return np.zeros((0, 2)), np.zeros((0, 3), dtype=int)

        # try structured reconstruction
        xs_u = np.unique(np.round(pts[:, 0], 12))
        ys_u = np.unique(np.round(pts[:, 1], 12))
        if xs_u.size * ys_u.size == pts.shape[0]:
            xs_sorted = np.sort(xs_u)
            ys_sorted = np.sort(ys_u)
            mask2d = np.zeros((ys_sorted.size, xs_sorted.size), dtype=bool)
            idx_map = {}
            cnt = 0
            for j, y in enumerate(ys_sorted):
                for i, x in enumerate(xs_sorted):
                    key = (round(float(x), 12), round(float(y), 12))
                    if key in corners and np.hypot(x - cx, y - cy) <= outer_radius + 1e-12:
                        mask2d[j, i] = True
                        idx_map[(j, i)] = cnt
                        cnt += 1
            kept_pts = []
            for j in range(ys_sorted.size):
                for i in range(xs_sorted.size):
                    if mask2d[j, i]:
                        kept_pts.append((xs_sorted[i], ys_sorted[j]))
            tris = []
            for j in range(ys_sorted.size - 1):
                for i in range(xs_sorted.size - 1):
                    if (
                        mask2d[j, i]
                        and mask2d[j, i + 1]
                        and mask2d[j + 1, i + 1]
                        and mask2d[j + 1, i]
                    ):
                        a = idx_map[(j, i)]
                        b = idx_map[(j, i + 1)]
                        c_idx = idx_map[(j + 1, i + 1)]
                        d = idx_map[(j + 1, i)]
                        tris.append((a, b, c_idx))
                        tris.append((a, c_idx, d))
            if len(kept_pts) == 0 or len(tris) == 0:
                if _HAS_SCIPY_DELAUNAY:
                    tri = Delaunay(pts)
                    return pts, tri.simplices
                else:
                    return pts, np.zeros((0, 3), dtype=int)
            return np.array(kept_pts, dtype=float), np.array(tris, dtype=int)
        else:
            if _HAS_SCIPY_DELAUNAY:
                tri = Delaunay(pts)
                return pts, tri.simplices
            else:
                return pts, np.zeros((0, 3), dtype=int)

    # ---------------------------------------------------------------------
    # Core public method: generate
    # ---------------------------------------------------------------------
    def generate(
        self,
        nx: int,
        ny: int,
        inner_poly: Polygon,
        center: tuple[float, float],
        outer_radius: float,
        transition_width: float,
        hmin: float,
        hmax: float,
        min_size_ratio: float = 0.5,
        mesh_strategy: str = "tensor",
        bg_mesh_strategy: str = "structured",
        delaunay_engine: str = "auto",
        use_background_mesh: bool = True,
        polygon_boundary: Sequence[tuple[float, float]] | None = None,
        write_mesh: str | None = None,
        target_num_nodes: float | None = None,
        complexity_nx: int | None = None,
        complexity_ny: int | None = None,
        bg_adapt_gradient_threshold: float | None = None,
        bg_adapt_max_levels: int = 0,
        refinement_polygons: list[tuple[Polygon, float]] | None = None,
        bg_adapt_anisotropic: bool = False,
        bg_adapt_anisotropic_oblique: bool = False,
        anisotropy_ratio_threshold: float = 2.0,
    ):
        """Generate a background metric mesh, register it in Gmsh and create a surface mesh.

        Args:
            nx, ny: base structured sampling resolution along x and y.
            inner_poly: Shapely polygon for the inner zone (hmin region).
            center: (cx, cy) center for outer circular restriction.
            outer_radius: Outer circular radius (same units as bbox/DEM).
            transition_width: width of transition band between inner and outer zones.
            hmin, hmax: minimum and maximum scalar sizes.
            min_size_ratio: clamp factor, sizes below hmin*min_size_ratio are clamped up.
            mesh_strategy: 'tensor' (preserve anisotropy, rescale tensors) or 'simple' (isotropic fallback).
            bg_mesh_strategy: 'structured', 'delaunay', or 'auto'.
            delaunay_engine: forwarded to delaunay_backends when used.
            use_background_mesh: if True call gmsh.model.mesh.setBackgroundMesh.
            polygon_boundary: optional polygon boundary coordinates for GEO construction.
            write_mesh: optional path to write final gmsh mesh (.msh).
            target_num_nodes: optional global target to compute beta* and scale metric.
            complexity_nx, complexity_ny: resolution for complexity integration (defaults to nx, ny).
            bg_adapt_gradient_threshold: if provided, enable gradient-driven adaptive structured sampling.
            bg_adapt_max_levels: maximum subdivision levels for adaptive samplers.
            refinement_polygons: optional list of (Polygon, local_hmin) to override hmin locally.
            bg_adapt_anisotropic: if True enable anisotropic axis-aligned adaptive splitting.
            bg_adapt_anisotropic_oblique: if True enable oblique anisotropic splitting (requires shapely).
            anisotropy_ratio_threshold: threshold (sqrt(lambda_max/lambda_min)) to trigger anisotropic split.

        Returns:
            nodes3d: (N,3) array of node coordinates after lifting z via ho.query_at.
            tri_idx: (M,3) array of triangle indices (0-based).

        Raises:
            RuntimeError on Gmsh failures or inability to produce triangles.
        """
        mesh_strategy = str(mesh_strategy)
        if mesh_strategy not in ("tensor", "simple"):
            raise ValueError("mesh_strategy must be 'tensor' or 'simple'")

        xmin, xmax, ymin, ymax = self.bbox
        if polygon_boundary is None:
            polygon_boundary = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

        # decide sampling strategy
        used_structured = bg_mesh_strategy == "structured"
        if bg_mesh_strategy == "auto":
            # try Delaunay quick check
            try:
                _ = delaunay_backends.delaunay_triangulation(
                    np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]), engine="auto", verbosity=0
                )
                used_structured = False
            except Exception:
                used_structured = True

        # pick adaptive sampler if requested
        sample_pts = None
        tri = np.zeros((0, 3), dtype=int)

        if bg_adapt_anisotropic_oblique:
            if self.verbosity:
                print("[ZoneTensorMesher] using oblique anisotropic adaptive sampling")
            sample_pts, tri = self._anisotropic_oblique_sampling(
                xmin,
                xmax,
                ymin,
                ymax,
                nx,
                ny,
                outer_radius,
                center,
                self.metric_sampler,
                anisotropy_ratio_threshold=anisotropy_ratio_threshold,
                max_levels=bg_adapt_max_levels,
            )
        elif bg_adapt_anisotropic:
            if self.verbosity:
                print("[ZoneTensorMesher] using axis-aligned anisotropic adaptive sampling")
            sample_pts, tri = self._anisotropic_axis_aligned_sampling(
                xmin,
                xmax,
                ymin,
                ymax,
                nx,
                ny,
                outer_radius,
                center,
                self.metric_sampler,
                anisotropy_ratio_threshold=anisotropy_ratio_threshold,
                max_levels=bg_adapt_max_levels,
            )
        elif bg_adapt_gradient_threshold and bg_adapt_gradient_threshold > 0.0:
            if self.verbosity:
                print("[ZoneTensorMesher] using gradient-driven adaptive structured sampling")

            # metric scalar function: representative length lg per sample
            def scalar_fn(pts):
                pts = np.asarray(pts, dtype=float)
                out = np.empty(pts.shape[0], dtype=float)
                for i, p in enumerate(pts):
                    try:
                        M = np.asarray(self.metric_sampler((float(p[0]), float(p[1]))), dtype=float)
                        out[i] = self._metric_geometric_length(M)
                    except Exception:
                        out[i] = np.nan
                return out

            sample_pts, tri = self._adaptive_structured_sampling(
                xmin,
                xmax,
                ymin,
                ymax,
                nx,
                ny,
                outer_radius,
                center,
                metric_scalar_fn=scalar_fn,
                grad_threshold=bg_adapt_gradient_threshold,
                max_levels=bg_adapt_max_levels,
            )
        elif used_structured:
            if self.verbosity:
                print("[ZoneTensorMesher] building regular structured background sampling")
            xs = np.linspace(xmin, xmax, nx)
            ys = np.linspace(ymin, ymax, ny)
            X, Y = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([X.ravel(), Y.ravel()])
            cx, cy = float(center[0]), float(center[1])
            dists = np.sqrt((pts_grid[:, 0] - cx) ** 2 + (pts_grid[:, 1] - cy) ** 2)
            inside_mask = dists <= float(outer_radius)
            keep_mask = inside_mask.reshape(ny, nx)
            sample_pts, tri = self._build_structured_triangles(xs, ys, keep_mask)
            if sample_pts.shape[0] == 0:
                raise RuntimeError(
                    "No structured background points found inside outer circle; increase nx/ny or outer_radius"
                )
        else:
            # Delaunay from structured grid points inside circle
            xs = np.linspace(xmin, xmax, nx)
            ys = np.linspace(ymin, ymax, ny)
            X, Y = np.meshgrid(xs, ys)
            pts_grid = np.column_stack([X.ravel(), Y.ravel()])
            cx, cy = float(center[0]), float(center[1])
            dists = np.sqrt((pts_grid[:, 0] - cx) ** 2 + (pts_grid[:, 1] - cy) ** 2)
            inside_mask = dists <= float(outer_radius)
            sample_pts = pts_grid[inside_mask]
            if sample_pts.shape[0] < 3:
                raise RuntimeError(
                    "Not enough sample points for Delaunay; increase nx/ny or outer_radius"
                )
            if self.verbosity:
                print(
                    f"[ZoneTensorMesher] triangulating {sample_pts.shape[0]} points using Delaunay (engine={delaunay_engine})"
                )
            tri = delaunay_backends.delaunay_triangulation(
                sample_pts, engine=delaunay_engine, verbosity=self.verbosity
            )
            if tri is None or len(tri) == 0:
                if self.verbosity:
                    print("[ZoneTensorMesher] Delaunay backend failed - falling back to structured")
                xs = np.linspace(xmin, xmax, nx)
                ys = np.linspace(ymin, ymax, ny)
                Xg, Yg = np.meshgrid(xs, ys)
                pts_grid = np.column_stack([Xg.ravel(), Yg.ravel()])
                dists = np.sqrt((pts_grid[:, 0] - cx) ** 2 + (pts_grid[:, 1] - cy) ** 2)
                inside_mask = dists <= float(outer_radius)
                keep_mask = inside_mask.reshape(ny, nx)
                sample_pts, tri = self._build_structured_triangles(xs, ys, keep_mask)

        if sample_pts is None or sample_pts.shape[0] == 0:
            raise RuntimeError("Background sampling produced no points")

        # compute per-sample sizes taking into account refinement polygons
        def compute_local_sizes(pts: np.ndarray) -> np.ndarray:
            pts = np.asarray(pts, dtype=float)
            sizes = compute_zone_size_for_points(pts, inner_poly, transition_width, hmin, hmax)
            if refinement_polygons:
                for poly, local_hmin in refinement_polygons:
                    mask = np.array(
                        [bool(poly.contains(Point(float(x), float(y)))) for x, y in pts], dtype=bool
                    )
                    sizes[mask] = np.minimum(sizes[mask], float(local_hmin))
            return sizes

        sizes = compute_local_sizes(sample_pts)
        min_allowed = float(hmin) * float(min_size_ratio)
        sizes = np.maximum(sizes, min_allowed)
        if self.verbosity >= 2:
            print(
                f"[ZoneTensorMesher] sizes range: {np.nanmin(sizes):.4g} .. {np.nanmax(sizes):.4g}"
            )

        # evaluate metrics and fill component arrays
        n = sample_pts.shape[0]
        m11 = np.empty(n, dtype=float)
        m12 = np.empty(n, dtype=float)
        m22 = np.empty(n, dtype=float)

        if mesh_strategy == "tensor":
            if self.verbosity:
                print(
                    "[ZoneTensorMesher] constructing tensor metric per sample (rescaling to local sizes)"
                )
            for i, (xy, h_local) in enumerate(zip(sample_pts, sizes, strict=False)):
                try:
                    M0 = np.asarray(self.metric_sampler((float(xy[0]), float(xy[1]))), dtype=float)
                    M0 = 0.5 * (M0 + M0.T)
                    eig = np.linalg.eigvalsh(M0)
                    if np.any(eig <= 0) or np.linalg.matrix_rank(M0) < 2:
                        Mscaled = np.eye(2) / (h_local * h_local)
                    else:
                        Mscaled = self._scale_tensor_to_length(M0, float(h_local))
                except Exception:
                    Mscaled = np.eye(2) / (h_local * h_local)
                a, b, c = self._sym_upper_from(Mscaled)
                m11[i], m12[i], m22[i] = a, b, c
                if (self.verbosity >= 2) and (i % 5000 == 0):
                    print(
                        f"[ZoneTensorMesher][debug] sample {i}/{n} size={h_local:.3g} m11={a:.3g}"
                    )
        else:
            if self.verbosity:
                print("[ZoneTensorMesher] constructing isotropic metrics (simple strategy)")
            val = 1.0 / (sizes**2)
            m11[:] = val
            m12[:] = 0.0
            m22[:] = val

        # compute global beta* if requested
        tmp_beta = 1.0
        if target_num_nodes is not None:
            cx_n = complexity_nx if complexity_nx is not None else nx
            cy_n = complexity_ny if complexity_ny is not None else ny
            if self.verbosity:
                print(
                    f"[ZoneTensorMesher] computing beta* for target {target_num_nodes} nodes using grid {cx_n}x{cy_n}"
                )
            try:
                beta_star, C, _, _, _, _, _ = metric_complexity.compute_beta_star_on_grid(
                    self.metric_sampler,
                    self.bbox,
                    cx_n,
                    cy_n,
                    float(target_num_nodes),
                    alpha=2.0,
                    mask=None,
                    n_jobs=1,
                    verbose=(self.verbosity > 1),
                )
            except Exception:
                # fallback: compute C from current (sample_pts,tri,m11,m12,m22)
                try:
                    C = metric_complexity.integrate_complexity_from_components_tris(
                        sample_pts, tri, m11, m12, m22, verbose=(self.verbosity > 1)
                    )
                    beta_star = metric_complexity.compute_beta_star(
                        float(target_num_nodes), C, alpha=2.0
                    )
                except Exception as e:
                    raise RuntimeError("Failed to compute complexity and beta*: " + str(e))
            tmp_beta = float(beta_star)
            if self.verbosity:
                print(f"[ZoneTensorMesher] computed beta* = {tmp_beta:.6e} (C={C:.6e})")
            m11 *= tmp_beta
            m12 *= tmp_beta
            m22 *= tmp_beta

        # write background .msh (meshio) with metric point_data
        pts3 = np.column_stack([sample_pts, np.zeros(len(sample_pts))])
        cells = [("triangle", tri)]
        point_data = {
            "metric_m11": m11.astype(float),
            "metric_m12": m12.astype(float),
            "metric_m22": m22.astype(float),
            "size_scalar_fallback": sizes.astype(float),
        }
        fh, tmp_bg = tempfile.mkstemp(suffix=".msh", prefix="zonetensor_bg_")
        os.close(fh)
        meshio.write_points_cells(tmp_bg, pts3, cells, point_data=point_data)
        if self.verbosity:
            print(f"[ZoneTensorMesher] wrote background .msh to {tmp_bg}")

        used_bg = False
        if use_background_mesh:
            try:
                gmsh.model.mesh.setBackgroundMesh(os.path.abspath(tmp_bg))
                used_bg = True
                if self.verbosity:
                    print("[ZoneTensorMesher] registered background mesh in gmsh")
            except Exception as e:
                used_bg = False
                if self.verbosity:
                    print(
                        "[ZoneTensorMesher] setBackgroundMesh failed; falling back to scalar point sizes. Error:",
                        e,
                    )

        # build GEO planar surface and generate mesh
        gmsh.model.add("zone_tensor_surface")
        boundary_point_tags = []
        for x, y in polygon_boundary:
            t = gmsh.model.geo.addPoint(float(x), float(y), 0.0, 1.0)
            boundary_point_tags.append(t)
        line_tags = []
        K = len(boundary_point_tags)
        for i in range(K):
            a = boundary_point_tags[i]
            b = boundary_point_tags[(i + 1) % K]
            line_tags.append(gmsh.model.geo.addLine(a, b))
        cl = gmsh.model.geo.addCurveLoop(line_tags)
        srf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()

        if not used_bg:
            # fallback: insert sample points in geo with scalar size as element size
            if self.verbosity:
                print("[ZoneTensorMesher] inserting sample points with scalar sizes (fallback)")
            for (x, y), s in zip(sample_pts, sizes, strict=False):
                gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(s))
            gmsh.model.geo.synchronize()

        if self.verbosity:
            print("[ZoneTensorMesher] generating 2D mesh in gmsh ...")
        gmsh.model.mesh.generate(2)

        if write_mesh:
            try:
                gmsh.write(write_mesh)
                if self.verbosity:
                    print(f"[ZoneTensorMesher] final mesh written to {write_mesh}")
            except Exception as e:
                if self.verbosity:
                    print("[ZoneTensorMesher] warning: cannot write final mesh:", e)

        # extract mesh and lift z using ho.query_at
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_list = []
        for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags, strict=False):
            if len(tags) == 0:
                continue
            num_nodes_per_elem = int(len(nodes) / len(tags))
            if num_nodes_per_elem == 3:
                tri_list.append(np.array(nodes, dtype=int).reshape(-1, 3))
        if len(tri_list) == 0:
            # try getElements without dim filter
            elem_types_all, elem_tags_all, elem_node_tags_all = gmsh.model.mesh.getElements()
            for nodes in elem_node_tags_all:
                if len(nodes) > 0 and (len(nodes) % 3 == 0):
                    tri_list.append(np.array(nodes, dtype=int).reshape(-1, 3))
        if len(tri_list) == 0:
            raise RuntimeError("Gmsh produced no triangular elements")

        tri_arr = np.vstack(tri_list).astype(int)
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        tri_idx = np.array([[tag_to_idx[t] for t in elem] for elem in tri_arr], dtype=int)

        nodes3d = coords.copy()
        for i, (x, y, z0) in enumerate(coords):
            try:
                z, *_ = self.ho.query_at((float(x), float(y)))
                nodes3d[i, 2] = float(z)
            except Exception:
                nodes3d[i, 2] = float(z0)

        # cleanup background file
        try:
            if os.path.exists(tmp_bg):
                os.remove(tmp_bg)
        except Exception:
            pass

        if self.verbosity:
            print(
                f"[ZoneTensorMesher] finished: nodes={len(nodes3d)}, tris={len(tri_idx)}, used_bg={used_bg}, beta_applied={tmp_beta}"
            )

        return nodes3d, tri_idx

    # ---------------------------------------------------------------------
    # Visualization helpers (wrap PVVisualizer)
    # ---------------------------------------------------------------------
    def visualize_background(
        self,
        bg_msh_path: str,
        scalar_name: str | None = None,
        show_vectors: bool = False,
        vector_scale: float = 1.0,
        downsample_vectors: int = 1000,
    ):
        """Visualize a background .msh file (metric scalar and principal directions)."""
        return self._pv.show_background_mesh_from_msh(
            bg_msh_path,
            scalar_name=scalar_name,
            show_vectors=show_vectors,
            vector_scale=vector_scale,
            downsample_vectors=downsample_vectors,
        )

    def visualize_mesh(
        self,
        nodes3d: np.ndarray,
        tri_idx: np.ndarray,
        scalars: np.ndarray | None = None,
        show_vertices: bool = False,
        vertex_size: float = 2.0,
        downsample_vertices: int = 2000,
        notebook: bool = False,
    ):
        """Visualize final triangular surface mesh."""
        return self._pv.show_mesh(
            nodes3d,
            tri_idx,
            scalars=scalars,
            show_vertices=show_vertices,
            vertex_size=vertex_size,
            downsample_vertices=downsample_vertices,
            notebook=notebook,
        )

    # ---------------------------------------------------------------------
    # Finalize gmsh
    # ---------------------------------------------------------------------
    def finalize(self):
        """Finalize gmsh session (safe to call multiple times)."""
        try:
            gmsh.finalize()
        except Exception:
            pass
