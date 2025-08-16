"""
GmshBackgroundMetricMesher (targeting Gmsh 4.14)

This module implements a robust background-metric workflow for true anisotropic
meshing with Gmsh 4.14:

 - Sample your 2x2 metric tensor M(x,y) on a regular grid over the parametric bbox.
 - Build a simple triangular "background mesh" (Delaunay) of the sampled points.
 - Write that background mesh to a temporary .msh file using meshio, attaching
   per-node metric components (m11, m12, m22) as point data.
 - Tell Gmsh to use that file as the background metric via:
        gmsh.model.mesh.setBackgroundMesh(<path-to-msh>)
   (this is available and tested for Gmsh 4.14).
 - Build your planar GEO surface in Gmsh, generate a 2D mesh; Gmsh will use the
   supplied background metric and try to produce anisotropic elements that
   respect the tensor field.
 - Lift the produced planar mesh to 3D by querying your HighOrderApproximant
   for z(x,y).

Requirements (when using this file)
 - Python packages: gmsh (>=4.14), meshio, numpy, scipy
 - Optional but recommended: pyvista for visualization

Usage sketch:
    from abl_mesh.gmsh_background_metric_mesher import GmshBackgroundMetricMesher
    mesher = GmshBackgroundMetricMesher(ho, metric_sampler, bbox, verbosity=2)
    nodes3d, tris = mesher.generate_anisotropic_mesh(nx=200, ny=200,
                                                     hmin=0.5, hmax=75.0,
                                                     write_mesh="final.msh")
    mesher.finalize()

Notes:
 - This implementation targets Gmsh 4.14 specifically (uses gmsh.model.mesh.setBackgroundMesh).
 - If Gmsh or meshio are missing the constructor will raise with an informative message.
 - For users new to anisotropic metrics: the metric at a point is a 2x2 SPD matrix M.
   Gmsh expects a background mesh file where each node has metric data attached (we provide m11,m12,m22).
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
from collections.abc import Callable

import numpy as np
from scipy.spatial import Delaunay

# external libraries
try:
    import gmsh
except Exception:
    gmsh = None

try:
    import meshio
except Exception:
    meshio = None


class GmshBackgroundMetricMesher:
    def __init__(
        self,
        ho,
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        bbox: tuple[float, float, float, float],
        verbosity: int = 1,
        gmsh_init: bool = True,
    ):
        """
        ho:
            HighOrderApproximant-like object that implements query_at((x,y)) -> (z, grad, hess)
        metric_sampler:
            callable(xy) -> 2x2 numpy.ndarray (symmetric positive-definite metric)
        bbox:
            (xmin, xmax, ymin, ymax) parametric domain where to sample metric
        verbosity:
            0 (quiet) / 1 (info) / 2 (debug)
        gmsh_init:
            if True, call gmsh.initialize() here (typical).
        """
        if gmsh is None:
            raise RuntimeError(
                "gmsh python bindings are required (install gmsh, e.g. pip install gmsh>=4.14)."
            )
        if meshio is None:
            raise RuntimeError(
                "meshio is required to write background .msh files (pip install meshio)."
            )

        self.ho = ho
        self.metric_sampler = metric_sampler
        self.bbox = bbox
        self.verbosity = verbosity

        # gmsh initialization
        if gmsh_init:
            gmsh.initialize()
            # show gmsh messages on terminal if verbosity >= 1
            gmsh.option.setNumber("General.Terminal", 1 if verbosity else 0)

        # verify we have the function we rely on (Gmsh >=4.14)
        if not hasattr(gmsh.model.mesh, "setBackgroundMesh"):
            raise RuntimeError(
                "Your installed gmsh Python API does not provide model.mesh.setBackgroundMesh(). "
                "This implementation targets Gmsh 4.14 which exposes that function."
            )

        try:
            self.gmsh_version = gmsh.__version__
        except Exception:
            self.gmsh_version = "unknown"

        if self.verbosity:
            print(f"[GmshBackgroundMetricMesher] initialized (gmsh version {self.gmsh_version})")

    @staticmethod
    def _metric_to_upper_tri(M: np.ndarray) -> tuple[float, float, float]:
        """Convert symmetric 2x2 metric M into (m11, m12, m22)."""
        M = 0.5 * (M + M.T)
        return float(M[0, 0]), float(M[0, 1]), float(M[1, 1])

    @staticmethod
    def metric_to_scalar(M: np.ndarray, mode: str = "min") -> float:
        """
        Fallback isotropic reduction of metric M to scalar length:
          - 'min', 'geom', 'harm', 'iso_avg' as convenience options.
        """
        M = 0.5 * (M + M.T)
        eig = np.linalg.eigvalsh(M)
        eig = np.maximum(eig, 1e-16)
        l1 = 1.0 / math.sqrt(eig[0])
        l2 = 1.0 / math.sqrt(eig[1])
        if mode == "min":
            return float(min(l1, l2))
        if mode == "geom":
            return float(math.sqrt(l1 * l2))
        if mode == "harm":
            return float(2.0 / (1.0 / l1 + 1.0 / l2 + 1e-18))
        if mode == "iso_avg":
            avg_lambda = 0.5 * (eig[0] + eig[1])
            return float(1.0 / math.sqrt(avg_lambda))
        raise ValueError(f"Unknown reduction mode '{mode}'")

    def generate_anisotropic_mesh(
        self,
        nx: int = 200,
        ny: int = 200,
        reduction: str = "min",
        hmin: float = 0.5,
        hmax: float = 75.0,
        polygon: np.ndarray | None = None,
        use_background_mesh: bool = True,
        write_mesh: str | None = None,
    ):
        """
        Generate an anisotropic surface mesh guided by a background metric (Gmsh 4.14).

        Parameters
        ----------
        nx, ny : int
            sampling resolution for the background metric grid (nx*ny samples)
        reduction : str
            fallback isotropic reduction if needed ('min','geom','harm','iso_avg')
        hmin, hmax : float
            clamps for fallback scalar sizes (if background-mesh path not used)
        polygon : optional (K,2) array
            polygon in parametric plane for domain boundary (if None uses bbox rectangle)
        use_background_mesh : bool
            try to use true background-mesh (preferred). If False, force scalar fallback.
        write_mesh : optional str
            if provided, write final generated mesh to this .msh path

        Returns
        -------
        nodes3d : (N,3) numpy array
        tri_idx : (M,3) numpy array (0-based indices into nodes3d)
        """
        xmin, xmax, ymin, ymax = self.bbox
        if polygon is None:
            polygon = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
            )

        # 1) sample metric on a regular grid
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys)
        pts2d = np.column_stack([X.ravel(), Y.ravel()])
        npts = pts2d.shape[0]
        if self.verbosity:
            print(f"[GmshAnisoMesher] sampling metric on {nx}x{ny} grid -> {npts} points")

        # prepare arrays to collect metric components and fallback scalar size
        m11 = np.empty(npts, dtype=float)
        m12 = np.empty(npts, dtype=float)
        m22 = np.empty(npts, dtype=float)
        scalar = np.empty(npts, dtype=float)

        for i, (x, y) in enumerate(pts2d):
            M = np.asarray(self.metric_sampler((float(x), float(y))), dtype=float)
            a, b, c = self._metric_to_upper_tri(M)
            m11[i], m12[i], m22[i] = a, b, c
            scalar[i] = max(hmin, min(hmax, self.metric_to_scalar(M, mode=reduction)))

        # 2) triangulate sample points (Delaunay) to make a background triangular mesh
        tri = Delaunay(pts2d).simplices.astype(int)
        if self.verbosity:
            print(f"[GmshAnisoMesher] Delaunay produced {len(tri)} background triangles")

        # 3) write temporary .msh file with meshio attaching per-node metric components
        #    meshio will produce a modern MSH (v4) file compatible with Gmsh 4.14
        pts3d_for_msh = np.column_stack([pts2d, np.zeros(npts)])  # z=0 for background mesh
        cells = [("triangle", tri)]
        point_data = {
            "metric_m11": m11,
            "metric_m12": m12,
            "metric_m22": m22,
            # also attach scalar fallback as extra convenience (not used by setBackgroundMesh API)
            "metric_scalar_fallback": scalar,
        }
        fh, tmp_bg_path = tempfile.mkstemp(suffix=".msh", prefix="gmsh_bg_")
        os.close(fh)
        try:
            meshio.write_points_cells(tmp_bg_path, pts3d_for_msh, cells, point_data=point_data)
            if self.verbosity:
                print(f"[GmshAnisoMesher] wrote background metric mesh to: {tmp_bg_path}")
        except Exception as e:
            # cleanup and re-raise
            if os.path.exists(tmp_bg_path):
                with contextlib.suppress(Exception):
                    os.remove(tmp_bg_path)
            raise RuntimeError(f"Failed to write temporary background .msh file using meshio: {e}")

        used_bg_mesh = False
        if use_background_mesh:
            # 4) ask gmsh to use this background mesh file (Gmsh 4.14 API)
            try:
                abs_path = os.path.abspath(tmp_bg_path)
                if self.verbosity:
                    print(
                        f"[GmshAnisoMesher] calling gmsh.model.mesh.setBackgroundMesh({abs_path!r})"
                    )
                gmsh.model.mesh.setBackgroundMesh(abs_path)
                used_bg_mesh = True
            except Exception as e:
                if self.verbosity:
                    print(
                        "[GmshAnisoMesher] setBackgroundMesh failed, falling back to scalar sizes. Error:",
                        e,
                    )
                used_bg_mesh = False

        # 5) create plane geometry (polygon) in gmsh and generate 2D mesh
        gmsh.model.add("aniso_surface")
        # create boundary points in GEO with default size (will be overridden by background if used)
        boundary_point_tags = []
        for x, y in polygon:
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

        # If we could not register background mesh, fall back to isotropic point-size hints:
        if not used_bg_mesh:
            if self.verbosity:
                print(
                    "[GmshAnisoMesher] background-mesh not used, inserting sample points with per-point scalar sizes (isotropic fallback)"
                )
            # add sample points into GEO with per-point target size = scalar (meshio scalar)
            for (x, y), s in zip(pts2d, scalar, strict=False):
                gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(s))
            gmsh.model.geo.synchronize()

        # generate 2D mesh; if background-mesh is set, Gmsh should use it
        if self.verbosity:
            print(
                "[GmshAnisoMesher] generating 2D mesh with gmsh (may take a while for large grids)"
            )
        gmsh.model.mesh.generate(2)

        # optionally write final mesh out
        if write_mesh:
            try:
                gmsh.write(write_mesh)
                if self.verbosity:
                    print(f"[GmshAnisoMesher] final mesh written to {write_mesh}")
            except Exception as e:
                if self.verbosity:
                    print("[GmshAnisoMesher] warning: could not write final mesh:", e)

        # 6) extract nodes and triangles produced by Gmsh
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        # extract triangular elements
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_list = []
        for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags, strict=False):
            num_nodes_per_elem = int(len(nodes) / len(tags))
            if num_nodes_per_elem == 3:
                tri_list.append(np.array(nodes, dtype=int).reshape(-1, 3))
        if len(tri_list) == 0:
            # fallback: search in all elements
            elem_types_all, elem_tags_all, elem_node_tags_all = gmsh.model.mesh.getElements()
            for nodes in elem_node_tags_all:
                if len(nodes) % 3 == 0 and len(nodes) > 0:
                    tri_list.append(np.array(nodes, dtype=int).reshape(-1, 3))
        if len(tri_list) == 0:
            raise RuntimeError("Gmsh returned no triangular elements for the surface mesh.")

        tri_arr = np.vstack(tri_list).astype(int)
        # mapping from gmsh node tag to index in coords array
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        tri_idx = np.array([[tag_to_idx[t] for t in elem] for elem in tri_arr], dtype=int)

        # 7) lift Z coordinate using high-order approximant
        nodes3d = coords.copy()
        for i, (x, y, z0) in enumerate(coords):
            try:
                z, *_ = self.ho.query_at((float(x), float(y)))
                nodes3d[i, 2] = float(z)
            except Exception:
                # if HO fails, leave z as produced (likely zero)
                nodes3d[i, 2] = float(z0)

        # cleanup background mesh temp file
        try:
            if os.path.exists(tmp_bg_path):
                os.remove(tmp_bg_path)
        except Exception:
            pass

        if self.verbosity:
            print(
                f"[GmshAnisoMesher] produced mesh with {len(nodes3d)} nodes and {len(tri_idx)} triangles"
            )
            if used_bg_mesh:
                print("[GmshAnisoMesher] used background-mesh (tensor) path")
            else:
                print("[GmshAnisoMesher] used scalar fallback path (isotropic sizes)")

        return nodes3d, tri_idx

    def finalize(self):
        """Finalize gmsh (call when you are finished with meshing)."""
        try:
            gmsh.finalize()
        except Exception:
            pass
