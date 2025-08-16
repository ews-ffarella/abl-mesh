"""
abl_mesh.surface_mesher
-----------------------

Adaptive surface mesher driver.

This module implements Algorithm 1 (Surface mesh adaptation) from the paper.
It uses a planar mesher to generate initial triangle mesh (Triangle wrapper recommended)
and drives a refinement loop using metric lengths:
 - edges whose metric-length > sqrt(2) are flagged for refinement (paper reference)
 - refinement is driven by asking the underlying mesher for smaller allowed area / element size

Practical notes:
 - The code expects the `triangle` Python package (https://pypi.org/project/triangle/) by
   Jonathan Shewchuk to be present for robust planar remeshing.
 - If triangle is not available the code raises an informative error; replace with your mesher.

API:
 - SurfaceMesher(topo, metric_field, hmax, hmin, verbosity)
 - mesher.adapt() -> returns nodes2d, tri (planar mesh)
 - optional: visualize via pyvista via utils.

Limitations:
 - The exact refinement operator (split particular triangles) is delegated to Triangle via size field.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from .metrics import metric_edge_length
from .topography import Topography

try:
    import triangle as tr

    TRI_AVAILABLE = True
except Exception:
    tr = None
    TRI_AVAILABLE = False


class SurfaceMesher:
    """
    Parameters
    ----------
    topo : Topography
        Input topography geometry (planar nodes + z + tri).
    metric_sampler : callable(xy) -> 2x2 metric matrix
        Function returning the combined metric M(x,y) used to measure edges.
    hmax : float
        Maximum element size (used for initial mesh).
    hmin : float
        Minimum element size (stop threshold).
    verbosity : int
        Verbosity level.
    """

    def __init__(
        self,
        topo: Topography,
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        hmax: float,
        hmin: float,
        verbosity: int = 1,
    ):
        self.topo = topo
        self.metric_sampler = metric_sampler
        self.hmax = float(hmax)
        self.hmin = float(hmin)
        self.verbosity = verbosity

        if not TRI_AVAILABLE:
            raise RuntimeError(
                "Triangle Python wrapper is required for surface meshing. "
                "Install 'triangle' (pip install triangle) or provide an alternative mesher."
            )

    def _initial_planar_mesh(self):
        """
        Create an initial uniform planar mesh in bounding box using triangle via 'a' max area option.
        We generate a convex polygon bounding box with the same parametric domain as input nodes
        and ask triangle to mesh it. For better results you can use the exact polygon boundary.
        """
        xy = self.topo.nodes2d
        xmin, ymin = xy.min(axis=0)
        xmax, ymax = xy.max(axis=0)
        # create a rectangle polygon with 4 vertices
        pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
        poly = {"vertices": pts, "segments": np.array([[0, 1], [1, 2], [2, 3], [3, 0]])}
        # area corresponding to hmax ~ equilateral triangle area ~ sqrt(3)/4 * h^2
        target_area = 0.65 * (math.sqrt(3) / 4.0) * (self.hmax**2)
        if self.verbosity:
            print(f"[surface] initial mesh target area ~ {target_area:.4g}")
        tri_in = {"vertices": pts, "segments": poly["segments"]}
        t = tr.triangulate(tri_in, f"pqa{target_area}")
        nodes = np.asarray(t["vertices"])
        tri = np.asarray(t["triangles"], dtype=int)
        return nodes, tri

    def _edge_metric_length(self, nodes: np.ndarray, edge: tuple[int, int]) -> float:
        p0 = tuple(nodes[edge[0]])
        p1 = tuple(nodes[edge[1]])
        return metric_edge_length(self.metric_sampler, p0, p1)

    def find_edges_to_refine(self, nodes: np.ndarray, tri: np.ndarray) -> list[tuple[int, int]]:
        """
        Identify edges with metric length > sqrt(2) (paper criterion with safety factor).
        Returns list of unique undirected edges (i,j) with i<j.
        """
        edges_set = set()
        for t in tri:
            edges_set.add(tuple(sorted((t[0], t[1]))))
            edges_set.add(tuple(sorted((t[1], t[2]))))
            edges_set.add(tuple(sorted((t[2], t[0]))))
        to_refine = []
        threshold = math.sqrt(2.0)
        for e in edges_set:
            L = self._edge_metric_length(nodes, e)
            if threshold < L:
                to_refine.append(e)
        if self.verbosity:
            print(f"[surface] edges to refine: {len(to_refine)} / {len(edges_set)}")
        return to_refine

    def adapt(self, max_iters: int = 12) -> tuple[np.ndarray, np.ndarray]:
        """
        Main adaptive loop (Algorithm 1 in paper simplified with Triangle driver).
        At each iteration:
         - compute metric-lengths and identify edges to refine
         - rebuild mesh with reduced target area on elements touching edges flagged
        Implementation detail:
         - We approximate per-element target area using a scalar representative size from metric:
           use smallest eigenvalue lambda_min of metric -> representative h = 1/sqrt(lambda_min)
         - We build a marker (per-vertex) size and pass to triangle via 'a' option is not per-vertex,
           so as practical approach we generate a new area target as average over domain but biased
           by flagged triangles.
        """
        # Initialize with coarse mesh
        nodes, tri = self._initial_planar_mesh()

        for it in range(max_iters):
            if self.verbosity:
                print(
                    f"[surface] adapt iter {it + 1}/{max_iters}, nodes={len(nodes)}, tri={len(tri)}"
                )
            # find edges to refine
            to_refine = self.find_edges_to_refine(nodes, tri)
            if not to_refine:
                if self.verbosity:
                    print("[surface] no edges to refine -> adapt finished")
                break
            # compute per-node size estimate s_i from metric eigenvalues at node
            sizes = np.zeros(len(nodes))
            for i, p in enumerate(nodes):
                M = self.metric_sampler((p[0], p[1]))
                eig = np.linalg.eigvalsh(M)
                eig_min = max(eig.min(), 1e-16)
                sizes[i] = 1.0 / np.sqrt(eig_min)
            # cap sizes between hmin and hmax
            sizes = np.clip(sizes, self.hmin, self.hmax)
            # Instruct triangle to refine: we form a list of points and per-vertex area (via 'a' we only have global area)
            # Practical compromise: compute a global target area proportional to mean size but reduced where many refinements requested
            mean_size = sizes.mean()
            # if many edges flagged reduce area (i.e., ask for smaller triangles)
            frac_flagged = len(to_refine) / (len(tri) * 3.0)
            # map size -> area (equilateral triangle approx)
            new_target_size = max(self.hmin, mean_size * (1.0 - 0.7 * frac_flagged))
            new_area = 0.65 * (math.sqrt(3) / 4.0) * (new_target_size**2)
            if self.verbosity:
                print(
                    f"[surface] adapt: mean_size={mean_size:.3g}, flagged_frac={frac_flagged:.3g}, new_size={new_target_size:.3g}"
                )
            # Remesh whole domain with smaller area to allow refinement in flagged areas
            # Using bounding rectangle approach again, but it would be better to pass domain polygon and holes.
            xmin, ymin = nodes[:, 0].min(), nodes[:, 1].min()
            xmax, ymax = nodes[:, 0].max(), nodes[:, 1].max()
            pts = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
            tri_in = {"vertices": pts, "segments": np.array([[0, 1], [1, 2], [2, 3], [3, 0]])}
            t = tr.triangulate(tri_in, f"pqa{new_area}")
            nodes = np.asarray(t["vertices"])
            tri = np.asarray(t["triangles"], dtype=int)
            # stop if target_size is already at minimal size
            if new_target_size <= self.hmin * 1.01:
                if self.verbosity:
                    print("[surface] reached near hmin; stopping adaptation")
                break
        # Map final planar nodes back to topography surface by sampling z
        zs = np.zeros(len(nodes))
        for i, p in enumerate(nodes):
            z, _g, _h = self.ho.query_at((p[0], p[1])) if hasattr(self, "ho") else (0.0, None, None)
            zs[i] = z
        # Note: we return planar nodes and connectivity. The caller may want (nodes,z)
        return nodes, tri


# Note:
# This SurfaceMesher uses triangle as a remesher with area-based control. The paper uses a metric-driven adaptation
# (anisotropic). Mapping an anisotropic metric directly to Triangle is non-trivial. A production implementation would
# either use an anisotropic remesher or implement local edge split/coarsen operators directly. Here we implement the
# same high-level adaptation loop and provide accurate metric measurements for the adaptation decision.
