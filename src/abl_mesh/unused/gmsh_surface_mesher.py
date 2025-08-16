"""
GmshSurfaceMesher

Use Gmsh Python bindings to generate a surface mesh guided by a size field
derived from an anisotropic metric.

Workflow:
 - sample the parametric domain at a set of points
 - at each sample evaluate the metric M(x) (2x2 SPD)
 - convert M(x) to a scalar local target size h(x) using a user-selected
   reduction (min-eigen, geometric mean, harmonic mean)
 - create a Gmsh planar surface (bounding polygon)
 - insert the sample points as GEO points with a per-point mesh size
 - generate a 2D triangular mesh on the plane
 - lift mesh node z-coordinates using the provided HighOrderApproximant (ho.query_at)
 - return mesh (nodes3d, tri) and optionally write .msh/.vtk

Requirements:
 - gmsh Python package (pip install gmsh) - tested with gmsh 4.x
 - numpy, scipy
 - optional: pyvista for 3D visualisation.

Usage:
  from abl_mesh.gmsh_surface_mesher import GmshSurfaceMesher
  mesher = GmshSurfaceMesher(ho, metric_sampler, bbox, verbosity=2)
  nodes3d, tri = mesher.generate(sample_density=2000, reduction='min', hmin=0.5, hmax=75.0)
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

# try gmsh import
try:
    import gmsh
except Exception:
    gmsh = None

# optional visualization
try:
    import pyvista as pv

    PV_AVAILABLE = True
except Exception:
    pv = None
    PV_AVAILABLE = False


class GmshSurfaceMesher:
    def __init__(
        self,
        ho,  # HighOrderApproximant-like object with method query_at((x,y)) -> (z, grad, hess)
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        bbox: tuple[float, float, float, float],
        verbosity: int = 1,
        gmsh_initialize: bool = True,
    ):
        """
        Parameters
        ----------
        ho : object
            High-order approximant that implements query_at((x,y)) -> (z, grad, hess)
        metric_sampler : callable
            function M = metric_sampler((x,y)) returning 2x2 positive definite metric matrix
        bbox : (xmin, xmax, ymin, ymax)
            sampling / geometry bounding box in parametric coordinates
        verbosity : int
            verbosity level
        gmsh_initialize: bool
            if True (default) gmsh.initialize() is called in constructor; otherwise the caller should initialize.
        """
        if gmsh is None:
            raise RuntimeError(
                "gmsh python bindings are required. Install 'gmsh' (pip install gmsh)."
            )

        self.ho = ho
        self.metric_sampler = metric_sampler
        self.bbox = bbox
        self.verbosity = verbosity

        if gmsh_initialize:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1 if verbosity else 0)

        # internal state
        self._geo_points = []
        self._geo_point_tags = []
        self._plane_surface_tag = None
        self._model_initialized = False

    # -------------------------
    # Metric -> scalar size reduction
    # -------------------------
    @staticmethod
    def metric_to_scalar(M: np.ndarray, mode: str = "min") -> float:
        """
        Convert a 2x2 metric M to a scalar target size h.

        modes:
         - 'min' : conservative minimum eigendirection length -> h = min(1/sqrt(lambda))
         - 'geom' : geometric mean of principal lengths -> h = sqrt(l1*l2)
         - 'harm' : harmonic mean: 2/(1/l1 + 1/l2)
         - 'iso_avg' : isotropic based on trace: h = 1 / sqrt(trace(M)/2)

        Where lambda are the eigenvalues of M and l = 1/sqrt(lambda) are lengths.
        """
        # ensure symmetric
        M = 0.5 * (M + M.T)
        eig = np.linalg.eigvalsh(M)
        # clamp eigenvalues to positive
        eig = np.maximum(eig, 1e-16)
        l1 = 1.0 / math.sqrt(eig[0])
        l2 = 1.0 / math.sqrt(eig[1])
        if mode == "min":
            return float(min(l1, l2))
        elif mode == "geom":
            return float(math.sqrt(l1 * l2))
        elif mode == "harm":
            return float(2.0 / (1.0 / l1 + 1.0 / l2 + 1e-18))
        elif mode == "iso_avg":
            avg_lambda = 0.5 * (eig[0] + eig[1])
            return float(1.0 / math.sqrt(avg_lambda))
        else:
            raise ValueError(f"Unknown reduction mode '{mode}'")

    # -------------------------
    # Sampling helper
    # -------------------------
    def sample_points_on_bbox(self, n_samples: int = 2000, strategy: str = "random"):
        """
        Return sample points (nx2) inside bbox for building the background size field.

        strategy:
         - 'random' : random uniform sampling
         - 'grid'   : approximate uniform grid with about n_samples points
         - 'nodes'  : expects that the caller will pass an explicit sample mesh (not used here)
        """
        xmin, xmax, ymin, ymax = self.bbox
        if strategy == "random":
            xs = np.random.uniform(xmin, xmax, size=n_samples)
            ys = np.random.uniform(ymin, ymax, size=n_samples)
            pts = np.column_stack([xs, ys])
            return pts
        elif strategy == "grid":
            # choose grid size approx sqrt(n_samples)
            m = int(np.sqrt(n_samples))
            xs = np.linspace(xmin, xmax, m)
            ys = np.linspace(ymin, ymax, m)
            X, Y = np.meshgrid(xs, ys)
            pts = np.column_stack([X.ravel(), Y.ravel()])
            return pts
        else:
            raise ValueError("Unknown sampling strategy")

    # -------------------------
    # Gmsh geometry creation
    # -------------------------
    def _create_planar_geo(self, polygon: np.ndarray | None = None):
        """
        Create a planar rectangular surface geometry in gmsh from bbox or user polygon.
        polygon: optional (K,2) polygon in parametric plane to be used as domain boundary.
        Returns model tags (points, curve_loop, plane_surface)
        """
        # clear previous model
        gmsh.model.add("surface_model")
        self._geo_point_tags = []
        if polygon is None:
            xmin, xmax, ymin, ymax = self.bbox
            polygon = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
            )
        # add geometry points (z=0)
        point_tags = []
        for i, (x, y) in enumerate(polygon):
            t = gmsh.model.geo.addPoint(
                float(x), float(y), 0.0, 1.0
            )  # default size, will be overridden
            point_tags.append(t)
        # add lines and loop
        line_tags = []
        K = len(point_tags)
        for i in range(K):
            a = point_tags[i]
            b = point_tags[(i + 1) % K]
            lt = gmsh.model.geo.addLine(a, b)
            line_tags.append(lt)
        cl = gmsh.model.geo.addCurveLoop(line_tags)
        srf = gmsh.model.geo.addPlaneSurface([cl])
        gmsh.model.geo.synchronize()
        self._plane_surface_tag = srf
        self._model_initialized = True
        if self.verbosity:
            print(f"[gmsh] created GEO plane surface with {len(point_tags)} boundary points")
        return point_tags, cl, srf

    # -------------------------
    # generate -> core
    # -------------------------
    def generate(
        self,
        sample_density: int = 2000,
        sample_strategy: str = "random",
        reduction_mode: str = "min",
        hmin: float = 0.5,
        hmax: float = 75.0,
        polygon: np.ndarray | None = None,
        write_mesh: str | None = None,
    ):
        """
        Main entry point.

        Parameters
        ----------
        sample_density : int
            number of sampling points for the background field
        sample_strategy : str
            'random' or 'grid' sampling inside bbox
        reduction_mode : str
            how to reduce 2x2 metric to scalar: 'min','geom','harm','iso_avg'
        hmin, hmax : float
            clamp resulting sizes
        polygon : optional numpy (K,2) array
            explicit boundary polygon to use instead of bbox
        write_mesh : optional str
            if provided, writes meshes to this .msh path

        Returns
        -------
        nodes3d : (N,3) ndarray
        tri : (M,3) ndarray
        """
        if gmsh is None:
            raise RuntimeError("gmsh python bindings are required.")

        # create geometry plane
        self._create_planar_geo(polygon=polygon)

        # sample points and compute scalar sizes
        pts = self.sample_points_on_bbox(n_samples=sample_density, strategy=sample_strategy)
        if self.verbosity:
            print(
                f"[gmsh] sampling {len(pts)} points for background size field (strategy={sample_strategy})"
            )

        sizes = np.zeros(len(pts), dtype=float)
        for i, (x, y) in enumerate(pts):
            M = self.metric_sampler((float(x), float(y)))
            h = self.metric_to_scalar(np.asarray(M, dtype=float), mode=reduction_mode)
            # clamp
            h = float(max(hmin, min(hmax, h)))
            sizes[i] = h

        # add sample points to GEO with assigned local mesh size
        sample_point_tags = []
        for (x, y), s in zip(pts, sizes, strict=False):
            tag = gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(s))
            sample_point_tags.append(tag)
        gmsh.model.geo.synchronize()
        if self.verbosity:
            print(
                f"[gmsh] created {len(sample_point_tags)} geo sample points with sizes in [{hmin},{hmax}]"
            )

        # Optionally: embed the sample points into the plane surface so Gmsh will consider them as constraints:
        # we create thin straight lines between boundary and points are not necessary; simply ensure they are
        # part of geometry (points exist) and Gmsh will use their target sizes during meshing.

        # Now generate a 2D mesh on the plane surface
        if self.verbosity:
            print("[gmsh] generating 2D mesh (this may take a bit)...")
        gmsh.model.mesh.generate(2)

        # optionally write intermediate mesh
        if write_mesh:
            gmsh.write(write_mesh)

        # get mesh nodes and elements (triangles)
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        # find triangle elements on the surface
        tria = []
        # Gmsh returns elements by element type: let's query triangles (type 2 for 2D triangles)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags, strict=False):
            # in Gmsh: type 2 = 3-node triangle; but this mapping depends on version; we'll treat any element with 3 nodes per element as triangle
            num_nodes_per_elem = int(len(nodes) / len(tags))
            if num_nodes_per_elem == 3:
                nodes_arr = np.array(nodes, dtype=int).reshape(-1, 3)
                tria.append(nodes_arr)
        if len(tria) == 0:
            # fallback: try all elements and pick those with 3 nodes
            elem_types_all, elem_tags_all, elem_node_tags_all = gmsh.model.mesh.getElements()
            for nodes in elem_node_tags_all:
                if len(nodes) == 0:
                    continue
                # guess tri if nodes multiple of 3 and typical
                if (len(nodes) % 3) == 0:
                    nodes_arr = np.array(nodes, dtype=int).reshape(-1, 3)
                    tria.append(nodes_arr)
        if len(tria) == 0:
            raise RuntimeError("No triangular elements found in generated mesh.")
        tri = np.vstack(tria).astype(int)

        # node tags -> mapping from tag to index in coords
        # gmsh node_tags correspond to rows in coords
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        # build triangle vertex indices as indices into coords array
        tri_idx = np.array([[tag_to_idx[t] for t in elem] for elem in tri], dtype=int)

        # Now lift the mesh nodes to 3D surface using ho.query_at
        nodes_xy = coords[:, :2]
        nodes3d = coords.copy()
        for i, (x, y, z0) in enumerate(coords):
            z, *_ = self.ho.query_at((float(x), float(y)))
            nodes3d[i, 2] = float(z)

        # update gmsh mesh nodes z coordinates as well (optional)
        # we will set mesh nodes back to gmsh to allow saving proper 3D mesh
        # gmsh expects node tags and coordinates flattened
        new_coords_flat = nodes3d.flatten().tolist()
        try:
            gmsh.model.mesh.setNodes(0, node_tags.tolist(), new_coords_flat)
        except Exception:
            # fallback to per-node set - slower
            for tag, xyz in zip(node_tags, nodes3d, strict=False):
                gmsh.model.mesh.setNode(tag, float(xyz[0]), float(xyz[1]), float(xyz[2]))

        # export final mesh if requested
        if write_mesh:
            gmsh.write(write_mesh)
            if self.verbosity:
                print(f"[gmsh] wrote mesh to {write_mesh!r}")

        # Return numpy arrays: nodes3d and tri_idx (triangles are indices into nodes3d)
        if self.verbosity:
            print(f"[gmsh] generated mesh with {len(nodes3d)} nodes and {len(tri_idx)} triangles")
        return nodes3d, tri_idx

    def finalize(self):
        """Call this when done to finalize gmsh (optional)."""
        try:
            gmsh.finalize()
        except Exception:
            pass

    # -------------------------
    # convenience visualizer
    # -------------------------
    def show_with_pyvista(
        self, nodes3d: np.ndarray, tri_idx: np.ndarray, scalars: np.ndarray | None = None
    ):
        if pv is None:
            raise RuntimeError("pyvista required for visualization (pip install pyvista).")
        mesh = pv.PolyData(
            nodes3d,
            np.hstack([np.full((tri_idx.shape[0], 1), 3), tri_idx]).astype(np.int64).ravel(),
        )
        pl = pv.Plotter()
        if scalars is not None:
            pl.add_mesh(mesh, scalars=scalars, show_edges=True)
        else:
            pl.add_mesh(mesh, color="lightgray", show_edges=True)
        pl.show()
