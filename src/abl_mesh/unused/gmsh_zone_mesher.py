"""
Zone-guided Gmsh mesher (Gmsh 4.14)

Creates a background .msh containing an isotropic metric derived from a
3-zone scalar size field and instructs Gmsh 4.14 to use it (setBackgroundMesh).
The background metric written is isotropic: M = (1/h^2) * I.

Dependencies:
 - gmsh (>=4.14), meshio, shapely, numpy, scipy

Primary class:
 - ZoneGmshMesher(ho, bbox, verbosity=1)
     .generate(nx, ny, inner_poly, center, outer_radius, transition_width, hmin, hmax, write_mesh=None)
     .finalize()

Rationale:
 - For the zoning you requested (outer circle hmax, transition linear ramp, inner polygon hmin)
   a scalar size field is exactly correct. We encode it as an isotropic metric background
   for Gmsh (m11 = m22 = 1/h^2, m12 = 0).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from scipy.spatial import Delaunay

try:
    import gmsh
except Exception:
    gmsh = None

try:
    import meshio
except Exception:
    meshio = None

from shapely.geometry import Polygon

from .zone_size import compute_zone_size_for_points


class ZoneGmshMesher:
    def __init__(
        self,
        ho,
        bbox: tuple[float, float, float, float],
        verbosity: int = 1,
        gmsh_init: bool = True,
    ):
        """
        Parameters
        ----------
        ho: object implementing ho.query_at((x,y)) -> (z, grad, hess)
        bbox: (xmin, xmax, ymin, ymax)
        verbosity: 0/1/2
        gmsh_init: whether to call gmsh.initialize() here
        """
        if gmsh is None:
            raise RuntimeError("gmsh bindings required (pip install 'gmsh>=4.14')")
        if meshio is None:
            raise RuntimeError("meshio required (pip install meshio)")
        self.ho = ho
        self.bbox = bbox
        self.verbosity = verbosity
        if gmsh_init:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 1 if verbosity else 0)
        # verify API
        if not hasattr(gmsh.model.mesh, "setBackgroundMesh"):
            raise RuntimeError("gmsh.model.mesh.setBackgroundMesh not found; need gmsh >= 4.14")
        try:
            self.gmsh_version = gmsh.__version__
        except Exception:
            self.gmsh_version = "unknown"
        if self.verbosity:
            print(f"[ZoneGmshMesher] gmsh version {self.gmsh_version}")

    def _make_background_msh_from_sizes(
        self, pts2d: np.ndarray, tri: np.ndarray, sizes: np.ndarray
    ) -> str:
        """
        Create temporary .msh (v4) that contains the triangular background mesh (pts2d, tri)
        and attach per-node metric components: metric_m11, metric_m12, metric_m22.
        We use an isotropic metric M = (1/h^2) * I.
        Returns path to temporary .msh file.
        """
        n = len(pts2d)
        # metric components
        m11 = (1.0 / (sizes**2)).astype(float)
        m22 = m11.copy()
        m12 = np.zeros_like(m11, dtype=float)

        # meshio expects 3D points, supply z=0
        pts3 = np.column_stack([pts2d, np.zeros(n)])
        cells = [("triangle", tri)]
        point_data = {
            "metric_m11": m11,
            "metric_m12": m12,
            "metric_m22": m22,
            "size_scalar": sizes.astype(float),
        }
        fh, tmpname = tempfile.mkstemp(suffix=".msh", prefix="gmsh_zone_bg_")
        os.close(fh)
        meshio.write_points_cells(tmpname, pts3, cells, point_data=point_data)
        if self.verbosity:
            print(f"[ZoneGmshMesher] wrote background .msh -> {tmpname}")
        return tmpname

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
        polygon_boundary: np.ndarray | None = None,
        use_background_mesh: bool = True,
        write_mesh: str | None = None,
    ):
        """
        Generate surface mesh using the 3-zone sizing.

        Parameters:
         - nx,ny: sampling resolution for background mesh
         - inner_poly: shapely Polygon -> interior zone (hmin)
         - center: (cx,cy) outer circle center
         - outer_radius: R for outer circular limit (points beyond are treated as outside domain)
         - transition_width: width outside inner_poly where size goes linearly to hmax
         - hmin/hmax: sizes
         - polygon_boundary: optional boundary polygon (K,2) for the GEO; default uses bbox rectangle
         - use_background_mesh: True attempt to use setBackgroundMesh
         - write_mesh: if provided the final gmsh mesh path to write

        Returns:
         - nodes3d (N,3) and tri_idx (M,3) (tri_idx are 0-based indices)
        """
        xmin, xmax, ymin, ymax = self.bbox
        if polygon_boundary is None:
            polygon_boundary = np.array(
                [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float
            )

        # 1) sample metric grid
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        X, Y = np.meshgrid(xs, ys)
        pts2d = np.column_stack([X.ravel(), Y.ravel()])

        # But limit sampling to points inside outer circle for efficiency
        cx, cy = float(center[0]), float(center[1])
        R = float(outer_radius)
        rs = np.sqrt((pts2d[:, 0] - cx) ** 2 + (pts2d[:, 1] - cy) ** 2)
        inside_circle = rs <= R
        sample_pts = pts2d[inside_circle]

        # 2) compute scalar sizes via zone rule
        sizes = compute_zone_size_for_points(sample_pts, inner_poly, transition_width, hmin, hmax)

        # 3) triangulate sampling points to create background mesh
        if len(sample_pts) < 3:
            raise RuntimeError(
                "not enough sample points inside outer circle; increase nx/ny or outer_radius"
            )
        tri = Delaunay(sample_pts).simplices.astype(int)

        # 4) write background .msh with isotropic tensors (1/h^2 * I)
        bg_path = self._make_background_msh_from_sizes(sample_pts, tri, sizes)

        used_bg = False
        if use_background_mesh:
            try:
                gmsh.model.mesh.setBackgroundMesh(os.path.abspath(bg_path))
                used_bg = True
                if self.verbosity:
                    print("[ZoneGmshMesher] background mesh registered in gmsh")
            except Exception as e:
                used_bg = False
                if self.verbosity:
                    print(
                        "[ZoneGmshMesher] setBackgroundMesh failed; falling back to per-point sizes. Error:",
                        e,
                    )

        # 5) build GEO planar domain (polygon_boundary) and optionally insert sample points as GEO points (fallback)
        gmsh.model.add("zone_surface")
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
            # Insert the sample points with scalar size values to guide Gmsh
            # (mesh will be isotropic but guided by local sizes)
            for (x, y), s in zip(sample_pts, sizes, strict=False):
                gmsh.model.geo.addPoint(float(x), float(y), 0.0, float(s))
            gmsh.model.geo.synchronize()
            if self.verbosity:
                print(
                    "[ZoneGmshMesher] inserted sample points with scalar sizes into GEO (fallback)"
                )

        # 6) generate 2D mesh
        if self.verbosity:
            print("[ZoneGmshMesher] generating 2D mesh in gmsh...")
        gmsh.model.mesh.generate(2)

        if write_mesh:
            gmsh.write(write_mesh)
            if self.verbosity:
                print(f"[ZoneGmshMesher] wrote mesh to {write_mesh}")

        # 7) extract nodes + triangles and lift to 3D with ho.query_at
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords).reshape(-1, 3)
        elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        tri_list = []
        for etype, tags, nodes in zip(elem_types, elem_tags, elem_node_tags, strict=False):
            num_nodes_per_elem = int(len(nodes) / len(tags))
            if num_nodes_per_elem == 3:
                tri_list.append(np.array(nodes, dtype=int).reshape(-1, 3))
        if len(tri_list) == 0:
            raise RuntimeError("no triangles produced by gmsh")
        tri_arr = np.vstack(tri_list).astype(int)
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        tri_idx = np.array([[tag_to_idx[t] for t in elem] for elem in tri_arr], dtype=int)

        # lift nodes z
        nodes3d = coords.copy()
        for i, (x, y, z0) in enumerate(coords):
            try:
                z, *_ = self.ho.query_at((float(x), float(y)))
                nodes3d[i, 2] = float(z)
            except Exception:
                nodes3d[i, 2] = float(z0)

        # cleanup temporary background mesh file
        try:
            if os.path.exists(bg_path):
                os.remove(bg_path)
        except Exception:
            pass

        if self.verbosity:
            print(
                f"[ZoneGmshMesher] produced {len(nodes3d)} nodes, {len(tri_idx)} triangles (used_bg={used_bg})"
            )

        return nodes3d, tri_idx

    def finalize(self):
        try:
            gmsh.finalize()
        except Exception:
            pass
