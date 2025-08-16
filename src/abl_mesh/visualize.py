"""
Visualization helpers using PyVista.

This module provides PVVisualizer, a small class with helper methods to visualize
triangular surface meshes and background metric .msh files (via meshio).
It supports showing vertex glyphs (downsampled or full), per-vertex scalars, and
metric principal direction arrows (downsampled).

Added:
- show_refinement_debug(polygons, principal_dirs, centers, scalars...) to overlay
  refinement cell polygons and metric principal directions for debugging adaptive samplers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

try:
    import pyvista as pv

    PV_AVAILABLE = True
except Exception:
    pv = None
    PV_AVAILABLE = False

try:
    import meshio

    MESHIO_AVAILABLE = True
except Exception:
    meshio = None
    MESHIO_AVAILABLE = False


class PVVisualizer:
    """Small helper wrapper for common PyVista visualizations.

    Args:
        verbosity: Verbosity level.
    """

    def __init__(self, verbosity: int = 1):
        if not PV_AVAILABLE:
            raise RuntimeError("pyvista is required for visualization (pip install pyvista)")
        self.verbosity = verbosity

    def _make_polydata(self, nodes: np.ndarray, tris: np.ndarray):
        faces = np.hstack([np.full((tris.shape[0], 1), 3, dtype=np.int64), tris.astype(np.int64)])
        faces = faces.flatten()
        mesh = pv.PolyData(nodes, faces)
        return mesh

    def show_mesh(
        self,
        nodes: np.ndarray,
        tris: np.ndarray,
        scalars: Sequence[float] | None = None,
        scalar_name: str = "scalar",
        show_vertices: bool = False,
        show_all_vertices: bool = False,
        vertex_size: float = 2.0,
        downsample_vertices: int = 2000,
        notebook: bool = False,
        show_edges: bool = False,
        cmap: str = "viridis",
    ):
        if not PV_AVAILABLE:
            raise RuntimeError("pyvista required for visualization")

        mesh = self._make_polydata(nodes, tris)
        if scalars is not None:
            mesh.point_data[scalar_name] = np.asarray(scalars, dtype=float)

        pl = pv.Plotter(notebook=notebook)
        if scalars is not None:
            pl.add_mesh(mesh, scalars=scalar_name, show_edges=show_edges, cmap=cmap)
        else:
            pl.add_mesh(mesh, color="lightgray", show_edges=show_edges)

        npts = nodes.shape[0]
        if show_all_vertices:
            if npts > 500000:
                print(
                    "[PVVisualizer] Warning: rendering all vertices (this may be slow or crash). Set show_all_vertices=False to downsample."
                )
            pl.add_points(
                nodes, color="black", point_size=vertex_size, render_points_as_spheres=True
            )
        elif show_vertices:
            if downsample_vertices <= 0 or downsample_vertices >= npts:
                idx = np.arange(npts)
            else:
                idx = np.linspace(0, npts - 1, downsample_vertices, dtype=int)
            pts = nodes[idx]
            pl.add_points(pts, color="black", point_size=vertex_size, render_points_as_spheres=True)

        pl.add_axes()
        pl.show()
        return pl

    def show_background_mesh_from_msh(
        self,
        msh_path: str,
        scalar_name: str | None = None,
        show_vectors: bool = False,
        vector_scale: float = 1.0,
        downsample_vectors: int = 1000,
    ):
        if not MESHIO_AVAILABLE:
            raise RuntimeError("meshio required to read background .msh")

        m = meshio.read(msh_path)
        points = m.points.copy()
        tri_cells = [c.data for c in m.cells if c.type in ("triangle", "triangles")]
        if len(tri_cells) == 0:
            tri_cells = [m.cells[0].data] if len(m.cells) > 0 else []
        if len(tri_cells) == 0:
            raise RuntimeError("No triangle cells found in background msh")
        tris = np.vstack(tri_cells)

        pd = m.point_data
        metric_present = all(k in pd for k in ("metric_m11", "metric_m12", "metric_m22"))
        scalar_present = scalar_name in pd if scalar_name else False

        rep_size = None
        if metric_present:
            m11 = np.asarray(pd["metric_m11"])
            m12 = np.asarray(pd["metric_m12"])
            m22 = np.asarray(pd["metric_m22"])
            sdet = np.sqrt(np.maximum(m11 * m22 - m12 * m12, 1e-18))
            rep_size = sdet
        elif scalar_present:
            rep_size = np.asarray(pd[scalar_name])
        else:
            for guess in ("size_scalar_fallback", "size", "h"):
                if guess in pd:
                    rep_size = np.asarray(pd[guess])
                    break

        mesh = self._make_polydata(points, tris)
        pl = pv.Plotter()
        if rep_size is not None:
            mesh.point_data["h_rep"] = rep_size
            pl.add_mesh(mesh, scalars="h_rep", cmap="viridis", show_edges=False)
        else:
            pl.add_mesh(mesh, color="lightgray", show_edges=True)

        if show_vectors and metric_present:
            npts = points.shape[0]
            if downsample_vectors >= npts:
                idx = np.arange(npts)
            else:
                idx = np.linspace(0, npts - 1, downsample_vectors, dtype=int)
            origins = points[idx]
            vecs = np.zeros_like(origins)
            for ii, i in enumerate(idx):
                M = np.array([[m11[i], m12[i]], [m12[i], m22[i]]])
                vals, vecs2 = np.linalg.eigh(M)
                principal = vecs2[:, np.argmax(vals)]
                vecs[ii, 0:2] = principal * float(1.0 / (np.sqrt(np.max(vals)) + 1e-12))
            vecs3 = np.column_stack([vecs[:, 0], vecs[:, 1], np.zeros(len(vecs))])
            pl.add_arrows(origins, vecs3, mag=vector_scale)

        pl.add_axes()
        pl.show()
        return pl

    def show_refinement_debug(
        self,
        polygons: list[Sequence[tuple[float, float]]] | None = None,
        centers: np.ndarray | None = None,
        principal_dirs: np.ndarray | None = None,
        levels: np.ndarray | None = None,
        title: str = "Refinement debug overlay",
        arrow_scale: float = 1.0,
    ):
        """Show polygons (wireframe), centers and principal direction arrows for debug.

        Args:
            polygons: list of polygons expressed as sequence of (x,y) tuples (closed or open).
            centers: (K,2) array of centers for principal directions.
            principal_dirs: (K,2) array of unit vectors for principal directions.
            levels: optional array of integer refinement levels for coloring.
            title: window title.
            arrow_scale: scale factor for arrows.
        """
        if not PV_AVAILABLE:
            raise RuntimeError("pyvista required for visualization")

        pl = pv.Plotter()
        if polygons:
            for idx, poly in enumerate(polygons):
                arr = np.asarray(poly, dtype=float)
                if arr.shape[0] < 2:
                    continue
                # Close polygon if necessary
                if not np.allclose(arr[0], arr[-1]):
                    arr = np.vstack([arr, arr[0]])
                # create polyline mesh
                line = pv.PolyData(arr)
                pl.add_mesh(line, color="black", line_width=2, style="wireframe")

        if centers is not None and principal_dirs is not None:
            centers = np.asarray(centers, dtype=float)
            dirs = np.asarray(principal_dirs, dtype=float)
            # build arrows
            arrows = []
            for c, d in zip(centers, dirs, strict=False):
                vec = np.array([d[0], d[1], 0.0], dtype=float) * arrow_scale
                arrow = pv.Arrow(
                    start=np.array([c[0], c[1], 0.0]) - vec * 0.5, direction=vec, scale="auto"
                )
                pl.add_mesh(arrow, color="red")
        pl.add_axes()
        pl.add_text(title, font_size=12)
        pl.show()
        return pl
