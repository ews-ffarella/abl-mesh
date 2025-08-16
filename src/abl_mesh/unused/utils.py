"""
abl_mesh.utils
--------------

Utility functions: simple I/O, visualizers (matplotlib + pyvista),
and small helpers used by examples.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyvista as pv

    PV_AVAILABLE = True
except Exception:
    pv = None
    PV_AVAILABLE = False


def plot_planar_mesh(nodes2d: np.ndarray, tri: np.ndarray, ax=None, title: str = "planar mesh"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    for t in tri:
        poly = np.vstack([nodes2d[t], nodes2d[t[0]]])
        ax.plot(poly[:, 0], poly[:, 1], "k-", linewidth=0.3)
    ax.set_title(title)
    ax.set_aspect("equal", "box")


def visualize_hybrid(
    nodes: np.ndarray, prisms: np.ndarray = None, tets: np.ndarray = None, scalars=None
):
    if not PV_AVAILABLE:
        raise RuntimeError("pyvista required for 3D visualization (pip install pyvista).")
    pl = pv.Plotter()
    pl.add_axes()
    if prisms is not None and len(prisms) > 0:
        # draw prism faces as triangular faces
        faces_pts = []
        for p in prisms:
            b0, b1, b2, t0, t1, t2 = p
            # bottom
            faces_pts.append([nodes[b0], nodes[b1], nodes[b2]])
            # top
            faces_pts.append([nodes[t0], nodes[t1], nodes[t2]])
        # convert to mesh of triangles for visualization
        tri_pts = np.vstack([np.vstack(f) for f in faces_pts])
        pl.add_mesh(pv.PolyData(tri_pts), color="lightgray", show_edges=True)
    if tets is not None and len(tets) > 0:
        # create UnstructuredGrid
        tet_cells = np.hstack([np.full((len(tets), 1), 4), tets]).astype(np.int64)
        grid = pv.UnstructuredGrid(tet_cells.flatten(), tet_type=pv.CellType.TETRA, points=nodes)
        pl.add_mesh(grid, show_edges=True, opacity=0.5)
    else:
        pl.add_points(nodes, point_size=2)
    pl.show()
