"""
Example usage of ZoneGmshMesher.

Generates a simple synthetic hill (same helper as main examples), defines a small inner polygon
(farm) as a shapely polygon and calls ZoneGmshMesher to produce a surface mesh with the three zones:
 - inside polygon -> hmin
 - transition band outside polygon up to transition_width -> linear ramp to hmax
 - outside transition -> hmax (and outside the outer circle we do not sample)

This script is minimal; adjust parameters to your real topography and polygon.
"""

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from abl_mesh.gmsh_zone_mesher import ZoneGmshMesher
from abl_mesh.topography import HighOrderApproximant, Topography


# simple synthetic hill as before
def synthetic_hill(nx=80, ny=80):
    xs = np.linspace(-500, 500, nx)
    ys = np.linspace(-500, 500, ny)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt((X / 300.0) ** 2 + (Y / 200.0) ** 2)
    Z = 100.0 * np.exp(-(R**2)) + 5.0 * np.sin(2 * X / 200.0) * np.cos(2 * Y / 250.0)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    zs = Z.ravel()
    from scipy.spatial import Delaunay

    tri = Delaunay(nodes).simplices
    return nodes, zs, tri


def main():
    nodes2d, zs, tri = synthetic_hill(80, 80)
    topo = Topography(nodes2d, zs, tri)
    ho = HighOrderApproximant(topo, degree=3)

    # bounding box
    xmin, ymin = nodes2d.min(axis=0)
    xmax, ymax = nodes2d.max(axis=0)
    bbox = (xmin, xmax, ymin, ymax)

    # define inner polygon (shapely). Example: a rotated rectangle / polygon inside domain
    inner_coords = [(-120, -80), (-40, -60), (0, 40), (-100, 120), (-180, 80)]
    inner_poly = Polygon(inner_coords)

    # define outer circular domain center+radius and transition width
    center = (0.0, 0.0)
    outer_radius = 500.0
    transition_width = 80.0

    hmin = 10.0
    hmax = 75.0

    mesher = ZoneGmshMesher(ho, bbox, verbosity=2)
    nodes3d, tri_idx = mesher.generate(
        nx=220,
        ny=220,
        inner_poly=inner_poly,
        center=center,
        outer_radius=outer_radius,
        transition_width=transition_width,
        hmin=hmin,
        hmax=hmax,
        polygon_boundary=None,
        use_background_mesh=True,
        write_mesh="zone_surface.msh",
    )
    mesher.finalize()

    print("Generated mesh nodes:", nodes3d.shape, "triangles:", tri_idx.shape)

    # quick matplotlib view of planar node distribution & polygon
    plt.figure(figsize=(6, 6))
    plt.triplot(nodes3d[:, 0], nodes3d[:, 1], tri_idx, linewidth=0.3, color="gray")
    xpoly, ypoly = inner_poly.exterior.xy
    plt.plot(xpoly, ypoly, "r-", linewidth=2, label="inner polygon")
    circ = plt.Circle(
        center, outer_radius, color="k", fill=False, linestyle="--", label="outer circle"
    )
    plt.gca().add_patch(circ)
    plt.axis("equal")
    plt.legend()
    plt.title("Generated surface mesh (plan view)")
    plt.show()


if __name__ == "__main__":
    main()
