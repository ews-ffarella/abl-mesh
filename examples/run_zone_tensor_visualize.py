"""
Example driver that demonstrates:
 - zone + tensor meshing (ZoneTensorMesher)
 - visualization of the background metric and final mesh with pyvista (PVVisualizer)
 - using an optional faster Delaunay backend if available via delaunay_backends

Edit the parameters below and run:
    python examples/run_zone_tensor_visualize.py

This example uses the synthetic hill used in other examples and a synthetic tensor metric.
"""

import numpy as np
from shapely.geometry import Polygon
from abl_mesh.zone_tensor_mesher import ZoneTensorMesher
from abl_mesh.topography import Topography, HighOrderApproximant
from examples.run_pipeline import synthetic_hill

def synthetic_metric_sampler(xy):
    x, y = float(xy[0]), float(xy[1])
    angle = 0.0 if x < 0 else np.pi/2.0
    lam1 = 1.0 / (20.0**2)
    lam2 = 1.0 / (5.0**2)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    D = np.diag([lam1, lam2])
    M = R @ D @ R.T
    return M

def main():
    # synthetic topo
    nodes2d, zs, tri = synthetic_hill(80,80)
    topo = Topography(nodes2d, zs, tri)
    ho = HighOrderApproximant(topo, degree=3)

    xmin, ymin = nodes2d.min(axis=0)
    xmax, ymax = nodes2d.max(axis=0)
    bbox = (xmin, xmax, ymin, ymax)

    # zone settings
    inner_coords = [(-120,-80), (-40,-60), (0,40), (-100,120), (-180,80)]
    inner_poly = Polygon(inner_coords)
    center = (0.0, 0.0)
    outer_radius = 500.0
    transition_width = 80.0
    hmin = 10.0
    hmax = 75.0

    mesher = ZoneTensorMesher(ho, synthetic_metric_sampler, bbox, verbosity=2)

    nodes3d, tris = mesher.generate(
        nx=220, ny=220,
        inner_poly=inner_poly,
        center=center,
        outer_radius=outer_radius,
        transition_width=transition_width,
        hmin=hmin, hmax=hmax,
        min_size_ratio=0.5,
        mesh_strategy='tensor',          # try 'simple' for isotropic fallback
        delaunay_engine='auto',          # try 'scipy' or 'triangle' if you installed them
        use_background_mesh=True,
        write_mesh="zone_tensor_final.msh"
    )

    # visualize generated mesh (heavy meshes will be slow - downsample vertices)
    mesher.visualize_mesh(nodes3d, tris, show_vertices=True, downsample_vertices=1500)

if __name__ == "__main__":
    main()