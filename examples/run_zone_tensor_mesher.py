"""
Example script demonstrating ZoneTensorMesher (Option A) with fallback 'simple'.

This demonstrates:
 - building a 3-zone scalar size field (inner polygon -> hmin, transition ramp, outer buffer -> hmax)
 - sampling an existing tensor metric (here we build a synthetic tensor field for demo)
 - rescaling the tensor per-zone so the anisotropy is preserved while the magnitude matches h(x)
 - instructing Gmsh 4.14 to use the background-metric .msh
 - generating the surface mesh and lifting z via a HighOrderApproximant-like object

Requirements:
    pip install gmsh==4.14.* meshio shapely numpy scipy pyvista
"""

import numpy as np
from shapely.geometry import Polygon

from abl_mesh.topography import HighOrderApproximant, Topography
from abl_mesh.zone_tensor_mesher import ZoneTensorMesher
from examples.run_pipeline import synthetic_hill

# --- synthetic hill for demo ---
nodes2d, zs, tri = synthetic_hill(nx=80, ny=80)
topo = Topography(nodes2d, zs, tri)
ho = HighOrderApproximant(topo, degree=3)


# --- synthetic tensor metric sampler for demo purposes ---
def synthetic_metric_sampler(xy):
    # Build a toy anisotropic metric that aligns with x-direction in left half and y-direction in right half
    x, y = float(xy[0]), float(xy[1])
    # base anisotropy: stronger in x on left side
    angle = 0.0 if x < 0 else np.pi / 2.0
    # eigenvalues (how strongly we compress in each principal direction)
    lam1 = 1.0 / (20.0**2)  # long direction length ~20
    lam2 = 1.0 / (5.0**2)  # short direction length ~5
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    D = np.diag([lam1, lam2])
    M = R @ D @ R.T
    return M


# --- zone definitions ---
xmin, ymin = nodes2d.min(axis=0)
xmax, ymax = nodes2d.max(axis=0)
bbox = (xmin, xmax, ymin, ymax)

# inner polygon (farm) example (within domain)
inner_coords = [(-120, -80), (-40, -60), (0, 40), (-100, 120), (-180, 80)]
inner_poly = Polygon(inner_coords)

center = (0.0, 0.0)
outer_radius = 500.0
transition_width = 80.0
hmin = 10.0
hmax = 75.0

mesher = ZoneTensorMesher(ho, synthetic_metric_sampler, bbox, verbosity=2)
nodes3d, tri_idx = mesher.generate(
    nx=220,
    ny=220,
    inner_poly=inner_poly,
    center=center,
    outer_radius=outer_radius,
    transition_width=transition_width,
    hmin=hmin,
    hmax=hmax,
    min_size_ratio=0.5,  # do not allow sizes < hmin*0.5
    mesh_strategy="tensor",  # 'tensor' uses rescaled tensors; use 'simple' to force isotropic
    write_mesh="zone_tensor_surface.msh",
)
mesher.finalize()

print("Done. Nodes:", nodes3d.shape, "Triangles:", tri_idx.shape)
