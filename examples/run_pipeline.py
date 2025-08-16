"""
Example driver for the ABL hybrid mesher pipeline.

Usage:
    python examples/run_pipeline.py --demo bolund

This script runs an end-to-end pipeline using the classes in abl_mesh.
It uses a synthetic hill if no dataset is provided.
"""

import numpy as np
import argparse
from abl_mesh.topography import Topography, HighOrderApproximant
from abl_mesh.hybrid import HybridMesher
from abl_mesh.utils import plot_planar_mesh, visualize_hybrid
import matplotlib.pyplot as plt

def synthetic_hill(nx=80, ny=80, scale=1.0):
    xs = np.linspace(-500,500,nx)
    ys = np.linspace(-500,500,ny)
    X,Y = np.meshgrid(xs,ys)
    R = np.sqrt((X/300.0)**2 + (Y/200.0)**2)
    Z = 100.0 * np.exp(-R**2) + 5.0*np.sin(2*X/200.0)*np.cos(2*Y/250.0)
    nodes = np.column_stack([X.ravel(), Y.ravel()])
    zs = Z.ravel()
    # generate a very simple triangulation via Delaunay (scipy)
    from scipy.spatial import Delaunay
    tri = Delaunay(nodes).simplices
    return nodes, zs, tri

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=['bolund','synthetic'], default='synthetic')
    args = parser.parse_args()

    if args.demo == 'synthetic':
        nodes2d, zs, tri = synthetic_hill(60,60)
    else:
        # Placeholder: user should replace with real dataset loader
        nodes2d, zs, tri = synthetic_hill(80,80)

    mesher = HybridMesher(nodes2d, zs, tri, verbosity=2)
    result = mesher.run(hmax=75.0, hmin=10.0, target_nodes_for_curvature=20000,
                        sbl_h0=1.0, sbl_h1=20.0, sbl_r=1.15, sbl_zbl=400.0,
                        top_ceiling_z=2000.0, optimize=True)
    # visualize planar surface mesh
    import matplotlib.pyplot as plt
    plot_planar_mesh(result['surface_nodes2d'], result['surface_tri'], title="Adapted surface mesh")
    plt.show()

    # Try a pyvista 3D visualization (if pyvista installed)
    try:
        visualize_hybrid(result['hybrid_nodes'], result['hybrid_prisms'], result['hybrid_tets'])
    except Exception as e:
        print("3D visualization skipped:", e)

if __name__ == "__main__":
    main()