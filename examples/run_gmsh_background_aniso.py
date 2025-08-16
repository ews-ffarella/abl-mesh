"""
Example: run anisotropic meshing with the GmshBackgroundMetricMesher.

This script uses the synthetic hill from examples/run_pipeline.py to demonstrate
the background-mesh (tensor) approach. It requires gmsh Python bindings and the
abl_mesh.HighOrderApproximant + MetricField objects available in the package.

Usage:
  python examples/run_gmsh_background_aniso.py
"""

from abl_mesh.gmsh_background_metric_mesher import GmshBackgroundMetricMesher
from abl_mesh.metrics import MetricField
from abl_mesh.topography import HighOrderApproximant, Topography
from examples.run_pipeline import synthetic_hill


def main():
    # Synthetic hill (reuse helper)
    nodes2d, zs, tri = synthetic_hill(nx=120, ny=120)
    topo = Topography(nodes2d, zs, tri)
    ho = HighOrderApproximant(topo, degree=3)

    # Build a metric field using MetricField (simple constant h_scalar here)
    metric_field = MetricField(ho, h_scalar=20.0)

    bbox = (nodes2d[:, 0].min(), nodes2d[:, 0].max(), nodes2d[:, 1].min(), nodes2d[:, 1].max())

    mesher = GmshBackgroundMetricMesher(ho, metric_field.combined_metric, bbox, verbosity=2)

    nodes3d, tri_idx = mesher.generate_anisotropic_mesh(
        nx=200,
        ny=200,
        reduction="min",
        hmin=1.0,
        hmax=75.0,
        use_background_mesh=True,
        write_mesh="out_aniso.msh",
    )

    print("Generated mesh:", nodes3d.shape, "nodes,", tri_idx.shape, "triangles")

    # finalize gmsh
    mesher.finalize()


if __name__ == "__main__":
    main()
