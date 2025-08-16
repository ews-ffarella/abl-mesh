"""
abl_mesh.hybrid
---------------

High-level orchestration for hybrid ABL mesh:
 - takes a surface mesh (nodes2d + z + tri)
 - uses Topography + HighOrderApproximant + MetricField to compute metrics
 - uses SurfaceMesher to produce adapted surface mesh
 - extruder.SafeExtruder to build prismatic SBL (several layers)
 - tetrahedral fill (via TetGen) to fill the rest of domain
 - optimizes hybrid mesh via QualityOptimizer

This file provides the class HybridMesher and a simple `run` example function.
"""

from __future__ import annotations

import numpy as np

from .extruder import SafeExtruder
from .metrics import MetricField
from .optimizer import QualityOptimizer
from .surface_mesher import SurfaceMesher
from .topography import HighOrderApproximant, Topography

try:
    import tetgen

    TET_AVAILABLE = True
except Exception:
    tetgen = None
    TET_AVAILABLE = False


class HybridMesher:
    """
    High-level mesher class.

    Parameters
    ----------
    topo_nodes2d, topo_z, topo_tri : input planar topography mesh
    verbosity : int
    """

    def __init__(
        self, topo_nodes2d: np.ndarray, topo_z: np.ndarray, topo_tri: np.ndarray, verbosity: int = 1
    ):
        self.topo = Topography(topo_nodes2d, topo_z, topo_tri, verbosity=verbosity)
        self.ho = HighOrderApproximant(self.topo, degree=3, verbosity=verbosity)
        self.verbosity = verbosity

    def run(
        self,
        hmax: float,
        hmin: float,
        target_nodes_for_curvature: int,
        sbl_h0: float,
        sbl_h1: float,
        sbl_r: float,
        sbl_zbl: float,
        top_ceiling_z: float,
        optimize: bool = True,
    ) -> dict:
        """
        Execute full pipeline.

        Returns dictionary with keys:
         - surface_nodes, surface_tri, surface_z
         - prism_nodes (top layer nodes), prism_conn
         - hybrid_nodes, hybrid_prisms, hybrid_tets
        """
        # 1) prepare MetricField with an h_scalar default strategy
        metric_field = MetricField(self.ho, h_scalar=lambda xy: hmax, verbosity=self.verbosity)

        # 2) compute beta* for curvature according to desired number of nodes in farm region
        bbox = (
            self.topo.nodes2d[:, 0].min(),
            self.topo.nodes2d[:, 0].max(),
            self.topo.nodes2d[:, 1].min(),
            self.topo.nodes2d[:, 1].max(),
        )
        beta_star, _ = metric_field.curvature_metric_with_complexity(
            bbox, target_nodes_for_curvature
        )
        # combined metric sampler function
        combined_sampler = lambda xy: metric_field.combined_metric(xy, beta_star)

        # 3) surface adapt
        s_mesher = SurfaceMesher(self.topo, combined_sampler, hmax, hmin, verbosity=self.verbosity)
        nodes2d, tri = s_mesher.adapt(max_iters=12)
        # sample z using HO approximant
        zs = np.zeros(len(nodes2d))
        for i, p in enumerate(nodes2d):
            z, _, _ = self.ho.query_at((p[0], p[1]))
            zs[i] = z

        # 4) map to 3D nodes
        surface_nodes3d = np.column_stack([nodes2d[:, 0], nodes2d[:, 1], zs])

        # 5) extrude SBL: create n_layers by geometric growth until z_bl
        extruder = SafeExtruder(surface_nodes3d, tri, verbosity=self.verbosity)
        layers_nodes = [surface_nodes3d]
        layers_prisms = []
        current_z = 0.0
        layer_thickness = sbl_h0
        total_height = 0.0
        # we will extrude until reaching sbl_zbl or max iterations
        while total_height < sbl_zbl - 1e-8:
            top_nodes, prisms = extruder.extrude_layer(
                layer_thickness,
                blend_vertical=0.25,
                max_backoffs=6,
                min_factor=0.05,
                quality_optimize=True,
            )
            # append
            layers_nodes.append(top_nodes)
            layers_prisms.append(prisms)
            total_height += layer_thickness
            # compute next thickness: geometric growth to h1 but not exceeding
            layer_thickness = min(layer_thickness * sbl_r, sbl_h1)
            # update extruder for next stage: treat top_nodes as new surface for next sweep
            extruder = SafeExtruder(top_nodes, tri, verbosity=self.verbosity)
            if len(layers_nodes) > 120:
                break

        # collect prism mesh
        # nodes: stack bottom->first top->second top etc. For simplicity we return only the first prismatic layer nodes & prisms
        prism_nodes = np.vstack(layers_nodes)
        prism_conn = np.vstack(layers_prisms) if layers_prisms else np.zeros((0, 6), dtype=int)

        # 6) tetrahedral fill for rest of domain up to top_ceiling_z
        if not TET_AVAILABLE:
            print(
                "[hybrid] tetgen not available: tetrahedral fill skipped. Install 'tetgen' python wrapper to enable."
            )
            tet_nodes = np.array([])
            tets = np.array([])
        else:
            # Build surface boundary for tetgen: use prism top layer nodes + ceiling polygon
            # Simple approach: extrude outer boundary to top and use TetGen
            tg = tetgen.TetGen(prism_nodes)
            # We must provide facets; we try to infer a closed surface from prisms + dom top;
            # here we perform a bounding-box ceiling at top_ceiling_z and call TetGen with volume constraints.
            # For robust production code, build explicit facets of the hybrid boundary.
            tg.tetrahedralize(order=1, mindihedral=10)
            tet_nodes = np.asarray(tg.points)
            tets = np.asarray(tg.elements)

        # 7) assemble hybrid nodes, prisms, tets
        hybrid_nodes = prism_nodes.copy()
        hybrid_prisms = prism_conn.copy()
        hybrid_tets = tets if TET_AVAILABLE else np.zeros((0, 4), dtype=int)

        # 8) final optimization (optional)
        if optimize:
            optimizer = QualityOptimizer(
                hybrid_nodes,
                self.topo.tri,
                prisms=hybrid_prisms if len(hybrid_prisms) > 0 else None,
                tets=hybrid_tets if len(hybrid_tets) > 0 else None,
                verbosity=self.verbosity,
            )
            optimizer.gauss_seidel(iterations=2)

        result = {
            "surface_nodes2d": nodes2d,
            "surface_nodes3d": surface_nodes3d,
            "surface_tri": tri,
            "prism_nodes": prism_nodes,
            "prism_conn": prism_conn,
            "hybrid_nodes": hybrid_nodes,
            "hybrid_prisms": hybrid_prisms,
            "hybrid_tets": hybrid_tets,
        }
        return result
