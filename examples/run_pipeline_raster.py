#!/usr/bin/env python3
"""
Driver script: full pipeline from raster -> HO approximant -> metric sampler ->
background-metric -> Gmsh surface mesh.

Enhancements:
- CLI flags to enable anisotropic oblique adaptive sampling:
    --bg-adapt-anisotropic-oblique : enable oblique (metric-aligned) refinement
- For debug: you can request the visualization overlay of refinement cells by using --visualize-debug
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys

try:
    import geopandas as gpd
except Exception:
    gpd = None

from shapely.geometry import Polygon

from abl_mesh.raster_topography import RasterHighOrderApproximant, RasterTopography
from abl_mesh.visualize import PVVisualizer
from abl_mesh.zone_tensor_mesher import ZoneTensorMesher


def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline driver: raster -> HO -> metric -> Gmsh surface"
    )
    p.add_argument("--raster", required=True, help="Path to input DEM raster (GeoTIFF).")
    p.add_argument(
        "--load-coeffs",
        default=None,
        help="Path to precomputed coefficients GeoTIFF to load before meshing.",
    )
    p.add_argument(
        "--export-coeffs",
        default=None,
        help="If set together with --do-precompute, export coefficients to this GeoTIFF path after precompute.",
    )
    p.add_argument(
        "--memmap-path",
        default=None,
        help="If set, precompute will write coefficients to this memmap file (out-of-core).",
    )
    p.add_argument(
        "--tile-size",
        default="256,256",
        help="Tile size for memmap precompute as 'rows,cols' (default '256,256').",
    )
    p.add_argument(
        "--hmin", type=float, required=True, help="Minimum size in center region (meters)."
    )
    p.add_argument("--hmax", type=float, required=True, help="Maximum size outside (meters).")
    p.add_argument("--bg-nx", type=int, default=400, help="Base background sampling nx.")
    p.add_argument("--bg-ny", type=int, default=400, help="Base background sampling ny.")
    p.add_argument(
        "--do-precompute",
        action="store_true",
        help="Precompute HO polynomial coefficients at raster cell centers.",
    )
    p.add_argument(
        "--precompute-njobs", type=int, default=-1, help="n_jobs for precompute_all_coeffs"
    )
    p.add_argument("--precompute-use-tqdm", action="store_true", help="Show tqdm during precompute")
    p.add_argument(
        "--export-coeffs-nodata",
        type=float,
        default=-9999.0,
        help="nodata sentinel when exporting GeoTIFF coefficients.",
    )
    p.add_argument("--visualize", action="store_true", help="Show pyvista visualizations.")
    p.add_argument(
        "--visualize-debug",
        action="store_true",
        help="Show refinement debug overlay after background generation (if available).",
    )
    p.add_argument("--verbosity", type=int, default=1, help="Verbosity (0..2).")

    # New anisotropic options
    p.add_argument(
        "--bg-adapt-anisotropic",
        action="store_true",
        help="Enable anisotropic (metric-aligned) adaptive background sampling.",
    )
    p.add_argument(
        "--bg-adapt-anisotropic-oblique",
        action="store_true",
        help="Enable oblique anisotropic splitting (metric-aligned cut lines).",
    )
    p.add_argument(
        "--anisotropy-threshold",
        type=float,
        default=2.0,
        help="Anisotropy eigenvalue ratio threshold to trigger anisotropic refinement (default 2.0).",
    )
    p.add_argument(
        "--bg-adapt-max-levels",
        type=int,
        default=0,
        help="Max adaptive subdivision levels for background (0 disables).",
    )

    return p.parse_args()


def configure_logging(verbosity: int):
    level = (
        logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    )
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    args = parse_args()
    configure_logging(args.verbosity)

    if not os.path.exists(args.raster):
        logging.error("Raster file not found: %s", args.raster)
        sys.exit(1)

    raster = RasterTopography(args.raster, verbosity=args.verbosity)
    ho = RasterHighOrderApproximant(raster, degree=3, verbosity=args.verbosity, precompute=False)

    # Optionally load precomputed coefficients from GeoTIFF
    if args.load_coeffs:
        if not os.path.exists(args.load_coeffs):
            logging.error("Requested coefficients file not found: %s", args.load_coeffs)
            sys.exit(1)
        ho.load_coeffs_geotiff(args.load_coeffs)
        logging.info("Loaded precomputed coefficients from %s", args.load_coeffs)

    # Optionally run precompute (either in-memory or memmap tiled)
    if args.do_precompute and (ho._precomputed is None):
        tile_rows, tile_cols = (int(s) for s in args.tile_size.split(","))
        logging.info(
            "Starting HO precompute (memmap=%s, tiles=%dx%d)",
            args.memmap_path,
            tile_rows,
            tile_cols,
        )
        ho.precompute_all_coeffs(
            n_jobs=args.precompute_njobs,
            use_tqdm=args.precompute_use_tqdm,
            memmap_path=(args.memmap_path if args.memmap_path else None),
            tile_size=(tile_rows, tile_cols),
        )
        logging.info(
            "Precompute finished. Valid coeffs: %d",
            int(ho._precompute_mask.sum()) if ho._precompute_mask is not None else 0,
        )

        if args.export_coeffs:
            logging.info("Exporting coefficients to GeoTIFF: %s", args.export_coeffs)
            ho.export_coeffs_geotiff(args.export_coeffs, nodata=args.export_coeffs_nodata)
            logging.info("Export completed.")

    # Build metric sampler from HO (curvature-based)
    def metric_sampler_from_ho(xy):
        _, _, hess = ho.query_at(xy)
        H = np.asarray(hess, dtype=float)
        H = 0.5 * (H + H.T)
        vals, vecs = np.linalg.eigh(H)
        vals = np.abs(vals)
        vals = np.maximum(vals, 1e-12)
        M = vecs @ np.diag(vals) @ vecs.T
        M += 1e-12 * np.eye(2)
        return M

    metric_sampler = metric_sampler_from_ho

    # center and outer defaults
    xmin, xmax, ymin, ymax = raster.bounds()
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))
    outer_radius = 0.5 * math.hypot(xmax - xmin, ymax - ymin)

    mesher = ZoneTensorMesher(
        ho, metric_sampler, bbox=raster.bounds(), verbosity=args.verbosity, gmsh_init=True
    )

    nodes3d, tri_idx = mesher.generate(
        nx=args.bg_nx,
        ny=args.bg_ny,
        inner_poly=Polygon(
            [
                (center[0] - 10, center[1] - 10),
                (center[0] + 10, center[1] - 10),
                (center[0] + 10, center[1] + 10),
                (center[0] - 10, center[1] + 10),
            ]
        ),
        center=center,
        outer_radius=outer_radius,
        transition_width=500.0,
        hmin=args.hmin,
        hmax=args.hmax,
        min_size_ratio=0.5,
        mesh_strategy="tensor",
        bg_mesh_strategy="auto",
        delaunay_engine="auto",
        use_background_mesh=True,
        write_mesh="final_surface.msh",
        target_num_nodes=None,
        complexity_nx=None,
        complexity_ny=None,
        bg_adapt_gradient_threshold=None,
        bg_adapt_max_levels=args.bg_adapt_max_levels,
        refinement_polygons=None,
        bg_adapt_anisotropic=args.bg_adapt_anisotropic,
        anisotropy_ratio_threshold=args.anisotropy_threshold,
    )

    logging.info("Generated final mesh nodes=%d tris=%d", nodes3d.shape[0], tri_idx.shape[0])

    if args.visualize:
        pv = PVVisualizer(verbosity=args.verbosity)
        pv.show_mesh(nodes3d, tri_idx, show_vertices=True, downsample_vertices=2000)

    mesher.finalize()


if __name__ == "__main__":
    main()
