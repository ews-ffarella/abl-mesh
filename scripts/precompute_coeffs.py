#!/usr/bin/env python3
"""
CLI helper to precompute HO polynomial coefficients for a DEM.

This script supports:
 - in-memory precompute (small rasters)
 - memmap-backed tiled precompute (out-of-core) using RasterHighOrderApproximant.precompute_all_coeffs(...)
 - optional export of the full coefficient stack to a multiband GeoTIFF

Typical usages:
  # memmap tiled precompute (recommended for large rasters)
  python scripts/precompute_coeffs.py --raster dem.tif --degree 3 --memmap-path /data/coeffs_p3.memmap \
      --tile-size 512,512 --n-jobs 8 --verbosity 1

  # precompute in memory and export to GeoTIFF
  python scripts/precompute_coeffs.py --raster dem.tif --do-precompute --export-coeffs coeffs_p3.tif

  # load existing memmap and just export GeoTIFF
  python scripts/precompute_coeffs.py --raster dem.tif --load-memmap /data/coeffs_p3.memmap --export-coeffs coeffs_p3.tif
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from abl_mesh.raster_topography import RasterHighOrderApproximant, RasterTopography


def parse_tile_size(s: str) -> tuple[int, int]:
    try:
        a, b = s.split(",")
        return int(a), int(b)
    except Exception:
        raise argparse.ArgumentTypeError("tile-size must be 'rows,cols' (e.g. 256,256)")


def build_parser():
    p = argparse.ArgumentParser(
        description="Precompute HO polynomial coefficients (memmap tiled option) and optionally export GeoTIFF"
    )
    p.add_argument("--raster", required=True, help="Input DEM (GeoTIFF)")
    p.add_argument(
        "--degree", type=int, default=3, help="Polynomial degree for HO approximant (default 3)"
    )
    p.add_argument(
        "--do-precompute",
        action="store_true",
        help="Run precompute now (if not set, load provided memmap or exit)",
    )
    p.add_argument(
        "--memmap-path",
        default=None,
        help="If provided run tiled memmap precompute writing to this path (NumPy memmap).",
    )
    p.add_argument(
        "--load-memmap",
        default=None,
        help="If provided, load existing memmap into the approximant (no precompute).",
    )
    p.add_argument(
        "--tile-size",
        type=parse_tile_size,
        default=(256, 256),
        help="Tile size for tiled precompute, as 'rows,cols'",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of joblib workers when computing fits inside a tile (multiprocessing). -1 uses all cores.",
    )
    p.add_argument("--use-tqdm", action="store_true", help="Show tqdm progress for the precompute.")
    p.add_argument(
        "--export-coeffs",
        default=None,
        help="If set, export coefficients to GeoTIFF after precompute/load.",
    )
    p.add_argument(
        "--export-nodata",
        type=float,
        default=-9999.0,
        help="Nodata sentinel to use when exporting GeoTIFF.",
    )
    p.add_argument("--verbosity", type=int, default=1, help="Verbosity (0 quiet, 1 info, 2 debug)")
    return p


def configure_logging(verbosity: int):
    lvl = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=lvl, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    p = build_parser()
    args = p.parse_args()
    configure_logging(args.verbosity)

    if not os.path.exists(args.raster):
        logging.error("Raster not found: %s", args.raster)
        sys.exit(1)

    rtopo = RasterTopography(args.raster, verbosity=args.verbosity)

    ho = RasterHighOrderApproximant(
        rtopo, degree=args.degree, verbosity=args.verbosity, precompute=False
    )

    # load memmap if requested (convenience)
    if args.load_memmap:
        if not os.path.exists(args.load_memmap):
            logging.error("Specified memmap not found: %s", args.load_memmap)
            sys.exit(1)
        # load mask if available (same path + .mask.npy)
        mask_path = args.load_memmap + ".mask.npy"
        ho.load_coeffs_memmap(args.load_memmap, mask_path if os.path.exists(mask_path) else None)
        logging.info("Loaded coefficients memmap from %s", args.load_memmap)
    elif args.do_precompute:
        # run precompute - memmap or in-memory
        if args.memmap_path:
            logging.info(
                "Starting tiled memmap precompute -> %s (tile %s, n_jobs=%d)",
                args.memmap_path,
                f"{args.tile_size[0]}x{args.tile_size[1]}",
                args.n_jobs,
            )
            ho.precompute_all_coeffs(
                n_jobs=args.n_jobs,
                use_tqdm=args.use_tqdm,
                memmap_path=args.memmap_path,
                memmap_dtype="float32",
                tile_size=args.tile_size,
            )
            logging.info("Memmap precompute finished: %s", args.memmap_path)
        else:
            logging.info(
                "Starting in-memory precompute (may use a lot of memory for big rasters)..."
            )
            ho.precompute_all_coeffs(
                n_jobs=args.n_jobs,
                use_tqdm=args.use_tqdm,
                memmap_path=None,
                tile_size=args.tile_size,
            )
            logging.info("In-memory precompute finished.")
    else:
        logging.error("No action requested: use --do-precompute or --load-memmap")
        sys.exit(1)

    # optionally export GeoTIFF
    if args.export_coeffs:
        outp = args.export_coeffs
        logging.info("Exporting coefficients to GeoTIFF: %s", outp)
        ho.export_coeffs_geotiff(outp, nodata=args.export_nodata, dtype="float32")
        logging.info("Export completed: %s", outp)

    logging.info("Script finished successfully.")


if __name__ == "__main__":
    main()
