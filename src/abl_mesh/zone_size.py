"""
Zone size helper

Compute a scalar size field h(x,y) from three zones:
 - inner polygon (Shapely polygon) -> hmin
 - transition band (outside polygon up to 'transition_width') -> linear ramp hmin -> hmax
 - outside transition -> hmax

Functions:
 - compute_zone_size_for_points(points, inner_poly, transition_width, hmin, hmax)
   points : (N,2) ndarray of (x,y)
   inner_poly : shapely.geometry.Polygon (interior zone with hmin)
   transition_width : float (ramp width outside the polygon)
   hmin, hmax : floats

Notes:
 - The polygon is given in the same parametric coordinates as the sampling bbox.
 - Points exactly on the polygon boundary get hmin.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon


def compute_zone_size_for_points(
    pts: np.ndarray,
    inner_poly: Polygon,
    transition_width: float,
    hmin: float,
    hmax: float,
) -> np.ndarray:
    """
    Compute per-point scalar size following the 3-zone rule.

    Parameters
    ----------
    pts : (N,2) array of sample points (x,y)
    inner_poly : shapely Polygon defining interior (hmin)
    transition_width : width (distance) outside polygon where sizes ramp to hmax
    hmin, hmax : float sizes

    Returns
    -------
    sizes : (N,) array of sizes clipped to [hmin,hmax]
    """
    if transition_width <= 0:
        raise ValueError("transition_width must be > 0")

    sizes = np.empty(len(pts), dtype=float)

    # Precompute polygon boundary geometry for distance queries
    poly_boundary = inner_poly.exterior

    for i, (x, y) in enumerate(pts):
        pt = Point(float(x), float(y))
        if inner_poly.contains(pt) or inner_poly.touches(pt):
            sizes[i] = float(hmin)
        else:
            # distance from point to polygon boundary (positive outside)
            d = poly_boundary.distance(pt)
            if d >= transition_width:
                sizes[i] = float(hmax)
            else:
                t = d / float(transition_width)
                sizes[i] = float(hmin + (hmax - hmin) * t)

    return sizes
