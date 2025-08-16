"""
Unit tests for metric_complexity utilities.

Requires pytest. These tests are lightweight and do not require gmsh or rasterio.
"""

import numpy as np

from abl_mesh import metric_complexity


def test_constant_isotropic_metric_grid():
    """
    For constant isotropic metric M = (1/h^2) * I on bbox,
    sqrt(det(M)) = 1/h^2 and C = area / h^2.
    """
    h_target = 5.0

    def sampler(xy):
        return np.array([[1.0 / (h_target**2), 0.0], [0.0, 1.0 / (h_target**2)]], dtype=float)

    bbox = (0.0, 10.0, 0.0, 20.0)  # area 200
    nx, ny = 50, 100
    C, xs, ys, m11, m12, m22 = metric_complexity.compute_complexity_on_grid(
        sampler, bbox, nx, ny, n_jobs=1, verbose=False
    )
    expected = ((bbox[1] - bbox[0]) * (bbox[3] - bbox[2])) / (h_target**2)
    # allow a small relative tolerance due to discrete sampling
    assert abs(C - expected) / max(expected, 1e-12) < 1e-3


def test_compute_beta_star_formula():
    C = 123.456
    num_nodes = 2000.0
    alpha = 2.0
    beta = metric_complexity.compute_beta_star(num_nodes, C, alpha)
    assert np.isfinite(beta)
    assert abs(beta - (num_nodes / (alpha * C))) < 1e-12
