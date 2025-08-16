"""
Metric complexity utilities.

This module implements the theory and numeric routines to compute the metric complexity
integral and the global scaling factor beta* used to scale a metric field so that the
resulting metric has (approximately) a prescribed complexity (hence a target node count).

Functions
---------
- compute_complexity_on_grid: Evaluate metric sampler on a structured grid and compute C via a Riemann sum.
- integrate_complexity_from_components_tris: Compute C given a triangulation and pointwise metric components.
- compute_beta_star: Compute beta* from a target node count and complexity.
- scaled_metric_sampler: Return a sampler that scales a metric by beta.
- compute_beta_star_on_grid: Convenience helper that samples and computes beta* in one call.

The implementation tries to be vectorized using NumPy and provides optional numba-accelerated
kernels for the sqrt(det) operation to speed up very large sample arrays.

Note
----
- metric_sampler: callable((x,y)) -> 2x2 numpy.ndarray (symmetric positive-definite)
    For efficiency you can instead precompute arrays m11,m12,m22 at sample points and use
    integrate_complexity_from_components_* functions.
- The structured-grid integrator uses area = dx * dy for each sample cell.
- The triangulation integrator uses triangle area and averages the integrand at triangle vertices.
- Small numerical eps is used to avoid negative determinants triggering NaNs.
- The module supports parallel sampling with joblib (multiprocessing backend) when evaluating
  an arbitrary Python metric_sampler on many sample points.

Parallel evaluation note
------------------------
This module supports parallel evaluation of metric_sampler across sample points using
joblib (if installed). Use n_jobs > 1 to enable parallel sampling. The sampling tasks are
CPU-bound and call arbitrary Python code (the metric sampler), so the 'multiprocessing'
(loky) backend is used by default by joblib to avoid GIL bottlenecks when possible.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Optional joblib for parallel evaluation
try:
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except Exception:
    _HAS_JOBLIB = False

# Optional numba for hot kernels
try:
    from numba import njit

    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# Small epsilon for numerical stability
_EPS = 1e-18


if _HAS_NUMBA:

    @njit(inline="always")
    def _sqrt_det_components_numba(m11: np.ndarray, m12: np.ndarray, m22: np.ndarray) -> np.ndarray:
        """Numba-jitted sqrt(det) kernel for arrays of components."""
        n = m11.size
        out = np.empty(n, dtype=np.float64)
        for i in range(n):
            det = m11[i] * m22[i] - m12[i] * m12[i]
            if det < _EPS:
                det = _EPS
            out[i] = np.sqrt(det)
        return out
else:

    def _sqrt_det_components_numba(m11: np.ndarray, m12: np.ndarray, m22: np.ndarray) -> np.ndarray:
        """NumPy fallback sqrt(det) kernel (vectorized)."""
        det = m11 * m22 - m12 * m12
        det = np.maximum(det, _EPS)
        return np.sqrt(det)


def _sqrt_det_from_components(m11: np.ndarray, m12: np.ndarray, m22: np.ndarray) -> np.ndarray:
    """Wrapper that returns sqrt(det(M)) for flattened arrays.

    Supports both numba accelerated and NumPy vectorized paths.

    Args:
        m11: Array of M[0,0].
        m12: Array of M[0,1] (== M[1,0]).
        m22: Array of M[1,1].

    Returns:
        Array of sqrt(det(M)) with the same shape as inputs (broadcasted).
    """
    flat_m11 = np.asarray(m11).ravel()
    flat_m12 = np.asarray(m12).ravel()
    flat_m22 = np.asarray(m22).ravel()
    out_flat = _sqrt_det_components_numba(flat_m11, flat_m12, flat_m22)
    return out_flat.reshape(np.asarray(m11).shape)


def compute_complexity_on_grid(
    metric_sampler: Callable[[tuple[float, float]], np.ndarray],
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    mask: np.ndarray | None = None,
    n_jobs: int = 1,
    verbose: bool = False,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the metric complexity integral C = ∫ sqrt(det(M(x))) dx on a structured grid.

    The function samples the provided metric_sampler at a regular grid of points defined by
    the bounding box and (nx, ny) resolution, computes sqrt(det(M)) at each sample and uses
    a Riemann sum with area element dx*dy.

    The sampling of an arbitrary Python metric_sampler can be parallelized using joblib
    (multiprocessing backend) by setting `n_jobs > 1`.

    Args:
        metric_sampler: Callable that returns a 2x2 symmetric positive-definite matrix for (x,y).
        bbox: (xmin, xmax, ymin, ymax) sampling domain.
        nx: Number of samples in x direction.
        ny: Number of samples in y direction.
        mask: Optional boolean mask array of shape (ny, nx) or (ny*nx,) indicating which samples to include.
        n_jobs: Number of parallel jobs to evaluate metric_sampler. If <= 1 sampling is serial.
        verbose: If True, prints progress and debug information.

    Returns:
        C: Float complexity approximation.
        xs: 1D array (nx,) of sample x coordinates.
        ys: 1D array (ny,) of sample y coordinates.
        m11: (ny, nx) array of sampled M[0,0].
        m12: (ny, nx) array of sampled M[0,1].
        m22: (ny, nx) array of sampled M[1,1].

    Raises:
        ValueError: If mask shape does not match sampling grid.
    """
    xmin, xmax, ymin, ymax = bbox
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    dx = (xmax - xmin) / max(1, nx - 1)
    dy = (ymax - ymin) / max(1, ny - 1)
    area = abs(dx * dy)

    # Build flattened point list in row-major order
    X, Y = np.meshgrid(xs, ys)  # shapes (ny, nx)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    npts = pts.shape[0]

    if mask is not None:
        mask_arr = np.asarray(mask).ravel()
        if mask_arr.shape[0] != npts:
            raise ValueError("mask must have shape (ny,nx) or (ny*nx,)")
    else:
        mask_arr = np.ones(npts, dtype=bool)

    if verbose:
        print(
            f"[metric_complexity] sampling grid {nx} x {ny} = {npts} points, active {mask_arr.sum()}"
        )

    def _eval_single(pt):
        M = metric_sampler((float(pt[0]), float(pt[1])))
        M = np.asarray(M, dtype=float)
        # symmetrize
        M = 0.5 * (M + M.T)
        return M[0, 0], M[0, 1], M[1, 1]

    if n_jobs is None or n_jobs <= 1 or not _HAS_JOBLIB:
        results = [
            _eval_single(pt) if m else (np.nan, np.nan, np.nan)
            for pt, m in zip(pts, mask_arr, strict=False)
        ]
    else:
        results = Parallel(n_jobs=n_jobs, backend="multiprocessing")(
            delayed(_eval_single)(pt) if m else (np.nan, np.nan, np.nan)
            for pt, m in zip(pts, mask_arr, strict=False)
        )

    arr = np.array(results, dtype=float)  # shape (npts, 3)
    m11 = arr[:, 0].reshape((ny, nx))
    m12 = arr[:, 1].reshape((ny, nx))
    m22 = arr[:, 2].reshape((ny, nx))

    sdet = _sqrt_det_from_components(m11, m12, m22)
    # zero out masked samples
    sdet_flat = sdet.ravel()
    sdet_flat[~mask_arr] = 0.0
    sdet = sdet_flat.reshape((ny, nx))

    C = float(np.sum(sdet) * area)
    if verbose:
        print(f"[metric_complexity] complexity C ≈ {C:.6e} (area per sample {area:.3g})")
    return C, xs, ys, m11, m12, m22


def integrate_complexity_from_components_tris(
    pts2d: np.ndarray,
    tris: np.ndarray,
    m11: np.ndarray,
    m12: np.ndarray,
    m22: np.ndarray,
    verbose: bool = False,
) -> float:
    """Integrate complexity C from a triangulation and per-vertex metric components.

    The integral is approximated by summing triangle areas times the average
    of sqrt(det(M)) at the triangle's vertices:
        C ≈ Σ_area(tri) * (1/3) Σ_i sqrt(det(M_i))

    Args:
        pts2d: (N,2) array of point coordinates.
        tris: (M,3) array of triangle indices (zero-based).
        m11, m12, m22: (N,) arrays of metric components at each point.
        verbose: If True prints debug information.

    Returns:
        C: float complexity estimate.
    """
    pts = np.asarray(pts2d, dtype=float)
    tri = np.asarray(tris, dtype=int)
    m11 = np.asarray(m11, dtype=float).ravel()
    m12 = np.asarray(m12, dtype=float).ravel()
    m22 = np.asarray(m22, dtype=float).ravel()

    if pts.shape[0] != m11.shape[0]:
        raise ValueError("pts2d length must match metric component arrays")

    sdet_nodes = _sqrt_det_from_components(m11, m12, m22).ravel()

    # Vectorized triangle area computation
    a = pts[tri[:, 0]]
    b = pts[tri[:, 1]]
    c = pts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)  # (M,)

    # Average sdet at triangle vertices
    avg_sdet = (sdet_nodes[tri[:, 0]] + sdet_nodes[tri[:, 1]] + sdet_nodes[tri[:, 2]]) / 3.0
    C = float(np.sum(areas * avg_sdet))
    if verbose:
        print(
            f"[metric_complexity] triangulation complexity C ≈ {C:.6e} (triangles {tri.shape[0]})"
        )
    return C


def compute_beta_star(num_nodes: float, complexity_C: float, alpha: float = 2.0) -> float:
    """Compute beta* from the target number of nodes and complexity.

    Formula:
        beta* = num_nodes / (alpha * C)

    Args:
        num_nodes: Desired number of nodes (float or int).
        complexity_C: Complexity integral value C (must be > 0).
        alpha: Proportionality constant (theory: alpha = 2.0 for 2D). Default is 2.0.

    Returns:
        beta_star: scaling factor > 0.

    Raises:
        ValueError: If complexity_C <= 0.
    """
    if complexity_C <= 0.0:
        raise ValueError("complexity_C must be positive to compute beta*")
    beta_star = float(num_nodes) / (float(alpha) * float(complexity_C))
    return beta_star


def scaled_metric_sampler(
    metric_sampler: Callable[[tuple[float, float]], np.ndarray], beta: float
) -> Callable[[tuple[float, float]], np.ndarray]:
    """Return a metric sampler that multiplies the sampled metric by beta.

    Args:
        metric_sampler: Original sampler callable((x,y)) -> 2x2 ndarray.
        beta: Positive scaling factor.

    Returns:
        Callable that returns beta * M(x).
    """

    def sampler(xy):
        M = np.asarray(metric_sampler((float(xy[0]), float(xy[1]))), dtype=float)
        M = 0.5 * (M + M.T)
        return float(beta) * M

    return sampler


def scale_metric_components_inplace(m11: np.ndarray, m12: np.ndarray, m22: np.ndarray, beta: float):
    """Scale metric component arrays in-place by beta.

    Args:
        m11, m12, m22: Arrays of metric components (modified in place).
        beta: Positive scaling factor.
    """
    m11 *= float(beta)
    m12 *= float(beta)
    m22 *= float(beta)


def compute_beta_star_on_grid(
    metric_sampler: Callable[[tuple[float, float]], np.ndarray],
    bbox: tuple[float, float, float, float],
    nx: int,
    ny: int,
    num_nodes_target: float,
    alpha: float = 2.0,
    mask: np.ndarray | None = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convenience pipeline: sample grid -> compute C -> compute beta*.

    Returns (beta_star, C, xs, ys, m11, m12, m22).
    """
    C, xs, ys, m11, m12, m22 = compute_complexity_on_grid(
        metric_sampler, bbox, nx, ny, mask=mask, n_jobs=n_jobs, verbose=verbose
    )
    beta_star = compute_beta_star(num_nodes_target, C, alpha=alpha)
    if verbose:
        print(
            f"[metric_complexity] computed beta* = {beta_star:.6e} (num_nodes={num_nodes_target}, alpha={alpha}, C={C:.6e})"
        )
    return beta_star, C, xs, ys, m11, m12, m22
