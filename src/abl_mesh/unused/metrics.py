"""
abl_mesh.metrics
----------------

Metric computation used by the surface adaptation:

 - Tangent metric (first fundamental form) scaled by local desired size h(x) (Eq. (8)-(10) in paper)
 - Curvature metric from Hessian eigen decomposition, scaled for a target complexity (Eqs. (11)-(16))

Implements:
 - MetricField: combines tangent & curvature metrics and produces final metric M(x).
 - metric_length: (approx) compute metric length of an edge via midpoint rule.

Notes:
 - Implementation follows paper notation; see HighOrderApproximant for geometry queries.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from abl_mesh.topography import HighOrderApproximant


def first_fundamental_form(jacobian: np.ndarray) -> np.ndarray:
    """
    Given Jacobian of the parameterization (3x2 matrix: dphi/dx, dphi/dy columns),
    return the 2x2 first fundamental form G = J^T J
    """
    return jacobian.T @ jacobian


class MetricField:
    """
    Metric field builder following Section 3.2 of the paper.

    Parameters
    ----------
    ho : HighOrderApproximant
        provides geometry queries (z, grad, hessian). For topographies, we use parametric
        representation phi(x,y) = (x, y, z(x,y)).
    h_scalar : function or float
        desired scalar element size field h(x,y). If a scalar is given it is constant.
    """

    def __init__(
        self,
        ho: HighOrderApproximant,
        h_scalar=1.0,
        alpha_complexity: float = 2.0,
        verbosity: int = 1,
    ):
        self.ho = ho
        self.h_scalar = (lambda xy: float(h_scalar)) if np.isscalar(h_scalar) else h_scalar
        self.alpha = alpha_complexity
        self.verbosity = verbosity

    def tangent_metric(self, xy: tuple[float, float]) -> np.ndarray:
        """
        Compute tangent metric M_T = (1/h^2) * (grad phi)^T (grad phi)
        where phi(x,y) = (x, y, z(x,y)) and grad phi is 3x2 Jacobian: [1 0; 0 1; dz/dx dz/dy]
        """
        z, grad, _ = self.ho.query_at(xy)
        dzdx, dzdy = grad
        J = np.array([[1.0, 0.0], [0.0, 1.0], [dzdx, dzdy]])
        G = first_fundamental_form(J)
        h = float(self.h_scalar(xy))
        M_T = G / (h * h)
        return M_T

    def curvature_metric_base(self, xy: tuple[float, float]) -> np.ndarray:
        """
        Compute curvature metric M_C^1 (beta=1) from Hessian H, Eq. (11).
        Returns the 2x2 symmetric PSD matrix V diag(|lambda1|,|lambda2|) V^T.
        """
        _, _, H = self.ho.query_at(xy)
        # Hessian H is 2x2
        eigvals, eigvecs = np.linalg.eigh(H)
        D = np.diag(np.abs(eigvals))
        M = eigvecs @ D @ eigvecs.T
        return M

    def curvature_metric_scaled(self, xy: tuple[float, float], beta: float) -> np.ndarray:
        """
        Return M_C^beta = V (beta * D) V^T == beta * M_C^1
        """
        base = self.curvature_metric_base(xy)
        return beta * base

    def complexity_of_metric(
        self,
        metric_sampler: Callable[[tuple[float, float]], np.ndarray],
        bbox: tuple[float, float, float, float],
        nx: int = 80,
        ny: int = 80,
    ) -> float:
        """
        Numerically estimate continuous complexity C = int sqrt(det M) dA
        using product quadrature on bounding box.
        bbox = (xmin,xmax,ymin,ymax)
        """
        xmin, xmax, ymin, ymax = bbox
        xs = np.linspace(xmin, xmax, nx)
        ys = np.linspace(ymin, ymax, ny)
        dx = (xmax - xmin) / (nx - 1)
        dy = (ymax - ymin) / (ny - 1)
        C = 0.0
        for xi in xs:
            for yi in ys:
                M = metric_sampler((xi, yi))
                det = np.linalg.det(M)
                det = max(det, 0.0)
                C += np.sqrt(det) * dx * dy
        return float(C)

    def curvature_metric_with_complexity(
        self,
        bbox: tuple[float, float, float, float],
        target_num_nodes: int,
        beta_initial: float = 1.0,
        sampler_res=(80, 80),
    ) -> tuple[float, np.ndarray]:
        """
        Compute the beta* value according to Eq. (15):
          beta* = NumNodes / (alpha * C(M_C^1))
        And return the scalar beta*.
        """
        if self.verbosity:
            print("[metric] estimating complexity of curvature metric on bbox", bbox)
        C = self.complexity_of_metric(
            lambda xy: self.curvature_metric_base(xy), bbox, nx=sampler_res[0], ny=sampler_res[1]
        )
        if C <= 0:
            if self.verbosity:
                print("[metric] warning: curvature complexity computed zero; returning beta=1")
            return 1.0, np.eye(2)
        beta_star = float(target_num_nodes) / (self.alpha * C)
        if self.verbosity:
            print(
                f"[metric] complexity C={C:.4g}, beta*={beta_star:.4g} for target nodes {target_num_nodes}"
            )
        # Return beta* and the base curvature matrix for reference
        return beta_star, None

    def combined_metric(self, xy: tuple[float, float], beta_star: float) -> np.ndarray:
        """
        Combine tangent and curvature metrics into a single metric.
        The paper uses different blends and intersection approaches; here we follow the
        practical approach: M = M_T + M_C^beta, then clamp eigenvalues to avoid extreme sizes.
        """
        MT = self.tangent_metric(xy)
        MC = self.curvature_metric_scaled(xy, beta_star)
        M = MT + MC
        # clamp eigenvalues to ensure numerical stability
        eigs, vecs = np.linalg.eigh(M)
        # clamp to small positive (avoid degeneracy)
        eigs_clamped = np.clip(eigs, 1e-12, 1e12)
        M_clamped = vecs @ np.diag(eigs_clamped) @ vecs.T
        return M_clamped


def metric_edge_length(
    M_func: Callable[[tuple[float, float]], np.ndarray],
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> float:
    """
    Approximate line integral sqrt((dx)^T M(x) (dx)) along straight segment [p0,p1]
    using the midpoint rule (order 2).
    """
    mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    dx = np.array([p1[0] - p0[0], p1[1] - p0[1]])
    Mmid = M_func(mid)
    val = np.sqrt(float(dx @ Mmid @ dx))
    return val
