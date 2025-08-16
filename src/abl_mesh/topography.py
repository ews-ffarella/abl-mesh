"""
abl_mesh.topography
-------------------

Topography modeling and high-order local approximant.

Implements:
 - Topography: holds a planar triangulation (2D parametric mesh) with heights z.
 - HighOrderApproximant: local least-squares polynomial fit of degree p (default p=3),
   enabling queries of z, grad z and Hessian at arbitrary parametric points.

This module implements the approach described in Section 3.1 of the paper:
 - build a cloud of neighbor points (default: p adjacency layers)
 - solve least-squares for polynomial basis (monomials up to degree p)
 - provide convenience methods:
     z(x,y), grad(x,y), hessian(x,y)

Notes:
 - The input geometry is expected to be provided as:
    nodes2d: (N,2) array of (x,y) param coords
    z:       (N,) array of heights
    tri:     (M,3) triangle connectivity (indices into nodes2d)
 - The polynomial fit is performed in the local (x,y) coordinates (no additional mapping)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq


class Topography:
    """
    Topography container.

    Attributes:
    -----------
    nodes2d : (N,2) ndarray
        Parametric coordinates (x,y) of input geometry vertices.
    z : (N,) ndarray
        Height values at nodes.
    tri : (M,3) ndarray
        Triangle connectivity (integer indices).

    """

    def __init__(self, nodes2d: np.ndarray, z: np.ndarray, tri: np.ndarray, verbosity: int = 1):
        assert nodes2d.shape[0] == z.shape[0]
        self.nodes2d = np.asarray(nodes2d, dtype=float)
        self.z = np.asarray(z, dtype=float)
        self.tri = np.asarray(tri, dtype=int)
        self.verbosity = verbosity

        # build node->triangle adjacency for neighbor gathering
        self.node_to_tri = [[] for _ in range(len(self.nodes2d))]
        for ei, t in enumerate(self.tri):
            for v in t:
                self.node_to_tri[v].append(ei)

        # build node->node adjacency (1-ring)
        self.node_to_node = [set() for _ in range(len(self.nodes2d))]
        for a, b, c in self.tri:
            self.node_to_node[a].update((b, c))
            self.node_to_node[b].update((a, c))
            self.node_to_node[c].update((a, b))
        self.node_to_node = [sorted(list(s)) for s in self.node_to_node]

    def locate_triangle(self, pt: tuple[float, float]) -> int | None:
        """
        Naive find which input triangle contains the parametric point `pt`.
        For large meshes replace by spatial acceleration (KDTree + bounding boxes).
        Returns triangle index or None if outside.
        """
        x, y = pt
        for ei, (i0, i1, i2) in enumerate(self.tri):
            p0 = self.nodes2d[i0]
            p1 = self.nodes2d[i1]
            p2 = self.nodes2d[i2]
            # barycentric
            mat = np.array([p1 - p0, p2 - p0]).T
            vec = np.array([x, y]) - p0
            try:
                uv = np.linalg.solve(mat, vec)
            except np.linalg.LinAlgError:
                continue
            u, v = uv
            if (u >= -1e-12) and (v >= -1e-12) and (u + v <= 1.0 + 1e-12):
                return ei
        return None


class HighOrderApproximant:
    """
    Provide local high-order polynomial approximations of the height field z(x,y).

    Parameters
    ----------
    topo : Topography
        Input topography mesh.
    degree : int
        Polynomial degree (default 3, as in the paper examples).
    adjacency : int
        Number of adjacency layers used to form the local cloud (default = degree).
    """

    def __init__(
        self, topo: Topography, degree: int = 3, adjacency: int | None = None, verbosity: int = 1
    ):
        self.topo = topo
        self.degree = degree
        self.adjacency = adjacency if adjacency is not None else degree
        self.verbosity = verbosity

        # prepare monomial index list up to degree p, consistent ordering
        self.monomials = []
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                self.monomials.append((i, j))
        self.num_coeffs = len(self.monomials)

    def _gather_cloud_indices(self, node_idx: int, min_pts: int | None = None) -> list[int]:
        """
        Gather a cloud of point indices around a given node by adjacency layers.
        Ensures at least num_coeffs points (increase layers if needed).
        """
        if min_pts is None:
            min_pts = self.num_coeffs + 3
        visited = {node_idx}
        frontier = [node_idx]
        layers = 0
        while len(visited) < min_pts and layers < max(6, self.adjacency + 3):
            layers += 1
            new_frontier = []
            for v in frontier:
                for nbr in self.topo.node_to_node[v]:
                    if nbr not in visited:
                        visited.add(nbr)
                        new_frontier.append(nbr)
            if not new_frontier:
                break
            frontier = new_frontier
        return sorted(visited)

    def _build_vandermonde(self, pts_xy: np.ndarray) -> np.ndarray:
        """
        Build Vandermonde matrix for monomials at pts_xy: shape (n_pts, num_coeffs)
        Order of cols corresponds to self.monomials
        """
        X = np.empty((pts_xy.shape[0], self.num_coeffs), dtype=float)
        for k, (i, j) in enumerate(self.monomials):
            X[:, k] = (pts_xy[:, 0] ** i) * (pts_xy[:, 1] ** j)
        return X

    def fit_at_node(self, node_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit a polynomial at the location of `node_idx` using least squares on the gathered cloud.
        Returns:
            coeffs : (num_coeffs,) polynomial coefficients in monomial order
            center : (2,) center point (x0, y0). Coeffs are expressed in absolute coordinates.
        Implementation detail:
            We do not subtract the center before fitting (could improve conditioning).
            For robust fits one could apply shift/scale; left simple here to match paper's exposition.
        """
        cloud = self._gather_cloud_indices(node_idx)
        pts = self.topo.nodes2d[cloud]
        zs = self.topo.z[cloud]
        A = self._build_vandermonde(pts)
        # solve least squares A c = z
        c, *_ = lstsq(A, zs)
        return c, self.topo.nodes2d[node_idx]

    def evaluate_poly(self, coeffs: np.ndarray, xy: np.ndarray) -> float:
        """
        Evaluate polynomial given coeffs at 2D point xy.
        """
        val = 0.0
        for k, (i, j) in enumerate(self.monomials):
            val += coeffs[k] * (xy[0] ** i) * (xy[1] ** j)
        return float(val)

    def gradient_and_hessian(self, coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Given polynomial coefficients, compute gradient and Hessian symbolic coefficients:
        - grad = [dz/dx, dz/dy] evaluated symbolically as linear forms of coeffs
        - Hessian 2x2: [d2z/dx2, d2z/dxdy; d2z/dydx, d2z/dy2]
        We return functions that evaluate grad and hessian at a point by reapplying monomial powers.
        For efficiency we produce arrays of exponents for evaluating derivatives.
        """

        # Precompute derivative monomial indices: for each coeff (i,j) derivative rules
        grad_i = np.array([i for (i, j) in self.monomials], dtype=int)
        grad_j = np.array([j for (i, j) in self.monomials], dtype=int)

        def eval_grad_at(xy):
            x, y = xy
            gx = 0.0
            gy = 0.0
            for k, (i, j) in enumerate(self.monomials):
                if i > 0:
                    gx += coeffs[k] * i * (x ** (i - 1)) * (y**j)
                if j > 0:
                    gy += coeffs[k] * j * (x**i) * (y ** (j - 1))
            return np.array([gx, gy], dtype=float)

        def eval_hess_at(xy):
            x, y = xy
            dxx = 0.0
            dxy = 0.0
            dyy = 0.0
            for k, (i, j) in enumerate(self.monomials):
                if i > 1:
                    dxx += coeffs[k] * i * (i - 1) * (x ** (i - 2)) * (y**j)
                if i > 0 and j > 0:
                    dxy += coeffs[k] * i * j * (x ** (i - 1)) * (y ** (j - 1))
                if j > 1:
                    dyy += coeffs[k] * j * (j - 1) * (x**i) * (y ** (j - 2))
            return np.array([[dxx, dxy], [dxy, dyy]], dtype=float)

        return eval_grad_at, eval_hess_at

    # Convenience higher-level API -----------------------------------------
    def query_at(self, xy: tuple[float, float]) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Query the smoothed height value & derivatives at XY.
        Strategy:
         - locate containing triangle (if possible)
         - choose the triangle centroid node index as center for local fit
         - fit local polynomial and evaluate at XY
        Returns:
         (z, grad, hessian)
        """
        tri_idx = self.topo.locate_triangle(xy)
        if tri_idx is None:
            # fallback: choose nearest node
            d2 = np.sum((self.topo.nodes2d - np.array(xy)) ** 2, axis=1)
            node_idx = int(np.argmin(d2))
        else:
            # choose first vertex of triangle as anchor
            node_idx = int(self.topo.tri[tri_idx, 0])

        coeffs, center = self.fit_at_node(node_idx)
        z = self.evaluate_poly(coeffs, np.array(xy))
        eval_grad, eval_hess = self.gradient_and_hessian(coeffs)
        grad = eval_grad(np.array(xy))
        hess = eval_hess(np.array(xy))
        return z, grad, hess

    def precompute_all_coeffs(self) -> list[np.ndarray]:
        """
        Optionally precompute polynomial coefficients at each node (costly, but speeds up many queries).
        Returns a list of coeff arrays indexed by node.
        """
        coeffs_list = []
        for ni in range(len(self.topo.nodes2d)):
            c, _ = self.fit_at_node(ni)
            coeffs_list.append(c)
            if self.verbosity and (ni % 500 == 0):
                print(f"[HO] precomputed coeffs at node {ni}/{len(self.topo.nodes2d)}")
        self._precomputed = coeffs_list
        return coeffs_list
