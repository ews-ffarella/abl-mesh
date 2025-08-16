"""
abl_mesh.optimizer
------------------

Mesh quality measures and optimization utilities.

Implements:
 - Scaled Jacobian / distortion measure for triangles, tetrahedra and prisms (conceptual).
 - Least-squares global objective (sum of squared distortions).
 - Local Gauss-Seidel node-wise minimization (per-node L-BFGS-B via scipy).

This module implements the optimization approach from Section 6, adapted to our element set.

Note:
 - For prisms we evaluate energy by splitting into tetrahedra (same splitting used for inversion test),
   compute tetra distortion and sum; the ideal prism is orthogonal extruded triangle of desired anisotropy.
 - This implementation focuses on shape quality (not size) as described in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def quality_triangle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Simple quality measure for triangle in (0,1], 1=equilateral of desired size.
    Here we use normalized area vs edge lengths (a practical algebraic measure).
    """
    e0 = np.linalg.norm(b - c)
    e1 = np.linalg.norm(c - a)
    e2 = np.linalg.norm(a - b)
    s = 0.5 * (e0 + e1 + e2)
    area = max(
        0.0,
        0.25
        * np.sqrt(max(0.0, (e0 + e1 + e2) * (-e0 + e1 + e2) * (e0 - e1 + e2) * (e0 + e1 - e2))),
    )
    if s <= 0 or area <= 0:
        return 0.0
    # normalized radius ratio (inradius / circumradius) ranges (0,0.5] ; 0.5 for equilateral
    r_in = area / s
    R = e0 * e1 * e2 / (4.0 * area) if area > 0 else 1e12
    if R <= 0:
        return 0.0
    scaled = (r_in / R) / 0.5
    return float(np.clip(scaled, 0.0, 1.0))


def tetra_quality(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """
    Simple scaled Jacobian proxy for tetrahedra.
    Uses volume scaled by squared edge lengths.
    """
    vol = abs(np.dot(np.cross(b - a, c - a), d - a)) / 6.0
    if vol <= 0:
        return 0.0
    # heuristic normalization factor using average edge length
    edges = [
        np.linalg.norm(b - a),
        np.linalg.norm(c - a),
        np.linalg.norm(d - a),
        np.linalg.norm(c - b),
        np.linalg.norm(d - b),
        np.linalg.norm(d - c),
    ]
    mean_e = np.mean(edges)
    denom = mean_e**3 + 1e-18
    q = (vol / denom) * 12.0  # scale to [0,1] typically
    return float(np.clip(q, 0.0, 1.0))


class QualityOptimizer:
    """
    Local Gauss-Seidel optimizer minimizing sum of squared distortions for a mesh.

    Parameters
    ----------
    nodes : (N,3) ndarray
        global nodes (modified in-place if optimize_inplace=True)
    tris : (M,3) ndarray
        surface triangles
    prisms : (P,6) ndarray, optional
        prism connectivity for volumetric layer
    tets : (K,4) ndarray, optional
        tetrahedra connectivity for tetrahedral region
    verbosity : int
    """

    def __init__(
        self,
        nodes: np.ndarray,
        tris: np.ndarray,
        prisms: np.ndarray = None,
        tets: np.ndarray = None,
        verbosity: int = 1,
    ):
        self.nodes = nodes
        self.tris = tris
        self.prisms = prisms
        self.tets = tets
        self.verbosity = verbosity

        # build node->elements adjacency
        N = len(nodes)
        self.node_to_elems = [[] for _ in range(N)]
        if tris is not None:
            for ei, t in enumerate(tris):
                for v in t:
                    self.node_to_elems[v].append(("tri", ei))
        if prisms is not None:
            for ei, p in enumerate(prisms):
                for v in p:
                    self.node_to_elems[v].append(("pri", ei))
        if tets is not None:
            for ei, tet in enumerate(tets):
                for v in tet:
                    self.node_to_elems[v].append(("tet", ei))

    def element_quality(self, elem_type: str, eid: int, nodes_local: np.ndarray) -> float:
        """
        Evaluate quality of a given element by type using nodes_local (global nodes array).
        """
        if elem_type == "tri":
            a, b, c = nodes_local[self.tris[eid]]
            return quality_triangle(a, b, c)
        if elem_type == "pri":
            # split prism into 3 tets like in extruder
            p = self.prisms[eid]
            b0, b1, b2, t0, t1, t2 = p
            a = nodes_local[b0]
            b = nodes_local[b1]
            c = nodes_local[b2]
            d = nodes_local[t0]
            q1 = tetra_quality(a, b, c, d)
            a = nodes_local[b1]
            b = nodes_local[b2]
            c = nodes_local[t0]
            d = nodes_local[t1]
            q2 = tetra_quality(a, b, c, d)
            a = nodes_local[b2]
            b = nodes_local[t0]
            c = nodes_local[t1]
            d = nodes_local[t2]
            q3 = tetra_quality(a, b, c, d)
            # aggregate quality
            return float((q1 + q2 + q3) / 3.0)
        if elem_type == "tet":
            a, b, c, d = nodes_local[self.tets[eid]]
            return tetra_quality(a, b, c, d)
        return 0.0

    def local_optimize_node(self, node_idx: int, maxiter: int = 40) -> bool:
        """
        Optimize a single free node's coordinates by minimizing sum of squared (1-quality).
        Returns True if optimization progressed.
        """
        x0 = self.nodes[node_idx].copy()

        # build local objective: sum_{adj elems} (1-q(elem))^2
        def obj(x):
            old = self.nodes[node_idx].copy()
            self.nodes[node_idx] = x
            val = 0.0
            for etype, eid in self.node_to_elems[node_idx]:
                q = self.element_quality(etype, eid, self.nodes)
                val += (1.0 - q) ** 2
            self.nodes[node_idx] = old
            return val

        try:
            res = minimize(obj, x0, method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-9})
            if res.success and np.linalg.norm(res.x - x0) > 1e-10:
                self.nodes[node_idx] = res.x
                return True
            return False
        except Exception as e:
            if self.verbosity:
                print(f"[opt] node {node_idx} optimization failed: {e}")
            return False

    def gauss_seidel(self, iterations: int = 2):
        """
        Run Gauss-Seidel passes over free nodes. Optimize only nodes which are adjacent to low-quality elements.
        """
        N = len(self.nodes)
        for it in range(iterations):
            if self.verbosity:
                print(f"[opt] GS iter {it + 1}/{iterations}")
            # find nodes adjacent to elements with q < threshold
            lowq_nodes = set()
            for v in range(N):
                for etype, eid in self.node_to_elems[v]:
                    q = self.element_quality(etype, eid, self.nodes)
                    if q < 0.6:
                        lowq_nodes.add(v)
                        break
            changed = 0
            for v in sorted(lowq_nodes):
                ok = self.local_optimize_node(v)
                if ok:
                    changed += 1
            if self.verbosity:
                print(f"[opt] changed {changed} nodes in pass {it + 1}")
            if changed == 0:
                break
