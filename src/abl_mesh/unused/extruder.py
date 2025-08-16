"""
abl_mesh.extruder
-----------------

Safe extrusion of surface mesh to create prismatic layers avoiding inverted prisms.
Implements the algorithm described in Section 5 of the paper and the approach in the assistant's earlier prototype.

Main class: SafeExtruder

Key behaviors:
 - compute pseudo-normals per node (area-weighted normal of adjacent triangles) blended with vertical
 - propose extruded top node positions with direction and target layer thickness
 - check prism inversion by splitting prisms into tets and verifying signed volumes
 - local backoff (halving per-node extrusion factors) for nodes participating in inverted prisms
 - final local untangling if needed
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def tetra_signed_volume(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed volume of tetra (a,b,c,d)."""
    return float(np.dot(np.cross(b - a, c - a), d - a) / 6.0)


class SafeExtruder:
    """
    Extrude a triangular surface mesh into a prismatic layer safely.

    Parameters
    ----------
    nodes : (N,3) ndarray
        Surface 3D nodes (x,y,z).
    tri : (M,3) ndarray
        Triangle connectivity referring to nodes indices.
    verbosity : int
        0/1/2 for logging verbosity.

    Usage:
        extruder = SafeExtruder(nodes, tri, verbosity=1)
        top_nodes, prism_conn = extruder.extrude_layer(h0=layer_thickness)
    """

    def __init__(self, nodes: np.ndarray, tri: np.ndarray, verbosity: int = 1):
        self.nodes = nodes.copy()
        self.tri = tri.copy().astype(int)
        self.verbosity = verbosity
        self.node_to_tri = self._build_node_triangle_adjacency(self.tri, len(self.nodes))

    @staticmethod
    def _build_node_triangle_adjacency(tri: np.ndarray, n_nodes: int):
        node_to_tri = [[] for _ in range(n_nodes)]
        for e, t in enumerate(tri):
            for v in t:
                node_to_tri[v].append(e)
        return node_to_tri

    def compute_pseudo_normals(self) -> np.ndarray:
        """
        Compute per-node pseudo-normal as normalized sum of triangle normals around the node.
        Returns array (N,3).
        """
        n_nodes = len(self.nodes)
        normals = np.zeros((n_nodes, 3), dtype=float)
        for i0, i1, i2 in self.tri:
            p0, p1, p2 = self.nodes[i0], self.nodes[i1], self.nodes[i2]
            tri_normal = np.cross(p1 - p0, p2 - p0)
            normals[i0] += tri_normal
            normals[i1] += tri_normal
            normals[i2] += tri_normal
        for i in range(n_nodes):
            nrm = np.linalg.norm(normals[i])
            if nrm < 1e-14:
                normals[i] = np.array([0.0, 0.0, 1.0])
            else:
                normals[i] = normals[i] / nrm
        return normals

    def extrude_layer(
        self,
        h0: float,
        blend_vertical: float = 0.25,
        max_backoffs: int = 8,
        min_factor: float = 0.05,
        quality_optimize: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extrude a single prismatic layer.

        Returns:
          top_nodes: (N,3) coordinates of extruded top nodes appended to mesh
          prism_conn: (M,6) prism connectivity [b0,b1,b2,t0,t1,t2] with top nodes indexed by offset N
        """
        n = len(self.nodes)
        normals = self.compute_pseudo_normals()
        vertical = np.array([0.0, 0.0, 1.0])
        dirs = normals * (1.0 - blend_vertical) + vertical * blend_vertical
        dirs_norm = np.linalg.norm(dirs, axis=1)
        dirs = dirs / dirs_norm[:, None]

        top_nodes = self.nodes + dirs * h0
        N = n
        prism_conn = []
        for tri in self.tri:
            b0, b1, b2 = tri
            prism_conn.append([b0, b1, b2, b0 + N, b1 + N, b2 + N])
        prism_conn = np.array(prism_conn, dtype=int)
        all_nodes = np.vstack([self.nodes, top_nodes])

        inverted = self._find_inverted_prisms(all_nodes, prism_conn)
        if self.verbosity >= 2:
            print(f"[extrude] initial inverted count {len(inverted)}")

        factors = np.ones(n, dtype=float)
        for backoff in range(max_backoffs):
            if not inverted:
                break
            affected_nodes = set()
            for pi in inverted:
                affected_nodes.update(prism_conn[pi][:3])
            affected_nodes = np.array(sorted(list(affected_nodes)), dtype=int)
            if self.verbosity >= 1:
                print(f"[extrude] backoff {backoff + 1}: reducing {len(affected_nodes)} nodes")
            factors[affected_nodes] *= 0.5
            factors = np.maximum(factors, min_factor)
            top_nodes = self.nodes + dirs * (h0 * factors[:, None])
            all_nodes = np.vstack([self.nodes, top_nodes])
            inverted = self._find_inverted_prisms(all_nodes, prism_conn)

        if inverted:
            # final local untangle heuristic
            if self.verbosity >= 1:
                print(
                    f"[extrude] still {len(inverted)} inverted after backoffs, performing local untangle"
                )
            all_nodes = self._local_untangle(all_nodes, prism_conn, inverted)

        # final quality optimize optional - simple local minimize on top nodes
        if quality_optimize:
            if self.verbosity >= 1:
                print("[extrude] running optional local quality optimization")
            all_nodes = self._local_optimize_quality(all_nodes, prism_conn)

        final_top = all_nodes[N:]
        return final_top, prism_conn

    def _find_inverted_prisms(self, all_nodes: np.ndarray, prism_conn: np.ndarray) -> list[int]:
        inverted = []
        tol = 1e-14
        for ei, prism in enumerate(prism_conn):
            b0, b1, b2, t0, t1, t2 = prism
            a = all_nodes[b0]
            b = all_nodes[b1]
            c = all_nodes[b2]
            d = all_nodes[t0]
            v1 = tetra_signed_volume(a, b, c, d)
            a = all_nodes[b1]
            b = all_nodes[b2]
            c = all_nodes[t0]
            d = all_nodes[t1]
            v2 = tetra_signed_volume(a, b, c, d)
            a = all_nodes[b2]
            b = all_nodes[t0]
            c = all_nodes[t1]
            d = all_nodes[t2]
            v3 = tetra_signed_volume(a, b, c, d)
            total = v1 + v2 + v3
            if (v1 <= tol) or (v2 <= tol) or (v3 <= tol) or (total <= tol):
                inverted.append(ei)
        return inverted

    def _local_untangle(
        self, all_nodes: np.ndarray, prism_conn: np.ndarray, inverted_prisms: list[int]
    ) -> np.ndarray:
        N = len(all_nodes) // 2
        top_nodes = all_nodes[N:].copy()
        bottom = all_nodes[:N]
        for pi in inverted_prisms:
            prism = prism_conn[pi]
            b0, b1, b2, t0, t1, t2 = prism
            bi = [b0, b1, b2]
            ti = [t0 - N, t1 - N, t2 - N]
            bottom_centroid = bottom[bi].mean(axis=0)
            for attempt in range(10):
                for j, tt in enumerate(ti):
                    vec = bottom_centroid - top_nodes[tt]
                    top_nodes[tt] += 0.3 * vec
                allr = np.vstack([bottom, top_nodes])
                inv = self._find_inverted_prisms(allr, prism_conn)
                if pi not in inv:
                    break
        return np.vstack([bottom, top_nodes])

    def _local_optimize_quality(self, all_nodes: np.ndarray, prism_conn: np.ndarray) -> np.ndarray:
        """
        Cheap global optimization of top nodes to increase positive tetra volumes.
        Uses scipy minimize on flattened top node coordinates with regularized inverse-volume cost.
        """
        N = len(all_nodes) // 2
        bottom = all_nodes[:N]
        top = all_nodes[N:].copy()
        prism_conn_local = prism_conn

        # cost: sum(1/(vol^2 + eps)) for tetra pieces
        eps = 1e-12

        def cost(x):
            top_nodes = x.reshape((-1, 3))
            A = np.vstack([bottom, top_nodes])
            total = 0.0
            for prism in prism_conn_local:
                b0, b1, b2, t0, t1, t2 = prism
                v1 = tetra_signed_volume(A[b0], A[b1], A[b2], A[t0])
                v2 = tetra_signed_volume(A[b1], A[b2], A[t0], A[t1])
                v3 = tetra_signed_volume(A[b2], A[t0], A[t1], A[t2])
                for v in (v1, v2, v3):
                    total += 1.0 / (v * v + eps)
            # keep close to proposed top to avoid large displacements
            reg = np.sum((top_nodes - top) ** 2) * 1e-4
            return total + reg

        x0 = top.flatten()
        try:
            res = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": 100, "ftol": 1e-8})
            top_opt = res.x.reshape((-1, 3))
            if self.verbosity >= 2:
                print("[extrude] optimization success:", res.success, res.message)
            return np.vstack([bottom, top_opt])
        except Exception as e:
            if self.verbosity:
                print("[extrude] optimization failed:", e)
            return all_nodes
