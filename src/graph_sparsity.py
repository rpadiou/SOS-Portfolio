"""
Chordal Sparsity for Sparse SOS Hierarchies — SOS-Portfolio
=============================================================

Implements Minimum Degree Ordering (MDO) chordal extension and maximal
clique extraction for the Waki–Kim–Kojima–Muramatsu sparse SOS hierarchy.

Theory
------
Given an asset correlation graph G = (V, E) where {i,j} ∈ E iff |ρ_{ij}| > θ,
the polynomial objective f(x) has correlative sparsity: each monomial's support
is contained in the neighbourhood of some vertex.

The chordal extension G' ⊇ G (add fill-in edges) produces a chordal graph
whose maximal cliques {I_1,...,I_p} satisfy the Running Intersection Property
(RIP) by construction. This enables the decomposition:

    M_d(y) ≽ 0  →  M_d^{I_k}(y) ≽ 0  for each k = 1,...,p

reducing SDP size from C(n+d,d)² to Σ_k C(|I_k|+d,d)².

Reference: Waki, Kim, Kojima, Muramatsu (2006). "Sums of Squares and
Semidefinite Program Relaxations for Polynomial Optimization Problems with
Structured Sparsity." SIAM J. Optim. 17(1):218–242.
"""

import numpy as np
from typing import List, Set, Tuple


class ChordalExtension:
    """
    Builds a chordal extension of a graph and extracts its maximal cliques.

    Parameters
    ----------
    adjacency : np.ndarray, shape (n, n)
        Symmetric binary adjacency matrix. Diagonal is ignored.
    """

    def __init__(self, adjacency: np.ndarray) -> None:
        n = adjacency.shape[0]
        assert adjacency.shape == (n, n), "adjacency must be square"
        self.n = n
        self._orig = adjacency.astype(bool).copy()
        np.fill_diagonal(self._orig, False)
        self._peo: List[int] = []
        self._chordal: np.ndarray = np.zeros((n, n), dtype=bool)
        self._cliques: List[List[int]] = []
        self._built = False

    # ── Public API ────────────────────────────────────────────────────────────

    def build(self) -> "ChordalExtension":
        """Run MDO, compute chordal extension and extract maximal cliques."""
        self._peo = self._minimum_degree_ordering()
        self._chordal = self._compute_chordal()
        self._cliques = self._extract_maximal_cliques()
        self._built = True
        return self

    @property
    def perfect_elimination_ordering(self) -> List[int]:
        if not self._built:
            self.build()
        return list(self._peo)

    @property
    def chordal_adjacency(self) -> np.ndarray:
        if not self._built:
            self.build()
        return self._chordal.copy()

    @property
    def maximal_cliques(self) -> List[List[int]]:
        if not self._built:
            self.build()
        return [list(c) for c in self._cliques]

    def verify_rip(self) -> bool:
        """
        Verify the Running Intersection Property for the clique sequence.

        RIP: for each k ≥ 1, I_k ∩ (∪_{j<k} I_j) ⊆ I_j for some j < k.
        Holds automatically for chordal graphs by the clique-tree theorem.
        """
        if not self._built:
            self.build()
        cliques = self._cliques
        for k in range(1, len(cliques)):
            union_prev: Set[int] = set()
            for j in range(k):
                union_prev |= set(cliques[j])
            intersection = set(cliques[k]) & union_prev
            if intersection and not any(
                intersection.issubset(set(cliques[j])) for j in range(k)
            ):
                return False
        return True

    def fill_in_count(self) -> int:
        """Number of edges added to create the chordal extension."""
        if not self._built:
            self.build()
        added = np.sum(self._chordal & ~self._orig)
        return int(added // 2)

    def summary(self) -> str:
        if not self._built:
            self.build()
        orig_edges = int(np.sum(self._orig) // 2)
        chordal_edges = int(np.sum(self._chordal) // 2)
        clique_sizes = [len(c) for c in self._cliques]
        lines = [
            f"Nodes: {self.n}",
            f"Original edges: {orig_edges}",
            f"Fill-in edges: {chordal_edges - orig_edges}",
            f"Maximal cliques: {len(self._cliques)}",
            f"Max clique size: {max(clique_sizes) if clique_sizes else 0}",
            f"Mean clique size: {np.mean(clique_sizes):.2f}" if clique_sizes else "",
            f"RIP satisfied: {self.verify_rip()}",
        ]
        return "\n".join(lines)

    # ── Internal algorithms ───────────────────────────────────────────────────

    def _minimum_degree_ordering(self) -> List[int]:
        """
        Greedy Minimum Degree Ordering.

        At each step, eliminate the un-eliminated vertex with the smallest
        degree in the current elimination graph (including fill-in edges added
        so far). Ties broken by smallest index.

        Complexity: O(n³) — acceptable for n ≤ 200.
        """
        n = self.n
        # Work on a mutable adjacency (will accumulate fill-in)
        adj = self._orig.copy()
        eliminated = np.zeros(n, dtype=bool)
        ordering: List[int] = []

        for _ in range(n):
            # Degree of each vertex = number of non-eliminated neighbours
            degrees = np.sum(adj & ~eliminated[np.newaxis, :], axis=1)
            degrees[eliminated] = n + 1  # sentinel for eliminated vertices

            v = int(np.argmin(degrees))
            ordering.append(v)
            eliminated[v] = True

            # Identify non-eliminated neighbours of v
            nbrs = np.where(adj[v] & ~eliminated)[0].tolist()

            # Fill-in: make all pairs of nbrs adjacent (clique formation)
            for i, u in enumerate(nbrs):
                for w in nbrs[i + 1 :]:
                    adj[u, w] = True
                    adj[w, u] = True

        # Store the final elimination graph (chordal extension)
        self._elimination_adj = adj
        return ordering

    def _compute_chordal(self) -> np.ndarray:
        """Return the chordal adjacency produced during MDO (includes fill-in)."""
        return self._elimination_adj.copy()

    def _extract_maximal_cliques(self) -> List[List[int]]:
        """
        Extract maximal cliques from the chordal graph using PEO.

        For vertex v at position i in PEO, the elimination clique is:
            C(v) = {v} ∪ { u : adj[v,u] = True, pos[u] > pos[v] }

        The set of elimination cliques covers all maximal cliques of a
        chordal graph (Gavril 1974).
        """
        n = self.n
        peo = self._peo
        adj = self._chordal

        # Position of each vertex in the PEO
        pos = np.empty(n, dtype=int)
        for i, v in enumerate(peo):
            pos[v] = i

        raw_cliques: List[Tuple[int, ...]] = []
        for v in peo:
            later_nbrs = [u for u in range(n) if adj[v, u] and pos[u] > pos[v]]
            clique = tuple(sorted([v] + later_nbrs))
            raw_cliques.append(clique)

        # Keep only maximal cliques (not a proper subset of any other)
        clique_sets = [set(c) for c in raw_cliques]
        maximal: List[Tuple[int, ...]] = []
        for i, c in enumerate(raw_cliques):
            if not any(
                clique_sets[i] < clique_sets[j]  # strict subset
                for j in range(len(raw_cliques))
                if j != i
            ):
                maximal.append(c)

        # Deduplicate while preserving order
        seen: Set[Tuple[int, ...]] = set()
        unique: List[List[int]] = []
        for c in maximal:
            if c not in seen:
                seen.add(c)
                unique.append(list(c))

        return unique


def build_correlation_graph(
    corr_matrix: np.ndarray, threshold: float = 0.15
) -> np.ndarray:
    """
    Build a binary adjacency matrix from a correlation matrix.

    An edge {i,j} is included iff |ρ_{ij}| ≥ threshold (i ≠ j).
    Self-loops are excluded.

    Parameters
    ----------
    corr_matrix : np.ndarray, shape (n, n)
        Correlation matrix (diagonal = 1).
    threshold : float
        Minimum |ρ| for an edge.

    Returns
    -------
    np.ndarray, shape (n, n)
        Binary adjacency matrix.
    """
    n = corr_matrix.shape[0]
    adj = (np.abs(corr_matrix) >= threshold).astype(bool)
    np.fill_diagonal(adj, False)
    return adj


def build_block_adjacency(n: int, block_size: int) -> np.ndarray:
    """
    Build a block-diagonal adjacency for n assets grouped into clusters.

    Within each cluster of size block_size, all pairs are connected.
    Inter-cluster edges are absent. This yields a maximally sparse structure
    (no fill-in needed — each block is already a clique).

    Parameters
    ----------
    n : int
        Total number of assets.
    block_size : int
        Size of each cluster.

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    adj = np.zeros((n, n), dtype=bool)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        for i in range(start, end):
            for j in range(start, end):
                if i != j:
                    adj[i, j] = True
    return adj
