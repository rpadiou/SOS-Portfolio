"""Chordal extension, maximal cliques, clique trees and the running intersection property.

Reference: Waki, Kim, Kojima, Muramatsu, "Sums of squares and semidefinite program
relaxations for polynomial optimization problems with structured sparsity",
SIAM J. Optim. 17(3):218-242, 2006.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np


def verify_rip(cliques: Sequence[Sequence[int]]) -> bool:
    """Running intersection property for the given clique ORDER.

    For every k >= 1 there must be j < k with I_k ∩ (I_0 ∪ ... ∪ I_{k-1}) ⊆ I_j.
    """
    sets = [set(c) for c in cliques]
    seen: set = set()
    for k, s in enumerate(sets):
        if k > 0:
            inter = s & seen
            if inter and not any(inter <= sets[j] for j in range(k)):
                return False
        seen |= s
    return True


def clique_tree(cliques: Sequence[Sequence[int]]) -> Tuple[List[List[int]], List[int]]:
    """Maximum-weight spanning tree of the clique intersection graph.

    Returns (ordered_cliques, parent) where ordered_cliques is a traversal from a
    root (so the order satisfies RIP when the cliques come from a chordal graph)
    and parent[k] is the position of the parent of clique k (-1 for the root).
    Components with empty intersections are attached arbitrarily.
    """
    cl = [sorted(c) for c in cliques]
    m = len(cl)
    if m == 0:
        return [], []
    sets = [set(c) for c in cl]
    w = np.array([[len(sets[i] & sets[j]) for j in range(m)] for i in range(m)])
    in_tree = [0]
    order = [0]
    par = {0: -1}
    best = {j: (w[0, j], 0) for j in range(1, m)}
    while len(in_tree) < m:
        j = max(best, key=lambda t: (best[t][0], -t))
        p = best.pop(j)[1]
        par[j] = p
        order.append(j)
        in_tree.append(j)
        for t in best:
            if w[j, t] > best[t][0]:
                best[t] = (w[j, t], j)
    pos = {c: i for i, c in enumerate(order)}
    return [cl[c] for c in order], [(-1 if par[c] == -1 else pos[par[c]]) for c in order]


def is_chordal(adj: np.ndarray, peo: Sequence[int]) -> bool:
    """True iff `peo` is a perfect elimination ordering of the graph `adj`."""
    n = adj.shape[0]
    pos = {v: i for i, v in enumerate(peo)}
    for v in peo:
        later = [u for u in range(n) if adj[v, u] and pos[u] > pos[v]]
        for a in range(len(later)):
            for b in range(a + 1, len(later)):
                if not adj[later[a], later[b]]:
                    return False
    return True


class ChordalExtension:
    """Minimum-degree chordal extension of a graph and its maximal cliques.

    Parameters
    ----------
    adjacency : (n, n) symmetric boolean array; the diagonal is ignored.
    """

    def __init__(self, adjacency: np.ndarray) -> None:
        n = adjacency.shape[0]
        assert adjacency.shape == (n, n), "adjacency must be square"
        self.n = n
        self._orig = adjacency.astype(bool).copy()
        np.fill_diagonal(self._orig, False)
        self._built = False

    def build(self) -> "ChordalExtension":
        n = self.n
        adj = self._orig.copy()
        eliminated = np.zeros(n, dtype=bool)
        peo: List[int] = []
        for _ in range(n):
            deg = np.sum(adj & ~eliminated[None, :], axis=1)
            deg[eliminated] = n + 1
            v = int(np.argmin(deg))
            peo.append(v)
            eliminated[v] = True
            nb = np.where(adj[v] & ~eliminated)[0]
            for a in range(len(nb)):
                for b in range(a + 1, len(nb)):
                    adj[nb[a], nb[b]] = adj[nb[b], nb[a]] = True
        self._peo = peo
        self._chordal = adj
        self._cliques = self._maximal_cliques(adj, peo)
        self._built = True
        return self

    @staticmethod
    def _maximal_cliques(adj: np.ndarray, peo: List[int]) -> List[List[int]]:
        n = adj.shape[0]
        pos = np.empty(n, dtype=int)
        for i, v in enumerate(peo):
            pos[v] = i
        raw = [tuple(sorted([v] + [u for u in range(n) if adj[v, u] and pos[u] > pos[v]])) for v in peo]
        sets = [set(c) for c in raw]
        out: List[List[int]] = []
        seen = set()
        for i, c in enumerate(raw):
            if c in seen or any(sets[i] < sets[j] for j in range(len(raw))):
                continue
            seen.add(c)
            out.append(list(c))
        return out

    def _ensure(self) -> None:
        if not self._built:
            self.build()

    @property
    def perfect_elimination_ordering(self) -> List[int]:
        self._ensure()
        return list(self._peo)

    @property
    def chordal_adjacency(self) -> np.ndarray:
        self._ensure()
        return self._chordal.copy()

    @property
    def maximal_cliques(self) -> List[List[int]]:
        """Maximal cliques ordered along a clique tree (RIP-compatible order)."""
        self._ensure()
        return clique_tree(self._cliques)[0]

    @property
    def clique_parents(self) -> List[int]:
        self._ensure()
        return clique_tree(self._cliques)[1]

    def verify_rip(self) -> bool:
        return verify_rip(self.maximal_cliques)

    def fill_in_count(self) -> int:
        self._ensure()
        return int(np.sum(self._chordal & ~self._orig) // 2)

    def summary(self) -> str:
        self._ensure()
        sizes = [len(c) for c in self._cliques]
        return "\n".join([
            f"nodes={self.n} edges={int(self._orig.sum() // 2)} fill-in={self.fill_in_count()}",
            f"maximal cliques={len(sizes)} max size={max(sizes, default=0)} RIP={self.verify_rip()}",
        ])


def build_correlation_graph(corr: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    adj = np.abs(corr) >= threshold
    np.fill_diagonal(adj, False)
    return adj


def build_block_adjacency(n: int, block_size: int) -> np.ndarray:
    adj = np.zeros((n, n), dtype=bool)
    for s in range(0, n, block_size):
        e = min(s + block_size, n)
        adj[s:e, s:e] = True
    np.fill_diagonal(adj, False)
    return adj
