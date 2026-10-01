"""Shared helpers: random correlatively sparse non-convex instances with overlapping cliques."""
import itertools

import numpy as np

from sos_portfolio import ChordalExtension, MultivariatePolynomial


def graph_edges(kind: str, n: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    if kind == "chain":      # cliques {0,1,2},{2,3,4},{4,5,6}...
        return [(i, j) for s in range(0, n - 2, 2) for i, j in itertools.combinations(range(s, s + 3), 2)]
    if kind == "star":       # hub 0 joined to every other vertex (cliques {0,i})
        return [(0, i) for i in range(1, n)]
    if kind == "random":     # sparse random graph; MDO adds fill-in
        return [(i, j) for i in range(n) for j in range(i + 1, n) if rng.rand() < 1.6 / n]
    if kind == "blocks":
        return [(i, j) for s in range(0, n, n // 2) for i, j in itertools.combinations(range(s, s + n // 2), 2)]
    raise ValueError(kind)


def cliques_of(kind: str, n: int, seed: int = 0):
    adj = np.zeros((n, n), dtype=bool)
    for i, j in graph_edges(kind, n, seed):
        adj[i, j] = adj[j, i] = True
    return ChordalExtension(adj).build().maximal_cliques


def random_sparse_poly(n: int, cliques, seed: int, skew: float = 3.0) -> MultivariatePolynomial:
    """Non-convex degree-4 polynomial whose monomials each lie in one clique."""
    rng = np.random.RandomState(seed)
    c = {}
    for cl in cliques:
        for i in cl:
            for deg, scale in [(2, 0.1), (3, skew), (4, 1.0)]:
                a = [0] * n
                a[i] = deg
                c[tuple(a)] = c.get(tuple(a), 0.0) + scale * rng.uniform(0.3, 1.2) * (rng.choice([-1, 1]) if deg == 3 else 1)
        for i, j in itertools.combinations(cl, 2):
            for e in [(1, 1), (2, 2), (1, 2), (2, 1)]:
                a = [0] * n
                a[i], a[j] = e
                c[tuple(a)] = c.get(tuple(a), 0.0) + rng.uniform(-0.5, 0.8) * (0.05 if sum(e) == 2 else 0.3)
    return MultivariatePolynomial(n, c)
