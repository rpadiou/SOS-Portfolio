"""The sparse relaxation (with sector-sum variables) is a relaxation of the dense one.

A dense moment vector is lifted to the sparse variables by substituting s_k = sum of the variables owned by
clique k (and the partial sums p_k for the chain budget) in every monomial. The lifted vector must satisfy
every sparse constraint, up to solver accuracy, with the dense objective value.
"""

import numpy as np
import pytest

from sos_portfolio import ChordalExtension, SyntheticMarket, build_objective_from_market, build_portfolio_relaxation
from sos_portfolio.polynomial_ring import MultivariatePolynomial


def lifted_violation(n, nc, seed, budget):
    m = SyntheticMarket.generate(n, nc, seed, skew_scale=3.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    dense = build_portfolio_relaxation(f, None, 0.95, 1.0, 2, "dense")
    rd = dense.solve("CLARABEL", max_threads=1, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    sparse = build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, "sparse", budget=budget)
    problem, y = sparse.relaxation.build()
    ix, dix = sparse.relaxation.indexer, dense.relaxation.indexer
    own = [o for o in sparse.owners if o]

    def var(i):
        return MultivariatePolynomial(n, {tuple(1 if j == i else 0 for j in range(n)): 1.0})

    sub = {n + k: MultivariatePolynomial(n, {tuple(1 if j == i else 0 for j in range(n)): 1.0 for i in o})
           for k, o in enumerate(own)}
    if budget == "chain" and len(own) > 2:
        run = None
        for k in range(len(own)):
            run = sub[n + k] if run is None else run + sub[n + k]
            sub[n + len(own) + k] = run
    base = sparse.n_total + 1
    values = np.zeros(ix.n_moments)
    for pos, key in enumerate(ix._keys):
        alpha, k = [0] * sparse.n_total, int(key)
        while k > 0:
            digit, k = k % base, k // base
            if digit:
                alpha[digit - 1] += 1
        p = MultivariatePolynomial(n, {(0,) * n: 1.0})
        for v, e in enumerate(alpha):
            for _ in range(e):
                p = p * (sub[v] if v >= n else var(v))
        values[pos] = sum(c * rd["moments"][dix.index(list(a))] for a, c in p.coeffs.items())
    y.value = values
    return rd["lower_bound"], float(problem.objective.value), max(float(np.max(c.violation())) for c in problem.constraints)


@pytest.mark.parametrize("n,nc,budget", [(4, 2, "star"), (6, 3, "chain"), (8, 2, "star")])
def test_dense_solution_lifts_to_a_sparse_feasible_point(n, nc, budget):
    lb_dense, objective, violation = lifted_violation(n, nc, 0, budget)
    assert objective == pytest.approx(lb_dense, abs=1e-9)
    assert violation < 1e-5


def test_cliques_without_running_intersection_property_raise_a_warning():
    from sos_portfolio import MultivariatePolynomial as P
    f = P(3, {(2, 0, 0): 1.0, (0, 2, 0): 1.0, (0, 0, 2): 1.0})
    with pytest.warns(UserWarning, match="running intersection"):
        build_portfolio_relaxation(f, [[0, 1], [1, 2], [2, 0]], 0.95, 1.0, 2, "sparse")
