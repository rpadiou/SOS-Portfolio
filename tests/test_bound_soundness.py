"""The SDP bound must not exceed the objective at any feasible point, up to solver accuracy.

Instances mimic experiment C (10 assets, heavy-tailed synthetic returns, weights capped at 0.25, w4 = 0 and w4 > 0).
Tolerance: the observed excess of the bound over the objective at the SOS point is below 2e-6 (relative to
max(1, |obj|)) when CLARABEL runs at 1e-9 and converges (results/bound_tolerance.csv); the test allows 5e-6.
"""
import numpy as np
import pytest

from sos_portfolio import MinimizerExtractor, build_portfolio_relaxation
from sos_portfolio import empirical as em

CAP, N, TOL = 0.25, 10, 5e-6

pytestmark = pytest.mark.slow


def feasible_points(rng, m=2000):
    pts = []
    while len(pts) < m - 300:
        x = rng.dirichlet(np.full(N, rng.choice([0.5, 1.0, 3.0])))
        if x.max() <= CAP:
            pts.append(x)
    for _ in range(300):                        # vertices and edges of {sum = 1, 0 <= x <= CAP}
        x = np.zeros(N)
        k = int(1 / CAP)
        idx = rng.permutation(N)
        x[idx[:k]] = CAP
        if rng.rand() < 0.5:
            x[idx[k - 1]] = CAP * rng.rand()
            x[idx[k]] = 1.0 - x.sum()
        pts.append(x)
    return np.array(pts)


@pytest.mark.parametrize("w4", [0.0, 1.0, 3.0])
def test_bound_below_objective_at_feasible_points(w4):
    rng = np.random.RandomState(0)
    L = np.linalg.cholesky(0.3 * np.ones((N, N)) + 0.7 * np.eye(N))
    R = (rng.standard_t(4, size=(756, N)) @ L.T) * 0.01
    Rc = em.prepare_window(R)
    blocks = em.blocks_of(N, None)
    lam = em.lambdas(Rc, blocks, (1.0, 1.0, w4))
    f = em.scale_polynomial(em.polynomial(Rc, blocks, lam), CAP)
    pr = build_portfolio_relaxation(f, None, 1 / CAP, 1 / CAP, 2, "dense", budget="star")
    r = pr.solve("CLARABEL", max_threads=1, tol_gap_abs=1e-9, tol_gap_rel=1e-9, tol_feas=1e-9)
    lb = r["lower_bound"]
    tol = TOL * max(1.0, abs(lb))
    X = feasible_points(rng)
    assert np.all(np.abs(X.sum(axis=1) - 1) < 1e-12) and X.max() <= CAP + 1e-12
    assert min(f(x / CAP) for x in X) >= lb - tol                                   # (i) random feasible points
    xl = em.solve_local(Rc, blocks, lam, cap=CAP)                                    # (iii) local solver upper bound
    assert f(xl / CAP) >= lb - tol
    pts = MinimizerExtractor(pr.relaxation, r["moments"], N).extract()["points"]     # (ii) extracted points, if any
    for p in ([] if pts is None else pts):
        if p.min() >= -1e-8 and abs(p.sum() - 1 / CAP) < 1e-6 and p.max() <= 1 + 1e-8:
            assert f(p) >= lb - tol
