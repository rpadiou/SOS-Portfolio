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


def exact_min(f, n, b_lo, b_hi, m):
    """Minimum of f on {x >= 0, b_lo <= sum x <= b_hi}, independent of the SDP: grid, then SLSQP from the best grid points."""
    import itertools

    from scipy.optimize import LinearConstraint, minimize
    pts = []
    for idx in itertools.product(range(m + 1), repeat=n):
        x = np.array(idx) / m
        if b_lo - 1e-12 <= x.sum() <= b_hi + 1e-12:
            pts.append((f(x), x))
    pts.sort(key=lambda t: t[0])
    best = pts[0][0]
    A = LinearConstraint(np.ones((1, n)), b_lo, b_hi)
    for _, x0 in pts[:25]:
        r = minimize(f, x0, jac=f.gradient, method="SLSQP", bounds=[(0, 1)] * n, constraints=A,
                     options={"ftol": 1e-16, "maxiter": 500})
        if r.x.min() >= -1e-12 and b_lo - 1e-10 <= r.x.sum() <= b_hi + 1e-10:
            best = min(best, f(np.clip(r.x, 0, 1)))
    return best


def small_instances():
    from sos_portfolio import SyntheticMarket, build_objective, build_objective_from_market, make_nonconvex
    out = [("benchmark2", build_objective(), 2, 0.95, 1.0, 60)]
    for seed in (0, 1):
        mk, _ = make_nonconvex(2, 1, seed, min_minima=2)
        out.append((f"nonconvex2-s{seed}", build_objective_from_market(mk), 2, mk.budget_lower, mk.budget_upper, 60))
    for seed in (0, 2, 5):
        mk = SyntheticMarket.generate(3, 1, seed, skew_scale=3.0, kurt_base=0.3, impact_base=0.05)
        out.append((f"syn3-s{seed}", build_objective_from_market(mk), 3, mk.budget_lower, mk.budget_upper, 30))
    return out


def test_bound_matches_independent_optimum_on_small_instances():
    """|lb - f*| against a minimum computed without the SDP (n = 2, 3).

    CLARABEL at tolerance 1e-10 stalls with residuals of order 1e-9 (status optimal_inaccurate), so 1e-9 cannot be
    enforced (results/small_instance_offsets.csv). Measured at tolerance 1e-10: lb - f* is at most +3.2e-9 and at least
    -8.0e-9. The test allows +1e-8
    (bound above the optimum: invalid) and -5e-8 (bound far below the optimum: loose).
    """
    for name, f, n, lo, hi, m in small_instances():
        fs = exact_min(f, n, lo, hi, m)
        r = build_portfolio_relaxation(f, None, lo, hi, 2, "dense").solve(
            "CLARABEL", max_threads=1, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        d = r["lower_bound"] - fs
        assert -5e-8 <= d <= 1e-8, (name, d)
