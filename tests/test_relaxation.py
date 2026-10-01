import numpy as np
import pytest

from sos_portfolio import (MinimizerExtractor, MomentRelaxation, MultivariatePolynomial, SyntheticMarket,
                           build_objective, build_objective_from_market, build_portfolio_relaxation,
                           certify, scipy_optimize)
from sos_portfolio.local_solver import analyze_local_minima
from sos_portfolio.sos_hierarchy import Constraint


def solve(f, cliques=None, structure="dense", d=2, **kw):
    pr = build_portfolio_relaxation(f, cliques, 0.95, 1.0, d, structure, **kw)
    return pr, pr.solve()


def test_two_asset_benchmark_is_convex_and_certified():
    f = build_objective()
    # E1: Hessian is positive definite on K -> benchmark is a validation case, not a non-convex one
    lam = min(np.linalg.eigvalsh(f.hessian(np.array([t, s - t])))[0]
              for s in (0.95, 0.975, 1.0) for t in np.linspace(0, s, 41))
    assert lam > 1.0
    pr, r = solve(f)
    assert r["status"] == "optimal"
    assert r["lower_bound"] == pytest.approx(0.187371, abs=2e-6)
    ex = MinimizerExtractor(pr.relaxation, r["moments"], 2)
    assert ex.flatness()["flat"]
    x = ex.extract()["points"]
    assert x.shape == (1, 2)
    np.testing.assert_allclose(x[0], [0.6058, 0.3442], atol=5e-4)
    c = certify(f, r["lower_bound"], [x[0]], 0.95, 1.0)
    assert c["certified"] and c["gap"] < 1e-6


def test_order_one_is_rejected_for_degree_four():
    with pytest.raises(ValueError, match="order >= 2"):
        solve(build_objective(), d=1)


def test_monotone_in_order_small_instance():
    m = SyntheticMarket.generate(3, 1, 0, skew_scale=3.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    l2 = solve(f, d=2)[1]["lower_bound"]
    l3 = solve(f, d=3)[1]["lower_bound"]
    assert l2 <= l3 + 1e-6


def test_constraint_outside_clique_is_an_error():
    f = MultivariatePolynomial(3, {(2, 0, 0): 1.0})
    g = MultivariatePolynomial(3, {(0, 0, 1): 1.0})
    with pytest.raises(ValueError):
        MomentRelaxation(3, f, [[0, 1], [1, 2]], [Constraint(g, 0)], 2)


def test_lower_bound_is_valid_on_random_feasible_points():
    m = SyntheticMarket.generate(6, 2, 0, skew_scale=2.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    cl = __import__("util").cliques_of("blocks", 6)
    lb = solve(f, cl, "sparse")[1]["lower_bound"]
    rng = np.random.RandomState(0)
    X = rng.dirichlet(np.ones(6) * 0.3, size=1000) * rng.uniform(0.95, 1.0, size=(1000, 1))
    assert min(f(x) for x in X) >= lb - 1e-6


def test_solver_status_is_reported_not_swallowed():
    pr, r = solve(build_objective())
    assert r["status"] in ("optimal", "optimal_inaccurate") and "inaccurate" in r
    assert r["iterations"] is not None and r["sizes"]["max_block"] > 0
