import numpy as np

from sos_portfolio import MultivariatePolynomial, SyntheticMarket, build_objective_from_market, scipy_optimize
from sos_portfolio.local_solver import (analyze_local_minima, feasibility_violation, kkt_residual, project_to_K)


def test_projection_is_feasible_and_idempotent():
    rng = np.random.RandomState(0)
    for _ in range(50):
        z = rng.randn(6)
        p = project_to_K(z, 0.95, 1.0)
        assert feasibility_violation(p, 0.95, 1.0) < 1e-9
        np.testing.assert_allclose(project_to_K(p, 0.95, 1.0), p, atol=1e-9)


def test_results_record_feasibility_and_kkt():
    m = SyntheticMarket.generate(6, 2, 0, skew_scale=2.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    out = scipy_optimize(f, 30, 0)
    assert all({"feasible", "converged", "n_iter", "kkt_res", "is_kkt"} <= set(r) for r in out["results"])
    assert feasibility_violation(out["x_opt"], 0.95, 1.0) <= 1e-8
    assert kkt_residual(f, out["x_opt"], 0.95, 1.0) < 1e-4
    assert analyze_local_minima(out["results"])["n_distinct_minima"] == 3      # E3 instance: 3 local minima


def test_convex_quadratic_single_minimum():
    f = MultivariatePolynomial(3, {(2, 0, 0): 1.0, (0, 2, 0): 1.0, (0, 0, 2): 1.0})
    out = scipy_optimize(f, 20, 1)
    np.testing.assert_allclose(out["x_opt"], [0.95 / 3] * 3, atol=1e-5)
    assert analyze_local_minima(out["results"])["n_distinct_minima"] == 1
