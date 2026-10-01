import numpy as np
import pytest
from scipy.optimize import minimize

from sos_portfolio import (ChordalExtension, MultivariatePolynomial, SyntheticMarket,
                           build_objective_from_market, build_portfolio_relaxation)


def robust_lb(f, terms, delta, cl=None, structure="dense"):
    return build_portfolio_relaxation(f, cl, 0.95, 1.0, 2, structure, robust_terms=terms,
                                      kappa_radius=delta).solve()


def test_one_variable_closed_form():
    # f = -x on [0.5, 1], kappa-term x^4 with weight 1: min -x + delta x^4 = -1 + delta for delta <= 1/4
    f = MultivariatePolynomial(1, {(1,): -1.0})
    for delta in (0.0, 0.1, 0.2):
        r = build_portfolio_relaxation(f, None, 0.5, 1.0, 2, "dense", robust_terms=[((4,), 1.0)], kappa_radius=delta).solve()
        assert r["lower_bound"] == pytest.approx(-1.0 + delta, abs=1e-5)


def test_delta_zero_equals_nominal_and_alias():
    m = SyntheticMarket.generate(4, 2, 0, skew_scale=2.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    nom = build_portfolio_relaxation(f, None, 0.95, 1.0, 2, "dense").solve()["lower_bound"]
    assert robust_lb(f, m.kappa_terms(), 0.0)["lower_bound"] == pytest.approx(nom, abs=1e-8)
    alias = build_portfolio_relaxation(f, None, 0.95, 1.0, 2, "dense", robust_terms=m.kappa_terms(),
                                       delta_robust=0.3).solve()["lower_bound"]
    assert alias == pytest.approx(robust_lb(f, m.kappa_terms(), 0.3)["lower_bound"], abs=1e-8)


@pytest.mark.parametrize("seed", range(10))
def test_bound_is_increasing_in_delta(seed):
    m = SyntheticMarket.generate(6, 2, seed, skew_scale=2.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    vals = [robust_lb(f, m.kappa_terms(), d, cl, "sparse")["lower_bound"] for d in (0.0, 0.05, 0.2, 0.5)]
    assert all(b >= a - 1e-6 for a, b in zip(vals, vals[1:]))
    assert vals[-1] > vals[0] + 1e-4


def test_matches_direct_minimisation_of_robust_objective():
    # min_x f(x) + delta * ||W z(x)||_2 by multi-start local search; equals the SDP bound when tight
    m = SyntheticMarket.generate(4, 2, 1, skew_scale=2.0, kurt_base=0.3, impact_base=0.05)
    f = build_objective_from_market(m)
    terms = m.kappa_terms()
    E = np.array([a for a, _ in terms])
    w = np.array([wt for _, wt in terms])
    delta = 0.4

    def robust_obj(x):
        return f(x) + delta * np.linalg.norm(w * np.prod(x[None, :] ** E, axis=1))

    rng = np.random.RandomState(0)
    best = np.inf
    for _ in range(40):
        x0 = rng.dirichlet(np.ones(4)) * rng.uniform(0.95, 1.0)
        r = minimize(robust_obj, x0, method="SLSQP", bounds=[(0, 1)] * 4,
                     constraints=[{"type": "ineq", "fun": lambda x: x.sum() - 0.95},
                                  {"type": "ineq", "fun": lambda x: 1.0 - x.sum()}], options={"ftol": 1e-13})
        if r.success and r.x.sum() >= 0.95 - 1e-9 and r.x.sum() <= 1 + 1e-9:
            best = min(best, r.fun)
    res = robust_lb(f, terms, delta)
    assert res["lower_bound"] <= best + 1e-6
    assert best - res["lower_bound"] < 1e-3    # tight on this instance
