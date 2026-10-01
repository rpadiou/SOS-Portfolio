import numpy as np
import pytest

from sos_portfolio import (ChordalExtension, MinimizerExtractor, SyntheticMarket, build_objective,
                           build_objective_from_market, build_portfolio_relaxation, certify, make_nonconvex,
                           scipy_optimize)


def pipeline(m):
    f = build_objective_from_market(m)
    cl = ChordalExtension(m.adjacency_matrix()).build().maximal_cliques
    pr = build_portfolio_relaxation(f, cl, m.budget_lower, m.budget_upper, 2, "sparse")
    r = pr.solve("CLARABEL")
    ub = scipy_optimize(f, 30, 0, m.budget_lower, m.budget_upper)
    ex = MinimizerExtractor(pr.relaxation, r["moments"], m.n)
    pts = ex.extract()
    cert = certify(f, r["lower_bound"], ([] if pts["points"] is None else list(pts["points"])) + [ex.mean_point()],
                   m.budget_lower, m.budget_upper)
    return r, ub, ex.flatness(), cert


def test_full_pipeline_n15_calibrated():
    r, ub, fl, cert = pipeline(SyntheticMarket.generate(15, 3, 0))
    assert r["status"] == "optimal"
    assert r["lower_bound"] <= ub["f_opt"] + 1e-6
    assert ub["f_opt"] - r["lower_bound"] < 1e-5
    assert fl["flat"] and cert["certified"] and cert["gap"] < 1e-6


def test_full_pipeline_n15_stress_global_certificate():
    m, info = make_nonconvex(15, 3, 0)
    assert info["n_local_minima"] >= 2
    r, ub, fl, cert = pipeline(m)
    assert r["status"] == "optimal" and cert["certified"]
    assert ub["f_opt"] - r["lower_bound"] < 1e-5      # local search found the global minimum here
    assert cert["f_hat"] - r["lower_bound"] < 1e-5


def test_hierarchy_monotone_two_assets_orders_2_3_4():
    f = build_objective()
    lb = [build_portfolio_relaxation(f, None, 0.95, 1.0, d, "dense").solve("CLARABEL")["lower_bound"] for d in (2, 3, 4)]
    ub = scipy_optimize(f, 20, 0)["f_opt"]
    assert lb[0] <= lb[1] + 1e-6 <= lb[2] + 2e-6 and lb[2] <= ub + 1e-6
    assert ub - lb[0] < 1e-6   # exact at d=2 on this instance
